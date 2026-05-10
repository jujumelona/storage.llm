#include "moe_pc_engine.h"
#include "moe_engine/include/parts/moe_fast_backend_types.h.inc"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>
#include <fstream>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

typedef int (*storagellm_tvm_kernel_fn_t)(
    int32_t backend,
    const moe_grouped_expert_device_task_t* tasks,
    uint32_t task_count,
    void* stream_or_queue);

struct storagellm_tvm_loaded_kernel {
    void* lib = nullptr;
    storagellm_tvm_kernel_fn_t fn = nullptr;
    int32_t backend = -1;
    std::string path;
    bool tried = false;
};

static std::mutex g_tvm_kernel_mutex;
static storagellm_tvm_loaded_kernel g_tvm_kernel;

static const char* tvm_env_name_for_backend(int32_t backend) {
    switch (backend) {
        case moe_BACKEND_CUDA:
            return "STORAGELLM_TVM_CUDA_MOE_LIB";
        case moe_BACKEND_METAL:
            return "STORAGELLM_TVM_METAL_MOE_LIB";
        case moe_BACKEND_VULKAN:
            return "STORAGELLM_TVM_VULKAN_MOE_LIB";
        case moe_BACKEND_OPENCL:
            return "STORAGELLM_TVM_OPENCL_MOE_LIB";
        case moe_BACKEND_CPU:
            return "STORAGELLM_TVM_CPU_MOE_LIB";
        default:
            return nullptr;
    }
}

static const char* tvm_backend_name_for_backend(int32_t backend) {
    switch (backend) {
        case moe_BACKEND_CUDA: return "CUDA";
        case moe_BACKEND_METAL: return "METAL";
        case moe_BACKEND_VULKAN: return "VULKAN";
        case moe_BACKEND_OPENCL: return "OPENCL";
        case moe_BACKEND_CPU: return "CPU";
        default: return "UNKNOWN";
    }
}


static std::string storagellm_trim_copy(std::string value) {
    while (!value.empty() && (value.back() == '\r' || value.back() == '\n' || value.back() == ' ' || value.back() == '\t')) value.pop_back();
    size_t i = 0;
    while (i < value.size() && (value[i] == ' ' || value[i] == '\t')) ++i;
    return value.substr(i);
}
static std::string storagellm_read_env_file_value(const char* key) {
    if (!key || !key[0]) return std::string();
    const char* explicit_file = std::getenv("STORAGELLM_SELECTED_BACKEND_ENV");
    std::vector<std::string> paths;
    if (explicit_file && explicit_file[0]) paths.emplace_back(explicit_file);
    paths.emplace_back("build/selected_backend.env");
    paths.emplace_back("selected_backend.env");
    for (const auto& p : paths) {
        std::ifstream in(p.c_str());
        if (!in.good()) continue;
        std::string line;
        while (std::getline(in, line)) {
            const size_t eq = line.find('=');
            if (eq == std::string::npos) continue;
            std::string k = storagellm_trim_copy(line.substr(0, eq));
            if (k == key) return storagellm_trim_copy(line.substr(eq + 1));
        }
    }
    return std::string();
}

static int storagellm_file_exists(const char* path) {
    if (!path || !path[0]) return 0;
#if defined(_WIN32)
    const DWORD attr = GetFileAttributesA(path);
    return attr != INVALID_FILE_ATTRIBUTES && !(attr & FILE_ATTRIBUTE_DIRECTORY);
#else
    return access(path, R_OK) == 0;
#endif
}

static std::string default_tvm_kernel_path_for_backend(int32_t backend) {
#if defined(_WIN32)
    const char* ext = ".dll";
#elif defined(__APPLE__)
    const char* ext = ".dylib";
#else
    const char* ext = ".so";
#endif

    const char* specific = nullptr;
    switch (backend) {
        case moe_BACKEND_CUDA: specific = "build/tvm_codegen/grouped_moe_cuda"; break;
        case moe_BACKEND_METAL: specific = "build/tvm_codegen/grouped_moe_metal"; break;
        case moe_BACKEND_VULKAN: specific = "build/tvm_codegen/grouped_moe_vulkan"; break;
        case moe_BACKEND_OPENCL: specific = "build/tvm_codegen/grouped_moe_opencl"; break;
        case moe_BACKEND_CPU: specific = "build/tvm_codegen/grouped_moe_cpu"; break;
        default: specific = "build/tvm_codegen/grouped_moe_auto"; break;
    }

    std::string p = specific;
    p += ext;
    if (storagellm_file_exists(p.c_str())) return p;

    if (backend == moe_BACKEND_CPU) {
        p = "build/tvm_codegen/grouped_moe_llvm";
        p += ext;
        if (storagellm_file_exists(p.c_str())) return p;
    }

    p = "build/tvm_codegen/grouped_moe_auto";
    p += ext;
    if (storagellm_file_exists(p.c_str())) return p;

    return std::string();
}

static void storagellm_tvm_unload_locked() {
    if (!g_tvm_kernel.lib) return;
#if defined(_WIN32)
    FreeLibrary(reinterpret_cast<HMODULE>(g_tvm_kernel.lib));
#else
    dlclose(g_tvm_kernel.lib);
#endif
    g_tvm_kernel.lib = nullptr;
    g_tvm_kernel.fn = nullptr;
    g_tvm_kernel.path.clear();
    g_tvm_kernel.backend = -1;
}

static int tvm_load_kernel_for_backend(int32_t backend) {
    std::lock_guard<std::mutex> guard(g_tvm_kernel_mutex);

    if (g_tvm_kernel.tried &&
        g_tvm_kernel.backend == backend &&
        g_tvm_kernel.fn != nullptr) {
        return 1;
    }

    if (g_tvm_kernel.backend != backend) {
        storagellm_tvm_unload_locked();
    }

    g_tvm_kernel.tried = true;
    g_tvm_kernel.backend = backend;
    g_tvm_kernel.fn = nullptr;

    const char* env_name = tvm_env_name_for_backend(backend);
    std::string path;

    if (env_name) {
        const char* env_path = std::getenv(env_name);
        if (env_path && env_path[0]) {
            path = env_path;
        }
        if (path.empty()) {
            path = storagellm_read_env_file_value(env_name);
        }
    }

    if (path.empty()) {
        const char* generic = std::getenv("STORAGELLM_TVM_MOE_LIB");
        if (generic && generic[0]) {
            path = generic;
        }
    }

    if (path.empty()) {
        path = default_tvm_kernel_path_for_backend(backend);
    }

    if (path.empty()) {
        return 0;
    }

#if defined(_WIN32)
    HMODULE lib = LoadLibraryA(path.c_str());
    if (!lib) {
        std::fprintf(stderr, "[storageLLM] failed to load TVM %s kernel: %s\n",
            tvm_backend_name_for_backend(backend), path.c_str());
        return 0;
    }
    g_tvm_kernel.lib = reinterpret_cast<void*>(lib);
    g_tvm_kernel.fn = reinterpret_cast<storagellm_tvm_kernel_fn_t>(
        GetProcAddress(lib, "storagellm_tvm_grouped_moe_entry"));
#else
    void* lib = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        std::fprintf(stderr, "[storageLLM] failed to load TVM %s kernel: %s\n",
            tvm_backend_name_for_backend(backend), dlerror());
        return 0;
    }
    g_tvm_kernel.lib = lib;
    g_tvm_kernel.fn = reinterpret_cast<storagellm_tvm_kernel_fn_t>(
        dlsym(lib, "storagellm_tvm_grouped_moe_entry"));
#endif

    g_tvm_kernel.path = path;

    if (!g_tvm_kernel.fn) {
        std::fprintf(stderr,
            "[storageLLM] TVM kernel library loaded but symbol "
            "storagellm_tvm_grouped_moe_entry was not found: %s\n",
            path.c_str());
        storagellm_tvm_unload_locked();
        return 0;
    }

    return 1;
}

extern "C" int storagellm_tvm_codegen_available(int32_t backend) {
    return tvm_load_kernel_for_backend(backend);
}

extern "C" int storagellm_tvm_codegen_grouped_moe_indexed_device_f32(
    int32_t backend,
    const moe_grouped_expert_device_task_t* tasks,
    uint32_t task_count,
    void* stream_or_queue) {
    if (!tasks || task_count == 0) {
        return 0;
    }
    if (!tvm_load_kernel_for_backend(backend)) {
        return 0;
    }

    storagellm_tvm_kernel_fn_t fn = nullptr;
    {
        std::lock_guard<std::mutex> guard(g_tvm_kernel_mutex);
        fn = g_tvm_kernel.fn;
    }

    if (!fn) {
        return 0;
    }
    return fn(backend, tasks, task_count, stream_or_queue);
}


extern "C" int storagellm_tvm_codegen_grouped_moe_indexed_device_f32_v2(
    const moe_fast_backend_dispatch_request_t* request
) {
    if (!request || request->abi_version != STORAGELLM_FAST_BACKEND_DISPATCH_ABI_V2) {
        return 0;
    }
    return storagellm_tvm_codegen_grouped_moe_indexed_device_f32(
        request->backend, request->tasks, request->task_count, request->legacy_stream_or_queue);
}

