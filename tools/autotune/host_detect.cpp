#include "host_detect.h"
#include "process.h"
#include <cstdlib>

namespace storagellm::autotune {

static bool cmd_ok(const std::string& cmd) {
    return run_capture(cmd).code == 0;
}

HostInfo detect_host() {
    HostInfo h;
#if defined(_WIN32)
    h.os = "windows";
#elif defined(__APPLE__)
    h.os = "macos";
#else
    h.os = "linux";
#endif
    h.python = find_executable({"python", "python3", "py"});
    h.cmake = find_executable({"cmake"});
    h.has_cuda = !find_executable({"nvcc"}).empty() || std::getenv("CUDA_PATH") || std::getenv("CUDA_HOME");
    h.has_rocm = !find_executable({"hipcc"}).empty() || std::getenv("ROCM_PATH") || dir_exists("/opt/rocm");
#if defined(__APPLE__)
    h.has_metal = true;
#else
    h.has_metal = false;
#endif
    h.has_vulkan = !find_executable({"vulkaninfo"}).empty() || std::getenv("VULKAN_SDK");
    h.has_opencl = !find_executable({"clinfo"}).empty() || std::getenv("OpenCL_ROOT") || std::getenv("OPENCL_ROOT");
    h.has_sycl = !find_executable({"icpx", "icx", "dpcpp"}).empty();
    return h;
}

std::vector<Candidate> make_backend_plan(const HostInfo& host) {
    std::vector<Candidate> v;
    if (host.has_cuda) {
        v.push_back({"cuda_cublaslt", "native", "cuda", "", "", 10});
        v.push_back({"tvm_cuda", "tvm", "cuda", "build/tvm_codegen/grouped_moe_cuda" + shared_library_suffix(), "STORAGELLM_TVM_CUDA_MOE_LIB", 20});
        v.push_back({"cuda_cutlass", "native_unimplemented", "cuda", "", "", 25});
    }
    if (host.has_rocm) {
        v.push_back({"rocm_hipblaslt", "native_unimplemented", "rocm", "", "", 28});
        v.push_back({"rocm_ck", "native_unimplemented", "rocm", "", "", 29});
        v.push_back({"tvm_rocm", "tvm", "rocm", "build/tvm_codegen/grouped_moe_rocm" + shared_library_suffix(), "STORAGELLM_TVM_ROCM_MOE_LIB", 30});
    }
    if (host.has_metal) {
        v.push_back({"metal_mps", "native_unimplemented", "metal", "", "", 35});
        v.push_back({"tvm_metal", "tvm", "metal", "build/tvm_codegen/grouped_moe_metal" + shared_library_suffix(), "STORAGELLM_TVM_METAL_MOE_LIB", 40});
    }
    if (host.has_vulkan) {
        v.push_back({"vulkan_coopmat", "native_unimplemented", "vulkan", "", "", 45});
        v.push_back({"tvm_vulkan", "tvm", "vulkan", "build/tvm_codegen/grouped_moe_vulkan" + shared_library_suffix(), "STORAGELLM_TVM_VULKAN_MOE_LIB", 50});
    }
    if (host.has_opencl) {
        v.push_back({"opencl_clblast", "native_unimplemented", "opencl", "", "", 55});
        v.push_back({"tvm_opencl", "tvm", "opencl", "build/tvm_codegen/grouped_moe_opencl" + shared_library_suffix(), "STORAGELLM_TVM_OPENCL_MOE_LIB", 60});
    }
    v.push_back({"cpu_native_f32", "native", "cpu", "", "", 900});
    v.push_back({"tvm_cpu", "tvm", "llvm", "build/tvm_codegen/grouped_moe_cpu" + shared_library_suffix(), "STORAGELLM_TVM_CPU_MOE_LIB", 1000});
    return v;
}

} // namespace storagellm::autotune
