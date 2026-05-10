#include "host_detect.h"
#include "process.h"
#include <cstdlib>
#include <sstream>

namespace storagellm::autotune {

static bool cmd_ok(const std::string& cmd, std::string* out = nullptr) {
    ProcessResult r = run_capture(cmd);
    if (out) *out = r.output;
    return r.code == 0;
}

static std::string first_line(std::string s) {
    std::istringstream is(s);
    std::string line;
    if (std::getline(is, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        return line;
    }
    return {};
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

    h.cuda_toolkit = !find_executable({"nvcc"}).empty() || std::getenv("CUDA_PATH") || std::getenv("CUDA_HOME");
    {
        std::string out;
        h.cuda_device = cmd_ok("nvidia-smi -L", &out) && !out.empty() && out.find("GPU") != std::string::npos;
        h.cuda_probe = h.cuda_device ? first_line(out) : (out.empty() ? "nvidia-smi unavailable or no NVIDIA GPU" : first_line(out));
    }

    h.rocm_toolkit = !find_executable({"hipcc"}).empty() || std::getenv("ROCM_PATH") || dir_exists("/opt/rocm");
    {
        std::string out;
        h.rocm_device = (cmd_ok("rocminfo", &out) && (out.find("gfx") != std::string::npos || out.find("Agent") != std::string::npos)) ||
                        (cmd_ok("rocm-smi -i", &out) && (out.find("GPU") != std::string::npos || out.find("card") != std::string::npos));
        h.rocm_probe = h.rocm_device ? first_line(out) : (out.empty() ? "rocminfo/rocm-smi unavailable or no ROCm GPU" : first_line(out));
    }

#if defined(__APPLE__)
    h.has_metal = true;
#else
    h.has_metal = false;
#endif

    {
        std::string out;
        const bool vulkan_tool = !find_executable({"vulkaninfo"}).empty() || std::getenv("VULKAN_SDK");
        h.vulkan_device = cmd_ok("vulkaninfo --summary", &out) &&
            (out.find("GPU") != std::string::npos || out.find("deviceName") != std::string::npos);
        h.has_vulkan = vulkan_tool || h.vulkan_device;
        h.vulkan_probe = h.vulkan_device ? first_line(out) : (out.empty() ? "vulkaninfo unavailable or no Vulkan device" : first_line(out));
    }

    {
        std::string out;
        const bool opencl_tool = !find_executable({"clinfo"}).empty() || std::getenv("OpenCL_ROOT") || std::getenv("OPENCL_ROOT");
        h.opencl_device = cmd_ok("clinfo -l", &out) &&
            (out.find("Platform") != std::string::npos || out.find("Device") != std::string::npos);
        h.has_opencl = opencl_tool || h.opencl_device;
        h.opencl_probe = h.opencl_device ? first_line(out) : (out.empty() ? "clinfo unavailable or no OpenCL device" : first_line(out));
    }

    {
        std::string out;
        const bool sycl_tool = !find_executable({"icpx", "icx", "dpcpp"}).empty();
        h.sycl_device = cmd_ok("sycl-ls", &out) &&
            (out.find("gpu") != std::string::npos || out.find("level_zero") != std::string::npos || out.find("opencl") != std::string::npos);
        h.has_sycl = sycl_tool || h.sycl_device;
        h.sycl_probe = h.sycl_device ? first_line(out) : (out.empty() ? "sycl-ls unavailable or no SYCL device" : first_line(out));
    }

    h.has_cuda = h.cuda_toolkit || h.cuda_device;
    h.has_rocm = h.rocm_toolkit || h.rocm_device;
    return h;
}

static Candidate make_candidate(
    const std::string& name,
    const std::string& kind,
    const std::string& target,
    const std::string& library,
    const std::string& env_key,
    int priority,
    bool runtime_device,
    bool true_kernel,
    bool fused_moe
) {
    Candidate c;
    c.name = name;
    c.kind = kind;
    c.tvm_target = target;
    c.library = library;
    c.env_key = env_key;
    c.priority = priority;
    c.runtime_device = runtime_device;
    c.true_kernel = true_kernel;
    c.fused_moe = fused_moe;
    return c;
}

std::vector<Candidate> make_backend_plan(const HostInfo& host) {
    std::vector<Candidate> v;

    // Only emit candidates that storagellm_host_autotune can actually execute
    // and time with a real fixture in this executable.  Platform/device support
    // that is merely detected is still reported in HostInfo, but it is not
    // advertised as an automatic fast-backend candidate until a real benchmark
    // path exists here.  This avoids "looks enabled" skeleton paths.
    if (host.cuda_device) {
        v.push_back(make_candidate("cuda_cublaslt", "native", "cuda", "", "", 10, true, true, false));
        v.push_back(make_candidate("cuda_cutlass", "native", "cuda", "", "", 20, true, true, true));
    }
    if (host.rocm_device) {
        v.push_back(make_candidate("rocm_hipblaslt", "native", "rocm", "", "", 35, true, true, true));
        v.push_back(make_candidate("rocm_ck", "native", "rocm", "", "", 36, true, true, true));
    }
    if (host.opencl_device) {
        v.push_back(make_candidate("opencl_clblast", "native", "opencl", "", "", 65, true, true, true));
    }

    // TVM CPU is measured in-process through a host pointer fixture.  Device TVM
    // is not listed here because a CUDA/HIP/Metal/Vulkan/OpenCL TVM module needs
    // a matching runtime context and device allocations; reporting it without
    // that measurement would be a fake fast path.
    v.push_back(make_candidate("cpu_native_f32", "native", "cpu", "", "", 900, true, true, true));
    v.push_back(make_candidate("tvm_cpu", "tvm", "llvm", "build/tvm_codegen/grouped_moe_cpu" + shared_library_suffix(), "STORAGELLM_TVM_CPU_MOE_LIB", 1000, true, true, true));
    return v;
}

} // namespace storagellm::autotune
