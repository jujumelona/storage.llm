#pragma once
#include <string>
#include <vector>

namespace storagellm::autotune {
struct HostInfo {
    std::string os;
    bool has_cuda = false;
    bool has_rocm = false;
    bool has_metal = false;
    bool has_vulkan = false;
    bool has_opencl = false;
    bool has_sycl = false;
    bool cuda_toolkit = false;
    bool cuda_device = false;
    bool rocm_toolkit = false;
    bool rocm_device = false;
    bool vulkan_device = false;
    bool opencl_device = false;
    bool sycl_device = false;
    std::string python;
    std::string cmake;
    std::string cuda_probe;
    std::string rocm_probe;
    std::string vulkan_probe;
    std::string opencl_probe;
    std::string sycl_probe;
};

struct Candidate {
    std::string name;
    std::string kind;
    std::string tvm_target;
    std::string library;
    std::string env_key;
    int priority = 1000;
    bool compiled = false;
    bool loadable = false;
    bool measured = false;
    bool runtime_device = false;
    bool true_kernel = false;
    bool fused_moe = false;
    bool verified = false;
    double latency_ms = 0.0;
    double correctness_max_abs = 0.0;
    double correctness_max_rel = 0.0;
    std::string validation;
    std::string reason;
};
}
