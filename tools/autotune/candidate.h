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
    std::string python;
    std::string cmake;
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
    double latency_ms = 0.0;
    std::string reason;
};
}
