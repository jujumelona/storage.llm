#include "files.h"
#include "process.h"
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

namespace storagellm::autotune {

void ensure_dir(const std::string& path) {
    std::error_code ec;
    fs::create_directories(fs::path(path), ec);
}

void write_text(const std::string& path, const std::string& text) {
    fs::path p(path);
    ensure_dir(p.parent_path().string().empty() ? "." : p.parent_path().string());
    std::ofstream out(p, std::ios::binary);
    out << text;
}

std::string read_text(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) out += " ";
                else out += c;
        }
    }
    return out;
}

void write_profile_json(const std::string& path, const Candidate& c) {
    std::ostringstream ss;
    ss << "{\n";
    ss << "  \"backend\": \"" << json_escape(c.tvm_target == "llvm" ? "cpu" : c.tvm_target) << "\",\n";
    ss << "  \"target\": \"" << json_escape(c.tvm_target) << "\",\n";
    ss << "  \"hidden\": 4096,\n";
    ss << "  \"intermediate\": 14336,\n";
    ss << "  \"dtype\": \"fp32\",\n";
    ss << "  \"max_experts\": 8,\n";
    ss << "  \"max_assignments\": 256,\n";
    ss << "  \"source\": \"cxx-host-autotune-default\"\n";
    ss << "}\n";
    write_text(path, ss.str());
}

void write_selected_env(const std::string& path, const Candidate* selected) {
    std::ostringstream ss;
    if (selected) {
        if (!selected->env_key.empty() && !selected->library.empty()) {
            ss << selected->env_key << "=" << selected->library << "\n";
        }
        ss << "STORAGELLM_SELECTED_FAST_BACKEND=" << selected->name << "\n";
        ss << "STORAGELLM_SELECTED_FAST_BACKEND_VERIFIED=" << (selected->verified ? 1 : 0) << "\n";
        ss << "STORAGELLM_SELECTED_FAST_BACKEND_LATENCY_MS=" << selected->latency_ms << "\n";
        ss << "STORAGELLM_SELECTED_FAST_BACKEND_VALIDATION=" << selected->validation << "\n";
        ss << "STORAGELLM_SELECTED_FAST_BACKEND_CORRECTNESS_MAX_ABS=" << selected->correctness_max_abs << "\n";
        ss << "STORAGELLM_SELECTED_FAST_BACKEND_CORRECTNESS_MAX_REL=" << selected->correctness_max_rel << "\n";
    }
    write_text(path, ss.str());
}

static void write_bool(std::ostream& os, bool v) { os << (v ? "true" : "false"); }

void write_report_json(const std::string& path, const HostInfo& host, const std::vector<Candidate>& candidates, const Candidate* selected) {
    std::ostringstream ss;
    ss << "{\n";
    ss << "  \"version\": 6,\n";
    ss << "  \"pipeline_owner\": \"C++ storagellm_host_autotune executable\",\n";
    ss << "  \"candidate_policy\": \"no skeleton candidates: only backends with an in-process real-device benchmark fixture are emitted as automatic candidates; detected-but-unmeasured platform support remains host diagnostics, not selectable success\",\n";
    ss << "  \"python_policy\": \"not engine runtime; only invoked for pip/TVM Python API codegen when TVM candidates are built\",\n";
    ss << "  \"cmake_policy\": \"build system only; user does not manually choose backend options in the default path\",\n";
    ss << "  \"host\": {\n";
    ss << "    \"os\": \"" << json_escape(host.os) << "\",\n";
    ss << "    \"python\": \"" << json_escape(host.python) << "\",\n";
    ss << "    \"cmake\": \"" << json_escape(host.cmake) << "\",\n";
    ss << "    \"cuda\": "; write_bool(ss, host.has_cuda); ss << ",\n";
    ss << "    \"cuda_toolkit\": "; write_bool(ss, host.cuda_toolkit); ss << ",\n";
    ss << "    \"cuda_device\": "; write_bool(ss, host.cuda_device); ss << ",\n";
    ss << "    \"cuda_probe\": \"" << json_escape(host.cuda_probe) << "\",\n";
    ss << "    \"rocm\": "; write_bool(ss, host.has_rocm); ss << ",\n";
    ss << "    \"rocm_toolkit\": "; write_bool(ss, host.rocm_toolkit); ss << ",\n";
    ss << "    \"rocm_device\": "; write_bool(ss, host.rocm_device); ss << ",\n";
    ss << "    \"rocm_probe\": \"" << json_escape(host.rocm_probe) << "\",\n";
    ss << "    \"metal\": "; write_bool(ss, host.has_metal); ss << ",\n";
    ss << "    \"vulkan\": "; write_bool(ss, host.has_vulkan); ss << ",\n";
    ss << "    \"vulkan_device\": "; write_bool(ss, host.vulkan_device); ss << ",\n";
    ss << "    \"vulkan_probe\": \"" << json_escape(host.vulkan_probe) << "\",\n";
    ss << "    \"opencl\": "; write_bool(ss, host.has_opencl); ss << ",\n";
    ss << "    \"opencl_device\": "; write_bool(ss, host.opencl_device); ss << ",\n";
    ss << "    \"opencl_probe\": \"" << json_escape(host.opencl_probe) << "\",\n";
    ss << "    \"sycl\": "; write_bool(ss, host.has_sycl); ss << ",\n";
    ss << "    \"sycl_device\": "; write_bool(ss, host.sycl_device); ss << ",\n";
    ss << "    \"sycl_probe\": \"" << json_escape(host.sycl_probe) << "\"\n";
    ss << "  },\n";
    ss << "  \"candidates\": [\n";
    for (size_t i = 0; i < candidates.size(); ++i) {
        const auto& c = candidates[i];
        ss << "    {\"name\": \"" << json_escape(c.name) << "\", \"kind\": \"" << json_escape(c.kind)
           << "\", \"target\": \"" << json_escape(c.tvm_target) << "\", \"library\": \"" << json_escape(c.library)
           << "\", \"compiled\": "; write_bool(ss, c.compiled); ss << ", \"loadable\": "; write_bool(ss, c.loadable);
        ss << ", \"runtime_device\": "; write_bool(ss, c.runtime_device);
        ss << ", \"true_kernel\": "; write_bool(ss, c.true_kernel);
        ss << ", \"fused_moe\": "; write_bool(ss, c.fused_moe);
        ss << ", \"verified\": "; write_bool(ss, c.verified);
        ss << ", \"measured\": "; write_bool(ss, c.measured); ss << ", \"latency_ms\": " << c.latency_ms
           << ", \"correctness_max_abs\": " << c.correctness_max_abs
           << ", \"correctness_max_rel\": " << c.correctness_max_rel
           << ", \"validation\": \"" << json_escape(c.validation) << "\""
           << ", \"reason\": \"" << json_escape(c.reason) << "\"}" << (i + 1 == candidates.size() ? "" : ",") << "\n";
    }
    ss << "  ],\n";
    ss << "  \"selected_backend\": ";
    if (selected) {
        ss << "{\"name\": \"" << json_escape(selected->name) << "\", \"latency_ms\": " << selected->latency_ms
           << ", \"verified\": "; write_bool(ss, selected->verified);
        ss << ", \"correctness_max_abs\": " << selected->correctness_max_abs
           << ", \"correctness_max_rel\": " << selected->correctness_max_rel
           << ", \"validation\": \"" << json_escape(selected->validation) << "\""
           << ", \"library\": \"" << json_escape(selected->library) << "\"}";
    } else {
        ss << "null";
    }
    ss << ",\n";
    ss << "  \"truth\": \"C++ auto pipeline is wired fail-closed: a backend is only selected when it is linked/loadable, has a runtime device when needed, produces output matching the C++ reference, and is actually measured. Backends without a complete benchmark+correctness fixture are not emitted as auto candidates, so they cannot be mistaken for maximum-speed success.\"\n";
    ss << "}\n";
    write_text(path, ss.str());
}

} // namespace storagellm::autotune
