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
    }
    write_text(path, ss.str());
}

static void write_bool(std::ostream& os, bool v) { os << (v ? "true" : "false"); }

void write_report_json(const std::string& path, const HostInfo& host, const std::vector<Candidate>& candidates, const Candidate* selected) {
    std::ostringstream ss;
    ss << "{\n";
    ss << "  \"version\": 5,\n";
    ss << "  \"pipeline_owner\": \"C++ storagellm_host_autotune executable\",\n";
    ss << "  \"python_policy\": \"not engine runtime; only invoked for pip/TVM Python API codegen when TVM candidates are built\",\n";
    ss << "  \"cmake_policy\": \"build system only; user does not manually choose backend options in the default path\",\n";
    ss << "  \"host\": {\n";
    ss << "    \"os\": \"" << json_escape(host.os) << "\",\n";
    ss << "    \"python\": \"" << json_escape(host.python) << "\",\n";
    ss << "    \"cmake\": \"" << json_escape(host.cmake) << "\",\n";
    ss << "    \"cuda\": "; write_bool(ss, host.has_cuda); ss << ",\n";
    ss << "    \"rocm\": "; write_bool(ss, host.has_rocm); ss << ",\n";
    ss << "    \"metal\": "; write_bool(ss, host.has_metal); ss << ",\n";
    ss << "    \"vulkan\": "; write_bool(ss, host.has_vulkan); ss << ",\n";
    ss << "    \"opencl\": "; write_bool(ss, host.has_opencl); ss << ",\n";
    ss << "    \"sycl\": "; write_bool(ss, host.has_sycl); ss << "\n";
    ss << "  },\n";
    ss << "  \"candidates\": [\n";
    for (size_t i = 0; i < candidates.size(); ++i) {
        const auto& c = candidates[i];
        ss << "    {\"name\": \"" << json_escape(c.name) << "\", \"kind\": \"" << json_escape(c.kind)
           << "\", \"target\": \"" << json_escape(c.tvm_target) << "\", \"library\": \"" << json_escape(c.library)
           << "\", \"compiled\": "; write_bool(ss, c.compiled); ss << ", \"loadable\": "; write_bool(ss, c.loadable);
        ss << ", \"measured\": "; write_bool(ss, c.measured); ss << ", \"latency_ms\": " << c.latency_ms
           << ", \"reason\": \"" << json_escape(c.reason) << "\"}" << (i + 1 == candidates.size() ? "" : ",") << "\n";
    }
    ss << "  ],\n";
    ss << "  \"selected_backend\": ";
    if (selected) {
        ss << "{\"name\": \"" << json_escape(selected->name) << "\", \"latency_ms\": " << selected->latency_ms
           << ", \"library\": \"" << json_escape(selected->library) << "\"}";
    } else {
        ss << "null";
    }
    ss << ",\n";
    ss << "  \"truth\": \"C++ auto pipeline is wired. CPU native, TVM CPU, and linked CUDA cuBLASLt native candidates can be measured automatically. Other SDK-specific GPU native adapters remain fail-closed unless their real device kernels are present and measured.\"\n";
    ss << "}\n";
    write_text(path, ss.str());
}

} // namespace storagellm::autotune
