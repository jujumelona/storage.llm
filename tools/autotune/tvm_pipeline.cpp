#include "tvm_pipeline.h"
#include "files.h"
#include "process.h"
#include <filesystem>
#include <sstream>
#include <cstdlib>

namespace fs = std::filesystem;

namespace storagellm::autotune {

static std::string tvm_install_backend(const HostInfo& host) {
    if (host.has_cuda) return "cuda";
    if (host.has_rocm) return "rocm";
    return "cpu";
}

void run_tvm_pipeline(const HostInfo& host, std::vector<Candidate>& candidates, int trials) {
    ensure_dir("build/tvm_codegen");
    const char* no_tvm = std::getenv("STORAGELLM_AUTOTUNE_NO_TVM");
    if (no_tvm && no_tvm[0] == '1') {
        for (auto& c : candidates) if (c.kind == "tvm") c.reason = "TVM helper skipped by STORAGELLM_AUTOTUNE_NO_TVM=1";
        return;
    }
    if (host.python.empty()) {
        for (auto& c : candidates) if (c.kind == "tvm") c.reason = "python not found; TVM pip/codegen helper cannot run";
        return;
    }

    const std::string py = shell_quote(host.python);
    const std::string installer = shell_quote("scripts/install_tvm_dependency.py");
    const std::string dep_status = shell_quote("build/tvm_codegen/tvm_dependency_status.json");
    std::string install_cmd = py + " " + installer + " --backend " + tvm_install_backend(host) + " --status-out " + dep_status;
    auto install = run_capture(install_cmd);

    for (auto& c : candidates) {
        if (c.kind != "tvm") continue;
        if (c.name != "tvm_cpu") {
            c.reason = "device TVM candidate skipped: no in-process real device fixture is available for safe measurement";
            continue;
        }
        const std::string profile = "build/tvm_codegen/profile_" + c.name + ".json";
        write_profile_json(profile, c);
        const std::string workdir = "build/tvm_tuning/" + c.name;
        const std::string outlib = c.library;
        std::ostringstream cmd;
        cmd << py << " " << shell_quote("moe_engine/backends/tvm_codegen/tvm_grouped_moe_tune.py")
            << " --profile " << shell_quote(profile)
            << " --work-dir " << shell_quote(workdir)
            << " --trials " << trials
            << " --codegen-out " << shell_quote(outlib);
        auto r = run_capture(cmd.str());
        if (file_exists(outlib)) {
            c.compiled = true;
            c.reason = "TVM helper compiled candidate on this host";
        } else {
            c.compiled = false;
            c.reason = "TVM candidate not compiled. installer rc=" + std::to_string(install.code) + ", tune/codegen rc=" + std::to_string(r.code);
        }
    }
}

} // namespace storagellm::autotune
