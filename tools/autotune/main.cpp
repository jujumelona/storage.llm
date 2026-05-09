#include "benchmark.h"
#include "files.h"
#include "host_detect.h"
#include "select.h"
#include "tvm_pipeline.h"
#include "transfer_profile.h"
#include <cstdlib>
#include <iostream>

using namespace storagellm::autotune;

static int env_int(const char* name, int fallback) {
    const char* v = std::getenv(name);
    if (!v || !v[0]) return fallback;
    try { return std::stoi(v); } catch (...) { return fallback; }
}

int main() {
    ensure_dir("build");
    ensure_dir("build/tvm_codegen");

    HostInfo host = detect_host();
    std::vector<Candidate> candidates = make_backend_plan(host);

    std::cout << "[storageLLM] C++ host autotune: detect -> TVM helper/codegen -> C++ benchmark -> select\n";
    std::cout << "[storageLLM] Python is helper-only for pip/TVM API, not engine runtime.\n";

    const int trials = env_int("STORAGELLM_TVM_TRIALS", 64);
    TransferProfile transfer = measure_transfer_profile();
    write_text("build/transfer_profile.json", transfer_report_json(transfer));

    run_tvm_pipeline(host, candidates, trials);
    benchmark_candidates(candidates);

    Candidate* selected = select_best(candidates);
    write_selected_env("build/selected_backend.env", selected);
    write_text("build/selected_backend.env", read_text("build/selected_backend.env") + transfer_env_text(transfer));
    write_report_json("build/auto_backend_report.json", host, candidates, selected);

    if (selected) {
        std::cout << "[storageLLM] selected backend: " << selected->name
                  << " latency_ms=" << selected->latency_ms << "\n";
        std::cout << "[storageLLM] wrote build/selected_backend.env and build/auto_backend_report.json\n";
        return 0;
    }
    std::cout << "[storageLLM] no measured fast backend selected; report written fail-closed.\n";
    return 1;
}
