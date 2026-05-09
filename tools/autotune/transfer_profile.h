#pragma once
#include <cstdint>
#include <string>

namespace storagellm::autotune {

struct TransferProfile {
    uint32_t hw_threads = 1;
    uint32_t probe_mb = 64;
    double ram_memcpy_gbps = 0.0;
    uint32_t ram_best_threads = 1;
    double storage_read_gbps = 0.0;
    uint32_t storage_best_threads = 1;
    double storage_write_gbps = 0.0;
    uint32_t io_workers = 1;
    uint32_t disk_workers = 1;
    uint32_t pinned_workers = 1;
    uint32_t gpu_workers = 1;
    uint32_t prefetch_window_layers = 1;
    uint32_t staging_mb = 128;
    std::string reason;
};

TransferProfile measure_transfer_profile();
std::string transfer_env_text(const TransferProfile& p);
std::string transfer_report_json(const TransferProfile& p);

} // namespace storagellm::autotune
