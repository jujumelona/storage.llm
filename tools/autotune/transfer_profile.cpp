#include "transfer_profile.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace storagellm::autotune {
namespace {

static uint32_t env_u32(const char* name, uint32_t fallback, uint32_t lo, uint32_t hi) {
    const char* v = std::getenv(name);
    if (!v || !v[0]) return fallback;
    char* end = nullptr;
    unsigned long parsed = std::strtoul(v, &end, 10);
    if (end == v || parsed == 0) return fallback;
    return std::max(lo, std::min<uint32_t>((uint32_t)parsed, hi));
}

static uint32_t hw_threads() {
    unsigned int n = std::thread::hardware_concurrency();
    return n ? std::min<uint32_t>(n, 256u) : 1u;
}

static double elapsed_ms(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static double memcpy_gbps_once(size_t bytes, uint32_t threads) {
    if (bytes < 1024 || threads == 0) return 0.0;
    std::vector<uint8_t> src(bytes);
    std::vector<uint8_t> dst(bytes);
    for (size_t i = 0; i < src.size(); ++i) src[i] = static_cast<uint8_t>(i * 131u + 7u);

    const int warmups = 2;
    const int min_iters = 4;
    const double min_ms = 120.0;
    auto run_copy = [&]() {
        std::vector<std::thread> workers;
        workers.reserve(threads);
        const size_t chunk = (bytes + threads - 1) / threads;
        for (uint32_t t = 0; t < threads; ++t) {
            const size_t begin = std::min(bytes, (size_t)t * chunk);
            const size_t end = std::min(bytes, begin + chunk);
            workers.emplace_back([&, begin, end]() {
                if (end > begin) {
                    std::memcpy(dst.data() + begin, src.data() + begin, end - begin);
                }
            });
        }
        for (auto& w : workers) w.join();
    };
    for (int i = 0; i < warmups; ++i) run_copy();

    int iters = 0;
    auto t0 = std::chrono::steady_clock::now();
    double ms = 0.0;
    do {
        run_copy();
        ++iters;
        ms = elapsed_ms(t0, std::chrono::steady_clock::now());
    } while ((iters < min_iters || ms < min_ms) && iters < 256);

    volatile uint8_t sink = dst[bytes / 2];
    (void)sink;
    const double gib = (double(bytes) * double(iters)) / (1024.0 * 1024.0 * 1024.0);
    return ms > 0.0 ? gib / (ms / 1000.0) : 0.0;
}

static double write_probe_file(const fs::path& path, size_t bytes) {
    std::vector<uint8_t> block(4 * 1024 * 1024);
    for (size_t i = 0; i < block.size(); ++i) block[i] = static_cast<uint8_t>(i * 17u + 3u);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) return 0.0;
    auto t0 = std::chrono::steady_clock::now();
    size_t done = 0;
    while (done < bytes) {
        const size_t n = std::min(block.size(), bytes - done);
        out.write(reinterpret_cast<const char*>(block.data()), static_cast<std::streamsize>(n));
        if (!out) return 0.0;
        done += n;
    }
    out.flush();
    auto t1 = std::chrono::steady_clock::now();
    const double ms = elapsed_ms(t0, t1);
    const double gib = double(bytes) / (1024.0 * 1024.0 * 1024.0);
    return ms > 0.0 ? gib / (ms / 1000.0) : 0.0;
}

static double read_probe_file_gbps(const fs::path& path, size_t bytes, uint32_t threads) {
    if (threads == 0 || bytes == 0) return 0.0;
    const size_t chunk = (bytes + threads - 1) / threads;
    std::atomic<uint64_t> checksum{0};
    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> workers;
    workers.reserve(threads);
    for (uint32_t t = 0; t < threads; ++t) {
        const size_t begin = std::min(bytes, (size_t)t * chunk);
        const size_t end = std::min(bytes, begin + chunk);
        if (end <= begin) continue;
        workers.emplace_back([&, begin, end]() {
            std::ifstream in(path, std::ios::binary);
            if (!in) return;
            in.seekg(static_cast<std::streamoff>(begin), std::ios::beg);
            std::vector<uint8_t> buf(1024 * 1024);
            size_t remaining = end - begin;
            uint64_t local = 0;
            while (remaining > 0 && in) {
                const size_t n = std::min(buf.size(), remaining);
                in.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(n));
                const size_t got = static_cast<size_t>(in.gcount());
                if (got == 0) break;
                for (size_t i = 0; i < got; i += 4096) local += buf[i];
                remaining -= got;
            }
            checksum.fetch_add(local, std::memory_order_relaxed);
        });
    }
    for (auto& w : workers) w.join();
    auto t1 = std::chrono::steady_clock::now();
    volatile uint64_t sink = checksum.load(std::memory_order_relaxed);
    (void)sink;
    const double ms = elapsed_ms(t0, t1);
    const double gib = double(bytes) / (1024.0 * 1024.0 * 1024.0);
    return ms > 0.0 ? gib / (ms / 1000.0) : 0.0;
}

static uint32_t clamp_u32(uint32_t v, uint32_t lo, uint32_t hi) {
    return std::max(lo, std::min(v, hi));
}

static uint32_t next_power_probe(uint32_t prev, uint32_t maxv) {
    if (prev >= maxv) return maxv + 1;
    return std::min(maxv, prev * 2u);
}

} // namespace

TransferProfile measure_transfer_profile() {
    TransferProfile p;
    p.hw_threads = hw_threads();
    p.probe_mb = env_u32("STORAGELLM_TRANSFER_PROBE_MB", 64, 8, 512);
    const size_t probe_bytes = size_t(p.probe_mb) * 1024ull * 1024ull;

    const uint32_t ram_thread_cap = clamp_u32(std::min(p.hw_threads, 16u), 1, 16);
    for (uint32_t t = 1; t <= ram_thread_cap; t = next_power_probe(t, ram_thread_cap)) {
        const double gbps = memcpy_gbps_once(probe_bytes, t);
        if (gbps > p.ram_memcpy_gbps) {
            p.ram_memcpy_gbps = gbps;
            p.ram_best_threads = t;
        }
    }

    fs::create_directories("build");
    fs::path probe_path = fs::path("build") / "storagellm_transfer_probe.bin";
    p.storage_write_gbps = write_probe_file(probe_path, probe_bytes);
    const uint32_t disk_thread_cap = clamp_u32(std::min(p.hw_threads, 8u), 1, 8);
    for (uint32_t t = 1; t <= disk_thread_cap; t = next_power_probe(t, disk_thread_cap)) {
        const double gbps = read_probe_file_gbps(probe_path, probe_bytes, t);
        if (gbps > p.storage_read_gbps) {
            p.storage_read_gbps = gbps;
            p.storage_best_threads = t;
        }
    }
    std::error_code ec;
    fs::remove(probe_path, ec);

    const bool slow_storage = p.storage_read_gbps > 0.0 && p.storage_read_gbps < 1.0;
    const bool fast_storage = p.storage_read_gbps >= 3.0;
    p.io_workers = clamp_u32(fast_storage ? 8u : slow_storage ? 2u : 4u, 1, std::min(16u, p.hw_threads));
    p.disk_workers = clamp_u32(std::max(p.storage_best_threads, slow_storage ? 2u : 4u), 1, std::min(16u, p.hw_threads));
    p.pinned_workers = clamp_u32(p.ram_best_threads >= 4 ? 3u : 1u, 1, std::min(8u, p.hw_threads));
    p.gpu_workers = clamp_u32(p.pinned_workers >= 3 ? 3u : 1u, 1, std::min(8u, p.hw_threads));

    if (p.storage_read_gbps <= 0.0) p.prefetch_window_layers = 1;
    else if (p.storage_read_gbps < 0.75) p.prefetch_window_layers = 1;
    else if (p.storage_read_gbps < 2.0) p.prefetch_window_layers = 2;
    else if (p.storage_read_gbps < 5.0) p.prefetch_window_layers = 3;
    else p.prefetch_window_layers = 4;

    const uint32_t base_slots = std::max(4u, p.pinned_workers * std::max(1u, p.gpu_workers) * 2u);
    p.staging_mb = clamp_u32(base_slots * 32u, 128u, 1024u);
    p.reason = "measured host RAM memcpy and storage read/write throughput in C++ autotune; derived default IO/pinned/GPU worker and staging/prefetch knobs for this machine";
    return p;
}

std::string transfer_env_text(const TransferProfile& p) {
    std::ostringstream ss;
    ss << "STORAGELLM_IO_WORKERS=" << p.io_workers << "\n";
    ss << "STORAGELLM_DISK_WORKERS=" << p.disk_workers << "\n";
    ss << "STORAGELLM_PINNED_WORKERS=" << p.pinned_workers << "\n";
    ss << "STORAGELLM_GPU_WORKERS=" << p.gpu_workers << "\n";
    ss << "STORAGELLM_PREFETCH_WINDOW_LAYERS=" << p.prefetch_window_layers << "\n";
    ss << "STORAGELLM_PINNED_STAGING_MB=" << p.staging_mb << "\n";
    ss << "STORAGELLM_MEASURED_RAM_GBPS=" << p.ram_memcpy_gbps << "\n";
    ss << "STORAGELLM_MEASURED_STORAGE_GBPS=" << p.storage_read_gbps << "\n";
    return ss.str();
}

std::string transfer_report_json(const TransferProfile& p) {
    std::ostringstream ss;
    ss << "{\n";
    ss << "  \"version\": 1,\n";
    ss << "  \"hw_threads\": " << p.hw_threads << ",\n";
    ss << "  \"probe_mb\": " << p.probe_mb << ",\n";
    ss << "  \"ram_memcpy_gbps\": " << p.ram_memcpy_gbps << ",\n";
    ss << "  \"ram_best_threads\": " << p.ram_best_threads << ",\n";
    ss << "  \"storage_read_gbps\": " << p.storage_read_gbps << ",\n";
    ss << "  \"storage_write_gbps\": " << p.storage_write_gbps << ",\n";
    ss << "  \"storage_best_threads\": " << p.storage_best_threads << ",\n";
    ss << "  \"selected_knobs\": {\n";
    ss << "    \"io_workers\": " << p.io_workers << ",\n";
    ss << "    \"disk_workers\": " << p.disk_workers << ",\n";
    ss << "    \"pinned_workers\": " << p.pinned_workers << ",\n";
    ss << "    \"gpu_workers\": " << p.gpu_workers << ",\n";
    ss << "    \"prefetch_window_layers\": " << p.prefetch_window_layers << ",\n";
    ss << "    \"pinned_staging_mb\": " << p.staging_mb << "\n";
    ss << "  },\n";
    ss << "  \"truth\": \"This reduces host-side movement bottlenecks by measuring RAM/storage throughput and auto-applying pipeline knobs. It does not prove bottleneck=0 for every model/device.\"\n";
    ss << "}\n";
    return ss.str();
}

} // namespace storagellm::autotune
