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

    const bool storage_unknown = p.storage_read_gbps <= 0.0;
    const bool slow_storage = !storage_unknown && p.storage_read_gbps < 1.0;
    const bool mid_storage = !storage_unknown && p.storage_read_gbps >= 1.0 && p.storage_read_gbps < 3.0;
    const bool fast_storage = p.storage_read_gbps >= 3.0;
    const bool very_fast_storage = p.storage_read_gbps >= 6.0;
    const bool ram_much_faster = p.ram_memcpy_gbps > 0.0 && p.storage_read_gbps > 0.0 &&
        p.ram_memcpy_gbps >= p.storage_read_gbps * 4.0;

    // Derive a full three-stage movement pipeline.  Disk workers hide storage
    // latency, pinned workers saturate host memory copy, GPU workers feed DMA
    // streams.  Caps are intentionally tied to the measured slowest stage so an
    // upstream queue does not explode while a downstream stage idles.
    const uint32_t worker_cap = std::min<uint32_t>(p.hw_threads, 32u);
    p.io_workers = clamp_u32(
        very_fast_storage ? 12u : fast_storage ? 8u : mid_storage ? 6u : slow_storage ? 4u : 4u,
        1, std::max<uint32_t>(1u, std::min<uint32_t>(worker_cap, 16u)));

    p.disk_workers = clamp_u32(
        std::max<uint32_t>(p.storage_best_threads, very_fast_storage ? 8u : fast_storage ? 6u : mid_storage ? 4u : 2u),
        1, std::max<uint32_t>(1u, std::min<uint32_t>(worker_cap, 16u)));

    p.pinned_workers = clamp_u32(
        p.ram_best_threads >= 8u ? 6u : p.ram_best_threads >= 4u ? 4u : p.ram_best_threads >= 2u ? 2u : 1u,
        1, std::max<uint32_t>(1u, std::min<uint32_t>(worker_cap, 8u)));

    // GPU copy workers are limited because too many H2D submitters can serialize
    // on driver locks.  Autotune still emits enough workers for double/triple
    // buffering on devices with multiple copy engines.
    p.gpu_workers = clamp_u32(
        p.pinned_workers >= 6u ? 4u : p.pinned_workers >= 4u ? 3u : p.pinned_workers >= 2u ? 2u : 1u,
        1, std::max<uint32_t>(1u, std::min<uint32_t>(worker_cap, 8u)));

    if (storage_unknown) p.prefetch_window_layers = 2;
    else if (p.storage_read_gbps < 0.75) p.prefetch_window_layers = 4;
    else if (p.storage_read_gbps < 2.0) p.prefetch_window_layers = 3;
    else if (p.storage_read_gbps < 5.0) p.prefetch_window_layers = 3;
    else p.prefetch_window_layers = 2;
    if (ram_much_faster && !very_fast_storage) {
        p.prefetch_window_layers = std::min<uint32_t>(8u, p.prefetch_window_layers + 1u);
    }

    const uint32_t base_slots = std::max(6u, p.pinned_workers * std::max(2u, p.gpu_workers) * 3u);
    p.staging_mb = clamp_u32(base_slots * 32u, 192u, very_fast_storage ? 2048u : 1536u);

    // Queue caps are emitted explicitly so runtime sizing is coherent with the
    // measured profile.  Values are large enough to keep stages busy, but bounded
    // to avoid RAM blow-up and latency spikes from stale speculative work.
    const double ratio = (p.storage_read_gbps > 0.0 && p.ram_memcpy_gbps > 0.0)
        ? std::max(0.25, std::min(8.0, p.ram_memcpy_gbps / p.storage_read_gbps))
        : 1.0;
    p.pipeline_depth_scale = slow_storage ? 2.0 : mid_storage ? 1.5 : very_fast_storage ? 0.85 : 1.0;
    const uint32_t disk_units = p.disk_workers * p.prefetch_window_layers * (slow_storage ? 256u : 128u);
    const uint32_t pinned_units = p.pinned_workers * std::max<uint32_t>(2u, p.gpu_workers) * 128u;
    const uint32_t gpu_units = p.gpu_workers * (very_fast_storage ? 192u : 128u);
    p.disk_stage_depth_cap = clamp_u32((uint32_t)std::ceil(disk_units * p.pipeline_depth_scale * std::min(2.0, ratio)), 512u, 65536u);
    p.pinned_stage_depth_cap = clamp_u32((uint32_t)std::ceil(pinned_units * p.pipeline_depth_scale), 512u, 65536u);
    p.gpu_stage_depth_cap = clamp_u32((uint32_t)std::ceil(gpu_units * p.pipeline_depth_scale / std::max(0.5, std::min(2.0, ratio))), 512u, 65536u);
    p.max_prefetch_queue = clamp_u32(
        std::max<uint32_t>({p.disk_stage_depth_cap, p.pinned_stage_depth_cap, p.gpu_stage_depth_cap}) * 2u,
        2048u,
        65536u);

    // Keep small tensors on the normal pinned path; direct device upload is best
    // for larger contiguous bundles where syscall/driver setup is amortized.
    p.direct_upload_min_mb = slow_storage ? 32u : fast_storage ? 16u : 8u;

    p.reason = "measured storage/RAM movement and emitted a bounded parallel disk->RAM->pinned->GPU pipeline profile; compute backend autotune remains separate";
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
    ss << "STORAGELLM_MAX_PREFETCH_QUEUE=" << p.max_prefetch_queue << "\n";
    ss << "STORAGELLM_DISK_STAGE_DEPTH_CAP=" << p.disk_stage_depth_cap << "\n";
    ss << "STORAGELLM_PINNED_STAGE_DEPTH_CAP=" << p.pinned_stage_depth_cap << "\n";
    ss << "STORAGELLM_GPU_STAGE_DEPTH_CAP=" << p.gpu_stage_depth_cap << "\n";
    ss << "STORAGELLM_DIRECT_UPLOAD_MIN_MB=" << p.direct_upload_min_mb << "\n";
    ss << "STORAGELLM_PIPELINE_DEPTH_SCALE=" << p.pipeline_depth_scale << "\n";
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
    ss << "    \"pinned_staging_mb\": " << p.staging_mb << ",\n";
    ss << "    \"max_prefetch_queue\": " << p.max_prefetch_queue << ",\n";
    ss << "    \"disk_stage_depth_cap\": " << p.disk_stage_depth_cap << ",\n";
    ss << "    \"pinned_stage_depth_cap\": " << p.pinned_stage_depth_cap << ",\n";
    ss << "    \"gpu_stage_depth_cap\": " << p.gpu_stage_depth_cap << ",\n";
    ss << "    \"direct_upload_min_mb\": " << p.direct_upload_min_mb << ",\n";
    ss << "    \"pipeline_depth_scale\": " << p.pipeline_depth_scale << "\n";
    ss << "  },\n";
    ss << "  \"truth\": \"This minimizes movement bottlenecks by measuring RAM/storage throughput, sizing queues, workers, staging and lookahead together, and letting runtime backpressure prevent oversupply. It does not prove physical bottleneck=0 for every model/device.\"\n";
    ss << "}\n";
    return ss.str();
}

} // namespace storagellm::autotune
