// Auto-tune environment loader.
// Reads process environment first, then build/selected_backend.env or
// selected_backend.env written by storagellm_host_autotune. This lets the C++
// engine consume host-local movement tuning without Python as runtime owner.

static std::string moe_autotune_trim(std::string s) {
    while (!s.empty() && (s.back() == '\r' || s.back() == '\n' || s.back() == ' ' || s.back() == '\t')) s.pop_back();
    size_t i = 0;
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
    return i ? s.substr(i) : s;
}

static std::string moe_autotune_file_value(const char* key) {
    if (!key || !key[0]) return {};
    const char* paths[] = {"build/selected_backend.env", "selected_backend.env"};
    for (const char* path : paths) {
        std::ifstream in(path, std::ios::binary);
        if (!in) continue;
        std::string line;
        const std::string prefix = std::string(key) + "=";
        while (std::getline(in, line)) {
            line = moe_autotune_trim(line);
            if (line.rfind(prefix, 0) == 0) {
                return moe_autotune_trim(line.substr(prefix.size()));
            }
        }
    }
    return {};
}

static std::string moe_autotune_value(const char* key) {
    const char* env = key ? std::getenv(key) : nullptr;
    if (env && env[0]) return std::string(env);
    return moe_autotune_file_value(key);
}

static uint32_t moe_autotune_u32(const char* key, uint32_t fallback, uint32_t min_value, uint32_t max_value) {
    const std::string value = moe_autotune_value(key);
    if (value.empty()) return fallback;
    char* end = nullptr;
    unsigned long parsed = std::strtoul(value.c_str(), &end, 10);
    if (end == value.c_str() || parsed == 0) return fallback;
    uint32_t v = static_cast<uint32_t>(std::min<unsigned long>(parsed, static_cast<unsigned long>(UINT32_MAX)));
    return std::max(min_value, std::min(v, max_value));
}

static uint64_t moe_autotune_u64(const char* key, uint64_t fallback, uint64_t min_value, uint64_t max_value) {
    const std::string value = moe_autotune_value(key);
    if (value.empty()) return fallback;
    char* end = nullptr;
    unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
    if (end == value.c_str() || parsed == 0) return fallback;
    uint64_t v = static_cast<uint64_t>(parsed);
    return std::max(min_value, std::min(v, max_value));
}

static double moe_autotune_f64(const char* key, double fallback, double min_value, double max_value) {
    const std::string value = moe_autotune_value(key);
    if (value.empty()) return fallback;
    char* end = nullptr;
    double parsed = std::strtod(value.c_str(), &end);
    if (end == value.c_str() || !std::isfinite(parsed)) return fallback;
    return std::max(min_value, std::min(parsed, max_value));
}

static uint32_t moe_autotune_round_up_pow2ish(uint32_t v) {
    if (v <= 64u) return 64u;
    uint32_t out = 64u;
    while (out < v && out < 65536u) out <<= 1u;
    return out;
}

static void moe_apply_autotune_io_env(moe_io_config_t* cfg) {
    if (!cfg) return;
    const uint32_t hw = moe_cpu_thread_count_hint();
    const uint32_t max_workers = std::max<uint32_t>(1u, std::min<uint32_t>(hw, 64u));
    cfg->io_worker_threads = moe_autotune_u32("STORAGELLM_IO_WORKERS", cfg->io_worker_threads, 1u, max_workers);
    cfg->disk_worker_threads = moe_autotune_u32("STORAGELLM_DISK_WORKERS", cfg->disk_worker_threads, 1u, max_workers);
    cfg->pinned_worker_threads = moe_autotune_u32("STORAGELLM_PINNED_WORKERS", cfg->pinned_worker_threads, 1u, max_workers);
    cfg->gpu_worker_threads = moe_autotune_u32("STORAGELLM_GPU_WORKERS", cfg->gpu_worker_threads, 1u, max_workers);
    cfg->prefetch_window_layers = moe_autotune_u32("STORAGELLM_PREFETCH_WINDOW_LAYERS", cfg->prefetch_window_layers, 1u, 64u);

    // Movement bottleneck profile from storagellm_host_autotune.  These are not
    // user-facing "modes"; they are measured host-local defaults that keep the
    // storage->RAM->pinned->GPU pipeline full without letting one stage create
    // unbounded backpressure for the next stage.
    cfg->max_prefetch_queue = moe_autotune_u32(
        "STORAGELLM_MAX_PREFETCH_QUEUE",
        cfg->max_prefetch_queue,
        512u,
        65536u);

    const uint64_t direct_upload_min_mb = moe_autotune_u64("STORAGELLM_DIRECT_UPLOAD_MIN_MB", 0, 1, 4096);
    if (direct_upload_min_mb > 0) {
        cfg->direct_upload_min_bytes = direct_upload_min_mb * 1024ull * 1024ull;
    }

    const uint64_t staging_mb = moe_autotune_u64("STORAGELLM_PINNED_STAGING_MB", 0, 16, 8192);
    if (staging_mb > 0) {
        cfg->pinned_staging_bytes = staging_mb * 1024ull * 1024ull;
    }
}

static void moe_apply_autotune_transfer_queue_caps(moe_pc_engine* engine) {
    if (!engine) return;

    const uint32_t disk_workers = std::max<uint32_t>(1u, runtime_disk_workers(engine));
    const uint32_t pinned_workers = std::max<uint32_t>(1u, runtime_pinned_workers(engine));
    const uint32_t gpu_workers = std::max<uint32_t>(1u, runtime_gpu_workers(engine));
    const uint32_t window = std::max<uint32_t>(1u,
        engine->runtime_prefetch_window_layers.load(std::memory_order_relaxed));

    const double storage_gbps = moe_autotune_f64("STORAGELLM_MEASURED_STORAGE_GBPS", 0.0, 0.0, 10000.0);
    const double ram_gbps = moe_autotune_f64("STORAGELLM_MEASURED_RAM_GBPS", 0.0, 0.0, 10000.0);
    const double bottleneck_ratio = (storage_gbps > 0.0 && ram_gbps > 0.0)
        ? std::max(0.125, std::min(8.0, ram_gbps / storage_gbps))
        : 1.0;
    const double depth_scale = moe_autotune_f64("STORAGELLM_PIPELINE_DEPTH_SCALE", 1.0, 0.25, 8.0);

    // Slow storage needs more lookahead to hide latency; very fast storage should
    // avoid oversupplying GPU queues beyond DMA stream capacity.
    uint32_t disk_base = disk_workers * window * 64u;
    if (storage_gbps > 0.0 && storage_gbps < 1.0) disk_base *= 2u;
    uint32_t pinned_base = pinned_workers * std::max<uint32_t>(4u, gpu_workers) * 64u;
    uint32_t gpu_base = gpu_workers * std::max<uint32_t>(4u, engine->backend_caps.copy_streams ? engine->backend_caps.copy_streams : 1u) * 64u;

    disk_base = static_cast<uint32_t>(std::min<double>(65536.0, std::ceil(disk_base * depth_scale * std::min(2.0, bottleneck_ratio))));
    pinned_base = static_cast<uint32_t>(std::min<double>(65536.0, std::ceil(pinned_base * depth_scale)));
    gpu_base = static_cast<uint32_t>(std::min<double>(65536.0, std::ceil(gpu_base * depth_scale / std::max(0.5, std::min(2.0, bottleneck_ratio)))));

    const uint32_t base_cap = std::max<uint32_t>(512u, engine->io_config.max_prefetch_queue);
    engine->disk_stage_depth_cap = std::max<uint32_t>(512u, std::min<uint32_t>(base_cap, moe_autotune_round_up_pow2ish(disk_base)));
    engine->pinned_stage_depth_cap = std::max<uint32_t>(512u, std::min<uint32_t>(base_cap, moe_autotune_round_up_pow2ish(pinned_base)));
    engine->gpu_stage_depth_cap = std::max<uint32_t>(512u, std::min<uint32_t>(std::max<uint32_t>(base_cap, 1024u), moe_autotune_round_up_pow2ish(gpu_base)));

    engine->disk_stage_depth_cap = moe_autotune_u32("STORAGELLM_DISK_STAGE_DEPTH_CAP", engine->disk_stage_depth_cap, 128u, 65536u);
    engine->pinned_stage_depth_cap = moe_autotune_u32("STORAGELLM_PINNED_STAGE_DEPTH_CAP", engine->pinned_stage_depth_cap, 128u, 65536u);
    engine->gpu_stage_depth_cap = moe_autotune_u32("STORAGELLM_GPU_STAGE_DEPTH_CAP", engine->gpu_stage_depth_cap, 128u, 65536u);
}

static void moe_apply_autotune_io_env(moe_pc_engine* engine) {
    if (!engine) return;
    const double measured_storage_gbps_for_caps = moe_autotune_f64(
        "STORAGELLM_MEASURED_STORAGE_GBPS", 0.0, 0.0, 10000.0);
    if (measured_storage_gbps_for_caps > 0.0) {
        engine->backend_caps.measured_storage_bw_bytes_per_sec =
            static_cast<uint64_t>(measured_storage_gbps_for_caps * 1024.0 * 1024.0 * 1024.0);
    }
    moe_apply_autotune_io_env(&engine->io_config);
    const uint32_t disk = moe_autotune_u32("STORAGELLM_DISK_WORKERS", 0, 1u, 64u);
    const uint32_t pinned = moe_autotune_u32("STORAGELLM_PINNED_WORKERS", 0, 1u, 64u);
    const uint32_t gpu = moe_autotune_u32("STORAGELLM_GPU_WORKERS", 0, 1u, 64u);
    if (disk || pinned || gpu) {
        runtime_store_io_workers(
            engine,
            disk ? disk : runtime_disk_workers(engine),
            pinned ? pinned : runtime_pinned_workers(engine),
            gpu ? gpu : runtime_gpu_workers(engine));
    }
    const uint32_t window = moe_autotune_u32("STORAGELLM_PREFETCH_WINDOW_LAYERS", 0, 1u, 64u);
    if (window) {
        engine->runtime_prefetch_window_layers.store(window, std::memory_order_relaxed);
        engine->adaptive_prefetch_window_last.store(window, std::memory_order_relaxed);
    }
    moe_apply_autotune_transfer_queue_caps(engine);
}


static void moe_reconcile_pipeline_caps_after_staging(moe_pc_engine* engine) {
    if (!engine) return;
    const uint32_t disk_workers = std::max<uint32_t>(1u, runtime_disk_workers(engine));
    const uint32_t pinned_workers = std::max<uint32_t>(1u, runtime_pinned_workers(engine));
    const uint32_t gpu_workers = std::max<uint32_t>(1u, runtime_gpu_workers(engine));
    const uint32_t window = std::max<uint32_t>(1u,
        engine->runtime_prefetch_window_layers.load(std::memory_order_relaxed));

    uint32_t pinned_slots = engine->staging_pinned_slot_count.load(std::memory_order_acquire);
    uint32_t gpu_slots = engine->staging_gpu_slot_count.load(std::memory_order_acquire);
    uint32_t direct_slots = engine->staging_direct_slot_count.load(std::memory_order_acquire);
    const uint32_t total_slots = static_cast<uint32_t>(engine->staging_slots.size());

    if (!engine->io_config.enable_multistage_pipeline || !engine->staging || total_slots == 0) {
        const uint32_t disk_floor = std::max<uint32_t>(128u, disk_workers * window * 32u);
        engine->disk_stage_depth_cap = std::max<uint32_t>(engine->disk_stage_depth_cap, disk_floor);
        engine->pinned_stage_depth_cap = std::min<uint32_t>(engine->pinned_stage_depth_cap, 128u);
        engine->gpu_stage_depth_cap = std::min<uint32_t>(engine->gpu_stage_depth_cap, 128u);
        return;
    }

    if (pinned_slots + gpu_slots + direct_slots == 0) {
        pinned_slots = std::max<uint32_t>(1u, total_slots / 3u);
        gpu_slots = std::max<uint32_t>(1u, total_slots / 3u);
        direct_slots = total_slots > pinned_slots + gpu_slots ? total_slots - pinned_slots - gpu_slots : 0u;
    }

    // Convert physical staging-slot capacity into queue-count caps.  This closes
    // the old hole where autotune could emit thousands of pinned/GPU queue slots
    // even though only tens of staging slots existed, causing slot spin/requeue storms.
    const uint32_t pinned_physical_cap = std::max<uint32_t>(
        64u,
        std::max<uint32_t>(pinned_workers * 8u, (pinned_slots + direct_slots) * 4u));
    const uint32_t gpu_physical_cap = std::max<uint32_t>(
        64u,
        std::max<uint32_t>(gpu_workers * 8u, (gpu_slots + direct_slots) * 4u));
    const uint32_t disk_supply_cap = std::max<uint32_t>(
        128u,
        std::max<uint32_t>(disk_workers * window * 64u,
            std::min<uint32_t>(engine->io_config.max_prefetch_queue,
                (pinned_physical_cap + gpu_physical_cap) * 2u)));

    engine->disk_stage_depth_cap = std::min<uint32_t>(
        std::max<uint32_t>(engine->disk_stage_depth_cap, disk_supply_cap),
        engine->io_config.max_prefetch_queue);
    engine->pinned_stage_depth_cap = std::min<uint32_t>(
        engine->pinned_stage_depth_cap,
        std::min<uint32_t>(engine->io_config.max_prefetch_queue, pinned_physical_cap));
    engine->gpu_stage_depth_cap = std::min<uint32_t>(
        engine->gpu_stage_depth_cap,
        std::min<uint32_t>(std::max<uint32_t>(engine->io_config.max_prefetch_queue, 128u), gpu_physical_cap));

    engine->disk_stage_depth_cap = std::max<uint32_t>(128u, engine->disk_stage_depth_cap);
    engine->pinned_stage_depth_cap = std::max<uint32_t>(64u, engine->pinned_stage_depth_cap);
    engine->gpu_stage_depth_cap = std::max<uint32_t>(64u, engine->gpu_stage_depth_cap);
}
