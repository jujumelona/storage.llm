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

static void moe_apply_autotune_io_env(moe_io_config_t* cfg) {
    if (!cfg) return;
    const uint32_t hw = moe_cpu_thread_count_hint();
    const uint32_t max_workers = std::max<uint32_t>(1u, std::min<uint32_t>(hw, 64u));
    cfg->io_worker_threads = moe_autotune_u32("STORAGELLM_IO_WORKERS", cfg->io_worker_threads, 1u, max_workers);
    cfg->disk_worker_threads = moe_autotune_u32("STORAGELLM_DISK_WORKERS", cfg->disk_worker_threads, 1u, max_workers);
    cfg->pinned_worker_threads = moe_autotune_u32("STORAGELLM_PINNED_WORKERS", cfg->pinned_worker_threads, 1u, max_workers);
    cfg->gpu_worker_threads = moe_autotune_u32("STORAGELLM_GPU_WORKERS", cfg->gpu_worker_threads, 1u, max_workers);
    cfg->prefetch_window_layers = moe_autotune_u32("STORAGELLM_PREFETCH_WINDOW_LAYERS", cfg->prefetch_window_layers, 1u, 64u);
    const uint64_t staging_mb = moe_autotune_u64("STORAGELLM_PINNED_STAGING_MB", 0, 16, 4096);
    if (staging_mb > 0) {
        cfg->pinned_staging_bytes = staging_mb * 1024ull * 1024ull;
    }
}

static void moe_apply_autotune_io_env(moe_pc_engine* engine) {
    if (!engine) return;
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
}
