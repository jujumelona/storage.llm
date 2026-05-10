
// CPU-only storage execution hooks.
//
// This is intentionally a separate generation path selector, not just a backend
// flag.  CPU-only decode must avoid the GPU offload/pinned/directstorage queues
// and must also avoid advertising graph_ir as the active forward path.  The
// actual math still reuses the proven attention/MLP functions, but the runtime
// plan, prefetch and health state are CPU-storage specific.

static int moe_cpu_storage_env_truthy(const char* name) {
    const char* v = std::getenv(name);
    if (!v || !*v) return 0;
    if (v[0] == '0') return 0;
    if ((v[0] == 'f' || v[0] == 'F') && (v[1] == 'a' || v[1] == 'A')) return 0;
    if ((v[0] == 'n' || v[0] == 'N') && (v[1] == 'o' || v[1] == 'O')) return 0;
    return 1;
}

static thread_local int g_moe_storage_prefetch_worker_tls = 0;

static int moe_cpu_storage_env_false(const char* name) {
    const char* v = std::getenv(name);
    if (!v || !*v) return 0;
    return v[0] == '0' || v[0] == 'f' || v[0] == 'F' || v[0] == 'n' || v[0] == 'N';
}

static int moe_cpu_storage_env_enabled_default_on(const char* name) {
    const char* v = std::getenv(name);
    if (!v || !*v) return 1;
    return !(v[0] == '0' || v[0] == 'f' || v[0] == 'F' || v[0] == 'n' || v[0] == 'N');
}

static int moe_storage_always_env_enabled() {
    return moe_cpu_storage_env_truthy("STORAGELLM_STORAGE_ALWAYS") ||
           moe_cpu_storage_env_truthy("STORAGELLM_STORAGE_REQUIRED") ||
           moe_cpu_storage_env_truthy("STORAGELLM_STORAGE_IO_PLANNER");
}

static void moe_cpu_storage_hint_mapped_range(const uint8_t* ptr, uint64_t bytes) {
    if (!ptr || bytes == 0 || !moe_cpu_storage_env_enabled_default_on("STORAGELLM_CPU_MADVISE_HOT_VIEWS")) {
        return;
    }
#if defined(__linux__)
    // madvise is only a kernel hint.  Unlike page-touching it does not force a
    // blocking read here, so it is safe for CPUStorageEngine plan creation.  Align
    // the range because Linux may reject unaligned addresses.
    const uintptr_t page = 4096u;
    uintptr_t begin = (uintptr_t)ptr;
    uintptr_t end = begin + (uintptr_t)bytes;
    uintptr_t aligned_begin = begin & ~(page - 1u);
    uintptr_t aligned_end = (end + page - 1u) & ~(page - 1u);
    if (aligned_end > aligned_begin) {
        (void)madvise((void*)aligned_begin, (size_t)(aligned_end - aligned_begin), MADV_WILLNEED);
    }
#else
    (void)ptr;
    (void)bytes;
#endif
}

static int moe_cpu_storage_tensor_is_hot_for_hint(const moe_tensor_record& rec) {
    return rec.residency_hint == moe_JUJU_RESIDENCY_FAST_MEM ||
           rec.residency_hint == moe_JUJU_RESIDENCY_FAST_MEM_STREAMABLE ||
           rec.prefetch_class == moe_JUJU_PREFETCH_STARTUP_HOT ||
           rec.prefetch_class == moe_JUJU_PREFETCH_LAYER_HOT ||
           rec.prefetch_class == moe_JUJU_PREFETCH_ROUTER_HOT ||
           rec.runtime_priority <= 20 ||
           rec.hotset_rank <= 128;
}

static int moe_cpu_storage_runtime_detected(const moe_pc_engine* engine) {
    return moe_storage_runtime_cpu_ram_path(engine);
}

static int moe_cpu_storage_explicit_slow_fallback_allowed() {
    return moe_cpu_storage_env_truthy("STORAGELLM_ALLOW_GRAPH_IR_SLOW_FALLBACK") ||
           moe_cpu_storage_env_truthy("STORAGELLM_EXPLICIT_SLOW_MODE");
}

static int moe_cpu_storage_strict_fast_path_enabled() {
    return moe_cpu_storage_env_truthy("STORAGELLM_STRICT_FAST_PATH") ||
           moe_cpu_storage_env_truthy("STORAGELLM_FAIL_ON_FALLBACK") ||
           moe_cpu_storage_env_truthy("STORAGELLM_NO_SLOW_FALLBACK");
}

static void moe_cpu_storage_disable_gpu_paths(moe_pc_engine* engine) {
    if (!engine) return;
    engine->config.prefer_gpu = 0;
    engine->config.preferred_backend = moe_BACKEND_CPU;
    engine->config.platform = moe_PLATFORM_CPU_ONLY;
    engine->config.vram_budget_bytes = 0;
    engine->config.common_vram_reserve_bytes = 0;

    engine->io_config.pinned_worker_threads = 0;
    engine->io_config.gpu_worker_threads = 0;
    engine->io_config.gpu_expert_scratch_slots = 0;
    engine->io_config.pinned_staging_bytes = 0;
    engine->io_config.directstorage_staging_queue_capacity = 0;
    engine->io_config.directstorage_gpu_queue_capacity = 0;
    engine->io_config.enable_pinned_host = 0;
    engine->io_config.enable_directstorage = 0;
    engine->io_config.enable_gpudirect_storage = 0;
    engine->io_config.enable_bam_nvme = 0;
    engine->io_config.enable_common_raw_vram_pin = 0;
    engine->io_config.internal_require_gpu_expert_batch_kernel = 0;
    engine->io_config.internal_verified_backend_only = 0;
    engine->io_config.enable_cold_expert_cpu_fallback =
        moe_cpu_storage_strict_fast_path_enabled() ? 0 : 1;
    engine->io_config.allow_required_sync_fallback =
        moe_cpu_storage_strict_fast_path_enabled() ? 0 : 1;
    if (engine->io_config.prefetch_window_layers == 0 || engine->io_config.prefetch_window_layers < 4) {
        engine->io_config.prefetch_window_layers = 4;
    }
    if (engine->io_config.max_prefetch_queue == 0 || engine->io_config.max_prefetch_queue < 16384) {
        engine->io_config.max_prefetch_queue = 16384;
    }
    engine->io_config.enable_compute_io_overlap = 1;
    engine->io_config.enable_multistage_pipeline = 1;
    engine->io_config.enable_lookahead_prefetch = 1;
    engine->io_config.enable_topology_prefetch = 1;
    engine->io_config.enable_mmap_prefetch = 1;
    engine->storage_runtime_path.store(moe_STORAGE_RUNTIME_PATH_CPU_RAM, std::memory_order_relaxed);
    engine->cpu_only_runtime.store(1, std::memory_order_relaxed);
    engine->gpu_path_disabled_for_cpu_only.store(1, std::memory_order_relaxed);
}

static uint32_t moe_cpu_storage_weight_format_countable(const moe_tensor_record& rec) {
    const int fmt = tensor_weight_format(rec);
    return moe_dot_weight_format_supported(fmt) ? 1u : 0u;
}


static uint8_t moe_cpu_storage_format_simple_dot(int weight_format) {
    return moe_cpu_simple_row_dot_for_format(weight_format) ? 1u : 0u;
}


static uint64_t moe_cpu_storage_env_u64_default(const char* name, uint64_t defv) {
    const char* v = std::getenv(name);
    if (!v || !*v) return defv;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(v, &end, 10);
    if (end == v) return defv;
    return (uint64_t)parsed;
}

static uint64_t moe_cpu_storage_layer_prefetch_budget_bytes() {
    // Large enough to queue the next layer's hot/common tensors, small enough
    // to avoid evicting the current token's working set from RAM on CPU-only.
    // CPU max mode benefits from a deeper page-cache runway, especially on NVMe.
    return moe_cpu_storage_env_u64_default(
        "STORAGELLM_CPU_LAYER_PREFETCH_BUDGET_BYTES",
        2048ull * 1024ull * 1024ull);
}

static int moe_cpu_storage_page_touch_enabled() {
    return moe_cpu_storage_env_truthy("STORAGELLM_CPU_PAGE_TOUCH");
}


static uint32_t moe_storage_io_parse_profile() {
    const char* v = std::getenv("STORAGELLM_STORAGE_PROFILE");
    if (!v || !*v || std::strcmp(v, "auto") == 0) return moe_STORAGE_PROFILE_AUTO;
    if (std::strcmp(v, "nvme") == 0 || std::strcmp(v, "nvme_ssd") == 0) return moe_STORAGE_PROFILE_NVME;
    if (std::strcmp(v, "ssd") == 0 || std::strcmp(v, "sata_ssd") == 0 || std::strcmp(v, "sata") == 0) return moe_STORAGE_PROFILE_SATA_SSD;
    if (std::strcmp(v, "hdd") == 0) return moe_STORAGE_PROFILE_HDD;
    if (std::strcmp(v, "network") == 0 || std::strcmp(v, "drive") == 0 || std::strcmp(v, "gdrive") == 0) return moe_STORAGE_PROFILE_NETWORK;
    if (std::strcmp(v, "ramdisk") == 0 || std::strcmp(v, "tmpfs") == 0 || std::strcmp(v, "ram") == 0) return moe_STORAGE_PROFILE_RAMDISK;
    return moe_STORAGE_PROFILE_AUTO;
}

static uint32_t moe_storage_io_worker_cap(uint32_t profile) {
    const uint32_t hw = std::max<uint32_t>(1u, moe_cpu_thread_count_hint());
    switch (profile) {
        case moe_STORAGE_PROFILE_NVME:
            return std::min<uint32_t>(64u, std::max<uint32_t>(16u, hw));
        case moe_STORAGE_PROFILE_SATA_SSD:
            return std::min<uint32_t>(32u, std::max<uint32_t>(8u, hw / 2u));
        case moe_STORAGE_PROFILE_HDD:
            return 2u;
        case moe_STORAGE_PROFILE_NETWORK:
            return std::min<uint32_t>(8u, std::max<uint32_t>(2u, hw / 4u));
        case moe_STORAGE_PROFILE_RAMDISK:
            return 0u;
        default:
            return std::min<uint32_t>(32u, std::max<uint32_t>(8u, hw / 2u));
    }
}

static uint32_t moe_storage_io_default_queue_depth(uint32_t profile) {
    if (const char* env = std::getenv("STORAGELLM_STORAGE_PREFETCH_QUEUE_DEPTH")) {
        char* end = nullptr;
        const unsigned long v = std::strtoul(env, &end, 10);
        if (end != env && v > 0) return (uint32_t)v;
    }
    const uint32_t workers = std::max<uint32_t>(1u, moe_storage_io_worker_cap(profile));
    switch (profile) {
        case moe_STORAGE_PROFILE_NVME:
            return std::max<uint32_t>(1024u, workers * 128u);
        case moe_STORAGE_PROFILE_SATA_SSD:
            return std::max<uint32_t>(256u, workers * 64u);
        case moe_STORAGE_PROFILE_HDD:
            return 16u;
        case moe_STORAGE_PROFILE_NETWORK:
            return std::max<uint32_t>(32u, workers * 16u);
        case moe_STORAGE_PROFILE_RAMDISK:
            return 0u;
        default:
            return std::max<uint32_t>(512u, workers * 96u);
    }
}

static uint32_t moe_storage_io_default_workers(uint32_t profile) {
    if (const char* env = std::getenv("STORAGELLM_STORAGE_PREFETCH_WORKERS")) {
        char* end = nullptr;
        const unsigned long v = std::strtoul(env, &end, 10);
        if (end != env) return (uint32_t)std::min<unsigned long>(v, 128ul);
    }
    const uint32_t hw = std::max<uint32_t>(1u, moe_cpu_thread_count_hint());
    switch (profile) {
        case moe_STORAGE_PROFILE_NVME:
            return std::min<uint32_t>(moe_storage_io_worker_cap(profile), std::max<uint32_t>(8u, hw / 2u));
        case moe_STORAGE_PROFILE_SATA_SSD:
            return std::min<uint32_t>(moe_storage_io_worker_cap(profile), std::max<uint32_t>(4u, hw / 4u));
        case moe_STORAGE_PROFILE_HDD:
            return 1u;
        case moe_STORAGE_PROFILE_NETWORK:
            return std::min<uint32_t>(moe_storage_io_worker_cap(profile), std::max<uint32_t>(2u, hw / 8u));
        case moe_STORAGE_PROFILE_RAMDISK:
            return 0u;
        default:
            return std::min<uint32_t>(moe_storage_io_worker_cap(profile), std::max<uint32_t>(6u, hw / 3u));
    }
}

static uint64_t moe_storage_io_merge_gap_bytes(uint32_t profile) {
    if (const char* env = std::getenv("STORAGELLM_STORAGE_PREFETCH_MERGE_GAP_BYTES")) {
        char* end = nullptr;
        const unsigned long long v = std::strtoull(env, &end, 10);
        if (end != env) return (uint64_t)v;
    }
    switch (profile) {
        case moe_STORAGE_PROFILE_NVME: return 256ull * 1024ull;
        case moe_STORAGE_PROFILE_SATA_SSD: return 128ull * 1024ull;
        case moe_STORAGE_PROFILE_HDD: return 1024ull * 1024ull;
        case moe_STORAGE_PROFILE_NETWORK: return 2ull * 1024ull * 1024ull;
        case moe_STORAGE_PROFILE_RAMDISK: return 0ull;
        default: return 256ull * 1024ull;
    }
}

static int moe_storage_io_enabled_default_on(const char* name) {
    return moe_cpu_storage_env_enabled_default_on(name);
}

static uint32_t moe_storage_io_auto_profile_for_model_root(const moe_pc_engine* engine, uint32_t parsed) {
    if (parsed != moe_STORAGE_PROFILE_AUTO || !engine) return parsed;
    const std::string& root = engine->model_root;
    if (root.find("/content/drive") != std::string::npos ||
        root.find("GoogleDrive") != std::string::npos ||
        root.find("gdrive") != std::string::npos) {
        return moe_STORAGE_PROFILE_NETWORK;
    }
    if (root.find("/dev/shm") != std::string::npos ||
        root.find("/run/shm") != std::string::npos ||
        root.find("tmpfs") != std::string::npos) {
        return moe_STORAGE_PROFILE_RAMDISK;
    }
    // Most Colab / local model directories in this project are SSD/NVMe-backed.
    // Keep AUTO as NVME-like so storage-required mode gets queue depth and
    // readahead by default instead of the generic conservative profile.
    if (moe_storage_always_env_enabled()) {
        return moe_STORAGE_PROFILE_NVME;
    }
    return parsed;
}

static int moe_storage_io_iouring_default_enabled(const moe_pc_engine* engine);
static int moe_storage_io_gpu_consumer_active(const moe_pc_engine* engine);
static void moe_storage_io_prefetch_range_execute(
    moe_pc_engine* engine,
    mmap_context_t* shard,
    const uint8_t* mapped_base,
    uint64_t offset,
    uint64_t bytes,
    uint64_t* bytes_io);


static int moe_storage_prefetch_inline_on_overflow_enabled() {
    // Queue-overflow behavior is a runtime policy, not per-request data.  Cache
    // it so the token hot path does not call getenv while saturated.
    static const int enabled = moe_cpu_storage_env_truthy("STORAGELLM_STORAGE_PREFETCH_INLINE_ON_OVERFLOW");
    return enabled;
}

static int moe_storage_async_prefetch_default_enabled(moe_pc_engine* engine) {
    if (!engine) return 0;
    if (!moe_cpu_storage_env_enabled_default_on("STORAGELLM_STORAGE_ASYNC_PREFETCH")) return 0;
    const uint32_t profile = engine->storage_profile.load(std::memory_order_relaxed);
    if (profile == moe_STORAGE_PROFILE_RAMDISK) return 0;
    const uint32_t workers = engine->storage_prefetch_workers.load(std::memory_order_relaxed);
    const uint32_t depth = engine->storage_prefetch_queue_depth.load(std::memory_order_relaxed);
    return workers > 0 && depth > 0;
}

static void moe_storage_prefetch_worker_loop(moe_pc_engine* engine) {
    if (!engine) return;
    g_moe_storage_prefetch_worker_tls = 1;
    std::vector<moe_storage_prefetch_request_t> batch;
    batch.reserve(8u);
    for (;;) {
        batch.clear();
        {
            std::unique_lock<std::mutex> lock(engine->storage_prefetch_queue_mutex);
            engine->storage_prefetch_queue_cv.wait(lock, [engine]() {
                return engine->storage_prefetch_stop.load(std::memory_order_acquire) ||
                       !engine->storage_prefetch_queue.empty();
            });
            if (engine->storage_prefetch_stop.load(std::memory_order_acquire) &&
                engine->storage_prefetch_queue.empty()) {
                break;
            }
            const uint32_t profile = engine->storage_profile.load(std::memory_order_relaxed);
            const size_t batch_cap = profile == moe_STORAGE_PROFILE_NVME ? 8u :
                                     profile == moe_STORAGE_PROFILE_SATA_SSD ? 4u : 2u;
            while (!engine->storage_prefetch_queue.empty() && batch.size() < batch_cap) {
                batch.push_back(engine->storage_prefetch_queue.front());
                engine->storage_prefetch_queue.pop_front();
            }
        }
        if (!batch.empty()) {
            engine->storage_async_prefetch_pending.fetch_sub(
                (uint64_t)batch.size(), std::memory_order_relaxed);
        }
        for (const moe_storage_prefetch_request_t& req : batch) {
            uint64_t bytes_io = 0;
            moe_storage_io_prefetch_range_execute(engine, req.shard, req.mapped_base, req.offset, req.bytes, &bytes_io);
            engine->storage_async_prefetch_completed.fetch_add(1, std::memory_order_relaxed);
        }
    }
    g_moe_storage_prefetch_worker_tls = 0;
}

static void moe_storage_async_prefetch_ensure_started(moe_pc_engine* engine) {
    if (!engine || !moe_storage_async_prefetch_default_enabled(engine)) {
        if (engine) engine->storage_async_prefetch_enabled.store(0, std::memory_order_relaxed);
        return;
    }
    if (!engine->storage_prefetch_worker_threads.empty()) {
        engine->storage_async_prefetch_enabled.store(1, std::memory_order_relaxed);
        return;
    }
    std::lock_guard<std::mutex> lock(engine->storage_prefetch_queue_mutex);
    if (!engine->storage_prefetch_worker_threads.empty()) {
        engine->storage_async_prefetch_enabled.store(1, std::memory_order_relaxed);
        return;
    }
    engine->storage_prefetch_stop.store(0, std::memory_order_release);
    uint32_t workers = engine->storage_prefetch_workers.load(std::memory_order_relaxed);
    const uint32_t profile = engine->storage_profile.load(std::memory_order_relaxed);
    if (workers == 0) workers = moe_storage_io_default_workers(profile);
    workers = std::min<uint32_t>(workers, moe_storage_io_worker_cap(profile));
    for (uint32_t i = 0; i < workers; ++i) {
        try {
            engine->storage_prefetch_worker_threads.emplace_back([engine]() { moe_storage_prefetch_worker_loop(engine); });
        } catch (...) {
            engine->storage_async_prefetch_dropped.fetch_add(1, std::memory_order_relaxed);
            break;
        }
    }
    engine->storage_async_prefetch_enabled.store(!engine->storage_prefetch_worker_threads.empty() ? 1 : 0, std::memory_order_relaxed);
}

static void moe_storage_async_prefetch_shutdown(moe_pc_engine* engine) {
    if (!engine) return;
    engine->storage_prefetch_stop.store(1, std::memory_order_release);
    engine->storage_prefetch_queue_cv.notify_all();
    for (std::thread& t : engine->storage_prefetch_worker_threads) {
        if (t.joinable()) t.join();
    }
    engine->storage_prefetch_worker_threads.clear();
    {
        std::lock_guard<std::mutex> lock(engine->storage_prefetch_queue_mutex);
        engine->storage_prefetch_queue.clear();
    }
    engine->storage_async_prefetch_pending.store(0, std::memory_order_relaxed);
    engine->storage_async_prefetch_enabled.store(0, std::memory_order_relaxed);
}

static int moe_storage_async_prefetch_enqueue(
    moe_pc_engine* engine,
    mmap_context_t* shard,
    const uint8_t* mapped_base,
    uint64_t offset,
    uint64_t bytes
) {
    if (!engine || !shard || !mapped_base || bytes == 0) return 0;
    if (g_moe_storage_prefetch_worker_tls) return 0;
    if (!engine->storage_async_prefetch_enabled.load(std::memory_order_relaxed)) {
        moe_storage_async_prefetch_ensure_started(engine);
        if (!engine->storage_async_prefetch_enabled.load(std::memory_order_relaxed)) return 0;
    }
    const uint32_t depth = std::max<uint32_t>(1u, engine->storage_prefetch_queue_depth.load(std::memory_order_relaxed));
    if (engine->storage_async_prefetch_pending.load(std::memory_order_relaxed) >= depth) {
        engine->storage_async_prefetch_dropped.fetch_add(1, std::memory_order_relaxed);
        return moe_storage_prefetch_inline_on_overflow_enabled() ? 0 : 1;
    }
    moe_storage_prefetch_request_t req{};
    req.shard = shard;
    req.mapped_base = mapped_base;
    req.offset = offset;
    req.bytes = bytes;
    req.profile = engine->storage_profile.load(std::memory_order_relaxed);
    req.gpu_consumer = (uint8_t)moe_storage_io_gpu_consumer_active(engine);
    {
        std::lock_guard<std::mutex> lock(engine->storage_prefetch_queue_mutex);
        if (engine->storage_prefetch_stop.load(std::memory_order_acquire)) return 0;
        if (engine->storage_prefetch_queue.size() >= (size_t)depth) {
            engine->storage_async_prefetch_dropped.fetch_add(1, std::memory_order_relaxed);
            if (!moe_storage_prefetch_inline_on_overflow_enabled()) {
                return 1;  // Drop prefetch instead of stalling token compute.
            }
            return 0;
        }
        engine->storage_prefetch_queue.push_back(req);
        engine->storage_async_prefetch_enqueued.fetch_add(1, std::memory_order_relaxed);
        engine->storage_async_prefetch_pending.fetch_add(1, std::memory_order_relaxed);
    }
    engine->storage_prefetch_queue_cv.notify_one();
    return 1;
}

static void moe_storage_io_prefetch_range_execute(
    moe_pc_engine* engine,
    mmap_context_t* shard,
    const uint8_t* mapped_base,
    uint64_t offset,
    uint64_t bytes,
    uint64_t* bytes_io);

static void moe_storage_io_apply_profile(moe_pc_engine* engine) {
    if (!engine) return;
    const uint32_t parsed_profile = moe_storage_io_parse_profile();
    const uint32_t profile = moe_storage_io_auto_profile_for_model_root(engine, parsed_profile);
    engine->storage_profile.store(profile, std::memory_order_relaxed);
    engine->storage_iouring_available.store(engine->ring_adapter.initialized ? 1 : 0, std::memory_order_relaxed);
    engine->storage_iouring_enabled.store((engine->ring_adapter.initialized && moe_storage_io_iouring_default_enabled(engine)) ? 1 : 0, std::memory_order_relaxed);
    engine->storage_gpu_staging_plan_ready.store(moe_storage_io_gpu_consumer_active(engine) ? 1 : 0, std::memory_order_relaxed);
    engine->storage_prefetch_workers.store(moe_storage_io_default_workers(profile), std::memory_order_relaxed);
    engine->storage_prefetch_queue_depth.store(moe_storage_io_default_queue_depth(profile), std::memory_order_relaxed);

    // Storage I/O planning is common to CPU and GPU/offload.  CPU-only uses it to
    // keep page faults out of the matvec path.  GPU/offload uses the same merged
    // span hints to make host staging reads less random.  RAM disk needs almost
    // none of this because the OS already serves pages from memory.
    if (profile == moe_STORAGE_PROFILE_RAMDISK) {
        engine->io_config.enable_mmap_prefetch = 0;
        return;
    }
    engine->io_config.enable_mmap_prefetch = 1;
    const uint32_t qd = moe_storage_io_default_queue_depth(profile);
    if (engine->io_config.max_prefetch_queue == 0 || moe_storage_always_env_enabled()) {
        engine->io_config.max_prefetch_queue = std::max<uint32_t>(engine->io_config.max_prefetch_queue, qd);
    }
    if (moe_storage_always_env_enabled() && engine->io_config.prefetch_window_layers < 2u && profile != moe_STORAGE_PROFILE_RAMDISK) {
        engine->io_config.prefetch_window_layers = 2u;
    }
}


static int moe_storage_io_profile_is_fast_ssd(uint32_t profile) {
    return profile == moe_STORAGE_PROFILE_NVME || profile == moe_STORAGE_PROFILE_SATA_SSD;
}

static int moe_storage_io_iouring_default_enabled(const moe_pc_engine* engine) {
    if (!engine) return 0;
    if (moe_cpu_storage_env_false("STORAGELLM_STORAGE_IOURING")) return 0;
    if (moe_cpu_storage_env_truthy("STORAGELLM_STORAGE_IOURING")) return 1;
    const uint32_t profile = engine->storage_profile.load(std::memory_order_relaxed);
    // io_uring/IORing is a fast SSD/NVMe path.  Keep it off for network/HDD by
    // default because queueing random reads can make those devices slower.
    return moe_storage_io_profile_is_fast_ssd(profile) ? 1 : 0;
}

static int moe_storage_io_gpu_consumer_active(const moe_pc_engine* engine) {
    if (!engine) return 0;
    if (engine->gpu_path_disabled_for_cpu_only.load(std::memory_order_relaxed)) return 0;
    if (!moe_storage_runtime_gpu_ram_path_has_device_memory(engine)) return 0;
    if (engine->backend_caps.supports_unified_memory && engine->backend_caps.unified_mmap_gpu_accessible) return 0;
    return engine->backend_caps.supports_pinned_host || engine->backend_caps.supports_fixed_read_staging ||
           engine->io_config.enable_directstorage || engine->io_config.enable_gpudirect_storage;
}

static int moe_storage_io_try_iouring_warm_range(
    moe_pc_engine* engine,
    mmap_context_t* shard,
    uint64_t offset,
    uint64_t bytes,
    uint64_t* bytes_io
) {
    if (!engine || !shard || bytes == 0) return 0;
    if (!engine->ring_adapter.initialized || !moe_storage_io_iouring_default_enabled(engine)) {
        if (engine) engine->storage_iouring_fallback_count.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    const uint64_t shard_size = (uint64_t)mmap_get_size(shard);
    if (offset > shard_size || bytes > shard_size - offset) return 0;
    if (!engine->staging || engine->staging_slots.empty() || engine->staging_slot_size == 0) {
        engine->storage_iouring_fallback_count.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    // Use the already registered staging slots as a warm buffer.  This is shared
    // by CPU and GPU: for CPU it pulls pages into the kernel cache before the
    // matvec thread touches them; for GPU it feeds the same pinned/staging pool
    // used by the H2D path.  We only use it from prefetch hooks, not from the dot
    // hot loop.
    const uint64_t max_chunk = std::min<uint64_t>(engine->staging_slot_size, 8ull * 1024ull * 1024ull);
    if (max_chunk == 0) return 0;
    uint64_t done = 0;
    uint32_t turn = (uint32_t)(engine->storage_iouring_submit_count.load(std::memory_order_relaxed) + 1u);
    while (done < bytes) {
        const uint64_t chunk = std::min<uint64_t>(max_chunk, bytes - done);
        const uint32_t slot = acquire_staging_slot(engine, 0u, turn++, chunk, true);
        if (slot == UINT32_MAX || !staging_slot_valid(engine, slot)) {
            engine->storage_iouring_fallback_count.fetch_add(1, std::memory_order_relaxed);
            if (slot != UINT32_MAX) release_staging_slot(engine, slot);
            return done > 0 ? 1 : 0;
        }
        const uint32_t io_gen = engine->prefer_overlapped_io_generation.load(std::memory_order_relaxed);
        void* user_data = reinterpret_cast<void*>((uintptr_t)std::max<uint32_t>(1u, io_gen));
#ifdef _WIN32
        void* file_handle = shard->hFile;
#else
        void* file_handle = nullptr;
#endif
        int submitted = engine->ring_adapter.prep_read_fixed(
            shard->fd,
            file_handle,
            engine->staging_slots[slot].buf,
            (size_t)chunk,
            offset + done,
            slot,
            user_data,
            0);
        if (submitted) {
            engine->ring_adapter.submit();
            void* got = engine->ring_adapter.wait_completion();
            if (got) {
                engine->storage_iouring_submit_count.fetch_add(1, std::memory_order_relaxed);
                engine->storage_iouring_bytes.fetch_add(chunk, std::memory_order_relaxed);
                if (bytes_io) *bytes_io += chunk;
                done += chunk;
            } else {
                engine->storage_iouring_fallback_count.fetch_add(1, std::memory_order_relaxed);
                release_staging_slot(engine, slot);
                return done > 0 ? 1 : 0;
            }
        } else {
            engine->storage_iouring_fallback_count.fetch_add(1, std::memory_order_relaxed);
            release_staging_slot(engine, slot);
            return done > 0 ? 1 : 0;
        }
        release_staging_slot(engine, slot);
    }
    return done > 0 ? 1 : 0;
}

static void moe_storage_io_prefetch_range_execute(
    moe_pc_engine* engine,
    mmap_context_t* shard,
    const uint8_t* mapped_base,
    uint64_t offset,
    uint64_t bytes,
    uint64_t* bytes_io
) {
    if (!engine || !shard || !mapped_base || bytes == 0) return;
    const uint64_t shard_size = (uint64_t)mmap_get_size(shard);
    if (offset > shard_size || bytes > shard_size - offset) return;

    const int gpu_consumer = moe_storage_io_gpu_consumer_active(engine);
    if (gpu_consumer) {
        engine->storage_gpu_staging_plan_ready.store(1, std::memory_order_relaxed);
        engine->storage_gpu_staging_hook_count.fetch_add(1, std::memory_order_relaxed);
        engine->storage_gpu_staging_bytes.fetch_add(bytes, std::memory_order_relaxed);
    } else {
        engine->storage_cpu_pagecache_hook_count.fetch_add(1, std::memory_order_relaxed);
        engine->storage_cpu_pagecache_bytes.fetch_add(bytes, std::memory_order_relaxed);
    }

    // Fast SSD/NVMe path: use io_uring/IORing registered staging buffers when
    // available.  Hints below remain as fallback/extra readahead; they are cheap
    // and also cover platforms without io_uring.
    if (moe_storage_io_try_iouring_warm_range(engine, shard, offset, bytes, bytes_io)) {
        // io_uring already touched the storage path; still keep madvise/fadvise
        // optional below because those affect mmap/page-cache policy.
    }

#if defined(__linux__)
    const int fd = shard->fd;
    if (fd >= 0 && moe_storage_io_enabled_default_on("STORAGELLM_STORAGE_POSIX_FADVISE")) {
        if (posix_fadvise(fd, (off_t)offset, (off_t)bytes, POSIX_FADV_WILLNEED) == 0) {
            engine->storage_io_fadvise_count.fetch_add(1, std::memory_order_relaxed);
        }
    }
    if (fd >= 0 && moe_storage_io_enabled_default_on("STORAGELLM_STORAGE_READAHEAD")) {
        // readahead is stronger than fadvise but still asynchronous enough for
        // SSD/NVMe.  Users can disable it on slow/network filesystems.
        if (readahead(fd, (off64_t)offset, (size_t)bytes) == 0) {
            engine->storage_io_readahead_count.fetch_add(1, std::memory_order_relaxed);
        }
    }
#else
    (void)engine;
#endif

    if (moe_storage_io_enabled_default_on("STORAGELLM_STORAGE_MADVISE")) {
        moe_cpu_storage_hint_mapped_range(mapped_base + offset, bytes);
        engine->storage_io_madvise_count.fetch_add(1, std::memory_order_relaxed);
    }

    if (moe_cpu_storage_page_touch_enabled()) {
        volatile const uint8_t* p = mapped_base + offset;
        uint64_t sink = 0;
        const uint64_t step = 64ull * 1024ull;
        for (uint64_t o = 0; o < bytes; o += step) {
            sink += p[o];
        }
        sink += p[bytes - 1u];
        (void)sink;
        engine->storage_io_page_touch_count.fetch_add(1, std::memory_order_relaxed);
    }

    engine->storage_io_prefetch_count.fetch_add(1, std::memory_order_relaxed);
    engine->storage_io_prefetch_bytes.fetch_add(bytes, std::memory_order_relaxed);
    if (bytes_io) *bytes_io += bytes;
}

static void moe_storage_io_prefetch_range(
    moe_pc_engine* engine,
    mmap_context_t* shard,
    const uint8_t* mapped_base,
    uint64_t offset,
    uint64_t bytes,
    uint64_t* bytes_io
) {
    if (engine && !g_moe_storage_prefetch_worker_tls &&
        moe_storage_async_prefetch_enqueue(engine, shard, mapped_base, offset, bytes)) {
        // Queue-only path: do not block compute on storage completion.  Accounting
        // happens in the worker when the actual warmup call executes.
        return;
    }
    if (engine) engine->storage_async_prefetch_inline.fetch_add(1, std::memory_order_relaxed);
    moe_storage_io_prefetch_range_execute(engine, shard, mapped_base, offset, bytes, bytes_io);
}

static void moe_storage_io_prefetch_hot_view_range(
    moe_pc_engine* engine,
    const moe_cpu_hot_tensor_view_t& view,
    uint64_t offset,
    uint64_t bytes,
    uint64_t* bytes_io
) {
    if (!view.ready || !view.base || !view.shard || !view.rec) return;
    moe_storage_io_prefetch_range(engine, view.shard, view.base, offset, bytes, bytes_io);
}

static void moe_storage_io_build_span_plan(moe_pc_engine* engine) {
    if (!engine || engine->storage_io_plan_ready.load(std::memory_order_acquire)) return;
    if (!engine->cpu_hot_tensor_views_ready.load(std::memory_order_acquire) ||
        engine->dynamic_shape.num_hidden_layers == 0) return;

    moe_storage_io_apply_profile(engine);
    const uint32_t profile = engine->storage_profile.load(std::memory_order_relaxed);
    const uint64_t merge_gap = moe_storage_io_merge_gap_bytes(profile);
    const uint32_t layers = engine->dynamic_shape.num_hidden_layers;
    std::vector<std::vector<moe_storage_io_span_t>> by_layer(layers);

    for (uint32_t i = 0; i < engine->cpu_hot_tensor_views.size(); ++i) {
        const moe_cpu_hot_tensor_view_t& view = engine->cpu_hot_tensor_views[i];
        if (!view.ready || !view.rec || !view.shard || !view.base || view.rec->info.weight_byte_length == 0) continue;
        const uint32_t layer = view.rec->info.layer;
        if (layer >= layers) continue;
        moe_storage_io_span_t span{};
        span.view_index = i;
        span.layer = layer;
        span.offset = view.base_offset;
        span.bytes = view.rec->info.weight_byte_length;
        by_layer[layer].push_back(span);
    }

    engine->storage_layer_span_offsets.assign((size_t)layers + 1u, 0u);
    engine->storage_io_spans.clear();
    uint64_t merged_count = 0;
    for (uint32_t layer = 0; layer < layers; ++layer) {
        auto& spans = by_layer[layer];
        std::sort(spans.begin(), spans.end(), [engine](const moe_storage_io_span_t& a, const moe_storage_io_span_t& b) {
            const moe_cpu_hot_tensor_view_t& va = engine->cpu_hot_tensor_views[a.view_index];
            const moe_cpu_hot_tensor_view_t& vb = engine->cpu_hot_tensor_views[b.view_index];
            if (va.shard != vb.shard) return va.shard < vb.shard;
            if (a.offset != b.offset) return a.offset < b.offset;
            return a.bytes < b.bytes;
        });
        engine->storage_layer_span_offsets[layer] = (uint32_t)engine->storage_io_spans.size();
        for (const moe_storage_io_span_t& s : spans) {
            if (!engine->storage_io_spans.empty()) {
                moe_storage_io_span_t& prev = engine->storage_io_spans.back();
                const moe_cpu_hot_tensor_view_t& pv = engine->cpu_hot_tensor_views[prev.view_index];
                const moe_cpu_hot_tensor_view_t& cv = engine->cpu_hot_tensor_views[s.view_index];
                const uint64_t prev_end = prev.offset + prev.bytes;
                if (prev.layer == s.layer && pv.shard == cv.shard && s.offset >= prev.offset &&
                    s.offset <= prev_end + merge_gap) {
                    const uint64_t end = std::max<uint64_t>(prev_end, s.offset + s.bytes);
                    prev.bytes = end - prev.offset;
                    prev.merged = 1u;
                    ++merged_count;
                    continue;
                }
            }
            engine->storage_io_spans.push_back(s);
        }
    }
    engine->storage_layer_span_offsets[layers] = (uint32_t)engine->storage_io_spans.size();
    engine->storage_layer_prefetch_epoch.assign(layers, 0u);
    engine->storage_io_span_count.store((uint64_t)engine->storage_io_spans.size(), std::memory_order_relaxed);
    engine->storage_io_merged_span_count.store(merged_count, std::memory_order_relaxed);
    engine->storage_io_plan_ready.store(!engine->storage_io_spans.empty() ? 1 : 0, std::memory_order_release);
}


static void moe_storage_io_build_expert_index(moe_pc_engine* engine) {
    if (!engine || engine->storage_expert_index_ready.load(std::memory_order_acquire)) return;
    if (!engine->cpu_hot_tensor_views_ready.load(std::memory_order_acquire)) return;
    const moe_storage_constants_t c = moe_engine_storage_constants_or_static(engine);
    const uint32_t layers = c.num_hidden_layers ? c.num_hidden_layers : engine->dynamic_shape.num_hidden_layers;
    const uint32_t experts = c.experts_per_moe_layer ? c.experts_per_moe_layer : engine->dynamic_shape.experts_per_moe_layer;
    if (layers == 0 || experts == 0) return;

    const size_t bucket_count = (size_t)layers * (size_t)experts;
    std::vector<std::vector<uint32_t>> buckets(bucket_count);
    for (uint32_t i = 0; i < engine->cpu_hot_tensor_views.size(); ++i) {
        const moe_cpu_hot_tensor_view_t& view = engine->cpu_hot_tensor_views[i];
        if (!view.ready || !view.rec || !view.shard || !view.base) continue;
        const moe_tensor_record& rec = *view.rec;
        if (rec.info.weight_byte_length == 0) continue;
        if (!(rec.info.proj == moe_PROJ_GATE || rec.info.proj == moe_PROJ_UP || rec.info.proj == moe_PROJ_DOWN)) continue;
        const uint32_t layer = rec.info.layer;
        const uint32_t expert = rec.info.expert;
        if (layer >= layers || expert >= experts) continue;
        buckets[(size_t)layer * experts + expert].push_back(i);
    }

    engine->storage_expert_view_offsets.assign(bucket_count + 1u, 0u);
    engine->storage_expert_prefetch_epoch.assign(bucket_count, 0u);
    engine->storage_expert_view_indices.clear();
    uint64_t nonempty = 0;
    uint64_t view_count = 0;
    for (size_t b = 0; b < bucket_count; ++b) {
        auto& v = buckets[b];
        std::sort(v.begin(), v.end(), [engine](uint32_t a, uint32_t bidx) {
            const moe_cpu_hot_tensor_view_t& va = engine->cpu_hot_tensor_views[a];
            const moe_cpu_hot_tensor_view_t& vb = engine->cpu_hot_tensor_views[bidx];
            if (va.shard != vb.shard) return va.shard < vb.shard;
            if (va.base_offset != vb.base_offset) return va.base_offset < vb.base_offset;
            return va.row_bytes < vb.row_bytes;
        });
        if (!v.empty()) ++nonempty;
        engine->storage_expert_view_offsets[b] = (uint32_t)engine->storage_expert_view_indices.size();
        for (uint32_t idx : v) {
            engine->storage_expert_view_indices.push_back(idx);
            ++view_count;
        }
    }
    engine->storage_expert_view_offsets[bucket_count] = (uint32_t)engine->storage_expert_view_indices.size();
    engine->storage_expert_index_entry_count.store(nonempty, std::memory_order_relaxed);
    engine->storage_expert_index_view_count.store(view_count, std::memory_order_relaxed);
    engine->storage_expert_index_ready.store(view_count > 0 ? 1 : 0, std::memory_order_release);
}

static uint64_t moe_storage_io_warm_layer_spans(
    moe_pc_engine* engine,
    uint32_t layer,
    uint64_t budget_bytes
) {
    if (!engine || !engine->storage_io_plan_ready.load(std::memory_order_acquire)) return 0;
    if (layer + 1u >= engine->storage_layer_span_offsets.size()) return 0;
    const uint32_t begin = engine->storage_layer_span_offsets[layer];
    const uint32_t end = engine->storage_layer_span_offsets[layer + 1u];
    uint64_t bytes = 0;
    uint64_t count = 0;
    for (uint32_t i = begin; i < end; ++i) {
        if (i >= engine->storage_io_spans.size()) break;
        const moe_storage_io_span_t& span = engine->storage_io_spans[i];
        if (span.view_index >= engine->cpu_hot_tensor_views.size()) continue;
        const moe_cpu_hot_tensor_view_t& view = engine->cpu_hot_tensor_views[span.view_index];
        if (budget_bytes > 0 && bytes >= budget_bytes) break;
        const uint64_t before = bytes;
        moe_storage_io_prefetch_hot_view_range(engine, view, span.offset, span.bytes, &bytes);
        if (bytes != before) ++count;
    }
    return count;
}


static void moe_cpu_storage_prefetch_hot_view(
    moe_pc_engine* engine,
    const moe_cpu_hot_tensor_view_t& view,
    uint64_t* bytes_io
) {
    if (!engine || !view.ready || !view.base || !view.rec || view.rec->info.weight_byte_length == 0) {
        return;
    }
    const uint64_t offset = view.base_offset;
    const uint64_t bytes = view.rec->info.weight_byte_length;
    moe_storage_io_prefetch_hot_view_range(engine, view, offset, bytes, bytes_io);
}

static void moe_cpu_storage_build_layer_hot_plan(moe_pc_engine* engine) {
    if (!engine || engine->cpu_layer_hot_plan_ready.load(std::memory_order_acquire)) {
        return;
    }
    if (!engine->cpu_hot_tensor_views_ready.load(std::memory_order_acquire) ||
        engine->dynamic_shape.num_hidden_layers == 0) {
        return;
    }
    const uint32_t layers = engine->dynamic_shape.num_hidden_layers;
    std::vector<std::vector<uint32_t>> by_layer(layers);
    for (uint32_t i = 0; i < engine->cpu_hot_tensor_views.size(); ++i) {
        const moe_cpu_hot_tensor_view_t& view = engine->cpu_hot_tensor_views[i];
        if (!view.ready || !view.rec || view.rec->info.weight_byte_length == 0) {
            continue;
        }
        const uint32_t layer = view.rec->info.layer;
        if (layer >= layers) {
            continue;
        }
        // Keep all executable layer-local tensors in the plan.  Sorting below
        // makes router/common/hot tensors land first under the prefetch budget.
        by_layer[layer].push_back(i);
    }
    uint64_t flat_count = 0;
    engine->cpu_layer_hot_offsets.assign((size_t)layers + 1u, 0u);
    for (uint32_t l = 0; l < layers; ++l) {
        auto& vec = by_layer[l];
        std::sort(vec.begin(), vec.end(), [engine](uint32_t a, uint32_t b) {
            const moe_tensor_record* ra = engine->cpu_hot_tensor_views[a].rec;
            const moe_tensor_record* rb = engine->cpu_hot_tensor_views[b].rec;
            const uint32_t pa = ra ? ((uint32_t)ra->runtime_priority * 4u + (uint32_t)ra->prefetch_priority) : 100000u;
            const uint32_t pb = rb ? ((uint32_t)rb->runtime_priority * 4u + (uint32_t)rb->prefetch_priority) : 100000u;
            if (pa != pb) return pa < pb;
            const uint32_t ea = ra ? (uint32_t)ra->execution_order : 100000u;
            const uint32_t eb = rb ? (uint32_t)rb->execution_order : 100000u;
            if (ea != eb) return ea < eb;
            const uint32_t ha = ra ? (uint32_t)ra->hotset_rank : 100000u;
            const uint32_t hb = rb ? (uint32_t)rb->hotset_rank : 100000u;
            if (ha != hb) return ha < hb;
            return a < b;
        });
        engine->cpu_layer_hot_offsets[l] = (uint32_t)flat_count;
        flat_count += vec.size();
    }
    engine->cpu_layer_hot_offsets[layers] = (uint32_t)flat_count;
    engine->cpu_layer_hot_tensor_indices.clear();
    engine->cpu_layer_hot_tensor_indices.reserve((size_t)flat_count);
    for (uint32_t l = 0; l < layers; ++l) {
        for (uint32_t idx : by_layer[l]) {
            engine->cpu_layer_hot_tensor_indices.push_back(idx);
        }
    }
    engine->cpu_layer_prefetch_epoch.assign(layers, 0u);
    engine->cpu_layer_hot_tensor_count.store(flat_count, std::memory_order_relaxed);
    engine->cpu_layer_hot_plan_ready.store(flat_count > 0 ? 1 : 0, std::memory_order_release);
    moe_storage_io_build_span_plan(engine);
}

static uint64_t moe_cpu_storage_warm_layer_hot_views(
    moe_pc_engine* engine,
    uint32_t layer,
    uint64_t budget_bytes
) {
    if (!engine || !engine->cpu_layer_hot_plan_ready.load(std::memory_order_acquire)) {
        return 0;
    }
    if (layer + 1u >= engine->cpu_layer_hot_offsets.size()) {
        return 0;
    }
    const uint32_t begin = engine->cpu_layer_hot_offsets[layer];
    const uint32_t end = engine->cpu_layer_hot_offsets[layer + 1u];
    uint64_t bytes = 0;
    uint64_t count = 0;
    for (uint32_t p = begin; p < end; ++p) {
        if (p >= engine->cpu_layer_hot_tensor_indices.size()) break;
        const uint32_t tensor_idx = engine->cpu_layer_hot_tensor_indices[p];
        if (tensor_idx >= engine->cpu_hot_tensor_views.size()) continue;
        const moe_cpu_hot_tensor_view_t& view = engine->cpu_hot_tensor_views[tensor_idx];
        if (!view.ready || !view.rec) continue;
        if (budget_bytes > 0 && bytes >= budget_bytes) break;
        const uint64_t before = bytes;
        moe_cpu_storage_prefetch_hot_view(engine, view, &bytes);
        if (bytes != before) {
            ++count;
        }
    }
    if (count) {
        engine->cpu_layer_hot_prefetch_count.fetch_add(count, std::memory_order_relaxed);
        engine->cpu_layer_hot_prefetch_bytes.fetch_add(bytes, std::memory_order_relaxed);
    }
    return count;
}

static void moe_cpu_storage_schedule_layer_hot_prefetch(
    moe_pc_engine* engine,
    uint32_t layer
) {
    if (!engine || !engine->cpu_storage_engine_ready.load(std::memory_order_relaxed)) {
        return;
    }
    if (!engine->cpu_layer_hot_plan_ready.load(std::memory_order_acquire)) {
        return;
    }
    const uint32_t epoch = engine->cpu_prefetch_epoch.load(std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(engine->cpu_layer_prefetch_mutex);
        if (layer >= engine->cpu_layer_prefetch_epoch.size()) {
            return;
        }
        if (engine->cpu_layer_prefetch_epoch[layer] == epoch) {
            engine->cpu_layer_prefetch_duplicate_skips.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        engine->cpu_layer_prefetch_epoch[layer] = epoch;
    }
    // Common SSD/NVMe span prefetch first: merged contiguous ranges feed both
    // CPU page-cache and GPU/offload host staging.  Fall back to per-tensor hints
    // only if the span planner was not built.
    const uint64_t budget = moe_cpu_storage_layer_prefetch_budget_bytes();
    if (!moe_storage_io_warm_layer_spans(engine, layer, budget)) {
        moe_cpu_storage_warm_layer_hot_views(engine, layer, budget);
    }
}

static void moe_cpu_storage_build_hot_tensor_views(moe_pc_engine* engine) {
    if (!engine || engine->tensors.empty()) {
        return;
    }
    if (engine->cpu_hot_tensor_views_ready.load(std::memory_order_acquire)) {
        return;
    }
    engine->cpu_hot_tensor_views.clear();
    engine->cpu_hot_tensor_views.resize(engine->tensors.size());
    uint64_t ready = 0;
    uint64_t simple = 0;
    uint64_t bytes = 0;
    for (size_t i = 0; i < engine->tensors.size(); ++i) {
        const moe_tensor_record& rec = engine->tensors[i];
        moe_cpu_hot_tensor_view_t view{};
        view.rec = &engine->tensors[i];
        if (rec.info.weight_byte_length == 0 || rec.weight_row_bytes == 0 || rec.info.rows == 0 || rec.info.cols == 0) {
            engine->cpu_hot_tensor_views[i] = view;
            continue;
        }
        const int declared = tensor_weight_format(rec);
        const int fmt = moe_effective_tensor_weight_format_for_row_bytes(&rec, declared);
        if (!moe_dot_weight_format_supported(fmt)) {
            engine->cpu_hot_tensor_views[i] = view;
            continue;
        }
        mmap_context_t* shard = get_or_map_tensor_shard(engine, rec);
        const uint8_t* base = shard ? static_cast<const uint8_t*>(mmap_get_ptr(shard, 0)) : nullptr;
        const uint64_t shard_size = shard ? (uint64_t)mmap_get_size(shard) : 0ull;
        if (!shard || !base || shard_size == 0) {
            engine->cpu_hot_tensor_views[i] = view;
            continue;
        }
        const uint64_t last_row = (uint64_t)rec.info.rows - 1ull;
        const uint64_t last_offset = rec.info.weight_byte_offset + last_row * rec.weight_row_bytes;
        if (rec.info.weight_byte_offset > shard_size ||
            last_offset > shard_size ||
            rec.weight_row_bytes > shard_size - last_offset) {
            engine->cpu_hot_tensor_views[i] = view;
            continue;
        }
        view.shard = shard;
        view.base = base;
        view.shard_size = shard_size;
        view.base_offset = rec.info.weight_byte_offset;
        view.row_bytes = rec.weight_row_bytes;
        view.rows = rec.info.rows;
        view.cols = rec.info.cols;
        view.weight_format = fmt;
        view.simple_dot = moe_cpu_storage_format_simple_dot(fmt);
        view.ready = 1u;
        ++ready;
        simple += view.simple_dot ? 1u : 0u;
        bytes += rec.info.weight_byte_length;
        if (moe_cpu_storage_tensor_is_hot_for_hint(rec)) {
            uint64_t ignored_bytes = 0;
            moe_storage_io_prefetch_hot_view_range(engine, view, view.base_offset, rec.info.weight_byte_length, &ignored_bytes);
        }
        engine->cpu_hot_tensor_views[i] = view;
    }
    engine->cpu_hot_tensor_view_count.store(ready, std::memory_order_relaxed);
    engine->cpu_hot_tensor_simple_dot_count.store(simple, std::memory_order_relaxed);
    engine->cpu_hot_tensor_bytes.store(bytes, std::memory_order_relaxed);
    engine->cpu_hot_tensor_views_ready.store(ready > 0 ? 1 : 0, std::memory_order_release);
    moe_cpu_storage_build_layer_hot_plan(engine);
}

static int moe_cpu_storage_fast_view_for_record(
    moe_pc_engine* engine,
    const moe_tensor_record* rec,
    uint32_t expected_cols,
    moe_tensor_row_fast_view_t* out
) {
    if (!engine || !rec || !out || expected_cols == 0 || rec->info.cols != expected_cols) {
        return 0;
    }
    if (!engine->cpu_hot_tensor_views_ready.load(std::memory_order_acquire)) {
        return 0;
    }
    const moe_tensor_record* base_rec = engine->tensors.empty() ? nullptr : engine->tensors.data();
    if (!base_rec || rec < base_rec || rec >= base_rec + engine->tensors.size()) {
        return 0;
    }
    const size_t idx = (size_t)(rec - base_rec);
    if (idx >= engine->cpu_hot_tensor_views.size()) {
        return 0;
    }
    const moe_cpu_hot_tensor_view_t& v = engine->cpu_hot_tensor_views[idx];
    if (!v.ready || v.rec != rec || !v.base || !v.shard || v.cols != expected_cols) {
        return 0;
    }
    *out = moe_tensor_row_fast_view_t{};
    out->rec = rec;
    out->shard = v.shard;
    out->base = v.base;
    out->shard_size = v.shard_size;
    out->base_offset = v.base_offset;
    out->row_bytes = v.row_bytes;
    out->rows = v.rows;
    out->cols = v.cols;
    out->weight_format = v.weight_format;
    out->simple_dot = moe_cpu_simple_row_dot_for_format(v.weight_format);
    engine->cpu_hot_tensor_reused.fetch_add(1, std::memory_order_relaxed);
    return 1;
}

static void moe_cpu_storage_build_execution_plan(moe_pc_engine* engine) {
    if (!engine) return;
    if (engine->cpu_storage_execution_plan_ready.load(std::memory_order_acquire)) return;
    if (!engine->dynamic_shape_valid.load(std::memory_order_relaxed) || engine->tensors.empty()) return;

    moe_storage_io_apply_profile(engine);

    uint64_t tensors = 0;
    uint64_t expert_tensors = 0;
    uint64_t bytes = 0;
    uint64_t supported = 0;
    for (const moe_tensor_record& rec : engine->tensors) {
        if (rec.info.weight_byte_length == 0) continue;
        ++tensors;
        bytes += rec.info.weight_byte_length;
        supported += moe_cpu_storage_weight_format_countable(rec);
        if (rec.info.proj == moe_PROJ_GATE || rec.info.proj == moe_PROJ_UP || rec.info.proj == moe_PROJ_DOWN) {
            ++expert_tensors;
        }
    }
    if (tensors == 0) return;
    engine->cpu_execution_plan_layers.store(engine->dynamic_shape.num_hidden_layers, std::memory_order_relaxed);
    engine->cpu_execution_plan_tensors.store(tensors, std::memory_order_relaxed);
    engine->cpu_execution_plan_expert_tensors.store(expert_tensors, std::memory_order_relaxed);
    engine->cpu_execution_plan_bytes.store(bytes, std::memory_order_relaxed);
    engine->cpu_quant_registry_ready.store(supported == tensors ? 1 : 0, std::memory_order_relaxed);
    const int storage_plan_ready =
        engine->juju_runtime_access_plan_ready.load(std::memory_order_relaxed) ||
        engine->offload_gguf_metadata_json.find("\"storage_execution_plan\"") != std::string::npos ||
        engine->offload_gguf_metadata_json.find("\"runtime_access_plan\"") != std::string::npos;
    engine->storage_format_plan_available.store(storage_plan_ready ? 1 : 0, std::memory_order_relaxed);
    moe_cpu_storage_build_hot_tensor_views(engine);
    moe_storage_io_build_span_plan(engine);
    moe_storage_io_build_expert_index(engine);
    engine->cpu_storage_execution_plan_ready.store(1, std::memory_order_release);
    engine->cpu_storage_engine_ready.store(1, std::memory_order_release);
}

static void moe_cpu_storage_select_forward_adapter(moe_pc_engine* engine) {
    if (!engine) return;
    const uint32_t runtime_path = moe_storage_runtime_path_cached(engine);
    const int cpu_only = runtime_path == moe_STORAGE_RUNTIME_PATH_CPU_RAM;
    engine->cpu_only_runtime.store(cpu_only ? 1 : 0, std::memory_order_relaxed);
    engine->gpu_path_disabled_for_cpu_only.store(cpu_only ? 1 : 0, std::memory_order_relaxed);
    if (cpu_only) {
        moe_cpu_storage_disable_gpu_paths(engine);
        moe_cpu_storage_build_execution_plan(engine);
        if (engine->cpu_storage_engine_ready.load(std::memory_order_relaxed)) {
            // CPU-storage is a real fast path, not graph_ir.  Do not advertise or
            // enter graph_ir here unless the operator explicitly requested a slow
            // fallback elsewhere.
            engine->forward_adapter.store(moe_FORWARD_ADAPTER_CPU_STORAGE, std::memory_order_relaxed);
            engine->forward_adapter_executable.store(1, std::memory_order_relaxed);
            engine->graph_ir_used_for_forward.store(0, std::memory_order_relaxed);
        } else if (!moe_cpu_storage_explicit_slow_fallback_allowed()) {
            engine->forward_adapter.store(moe_FORWARD_ADAPTER_CPU_STORAGE, std::memory_order_relaxed);
            engine->forward_adapter_executable.store(0, std::memory_order_relaxed);
            engine->graph_ir_used_for_forward.store(0, std::memory_order_relaxed);
        }
        return;
    }

    const int gpu_ready = runtime_path == moe_STORAGE_RUNTIME_PATH_GPU_RAM;
    if (gpu_ready) {
        engine->cpu_only_runtime.store(0, std::memory_order_relaxed);
        engine->gpu_path_disabled_for_cpu_only.store(0, std::memory_order_relaxed);
        // Storage span planning is backend-common: GPU/offload still reads
        // weights from SSD into host/device staging and benefits from the same
        // merged layer-span hints.
        moe_cpu_storage_build_execution_plan(engine);
        engine->forward_adapter.store(moe_FORWARD_ADAPTER_GPU_OFFLOAD, std::memory_order_relaxed);
        engine->forward_adapter_executable.store(1, std::memory_order_relaxed);
        engine->graph_ir_used_for_forward.store(0, std::memory_order_relaxed);
    }
}

static void moe_cpu_storage_ensure_runtime(moe_pc_engine* engine) {
    if (!engine) return;
    moe_cpu_storage_select_forward_adapter(engine);
}

static int moe_cpu_storage_forward_allowed(moe_pc_engine* engine, moe_generation_stats_t* stats) {
    if (!engine) return 0;
    moe_cpu_storage_ensure_runtime(engine);
    if (!moe_cpu_storage_runtime_detected(engine)) return 1;
    if (engine->cpu_storage_engine_ready.load(std::memory_order_relaxed)) return 1;
    if (moe_cpu_storage_explicit_slow_fallback_allowed()) return 1;
    moe_generation_set_error(stats, "CPU-only runtime detected but CPUStorageEngine is not ready; refusing graph_ir slow fallback");
    return 0;
}

static void moe_cpu_storage_begin_token(moe_pc_engine* engine, uint32_t position, uint32_t token_id) {
    if (!engine || !engine->cpu_storage_engine_ready.load(std::memory_order_relaxed)) return;
    (void)position;
    (void)token_id;
    const int graph_contract_active =
        engine->forward_adapter.load(std::memory_order_relaxed) == moe_FORWARD_ADAPTER_GRAPH_IR &&
        engine->offload_graph_ir_required.load(std::memory_order_relaxed) &&
        engine->offload_graph_ir_ready.load(std::memory_order_relaxed);
    engine->graph_ir_used_for_forward.store(graph_contract_active ? 1 : 0, std::memory_order_relaxed);
    engine->generation_phase.store(moe_GEN_PHASE_INIT, std::memory_order_relaxed);
    engine->cpu_prefetch_epoch.fetch_add(1u, std::memory_order_relaxed);
    // Warm the first layers before the loop reaches them.  Existing IO queues
    // fetch required spans; CPU layer-hot plan issues cheap page-cache hints
    // without scanning all tensors.
    const moe_storage_constants_t runtime_c = moe_engine_storage_constants_or_static(engine);
    const uint32_t layers = runtime_c.num_hidden_layers;
    const uint32_t depth = std::max<uint32_t>(1u, std::min<uint32_t>(engine->io_config.prefetch_window_layers ? engine->io_config.prefetch_window_layers : 6u, 8u));
    for (uint32_t l = 0; l < layers && l < depth; ++l) {
        moe_cpu_storage_schedule_layer_hot_prefetch(engine, l);
        moe_enqueue_attention_prefetch(engine, (int32_t)l);
        moe_pc_engine_prefetch_moe_lookahead(engine, (int)l, nullptr, 0, nullptr, 0);
        engine->cpu_prefetch_scheduled.fetch_add(1, std::memory_order_relaxed);
    }
}

static void moe_cpu_storage_begin_layer(moe_pc_engine* engine, uint32_t layer, const moe_storage_constants_t* c) {
    if (!engine || !c || !engine->cpu_storage_engine_ready.load(std::memory_order_relaxed)) return;
    engine->cpu_layer_begin_count.fetch_add(1, std::memory_order_relaxed);
    const uint32_t depth = std::max<uint32_t>(1u, std::min<uint32_t>(engine->io_config.prefetch_window_layers ? engine->io_config.prefetch_window_layers : 6u, 8u));
    for (uint32_t d = 1; d <= depth; ++d) {
        const uint32_t next = layer + d;
        if (next >= c->num_hidden_layers) break;
        moe_cpu_storage_schedule_layer_hot_prefetch(engine, next);
        moe_enqueue_attention_prefetch(engine, (int32_t)next);
        // Shared router/common spans can be prefetched before router top-k is known.
        moe_pc_engine_prefetch_moe_lookahead(engine, (int)next, nullptr, 0, nullptr, 0);
        engine->cpu_prefetch_scheduled.fetch_add(1, std::memory_order_relaxed);
    }
}

static void moe_cpu_storage_after_router_selected(
    moe_pc_engine* engine,
    int32_t layer,
    const std::vector<int>& selected_experts
) {
    if (!engine || !engine->cpu_storage_engine_ready.load(std::memory_order_relaxed)) return;
    if (layer < 0 || selected_experts.empty()) return;
    moe_pc_engine_prefetch_moe_lookahead(
        engine,
        (int)layer,
        selected_experts.data(),
        (uint32_t)selected_experts.size(),
        nullptr,
        0);

    // Storage-required mode: router-selected experts are no longer speculative.
    // Use the per-(layer, expert) index built from JUJU/IDX offsets; this avoids
    // scanning every tensor/view for every selected expert on the token hot path.
    if (engine->cpu_hot_tensor_views_ready.load(std::memory_order_acquire)) {
        uint64_t bytes = 0;
        const moe_storage_constants_t c = moe_engine_storage_constants_or_static(engine);
        const uint32_t layers = c.num_hidden_layers ? c.num_hidden_layers : engine->dynamic_shape.num_hidden_layers;
        const uint32_t experts = c.experts_per_moe_layer ? c.experts_per_moe_layer : engine->dynamic_shape.experts_per_moe_layer;
        const bool indexed = engine->storage_expert_index_ready.load(std::memory_order_acquire) &&
            experts > 0 && (uint32_t)layer < layers &&
            engine->storage_expert_view_offsets.size() >= (size_t)layers * ((size_t)experts + 0u);
        if (indexed) {
            for (const int expert_id : selected_experts) {
                if (expert_id < 0 || (uint32_t)expert_id >= experts) continue;
                const size_t b = (size_t)(uint32_t)layer * (size_t)experts + (uint32_t)expert_id;
                if (b + 1u >= engine->storage_expert_view_offsets.size()) continue;
                const uint32_t epoch = engine->cpu_prefetch_epoch.load(std::memory_order_relaxed);
                bool duplicate_bucket = false;
                {
                    std::lock_guard<std::mutex> lock(engine->storage_expert_prefetch_mutex);
                    if (b < engine->storage_expert_prefetch_epoch.size()) {
                        if (engine->storage_expert_prefetch_epoch[b] == epoch) {
                            duplicate_bucket = true;
                        } else {
                            engine->storage_expert_prefetch_epoch[b] = epoch;
                        }
                    }
                }
                if (duplicate_bucket) {
                    engine->storage_selected_expert_duplicate_skips.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                const uint32_t begin = engine->storage_expert_view_offsets[b];
                const uint32_t end = engine->storage_expert_view_offsets[b + 1u];
                for (uint32_t p = begin; p < end && p < engine->storage_expert_view_indices.size(); ++p) {
                    const uint32_t view_idx = engine->storage_expert_view_indices[p];
                    if (view_idx >= engine->cpu_hot_tensor_views.size()) continue;
                    const moe_cpu_hot_tensor_view_t& view = engine->cpu_hot_tensor_views[view_idx];
                    if (!view.ready || !view.rec) continue;
                    moe_storage_io_prefetch_hot_view_range(engine, view, view.base_offset, view.rec->info.weight_byte_length, &bytes);
                }
                engine->storage_selected_expert_index_hits.fetch_add(1, std::memory_order_relaxed);
            }
        } else {
            engine->storage_selected_expert_linear_fallbacks.fetch_add(1, std::memory_order_relaxed);
            for (const int expert_id : selected_experts) {
                for (uint32_t i = 0; i < engine->cpu_hot_tensor_views.size(); ++i) {
                    const moe_cpu_hot_tensor_view_t& view = engine->cpu_hot_tensor_views[i];
                    if (!view.ready || !view.rec) continue;
                    const moe_tensor_record& rec = *view.rec;
                    if ((int32_t)rec.info.layer != layer) continue;
                    if ((int32_t)rec.info.expert != (int32_t)expert_id) continue;
                    if (!(rec.info.proj == moe_PROJ_GATE || rec.info.proj == moe_PROJ_UP || rec.info.proj == moe_PROJ_DOWN)) continue;
                    moe_storage_io_prefetch_hot_view_range(engine, view, view.base_offset, rec.info.weight_byte_length, &bytes);
                }
            }
        }
        engine->storage_selected_expert_prefetch_bytes.fetch_add(bytes, std::memory_order_relaxed);
    }
    engine->cpu_selected_expert_prefetch_scheduled.fetch_add(1, std::memory_order_relaxed);
}
