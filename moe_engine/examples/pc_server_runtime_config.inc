
static int server_env_truthy_local(const char* name) {
    const char* v = std::getenv(name);
    if (!v || !*v || v[0] == '0') return 0;
    if ((v[0] == 'f' || v[0] == 'F') && (v[1] == 'a' || v[1] == 'A')) return 0;
    if ((v[0] == 'n' || v[0] == 'N') && (v[1] == 'o' || v[1] == 'O')) return 0;
    return 1;
}


static int server_strict_fast_path_enabled() {
    return server_env_truthy_local("STORAGELLM_STRICT_FAST_PATH") ||
           server_env_truthy_local("STORAGELLM_FAIL_ON_FALLBACK") ||
           server_env_truthy_local("STORAGELLM_NO_SLOW_FALLBACK");
}

static int server_cpu_storage_max_enabled(const moe_backend_caps_t& caps) {
    return caps.backend == moe_BACKEND_CPU ||
           server_env_truthy_local("STORAGELLM_CPU_ONLY") ||
           server_env_truthy_local("STORAGELLM_CPU_STORAGE_ENGINE") ||
           server_env_truthy_local("STORAGELLM_CPU_STORAGE_MAX") ||
           server_env_truthy_local("STORAGELLM_CPU_PIPELINE");
}

static uint64_t server_auto_ram_budget_bytes(int cpu_storage_mode) {
    const uint64_t available = storagellm::available_ram_bytes();
    const uint64_t gib = 1024ull * 1024ull * 1024ull;
    if (const char* env = std::getenv("STORAGELLM_RAM_RESERVE_BYTES")) {
        char* end = nullptr;
        const unsigned long long parsed = std::strtoull(env, &end, 10);
        if (end && end != env && *end == '\0' && parsed > 0) {
            const uint64_t reserve = (uint64_t)parsed;
            if (available > reserve) {
                return available - reserve;
            }
        }
    }
    const uint64_t reserve =
        available >= 64ull * gib ? 3ull * gib :
        available >= 32ull * gib ? 2ull * gib : 768ull * 1024ull * 1024ull;
    if (available > reserve * 2ull) {
        const uint64_t available_budget = available - reserve;
        if (cpu_storage_mode) {
            const uint64_t total = storagellm::total_ram_bytes();
            const uint64_t total_budget = total > 0 ? (total * 80ull) / 100ull : 0ull;
            const uint64_t reclaim_cap = available + 8ull * gib;
            if (total_budget > available_budget) {
                return std::min<uint64_t>(total_budget, reclaim_cap);
            }
        }
        return available_budget;
    }
    return (available * 9ull) / 10ull;
}

static void apply_detected_budgets(server_options* opts, const moe_backend_caps_t& caps) {
    if (!opts) {
        return;
    }
    const int cpu_storage_mode = server_cpu_storage_max_enabled(caps);
    if (cpu_storage_mode) {
        opts->engine_config.prefer_gpu = 0;
        opts->engine_config.preferred_backend = moe_BACKEND_CPU;
        opts->engine_config.platform = moe_PLATFORM_CPU_ONLY;
        opts->engine_config.vram_budget_bytes = 0;
        opts->vram_budget_override = true;
    }
    if (!opts->vram_budget_override) {
        opts->engine_config.vram_budget_bytes = caps.recommended_vram_cache_bytes;
        if (caps.device_memory_bytes > 0) {
            const uint64_t detected_budget = (caps.device_memory_bytes * 9ull) / 10ull;
            opts->engine_config.vram_budget_bytes = std::max<uint64_t>(
                opts->engine_config.vram_budget_bytes,
                detected_budget
            );
        }
    }
    if (!opts->ram_budget_override) {
        opts->engine_config.ram_budget_bytes = server_auto_ram_budget_bytes(cpu_storage_mode);
    }
    opts->engine_config.allow_db_streaming = 1;
}

static moe_io_config_t make_server_io_config(const server_options& opts, const moe_backend_caps_t& caps) {
    moe_io_config_t io = moe_pc_default_io_config();
    const moe_storage_constants_t* storage = moe_storage_constants();
    const uint64_t expert_min_bytes = storage ? storage->normal_bundle_min_bytes : 8ull * 1024ull * 1024ull;
    const uint64_t expert_max_bytes = storage && storage->normal_bundle_max_bytes ?
        storage->normal_bundle_max_bytes : expert_min_bytes;
    io.enable_compute_io_overlap = 1;
    // CPU storage mode is exactly where the multistage pipeline matters most:
    // the compute thread should keep running while disk/page-cache prefetch warms
    // the next layer.  The previous server override disabled it for CPU and
    // caused the token loop to alternate between compute and I/O waits.
    io.enable_multistage_pipeline = 1;
    io.enable_lookahead_prefetch = 1;
    io.enable_topology_prefetch = 1;
    io.enable_mmap_prefetch = 1;
    io.enable_pinned_host = server_cpu_storage_max_enabled(caps) ? 0 : caps.supports_pinned_host;
    io.enable_directstorage = server_cpu_storage_max_enabled(caps) ? 0 : caps.supports_directstorage;
    io.enable_gpudirect_storage = server_cpu_storage_max_enabled(caps) ? 0 : caps.supports_gpudirect_storage;
    io.enable_bam_nvme = server_cpu_storage_max_enabled(caps) ? 0 : caps.supports_gpu_direct_file_io;
    // Max-speed GPU/offload server path: keep common tensors resident in VRAM
    // when a discrete GPU/cache budget exists so attention/common projections do
    // not bounce through host memory every token.
    io.enable_common_raw_vram_pin = server_cpu_storage_max_enabled(caps) ? 0 : 1;
    io.prefetch_window_layers = std::max<uint32_t>(server_cpu_storage_max_enabled(caps) ? 4u : 6u, io.prefetch_window_layers);
    io.max_prefetch_queue = std::max<uint32_t>(io.max_prefetch_queue, server_cpu_storage_max_enabled(caps) ? 16384u : 32768u);
    io.topology_max_candidates = std::max<uint32_t>(io.topology_max_candidates, 4u);
    io.topology_max_edges_per_source = std::max<uint32_t>(io.topology_max_edges_per_source, 64u);
    io.directstorage_staging_queue_capacity = std::max<uint32_t>(io.directstorage_staging_queue_capacity, 4096u);
    io.directstorage_gpu_queue_capacity = std::max<uint32_t>(io.directstorage_gpu_queue_capacity, 8192u);
    const int cpu_storage_mode = server_cpu_storage_max_enabled(caps);
    io.ram_eviction_target_fill_pct = std::max<uint8_t>(
        io.ram_eviction_target_fill_pct,
        cpu_storage_mode ? 97u : 85u);
    io.ram_eviction_threshold_pct = std::max<uint8_t>(
        io.ram_eviction_threshold_pct,
        cpu_storage_mode ? 98u : 90u);
    if (cpu_storage_mode) {
        const uint32_t hw_cpu = std::max<uint32_t>(1u, std::thread::hardware_concurrency());
        const uint32_t cpu_io_workers =
            hw_cpu >= 32u ? 10u :
            hw_cpu >= 16u ? 6u :
            hw_cpu >= 8u  ? 4u : 1u;
        io.io_worker_threads = cpu_io_workers;
        io.disk_worker_threads = cpu_io_workers;
        io.pinned_worker_threads = 0;
        io.gpu_worker_threads = 0;
        io.prefetch_window_layers = hw_cpu >= 16u ? 6u : 4u;
        io.max_prefetch_queue = hw_cpu >= 16u ? 32768u : 16384u;
        io.pinned_staging_bytes = 0;
        io.directstorage_staging_queue_capacity = 0;
        io.directstorage_gpu_queue_capacity = 0;
        io.allow_required_sync_fallback = 1;
        io.enable_cold_expert_cpu_fallback = 1;
        io.ram_eviction_target_fill_pct = 97;
        io.ram_eviction_threshold_pct = 98;
        io.internal_require_gpu_expert_batch_kernel = 0;
        io.internal_verified_backend_only = 0;
    } else {
        const uint32_t hw_gpu = std::max<uint32_t>(1u, std::thread::hardware_concurrency());
        const uint32_t stream_workers = std::max<uint32_t>(2u, caps.copy_streams ? caps.copy_streams * 2u : 2u);
        io.disk_worker_threads = hw_gpu >= 32u ? 12u : (hw_gpu >= 16u ? 8u : 4u);
        io.pinned_worker_threads = std::min<uint32_t>(16u, std::max<uint32_t>(4u, stream_workers));
        io.gpu_worker_threads = std::min<uint32_t>(16u, std::max<uint32_t>(4u, stream_workers));
        io.gpu_expert_scratch_slots = std::max<uint32_t>(32u, io.gpu_worker_threads * 4u);
    }
    io.direct_upload_min_bytes = expert_min_bytes;
    const uint32_t active_workers =
        io.disk_worker_threads + io.pinned_worker_threads + io.gpu_worker_threads;
    const uint64_t slot_bytes = expert_max_bytes + 4ull * 1024ull * 1024ull;
    const uint64_t worker_slots = std::max<uint32_t>(16u, active_workers * 3u);
    const uint64_t worker_staging = slot_bytes * worker_slots;
    if (!server_cpu_storage_max_enabled(caps)) {
        io.pinned_staging_bytes = std::max<uint64_t>(
            std::max<uint64_t>(caps.recommended_pinned_bytes, worker_staging),
            512ull * 1024ull * 1024ull);
    }
    if (server_strict_fast_path_enabled()) {
        // Strict mode is profile-aware.  CPU-only strict mode refuses hidden
        // storage sync fallback, but it must not require GPU kernels.  GPU/offload
        // strict mode requires the verified fast backend and grouped expert kernel.
        io.allow_required_sync_fallback = 0;
        io.require_bundle_native_experts = 1;
        if (server_cpu_storage_max_enabled(caps)) {
            io.enable_cold_expert_cpu_fallback = 1;
            io.internal_verified_backend_only = 0;
            io.internal_require_gpu_expert_batch_kernel = 0;
        } else {
            io.enable_cold_expert_cpu_fallback = 0;
            io.internal_verified_backend_only = 1;
            io.internal_require_gpu_expert_batch_kernel = 1;
        }
    }
    return io;
}

static void print_optimization_plan(moe_pc_engine_t* engine) {
    moe_runtime_optimization_plan_t plan{};
    if (!moe_pc_engine_get_optimization_plan(engine, moe_PHASE_BOTH, &plan)) {
        return;
    }
    std::cout << "Optimization: native_nvfp4=" << plan.use_native_nvfp4_tensor_cores
              << " native_mxfp4=" << plan.use_native_mxfp4_tensor_cores
              << " fused_fp4=" << plan.use_fused_fp4_dequant_matmul
              << " scale4_fusion=" << plan.use_scale4_decode_fusion
              << " async_copy=" << plan.use_async_copy_stream
              << " vram_hot=" << plan.use_vram_hot_cache
              << " common_vram=" << plan.recommended_vram_common_resident_bytes
              << " expert_vram=" << plan.recommended_vram_expert_cache_bytes
              << " pinned_ram=" << plan.use_pinned_ram_cache
              << " streams=" << plan.compute_streams << "/"
              << plan.copy_streams << "/" << plan.prefetch_streams << "\n";
}
