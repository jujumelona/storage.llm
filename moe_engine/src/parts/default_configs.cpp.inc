


static int moe_env_truthy_local(const char* name) {
    return moe_perf_env_truthy(name);
}

static int moe_strict_fast_path_enabled() {
    return moe_perf_strict_fast_path_enabled();
}

static int moe_cpu_storage_max_enabled() {
    return moe_perf_cpu_storage_mode_enabled();
}

static int moe_storage_always_enabled() {
    // StorageLLM format/runtime target: weights remain storage-resident and RAM
    // is a hotset/page-cache/scratch tier.  This is backend-common; unlike
    // CPU_STORAGE_MAX it must not force GPU environments onto CPU.
    return moe_perf_storage_required_enabled();
}

static uint32_t moe_default_io_worker_threads(uint32_t hw) {
    if (hw >= 32u) return 8u;
    if (hw >= 16u) return 6u;
    if (hw >= 8u) return 4u;
    return 2u;
}

static uint32_t moe_default_disk_worker_threads(uint32_t hw) {
    if (hw >= 32u) return 16u;
    if (hw >= 16u) return 8u;
    if (hw >= 8u) return 4u;
    return hw >= 4u ? 2u : 1u;
}

static uint32_t moe_balanced_disk_worker_threads(uint32_t hw, uint32_t io_workers, uint32_t gpu_workers) {
    const uint32_t default_disk = moe_default_disk_worker_threads(hw);
    const uint32_t balanced = std::max(1u, io_workers + std::max(1u, gpu_workers));
    return std::min(default_disk, balanced);
}

static uint64_t moe_default_pinned_staging_bytes(
    uint64_t normal_min,
    uint64_t normal_max,
    uint32_t io_workers,
    uint32_t gpu_workers
) {
    const uint32_t gpu = std::max(1u, gpu_workers);
    uint64_t slot_count = (uint64_t)std::max(1u, io_workers) * (uint64_t)gpu * 2ull;
    // Max-speed default: keep enough pinned slots to overlap SSD read, host copy,
    // and device upload without immediately starving the queue on large MoE bundles.
    // Still bounded so machines without GDS/DirectStorage do not pin unbounded RAM.
    slot_count = std::max<uint64_t>(8ull, std::min<uint64_t>(slot_count, 32ull));
    uint64_t desired = normal_max > 0 && slot_count <= UINT64_MAX / normal_max
        ? normal_max * slot_count : UINT64_MAX;
    const uint64_t ram_cap = storagellm::default_ram_budget_bytes(25);
    if (ram_cap > 0) {
        desired = std::min<uint64_t>(desired, ram_cap);
    }
    const uint64_t normal_floor = normal_min <= UINT64_MAX / 4ull ? normal_min * 4ull : UINT64_MAX;
    const uint64_t floor = std::max<uint64_t>(128ull * 1024ull * 1024ull, normal_floor);
    const uint64_t cap = ram_cap ? ram_cap : UINT64_MAX;
    return std::min<uint64_t>(std::max<uint64_t>(desired, std::min<uint64_t>(floor, cap)), cap);
}

moe_pc_engine_config_t moe_pc_default_config(void) {
    moe_pc_engine_config_t cfg{};
    cfg.vram_budget_bytes = 0;
    cfg.common_vram_reserve_bytes = 0;
    cfg.ram_budget_bytes = storagellm::default_ram_budget_bytes(
        moe_cpu_storage_max_enabled() ? 80u : 60u);
    cfg.db_budget_bytes = UINT64_MAX / 4;
    cfg.kv_mode = moe_KV_MODE_QKV;
    cfg.prefer_gpu = moe_cpu_storage_max_enabled() ? 0 : 1;
    cfg.allow_db_streaming = 1;
    cfg.preferred_backend = moe_cpu_storage_max_enabled() ? moe_BACKEND_CPU : moe_BACKEND_AUTO;
    cfg.platform = moe_cpu_storage_max_enabled() ? moe_PLATFORM_CPU_ONLY : moe_PLATFORM_AUTO;
    cfg.model_lib_cache_dir = nullptr;
    cfg.model_lib_path = nullptr;
    cfg.model_lib_compile_command = nullptr;
    cfg.enable_model_lib_jit = moe_cpu_storage_max_enabled() ? 0 : 0;
    cfg.vram_kv_ratio = 0.0f;  // auto
    cfg.device_id = -1;  // auto
    cfg.max_context_len = 0;  // use default (8192)
    cfg.kv_offload_mode = moe_KV_OFFLOAD_DISABLED;
    cfg.kv_offload_enable_for_long_context_only = 1;
    cfg.kv_offload_min_context_tokens = 32768;
    return cfg;
}

moe_io_config_t moe_pc_default_io_config(void) {
    moe_io_config_t cfg{};
    const moe_storage_constants_t* storage = moe_storage_constants();
    const uint64_t normal_min = storage ? storage->normal_bundle_min_bytes : 8ull * 1024ull * 1024ull;
    const uint64_t normal_max = storage ? storage->normal_bundle_max_bytes : normal_min;
    const uint32_t hw = moe_cpu_thread_count_hint();
    cfg.io_worker_threads = moe_default_io_worker_threads(hw);
    // Max-speed storage/GPU defaults. configure_io() may still adapt these to
    // measured bandwidth and backend caps, but the baseline should keep the
    // disk->pinned->GPU pipeline full instead of starting conservative.
    cfg.prefetch_window_layers = hw >= 16u ? 6u : (hw >= 8u ? 4u : 2u);
    cfg.max_prefetch_queue = hw >= 16u ? 32768u : (hw >= 8u ? 16384u : 8192u);
    cfg.topology_max_candidates = 4;
    cfg.topology_max_edges_per_source = 64;
    cfg.pinned_worker_threads = hw >= 32 ? 8u : hw >= 16 ? 6u : hw >= 8 ? 4u : 1u;
    cfg.gpu_worker_threads = hw >= 32 ? 8u : hw >= 16 ? 6u : hw >= 8 ? 4u : 1u;
    cfg.gpu_expert_scratch_slots = std::max<uint32_t>(32u, cfg.gpu_worker_threads * 4u);
    cfg.disk_worker_threads = moe_balanced_disk_worker_threads(hw, cfg.io_worker_threads, cfg.gpu_worker_threads);
    cfg.pinned_staging_bytes = moe_default_pinned_staging_bytes(
        normal_min,
        normal_max,
        cfg.io_worker_threads,
        cfg.gpu_worker_threads);
    cfg.direct_upload_min_bytes = normal_min;
    cfg.vram_safety_margin_bytes = 128ull * 1024ull * 1024ull;
    cfg.directstorage_staging_queue_capacity = 4096;
    cfg.directstorage_gpu_queue_capacity = 8192;
    cfg.directstorage_require_completion_event = 1;
    cfg.prefill_group_wait_us = 100;
    cfg.internal_require_gpu_expert_batch_kernel = 1;
    cfg.internal_verified_backend_only = 1;
    cfg.enable_compute_io_overlap = 1;
    cfg.enable_multistage_pipeline = 1;
    cfg.enable_lookahead_prefetch = 1;
    cfg.enable_topology_prefetch = 1;
    cfg.enable_mmap_prefetch = 1;
    cfg.enable_pinned_host = 1;
    cfg.enable_directstorage = 1;
    cfg.enable_gpudirect_storage = 1;
    cfg.enable_bam_nvme = 1;
    cfg.enable_common_raw_vram_pin = 1;
    cfg.allow_required_sync_fallback = 0;
    cfg.require_bundle_native_experts = 1;
    cfg.enable_cold_expert_cpu_fallback = 0;
    cfg.cpu_fallback_max_tokens = 1;
    cfg.expert_group_gpu_min_assignments = 2;
    cfg.int4_fallback_threshold = 0.50f;
    cfg.metal_memory_budget_pct = 0;
    cfg.madv_cold_threshold_pct = 30;
    cfg.vulkan_memory_headroom_pct = 10;
    // BUGFIX 902: Level Zero iGPU needs higher headroom (20% default)
    cfg.level_zero_memory_headroom_pct = 20;
    cfg.enable_bw_benchmark = 1;
    cfg.vram_eviction_threshold_pct = 88;
    cfg.attention_prefetch_threshold_pct = 92;
    cfg.ram_eviction_target_fill_pct = 85;  // Max-speed default keeps a larger RAM hotset
    if (moe_storage_always_enabled()) {
        // Backend-common SSD/RAM optimization.  CPU consumes this through
        // mmap/page-cache; GPU/offload consumes the same merged span plan through
        // host staging and device cache.  Do not disable GPU here.
        cfg.enable_compute_io_overlap = 1;
        cfg.enable_multistage_pipeline = 1;
        cfg.enable_lookahead_prefetch = 1;
        cfg.enable_topology_prefetch = 1;
        cfg.enable_mmap_prefetch = 1;
        cfg.prefetch_window_layers = hw >= 32u ? 8u : (hw >= 16u ? 6u : (hw >= 8u ? 4u : 2u));
        cfg.max_prefetch_queue = hw >= 32u ? 65536u : (hw >= 16u ? 32768u : (hw >= 8u ? 16384u : 8192u));
        cfg.disk_worker_threads = std::max<uint32_t>(cfg.disk_worker_threads, hw >= 32u ? 16u : (hw >= 16u ? 12u : (hw >= 8u ? 6u : 2u)));
        cfg.io_worker_threads = std::max<uint32_t>(cfg.io_worker_threads, hw >= 32u ? 16u : (hw >= 16u ? 10u : (hw >= 8u ? 6u : 2u)));
        cfg.pinned_worker_threads = std::max<uint32_t>(cfg.pinned_worker_threads, hw >= 32u ? 10u : (hw >= 16u ? 8u : (hw >= 8u ? 4u : 1u)));
        cfg.gpu_worker_threads = std::max<uint32_t>(cfg.gpu_worker_threads, hw >= 32u ? 10u : (hw >= 16u ? 8u : (hw >= 8u ? 4u : 1u)));
        cfg.ram_eviction_target_fill_pct = 94;
        cfg.ram_eviction_threshold_pct = 98;
    }
    if (moe_cpu_storage_max_enabled()) {
        const uint32_t hw_cpu = std::max<uint32_t>(1u, hw);
        // CPU-only needs overlap, but compute must not starve page-cache/NVMe
        // workers.  Reserve 1-2 cores for IO on larger machines and keep GPU
        // staging completely disabled.
        const uint32_t cpu_io_workers =
            hw_cpu >= 32u ? 16u :
            hw_cpu >= 16u ? 10u :
            hw_cpu >= 8u  ? 6u : 2u;
        cfg.io_worker_threads = cpu_io_workers;
        cfg.disk_worker_threads = cpu_io_workers;
        cfg.pinned_worker_threads = 0;
        cfg.gpu_worker_threads = 0;
        cfg.gpu_expert_scratch_slots = 0;
        cfg.prefetch_window_layers = hw_cpu >= 32u ? 8u : (hw_cpu >= 16u ? 6u : 4u);
        cfg.max_prefetch_queue = hw_cpu >= 32u ? 65536u : (hw_cpu >= 16u ? 32768u : 16384u);
        cfg.pinned_staging_bytes = 0;
        cfg.directstorage_staging_queue_capacity = 0;
        cfg.directstorage_gpu_queue_capacity = 0;
        cfg.internal_require_gpu_expert_batch_kernel = 0;
        cfg.internal_verified_backend_only = 0;
        cfg.enable_compute_io_overlap = 1;
        // Keep CPU pipeline enabled: this now means layer-hot page-cache hints
        // plus async IO workers, not GPU copy staging.
        cfg.enable_multistage_pipeline = 1;
        cfg.enable_lookahead_prefetch = 1;
        cfg.enable_topology_prefetch = 1;
        cfg.enable_mmap_prefetch = 1;
        cfg.enable_pinned_host = 0;
        cfg.enable_directstorage = 0;
        cfg.enable_gpudirect_storage = 0;
        cfg.enable_bam_nvme = 0;
        cfg.enable_common_raw_vram_pin = 0;
        cfg.allow_required_sync_fallback = 1;
        cfg.enable_cold_expert_cpu_fallback = 1;
        cfg.ram_eviction_target_fill_pct = 94;
        cfg.ram_eviction_threshold_pct = 98;
    }
    moe_apply_autotune_io_env(&cfg);
    if (moe_strict_fast_path_enabled()) {
        // Verification mode is profile-aware.  CPU-only strict mode must not
        // demand a GPU batch kernel; GPU/offload strict mode should fail fast
        // when the model library/backend kernel is missing.
        cfg.allow_required_sync_fallback = 0;
        cfg.require_bundle_native_experts = 1;
        if (moe_cpu_storage_max_enabled()) {
            cfg.enable_cold_expert_cpu_fallback = 1;   // CPU is the target path, not a fallback here.
            cfg.internal_verified_backend_only = 0;
            cfg.internal_require_gpu_expert_batch_kernel = 0;
        } else {
            cfg.enable_cold_expert_cpu_fallback = 0;
            cfg.internal_verified_backend_only = 1;
            cfg.internal_require_gpu_expert_batch_kernel = 1;
        }
    }
    return cfg;
}
