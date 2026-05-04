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
    slot_count = std::max<uint64_t>(4ull, std::min<uint64_t>(slot_count, 64ull));
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
    cfg.ram_budget_bytes = storagellm::default_ram_budget_bytes(60);
    cfg.db_budget_bytes = UINT64_MAX / 4;
    cfg.kv_mode = moe_KV_MODE_QKV;
    cfg.prefer_gpu = 1;
    cfg.allow_db_streaming = 1;
    cfg.preferred_backend = moe_BACKEND_AUTO;
    cfg.platform = moe_PLATFORM_AUTO;
    cfg.model_lib_cache_dir = nullptr;
    cfg.model_lib_path = nullptr;
    cfg.model_lib_compile_command = nullptr;
    cfg.enable_model_lib_jit = 0;
    return cfg;
}

moe_io_config_t moe_pc_default_io_config(void) {
    moe_io_config_t cfg{};
    const moe_storage_constants_t* storage = moe_storage_constants();
    const uint64_t normal_min = storage ? storage->normal_bundle_min_bytes : 8ull * 1024ull * 1024ull;
    const uint64_t normal_max = storage ? storage->normal_bundle_max_bytes : normal_min;
    const uint32_t hw = moe_cpu_thread_count_hint();
    cfg.io_worker_threads = moe_default_io_worker_threads(hw);
    cfg.prefetch_window_layers = 6;
    cfg.max_prefetch_queue = 12288;
    cfg.topology_max_candidates = 2;
    cfg.topology_max_edges_per_source = 32;
    cfg.pinned_worker_threads = hw >= 32 ? 4u : hw >= 16 ? 3u : 1u;
    cfg.gpu_worker_threads = hw >= 32 ? 4u : hw >= 16 ? 3u : 1u;
    cfg.disk_worker_threads = moe_balanced_disk_worker_threads(hw, cfg.io_worker_threads, cfg.gpu_worker_threads);
    cfg.pinned_staging_bytes = moe_default_pinned_staging_bytes(
        normal_min,
        normal_max,
        cfg.io_worker_threads,
        cfg.gpu_worker_threads);
    cfg.direct_upload_min_bytes = normal_min;
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
    cfg.int4_fallback_threshold = 0.50f;
    cfg.metal_memory_budget_pct = 0;
    cfg.madv_cold_threshold_pct = 30;
    cfg.vulkan_memory_headroom_pct = 10;
    cfg.level_zero_memory_headroom_pct = 10;
    return cfg;
}

