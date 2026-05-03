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
    cfg.disk_worker_threads = moe_default_disk_worker_threads(hw);
    cfg.pinned_worker_threads = hw >= 32 ? 4u : hw >= 16 ? 3u : 1u;
    cfg.gpu_worker_threads = hw >= 32 ? 4u : hw >= 16 ? 3u : 1u;
    cfg.pinned_staging_bytes = std::max<uint64_t>(512ull * 1024ull * 1024ull, normal_max * 16ull);
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
    cfg.vulkan_memory_headroom_pct = 10;
    cfg.level_zero_memory_headroom_pct = 10;
    return cfg;
}

