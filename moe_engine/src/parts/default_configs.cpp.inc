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
    // BUGFIX 993: Cap default staging slots to prevent excessive pinned memory ★★ IMPORTANT
    // Problem: 32-core machine → io=8, gpu=4 → slot_count=64 → 64 × 32MB = 2GB pinned
    // This wastes RAM, especially with GDS (staging bypassed) or CPU-only (RAM > throughput).
    // Solution: Cap at 16 slots (512MB). moe_ensure_staging_capacity grows if genuinely needed.
    // 16 slots provides sufficient pipeline depth for 4 GPU workers × 4 batch.
    slot_count = std::max<uint64_t>(4ull, std::min<uint64_t>(slot_count, 16ull));
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
    // BUGFIX 965: Set default to 0 (auto-calculate) instead of 6 ★★
    // Problem: Value 6 was immediately overwritten by io_config_api.cpp.inc VRAM-based
    // logic. But if user calls moe_pc_default_io_config() without configure_io(),
    // the stale 6 leaks through. 0 = "auto" is consistent with the documented
    // (using_default_io_config || == 0) auto-calculation condition.
    // Safety: common_raw_prefetch.cpp.inc::runtime_prefetch_window() already
    // falls back to 1 when both runtime and config values are 0.
    cfg.prefetch_window_layers = 0;
    cfg.max_prefetch_queue = 12288;
    cfg.topology_max_candidates = 2;
    cfg.topology_max_edges_per_source = 32;
    cfg.pinned_worker_threads = hw >= 32 ? 4u : hw >= 16 ? 3u : 1u;
    cfg.gpu_worker_threads = hw >= 32 ? 4u : hw >= 16 ? 3u : 1u;
    cfg.gpu_expert_scratch_slots = std::max<uint32_t>(16u, cfg.gpu_worker_threads * 2u);
    cfg.disk_worker_threads = moe_balanced_disk_worker_threads(hw, cfg.io_worker_threads, cfg.gpu_worker_threads);
    cfg.pinned_staging_bytes = moe_default_pinned_staging_bytes(
        normal_min,
        normal_max,
        cfg.io_worker_threads,
        cfg.gpu_worker_threads);
    cfg.direct_upload_min_bytes = normal_min;
    cfg.vram_safety_margin_bytes = 256ull * 1024ull * 1024ull;
    cfg.directstorage_staging_queue_capacity = 1024;
    cfg.directstorage_gpu_queue_capacity = 1024;
    cfg.directstorage_require_completion_event = 1;
    cfg.prefill_group_wait_us = 200;
    cfg.require_gpu_expert_batch_kernel_for_fast_mode = 1;
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
    cfg.ram_eviction_target_fill_pct = 75;  // Default 75% RAM fill target
    return cfg;
}
