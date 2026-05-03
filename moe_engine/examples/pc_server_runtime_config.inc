static uint64_t server_auto_ram_budget_bytes() {
    const uint64_t available = storagellm::available_ram_bytes();
    const uint64_t gib = 1024ull * 1024ull * 1024ull;
    const uint64_t reserve =
        available >= 64ull * gib ? 3ull * gib :
        available >= 32ull * gib ? 2ull * gib : 1ull * gib;
    if (available > reserve * 2ull) {
        return available - reserve;
    }
    return (available * 9ull) / 10ull;
}

static void apply_detected_budgets(server_options* opts, const moe_backend_caps_t& caps) {
    if (!opts) {
        return;
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
        opts->engine_config.ram_budget_bytes = server_auto_ram_budget_bytes();
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
    io.enable_multistage_pipeline = 1;
    io.enable_lookahead_prefetch = 1;
    io.enable_topology_prefetch = 1;
    io.enable_mmap_prefetch = 1;
    io.enable_pinned_host = caps.supports_pinned_host;
    io.enable_directstorage = caps.supports_directstorage;
    io.enable_gpudirect_storage = caps.supports_gpudirect_storage;
    io.enable_bam_nvme = caps.supports_gpu_direct_file_io;
    // Keep common raw pinning opt-in. Server auto startup must stay portable
    // across CPU and GPU backends instead of assuming one device allocator path.
    io.enable_common_raw_vram_pin = 0;
    io.prefetch_window_layers = std::max<uint32_t>(2u, io.prefetch_window_layers);
    io.disk_worker_threads = 4;
    io.pinned_worker_threads = caps.copy_streams > 1 ? 4u : 2u;
    io.gpu_worker_threads = caps.copy_streams ? caps.copy_streams : 1u;
    io.direct_upload_min_bytes = expert_min_bytes;
    const uint32_t active_workers =
        io.disk_worker_threads + io.pinned_worker_threads + io.gpu_worker_threads;
    const uint64_t slot_bytes = expert_max_bytes + 4ull * 1024ull * 1024ull;
    const uint64_t worker_slots = std::max<uint32_t>(8u, active_workers * 2u);
    const uint64_t worker_staging = slot_bytes * worker_slots;
    io.pinned_staging_bytes = std::max<uint64_t>(
        std::max<uint64_t>(caps.recommended_pinned_bytes, worker_staging),
        512ull * 1024ull * 1024ull);
    return io;
}

static void print_optimization_plan(moe_pc_engine_t* engine) {
    moe_runtime_optimization_plan_t plan{};
    if (!moe_pc_engine_get_optimization_plan(engine, moe_PHASE_BOTH, &plan)) {
        return;
    }
    std::cout << "Optimization: fused_fp4=" << plan.use_fused_fp4_dequant_matmul
              << " scale4_fusion=" << plan.use_scale4_decode_fusion
              << " async_copy=" << plan.use_async_copy_stream
              << " vram_hot=" << plan.use_vram_hot_cache
              << " common_vram=" << plan.recommended_vram_common_resident_bytes
              << " expert_vram=" << plan.recommended_vram_expert_cache_bytes
              << " pinned_ram=" << plan.use_pinned_ram_cache
              << " streams=" << plan.compute_streams << "/"
              << plan.copy_streams << "/" << plan.prefetch_streams << "\n";
}
