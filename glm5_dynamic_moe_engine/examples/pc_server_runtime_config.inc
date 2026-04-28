static void apply_detected_budgets(server_options* opts, const glm5_backend_caps_t& caps) {
    if (!opts) {
        return;
    }
    if (!opts->vram_budget_override) {
        opts->engine_config.vram_budget_bytes = caps.recommended_vram_cache_bytes;
    }
    if (!opts->ram_budget_override) {
        opts->engine_config.ram_budget_bytes = storagellm::default_ram_budget_bytes(60);
    }
    opts->engine_config.allow_db_streaming = 1;
}

static glm5_io_config_t make_server_io_config(const server_options& opts, const glm5_backend_caps_t& caps) {
    glm5_io_config_t io = glm5_pc_default_io_config();
    const glm5_storage_constants_t* storage = glm5_storage_constants();
    const uint64_t expert_bytes = storage ? storage->normal_bundle_min_bytes : 8ull * 1024ull * 1024ull;
    io.enable_compute_io_overlap = 1;
    io.enable_multistage_pipeline = 1;
    io.enable_lookahead_prefetch = 1;
    io.enable_topology_prefetch = 1;
    io.enable_mmap_prefetch = 1;
    io.enable_pinned_host = caps.supports_pinned_host;
    io.enable_directstorage = caps.supports_directstorage;
    io.enable_gpudirect_storage = caps.supports_gpudirect_storage;
    io.enable_bam_nvme = caps.supports_gpu_direct_file_io;
    io.prefetch_window_layers = std::max<uint32_t>(2u, io.prefetch_window_layers);
    io.disk_worker_threads = 4;
    io.pinned_worker_threads = caps.copy_streams > 1 ? 4u : 2u;
    io.gpu_worker_threads = caps.copy_streams ? caps.copy_streams : 1u;
    io.direct_upload_min_bytes = expert_bytes;
    io.pinned_staging_bytes = std::max<uint64_t>(caps.recommended_pinned_bytes, expert_bytes * 3ull);
    if (opts.engine_config.ram_budget_bytes > 0) {
        io.pinned_staging_bytes = std::min<uint64_t>(io.pinned_staging_bytes, opts.engine_config.ram_budget_bytes / 4ull);
    }
    return io;
}

static void print_optimization_plan(glm5_pc_engine_t* engine) {
    glm5_runtime_optimization_plan_t plan{};
    if (!glm5_pc_engine_get_optimization_plan(engine, GLM5_PHASE_BOTH, &plan)) {
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
