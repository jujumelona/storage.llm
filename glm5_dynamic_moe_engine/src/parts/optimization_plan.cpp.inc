int glm5_pc_engine_get_optimization_plan(
    const glm5_pc_engine_t* engine,
    glm5_inference_phase_t phase,
    glm5_runtime_optimization_plan_t* out_plan
) {
    if (!engine || !out_plan) {
        return 0;
    }

    std::memset(out_plan, 0, sizeof(*out_plan));
    const glm5_backend_caps_t& caps = engine->backend_caps;
    const glm5_io_config_t& io = engine->io_config;
    const int gpu_available = caps_gpu_backend_available(caps);
    const glm5_io_path_t active_path = choose_active_io_path(engine);
    const int direct_to_gpu =
        active_path == GLM5_IO_PATH_DIRECTSTORAGE ||
        active_path == GLM5_IO_PATH_GPUDIRECT_STORAGE ||
        active_path == GLM5_IO_PATH_BAM_NVME;

    out_plan->backend = caps.backend;
    out_plan->platform = caps.platform;
    out_plan->phase = phase;
    out_plan->compute_streams = caps.compute_streams ? caps.compute_streams : 1;
    out_plan->copy_streams = caps.copy_streams;
    out_plan->prefetch_streams = caps.prefetch_streams ? caps.prefetch_streams : 1;
    out_plan->recommended_vram_total_cache_bytes =
        engine->config.vram_budget_bytes ? engine->config.vram_budget_bytes : caps.recommended_vram_cache_bytes;
    out_plan->recommended_vram_common_resident_bytes = engine->common_vram_reserved_bytes.load();
    out_plan->recommended_vram_expert_cache_bytes = vram_expert_budget_bytes(engine);
    const uint64_t recommended_staging = glm5_recommended_staging_bytes(engine);
    const uint64_t current_or_backend_staging =
        engine->staging_size ? engine->staging_size : caps.recommended_pinned_bytes;
    out_plan->recommended_pinned_staging_bytes = engine_bypass_staging_pipeline(engine) ?
        0 : std::max(current_or_backend_staging, recommended_staging);
    out_plan->use_model_shape_constants = 1;
    out_plan->use_static_memory_pool = 1;
    out_plan->use_expert_prefetch = io.enable_lookahead_prefetch || io.enable_topology_prefetch;
    out_plan->use_vram_hot_cache = gpu_available && engine->config.vram_budget_bytes > 0;
    out_plan->use_pinned_ram_cache =
        caps.supports_pinned_host && engine->staging_pinned && !engine_bypass_staging_pipeline(engine);
    out_plan->use_storage_streaming = engine->config.allow_db_streaming;
    out_plan->use_async_copy_stream =
        caps.supports_async_copy && caps.supports_backend_async_api && io.enable_compute_io_overlap;
    out_plan->use_direct_to_gpu_io = direct_to_gpu;
    out_plan->use_fused_fp4_dequant_matmul = gpu_available && caps.supports_fp4;
    out_plan->use_scale4_decode_fusion = gpu_available && caps.supports_scale4;
    out_plan->use_quant_family_dispatch = 1;
    out_plan->use_reference_weight_dequant =
        caps.supports_weight_q2 || caps.supports_weight_q3 ||
        caps.supports_weight_q4 || caps.supports_weight_q8;
    out_plan->preferred_weight_bits = engine->offload_weight_quant_bits.load(std::memory_order_relaxed);
    out_plan->preferred_weight_encoding = engine->offload_weight_encoding.load(std::memory_order_relaxed);
    out_plan->use_cuda_graphs = caps.backend == GLM5_BACKEND_CUDA && caps.supports_cuda;
    out_plan->use_metal_command_buffers = caps.backend == GLM5_BACKEND_METAL && caps.supports_metal;
    out_plan->use_paged_kv_cache = gpu_available;
    out_plan->use_prefill_decode_split = 1;
    return 1;
}

