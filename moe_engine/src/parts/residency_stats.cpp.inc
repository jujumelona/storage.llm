int moe_pc_engine_get_stats(const moe_pc_engine_t* engine, moe_pc_engine_stats_t* out_stats) {
    if (!engine || !out_stats) {
        return 0;
    }
    std::memset(out_stats, 0, sizeof(*out_stats));
    out_stats->vram_used_bytes = tier_used(engine, moe_TIER_VRAM);
    out_stats->ram_used_bytes = tier_used(engine, moe_TIER_RAM);
    out_stats->db_used_bytes = tier_used(engine, moe_TIER_DB);
    out_stats->common_vram_reserved_bytes = engine->common_vram_reserved_bytes.load();
    out_stats->vram_expert_used_bytes =
        out_stats->vram_used_bytes > out_stats->common_vram_reserved_bytes ?
        out_stats->vram_used_bytes - out_stats->common_vram_reserved_bytes : 0;
    out_stats->common_raw_prefetched_bytes = engine->common_raw_prefetched_bytes.load();
    out_stats->common_raw_prefetched_spans = engine->common_raw_prefetched_spans.load();
    out_stats->device_allocated_bytes = engine->device_allocated_bytes.load();
    out_stats->device_allocation_count = engine->device_allocation_count.load();
    out_stats->device_fixed_bytes = engine->device_fixed_bytes.load();
    out_stats->device_activation_bytes = engine->device_activation_bytes.load();
    out_stats->device_expert_bytes = engine->device_expert_bytes.load();
    out_stats->device_stream_bytes = engine->device_stream_bytes.load();
    out_stats->device_total_bytes = engine->device_total_bytes.load();
    out_stats->device_free_bytes = engine->device_free_bytes.load();
    out_stats->model_lib_loaded = engine->model_lib_loaded;
    out_stats->model_lib_generated = engine->model_lib_generated;
    out_stats->model_lib_compile_attempted = engine->model_lib_compile_attempted;
    out_stats->model_lib_compile_succeeded = engine->model_lib_compile_succeeded;
    out_stats->model_lib_abi_version = engine->model_lib_abi_version;
    if (!engine->model_lib_path.empty()) {
        std::strncpy(out_stats->model_lib_path, engine->model_lib_path.c_str(), sizeof(out_stats->model_lib_path) - 1);
    }
    out_stats->common_raw_tensor_count = moe_common_raw_tensor_count();
    out_stats->vram_expert_count = engine->vram_expert_count.load();
    out_stats->ram_expert_count = engine->ram_expert_count.load();
    out_stats->db_expert_count = engine->db_expert_count.load();
    out_stats->hot_expert_count = out_stats->vram_expert_count + out_stats->ram_expert_count + out_stats->db_expert_count;
    out_stats->generation_started = engine->generation_started.load(std::memory_order_relaxed);
    out_stats->generation_completed = engine->generation_completed.load(std::memory_order_relaxed);
    out_stats->generation_failed = engine->generation_failed.load(std::memory_order_relaxed);
    out_stats->generation_active = engine->generation_active.load(std::memory_order_relaxed);
    out_stats->generation_token = engine->generation_token.load(std::memory_order_relaxed);
    out_stats->generation_layer = engine->generation_layer.load(std::memory_order_relaxed);
    out_stats->generation_phase = engine->generation_phase.load(std::memory_order_relaxed);
    out_stats->kv_mode = engine->config.kv_mode;
    out_stats->offload_gguf_valid = engine->offload_gguf_valid.load(std::memory_order_relaxed);
    out_stats->offload_gguf_file_count = engine->offload_gguf_file_count.load(std::memory_order_relaxed);
    out_stats->qkv_forced_by_format = engine->qkv_forced_by_format.load(std::memory_order_relaxed);
    if (engine->offload_qkv_cfg_available.load(std::memory_order_acquire)) {
        out_stats->qkv_k_bits = (uint32_t)std::max(0, engine->offload_qkv_cfg.k_bits);
        out_stats->qkv_v_bits = (uint32_t)std::max(0, engine->offload_qkv_cfg.v_bits);
        out_stats->qkv_group_size = engine->offload_qkv_cfg.group_size;
        out_stats->qkv_page_size_tokens = engine->offload_qkv_cfg.page_size_tokens;
        out_stats->qkv_sink_tokens = engine->offload_qkv_cfg.sink_tokens;
    }
    out_stats->weight_quant_bits = engine->offload_weight_quant_bits.load(std::memory_order_relaxed);
    out_stats->weight_quant_encoding = engine->offload_weight_encoding.load(std::memory_order_relaxed);
    out_stats->weight_quant_block_size = engine->offload_weight_block_size.load(std::memory_order_relaxed);
    out_stats->optimization_catalog_count =
        engine->offload_optimization_catalog_count.load(std::memory_order_relaxed);
    out_stats->juju_strategy_valid = engine->juju_strategy_valid.load(std::memory_order_relaxed);
    out_stats->juju_execution_path = (uint32_t)engine->juju_execution_path;
    out_stats->juju_prefetch_depth = engine->juju_pipeline_config.prefetch_depth;
    out_stats->juju_io_threads = engine->juju_pipeline_config.io_threads;
    out_stats->juju_segment_count = engine->juju_pipeline_config.segment_count;
    out_stats->juju_block_size = engine->juju_pipeline_config.block_size;
    out_stats->juju_segment_bytes = engine->juju_pipeline_config.segment_bytes;
    out_stats->juju_tier0_expert_count = engine->juju_pipeline_config.tier0_expert_count;
    out_stats->juju_tier1_expert_count = engine->juju_pipeline_config.tier1_expert_count;
    out_stats->juju_compute_ms = engine->juju_pipeline_config.compute_ms;
    out_stats->juju_tier1_to_tier0_ms = engine->juju_pipeline_config.tier1_to_tier0_ms;
    out_stats->juju_tier2_to_tier1_ms = engine->juju_pipeline_config.tier2_to_tier1_ms;
    out_stats->offload_gguf_tensor_count =
        engine->offload_gguf_tensor_count.load(std::memory_order_relaxed);
    out_stats->offload_gguf_executable_tensor_count =
        engine->offload_gguf_executable_tensor_count.load(std::memory_order_relaxed);
    if (!engine->offload_gguf_first_file.empty()) {
        std::strncpy(
            out_stats->offload_gguf_first_file,
            engine->offload_gguf_first_file.c_str(),
            sizeof(out_stats->offload_gguf_first_file) - 1
        );
    }
    if (!engine->offload_weight_quant_family.empty()) {
        std::strncpy(
            out_stats->weight_quant_family,
            engine->offload_weight_quant_family.c_str(),
            sizeof(out_stats->weight_quant_family) - 1
        );
    }
    if (!engine->offload_weight_kernel_family.empty()) {
        std::strncpy(
            out_stats->weight_kernel_family,
            engine->offload_weight_kernel_family.c_str(),
            sizeof(out_stats->weight_kernel_family) - 1
        );
    }
    return 1;
}
