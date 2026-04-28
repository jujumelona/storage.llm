int glm5_pc_engine_get_stats(const glm5_pc_engine_t* engine, glm5_pc_engine_stats_t* out_stats) {
    if (!engine || !out_stats) {
        return 0;
    }
    std::memset(out_stats, 0, sizeof(*out_stats));
    out_stats->vram_used_bytes = tier_used(engine, GLM5_TIER_VRAM);
    out_stats->ram_used_bytes = tier_used(engine, GLM5_TIER_RAM);
    out_stats->db_used_bytes = tier_used(engine, GLM5_TIER_DB);
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
    out_stats->common_raw_tensor_count = glm5_common_raw_tensor_count();
    out_stats->vram_expert_count = engine->vram_expert_count.load();
    out_stats->ram_expert_count = engine->ram_expert_count.load();
    out_stats->db_expert_count = engine->db_expert_count.load();
    out_stats->hot_expert_count = out_stats->vram_expert_count + out_stats->ram_expert_count + out_stats->db_expert_count;
    out_stats->kv_mode = engine->config.kv_mode;
    return 1;
}
