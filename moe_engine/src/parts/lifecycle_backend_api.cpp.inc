moe_pc_engine_t* moe_pc_engine_create(const moe_pc_engine_config_t* config) {
    auto* engine = new moe_pc_engine();
    engine->config = config ? *config : moe_pc_default_config();
    qkv_set_mode(engine->config.kv_mode == moe_KV_MODE_QKV);
    engine->expert_states.assign(expert_slot_count(), moe_block_state{});
    tensor_slots_reset(engine);
    engine->tensors.reserve(tensor_record_capacity());
    engine->shard_mmaps.reserve(105);
    engine->topology_edges.reserve(19456);
    engine->prefetch_plan_seen_gen.reserve(24576);
    fill_backend_caps(engine->config.preferred_backend, engine->config.platform, &engine->backend_caps);
    model_lib_prepare(engine);
    if (engine->config.prefer_gpu && engine->config.vram_budget_bytes == 0 &&
        caps_gpu_backend_available(engine->backend_caps)) {
        engine->config.vram_budget_bytes = engine->backend_caps.recommended_vram_cache_bytes;
    }
    apply_common_vram_reservation(engine);
    moe_pc_engine_configure_io(engine, nullptr);
    ensure_cpu_row_workers(engine, 0);
    return engine;
}

void moe_pc_engine_destroy(moe_pc_engine_t* engine) {
    if (!engine) {
        return;
    }
    if (!engine->topology_path.empty()) {
        moe_pc_engine_save_topology(engine, nullptr);
    }
    stop_cpu_row_workers(engine);
    stop_io_workers(engine);
    for (size_t i = 0; i < engine->qkv_head_state_slots.size(); ++i) {
        if (i < engine->qkv_head_state_valid.size() && engine->qkv_head_state_valid[i]) {
            qkv_free(&engine->qkv_head_state_slots[i]);
        }
    }
    engine->qkv_head_state_slots.clear();
    engine->qkv_head_state_valid.clear();
    engine->qkv_head_state_locks.clear();
    device_stream_pool_destroy(engine);
    device_allocator_shutdown(engine);
    model_lib_unload(engine);
    {
        std::unique_lock<std::shared_mutex> lock(engine->shard_mutex);
        for (auto& item : engine->shard_mmaps) {
            unload_mapped_shard(item.second);
        }
        engine->shard_mmaps.clear();
        engine->shard_mmaps_by_id.clear();
        engine->shard_mmap_cvs.clear();
        engine->locked_mmaps.clear();
    }
    reset_source_mmap_slots(engine);
    {
        std::lock_guard<std::mutex> lock(engine->raw_file_size_cache_mutex);
        engine->raw_file_size_cache.clear();
    }
    {
        std::unique_lock<std::shared_mutex> lock(engine->raw_tensor_cache_mutex);
        engine->raw_tensor_cache.clear();
    }
    if (engine->scale4) {
        moe_scale4_close(engine->scale4);
        engine->scale4 = nullptr;
    }
    // Bug 13 Fix: Call reset_staging_slots BEFORE free_staging to prevent use-after-free.
    // Workers may still reference staging_slots[i].buf in 5ms race window.
    // reset_staging_slots clears in_use flags and waits for workers to exit.
    reset_staging_slots(engine);
    free_staging(engine->staging, engine->staging_size, engine->staging_pinned);
    engine->staging = nullptr;
    engine->staging_size = 0;
    engine->staging_pinned = false;
    delete engine;
}

int moe_pc_detect_backend_caps(
    moe_backend_t preferred_backend,
    moe_platform_t platform,
    moe_backend_caps_t* out_caps
) {
    if (!out_caps) {
        return 0;
    }
    fill_backend_caps(preferred_backend, platform, out_caps);
    return 1;
}

int moe_pc_engine_get_backend_caps(const moe_pc_engine_t* engine, moe_backend_caps_t* out_caps) {
    if (!engine || !out_caps) {
        return 0;
    }
    *out_caps = engine->backend_caps;
    const uint64_t recommended_staging = moe_recommended_staging_bytes(engine);
    if (recommended_staging > out_caps->recommended_pinned_bytes) {
        out_caps->recommended_pinned_bytes = recommended_staging;
    }
    return 1;
}
