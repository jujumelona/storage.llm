static void demote_failed_device_expert(glm5_pc_engine* engine, const glm5_tensor_record& rec) {
    if (!engine) {
        return;
    }
    const size_t slot = expert_dense_slot((int)rec.info.layer, (int)rec.info.expert);
    if (slot == tensor_slot_invalid() || slot >= engine->expert_states.size()) {
        return;
    }
    const int layer = (int)rec.info.layer;
    const int expert = (int)rec.info.expert;
    // Fix 2: Release map_mutex before replace_evict_candidate (acquires heap_mutex).
    // Lock ordering: device_mutex -> heap_mutex -> map_mutex.
    // Holding map_mutex while acquiring heap_mutex is REVERSE ORDER = deadlock.
    glm5_block_state state_copy{};
    {
        std::lock_guard<std::mutex> lock(engine->map_mutex);
        glm5_block_state& state = engine->expert_states[slot];
        if (!state.active || state.tier != GLM5_TIER_VRAM) {
            return;
        }
        subtract_tier_usage(engine, GLM5_TIER_VRAM, state.bytes);
        engine->vram_expert_count.fetch_sub(1);
        state.tier = tier_used(engine, GLM5_TIER_RAM) + state.bytes <= tier_budget(engine, GLM5_TIER_RAM) ?
            GLM5_TIER_RAM : GLM5_TIER_DB;
        if (state.tier == GLM5_TIER_RAM) engine->ram_expert_count.fetch_add(1);
        else if (state.tier == GLM5_TIER_DB) engine->db_expert_count.fetch_add(1);
        add_tier_usage(engine, state.tier, state.bytes);
        state_copy = state;
    }
    replace_evict_candidate(engine, slot, state_copy);
    device_free_expert(engine, layer, expert);
}

static int prefetch_tensor_pinned_to_gpu(
    glm5_pc_engine* engine,
    const glm5_tensor_record& rec,
    uint32_t staging_slot = UINT32_MAX,
    uint64_t staging_bytes = 0,
    int sync_after = 1,
    int allow_sync_fallback = 1
) {
    if (!engine) {
        return 0;
    }
    const uint64_t bytes = tensor_stream_bytes(rec);
    int copied = 0;
    if (staging_slot < engine->staging_slots.size() &&
        staging_bytes >= bytes &&
        engine->staging_slots[staging_slot].buf) {
        copied = device_copy_tensor_from_host(
            engine,
            rec,
            GLM5_DEVICE_POOL_EXPERT,
            engine->staging_slots[staging_slot].buf,
            staging_bytes
        );
    } else {
        copied = device_copy_tensor(engine, rec, GLM5_DEVICE_POOL_EXPERT, allow_sync_fallback);
    }
    if (copied == GLM5_DEVICE_COPY_BACKPRESSURE) {
        return GLM5_DEVICE_COPY_BACKPRESSURE;
    }
    if (!copied) {
        demote_failed_device_expert(engine, rec);
        return 0;
    }
    if (sync_after) {
        device_stream_sync_current(engine);
    }
    engine->io_atomic.bytes_pinned_to_gpu.fetch_add(bytes);
    engine->io_atomic.gpu_stage_done.fetch_add(1);
    return 1;
}

static int prefetch_tensor_direct_to_gpu(glm5_pc_engine* engine, const glm5_tensor_record& rec) {
    if (!engine) {
        return 0;
    }
    const uint64_t bytes = tensor_stream_bytes(rec);
    if (!device_copy_tensor(engine, rec, GLM5_DEVICE_POOL_EXPERT)) {
        demote_failed_device_expert(engine, rec);
        return 0;
    }
    device_stream_sync_current(engine);
    engine->io_atomic.bytes_direct_to_gpu.fetch_add(bytes);
    engine->io_atomic.direct_stage_done.fetch_add(1);
    return 1;
}

static int prefetch_tensor_record(glm5_pc_engine* engine, const glm5_tensor_record& rec, uint32_t worker_index = 0) {
    if (!prefetch_tensor_disk_to_ram(engine, rec)) {
        return 0;
    }
    // Fix 39: Removed staging acquire+copy+release that performed no GPU transfer.
    return 1;
}

