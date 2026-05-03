static void demote_failed_device_expert(moe_pc_engine* engine, const moe_tensor_record& rec) {
    if (!engine) {
        return;
    }
    const size_t slot = expert_dense_slot(engine, (int)rec.info.layer, (int)rec.info.expert);
    if (slot == tensor_slot_invalid() || slot >= engine->expert_states.size()) {
        return;
    }
    const int layer = (int)rec.info.layer;
    const int expert = (int)rec.info.expert;
    // Fix 2: Release map_mutex before replace_evict_candidate (acquires heap_mutex).
    // Lock ordering: device_mutex -> heap_mutex -> map_mutex.
    // Holding map_mutex while acquiring heap_mutex is REVERSE ORDER = deadlock.
    moe_block_state state_copy{};
    {
        std::lock_guard<std::mutex> lock(engine->map_mutex);
        moe_block_state& state = engine->expert_states[slot];
        if (!state.active || state.tier != moe_TIER_VRAM) {
            return;
        }
        subtract_tier_usage(engine, moe_TIER_VRAM, state.bytes);
        // BUGFIX 608: Check vram_expert_count > 0 before fetch_sub ★
        // Problem: vram_expert_count can underflow if fetch_sub called when count is 0
        // Solution: Check before decrement
        if (engine->vram_expert_count.load(std::memory_order_relaxed) > 0) {
            engine->vram_expert_count.fetch_sub(1);
        }
        state.tier = tier_used(engine, moe_TIER_RAM) + state.bytes <= tier_budget(engine, moe_TIER_RAM) ?
            moe_TIER_RAM : moe_TIER_DB;
        // BUGFIX 153: Add memory_order_relaxed for statistics counters
        if (state.tier == moe_TIER_RAM) engine->ram_expert_count.fetch_add(1, std::memory_order_relaxed);
        else if (state.tier == moe_TIER_DB) engine->db_expert_count.fetch_add(1, std::memory_order_relaxed);
        add_tier_usage(engine, state.tier, state.bytes);
        state_copy = state;
    }
    replace_evict_candidate(engine, slot, state_copy);
    device_free_expert(engine, layer, expert);
}

static int prefetch_tensor_pinned_to_gpu(
    moe_pc_engine* engine,
    const moe_tensor_record& rec,
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
            moe_DEVICE_POOL_EXPERT,
            engine->staging_slots[staging_slot].buf,
            staging_bytes,
            staging_slot
        );
    } else {
        copied = device_copy_tensor(engine, rec, moe_DEVICE_POOL_EXPERT, allow_sync_fallback);
    }
    if (copied == moe_DEVICE_COPY_BACKPRESSURE) {
        return moe_DEVICE_COPY_BACKPRESSURE;
    }
    if (!copied) {
        demote_failed_device_expert(engine, rec);
        return 0;
    }
    if (sync_after) {
        device_stream_sync_current(engine);
    }
    // BUGFIX 153: Add memory_order_relaxed for statistics counters
    engine->io_atomic.bytes_pinned_to_gpu.fetch_add(bytes, std::memory_order_relaxed);
    engine->io_atomic.gpu_stage_done.fetch_add(1, std::memory_order_relaxed);
    return 1;
}

static int prefetch_expert_triplet_pinned_to_gpu_bundle(
    moe_pc_engine* engine,
    const moe_prefetch_task& task,
    int sync_after = 1
) {
    if (!engine || !task.bundle_triplet ||
        task.bundle_layer < 0 || task.bundle_expert < 0 ||
        task.staging_slot >= engine->staging_slots.size() ||
        !engine->staging_slots[task.staging_slot].buf) {
        return 0;
    }
    const moe_projection_t projs[3] = {moe_PROJ_GATE, moe_PROJ_UP, moe_PROJ_DOWN};
    uint64_t copied_total = 0;
    for (uint32_t i = 0; i < 3u; ++i) {
        const moe_tensor_record* rec = find_tensor_record(
            engine,
            task.bundle_layer,
            task.bundle_expert,
            projs[i]);
        if (!rec || task.bundle_bytes[i] == 0 ||
            task.bundle_offsets[i] > engine->staging_slots[task.staging_slot].size ||
            task.bundle_bytes[i] > engine->staging_slots[task.staging_slot].size - task.bundle_offsets[i]) {
            if (rec) {
                demote_failed_device_expert(engine, *rec);
            }
            release_staging_slot(engine, task.staging_slot);
            return 0;
        }
        const uint8_t* host = engine->staging_slots[task.staging_slot].buf + task.bundle_offsets[i];
        const int ok = device_copy_tensor_from_host(
            engine,
            *rec,
            moe_DEVICE_POOL_EXPERT,
            host,
            task.bundle_bytes[i],
            task.staging_slot,
            0);
        if (!ok) {
            demote_failed_device_expert(engine, *rec);
            release_staging_slot(engine, task.staging_slot);
            return 0;
        }
        copied_total += task.bundle_bytes[i];
    }
    release_staging_slot(engine, task.staging_slot);
    if (sync_after) {
        device_stream_sync_current(engine);
    }
    engine->io_atomic.bytes_pinned_to_gpu.fetch_add(copied_total, std::memory_order_relaxed);
    engine->io_atomic.gpu_stage_done.fetch_add(1, std::memory_order_relaxed);
    return 1;
}

static int prefetch_tensor_direct_to_gpu(moe_pc_engine* engine, const moe_tensor_record& rec) {
    if (!engine) {
        return 0;
    }
    const uint64_t bytes = tensor_stream_bytes(rec);
    int copied = device_copy_tensor_gpudirect_storage(engine, rec, moe_DEVICE_POOL_EXPERT);
    if (!copied) {
        // CRITICAL FIX: Allow sync fallback when staging slots are exhausted.
        // Before: allow_sync_fallback=0 caused infinite BACKPRESSURE loop when slots unavailable.
        // After: allow_sync_fallback=1 enables sync DMA as last resort, ensuring forward progress.
        // Sync DMA blocks IO worker but completes the transfer; BACKPRESSURE blocks forever.
        copied = device_copy_tensor(engine, rec, moe_DEVICE_POOL_EXPERT, 1);
    }
    if (copied == moe_DEVICE_COPY_BACKPRESSURE) {
        return moe_DEVICE_COPY_BACKPRESSURE;
    }
    if (!copied) {
        demote_failed_device_expert(engine, rec);
        return 0;
    }
    device_stream_sync_current(engine);
    // BUGFIX 153: Add memory_order_relaxed for statistics counters
    engine->io_atomic.bytes_direct_to_gpu.fetch_add(bytes, std::memory_order_relaxed);
    engine->io_atomic.direct_stage_done.fetch_add(1, std::memory_order_relaxed);
    return 1;
}

static int prefetch_tensor_record(moe_pc_engine* engine, const moe_tensor_record& rec, uint32_t worker_index = 0) {
    if (!prefetch_tensor_disk_to_ram(engine, rec)) {
        return 0;
    }
    // Fix 39: Removed staging acquire+copy+release that performed no GPU transfer.
    return 1;
}

