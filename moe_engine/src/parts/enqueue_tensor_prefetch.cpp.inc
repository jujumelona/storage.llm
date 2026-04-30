static int enqueue_tensor_prefetch(
    moe_pc_engine* engine,
    int layer,
    int expert,
    moe_projection_t proj,
    moe_tier_t target_tier,
    int stream_state_prechecked = 0
) {
    if (!engine) {
        return 0;
    }
    const size_t tensor_index = find_tensor_record_index(engine, layer, expert, proj);
    if (tensor_index == tensor_slot_invalid() || tensor_index >= engine->tensors.size()) {
        return 0;
    }
    // Bug #19: Skip if expert is already in STREAM state (transfer in-flight from another thread)
    // thread_local seen_set only deduplicates within one thread; STREAM catches cross-thread dupes
    if (!stream_state_prechecked && expert >= 0) {
        const size_t eslot = expert_dense_slot(layer, expert);
        std::lock_guard<std::mutex> lock(engine->map_mutex);
        if (eslot < engine->expert_states.size() && engine->expert_states[eslot].active &&
            engine->expert_states[eslot].tier == moe_TIER_STREAM) {
            return 1;  // Already being transferred ??skip silently
        }
    }
    if (!engine->io_config.enable_compute_io_overlap) {
        // Fix 3: Even without IO overlap, use async path if IO workers are ready.
        // Only fall back to synchronous when IO workers aren't running.
        if (ensure_io_ready(engine)) {
            // IO workers available - use async path instead of blocking caller
            engine->io_atomic.active_path.store(static_cast<uint32_t>(choose_active_io_path(engine)));
            moe_prefetch_task task{};
            task.tensor_index = tensor_index;
            task.target_tier = target_tier;
            task.urgent = 1; // Mark urgent since caller expects immediate completion
            if (enqueue_stage_or_retry(engine, engine->disk_stage, task)) {
                engine->io_atomic.queued_requests.fetch_add(1);
                return 1;
            }
        }
        // Fallback: truly synchronous (no IO workers)
        engine->io_atomic.active_path.store(static_cast<uint32_t>(choose_active_io_path(engine)));
        const int ok = prefetch_tensor_record(engine, engine->tensors[tensor_index]);
        if (ok && target_tier == moe_TIER_VRAM &&
            engine_has_device_upload_workers(engine)) {
            const int gpu_ok = prefetch_tensor_pinned_to_gpu(engine, engine->tensors[tensor_index]);
            device_stream_destroy_current(engine);
            if (!gpu_ok) {
                demote_failed_device_expert(engine, engine->tensors[tensor_index]);
            }
        }
        if (ok) {
            engine->io_atomic.queued_requests.fetch_add(1);
            engine->io_atomic.completed_requests.fetch_add(1);
        } else {
            engine->io_atomic.failed_requests.fetch_add(1);
        }
        return ok;
    }
    if (!ensure_io_ready(engine)) {
        return 0;
    }
    engine->io_atomic.active_path.store(static_cast<uint32_t>(choose_active_io_path(engine)));
    moe_prefetch_task task{};
    task.tensor_index = tensor_index;
    task.target_tier = target_tier;
    task.urgent = target_tier == moe_TIER_VRAM ? 1 : 0;
    const int queued = (task.urgent || target_tier == moe_TIER_VRAM) ?
        enqueue_stage_or_retry(engine, engine->disk_stage, task) :
        enqueue_stage(engine, engine->disk_stage, task);
    if (!queued) {
        return 0;
    }
    // Fix: Only count queued_requests after successful enqueue
    engine->io_atomic.queued_requests.fetch_add(1);
    return 1;
}
