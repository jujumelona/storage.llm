static void moe_prefetch_task_apply_policy(
    moe_pc_engine* engine,
    moe_prefetch_task& task,
    int requested_sync_fallback,
    moe_prefetch_reason_t reason,
    int urgent_required
) {
    const bool required_now =
        reason == moe_PREFETCH_REASON_ROUTER_SELECTED ||
        urgent_required != 0;
    task.prefetch_reason = reason;
    task.required_now = required_now ? 1u : 0u;
    task.speculative = required_now ? 0u : 1u;
    task.urgent_required = urgent_required ? 1u : 0u;
    task.allow_sync_fallback =
        required_now &&
        requested_sync_fallback &&
        engine &&
        engine->io_config.allow_required_sync_fallback ? 1 : 0;
    if (task.speculative) {
        task.allow_sync_fallback = 0;
    }
}

static int enqueue_tensor_prefetch(
    moe_pc_engine* engine,
    int layer,
    int expert,
    moe_projection_t proj,
    moe_tier_t target_tier,
    int stream_state_prechecked = 0,
    const vram_reservation_token& reservation = vram_reservation_token{},
    int allow_sync_fallback = 0,
    moe_prefetch_reason_t prefetch_reason = moe_PREFETCH_REASON_LOOKAHEAD,
    int urgent_required = 0
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
        const size_t eslot = expert_dense_slot(engine, layer, expert);
        std::lock_guard<std::mutex> lock(engine->map_mutex);
        if (eslot < engine->expert_states.size() && engine->expert_states[eslot].active &&
            engine->expert_states[eslot].tier == moe_TIER_STREAM) {
            if (reservation.valid) {
                device_vram_release_reservation_token(engine, reservation);
            }
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
            task.vram_token = reservation;
            task.urgent = 1; // Mark urgent since caller expects immediate completion
            moe_prefetch_task_apply_policy(
                engine,
                task,
                allow_sync_fallback ? 1 : 0,
                prefetch_reason,
                urgent_required);
            if (enqueue_stage_or_retry(engine, engine->disk_stage, task)) {
                engine->io_atomic.queued_requests.fetch_add(1);
                return 1;
            }
            if (task.vram_token.valid) {
                device_vram_release_reservation_token(engine, task.vram_token);
            }
        }
        // Fallback: truly synchronous (no IO workers)
        engine->io_atomic.active_path.store(static_cast<uint32_t>(choose_active_io_path(engine)));
        const int ok = prefetch_tensor_record(engine, engine->tensors[tensor_index]);
        if (ok && target_tier == moe_TIER_VRAM &&
            engine_has_device_upload_workers(engine)) {
            const int gpu_ok = prefetch_tensor_pinned_to_gpu(
                engine,
                engine->tensors[tensor_index],
                UINT32_MAX,
                0,
                1,
                allow_sync_fallback ? 1 : 0,
                reservation);
            device_stream_destroy_current(engine);
            if (gpu_ok == moe_DEVICE_COPY_BACKPRESSURE) {
                if (reservation.valid) {
                    device_vram_release_reservation_token(engine, reservation);
                }
                demote_failed_device_expert(engine, engine->tensors[tensor_index]);
            } else if (!gpu_ok) {
                demote_failed_device_expert(engine, engine->tensors[tensor_index]);
            }
        } else if (ok && target_tier == moe_TIER_VRAM) {
            if (reservation.valid) {
                device_vram_release_reservation_token(engine, reservation);
            }
            demote_failed_device_expert(engine, engine->tensors[tensor_index]);
        } else if (!ok && reservation.valid) {
            device_vram_release_reservation_token(engine, reservation);
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
        if (reservation.valid) {
            device_vram_release_reservation_token(engine, reservation);
        }
        return 0;
    }
    engine->io_atomic.active_path.store(static_cast<uint32_t>(choose_active_io_path(engine)));
    moe_prefetch_task task{};
    task.tensor_index = tensor_index;
    task.target_tier = target_tier;
    task.vram_token = reservation;
    const moe_tensor_record& rec = engine->tensors[tensor_index];
    task.urgent = (target_tier == moe_TIER_VRAM ||
        rec.prefetch_priority >= 95 ||
        rec.prefetch_class == moe_JUJU_PREFETCH_STARTUP_HOT ||
        rec.prefetch_class == moe_JUJU_PREFETCH_ROUTER_HOT) ? 1 : 0;
    moe_prefetch_task_apply_policy(
        engine,
        task,
        allow_sync_fallback ? 1 : 0,
        prefetch_reason,
        urgent_required);
    const int queued = (task.urgent || target_tier == moe_TIER_VRAM) ?
        enqueue_stage_or_retry(engine, engine->disk_stage, task) :
        enqueue_stage(engine, engine->disk_stage, task);
    if (!queued) {
        if (task.vram_token.valid) {
            device_vram_release_reservation_token(engine, task.vram_token);
        }
        // BUGFIX 237: Track prefetch admission failures
        // Problem: enqueue failure returns 0 without updating statistics
        // Impact: /health monitoring shows incorrect hit rate (failures not counted)
        // Solution: Increment failed_requests counter on enqueue failure
        engine->io_atomic.failed_requests.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    // Fix: Only count queued_requests after successful enqueue
    engine->io_atomic.queued_requests.fetch_add(1);
    return 1;
}
