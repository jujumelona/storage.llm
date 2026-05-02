static void pinned_worker_loop(moe_pc_engine* engine, uint32_t worker_index) {
    if (!engine) {
        return;
    }

    // BUGFIX 129: Add InflightGuard to prevent inflight count leaks ★★
    // Problem: Worker terminates while task is popped → inflight stuck
    // Solution: RAII guard ensures finish_stage_task on all exit paths
    // BUGFIX 255: Guard must be activated AFTER pop_stage_task succeeds ★★★
    // Problem: Same as BUGFIX 251 - guard created before pop → underflow on pop failure
    // Solution: Create guard inactive, activate only after successful pop
    struct InflightGuard {
        moe_pc_engine* e;
        moe_stage_queue* stage;
        bool active = false;
        bool finished = false;

        ~InflightGuard() {
            if (e && stage && active && !finished) {
                // Abnormal exit - decrement inflight
                finish_stage_task(e, *stage, 0, 1);
            }
        }

        void activate() {
            active = true;
        }

        void mark_finished() {
            finished = true;
        }
    } inflight_guard{engine, &engine->pinned_stage};

    uint32_t pingpong = 0;
    for (;;) {
        moe_prefetch_task task{};
        if (!pop_stage_task(engine, engine->pinned_stage, &task, &engine->pinned_worker_target, worker_index)) {
            return;
        }
        // BUGFIX 255: Activate guard only after successful pop
        inflight_guard.activate();
        int ok = 0;
        int terminal = 1;
        // BUGFIX 567: Check tensor_index >= 0 before size comparison ★
        // Problem: task.tensor_index can be negative
        // Comparison with size_t causes wraparound
        // Solution: Check >= 0 first
        if (task.tensor_index >= 0 && (size_t)task.tensor_index < engine->tensors.size()) {
            uint64_t staging_bytes = 0;
            if (task.staging_valid && task.staging_slot != UINT32_MAX) {
                staging_bytes = task.staging_bytes;
                ok = 1;
            } else {
                const uint64_t required_bytes = tensor_stream_bytes(engine->tensors[task.tensor_index]);
                // BUGFIX 568: Check required_bytes > 0 before acquire_staging_slot ★
                // Problem: tensor_stream_bytes can return 0
                // Solution: Check before allocation
                if (required_bytes == 0) {
                    ok = 0;
                    finish_stage_task(engine, engine->pinned_stage, ok, terminal);
                    inflight_guard.mark_finished();
                    if (worker_index >= engine->pinned_worker_target.load()) {
                        return;
                    }
                    continue;
                }
                task.staging_slot = acquire_staging_slot(
                    engine,
                    worker_index,
                    pingpong++,
                    required_bytes,
                    true,
                    false
                );
                if (task.staging_slot == UINT32_MAX && engine->io_stop.load()) {
                    finish_stage_task(engine, engine->pinned_stage, 0, terminal);
                    return;
                }
                if (task.staging_slot == UINT32_MAX) {
                    // Bug 3 Fix: Separate staging_slot_waits from staging_slot_deficit
                    // staging_slot_deficit = structural shortage (update_staging_slot_deficit)
                    // staging_slot_waits = transient wait events (pinned_worker retries)
                    engine->io_atomic.staging_slot_waits.fetch_add(1);
                    // Problem 1 Fix: Preserve original urgent flag from router-selected experts.
                    // Only promote to urgent after 3+ retries, never demote existing urgent=1.
                    if (!task.urgent && task.requeue_count >= 3u) {
                        task.urgent = 1;
                    }
                    if (task.requeue_count < UINT8_MAX) {
                        ++task.requeue_count;
                    }
                    if (enqueue_stage_or_retry(engine, engine->pinned_stage, task)) {
                        ok = 1;
                        terminal = 0;
                        engine->pinned_stage.cv.notify_one();
                    } else {
                        // BUGFIX 64: Issue 8 - Track dropped_requests when enqueue fails
                        // FIX 7: Requeue to disk_stage as urgent fallback instead of
                        // silently dropping. Generation loop would otherwise block forever
                        // waiting for an expert that will never arrive.
                        if (task.requeue_count < 10u) {
                            task.urgent = 1;
                            task.stage = moe_TRANSFER_STAGE_DISK_TO_RAM;
                            if (enqueue_stage_or_retry(engine, engine->disk_stage, task)) {
                                engine->disk_stage.cv.notify_one();
                                ok = 1;
                                terminal = 0;
                            } else {
                                engine->io_atomic.dropped_requests.fetch_add(1, std::memory_order_relaxed);
                            }
                        } else {
                            // Exceeded requeue limit — genuinely drop
                            // BUGFIX: Downgrade to RAM tier to prevent permanent stall
                            const uint64_t ram_budget = tier_budget(engine, moe_TIER_RAM);
                            if (ram_budget > 0 && tier_used(engine, moe_TIER_RAM) < (ram_budget * 9 / 10)) {
                                task.target_tier = moe_TIER_RAM;
                                task.urgent = 1;
                                task.requeue_count = 0;
                                if (enqueue_stage_or_retry(engine, engine->disk_stage, task)) {
                                    engine->disk_stage.cv.notify_one();
                                } else {
                                    engine->io_atomic.dropped_requests.fetch_add(1, std::memory_order_relaxed);
                                }
                            } else {
                                engine->io_atomic.dropped_requests.fetch_add(1, std::memory_order_relaxed);
                            }
                        }
                    }
                    finish_stage_task(engine, engine->pinned_stage, ok, terminal);
                    inflight_guard.mark_finished();  // BUGFIX 129: Mark as finished

                    if (worker_index >= engine->pinned_worker_target.load()) {
                        return;
                    }
                    continue;
                }
                ok = prefetch_tensor_ram_to_pinned(
                    engine,
                    engine->tensors[task.tensor_index],
                    task.staging_slot,
                    &staging_bytes,
                    task.target_tier
                );
            }
            if (ok && task.target_tier == moe_TIER_VRAM) {
                moe_prefetch_task next = task;
                next.stage = moe_TRANSFER_STAGE_PINNED_TO_GPU;
                next.staging_bytes = staging_bytes;
                next.staging_valid = staging_bytes == tensor_stream_bytes(engine->tensors[task.tensor_index]) ? 1 : 0;
                if (!next.staging_valid) {
                    release_staging_slot(engine, task.staging_slot);
                    next.staging_slot = UINT32_MAX;
                }
                if (enqueue_stage_or_retry(engine, engine->gpu_stage, next)) {
                    terminal = 0;
                    engine->gpu_stage.cv.notify_one();
                } else {
                    bool queued_retry = false;
                    for (uint32_t retry = 0; retry < 3u && !engine->io_stop.load(); ++retry) {
                        next.stage = moe_TRANSFER_STAGE_PINNED_TO_GPU;
                        next.urgent = 1;
                        if (enqueue_stage_or_retry(engine, engine->gpu_stage, next)) {
                            terminal = 0;
                            engine->gpu_stage.cv.notify_one();
                            queued_retry = true;
                            break;
                        }
                        // Fix: Do NOT push back to pinned_stage, this causes self-ping-pong spin livelocks!
                        // If gpu_stage is full, we must wait, not spin.
                        std::unique_lock<std::mutex> lock(engine->gpu_stage.mtx);
                        engine->gpu_stage.pop_cv.wait_for(lock, std::chrono::milliseconds(1));
                    }
                    // Bug 22 Fix: Sync fallback when gpu_stage is saturated
                    // Without this, staging data is discarded and must be re-fetched from disk
                    if (!queued_retry && next.staging_valid && next.staging_slot != UINT32_MAX) {
                        const moe_tensor_record& rec = engine->tensors[task.tensor_index];
                        const int sync_ok = prefetch_tensor_pinned_to_gpu(
                            engine, rec, next.staging_slot, next.staging_bytes, 1, 1);
                        if (sync_ok) {
                            ok = 1;
                            queued_retry = true;
                            // sync_ok=1이면 device_copy_tensor_from_host 내부에서 이미 release됨
                            // next.staging_slot을 UINT32_MAX로 무효화
                            next.staging_slot = UINT32_MAX;
                        }
                        release_staging_slot(engine, next.staging_slot); // UINT32_MAX면 내부에서 skip됨
                    } else if (!queued_retry) {
                        ok = 0;
                        if (next.staging_slot != UINT32_MAX) {
                            release_staging_slot(engine, next.staging_slot);
                        }
                    }
                }
            } else {
                release_staging_slot(engine, task.staging_slot);
            }
        }

        finish_stage_task(engine, engine->pinned_stage, ok, terminal);
        inflight_guard.mark_finished();  // BUGFIX 129: Mark as finished to prevent double-decrement

        if (worker_index >= engine->pinned_worker_target.load()) {
            return;
        }
    }
}

