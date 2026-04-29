static void pinned_worker_loop(glm5_pc_engine* engine, uint32_t worker_index) {
    if (!engine) {
        return;
    }
    uint32_t pingpong = 0;
    for (;;) {
        glm5_prefetch_task task{};
        if (!pop_stage_task(engine, engine->pinned_stage, &task, &engine->pinned_worker_target, worker_index)) {
            return;
        }
        int ok = 0;
        int terminal = 1;
        if (task.tensor_index < engine->tensors.size()) {
            uint64_t staging_bytes = 0;
            if (task.staging_valid && task.staging_slot != UINT32_MAX) {
                staging_bytes = task.staging_bytes;
                ok = 1;
            } else {
                task.staging_slot = acquire_staging_slot(
                    engine,
                    worker_index,
                    pingpong++,
                    task.target_tier != GLM5_TIER_VRAM
                );
                if (task.staging_slot == UINT32_MAX && engine->io_stop.load()) {
                    finish_stage_task(engine, engine->pinned_stage, 0, terminal);
                    return;
                }
                if (task.staging_slot == UINT32_MAX) {
                    engine->io_atomic.staging_slot_deficit.fetch_add(1);
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
                    }
                    finish_stage_task(engine, engine->pinned_stage, ok, terminal);
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
            if (ok && task.target_tier == GLM5_TIER_VRAM) {
                glm5_prefetch_task next = task;
                next.stage = GLM5_TRANSFER_STAGE_PINNED_TO_GPU;
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
                        next.stage = GLM5_TRANSFER_STAGE_PINNED_TO_GPU;
                        next.urgent = 1;
                        if (enqueue_stage_or_retry(engine, engine->gpu_stage, next)) {
                            terminal = 0;
                            engine->gpu_stage.cv.notify_one();
                            queued_retry = true;
                            break;
                        }
                        next.stage = GLM5_TRANSFER_STAGE_RAM_TO_PINNED;
                        next.urgent = 1;
                        if (next.staging_valid &&
                            enqueue_stage_or_retry(engine, engine->pinned_stage, next)) {
                            terminal = 0;
                            engine->pinned_stage.cv.notify_one();
                            queued_retry = true;
                            break;
                        }
                        std::unique_lock<std::mutex> lock(engine->gpu_stage.mtx);
                        engine->gpu_stage.pop_cv.wait_for(lock, std::chrono::milliseconds(1));
                    }
                    // Bug 22 Fix: Sync fallback when gpu_stage is saturated
                    // Without this, staging data is discarded and must be re-fetched from disk
                    if (!queued_retry && next.staging_valid && next.staging_slot != UINT32_MAX) {
                        const glm5_tensor_record& rec = engine->tensors[task.tensor_index];
                        const int sync_ok = prefetch_tensor_pinned_to_gpu(
                            engine, rec, next.staging_slot, 0, 1, 1);
                        if (sync_ok) {
                            ok = 1;
                            queued_retry = true;
                        }
                        release_staging_slot(engine, next.staging_slot);
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
        if (worker_index >= engine->pinned_worker_target.load()) {
            return;
        }
    }
}

