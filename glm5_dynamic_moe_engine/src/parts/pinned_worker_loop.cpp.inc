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
                    // Bug #3: Sleep before re-enqueue to prevent CPU spin
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    task.urgent = 1;
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
                    &staging_bytes
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
                    next.stage = GLM5_TRANSFER_STAGE_RAM_TO_PINNED;
                    next.urgent = 1;
                    if (next.staging_valid &&
                        enqueue_stage_or_retry(engine, engine->pinned_stage, next)) {
                        terminal = 0;
                        engine->pinned_stage.cv.notify_one();
                    } else {
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

