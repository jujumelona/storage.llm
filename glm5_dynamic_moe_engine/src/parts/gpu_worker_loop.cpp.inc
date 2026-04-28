static void gpu_worker_loop(glm5_pc_engine* engine, uint32_t worker_index) {
    if (!engine) {
        return;
    }
    for (;;) {
        glm5_prefetch_task task{};
        if (!pop_stage_task(engine, engine->gpu_stage, &task, &engine->gpu_worker_target, worker_index)) {
            device_stream_destroy_current(engine);
            return;
        }

        constexpr uint32_t batch_max = 8;
        glm5_prefetch_task batch[batch_max];
        int ok[batch_max] = {};
        int terminal[batch_max] = {};
        uint32_t batch_count = 0;
        batch[batch_count++] = task;

        {
            std::unique_lock<std::mutex> lock(engine->gpu_stage.mtx);
            uint32_t extra_popped = 0;
            while (batch_count < batch_max && !engine->gpu_stage.urgent_queue.empty()) {
                batch[batch_count] = engine->gpu_stage.urgent_queue.front();
                engine->gpu_stage.urgent_queue.pop_front();
                engine->gpu_stage.inflight.fetch_add(1);
                ++batch_count;
                ++extra_popped;
            }
            while (batch_count < batch_max && !engine->gpu_stage.queue.empty()) {
                batch[batch_count] = engine->gpu_stage.queue.front();
                engine->gpu_stage.queue.pop_front();
                engine->gpu_stage.inflight.fetch_add(1);
                ++batch_count;
                ++extra_popped;
            }
            engine->gpu_stage.depth.store(static_cast<uint32_t>(
                engine->gpu_stage.queue.size() + engine->gpu_stage.urgent_queue.size()));
            queue_total_depth_sub(engine, extra_popped);
        }

        for (uint32_t i = 0; i < batch_count; ++i) {
            terminal[i] = 1;
            if (batch[i].tensor_index < engine->tensors.size()) {
                ok[i] = prefetch_tensor_pinned_to_gpu(
                    engine,
                    engine->tensors[batch[i].tensor_index],
                    batch[i].staging_valid ? batch[i].staging_slot : UINT32_MAX,
                    batch[i].staging_bytes,
                    0,
                    0
                );
                if (ok[i] == GLM5_DEVICE_COPY_BACKPRESSURE) {
                    // Keep pinned data! Re-enqueue to gpu_stage for retry.
                    // Discarding to disk_stage wastes the RAM→pinned copy.
                    glm5_prefetch_task retry = batch[i];
                    retry.stage = GLM5_TRANSFER_STAGE_PINNED_TO_GPU;
                    retry.urgent = 0;  // back of queue — wait for VRAM to free
                    if (enqueue_stage_or_retry(engine, engine->gpu_stage, retry)) {
                        ok[i] = 1;
                        terminal[i] = 0;
                        engine->gpu_stage.cv.notify_one();
                    } else {
                        // gpu_stage full: release slot and fall back to disk retry
                        if (batch[i].staging_slot != UINT32_MAX) {
                            release_staging_slot(engine, batch[i].staging_slot);
                            batch[i].staging_slot = UINT32_MAX;
                        }
                        glm5_prefetch_task disk_retry = batch[i];
                        disk_retry.stage = GLM5_TRANSFER_STAGE_DISK_TO_RAM;
                        disk_retry.staging_slot = UINT32_MAX;
                        disk_retry.staging_bytes = 0;
                        disk_retry.staging_valid = 0;
                        disk_retry.urgent = 1;
                        ok[i] = enqueue_stage_or_retry(engine, engine->disk_stage, disk_retry) ? 1 : 0;
                        if (ok[i]) {
                            terminal[i] = 0;
                            engine->disk_stage.cv.notify_one();
                        }
                    }
                }
            }
        }

        device_stream_sync_current(engine);

        // Fix 29 + Bug #7: Guard against double-release.
        // If prefetch_tensor_pinned_to_gpu pushed the slot to s_tl_deferred_staging_slots,
        // device_stream_sync_current already released it. Check before outer release.
        for (uint32_t i = 0; i < batch_count; ++i) {
            if (batch[i].staging_slot != UINT32_MAX && !ok[i]) {
                // Copy failed before async enqueue — slot was NOT deferred, safe to release
                release_staging_slot(engine, batch[i].staging_slot);
            }
            // If ok[i]==true, slot was consumed by async copy and released via
            // device_stream_sync_current → deferred path. Do NOT release again.
        }
        for (uint32_t i = 0; i < batch_count; ++i) {
            finish_stage_task(engine, engine->gpu_stage, ok[i], terminal[i]);
        }

        if (worker_index >= engine->gpu_worker_target.load()) {
            device_stream_destroy_current(engine);
            return;
        }
    }
}
