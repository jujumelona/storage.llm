static void gpu_worker_loop(glm5_pc_engine* engine, uint32_t worker_index) {
    if (!engine) {
        return;
    }
    
    // Problem 6 Fix: RAII guard to prevent CUDA stream leak on abnormal termination
    // If gpu_worker exits via exception or forced termination (not normal pop_stage_task→false),
    // device_stream_destroy_current would not be called → stream leak.
    // This guard ensures cleanup regardless of exit path.
    struct StreamGuard {
        glm5_pc_engine* e;
        ~StreamGuard() { 
            if (e) {
                device_stream_destroy_current(e);
            }
        }
    } stream_guard{engine};
    
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
                    // Bug 4 Fix: Preserve staging_slot when staging_valid==1 to avoid wasting RAM→pinned copy
                    // BACKPRESSURE means GPU is temporarily full. If staging data is valid, keep the slot
                    // and retry gpu_stage with 5ms wait. Only discard as last resort after retry fails.
                    glm5_prefetch_task once_more = batch[i];
                    once_more.stage = GLM5_TRANSFER_STAGE_PINNED_TO_GPU;
                    once_more.urgent = 1;

                    // If staging data is valid, preserve it and retry with longer wait
                    if (batch[i].staging_valid && batch[i].staging_slot != UINT32_MAX) {
                        {
                            std::unique_lock<std::mutex> lk(engine->gpu_stage.mtx);
                            engine->gpu_stage.pop_cv.wait_for(lk, std::chrono::milliseconds(5));
                        }
                        if (enqueue_stage_or_retry(engine, engine->gpu_stage, once_more)) {
                            ok[i] = 1;
                            terminal[i] = 0;
                            engine->gpu_stage.cv.notify_one();
                        } else {
                            // Last resort: discard staging slot after retry failed
                            release_staging_slot(engine, batch[i].staging_slot);
                            batch[i].staging_slot = UINT32_MAX;
                            glm5_prefetch_task pinned_retry = batch[i];
                            pinned_retry.stage = GLM5_TRANSFER_STAGE_RAM_TO_PINNED;
                            pinned_retry.staging_slot = UINT32_MAX;
                            pinned_retry.staging_bytes = 0;
                            pinned_retry.staging_valid = 0;
                            pinned_retry.urgent = 1;
                            ok[i] = enqueue_stage_or_retry(engine, engine->pinned_stage, pinned_retry) ? 1 : 0;
                            if (ok[i]) {
                                terminal[i] = 0;
                                engine->pinned_stage.cv.notify_one();
                            } else {
                                engine->io_atomic.dropped_requests.fetch_add(1, std::memory_order_relaxed);
                            }
                        }
                    } else {
                        // No valid staging data, use shorter 2ms wait
                        {
                            std::unique_lock<std::mutex> lk(engine->gpu_stage.mtx);
                            engine->gpu_stage.pop_cv.wait_for(lk, std::chrono::milliseconds(2));
                        }
                        if (enqueue_stage_or_retry(engine, engine->gpu_stage, once_more)) {
                            ok[i] = 1;
                            terminal[i] = 0;
                            engine->gpu_stage.cv.notify_one();
                        } else {
                            if (batch[i].staging_slot != UINT32_MAX) {
                                release_staging_slot(engine, batch[i].staging_slot);
                                batch[i].staging_slot = UINT32_MAX;
                            }
                            glm5_prefetch_task pinned_retry = batch[i];
                            pinned_retry.stage = GLM5_TRANSFER_STAGE_RAM_TO_PINNED;
                            pinned_retry.staging_slot = UINT32_MAX;
                            pinned_retry.staging_bytes = 0;
                            pinned_retry.staging_valid = 0;
                            pinned_retry.urgent = 1;
                            ok[i] = enqueue_stage_or_retry(engine, engine->pinned_stage, pinned_retry) ? 1 : 0;
                            if (ok[i]) {
                                terminal[i] = 0;
                                engine->pinned_stage.cv.notify_one();
                            } else {
                                engine->io_atomic.dropped_requests.fetch_add(1, std::memory_order_relaxed);
                            }
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
