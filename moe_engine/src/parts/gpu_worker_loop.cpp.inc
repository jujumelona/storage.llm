static void gpu_worker_loop(moe_pc_engine* engine, uint32_t worker_index) {
    if (!engine) {
        return;
    }

    // Problem 6 Fix: RAII guard to prevent CUDA stream leak on abnormal termination
    // If gpu_worker exits via exception or forced termination (not normal pop_stage_task?�false),
    // device_stream_destroy_current would not be called ??stream leak.
    // This guard ensures cleanup regardless of exit path.
    // BUG 4 FIX: Make 'e' mutable to prevent double-destroy on normal exit ??
    struct StreamGuard {
        moe_pc_engine* e;
        ~StreamGuard() {
            if (e) {
                device_stream_destroy_current(e);
            }
        }
    } stream_guard{engine};

    for (;;) {
        moe_prefetch_task task{};
        if (!pop_stage_task(engine, engine->gpu_stage, &task, &engine->gpu_worker_target, worker_index)) {
            stream_guard.e = nullptr;  // BUG 4 FIX: Prevent double-destroy
            device_stream_destroy_current(engine);
            return;
        }

        // PIPELINE FIX 13: Reduce batch size from 8 to 4 ?�★??
        // Update (Issue 2): Use dynamic batch max (up to 8)
        const uint32_t batch_max_dyn = moe_gpu_stage_batch_max(engine);
        constexpr uint32_t batch_capacity = 8;
        moe_prefetch_task batch[batch_capacity];
        int ok[batch_capacity] = {};
        int terminal[batch_capacity] = {};
        uint32_t batch_count = 0;
        batch[batch_count++] = task;

        {
            std::unique_lock<std::mutex> lock(engine->gpu_stage.mtx);
            uint32_t extra_popped = 0;
            while (batch_count < batch_max_dyn && !engine->gpu_stage.urgent_queue.empty()) {
                batch[batch_count] = engine->gpu_stage.urgent_queue.front();
                engine->gpu_stage.urgent_queue.pop_front();
                engine->gpu_stage.inflight.fetch_add(1);
                ++batch_count;
                ++extra_popped;
            }
            while (batch_count < batch_max_dyn && !engine->gpu_stage.queue.empty()) {
                batch[batch_count] = engine->gpu_stage.queue.front();
                engine->gpu_stage.queue.pop_front();
                engine->gpu_stage.inflight.fetch_add(1);
                ++batch_count;
                ++extra_popped;
            }
            // BUGFIX 97: Cache size calculation in gpu_worker
            const uint32_t new_depth = static_cast<uint32_t>(
                engine->gpu_stage.queue.size() + engine->gpu_stage.urgent_queue.size());
            engine->gpu_stage.depth.store(new_depth);
            queue_total_depth_sub(engine, extra_popped);
        }

        // Double-buffering: acquire secondary DMA stream for compute/transfer overlap
        // Research: Splitwise (2024), FlashInfer (2024)
        // While expert[i] computes on the primary stream, expert[i+1]'s
        // DMA can overlap on the secondary stream. This hides transfer
        // latency behind compute for consecutive expert execution.
        CUstream dma_overlap_stream = nullptr;
        if (batch_count > 1 && engine->stream_pool_available.load(std::memory_order_acquire) > 0) {
            std::lock_guard<std::mutex> lock(engine->stream_pool_mutex);
            if (!engine->stream_pool.empty()) {
                dma_overlap_stream = engine->stream_pool.back();
                engine->stream_pool.pop_back();
                engine->stream_pool_available.store(
                    static_cast<uint32_t>(engine->stream_pool.size()),
                    std::memory_order_release);
            }
        }

        for (uint32_t i = 0; i < batch_count; ++i) {
            terminal[i] = 1;
            if (batch[i].raw_span_index >= 0) {
                const moe_storage_raw_span_spec_t* span = moe_storage_raw_span_at((uint32_t)batch[i].raw_span_index);
                ok[i] = span ? prefetch_one_raw_span(engine, *span, batch[i].raw_pool, 1) : 0;
            } else if (batch[i].tensor_index < engine->tensors.size()) {
                ok[i] = prefetch_tensor_pinned_to_gpu(
                    engine,
                    engine->tensors[batch[i].tensor_index],
                    batch[i].staging_valid ? batch[i].staging_slot : UINT32_MAX,
                    batch[i].staging_bytes,
                    0,
                    0
                );
                if (ok[i] == moe_DEVICE_COPY_BACKPRESSURE) {
                    // PIPELINE FIX 4: Release staging slot immediately if no data ?�★??
                    // Old: Hold slot during 5ms wait ??blocks pinned_worker/io_worker
                    // New: Release slot if staging_valid==0 (no data yet) ??unblock pipeline
                    // Keep slot if staging_valid==1 (data already copied) ??preserve work
                    if (!batch[i].staging_valid && batch[i].staging_slot != UINT32_MAX) {
                        release_staging_slot(engine, batch[i].staging_slot);
                        batch[i].staging_slot = UINT32_MAX;
                    }

                    // Bug 4 Fix: Preserve staging_slot when staging_valid==1 to avoid wasting RAM?�pinned copy
                    // BACKPRESSURE means GPU is temporarily full. If staging data is valid, keep the slot
                    // and retry gpu_stage with 5ms wait. Only discard as last resort after retry fails.
                    moe_prefetch_task once_more = batch[i];
                    once_more.stage = moe_TRANSFER_STAGE_PINNED_TO_GPU;
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
                            if (batch[i].staging_slot != UINT32_MAX) {
                                release_staging_slot(engine, batch[i].staging_slot);
                                batch[i].staging_slot = UINT32_MAX;
                            }
                            engine->io_atomic.dropped_requests.fetch_add(1, std::memory_order_relaxed);
                            ok[i] = 0;
                            terminal[i] = 1; // Drop it, avoid ping-pong livelock
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
                            engine->io_atomic.dropped_requests.fetch_add(1, std::memory_order_relaxed);
                            ok[i] = 0;
                            terminal[i] = 1; // Drop it, avoid ping-pong livelock
                        }
                    }
                }
            }
            device_drain_deferred_staging_copies(engine, false);
        }

        device_drain_deferred_staging_copies(engine, true);

        // Return double-buffering DMA stream to pool after batch completes
        if (dma_overlap_stream) {
            // Sync the DMA stream before returning to ensure all transfers complete
            device_backend_stream_sync(engine, dma_overlap_stream);
            std::lock_guard<std::mutex> lock(engine->stream_pool_mutex);
            engine->stream_pool.push_back(dma_overlap_stream);
            engine->stream_pool_available.store(
                static_cast<uint32_t>(engine->stream_pool.size()),
                std::memory_order_release);
            dma_overlap_stream = nullptr;
        }

        // Successful async copies release their staging slot through per-copy events.
        // Failed copies never registered an event, so they are safe to release here.
        for (uint32_t i = 0; i < batch_count; ++i) {
            if (batch[i].staging_slot != UINT32_MAX && !ok[i]) {
                // Copy failed before async enqueue ??slot was NOT deferred, safe to release
                release_staging_slot(engine, batch[i].staging_slot);
            }
        }
        for (uint32_t i = 0; i < batch_count; ++i) {
            finish_stage_task(engine, engine->gpu_stage, ok[i], terminal[i]);
        }

        if (worker_index >= engine->gpu_worker_target.load()) {
            // BUGFIX 111: Prevent CUDA stream double-destroy on dynamic worker downscaling
            // Old: device_stream_destroy_current called directly, but stream_guard.e was not cleared.
            //      When function returns, ~StreamGuard calls it AGAIN -> CUDA_ERROR_INVALID_HANDLE (Crash)
            // New: Clear stream_guard.e before manual destroy, matching pop_stage_task failure path.
            stream_guard.e = nullptr;
            device_stream_destroy_current(engine);
            return;
        }
    }
}

