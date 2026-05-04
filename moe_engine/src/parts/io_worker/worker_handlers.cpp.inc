                        ok = preserve_requeue(task, moe_TRANSFER_STAGE_DISK_TO_RAM);
                        terminal = 0;
                    }
                }
            } else {
                bool staged_direct_to_gpu = false;
                if (task.bundle_triplet && task.target_tier == moe_TIER_VRAM) {
                    const uint64_t bundle_bytes = expert_triplet_or_hardcoded_bytes(
                        engine, task.bundle_layer, task.bundle_expert);
                    if (active_path == moe_IO_PATH_PINNED_HOST &&
                        engine_has_device_upload_workers(engine) &&
                        engine->backend_caps.supports_fixed_read_staging &&
                        engine->io_config.enable_multistage_pipeline &&
                        engine->staging &&
                        engine->staging_pinned &&
                        bundle_bytes > 0) {
                        const uint32_t gpu_hard_cap = engine->gpu_stage_depth_cap;
                        const uint32_t gpu_depth_pre = engine->gpu_stage.depth.load(std::memory_order_relaxed);
                        if (gpu_hard_cap > 0 && gpu_depth_pre >= gpu_hard_cap) {
                            ok = preserve_requeue(task, moe_TRANSFER_STAGE_DISK_TO_RAM);
                            terminal = 0;
                            staged_direct_to_gpu = true;
                        } else {
                            uint32_t slot = acquire_staging_slot(
                                engine,
                                worker_index,
                                staging_turn++,
                                bundle_bytes,
                                false,
                                true
                            );
                            if (slot != UINT32_MAX) {
                                moe_prefetch_task next = task;
                                next.stage = moe_TRANSFER_STAGE_PINNED_TO_GPU;
                                next.staging_slot = slot;
                                next.staging_bytes = 0;
                                next.staging_valid = 0;
                                uint64_t staging_offset = 0;
                                int bundle_ok = 1;
                                const moe_projection_t projs[3] = {moe_PROJ_GATE, moe_PROJ_UP, moe_PROJ_DOWN};
                                for (uint32_t p = 0; p < 3u; ++p) {
                                    const moe_tensor_record* brec = find_tensor_record(
                                        engine, task.bundle_layer, task.bundle_expert, projs[p]);
                                    if (!brec) {
                                        bundle_ok = 0;
                                        break;
                                    }
                                    next.bundle_offsets[p] = staging_offset;
                                    uint64_t copied = 0;
                                    if (!prefetch_tensor_ram_to_pinned_at(
                                            engine, *brec, slot, &staging_offset, &copied, task.target_tier) ||
                                        copied != tensor_stream_bytes(*brec)) {
                                        bundle_ok = 0;
                                        break;
                                    }
                                    next.bundle_bytes[p] = copied;
                                }
                                if (bundle_ok && staging_offset == bundle_bytes) {
                                    next.staging_bytes = staging_offset;
                                    next.staging_valid = 1;
                                    engine->io_atomic.pinned_stage_done.fetch_add(1, std::memory_order_relaxed);
                                    if (enqueue_stage_or_retry(engine, engine->gpu_stage, next)) {
                                        ok = 1;
                                        terminal = 0;
                                        staged_direct_to_gpu = true;
                                        engine->gpu_stage.cv.notify_one();
                                    } else {
                                        release_staging_slot(engine, slot);
                                        ok = preserve_requeue(task, moe_TRANSFER_STAGE_DISK_TO_RAM);
                                        terminal = 0;
                                        staged_direct_to_gpu = true;
                                    }
                                } else {
                                    release_staging_slot(engine, slot);
                                    ok = preserve_requeue(task, moe_TRANSFER_STAGE_DISK_TO_RAM);
                                    terminal = 0;
                                    staged_direct_to_gpu = true;
                                }
                            } else {
                                ok = preserve_requeue(task, moe_TRANSFER_STAGE_DISK_TO_RAM);
                                terminal = 0;
                                staged_direct_to_gpu = true;
                            }
                        }
                    } else {
                        ok = preserve_requeue(task, moe_TRANSFER_STAGE_DISK_TO_RAM);
                        terminal = 0;
                        staged_direct_to_gpu = true;
                    }
                }
                if (!staged_direct_to_gpu &&
                    task.target_tier == moe_TIER_VRAM &&
                    active_path == moe_IO_PATH_PINNED_HOST &&
                    engine_has_device_upload_workers(engine) &&
                    engine->backend_caps.supports_fixed_read_staging &&
                    engine->io_config.enable_multistage_pipeline &&
                    engine->staging &&
                    engine->staging_pinned &&
                    tensor_stream_bytes(rec) >= engine->io_config.direct_upload_min_bytes) {
                    const uint32_t gpu_hard_cap = engine->gpu_stage_depth_cap;
                    const uint32_t gpu_depth_pre = engine->gpu_stage.depth.load(std::memory_order_relaxed);
                    if (gpu_hard_cap > 0 && gpu_depth_pre >= gpu_hard_cap) {
                        ok = preserve_requeue(task, moe_TRANSFER_STAGE_DISK_TO_RAM);
                        terminal = 0;
                        staged_direct_to_gpu = true;
                    } else {
                        // BUGFIX 566: Check tensor_stream_bytes before acquire_staging_slot ??                        // Problem: tensor_stream_bytes can return 0, causing invalid slot allocation
                        // Solution: Check size > 0 before acquiring slot
                        const uint64_t stream_bytes = tensor_stream_bytes(rec);
                        if (stream_bytes == 0) {
                            ok = 0;
                            staged_direct_to_gpu = true;
                        } else {
                        uint32_t slot = acquire_staging_slot(
                            engine,
                            worker_index,
                            staging_turn++,
                            stream_bytes,
                            false,
                            true
                        );
                        if (slot != UINT32_MAX) {
                            uint64_t staging_bytes = 0;
                            if (prefetch_tensor_file_to_pinned(engine, rec, &slot, &staging_bytes)) {
                                moe_prefetch_task next = task;
                                next.stage = moe_TRANSFER_STAGE_PINNED_TO_GPU;
                                next.staging_slot = slot;
                                next.staging_bytes = staging_bytes;
                                next.staging_valid = staging_bytes == tensor_stream_bytes(rec) ? 1 : 0;
                                if (next.staging_valid &&
                                    enqueue_stage_or_retry(engine, engine->gpu_stage, next)) {
                                    ok = 1;
                                    terminal = 0;
                                    staged_direct_to_gpu = true;
                                    engine->gpu_stage.cv.notify_one();
                                } else {
                                    release_staging_slot(engine, slot);
                                    if (next.staging_valid) {
                                        ok = preserve_requeue(task, moe_TRANSFER_STAGE_DISK_TO_RAM);
                                        terminal = 0;
                                        staged_direct_to_gpu = true;
                                    }
                                }
                            } else {
                                release_staging_slot(engine, slot);
                            }
                        }
                        }  // BUGFIX 566: Close if (stream_bytes > 0) block
                    }
                }
                if (!staged_direct_to_gpu) {
                // Critical Fix 5: Check pinned_stage capacity BEFORE disk read
                // Reading from disk then discovering pinned_stage is full wastes disk bandwidth.
                // Check hard-cap before prefetch_tensor_disk_to_ram to avoid unnecessary I/O.
                // BUGFIX 22: Snapshot pinned_depth once to avoid TOCTOU ?占썩쁾
                // BUGFIX NEW-28: Remove duplicate condition check
                if (task.target_tier == moe_TIER_VRAM &&
                    engine->io_config.enable_multistage_pipeline && engine->staging) {
                    const uint32_t pinned_depth = engine->pinned_stage.depth.load(std::memory_order_relaxed);
                    const uint32_t pinned_hard_cap = engine->pinned_stage_depth_cap;
                    const uint32_t pinned_soft_cap = (pinned_hard_cap * 3u) / 4u;

                    if (pinned_hard_cap > 0 && pinned_depth >= pinned_hard_cap) {
                        if (task.requeue_count > 0) {
                            std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        }
                        ok = preserve_requeue(task, moe_TRANSFER_STAGE_DISK_TO_RAM);
                        terminal = 0;
                        finish_stage_task(engine, engine->disk_stage, ok, terminal);
                        inflight_guard.mark_finished();  // BUGFIX 285: Mark as finished before continue ??                        if (worker_index >= engine->disk_worker_target.load()) {
                        if (worker_index >= engine->disk_worker_target.load()) {
                            exit_worker();
                            return;
                        }
                        continue;
                    }

                    if (pinned_soft_cap > 0 && pinned_depth >= pinned_soft_cap) {
                        ok = preserve_requeue(task, moe_TRANSFER_STAGE_DISK_TO_RAM);
                        terminal = 0;
                        finish_stage_task(engine, engine->disk_stage, ok, terminal);
                        inflight_guard.mark_finished();  // BUGFIX 285: Mark as finished before continue ??                        if (worker_index >= engine->disk_worker_target.load()) {
                        if (worker_index >= engine->disk_worker_target.load()) {
                            exit_worker();
                            return;
                        }
                        continue;
                    }
                }
                ok = prefetch_tensor_disk_to_ram(
                    engine,
                    rec,
                    task.target_tier == moe_TIER_VRAM &&
                    engine->io_config.enable_multistage_pipeline &&
                    engine->staging &&
                    engine->ring_adapter.initialized);
                if (ok && task.target_tier == moe_TIER_VRAM &&
                    engine->io_config.enable_multistage_pipeline && engine->staging) {
                    if (active_path == moe_IO_PATH_ZERO_COPY_HOST) {
                        moe_prefetch_task next = task;
                        next.stage = moe_TRANSFER_STAGE_PINNED_TO_GPU;
                        next.staging_slot = UINT32_MAX;
                        next.staging_bytes = 0;
                        next.staging_valid = 0;
                        if (enqueue_stage_or_retry(engine, engine->gpu_stage, next)) {
                            terminal = 0;
                            engine->gpu_stage.cv.notify_one();
                        } else {
                            ok = preserve_requeue(next, moe_TRANSFER_STAGE_DISK_TO_RAM);
                            terminal = 0;
                        }
                    } else {
                        moe_prefetch_task next = task;
                        next.stage = moe_TRANSFER_STAGE_RAM_TO_PINNED;
                        if (enqueue_stage_or_retry(engine, engine->pinned_stage, next)) {
                            terminal = 0;
                            engine->pinned_stage.cv.notify_one();
                        } else {
                            ok = preserve_requeue(next, moe_TRANSFER_STAGE_RAM_TO_PINNED);
                            terminal = 0;
                        }
                    }
                } else if (ok && task.target_tier == moe_TIER_VRAM &&
                    engine_has_device_upload_workers(engine)) {
                    ok = prefetch_tensor_pinned_to_gpu(engine, rec, UINT32_MAX, 0, 1, 0);
                    if (ok == moe_DEVICE_COPY_BACKPRESSURE) {
                        // Bug #4: GPU full, re-enqueue to gpu_stage not disk_stage
                        task.urgent = 1;
                        task.stage = moe_TRANSFER_STAGE_PINNED_TO_GPU;
                        ok = enqueue_stage_or_retry(engine, engine->gpu_stage, task) ? 1 : 0;
                        if (ok) {
                            terminal = 0;
                            engine->gpu_stage.cv.notify_one();
                        }
                    }
                }
                }
            }
            }
        }
        finish_stage_task(engine, engine->disk_stage, ok, terminal);
        inflight_guard.mark_finished();  // BUGFIX 129: Mark as finished to prevent double-decrement

        if (worker_index >= engine->disk_worker_target.load()) {
            exit_worker();
            return;
        }
    }
}
