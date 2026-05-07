                        ok = preserve_requeue(task, moe_TRANSFER_STAGE_DISK_TO_RAM);
                        terminal = 0;
                    }
                }
            } else {
                bool staged_direct_to_gpu = false;
                if (task.bundle_triplet && task.target_tier == moe_TIER_VRAM) {
                    const moe_projection_t bundle_projs[3] = {moe_PROJ_GATE, moe_PROJ_UP, moe_PROJ_DOWN};
                    const moe_tensor_record* bundle_recs[3] = {};
                    for (uint32_t p = 0; p < 3u; ++p) {
                        bundle_recs[p] = find_tensor_record(
                            engine, task.bundle_layer, task.bundle_expert, bundle_projs[p]);
                    }
                    uint64_t bundle_bytes = expert_triplet_or_hardcoded_bytes(
                        engine, task.bundle_layer, task.bundle_expert);
                    int contiguous_bundle = 0;
                    uint64_t contiguous_bundle_offset = 0;
                    uint64_t contiguous_bundle_size = 0;
                    if (bundle_recs[0] && bundle_recs[1] && bundle_recs[2] &&
                        bundle_recs[0]->bundle_native &&
                        bundle_recs[1]->bundle_native &&
                        bundle_recs[2]->bundle_native &&
                        bundle_recs[0]->resolved_path_index == bundle_recs[1]->resolved_path_index &&
                        bundle_recs[0]->resolved_path_index == bundle_recs[2]->resolved_path_index &&
                        bundle_recs[0]->bundle_id == bundle_recs[1]->bundle_id &&
                        bundle_recs[0]->bundle_id == bundle_recs[2]->bundle_id &&
                        bundle_recs[0]->bundle_offset == bundle_recs[1]->bundle_offset &&
                        bundle_recs[0]->bundle_offset == bundle_recs[2]->bundle_offset &&
                        bundle_recs[0]->bundle_size == bundle_recs[1]->bundle_size &&
                        bundle_recs[0]->bundle_size == bundle_recs[2]->bundle_size &&
                        bundle_recs[0]->bundle_size > 0) {
                        contiguous_bundle = 1;
                        contiguous_bundle_offset = bundle_recs[0]->bundle_offset;
                        contiguous_bundle_size = bundle_recs[0]->bundle_size;
                        bundle_bytes = contiguous_bundle_size;
                    }
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
                                if (contiguous_bundle) {
                                    mmap_context_t* shard = get_or_map_tensor_shard(engine, *bundle_recs[0]);
                                    if (!shard || !mmap_get_ptr(shard, 0)) {
                                        bundle_ok = 0;
                                    } else {
                                        const uint64_t file_size = static_cast<uint64_t>(mmap_get_size(shard));
                                        if (contiguous_bundle_offset > file_size ||
                                            contiguous_bundle_size > file_size - contiguous_bundle_offset ||
                                            contiguous_bundle_size > engine->staging_slots[slot].size) {
                                            bundle_ok = 0;
                                        } else {
                                            const uint8_t* src = static_cast<const uint8_t*>(
                                                mmap_get_ptr(shard, static_cast<size_t>(contiguous_bundle_offset)));
                                            if (!src) {
                                                bundle_ok = 0;
                                            } else {
                                                const uint64_t copied = copy_mapped_range_to_staging_linear(
                                                    engine, src, contiguous_bundle_size, slot, &staging_offset);
                                                if (copied != contiguous_bundle_size ||
                                                    staging_offset != contiguous_bundle_size) {
                                                    bundle_ok = 0;
                                                } else {
                                                    engine->io_atomic.bytes_ram_to_pinned.fetch_add(copied, std::memory_order_relaxed);
                                                    for (uint32_t p = 0; p < 3u; ++p) {
                                                        next.bundle_offsets[p] = bundle_recs[p]->bundle_member_offset;
                                                        next.bundle_bytes[p] = tensor_stream_bytes(*bundle_recs[p]);
                                                        if (next.bundle_offsets[p] > contiguous_bundle_size ||
                                                            next.bundle_bytes[p] > contiguous_bundle_size - next.bundle_offsets[p]) {
                                                            bundle_ok = 0;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    for (uint32_t p = 0; p < 3u; ++p) {
                                        const moe_tensor_record* brec = bundle_recs[p];
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
                                }
                                if (!bundle_ok) {
                                    for (uint32_t p = 0; p < 3u; ++p) {
                                        next.bundle_offsets[p] = 0;
                                        next.bundle_bytes[p] = 0;
                                    }
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
                        // Check tensor_stream_bytes before acquire_staging_slot.
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
                        inflight_guard.mark_finished();
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
                        inflight_guard.mark_finished();
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
                    ok = prefetch_tensor_pinned_to_gpu(engine, rec, UINT32_MAX, 0, 1, 0, task.vram_token);
                    if (ok != moe_DEVICE_COPY_BACKPRESSURE) {
                        task.vram_token.valid = false;
                    }
                    if (ok == moe_DEVICE_COPY_BACKPRESSURE) {
                        // Bug #4: GPU full, re-enqueue to gpu_stage not disk_stage
                        task.urgent = 1;
                        task.stage = moe_TRANSFER_STAGE_PINNED_TO_GPU;
                        ok = enqueue_stage_or_retry(engine, engine->gpu_stage, task) ? 1 : 0;
                        if (ok) {
                            terminal = 0;
                            engine->gpu_stage.cv.notify_one();
                        } else if (task.vram_token.valid) {
                            device_vram_release_reservation_token(engine, task.vram_token);
                            task.vram_token.valid = false;
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
