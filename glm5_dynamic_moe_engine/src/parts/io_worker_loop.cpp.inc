static int prefetch_one_raw_span(
    glm5_pc_engine* e,
    const glm5_storage_raw_span_spec_t& s,
    uint8_t pool,
    int allow_sync_fallback = 1
);

static int io_worker_requeue_disk_task(
    glm5_pc_engine* engine,
    const glm5_prefetch_task& task,
    glm5_transfer_stage_t stage = GLM5_TRANSFER_STAGE_DISK_TO_RAM,
    uint32_t attempts = 1,
    uint32_t backoff_ms = 0
) {
    if (!engine) {
        return 0;
    }
    if (attempts == 0) {
        attempts = 1;
    }
    for (uint32_t i = 0; i < attempts && !engine->io_stop.load(); ++i) {
        glm5_prefetch_task retry = task;
        retry.stage = stage;
        retry.urgent = (retry.requeue_count >= 3u || retry.stage == GLM5_TRANSFER_STAGE_RAM_TO_PINNED) ? 1 : 0;
        if (retry.requeue_count < UINT8_MAX) {
            ++retry.requeue_count;
        }
        const int ok = enqueue_stage_or_retry(engine, engine->disk_stage, retry) ? 1 : 0;
        if (ok) {
            engine->disk_stage.cv.notify_one();
            return 1;
        }
        if (backoff_ms > 0 && i + 1u < attempts) {
            std::unique_lock<std::mutex> lock(engine->disk_stage.mtx);
            engine->disk_stage.pop_cv.wait_for(lock, std::chrono::milliseconds(backoff_ms));
        }
    }
    return 0;
}

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((noinline))
#endif
static uint64_t io_worker_device_allocated_bytes(const glm5_pc_engine* engine) {
    return engine ? engine->device_allocated_bytes.load(std::memory_order_relaxed) : 0;
}

static void proactive_eviction_worker_loop(glm5_pc_engine* engine) {
    if (!engine) {
        return;
    }
    while (!engine->io_stop.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        if (engine->io_stop.load(std::memory_order_relaxed)) {
            break;
        }
        const uint64_t vram_bud = tier_budget(engine, GLM5_TIER_VRAM);
        const uint64_t ram_bud = tier_budget(engine, GLM5_TIER_RAM);
        const uint64_t vram_used = tier_used(engine, GLM5_TIER_VRAM);
        const uint64_t vram_target = (vram_bud * 80ull) / 100ull;
        if (vram_bud > 0 && vram_used > (vram_bud * 90ull) / 100ull && vram_used > vram_target) {
            std::vector<std::pair<int, int>> freed;
            demote_cold_until(engine, GLM5_TIER_VRAM, vram_used - vram_target, freed);
            if (!freed.empty() && !engine->io_stop.load(std::memory_order_relaxed)) {
                device_free_experts_batched(engine, freed, 8);
            }
        }
        const uint64_t ram_used = tier_used(engine, GLM5_TIER_RAM);
        const uint64_t ram_target = (ram_bud * 80ull) / 100ull;
        if (ram_bud > 0 && ram_used > (ram_bud * 90ull) / 100ull && ram_used > ram_target) {
            std::vector<std::pair<int, int>> ignored;
            demote_cold_until(engine, GLM5_TIER_RAM, ram_used - ram_target, ignored);
        }
    }
}

static void io_worker_loop(glm5_pc_engine* engine, uint32_t worker_index) {
    if (!engine) {
        return;
    }

    // BUGFIX 27: Add StreamGuard to prevent CUDA stream leak ★★
    // Old: device_stream_destroy_current only on normal exit paths
    // New: RAII guard ensures cleanup on exception/abnormal termination
    // Matches gpu_worker_loop pattern for consistency
    struct StreamGuard {
        glm5_pc_engine* e;
        ~StreamGuard() {
            if (e) {
                device_stream_destroy_current(e);
            }
        }
    } stream_guard{engine};

    glm5_io_path_t active_path = choose_active_io_path(engine);
    engine->io_atomic.active_path.store(static_cast<uint32_t>(active_path));
    uint32_t refresh_turn = 0;
    const uint64_t initial_budget = device_budget_bytes(engine);
    const uint64_t initial_allocated = io_worker_device_allocated_bytes(engine);
    uint64_t last_vram_room = initial_budget > initial_allocated ? initial_budget - initial_allocated : 0;
    std::deque<std::pair<glm5_prefetch_task, glm5_transfer_stage_t>> local_requeues;
    auto preserve_requeue = [&](const glm5_prefetch_task& retry_task, glm5_transfer_stage_t retry_stage) {
        if (!io_worker_requeue_disk_task(engine, retry_task, retry_stage, 3u, 1u)) {
            // Bug Fix: Cap local_requeues size to prevent unbounded growth
            // When both disk_stage and gpu_stage are full, local_requeues grows indefinitely
            const uint32_t cap = std::max(8u, engine->io_config.max_prefetch_queue / 2u);
            if (local_requeues.size() < cap) {
                local_requeues.emplace_back(retry_task, retry_stage);
            } else {
                engine->io_atomic.dropped_requests.fetch_add(1, std::memory_order_relaxed);
            }
        }
        return 1;
    };
    auto drain_local_requeues = [&]() {
        while (!local_requeues.empty() && !engine->io_stop.load(std::memory_order_relaxed)) {
            const auto pending = local_requeues.front();
            if (io_worker_requeue_disk_task(engine, pending.first, pending.second, 5u, 1u)) {
                local_requeues.pop_front();
            } else {
                engine->io_atomic.dropped_requests.fetch_add(1, std::memory_order_relaxed);
                local_requeues.pop_front();
            }
        }
    };
    auto exit_worker = [&]() {
        drain_local_requeues();
        stream_guard.e = nullptr;
        device_stream_destroy_current(engine);
    };
    for (;;) {
        if (!local_requeues.empty()) {
            const auto pending = local_requeues.front();
            if (io_worker_requeue_disk_task(engine, pending.first, pending.second, 1u, 0u)) {
                local_requeues.pop_front();
                continue;
            }
            // PIPELINE FIX 10: Wait on correct CV based on requeue target ★★★
            // Old: Always waited on disk_stage.pop_cv even when gpu_stage was full
            // New: Wait on gpu_stage.pop_cv when target is GPU, disk_stage.pop_cv otherwise
            // Eliminates 2ms polling delay when GPU stage is the actual bottleneck
            glm5_stage_queue* target_stage = (pending.second == GLM5_TRANSFER_STAGE_PINNED_TO_GPU)
                ? &engine->gpu_stage : &engine->disk_stage;
            std::unique_lock<std::mutex> lock(target_stage->mtx);
            target_stage->pop_cv.wait_for(lock, std::chrono::milliseconds(2));
            continue;
        }
        // Fix 10: GDS/mmap routing changes rarely — refresh every 10 tasks.
        // VRAM pressure changes fast; refresh when allocator budget headroom shifts >10%.
        const uint64_t cur_budget = device_budget_bytes(engine);
        const uint64_t cur_allocated = io_worker_device_allocated_bytes(engine);
        const uint64_t cur_vram_room = cur_budget > cur_allocated ? cur_budget - cur_allocated : 0;
        const bool vram_shifted = last_vram_room == 0
            ? cur_vram_room > 0
            : (cur_vram_room > last_vram_room + last_vram_room / 10 ||
               cur_vram_room < last_vram_room - last_vram_room / 10);
        if (vram_shifted || (refresh_turn++ % 10u) == 0u) {
            active_path = static_cast<glm5_io_path_t>(
                engine->io_atomic.active_path.load(std::memory_order_relaxed));
            last_vram_room = cur_vram_room;
        }
        glm5_prefetch_task task{};
        if (!pop_stage_task(engine, engine->disk_stage, &task, &engine->disk_worker_target, worker_index)) {
            exit_worker();
            return;
        }

        // Bug 2 Fix: Lookahead touch_mapped_range to prevent blocking page faults
        // touch_mapped_range (MADV_WILLNEED/PrefetchVirtualMemory) is async. If we call it
        // immediately before memcpy, pages may not be in RAM yet → page fault blocks IO worker.
        // Solution: peek next task and touch its pages NOW, while processing current task.
        // By the time we process next task, its pages will likely be in RAM.
        // PIPELINE FIX 11: Multi-task lookahead for multiple disk workers ★★
        // Old: Single lookahead → only helps 1 worker, others still hit page faults
        // New: Lookahead N tasks where N = disk_workers → all workers benefit
        {
            const uint32_t max_lookahead = std::min(runtime_disk_workers(engine), 4u);
            std::vector<int32_t> lookahead_indices;
            lookahead_indices.reserve(max_lookahead);
            {
                std::unique_lock<std::mutex> peek_lock(engine->disk_stage.mtx);
                auto collect_indices = [&](const std::deque<glm5_prefetch_task>& queue) {
                    for (const auto& lookahead_task : queue) {
                        if (lookahead_indices.size() >= max_lookahead) break;
                        if (lookahead_task.tensor_index >= 0 &&
                            (size_t)lookahead_task.tensor_index < engine->tensors.size() &&
                            lookahead_task.raw_span_index < 0) {
                            lookahead_indices.push_back(lookahead_task.tensor_index);
                        }
                    }
                };
                collect_indices(engine->disk_stage.urgent_queue);
                collect_indices(engine->disk_stage.queue);
            }
            for (int32_t tensor_index : lookahead_indices) {
                if (tensor_index < 0 || (size_t)tensor_index >= engine->tensors.size()) {
                    continue;
                }
                const glm5_tensor_record& next_rec = engine->tensors[tensor_index];
                mmap_context_t* next_shard = get_or_map_tensor_shard(engine, next_rec);
                if (next_shard) {
                    const uint8_t* next_src = static_cast<const uint8_t*>(mmap_get_ptr(next_shard, 0));
                    if (next_src) {
                        touch_mapped_range(next_src, tensor_stream_bytes(next_rec));
                    }
                }
            }
        }

        int ok = 0;
        int terminal = 1;
        if (task.raw_span_index >= 0) {
            const glm5_storage_raw_span_spec_t* span = glm5_storage_raw_span_at((uint32_t)task.raw_span_index);
            ok = span ? prefetch_one_raw_span(engine, *span, task.raw_pool, 0) : 0;
            // Bug Fix: Raw span BACKPRESSURE handling (same as tensor path)
            // Without this, staging slot exhaustion causes sync H2D fallback, blocking IO worker
            if (ok == GLM5_DEVICE_COPY_BACKPRESSURE) {
                task.urgent = 1;
                task.stage = GLM5_TRANSFER_STAGE_PINNED_TO_GPU;
                ok = enqueue_stage_or_retry(engine, engine->gpu_stage, task) ? 1 : 0;
                if (ok) {
                    terminal = 0;
                    engine->gpu_stage.cv.notify_one();
                } else {
                    ok = preserve_requeue(task, GLM5_TRANSFER_STAGE_DISK_TO_RAM);
                    terminal = 0;
                }
            } else if (ok) {
                const size_t deferred_copies = device_deferred_staging_copy_count(engine);
                if (deferred_copies > 0) {
                    const bool should_flush = deferred_copies >= glm5_gpu_stage_batch_max
                        || (deferred_copies >= 1u
                            && engine->disk_stage.depth.load(std::memory_order_relaxed) == 0
                            && engine->disk_stage.inflight.load(std::memory_order_relaxed) <= 1u);
                    if (should_flush) {
                        device_drain_deferred_staging_copies(engine, true);
                    }
                }
            }
        } else if (task.tensor_index < engine->tensors.size()) {
            const glm5_tensor_record& rec = engine->tensors[task.tensor_index];
            if (task.stage == GLM5_TRANSFER_STAGE_RAM_TO_PINNED &&
                engine_bypass_staging_pipeline(engine)) {
                ok = prefetch_tensor_disk_to_ram(engine, rec);
            } else if (task.stage == GLM5_TRANSFER_STAGE_RAM_TO_PINNED &&
                engine->io_config.enable_multistage_pipeline && engine->staging) {
                if (enqueue_stage_or_retry(engine, engine->pinned_stage, task)) {
                    ok = 1;
                    terminal = 0;
                    engine->pinned_stage.cv.notify_one();
                } else {
                    ok = preserve_requeue(task, GLM5_TRANSFER_STAGE_RAM_TO_PINNED);
                    terminal = 0;
                }
            } else if (should_use_direct_device_path(engine, rec, task.target_tier, active_path)) {
                ok = prefetch_tensor_direct_to_gpu(engine, rec);
                if (ok == GLM5_DEVICE_COPY_BACKPRESSURE) {
                    task.urgent = 1;
                    task.stage = GLM5_TRANSFER_STAGE_PINNED_TO_GPU;
                    ok = enqueue_stage_or_retry(engine, engine->gpu_stage, task) ? 1 : 0;
                    if (ok) {
                        terminal = 0;
                        engine->gpu_stage.cv.notify_one();
                    } else {
                        ok = preserve_requeue(task, GLM5_TRANSFER_STAGE_DISK_TO_RAM);
                        terminal = 0;
                    }
                }
            } else {
                bool staged_direct_to_gpu = false;
                if (task.target_tier == GLM5_TIER_VRAM &&
                    active_path == GLM5_IO_PATH_PINNED_HOST &&
                    engine_has_device_upload_workers(engine) &&
                    engine->backend_caps.supports_fixed_read_staging &&
                    engine->io_config.enable_multistage_pipeline &&
                    engine->staging &&
                    engine->staging_pinned &&
                    tensor_stream_bytes(rec) >= engine->io_config.direct_upload_min_bytes) {
                    const uint32_t gpu_hard_cap = engine->io_config.max_prefetch_queue;
                    const uint32_t gpu_depth_pre = engine->gpu_stage.depth.load(std::memory_order_relaxed);
                    if (gpu_hard_cap > 0 && gpu_depth_pre >= gpu_hard_cap) {
                        ok = preserve_requeue(task, GLM5_TRANSFER_STAGE_DISK_TO_RAM);
                        terminal = 0;
                        staged_direct_to_gpu = true;
                    } else {
                        const uint32_t slot = acquire_staging_slot(
                            engine,
                            worker_index,
                            refresh_turn,
                            false,
                            true
                        );
                        if (slot != UINT32_MAX) {
                            uint64_t staging_bytes = 0;
                            if (prefetch_tensor_file_to_pinned(engine, rec, slot, &staging_bytes)) {
                                glm5_prefetch_task next = task;
                                next.stage = GLM5_TRANSFER_STAGE_PINNED_TO_GPU;
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
                                        ok = preserve_requeue(task, GLM5_TRANSFER_STAGE_DISK_TO_RAM);
                                        terminal = 0;
                                        staged_direct_to_gpu = true;
                                    }
                                }
                            } else {
                                release_staging_slot(engine, slot);
                            }
                        }
                    }
                }
                if (!staged_direct_to_gpu) {
                // Critical Fix 5: Check pinned_stage capacity BEFORE disk read
                // Reading from disk then discovering pinned_stage is full wastes disk bandwidth.
                // Check hard-cap before prefetch_tensor_disk_to_ram to avoid unnecessary I/O.
                // BUGFIX 22: Snapshot pinned_depth once to avoid TOCTOU ★★
                if (task.target_tier == GLM5_TIER_VRAM &&
                    engine->io_config.enable_multistage_pipeline && engine->staging) {
                    const uint32_t pinned_depth = engine->pinned_stage.depth.load(std::memory_order_relaxed);
                    const uint32_t pinned_hard_cap = engine->io_config.max_prefetch_queue;
                    const uint32_t pinned_soft_cap = (pinned_hard_cap * 3u) / 4u;

                    if (pinned_hard_cap > 0 && pinned_depth >= pinned_hard_cap) {
                        if (task.requeue_count > 0) {
                            std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        }
                        ok = preserve_requeue(task, GLM5_TRANSFER_STAGE_DISK_TO_RAM);
                        terminal = 0;
                        finish_stage_task(engine, engine->disk_stage, ok, terminal);
                        if (worker_index >= engine->disk_worker_target.load()) {
                            exit_worker();
                            return;
                        }
                        continue;
                    }

                    if (pinned_soft_cap > 0 && pinned_depth >= pinned_soft_cap) {
                        ok = preserve_requeue(task, GLM5_TRANSFER_STAGE_DISK_TO_RAM);
                        terminal = 0;
                        finish_stage_task(engine, engine->disk_stage, ok, terminal);
                        if (worker_index >= engine->disk_worker_target.load()) {
                            exit_worker();
                            return;
                        }
                        continue;
                    }
                }
                ok = prefetch_tensor_disk_to_ram(engine, rec);
                if (ok && task.target_tier == GLM5_TIER_VRAM &&
                    engine->io_config.enable_multistage_pipeline && engine->staging) {
                    glm5_prefetch_task next = task;
                    next.stage = GLM5_TRANSFER_STAGE_RAM_TO_PINNED;
                    if (enqueue_stage_or_retry(engine, engine->pinned_stage, next)) {
                        terminal = 0;
                        engine->pinned_stage.cv.notify_one();
                    } else {
                        ok = preserve_requeue(next, GLM5_TRANSFER_STAGE_RAM_TO_PINNED);
                        terminal = 0;
                    }
                } else if (ok && task.target_tier == GLM5_TIER_VRAM &&
                    engine_has_device_upload_workers(engine)) {
                    ok = prefetch_tensor_pinned_to_gpu(engine, rec, UINT32_MAX, 0, 1, 0);
                    if (ok == GLM5_DEVICE_COPY_BACKPRESSURE) {
                        // Bug #4: GPU full, re-enqueue to gpu_stage not disk_stage
                        task.urgent = 1;
                        task.stage = GLM5_TRANSFER_STAGE_PINNED_TO_GPU;
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

        finish_stage_task(engine, engine->disk_stage, ok, terminal);

        if (worker_index >= engine->disk_worker_target.load()) {
            exit_worker();
            return;
        }
    }
}
