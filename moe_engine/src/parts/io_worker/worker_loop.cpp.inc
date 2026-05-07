        }
    } stream_guard{engine};

    // BUGFIX 129: Add InflightGuard to prevent inflight count leaks ?끸쁾
    // Problem: Worker terminates (worker_index >= worker_target) while task is popped
    // If a worker exits after pop, finish_stage_task must still run.
    // Solution: RAII guard ensures finish_stage_task on all exit paths
    // Guard must be activated after pop_stage_task succeeds.
    // but destructor tries to decrement ??underflow
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
            finished = false;
        }

        void mark_finished() {
            finished = true;
            active = false;
        }
    } inflight_guard{engine, &engine->disk_stage};

    moe_io_path_t active_path = choose_active_io_path(engine);
    engine->io_atomic.active_path.store(static_cast<uint32_t>(active_path));
    const uint64_t initial_budget = device_budget_bytes(engine);
    // BUGFIX 899 follow-up: Remove device_allocated double subtraction
    uint64_t last_vram_room = initial_budget;
    uint32_t staging_turn = 0;
    std::deque<std::pair<moe_prefetch_task, moe_transfer_stage_t>> local_requeues;
    const auto should_stop = [&]() {
        return engine->io_stop.load(std::memory_order_relaxed);
    };
    auto preserve_requeue = [&](const moe_prefetch_task& retry_task, moe_transfer_stage_t retry_stage) {
        if (!io_worker_requeue_disk_task(engine, retry_task, retry_stage, 3u, 1u)) {
            const uint32_t cap = std::max(8u, std::min(64u, engine->disk_stage_depth_cap / 8u));
            if (local_requeues.size() < cap) {
                local_requeues.emplace_back(retry_task, retry_stage);
                uint64_t task_bytes = 0;
                if (retry_task.tensor_index >= 0 && (size_t)retry_task.tensor_index < engine->tensors.size()) {
                    task_bytes = tensor_stream_bytes(engine->tensors[retry_task.tensor_index]);
                } else if (retry_task.raw_span_index >= 0) {
                    const moe_storage_raw_span_spec_t* span = moe_storage_raw_span_at((uint32_t)retry_task.raw_span_index);
                    if (span) {
                        task_bytes = span->bytes;
                    }
                }
                if (task_bytes > 0) {
                    add_tier_usage(engine, moe_TIER_RAM, task_bytes);
                }
            } else {
                if (retry_task.vram_token.valid) {
                    device_vram_release_reservation_token(engine, retry_task.vram_token);
                }
                engine->io_atomic.dropped_requests.fetch_add(1, std::memory_order_relaxed);
            }
        }
        return 1;
    };
    auto drain_local_requeues = [&]() {
        while (!local_requeues.empty() && !should_stop()) {
            const auto pending = local_requeues.front();
            uint64_t task_bytes = 0;
            if (pending.first.tensor_index >= 0 && (size_t)pending.first.tensor_index < engine->tensors.size()) {
                task_bytes = tensor_stream_bytes(engine->tensors[pending.first.tensor_index]);
            } else if (pending.first.raw_span_index >= 0) {
                const moe_storage_raw_span_spec_t* span = moe_storage_raw_span_at((uint32_t)pending.first.raw_span_index);
                if (span) {
                    task_bytes = span->bytes;
                }
            }
            if (io_worker_requeue_disk_task(engine, pending.first, pending.second, 5u, 1u)) {
                local_requeues.pop_front();
                if (task_bytes > 0) {
                    subtract_tier_usage(engine, moe_TIER_RAM, task_bytes);
                }
            } else {
                if (pending.first.vram_token.valid) {
                    device_vram_release_reservation_token(engine, pending.first.vram_token);
                }
                engine->io_atomic.dropped_requests.fetch_add(1, std::memory_order_relaxed);
                local_requeues.pop_front();
                if (task_bytes > 0) {
                    subtract_tier_usage(engine, moe_TIER_RAM, task_bytes);
                }
            }
        }
    };
    auto exit_worker = [&]() {
        drain_local_requeues();
        stream_guard.e = nullptr;
        device_stream_destroy_current(engine);
    };
    // BUGFIX 84: Issue 6 - Limit local_requeues retry attempts ?끸쁾
    // Problem: Infinite retry when both disk_stage and gpu_stage full
    // io_worker stuck in requeue loop ??no progress
    // Solution: Max 100 retries per item, then drop
    uint32_t requeue_retry_count = 0;
    constexpr uint32_t MAX_REQUEUE_RETRIES = 100;

    for (;;) {
        device_drain_deferred_staging_copies(engine, false);
        drain_directstorage_staging_copies(engine);
        if (!local_requeues.empty()) {
            const auto pending = local_requeues.front();
            uint64_t task_bytes = 0;
            if (pending.first.tensor_index >= 0 && (size_t)pending.first.tensor_index < engine->tensors.size()) {
                task_bytes = tensor_stream_bytes(engine->tensors[pending.first.tensor_index]);
            } else if (pending.first.raw_span_index >= 0) {
                const moe_storage_raw_span_spec_t* span = moe_storage_raw_span_at((uint32_t)pending.first.raw_span_index);
                if (span) {
                    task_bytes = span->bytes;
                }
            }
            if (io_worker_requeue_disk_task(engine, pending.first, pending.second, 1u, 0u)) {
                local_requeues.pop_front();
                if (task_bytes > 0) {
                    subtract_tier_usage(engine, moe_TIER_RAM, task_bytes);
                }
                requeue_retry_count = 0;
                continue;
            }

            if (++requeue_retry_count >= MAX_REQUEUE_RETRIES) {
                if (pending.first.vram_token.valid) {
                    device_vram_release_reservation_token(engine, pending.first.vram_token);
                }
                engine->io_atomic.dropped_requests.fetch_add(1, std::memory_order_relaxed);
                local_requeues.pop_front();
                if (task_bytes > 0) {
                    subtract_tier_usage(engine, moe_TIER_RAM, task_bytes);
                }
                requeue_retry_count = 0;
                continue;
            }

            moe_stage_queue* target_stage = (pending.second == moe_TRANSFER_STAGE_PINNED_TO_GPU)
                ? &engine->gpu_stage : &engine->disk_stage;
            std::unique_lock<std::mutex> lock(target_stage->mtx);
            target_stage->pop_cv.wait_for(lock, std::chrono::milliseconds(2));
            continue;
        }
        requeue_retry_count = 0;  // Reset when queue empty
        // Fix 10: GDS/mmap routing changes rarely ??refresh every 10 tasks.
        // VRAM pressure changes fast; refresh when allocator budget headroom shifts >10%.
        const uint64_t cur_budget = device_budget_bytes(engine);
        // BUGFIX 899 follow-up: Remove device_allocated double subtraction ★★
        // Problem: device_budget_bytes() already returns free VRAM
        // Solution: Do not subtract cur_allocated again
        const uint64_t cur_vram_room = cur_budget;
        const bool vram_shifted = last_vram_room == 0
            ? cur_vram_room > 0
            : (cur_vram_room > last_vram_room + last_vram_room / 10 ||
               cur_vram_room < last_vram_room - last_vram_room / 10);
        if (vram_shifted) {
            active_path = static_cast<moe_io_path_t>(
                engine->io_atomic.active_path.load(std::memory_order_relaxed));
            last_vram_room = cur_vram_room;
        }
        moe_prefetch_task task{};
        if (!pop_stage_task(engine, engine->disk_stage, &task, &engine->disk_worker_target, worker_index)) {
            exit_worker();
            return;
        }
        // BUGFIX 251: Activate guard only after successful pop
        inflight_guard.activate();

        // Bug 2 Fix: Lookahead touch_mapped_range to prevent blocking page faults
        // touch_mapped_range (MADV_WILLNEED/PrefetchVirtualMemory) is async. If we call it
        // immediately before memcpy, pages may not be in RAM yet ??page fault blocks IO worker.
        // Solution: peek next task and touch its pages NOW, while processing current task.
        // By the time we process next task, its pages will likely be in RAM.
        // PIPELINE FIX 11: Multi-task lookahead for multiple disk workers ?占썩쁾
        // Old: Single lookahead ??only helps 1 worker, others still hit page faults
        // New: Lookahead N tasks where N = disk_workers ??all workers benefit
        // BUGFIX 34: Increase lookahead cap for high disk worker counts ?占썩쁾
        // Old: min(disk_workers, 4) ??workers 5+ still hit page faults
        // New: min(disk_workers * 2, 8) ??scales with worker count, caps at 8
        // Rationale: OS prefetcher can handle 8 concurrent MADV_WILLNEED hints efficiently
        {
            const uint32_t max_lookahead = std::min(runtime_disk_workers(engine) * 4u, 16u);
            std::vector<int32_t> lookahead_indices;
            lookahead_indices.reserve(max_lookahead);
            {
                std::unique_lock<std::mutex> peek_lock(engine->disk_stage.mtx);
                auto collect_indices = [&](const std::deque<moe_prefetch_task>& queue) {
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
                // BUGFIX 322: Check tensor_index bounds before accessing tensors array ??                // Problem: tensor_index can be corrupted or out of bounds
                // Accessing engine->tensors[tensor_index] with invalid index causes crash
                // Solution: Check bounds before access
                if (tensor_index < 0 || (size_t)tensor_index >= engine->tensors.size()) {
                    continue;
                }
                const moe_tensor_record& next_rec = engine->tensors[tensor_index];
                mmap_context_t* next_shard = get_or_map_tensor_shard(engine, next_rec);
                // Check next_shard validity before accessing it.
                // Solution: Check next_shard before use
                if (next_shard) {
                    const uint8_t* next_src = static_cast<const uint8_t*>(mmap_get_ptr(next_shard, 0));
                    // Check tensor_stream_bytes before touch_mapped_range.
                    // Solution: Check size > 0 before touch
                    const uint64_t touch_size = tensor_stream_bytes(next_rec);
                    if (next_src && touch_size > 0) {
                        touch_mapped_range(engine, next_src, touch_size);
                    }
                }
            }
        }

        int ok = 0;
        int terminal = 1;
        if (task.raw_span_index >= 0) {
            const moe_storage_raw_span_spec_t* span = moe_storage_raw_span_at((uint32_t)task.raw_span_index);
            ok = span ? prefetch_one_raw_span(engine, *span, task.raw_pool, 0) : 0;
            // Bug Fix: Raw span BACKPRESSURE handling (same as tensor path)
            // Without this, staging slot exhaustion causes sync H2D fallback, blocking IO worker
            if (ok == moe_DEVICE_COPY_BACKPRESSURE) {
                task.urgent = 1;
                task.stage = moe_TRANSFER_STAGE_PINNED_TO_GPU;
                ok = enqueue_stage_or_retry(engine, engine->gpu_stage, task) ? 1 : 0;
                if (ok) {
                    terminal = 0;
                    engine->gpu_stage.cv.notify_one();
                } else {
                    ok = preserve_requeue(task, moe_TRANSFER_STAGE_DISK_TO_RAM);
                    terminal = 0;
                }
            } else if (ok) {
                const size_t deferred_copies = device_deferred_staging_copy_count(engine);
                if (deferred_copies > 0) {
                    const bool should_flush = deferred_copies >= moe_gpu_stage_batch_max(engine)
                        || (deferred_copies >= 1u
                            && engine->disk_stage.depth.load(std::memory_order_relaxed) == 0
                            && engine->disk_stage.inflight.load(std::memory_order_relaxed) <= 1u);
                    if (should_flush) {
                        device_drain_deferred_staging_copies(engine, true);
                    }
                }
            }
        } else if (task.tensor_index < engine->tensors.size()) {
            // BUGFIX 324: Add explicit bounds check for tensor_index ??            // Problem: task.tensor_index can be negative or out of bounds
            // Accessing engine->tensors[task.tensor_index] with invalid index causes crash
            // Solution: Check task.tensor_index >= 0 before use
            if (task.tensor_index < 0) {
                ok = 0;
            } else {
                const moe_tensor_record& rec = engine->tensors[task.tensor_index];
            if (task.stage == moe_TRANSFER_STAGE_RAM_TO_PINNED &&
                engine_bypass_staging_pipeline(engine)) {
                ok = prefetch_tensor_disk_to_ram(engine, rec);
            } else if (task.stage == moe_TRANSFER_STAGE_RAM_TO_PINNED &&
                engine->io_config.enable_multistage_pipeline && engine->staging) {
                if (enqueue_stage_or_retry(engine, engine->pinned_stage, task)) {
                    ok = 1;
                    terminal = 0;
                    engine->pinned_stage.cv.notify_one();
                } else {
                    ok = preserve_requeue(task, moe_TRANSFER_STAGE_RAM_TO_PINNED);
                    terminal = 0;
                }
            } else if (task.bundle_triplet &&
                task.target_tier == moe_TIER_VRAM &&
                (active_path == moe_IO_PATH_GPUDIRECT_STORAGE ||
                 active_path == moe_IO_PATH_DIRECTSTORAGE_GPU)) {
                ok = prefetch_expert_bundle_direct_to_gpu_bundle(engine, task, worker_index);
                if (ok > 0) {
                    task.vram_token.valid = false;
                } else if (ok == moe_DEVICE_COPY_BACKPRESSURE) {
                    ok = prefetch_tensor_direct_to_gpu(
                        engine,
                        rec,
                        task.vram_token,
                        task.allow_sync_fallback);
                    if (ok != moe_DEVICE_COPY_BACKPRESSURE) {
                        task.vram_token.valid = false;
                    }
                } else {
                    task.vram_token.valid = false;
                }
                if (ok == moe_DEVICE_COPY_BACKPRESSURE) {
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
            } else if (should_use_direct_device_path(engine, rec, task.target_tier, active_path)) {
                ok = prefetch_tensor_direct_to_gpu(
                    engine,
                    rec,
                    task.vram_token,
                    task.allow_sync_fallback);
                if (ok != moe_DEVICE_COPY_BACKPRESSURE) {
                    task.vram_token.valid = false;
                }
                if (ok == moe_DEVICE_COPY_BACKPRESSURE) {
                    task.urgent = 1;
                    task.stage = moe_TRANSFER_STAGE_PINNED_TO_GPU;
                    ok = enqueue_stage_or_retry(engine, engine->gpu_stage, task) ? 1 : 0;
                    if (ok) {
                        terminal = 0;
                        engine->gpu_stage.cv.notify_one();
                    } else {
