static int prefetch_one_raw_span(
    moe_pc_engine* e,
    const moe_storage_raw_span_spec_t& s,
    uint8_t pool,
    int allow_sync_fallback = 1
);

// BUGFIX 62: Forward declaration for dynamic batch_max function ?�★
static uint32_t moe_gpu_stage_batch_max(const moe_pc_engine* engine);

static int io_worker_requeue_disk_task(
    moe_pc_engine* engine,
    const moe_prefetch_task& task,
    moe_transfer_stage_t stage = moe_TRANSFER_STAGE_DISK_TO_RAM,
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
        moe_prefetch_task retry = task;
        retry.stage = stage;
        retry.urgent = (retry.requeue_count >= 3u || retry.stage == moe_TRANSFER_STAGE_RAM_TO_PINNED) ? 1 : 0;
        if (retry.requeue_count < UINT8_MAX) {
            ++retry.requeue_count;
        }
        // BUGFIX 711: Respect disk_stage_depth_cap in requeue path ★★★
        if (engine->disk_stage_depth_cap > 0 &&
            engine->disk_stage.depth.load(std::memory_order_relaxed) >= engine->disk_stage_depth_cap) {
            // Queue full - wait
            std::this_thread::yield();
            continue;
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
static uint64_t io_worker_device_allocated_bytes(const moe_pc_engine* engine) {
    return engine ? engine->device_allocated_bytes.load(std::memory_order_relaxed) : 0;
}

static void proactive_eviction_worker_loop(moe_pc_engine* engine) {
    if (!engine) {
        return;
    }
    // BUGFIX 95: Issue 16 - Cache io_stop load outside loop ★
    // Problem: Atomic load on every iteration adds overhead
    // Solution: Cache load, re-check only after CV wakeup
    const auto should_stop = [&]() { return engine->io_stop.load(std::memory_order_relaxed); };
    while (!should_stop()) {
        // BUGFIX 81: Problem 6 - Replace 5ms polling with event-driven CV ★★
        // Old: 5ms sleep caused CPU overhead and potential oscillation.
        // New: Wait on vram_eviction_cv. Triggered instantly when allocation > 85%.
        // Fallback to 50ms wait to periodically check RAM eviction.
        {
            std::unique_lock<std::mutex> lock(engine->vram_eviction_mutex);
            engine->vram_eviction_cv.wait_for(lock, std::chrono::milliseconds(25), [&]() {
                return engine->vram_eviction_needed.load(std::memory_order_relaxed) ||
                       engine->ram_eviction_needed.load(std::memory_order_relaxed) ||
                       engine->io_stop.load(std::memory_order_relaxed);
            });
            engine->vram_eviction_needed.store(false, std::memory_order_relaxed);
            engine->ram_eviction_needed.store(false, std::memory_order_relaxed);
        }
        if (engine->io_stop.load(std::memory_order_relaxed)) {
            break;
        }
        const uint64_t vram_bud = tier_budget(engine, moe_TIER_VRAM);
        const uint64_t ram_bud = tier_budget(engine, moe_TIER_RAM);
        const uint64_t ram_used = tier_budget(engine, moe_TIER_RAM) > 0 ? tier_used(engine, moe_TIER_RAM) : 0;
        const uint64_t vram_used = tier_used(engine, moe_TIER_VRAM);

        // BUGFIX 57: Issue 2 - Lower proactive eviction trigger to 85% ★★
        // Problem: Attention prefetch cutoff at 87%, proactive eviction at 90%
        // →3% dead zone where attention blocked but eviction not started
        // →CPU fallback or mmap streaming for attention weights
        // Solution: Lower eviction trigger to 85%, target to 75%
        // Ensures eviction starts BEFORE attention prefetch cutoff
        // Trade-off: More frequent eviction, but prevents attention stalls
        // BUGFIX 99: Precompute percentage thresholds
        const uint64_t vram_target_pct  = engine->io_config.vram_eviction_threshold_pct ? engine->io_config.vram_eviction_threshold_pct : 75u;
        const uint64_t vram_trigger_pct  = vram_target_pct + 10u <= 100u ? vram_target_pct + 10u : 100u;
        const uint64_t vram_target = (vram_bud * vram_target_pct) / 100ull;
        const uint64_t vram_threshold_85 = (vram_bud * vram_trigger_pct) / 100ull;
        if (vram_bud > 0 && vram_used > vram_threshold_85 && vram_used > vram_target) {
            std::vector<std::pair<int, int>> freed;
            demote_cold_until(engine, moe_TIER_VRAM, vram_bud - vram_target, freed);
            if (!freed.empty() && !engine->io_stop.load(std::memory_order_relaxed)) {
                // BUGFIX 36: Reduce batch_size from 8 to 4 to lower mutex hold time ?★??
                // Old: 8 × cuMemFree (100µs each) = 800µs device_mutex hold
                // New: 4 × cuMemFree = 400µs device_mutex hold
                // All device_copy_tensor calls wait on device_mutex ??halving hold time improves throughput
                // Trade-off: More batches needed for large evictions, but each batch blocks less
                device_free_experts_batched(engine, freed, 4);
            }
        }

        // BUGFIX 124: Issue 29 - Skip RAM eviction for Metal unified memory ★★
        // Metal uses unified memory budget tracked in VRAM only (ram_budget_bytes = 0).
        // RAM eviction would interfere with OS's native unified memory pressure management.
        // Metal automatically pages out to backing store when needed.
        const bool skip_ram_eviction = (engine->backend_caps.backend == moe_BACKEND_METAL &&
                                        engine->backend_caps.supports_unified_memory);

        if (!skip_ram_eviction) {
            const uint64_t ram_target_pct  = engine->io_config.ram_eviction_threshold_pct  ? engine->io_config.ram_eviction_threshold_pct  : 80u;
            const uint64_t ram_trigger_pct  = ram_target_pct + 5u <= 100u ? ram_target_pct + 5u : 100u;
            const uint64_t ram_target = (ram_bud * ram_target_pct) / 100ull;
            const uint64_t ram_threshold_85 = (ram_bud * ram_trigger_pct) / 100ull;
            if (ram_bud > 0 && ram_used > ram_threshold_85 && ram_used > ram_target) {
                std::vector<std::pair<int, int>> ignored;
                demote_cold_until(engine, moe_TIER_RAM, ram_bud - ram_target, ignored);
            }
        }

        // BUGFIX 235: Check and compact eviction heaps if needed
        // Problem: pop_valid_evict_candidate sets evict_heap_compact_needed flag when stale ratio high,
        // but proactive_eviction_worker_loop never checks this flag to actually compact the heap
        // Solution: Check flag and compact all heaps periodically
        // Impact: Prevents heap from accumulating stale entries
        if (engine->evict_heap_compact_needed.load(std::memory_order_relaxed)) {
            engine->evict_heap_compact_needed.store(false, std::memory_order_relaxed);
            for (int idx = 0; idx < 3; ++idx) {
                compact_evict_heap(engine, idx);
            }
        }

        // BUGFIX 문제B: Adaptive queue depth adjustment ★★
        // Problem: Static queue depth caps don't adapt to runtime bottlenecks
        // pinned_stage saturation → disk_stage should reduce cap
        // gpu_stage idle → pinned_stage can increase cap
        // Solution: Periodically adjust caps based on queue utilization
        // Trade-off: Increased pinned cap uses more staging memory
        {
            const uint32_t disk_depth = engine->disk_stage.depth.load(std::memory_order_relaxed);
            const uint32_t pin_depth = engine->pinned_stage.depth.load(std::memory_order_relaxed);
            const uint32_t gpu_depth = engine->gpu_stage.depth.load(std::memory_order_relaxed);
            const uint32_t base = engine->io_config.max_prefetch_queue;

            // pinned_stage saturated → reduce disk_stage cap to prevent overflow
            if (pin_depth > engine->pinned_stage_depth_cap * 9 / 10) {
                const uint32_t new_disk_cap = std::max(
                    engine->disk_stage_depth_cap * 3 / 4,
                    std::max(512u, base / 4)  // Never go below 512 or base/4
                );
                if (new_disk_cap != engine->disk_stage_depth_cap) {
                    engine->disk_stage_depth_cap = new_disk_cap;
                }
            }

            // gpu_stage idle AND pinned_stage underutilized → increase pinned cap
            if (gpu_depth < 4 && pin_depth < engine->pinned_stage_depth_cap * 3 / 10) {
                const uint32_t max_pin_cap = std::min(base * 2,
                    (uint32_t)(engine->staging_size / (1024 * 1024)));  // Limit by staging memory
                const uint32_t new_pin_cap = std::min(
                    engine->pinned_stage_depth_cap * 5 / 4,
                    max_pin_cap
                );
                if (new_pin_cap != engine->pinned_stage_depth_cap &&
                    new_pin_cap <= max_pin_cap) {
                    engine->pinned_stage_depth_cap = new_pin_cap;
                }
            }
        }

    }
}

static void io_worker_loop(moe_pc_engine* engine, uint32_t worker_index) {
    if (!engine) {
        return;
    }

    // BUGFIX 27: Add StreamGuard to prevent CUDA stream leak ?�★
    // Old: device_stream_destroy_current only on normal exit paths
    // New: RAII guard ensures cleanup on exception/abnormal termination
    // Matches gpu_worker_loop pattern for consistency
    struct StreamGuard {
        moe_pc_engine* e;
        ~StreamGuard() {
            if (e) {
                device_stream_destroy_current(e);
            }
        }
    } stream_guard{engine};

    // BUGFIX 129: Add InflightGuard to prevent inflight count leaks ★★
    // Problem: Worker terminates (worker_index >= worker_target) while task is popped
    // inflight incremented → worker exits → finish_stage_task never called → inflight stuck
    // Solution: RAII guard ensures finish_stage_task on all exit paths
    // BUGFIX 251: Guard must be activated AFTER pop_stage_task succeeds ★★★
    // Problem: Guard created before pop_stage_task → if pop fails, inflight not incremented
    // but destructor tries to decrement → underflow
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
    const uint64_t initial_allocated = io_worker_device_allocated_bytes(engine);
    uint64_t last_vram_room = initial_budget > initial_allocated ? initial_budget - initial_allocated : 0;
    uint32_t staging_turn = 0;
    std::deque<std::pair<moe_prefetch_task, moe_transfer_stage_t>> local_requeues;
    const auto should_stop = [&]() {
        return engine->io_stop.load(std::memory_order_relaxed);
    };
    auto preserve_requeue = [&](const moe_prefetch_task& retry_task, moe_transfer_stage_t retry_stage) {
        if (!io_worker_requeue_disk_task(engine, retry_task, retry_stage, 3u, 1u)) {
            // Bug Fix: Cap local_requeues size to prevent unbounded growth
            // When both disk_stage and gpu_stage are full, local_requeues grows indefinitely
            const uint32_t cap = std::max(8u, engine->disk_stage_depth_cap);
            if (local_requeues.size() < cap) {
                local_requeues.emplace_back(retry_task, retry_stage);
            } else {
                engine->io_atomic.dropped_requests.fetch_add(1, std::memory_order_relaxed);
            }
        }
        return 1;
    };
    auto drain_local_requeues = [&]() {
        // BUGFIX 95: Cache io_stop check in requeue loop
        while (!local_requeues.empty() && !should_stop()) {
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
    // BUGFIX 84: Issue 6 - Limit local_requeues retry attempts ★★
    // Problem: Infinite retry when both disk_stage and gpu_stage full
    // io_worker stuck in requeue loop → no progress
    // Solution: Max 100 retries per item, then drop
    uint32_t requeue_retry_count = 0;
    constexpr uint32_t MAX_REQUEUE_RETRIES = 100;

    for (;;) {
        if (!local_requeues.empty()) {
            const auto pending = local_requeues.front();
            if (io_worker_requeue_disk_task(engine, pending.first, pending.second, 1u, 0u)) {
                local_requeues.pop_front();
                requeue_retry_count = 0;  // Reset on success
                continue;
            }

            // BUGFIX 84: Check retry limit
            if (++requeue_retry_count >= MAX_REQUEUE_RETRIES) {
                engine->io_atomic.dropped_requests.fetch_add(1, std::memory_order_relaxed);
                local_requeues.pop_front();
                requeue_retry_count = 0;
                continue;
            }

            // PIPELINE FIX 10: Wait on correct CV based on requeue target ★★
            // Old: Always waited on disk_stage.pop_cv even when gpu_stage was full
            // New: Wait on gpu_stage.pop_cv when target is GPU, disk_stage.pop_cv otherwise
            // Eliminates 2ms polling delay when GPU stage is the actual bottleneck
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
        const uint64_t cur_allocated = io_worker_device_allocated_bytes(engine);
        const uint64_t cur_vram_room = cur_budget > cur_allocated ? cur_budget - cur_allocated : 0;
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
        // PIPELINE FIX 11: Multi-task lookahead for multiple disk workers ?�★
        // Old: Single lookahead ??only helps 1 worker, others still hit page faults
        // New: Lookahead N tasks where N = disk_workers ??all workers benefit
        // BUGFIX 34: Increase lookahead cap for high disk worker counts ?�★
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
                // BUGFIX 322: Check tensor_index bounds before accessing tensors array ★
                // Problem: tensor_index can be corrupted or out of bounds
                // Accessing engine->tensors[tensor_index] with invalid index causes crash
                // Solution: Check bounds before access
                if (tensor_index < 0 || (size_t)tensor_index >= engine->tensors.size()) {
                    continue;
                }
                const moe_tensor_record& next_rec = engine->tensors[tensor_index];
                mmap_context_t* next_shard = get_or_map_tensor_shard(engine, next_rec);
                // BUGFIX 323: Check next_shard validity before accessing ★
                // Problem: get_or_map_tensor_shard can return null
                // Solution: Check next_shard before use
                if (next_shard) {
                    const uint8_t* next_src = static_cast<const uint8_t*>(mmap_get_ptr(next_shard, 0));
                    // BUGFIX 565: Check tensor_stream_bytes before touch_mapped_range ★
                    // Problem: tensor_stream_bytes can return 0, causing invalid touch
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
            // BUGFIX 324: Add explicit bounds check for tensor_index ★
            // Problem: task.tensor_index can be negative or out of bounds
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
            } else if (should_use_direct_device_path(engine, rec, task.target_tier, active_path)) {
                ok = prefetch_tensor_direct_to_gpu(engine, rec);
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
                }
            } else {
                bool staged_direct_to_gpu = false;
                if (task.target_tier == moe_TIER_VRAM &&
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
                        // BUGFIX 566: Check tensor_stream_bytes before acquire_staging_slot ★
                        // Problem: tensor_stream_bytes can return 0, causing invalid slot allocation
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
                // BUGFIX 22: Snapshot pinned_depth once to avoid TOCTOU ?�★
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
                        inflight_guard.mark_finished();  // BUGFIX 285: Mark as finished before continue ★
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
                        inflight_guard.mark_finished();  // BUGFIX 285: Mark as finished before continue ★
                        if (worker_index >= engine->disk_worker_target.load()) {
                            exit_worker();
                            return;
                        }
                        continue;
                    }
                }
                ok = prefetch_tensor_disk_to_ram(engine, rec);
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

