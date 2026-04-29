static int prefetch_one_raw_span(glm5_pc_engine* e, const glm5_storage_raw_span_spec_t& s, uint8_t pool);

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

static void io_worker_loop(glm5_pc_engine* engine, uint32_t worker_index) {
    if (!engine) {
        return;
    }
    glm5_io_path_t active_path = choose_active_io_path(engine);
    engine->io_atomic.active_path.store(static_cast<uint32_t>(active_path));
    uint32_t refresh_turn = 0;
    const uint64_t initial_budget = device_budget_bytes(engine);
    const uint64_t initial_allocated = io_worker_device_allocated_bytes(engine);
    uint64_t last_vram_room = initial_budget > initial_allocated ? initial_budget - initial_allocated : 0;
    std::deque<std::pair<glm5_prefetch_task, glm5_transfer_stage_t>> local_requeues;
    auto preserve_requeue = [&](const glm5_prefetch_task& retry_task, glm5_transfer_stage_t retry_stage) {
        if (!io_worker_requeue_disk_task(engine, retry_task, retry_stage, 3u, 1u)) {
            local_requeues.emplace_back(retry_task, retry_stage);
        }
        return 1;
    };
    for (;;) {
        if (!local_requeues.empty()) {
            const auto pending = local_requeues.front();
            if (io_worker_requeue_disk_task(engine, pending.first, pending.second, 1u, 0u)) {
                local_requeues.pop_front();
                continue;
            }
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
            device_stream_destroy_current(engine);
            return;
        }
        int ok = 0;
        int terminal = 1;
        if (task.raw_span_index >= 0) {
            const glm5_storage_raw_span_spec_t* span = glm5_storage_raw_span_at((uint32_t)task.raw_span_index);
            ok = span ? prefetch_one_raw_span(engine, *span, task.raw_pool) : 0;
            if (ok) {
                auto it = s_tl_deferred_staging_slots.find(engine);
                if (it != s_tl_deferred_staging_slots.end()) {
                    const bool should_flush = it->second.size() >= 4u
                        || (it->second.size() >= 1u
                            && engine->disk_stage.depth.load(std::memory_order_relaxed) == 0
                            && engine->disk_stage.inflight.load(std::memory_order_relaxed) <= 1u);
                    if (should_flush) {
                        device_stream_sync_current(engine);
                        device_complete_pending_copies(engine);
                        for (uint32_t sidx : it->second) {
                            release_staging_slot(engine, sidx);
                        }
                        it->second.clear();
                    }
                }
            }
        } else if (task.tensor_index < engine->tensors.size()) {
            const glm5_tensor_record& rec = engine->tensors[task.tensor_index];
            if (task.stage == GLM5_TRANSFER_STAGE_RAM_TO_PINNED &&
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
                if (task.target_tier == GLM5_TIER_VRAM &&
                    engine->io_config.enable_multistage_pipeline && engine->staging) {
                    const uint32_t pinned_depth = engine->pinned_stage.depth.load(std::memory_order_relaxed);
                    const uint32_t pinned_soft_cap = (engine->io_config.max_prefetch_queue * 3u) / 4u;
                    if (pinned_soft_cap > 0 && pinned_depth >= pinned_soft_cap) {
                        ok = preserve_requeue(task, GLM5_TRANSFER_STAGE_DISK_TO_RAM);
                        terminal = 0;
                        finish_stage_task(engine, engine->disk_stage, ok, terminal);
                        if (worker_index >= engine->disk_worker_target.load()) {
                            device_stream_destroy_current(engine);
                            return;
                        }
                        continue;
                    }
                }
                ok = prefetch_tensor_disk_to_ram(engine, rec);
                if (ok && task.target_tier == GLM5_TIER_VRAM &&
                    engine->io_config.enable_multistage_pipeline && engine->staging) {
                    const uint32_t pinned_hard_cap = engine->io_config.max_prefetch_queue;
                    const uint32_t pinned_depth_after_read =
                        engine->pinned_stage.depth.load(std::memory_order_relaxed);
                    if (pinned_hard_cap > 0 && pinned_depth_after_read >= pinned_hard_cap) {
                        ok = preserve_requeue(task, GLM5_TRANSFER_STAGE_RAM_TO_PINNED);
                        terminal = 0;
                        finish_stage_task(engine, engine->disk_stage, ok, terminal);
                        if (worker_index >= engine->disk_worker_target.load()) {
                            device_stream_destroy_current(engine);
                            return;
                        }
                        continue;
                    }
                    glm5_prefetch_task next = task;
                    next.stage = GLM5_TRANSFER_STAGE_RAM_TO_PINNED;
                    if (enqueue_stage_or_retry(engine, engine->pinned_stage, next)) {
                        terminal = 0;
                        engine->pinned_stage.cv.notify_one();
                    } else {
                        ok = preserve_requeue(next, GLM5_TRANSFER_STAGE_RAM_TO_PINNED);
                        terminal = 0;
                    }
                } else if (ok && task.target_tier == GLM5_TIER_VRAM) {
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

        finish_stage_task(engine, engine->disk_stage, ok, terminal);

        // Problem 2 Fix: Proactive eviction at 75% threshold to prevent latency spikes.
        // Without this, demote_cold_until only runs when admission is nearly full, causing synchronous
        // eviction on every prefetch admission. This background eviction keeps headroom.
        if ((refresh_turn % 32u) == 0u) {
            bool ev_expected = false;
            if (engine->io_proactive_eviction_in_progress.compare_exchange_strong(
                    ev_expected, true, std::memory_order_acquire, std::memory_order_relaxed)) {
                struct ProactiveGuard {
                    std::atomic<bool>* flag;
                    ~ProactiveGuard() { flag->store(false, std::memory_order_release); }
                } pguard{&engine->io_proactive_eviction_in_progress};

                const uint64_t vram_bud = tier_budget(engine, GLM5_TIER_VRAM);
                const uint64_t ram_bud = tier_budget(engine, GLM5_TIER_RAM);
                if (vram_bud > 0 && tier_used(engine, GLM5_TIER_VRAM) > (vram_bud * 75ull) / 100ull) {
                    std::vector<std::pair<int, int>> freed;
                    demote_cold_until(engine, GLM5_TIER_VRAM,
                        tier_used(engine, GLM5_TIER_VRAM) - (vram_bud * 65ull) / 100ull, freed);
                    if (!freed.empty()) {
                        device_free_experts_batched(engine, freed, 8);
                    }
                }
                if (ram_bud > 0 && tier_used(engine, GLM5_TIER_RAM) > (ram_bud * 75ull) / 100ull) {
                    std::vector<std::pair<int, int>> freed_dummy;
                    demote_cold_until(engine, GLM5_TIER_RAM,
                        tier_used(engine, GLM5_TIER_RAM) - (ram_bud * 65ull) / 100ull, freed_dummy);
                }
            }
        }

        if (worker_index >= engine->disk_worker_target.load()) {
            device_stream_destroy_current(engine);
            return;
        }
    }
}
