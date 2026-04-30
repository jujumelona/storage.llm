static int glm5_pc_engine_prefetch_expert_triplet_impl(
    glm5_pc_engine_t* engine,
    int layer,
    int expert,
    glm5_tier_t preferred_tier,
    int notify_workers,
    int stream_state_prechecked = 0
) {
    if (!engine || layer < 0 || expert < 0) {
        return 0;
    }
    const uint64_t bytes = expert_triplet_or_hardcoded_bytes(engine, layer, expert);
    if (bytes == 0) {
        return 0;
    }
    const int requested = glm5_pc_engine_request_expert(engine, layer, expert, bytes, preferred_tier);
    if (!requested) {
        return 0;
    }
    if (preferred_tier == GLM5_TIER_DB || preferred_tier == GLM5_TIER_STREAM) {
        return requested;
    }

    const glm5_projection_t projs[3] = {GLM5_PROJ_GATE, GLM5_PROJ_UP, GLM5_PROJ_DOWN};
    int queued = 0;
    for (auto proj : projs) {
        queued += enqueue_tensor_prefetch(engine, layer, expert, proj, preferred_tier, stream_state_prechecked);
    }
    if (notify_workers) {
        for (int i = 0; i < queued; ++i) {
            engine->disk_stage.cv.notify_one();
        }
    }
    return queued > 0 ? 1 : requested;
}

int glm5_pc_engine_prefetch_expert_triplet(
    glm5_pc_engine_t* engine,
    int layer,
    int expert,
    glm5_tier_t preferred_tier
) {
    return glm5_pc_engine_prefetch_expert_triplet_impl(
        engine,
        layer,
        expert,
        preferred_tier,
        1
    );
}

int glm5_pc_engine_startup_warm_model(glm5_pc_engine_t* engine) {
    if (!engine || engine->tensors.empty()) {
        return 0;
    }
    if (!ensure_io_ready(engine)) {
        return 0;
    }
    const glm5_storage_constants_t* c = glm5_storage_constants();
    if (!c || c->experts_per_moe_layer == 0) {
        return 0;
    }
    const uint64_t vram_limit = tier_budget(engine, GLM5_TIER_VRAM) ?
        (tier_budget(engine, GLM5_TIER_VRAM) * 90ull) / 100ull : 0;
    const uint64_t ram_limit = tier_budget(engine, GLM5_TIER_RAM) ?
        (tier_budget(engine, GLM5_TIER_RAM) * 90ull) / 100ull : 0;
    uint64_t planned_vram = tier_used(engine, GLM5_TIER_VRAM);
    uint64_t planned_ram = tier_used(engine, GLM5_TIER_RAM);
    uint32_t queued = 0;
    const uint32_t queue_cap = engine->io_config.max_prefetch_queue ?
        engine->io_config.max_prefetch_queue : 4096u;
    const uint32_t queue_high = queue_cap > 32u ? queue_cap - 32u : queue_cap;

    for (uint32_t layer = c->first_moe_layer; layer <= c->last_moe_layer; ++layer) {
        for (uint32_t expert = 0; expert < c->experts_per_moe_layer; ++expert) {
            const uint64_t bytes = expert_triplet_or_hardcoded_bytes(
                engine,
                (int)layer,
                (int)expert
            );
            if (bytes == 0) {
                continue;
            }
            glm5_tier_t target = GLM5_TIER_DB;
            if (vram_limit > 0 && planned_vram + bytes <= vram_limit) {
                target = GLM5_TIER_VRAM;
                planned_vram += bytes;
            } else {
                // Stop queuing once VRAM budget is reached.
                // Do NOT queue RAM experts during startup_warm to prevent massive disk IO
                // and 60-second timeouts. RAM experts will be page-faulted on demand.
                return queued > 0 ? 1 : 0;
            }

            while (!engine->io_stop.load(std::memory_order_relaxed) &&
                   engine->total_queue_depth.load(std::memory_order_relaxed) >= queue_high) {
                // BUGFIX 24: Use cv wait instead of polling ★★
                // Old: 10ms sleep polling → slow response, unnecessary wakeups
                // New: io_done_cv wait → immediate response when queue drains
                std::unique_lock<std::mutex> lock(engine->done_mutex);
                engine->io_done_cv.wait_for(lock, std::chrono::milliseconds(100), [&] {
                    return engine->io_stop.load(std::memory_order_relaxed) ||
                           engine->total_queue_depth.load(std::memory_order_relaxed) < queue_high;
                });
            }
            if (engine->io_stop.load(std::memory_order_relaxed)) {
                return queued > 0 ? 1 : 0;
            }
            if (glm5_pc_engine_prefetch_expert_triplet_impl(
                    engine,
                    (int)layer,
                    (int)expert,
                    target,
                    0,
                    1)) {
                ++queued;
                if ((queued & 31u) == 0u) {
                    engine->disk_stage.cv.notify_all();
                }
                if ((queued & 127u) == 0u) {
                    std::fprintf(stderr,
                        "[storagellm] startup warm queued experts=%u ram=%llu vram=%llu queue=%u\n",
                        queued,
                        (unsigned long long)tier_used(engine, GLM5_TIER_RAM),
                        (unsigned long long)tier_used(engine, GLM5_TIER_VRAM),
                        engine->total_queue_depth.load(std::memory_order_relaxed));
                    std::fflush(stderr);
                }
            }
        }
    }
    if (queued > 0) {
        engine->disk_stage.cv.notify_all();
    }

    // CRITICAL FIX: Wait for actual loading to complete
    // startup_warm only queues experts but doesn't wait for IO workers to load them.
    // This causes "ready" state with 0 RAM/VRAM usage and hang on first token.
    // Wait until queue is processed and experts are actually in memory.
    if (queued > 0) {
        fprintf(stderr, "[storagellm] startup warm: waiting for %u experts to load...\n", queued);
        const auto start_time = std::chrono::steady_clock::now();
        const uint32_t initial_completed = engine->io_atomic.completed_requests.load();
        const uint32_t initial_failed = engine->io_atomic.failed_requests.load();

        while (!engine->io_stop.load(std::memory_order_relaxed)) {
            const uint32_t depth = engine->total_queue_depth.load(std::memory_order_relaxed);
            const uint32_t completed = engine->io_atomic.completed_requests.load();
            const uint32_t failed = engine->io_atomic.failed_requests.load();
            const uint32_t processed_completed = completed >= initial_completed ? completed - initial_completed : 0;
            const uint32_t processed_failed = failed >= initial_failed ? failed - initial_failed : 0;
            const uint32_t processed = processed_completed + processed_failed;

            // PIPELINE FIX 15: Check inflight counters to ensure all DMA completed ★★★
            // Old: depth==0 only checks queue empty, not actual work completion
            // New: Also check inflight==0 for all stages → ensures GPU DMA finished
            // Prevents "ready" state while cuMemcpyHtoDAsync still in progress
            const bool all_idle =
                engine->disk_stage.inflight.load(std::memory_order_acquire) == 0 &&
                engine->pinned_stage.inflight.load(std::memory_order_acquire) == 0 &&
                engine->gpu_stage.inflight.load(std::memory_order_acquire) == 0;

            // Check if all queued experts are processed
            if (depth == 0 && all_idle && processed >= queued) {
                fprintf(stderr, "[storagellm] startup warm: complete (processed=%u)\n", processed);
                break;
            }

            // Progress report every 2 seconds
            const auto elapsed = std::chrono::steady_clock::now() - start_time;
            if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() % 2 == 0) {
                fprintf(stderr, "[storagellm] startup warm: queue=%u completed=%u failed=%u ram=%llu vram=%llu\n",
                    depth, processed, failed,
                    (unsigned long long)tier_used(engine, GLM5_TIER_RAM),
                    (unsigned long long)tier_used(engine, GLM5_TIER_VRAM));
            }

            // Timeout after 60 seconds
            if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() > 60) {
                fprintf(stderr, "[storageLLM ERROR] startup warm timeout after 60s (queue=%u completed=%u failed=%u)\n",
                    depth, processed, failed);
                return 0;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    return queued > 0 ? 1 : 0;
}

static int prefetch_expert_batch_raw(
    glm5_pc_engine_t* engine,
    int layer,
    const int* experts,
    uint32_t expert_count,
    glm5_tier_t preferred_tier
) {
    if (!engine || !experts || expert_count == 0) {
        return 0;
    }
    std::vector<uint8_t> stream_inflight(expert_count, 0);
    {
        std::lock_guard<std::mutex> lock(engine->map_mutex);
        for (uint32_t i = 0; i < expert_count; ++i) {
            const size_t slot = expert_dense_slot(layer, experts[i]);
            if (slot < engine->expert_states.size()) {
                const glm5_block_state& state = engine->expert_states[slot];
                stream_inflight[i] = state.active && state.tier == GLM5_TIER_STREAM ? 1u : 0u;
            }
        }
    }
    int ok_count = 0;
    for (uint32_t i = 0; i < expert_count; ++i) {
        if (stream_inflight[i]) {
            ++ok_count;
            continue;
        }
        ok_count += glm5_pc_engine_prefetch_expert_triplet_impl(
            engine,
            layer,
            experts[i],
            preferred_tier,
            1,
            1
        ) ? 1 : 0;
    }
    return ok_count > 0 ? 1 : 0;
}

int glm5_pc_engine_prefetch_moe_lookahead(
    glm5_pc_engine_t* engine,
    int current_layer,
    const int* selected_experts,
    uint32_t selected_count,
    const int* next_layer_candidates,
    uint32_t next_count
) {
    if (!engine || current_layer < 0) {
        return 0;
    }
    std::lock_guard<std::mutex> plan_lock(engine->prefetch_plan_mutex);
    prefetch_plan_clear_seen(engine);

    int ok_count = 0;
    release_moe_activation_raw(engine);
    prefetch_moe_shared_compute_raw(engine, current_layer);
    if (selected_experts && selected_count > 0) {
        ok_count += prefetch_expert_batch_raw(
            engine,
            current_layer,
            selected_experts,
            selected_count,
            vram_expert_budget_bytes(engine) > 0 ? GLM5_TIER_VRAM : GLM5_TIER_RAM
        ) ? 1 : 0;
    }
    if (engine->io_config.enable_lookahead_prefetch && next_layer_candidates && next_count > 0) {
        ok_count += prefetch_expert_batch_raw(
            engine,
            current_layer + 1,
            next_layer_candidates,
            next_count,
            GLM5_TIER_RAM
        ) ? 1 : 0;
        prefetch_moe_activation_raw(engine, current_layer + 1);
    }
    return ok_count > 0 ? 1 : 0;
}
