static moe_tier_t moe_hot_miss_preferred_tier(
    moe_pc_engine_t* engine,
    int layer,
    int expert,
    uint64_t bytes,
    moe_tier_t preferred_tier
) {
    if (!engine || preferred_tier != moe_TIER_VRAM || bytes == 0 ||
        engine->generation_phase.load(std::memory_order_relaxed) == moe_GEN_PHASE_PREFILL) {
        return preferred_tier;
    }
    const size_t slot = expert_dense_slot(engine, layer, expert);
    if (slot >= engine->expert_states.size()) {
        return preferred_tier;
    }
    {
        std::lock_guard<std::mutex> lock(engine->map_mutex);
        const moe_block_state& state = engine->expert_states[slot];
        if (!state.active || state.tier != moe_TIER_RAM) {
            return preferred_tier;
        }
    }
    const float dma_gbps = engine->backend_caps.dma_bandwidth_gbps > 0.0f ?
        engine->backend_caps.dma_bandwidth_gbps : 12.0f;
    const float pcie_ms = (float)bytes / (dma_gbps * 1e9f / 1000.0f);
    constexpr float cpu_gflops = 100.0f;
    constexpr uint32_t token_count = 1u;
    const float compute_gflops = ((float)bytes / 2.0f * 2.0f * (float)token_count) / 1e9f;
    const float cpu_ms = (compute_gflops / cpu_gflops) * 1000.0f;
    return cpu_ms < pcie_ms ? moe_TIER_RAM : preferred_tier;
}

static int moe_pc_engine_prefetch_expert_triplet_impl(
    moe_pc_engine_t* engine,
    int layer,
    int expert,
    moe_tier_t preferred_tier,
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
    preferred_tier = juju_hint_target_tier(
        engine,
        layer,
        expert,
        preferred_tier,
        notify_workers ? moe_PREFETCH_REASON_ROUTER_SELECTED : moe_PREFETCH_REASON_LOOKAHEAD);
    preferred_tier = moe_hot_miss_preferred_tier(engine, layer, expert, bytes, preferred_tier);
    const int requested = moe_pc_engine_request_expert(engine, layer, expert, bytes, preferred_tier);
    if (!requested) {
        return 0;
    }
    if (preferred_tier == moe_TIER_DB || preferred_tier == moe_TIER_STREAM) {
        return requested;
    }

    const moe_projection_t projs[3] = {moe_PROJ_GATE, moe_PROJ_UP, moe_PROJ_DOWN};
    if (preferred_tier == moe_TIER_VRAM &&
        engine->io_config.enable_compute_io_overlap &&
        engine->io_config.enable_multistage_pipeline &&
        engine->staging &&
        engine->staging_slot_size >= bytes &&
        engine_has_device_upload_workers(engine) &&
        choose_active_io_path(engine) == moe_IO_PATH_PINNED_HOST &&
        ensure_io_ready(engine)) {
        const size_t gate_index = find_tensor_record_index(engine, layer, expert, moe_PROJ_GATE);
        const size_t up_index = find_tensor_record_index(engine, layer, expert, moe_PROJ_UP);
        const size_t down_index = find_tensor_record_index(engine, layer, expert, moe_PROJ_DOWN);
        if (gate_index != tensor_slot_invalid() && gate_index < engine->tensors.size() &&
            up_index != tensor_slot_invalid() && up_index < engine->tensors.size() &&
            down_index != tensor_slot_invalid() && down_index < engine->tensors.size()) {
            engine->io_atomic.active_path.store(
                static_cast<uint32_t>(moe_IO_PATH_PINNED_HOST),
                std::memory_order_relaxed);
            moe_prefetch_task task{};
            task.tensor_index = gate_index;
            task.target_tier = preferred_tier;
            task.bundle_triplet = 1;
            task.bundle_layer = layer;
            task.bundle_expert = expert;
            task.urgent = 1;
            if (enqueue_stage_or_retry(engine, engine->disk_stage, task)) {
                engine->io_atomic.queued_requests.fetch_add(1, std::memory_order_relaxed);
                if (notify_workers) {
                    engine->disk_stage.cv.notify_one();
                }
                return 1;
            }
        }
    }
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

int moe_pc_engine_prefetch_expert_triplet(
    moe_pc_engine_t* engine,
    int layer,
    int expert,
    moe_tier_t preferred_tier
) {
    return moe_pc_engine_prefetch_expert_triplet_impl(
        engine,
        layer,
        expert,
        preferred_tier,
        1
    );
}

int moe_pc_engine_startup_warm_model(moe_pc_engine_t* engine) {
    if (!engine || engine->tensors.empty()) {
        return 0;
    }
    if (!ensure_io_ready(engine)) {
        return 0;
    }
    const moe_storage_constants_t runtime_c = moe_engine_storage_constants_or_static(engine);
    const moe_storage_constants_t* c = &runtime_c;
    if (c->experts_per_moe_layer == 0) {
        return 0;
    }
    // BUGFIX 99: Precompute percentage thresholds
    const uint64_t vram_limit = tier_budget(engine, moe_TIER_VRAM) ?
        (tier_budget(engine, moe_TIER_VRAM) * 90ull) / 100ull : 0;
    const uint64_t ram_limit = tier_budget(engine, moe_TIER_RAM) ?
        (tier_budget(engine, moe_TIER_RAM) * 90ull) / 100ull : 0;
    uint64_t planned_vram = tier_used(engine, moe_TIER_VRAM);
    uint64_t planned_ram = tier_used(engine, moe_TIER_RAM);
    uint32_t queued = 0;
    const uint32_t queue_cap = engine->io_config.max_prefetch_queue ?
        engine->io_config.max_prefetch_queue : 4096u;
    const uint32_t queue_high = queue_cap > 32u ? queue_cap - 32u : queue_cap;

    struct startup_expert_candidate_t {
        uint32_t layer = 0;
        uint32_t expert = 0;
        uint64_t bytes = 0;
        uint16_t priority = 0;
    };
    std::vector<startup_expert_candidate_t> candidates;
    const uint64_t layer_count =
        c->last_moe_layer >= c->first_moe_layer ?
        (uint64_t)c->last_moe_layer - (uint64_t)c->first_moe_layer + 1ull : 0ull;
    if (layer_count > 0 && layer_count <= 10000ull &&
        c->experts_per_moe_layer <= 10000u &&
        layer_count <= UINT64_MAX / (uint64_t)c->experts_per_moe_layer) {
        candidates.reserve((size_t)(layer_count * (uint64_t)c->experts_per_moe_layer));
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
                startup_expert_candidate_t cand{};
                cand.layer = layer;
                cand.expert = expert;
                cand.bytes = bytes;
                cand.priority = expert_prefetch_priority_hint(engine, (int)layer, (int)expert);
                candidates.push_back(cand);
            }
        }
    }
    std::stable_sort(candidates.begin(), candidates.end(),
        [](const startup_expert_candidate_t& a, const startup_expert_candidate_t& b) {
            if (a.priority != b.priority) {
                return a.priority > b.priority;
            }
            if (a.expert != b.expert) {
                return a.expert < b.expert;
            }
            return a.layer < b.layer;
        });

    for (const startup_expert_candidate_t& cand : candidates) {
            const uint32_t layer = cand.layer;
            const uint32_t expert = cand.expert;
            const uint64_t bytes = cand.bytes;
            moe_tier_t target = moe_TIER_DB;
            if (vram_limit > 0 && planned_vram + bytes <= vram_limit) {
                target = moe_TIER_VRAM;
                planned_vram += bytes;
            } else if (ram_limit > 0 && planned_ram + bytes <= ram_limit) {
                target = moe_TIER_RAM;
                planned_ram += bytes;
            } else {
                // Stop once the active compute tiers are full. CPU-only roots have
                // no VRAM budget, so RAM warm is the primary way to avoid first-token
                // cold page-fault storms.
                break;
            }

            // BUGFIX 95: Cache atomic loads in throttle loop
            while (true) {
                const bool stop = engine->io_stop.load(std::memory_order_relaxed);
                const uint32_t depth = engine->total_queue_depth.load(std::memory_order_relaxed);
                if (stop || depth < queue_high) break;
                // BUGFIX 24: Use cv wait instead of polling ?�★
                // Old: 10ms sleep polling ??slow response, unnecessary wakeups
                // New: io_done_cv wait ??immediate response when queue drains
                std::unique_lock<std::mutex> lock(engine->done_mutex);
                engine->io_done_cv.wait_for(lock, std::chrono::milliseconds(100), [&] {
                    return engine->io_stop.load(std::memory_order_relaxed) ||
                           engine->total_queue_depth.load(std::memory_order_relaxed) < queue_high;
                });
            }
            if (engine->io_stop.load(std::memory_order_relaxed)) {
                return queued > 0 ? 1 : 0;
            }
            if (moe_pc_engine_prefetch_expert_triplet_impl(
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
                        (unsigned long long)tier_used(engine, moe_TIER_RAM),
                        (unsigned long long)tier_used(engine, moe_TIER_VRAM),
                        engine->total_queue_depth.load(std::memory_order_relaxed));
                    std::fflush(stderr);
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

        // BUGFIX 95: Cache atomic loads in wait loop
        while (true) {
            const bool stop = engine->io_stop.load(std::memory_order_relaxed);
            const uint32_t depth = engine->total_queue_depth.load(std::memory_order_relaxed);
            const uint32_t completed = engine->io_atomic.completed_requests.load();
            if (stop) break;
            const uint32_t failed = engine->io_atomic.failed_requests.load();
            const uint32_t processed_completed = completed >= initial_completed ? completed - initial_completed : 0;
            const uint32_t processed_failed = failed >= initial_failed ? failed - initial_failed : 0;
            const uint32_t processed = processed_completed + processed_failed;

            // PIPELINE FIX 15: Check inflight counters to ensure all DMA completed ?�★??            // Old: depth==0 only checks queue empty, not actual work completion
            // New: Also check inflight==0 for all stages ??ensures GPU DMA finished
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
                    (unsigned long long)tier_used(engine, moe_TIER_RAM),
                    (unsigned long long)tier_used(engine, moe_TIER_VRAM));
            }

            // Timeout after 60 seconds
            if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() > 60) {
                fprintf(stderr, "[storageLLM ERROR] startup warm timeout after 60s (queue=%u completed=%u failed=%u)\n",
                    depth, processed, failed);
                return 0;
            }

            // BUGFIX 28: Use io_done_cv.wait_for instead of sleep polling ?�★??            // Old: sleep_for(100ms) ??up to 100ms delay after loading completes
            // New: io_done_cv.wait_for ??immediate wakeup when IO finishes
            // Reduces startup latency by up to 100ms per check iteration
            std::unique_lock<std::mutex> lock(engine->done_mutex);
            engine->io_done_cv.wait_for(lock, std::chrono::milliseconds(100), [&] {
                const uint32_t depth = engine->total_queue_depth.load(std::memory_order_relaxed);
                const uint32_t completed = engine->io_atomic.completed_requests.load();
                const uint32_t failed = engine->io_atomic.failed_requests.load();
                const uint32_t processed_completed = completed >= initial_completed ? completed - initial_completed : 0;
                const uint32_t processed_failed = failed >= initial_failed ? failed - initial_failed : 0;
                const uint32_t processed = processed_completed + processed_failed;
                const bool all_idle =
                    engine->disk_stage.inflight.load(std::memory_order_acquire) == 0 &&
                    engine->pinned_stage.inflight.load(std::memory_order_acquire) == 0 &&
                    engine->gpu_stage.inflight.load(std::memory_order_acquire) == 0;
                return engine->io_stop.load(std::memory_order_relaxed) ||
                       (depth == 0 && all_idle && processed >= queued);
            });
        }
    }

    return queued > 0 ? 1 : 0;
}

static int prefetch_expert_batch_raw(
    moe_pc_engine_t* engine,
    int layer,
    const int* experts,
    uint32_t expert_count,
    moe_tier_t preferred_tier
) {
    if (!engine || !experts || expert_count == 0) {
        return 0;
    }
    std::vector<uint8_t> stream_inflight(expert_count, 0);
    {
        std::lock_guard<std::mutex> lock(engine->map_mutex);
        for (uint32_t i = 0; i < expert_count; ++i) {
            const size_t slot = expert_dense_slot(engine, layer, experts[i]);
            if (slot < engine->expert_states.size()) {
                const moe_block_state& state = engine->expert_states[slot];
                stream_inflight[i] = state.active && state.tier == moe_TIER_STREAM ? 1u : 0u;
            }
        }
    }
    int ok_count = 0;
    for (uint32_t i = 0; i < expert_count; ++i) {
        if (stream_inflight[i]) {
            ++ok_count;
            continue;
        }
        ok_count += moe_pc_engine_prefetch_expert_triplet_impl(
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

int moe_pc_engine_prefetch_moe_lookahead(
    moe_pc_engine_t* engine,
    int current_layer,
    const int* selected_experts,
    uint32_t selected_count,
    const int* next_layer_candidates,
    uint32_t next_count
) {
    if (!engine || current_layer < 0) {
        return 0;
    }
    // prefetch_plan_clear_seen and duplicate checks lock internally.
    // Holding the same mutex here deadlocks through prefetch plan helpers.
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
            vram_expert_budget_bytes(engine) > 0 ? moe_TIER_VRAM : moe_TIER_RAM
        ) ? 1 : 0;
    }
    const uint32_t phase = engine->generation_phase.load(std::memory_order_relaxed);
    if (phase != moe_GEN_PHASE_PREFILL &&
        engine->io_config.enable_lookahead_prefetch &&
        next_layer_candidates &&
        next_count > 0) {
        std::vector<int> ram_candidates;
        std::vector<int> vram_candidates;
        if (next_count <= 2u) {
            ram_candidates.assign(next_layer_candidates, next_layer_candidates + next_count);
        } else {
            const uint32_t keep_count = std::max(1u, (next_count + 1u) / 2u);
            const uint32_t vram_count = std::max(1u, (next_count * 30u + 99u) / 100u);
            for (uint32_t i = 0; i < keep_count; ++i) {
                if (i < vram_count) {
                    vram_candidates.push_back(next_layer_candidates[i]);
                } else {
                    ram_candidates.push_back(next_layer_candidates[i]);
                }
            }
        }
        if (!vram_candidates.empty()) {
            ok_count += prefetch_expert_batch_raw(
                engine,
                current_layer + 1,
                vram_candidates.data(),
                (uint32_t)vram_candidates.size(),
                moe_TIER_VRAM
            ) ? 1 : 0;
        }
        if (!ram_candidates.empty()) {
            ok_count += prefetch_expert_batch_raw(
                engine,
                current_layer + 1,
                ram_candidates.data(),
                (uint32_t)ram_candidates.size(),
                moe_TIER_RAM
            ) ? 1 : 0;
        }
        if (!vram_candidates.empty() || !ram_candidates.empty()) {
            prefetch_moe_activation_raw(engine, current_layer + 1);
        }
    }
    return ok_count > 0 ? 1 : 0;
}

// ============================================================================
// EAGLE-2 Draft Verification (ICML 2025)
// ============================================================================
// Speculative decoding draft verification structure. Batch verification requires
// parallel forward pass over the drafted tree. Currently pending batch API.
// ============================================================================
