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
    // BUGFIX 746: Check dma_gbps for zero before division ★★★
    if (dma_gbps <= 0.0f) {
        return preferred_tier;
    }
    const float pcie_ms = (float)bytes / (dma_gbps * 1e9f / 1000.0f);
    // BUGFIX 747: Check pcie_ms for NaN/Inf ★★
    if (!std::isfinite(pcie_ms)) {
        return preferred_tier;
    }
    const uint32_t hw = std::max<uint32_t>(1u, moe_cpu_thread_count_hint());
    const float cpu_gflops =
        (engine->backend_caps.backend == moe_BACKEND_CPU && engine->backend_caps.tflops_hint > 0.0) ?
        std::max(100.0f, (float)(engine->backend_caps.tflops_hint * 1000.0)) :
        std::min(4000.0f, std::max(250.0f, (float)hw * 150.0f));
    constexpr uint32_t token_count = 1u;
    const float compute_gflops = ((float)bytes / 2.0f * 2.0f * (float)token_count) / 1e9f;
    const float cpu_ms = (compute_gflops / cpu_gflops) * 1000.0f;
    // BUGFIX 748: Check cpu_ms for NaN/Inf ★★
    if (!std::isfinite(cpu_ms)) {
        return preferred_tier;
    }
    return cpu_ms < pcie_ms ? moe_TIER_RAM : preferred_tier;
}

static void prefetch_adjust_expert_count(moe_pc_engine_t* engine, moe_tier_t tier, int delta) {
    if (!engine || delta == 0) {
        return;
    }
    std::atomic<uint32_t>* counter = nullptr;
    if (tier == moe_TIER_VRAM) counter = &engine->vram_expert_count;
    else if (tier == moe_TIER_RAM) counter = &engine->ram_expert_count;
    else if (tier == moe_TIER_DB) counter = &engine->db_expert_count;
    if (!counter) {
        return;
    }
    if (delta > 0) {
        counter->fetch_add((uint32_t)delta, std::memory_order_relaxed);
    } else {
        uint32_t cur = counter->load(std::memory_order_relaxed);
        while (cur > 0) {
            if (counter->compare_exchange_weak(
                    cur, cur - 1u,
                    std::memory_order_relaxed,
                    std::memory_order_relaxed)) {
                break;
            }
        }
    }
}

static void prefetch_rollback_unqueued_expert(
    moe_pc_engine_t* engine,
    int layer,
    int expert,
    const moe_block_state& before_state,
    bool had_before_state
) {
    if (!engine || layer < 0 || expert < 0) {
        return;
    }
    const size_t slot = expert_dense_slot(engine, layer, expert);
    if (slot == tensor_slot_invalid() || slot >= engine->expert_states.size()) {
        return;
    }
    moe_block_state restored{};
    bool push_restored = false;
    {
        std::lock_guard<std::mutex> lock(engine->map_mutex);
        moe_block_state& state = engine->expert_states[slot];
        if (state.active) {
            subtract_tier_usage(engine, state.tier, state.bytes);
            prefetch_adjust_expert_count(engine, state.tier, -1);
        }
        if (had_before_state) {
            state = before_state;
            if (state.active) {
                add_tier_usage(engine, state.tier, state.bytes);
                prefetch_adjust_expert_count(engine, state.tier, 1);
                restored = state;
                push_restored = true;
            }
        } else {
            state = moe_block_state{};
        }
    }
    if (push_restored) {
        replace_evict_candidate(engine, slot, restored);
    }
}

static int moe_pc_engine_prefetch_expert_triplet_impl(
    moe_pc_engine_t* engine,
    int layer,
    int expert,
    moe_tier_t preferred_tier,
    int notify_workers,
    int stream_state_prechecked = 0,
    const vram_reservation_token& reservation = vram_reservation_token{},
    int allow_sync_fallback = 0,
    moe_prefetch_reason_t prefetch_reason = moe_PREFETCH_REASON_LOOKAHEAD,
    int urgent_required = 0
) {
    if (!engine || layer < 0 || expert < 0) {
        return 0;
    }
    const uint64_t bytes = expert_triplet_or_hardcoded_bytes(engine, layer, expert);
    if (bytes == 0) {
        return 0;
    }
    moe_block_state before_state{};
    bool had_before_state = false;
    {
        const size_t slot = expert_dense_slot(engine, layer, expert);
        std::lock_guard<std::mutex> lock(engine->map_mutex);
        if (slot < engine->expert_states.size() && engine->expert_states[slot].active) {
            before_state = engine->expert_states[slot];
            had_before_state = true;
        }
    }
    (void)notify_workers;
    const int required_now =
        prefetch_reason == moe_PREFETCH_REASON_ROUTER_SELECTED ||
        urgent_required != 0 ? 1 : 0;
    const int effective_allow_sync_fallback =
        required_now && engine->io_config.allow_required_sync_fallback ? 1 : 0;
    preferred_tier = juju_hint_target_tier(
        engine,
        layer,
        expert,
        preferred_tier,
        prefetch_reason);
    preferred_tier = moe_hot_miss_preferred_tier(engine, layer, expert, bytes, preferred_tier);
    vram_reservation_token effective_reservation = reservation;
    if (preferred_tier == moe_TIER_VRAM && !effective_reservation.valid) {
        if (!device_vram_try_reserve_token(
                engine,
                bytes,
                moe_DEVICE_POOL_EXPERT,
                &effective_reservation)) {
            preferred_tier = moe_TIER_RAM;
        }
    }
    const int requested = moe_pc_engine_request_expert(engine, layer, expert, bytes, preferred_tier);
    if (!requested) {
        if (effective_reservation.valid) {
            device_vram_release_reservation_token(engine, effective_reservation);
        }
        return 0;
    }
    if (had_before_state &&
        before_state.active &&
        before_state.tier != moe_TIER_STREAM &&
        before_state.bytes == bytes &&
        tier_speed_rank(before_state.tier) <= tier_speed_rank(preferred_tier)) {
        if (effective_reservation.valid) {
            device_vram_release_reservation_token(engine, effective_reservation);
        }
        return 1;
    }
    if (preferred_tier == moe_TIER_DB || preferred_tier == moe_TIER_STREAM) {
        if (effective_reservation.valid) {
            device_vram_release_reservation_token(engine, effective_reservation);
        }
        return requested;
    }

    const moe_projection_t projs[3] = {moe_PROJ_GATE, moe_PROJ_UP, moe_PROJ_DOWN};
    const moe_io_path_t active_io_path = choose_active_io_path(engine);
    const bool direct_bundle_io =
        active_io_path == moe_IO_PATH_GPUDIRECT_STORAGE ||
        active_io_path == moe_IO_PATH_DIRECTSTORAGE_GPU;
    const bool pinned_bundle_io =
        active_io_path == moe_IO_PATH_PINNED_HOST &&
        engine->io_config.enable_multistage_pipeline &&
        engine->staging &&
        engine->staging_slot_size >= bytes;
    if (preferred_tier == moe_TIER_VRAM &&
        engine->io_config.enable_compute_io_overlap &&
        engine_has_device_upload_workers(engine) &&
        (direct_bundle_io || pinned_bundle_io) &&
        ensure_io_ready(engine)) {
        const size_t gate_index = find_tensor_record_index(engine, layer, expert, moe_PROJ_GATE);
        const size_t up_index = find_tensor_record_index(engine, layer, expert, moe_PROJ_UP);
        const size_t down_index = find_tensor_record_index(engine, layer, expert, moe_PROJ_DOWN);
        if (gate_index != tensor_slot_invalid() && gate_index < engine->tensors.size() &&
            up_index != tensor_slot_invalid() && up_index < engine->tensors.size() &&
            down_index != tensor_slot_invalid() && down_index < engine->tensors.size()) {
            engine->io_atomic.active_path.store(
                static_cast<uint32_t>(active_io_path),
                std::memory_order_relaxed);
            moe_prefetch_task task{};
            task.tensor_index = gate_index;
            task.target_tier = preferred_tier;
            task.vram_token = effective_reservation;
            task.bundle_triplet = 1;
            task.bundle_layer = layer;
            task.bundle_expert = expert;
            task.urgent = 1;
            task.allow_sync_fallback = effective_allow_sync_fallback;
            task.prefetch_reason = prefetch_reason;
            task.required_now = required_now ? 1u : 0u;
            task.speculative = required_now ? 0u : 1u;
            task.urgent_required = required_now ? 1u : 0u;
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
    uint64_t token_bytes_assigned = 0;
    for (auto proj : projs) {
        vram_reservation_token sub_token{};
        if (effective_reservation.valid) {
            const moe_tensor_record* rec = find_tensor_record(engine, layer, expert, proj);
            const uint64_t rec_bytes = rec ? tensor_stream_bytes(*rec) : 0;
            if (rec_bytes > 0 && token_bytes_assigned < effective_reservation.bytes) {
                sub_token = effective_reservation;
                sub_token.bytes = std::min<uint64_t>(rec_bytes, effective_reservation.bytes - token_bytes_assigned);
                token_bytes_assigned += sub_token.bytes;
            }
        }
        queued += enqueue_tensor_prefetch(
            engine,
            layer,
            expert,
            proj,
            preferred_tier,
            stream_state_prechecked,
            sub_token,
            effective_allow_sync_fallback,
            prefetch_reason,
            required_now);
    }
    if (effective_reservation.valid && effective_reservation.bytes > token_bytes_assigned) {
        vram_reservation_token rest = effective_reservation;
        rest.bytes -= token_bytes_assigned;
        device_vram_release_reservation_token(engine, rest);
    }
    if (notify_workers) {
        for (int i = 0; i < queued; ++i) {
            engine->disk_stage.cv.notify_one();
        }
    }
    if (queued > 0) {
        return 1;
    }
    prefetch_rollback_unqueued_expert(engine, layer, expert, before_state, had_before_state);
    return 0;
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
        1,
        0,
        vram_reservation_token{},
        1,
        moe_PREFETCH_REASON_ROUTER_SELECTED,
        1
    );
}

static uint32_t startup_warm_stage_load(const moe_stage_queue& stage) {
    return stage.depth.load(std::memory_order_relaxed) +
           stage.inflight.load(std::memory_order_relaxed);
}

static uint64_t startup_warm_stage_byte_load(const moe_stage_queue& stage) {
    const uint64_t queued = stage.queued_bytes.load(std::memory_order_relaxed);
    const uint64_t inflight = stage.inflight_bytes.load(std::memory_order_relaxed);
    return queued > UINT64_MAX - inflight ? UINT64_MAX : queued + inflight;
}

static bool startup_warm_disk_has_byte_room(moe_pc_engine_t* engine, uint64_t bytes) {
    if (!engine || bytes == 0) {
        return true;
    }
    const uint64_t cap = moe_stage_byte_cap(engine, engine->disk_stage);
    if (cap == 0) {
        return true;
    }
    const uint64_t used = startup_warm_stage_byte_load(engine->disk_stage);
    return used <= cap && bytes <= cap - used;
}

static bool startup_warm_io_idle(moe_pc_engine_t* engine) {
    if (!engine) {
        return true;
    }
    const moe_io_path_t path = choose_active_io_path(engine);
    const bool device_copy_path =
        engine_has_device_upload_workers(engine) ||
        path == moe_IO_PATH_GPUDIRECT_STORAGE ||
        path == moe_IO_PATH_DIRECTSTORAGE_GPU;
    const bool gds_idle =
        !device_copy_path ||
        engine->gds_blocking_pending.load(std::memory_order_acquire) == 0;
    return engine->total_queue_depth.load(std::memory_order_acquire) == 0 &&
           startup_warm_stage_load(engine->disk_stage) == 0 &&
           startup_warm_stage_load(engine->pinned_stage) == 0 &&
           startup_warm_stage_load(engine->gpu_stage) == 0 &&
           gds_idle;
}

static size_t startup_warm_cpu_ram_default_cap(
    moe_pc_engine_t* engine,
    const std::vector<uint64_t>& candidate_bytes,
    uint64_t graph_cap
) {
    if (!engine || candidate_bytes.empty() || graph_cap == 0) {
        return 0;
    }
    uint64_t total_bytes = 0;
    for (uint64_t bytes : candidate_bytes) {
        if (bytes > UINT64_MAX - total_bytes) {
            total_bytes = UINT64_MAX;
            break;
        }
        total_bytes += bytes;
    }
    const uint64_t avg_bytes = std::max<uint64_t>(
        1ull,
        total_bytes / (uint64_t)candidate_bytes.size());
    const uint32_t hw = std::max<uint32_t>(1u, moe_cpu_thread_count_hint());
    uint64_t concurrency_cap = 512ull;
    if (hw > 8u) {
        concurrency_cap = 1280ull;
    } else if (hw > 4u) {
        concurrency_cap = 768ull;
    }

    uint64_t memory_cap = concurrency_cap;
    const uint64_t available = storagellm::available_ram_bytes();
    const uint64_t rss = storagellm::current_process_rss_bytes();
    const uint64_t common_raw = engine->common_raw_prefetched_bytes.load(std::memory_order_relaxed);
    const uint64_t reserve_floor =
        available <= 24ull * 1024ull * 1024ull * 1024ull
            ? 4ull * 1024ull * 1024ull * 1024ull
            : 2ull * 1024ull * 1024ull * 1024ull;
    const uint64_t reserve = std::max<uint64_t>(
        reserve_floor,
        std::max<uint64_t>(rss >> 1u, common_raw));
    if (available > reserve) {
        const uint64_t usable = (available - reserve) * 65ull / 100ull;
        memory_cap = std::max<uint64_t>(128ull, usable / avg_bytes);
    }

    const uint64_t capped = std::min<uint64_t>(
        graph_cap,
        std::min<uint64_t>(concurrency_cap, memory_cap));
    return (size_t)std::min<uint64_t>(capped, (uint64_t)SIZE_MAX);
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
        (tier_budget(engine, moe_TIER_VRAM) * 92ull) / 100ull : 0;
    const uint64_t ram_warm_pct = std::min<uint64_t>(
        engine->io_config.ram_eviction_target_fill_pct > 0
            ? engine->io_config.ram_eviction_target_fill_pct : 75ull,
        85ull);
    const uint64_t ram_limit = tier_budget(engine, moe_TIER_RAM) ?
        (tier_budget(engine, moe_TIER_RAM) * ram_warm_pct) / 100ull : 0;
    uint64_t planned_vram = tier_used(engine, moe_TIER_VRAM);
    uint64_t planned_ram = tier_used(engine, moe_TIER_RAM);
    uint32_t queued = 0;
    const uint32_t total_queue_cap = engine->io_config.max_prefetch_queue ?
        engine->io_config.max_prefetch_queue : 4096u;
    const uint32_t disk_queue_cap = engine->disk_stage_depth_cap > 0 ?
        engine->disk_stage_depth_cap : total_queue_cap;
    const uint32_t queue_headroom = std::max<uint32_t>(
        64u,
        std::max<uint32_t>(runtime_disk_workers(engine), 1u) * 16u);
    const uint32_t total_queue_high = total_queue_cap > queue_headroom ?
        total_queue_cap - queue_headroom : total_queue_cap;
    const uint32_t disk_queue_high = disk_queue_cap > queue_headroom ?
        disk_queue_cap - queue_headroom : disk_queue_cap;

    struct startup_expert_candidate_t {
        uint32_t layer = 0;
        uint32_t expert = 0;
        uint64_t bytes = 0;
        uint16_t priority = 0;
    };
    std::vector<startup_expert_candidate_t> candidates;
    std::vector<uint64_t> candidate_bytes;
    const uint64_t layer_count =
        c->last_moe_layer >= c->first_moe_layer ?
        (uint64_t)c->last_moe_layer - (uint64_t)c->first_moe_layer + 1ull : 0ull;
    if (layer_count > 0 && layer_count <= 10000ull &&
        c->experts_per_moe_layer <= 10000u &&
        layer_count <= UINT64_MAX / (uint64_t)c->experts_per_moe_layer) {
        candidates.reserve((size_t)(layer_count * (uint64_t)c->experts_per_moe_layer));
        candidate_bytes.reserve((size_t)(layer_count * (uint64_t)c->experts_per_moe_layer));
        for (uint32_t layer = c->first_moe_layer; layer <= c->last_moe_layer; ++layer) {
            if (!moe_layer_is_moe_runtime(engine, c, (int32_t)layer)) {
                continue;
            }
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
                candidate_bytes.push_back(bytes);
            }
        }
    }
    auto startup_warm_expert_spread_key = [](uint32_t expert) -> uint32_t {
        uint32_t x = expert * 2654435761u;
        x ^= x >> 16u;
        x *= 2246822519u;
        x ^= x >> 13u;
        return x;
    };
    std::stable_sort(candidates.begin(), candidates.end(),
        [&](const startup_expert_candidate_t& a, const startup_expert_candidate_t& b) {
            if (a.priority != b.priority) {
                return a.priority > b.priority;
            }
            const uint32_t a_key = startup_warm_expert_spread_key(a.expert);
            const uint32_t b_key = startup_warm_expert_spread_key(b.expert);
            if (a_key != b_key) {
                return a_key < b_key;
            }
            if (a.layer != b.layer) {
                return a.layer < b.layer;
            }
            return a.expert < b.expert;
        });

    size_t warm_candidate_limit = candidates.size();
    if (const char* env = std::getenv("STORAGELLM_STARTUP_WARM_MAX_EXPERTS")) {
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(env, &end, 10);
        if (end && end != env && parsed > 0ul) {
            warm_candidate_limit = std::min<size_t>(warm_candidate_limit, (size_t)parsed);
        }
    } else if (moe_storage_runtime_cpu_ram_path(engine) && layer_count > 0) {
        const uint64_t experts_per_layer =
            std::min<uint64_t>((uint64_t)c->experts_per_moe_layer, 64ull);
        const uint64_t graph_cap =
            layer_count > UINT64_MAX / experts_per_layer ?
            UINT64_MAX : layer_count * experts_per_layer;
        const size_t cpu_storage_cap =
            startup_warm_cpu_ram_default_cap(engine, candidate_bytes, graph_cap);
        if (cpu_storage_cap > 0) {
            warm_candidate_limit = std::min<size_t>(warm_candidate_limit, cpu_storage_cap);
        }
    }
    if (warm_candidate_limit < candidates.size()) {
        std::fprintf(stderr,
            "[storagellm] startup warm capped: candidates=%zu cap=%zu cpu_storage=%d ram_limit=%llu vram_limit=%llu\n",
            candidates.size(),
            warm_candidate_limit,
            moe_storage_runtime_cpu_ram_path(engine) ? 1 : 0,
            (unsigned long long)ram_limit,
            (unsigned long long)vram_limit);
    }
    size_t warm_candidate_index = 0;
    for (const startup_expert_candidate_t& cand : candidates) {
            if (warm_candidate_index++ >= warm_candidate_limit) {
                break;
            }
            const uint32_t layer = cand.layer;
            const uint32_t expert = cand.expert;
            const uint64_t bytes = cand.bytes;
            moe_tier_t target = moe_TIER_DB;
            if (vram_limit > 0 && planned_vram + bytes <= vram_limit) {
                target = moe_TIER_VRAM;
            } else if (ram_limit > 0 && planned_ram + bytes <= ram_limit) {
                target = moe_TIER_RAM;
            } else {
                // Stop once the active compute tiers are full. CPU-only roots have
                // no VRAM budget, so RAM warm is the primary way to avoid first-token
                // cold page-fault storms.
                break;
            }

            // BUGFIX 95: Cache atomic loads in throttle loop
            while (true) {
                const bool stop = engine->io_stop.load(std::memory_order_relaxed);
                const uint32_t total_depth = engine->total_queue_depth.load(std::memory_order_relaxed);
                const uint32_t disk_depth = startup_warm_stage_load(engine->disk_stage);
                const bool disk_byte_room = startup_warm_disk_has_byte_room(engine, bytes);
                if (stop || (total_depth < total_queue_high && disk_depth < disk_queue_high && disk_byte_room)) break;
                // BUGFIX 24: Use cv wait instead of polling ?�★
                // Old: 10ms sleep polling ??slow response, unnecessary wakeups
                // New: io_done_cv wait ??immediate response when queue drains
                std::unique_lock<std::mutex> lock(engine->done_mutex);
                engine->io_done_cv.wait_for(lock, std::chrono::milliseconds(100), [&] {
                    return engine->io_stop.load(std::memory_order_relaxed) ||
                           (engine->total_queue_depth.load(std::memory_order_relaxed) < total_queue_high &&
                            startup_warm_stage_load(engine->disk_stage) < disk_queue_high &&
                            startup_warm_disk_has_byte_room(engine, bytes));
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
                    1,
                    vram_reservation_token{},
                    0,
                    moe_PREFETCH_REASON_STARTUP_WARM,
                    0)) {
                if (target == moe_TIER_VRAM) {
                    planned_vram += bytes;
                } else if (target == moe_TIER_RAM) {
                    planned_ram += bytes;
                }
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
        uint32_t timeout_sec = 300u;
        if (const char* env = std::getenv("STORAGELLM_STARTUP_WARM_TIMEOUT_SEC")) {
            char* end = nullptr;
            const unsigned long parsed = std::strtoul(env, &end, 10);
            if (end && end != env && parsed > 0ul) {
                timeout_sec = (uint32_t)std::min<unsigned long>(parsed, 3600ul);
            }
        }

        int last_reported_second = -1;
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

            // Completion is a queue/stage property. queued counts experts, while
            // completed_requests counts tensor tasks and can use a different unit.
            if (startup_warm_io_idle(engine)) {
                fprintf(stderr, "[storagellm] startup warm: complete (processed=%u)\n", processed);
                break;
            }

            // Progress report every 2 seconds
            const auto elapsed = std::chrono::steady_clock::now() - start_time;
            const int elapsed_sec = (int)std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
            if ((elapsed_sec % 2) == 0 && elapsed_sec != last_reported_second) {
                last_reported_second = elapsed_sec;
                fprintf(stderr, "[storagellm] startup warm: queue=%u disk=%u pinned=%u gpu=%u completed=%u failed=%u ram=%llu vram=%llu\n",
                    depth,
                    startup_warm_stage_load(engine->disk_stage),
                    startup_warm_stage_load(engine->pinned_stage),
                    startup_warm_stage_load(engine->gpu_stage),
                    processed,
                    failed,
                    (unsigned long long)tier_used(engine, moe_TIER_RAM),
                    (unsigned long long)tier_used(engine, moe_TIER_VRAM));
            }

            if ((uint32_t)elapsed_sec > timeout_sec) {
                fprintf(stderr, "[storageLLM ERROR] startup warm timeout after %us (queue=%u disk=%u pinned=%u gpu=%u completed=%u failed=%u)\n",
                    timeout_sec,
                    depth,
                    startup_warm_stage_load(engine->disk_stage),
                    startup_warm_stage_load(engine->pinned_stage),
                    startup_warm_stage_load(engine->gpu_stage),
                    processed,
                    failed);
                return 0;
            }

            // BUGFIX 28: Use io_done_cv.wait_for instead of sleep polling ?�★??            // Old: sleep_for(100ms) ??up to 100ms delay after loading completes
            // New: io_done_cv.wait_for ??immediate wakeup when IO finishes
            // Reduces startup latency by up to 100ms per check iteration
            std::unique_lock<std::mutex> lock(engine->done_mutex);
            engine->io_done_cv.wait_for(lock, std::chrono::milliseconds(100), [&] {
                return engine->io_stop.load(std::memory_order_relaxed) ||
                       startup_warm_io_idle(engine);
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
    moe_tier_t preferred_tier,
    int allow_sync_fallback = 0,
    moe_prefetch_reason_t prefetch_reason = moe_PREFETCH_REASON_LOOKAHEAD,
    int urgent_required = 0
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
            1,
            vram_reservation_token{},
            allow_sync_fallback,
            prefetch_reason,
            urgent_required
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
            vram_expert_budget_bytes(engine) > 0 ? moe_TIER_VRAM : moe_TIER_RAM,
            1,
            moe_PREFETCH_REASON_ROUTER_SELECTED,
            1
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
                moe_TIER_VRAM,
                0,
                moe_PREFETCH_REASON_LOOKAHEAD,
                0
            ) ? 1 : 0;
        }
        if (!ram_candidates.empty()) {
            ok_count += prefetch_expert_batch_raw(
                engine,
                current_layer + 1,
                ram_candidates.data(),
                (uint32_t)ram_candidates.size(),
                moe_TIER_RAM,
                0,
                moe_PREFETCH_REASON_LOOKAHEAD,
                0
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
