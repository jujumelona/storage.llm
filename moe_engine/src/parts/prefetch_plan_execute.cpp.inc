int moe_pc_engine_execute_prefetch_plan(
    moe_pc_engine_t* engine,
    const moe_expert_prefetch_item_t* items,
    uint32_t item_count
) {
    if (!engine || !items || item_count == 0) {
        return 0;
    }
    // BUGFIX 93: Issue 17 - Batch map_mutex locks in prefetch_plan_execute ★★★
    // Problem: Lock map_mutex once per item in loop (item_count × lock overhead)
    // High contention when many items (typical: 50-100 items per prefetch plan)
    // Solution: Single lock for entire loop
    // Trade-off: Longer lock hold time, but eliminates lock/unlock overhead
    std::vector<uint8_t> stream_inflight(item_count, 0);
    {
        std::lock_guard<std::mutex> lock(engine->map_mutex);
        for (uint32_t i = 0; i < item_count; ++i) {
            const size_t slot = expert_dense_slot((int)items[i].layer, (int)items[i].expert);
            if (slot < engine->expert_states.size()) {
                const moe_block_state& state = engine->expert_states[slot];
                stream_inflight[i] = state.active && state.tier == moe_TIER_STREAM ? 1u : 0u;
            }
        }
    }

    // Problem 5 Fix: Split into two passes to reduce per-item mutex chaining.
    // Pass 1: Admission + eviction decisions (heap_mutex + map_mutex per item).
    // Pass 2: Enqueue all items (map_mutex once per item).
    // This prevents worst-case heap_mutex ??(map_mutex × n evictions) ??device_mutex ??map_mutex
    // from happening item_count times in a tight loop.

    std::vector<moe_tier_t> admitted_tiers(item_count, moe_TIER_DB);
    std::vector<std::vector<std::pair<int, int>>> all_to_free(item_count);

    // Pass 1: Admission and eviction decisions
    uint64_t batch_ram_reserved = 0;
    for (uint32_t i = 0; i < item_count; ++i) {
        if (stream_inflight[i]) {
            continue;
        }
        admitted_tiers[i] = prefetch_admitted_tier(engine, items[i], all_to_free[i]);
    }

    // BUGFIX 60: Issue 3 - TOCTOU deferred free pattern ?�★
    // Problem: Pass 1 collects to_free with map_mutex, releases lock,
    // then device_free_experts_batched. Between these, IO worker can start DMA
    // on same expert ??inflight_copies != 0 ??free fails or crashes.
    // Solution: Set pending_free flag in Pass 1 under map_mutex.
    // device_free_experts_batched checks pending_free, skips if inflight.
    // DMA completion handler (device_copy_complete) checks pending_free and
    // calls device_free_expert_if_pending_evict_locked to complete deferred free.
    // This pattern already exists in device_free_expert_if_pending_evict_locked.

    std::vector<std::pair<int, int>> flat_to_free;
    for (auto& to_free : all_to_free) {
        flat_to_free.insert(flat_to_free.end(), to_free.begin(), to_free.end());
    }

    // Set pending_free flags under map_mutex before actual free
    // This prevents IO workers from starting new DMA on these experts
    if (!flat_to_free.empty()) {
        std::lock_guard<std::mutex> lock(engine->map_mutex);
        for (const auto& victim : flat_to_free) {
            const size_t slot = expert_dense_slot(victim.first, victim.second);
            if (slot < engine->expert_states.size()) {
                moe_block_state& state = engine->expert_states[slot];
                if (state.active && state.tier == moe_TIER_VRAM) {
                    state.pending_evict = true;
                }
            }
        }
    }

    // Now safe to call device_free_experts_batched
    // It will check pending_evict and inflight_copies
    // If inflight, free is deferred to DMA completion handler
    if (!flat_to_free.empty()) {
        device_free_experts_batched(engine, flat_to_free, 4);
    }

    // Re-check RAM-admitted speculative items after freeing victims. Reserve
    // with the tier counter itself; a stale remaining-bytes snapshot can race
    // with other workers and over-admit VRAM work.
    for (uint32_t i = 0; i < item_count; ++i) {
        if (admitted_tiers[i] == moe_TIER_RAM
            && items[i].target_tier == moe_TIER_VRAM
            && !prefetch_is_router_selected(items[i])) {
            if (try_add_tier_usage(engine, moe_TIER_VRAM, items[i].bytes,
                                   tier_budget(engine, moe_TIER_VRAM))) {
                admitted_tiers[i] = moe_TIER_VRAM;
                // Prevent global RAM reservation leaks when upgrading RAM admission to VRAM.
                // prefetch_admitted_tier adds to global pending_ram_reservation.
                // If we change tier to VRAM, the cleanup loop at the end (which checks for RAM tier)
                // will skip this item, leaking the reservation permanently.
                prefetch_release_pending_ram_reservation(engine, items[i].bytes);
                if (batch_ram_reserved >= items[i].bytes) {
                    batch_ram_reserved -= items[i].bytes;
                }
            }
        }
    }

    // Pass 2: Enqueue all items
    int ok_count = 0;
    for (uint32_t i = 0; i < item_count; ++i) {
        if (stream_inflight[i]) {
            ++ok_count;
            continue;
        }
        const int queued = moe_pc_engine_prefetch_expert_triplet_impl(
            engine,
            (int)items[i].layer,
            (int)items[i].expert,
            admitted_tiers[i],
            0,
            1
        );
        if (queued) {
            ++ok_count;
        } else if (items[i].bytes > 0) {
            // Bug 10: Re-check STREAM state before fallback to avoid tier corruption
            // Between snapshot and here, another IO worker may have started streaming this expert
            bool is_streaming = false;
            {
                std::lock_guard<std::mutex> lock(engine->map_mutex);
                const size_t slot = expert_dense_slot((int)items[i].layer, (int)items[i].expert);
                if (slot < engine->expert_states.size()) {
                    const moe_block_state& state = engine->expert_states[slot];
                    is_streaming = state.active && state.tier == moe_TIER_STREAM;
                }
            }
            if (is_streaming) {
                ++ok_count;
            } else {
                ok_count += moe_pc_engine_request_expert(
                    engine,
                    (int)items[i].layer,
                    (int)items[i].expert,
                    items[i].bytes,
                    admitted_tiers[i]
                ) ? 1 : 0;
            }
        }
        if (admitted_tiers[i] == moe_TIER_RAM &&
            !prefetch_is_router_selected(items[i]) &&
            items[i].bytes > 0) {
            prefetch_release_pending_ram_reservation(engine, items[i].bytes);
        } else if (admitted_tiers[i] == moe_TIER_VRAM && items[i].bytes > 0) {
            prefetch_release_pending_vram_reservation(engine, items[i].bytes);
        }
    }
    if (ok_count > 0) {
        const uint32_t wake_count = std::min(static_cast<uint32_t>(ok_count), runtime_disk_workers(engine));
        for (uint32_t i = 0; i < wake_count; ++i) {
            engine->disk_stage.cv.notify_one();
        }
    }
    return ok_count > 0 ? 1 : 0;
}

int moe_pc_engine_set_topology_path(moe_pc_engine_t* engine, const char* path) {
    if (!engine) {
        return 0;
    }
    std::unique_lock<std::shared_mutex> lock(engine->topology_mutex);
    engine->topology_path = path ? path : "";
    return 1;
}



