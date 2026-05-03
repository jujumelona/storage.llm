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
            const size_t slot = expert_dense_slot(engine, (int)items[i].layer, (int)items[i].expert);
            if (slot < engine->expert_states.size()) {
                const moe_block_state& state = engine->expert_states[slot];
                stream_inflight[i] = state.active && state.tier == moe_TIER_STREAM ? 1u : 0u;
            }
        }
    }

    // Problem 5 Fix: Split into two passes to reduce per-item mutex chaining.
    // Pass 1: Admission + eviction decisions (heap_mutex + map_mutex per item).
    // Pass 2: Enqueue all items (map_mutex once per item).
    // This prevents worst-case heap_mutex → map_mutex × n evictions → device_mutex → map_mutex
    // from happening item_count times in a tight loop.

    std::vector<moe_tier_t> admitted_tiers(item_count, moe_TIER_DB);
    std::vector<uint8_t> vram_pending_reserved(item_count, 0u);
    std::vector<std::vector<std::pair<int, int>>> all_to_free(item_count);

    // Fix 37: Snapshot stage loads once before admission loop
    // Eliminates 6 atomic reads per item (hundreds of items → thousands of reads)
    // Stale snapshots are acceptable: queue pressure changes on 10s of ms timescale
    const uint32_t snap_disk_load = prefetch_stage_load(engine->disk_stage);
    const uint32_t snap_pinned_load = prefetch_stage_load(engine->pinned_stage);
    const uint32_t snap_gpu_load = prefetch_stage_load(engine->gpu_stage);

    for (uint32_t i = 0; i < item_count; ++i) {
        if (stream_inflight[i]) {
            continue;
        }
        // Use snapshot-based pressure check instead of per-item atomic reads
        bool under_pressure = false;
        if (engine->disk_stage_depth_cap > 0 &&
            snap_disk_load * 4u >= engine->disk_stage_depth_cap * 3u) {
            under_pressure = true;
        } else if (items[i].target_tier == moe_TIER_VRAM &&
            ((engine->pinned_stage_depth_cap > 0 &&
              snap_pinned_load * 4u >= engine->pinned_stage_depth_cap * 3u) ||
             (engine->gpu_stage_depth_cap > 0 &&
              snap_gpu_load * 4u >= engine->gpu_stage_depth_cap * 3u))) {
            under_pressure = true;
        } else if (items[i].target_tier == moe_TIER_RAM) {
            const uint32_t pinned_workers = runtime_pinned_workers(engine);
            const uint32_t dyn_batch_max = moe_gpu_stage_batch_max(engine);
            const uint32_t pinned_threshold = std::min(
                engine->io_config.max_prefetch_queue / 4u,
                std::max(64u, pinned_workers * dyn_batch_max * 12u));
            if (snap_pinned_load >= pinned_threshold) {
                under_pressure = true;
            }
        }
        moe_expert_prefetch_item_t adjusted_item = items[i];
        if (under_pressure && !prefetch_is_router_selected(items[i])) {
            adjusted_item.target_tier = moe_TIER_RAM;
        }
        admitted_tiers[i] = prefetch_admitted_tier(engine, adjusted_item, all_to_free[i]);
        vram_pending_reserved[i] =
            (admitted_tiers[i] == moe_TIER_VRAM && items[i].target_tier == moe_TIER_VRAM) ? 1u : 0u;
    }

    // BUGFIX 60: Issue 3 - TOCTOU deferred free pattern
    std::vector<std::pair<int, int>> flat_to_free;
    for (auto& to_free : all_to_free) {
        flat_to_free.insert(flat_to_free.end(), to_free.begin(), to_free.end());
    }

    // Set pending_free flags under map_mutex before actual free
    if (!flat_to_free.empty()) {
        std::lock_guard<std::mutex> lock(engine->map_mutex);
        for (const auto& victim : flat_to_free) {
            const size_t slot = expert_dense_slot(engine, victim.first, victim.second);
            if (slot < engine->expert_states.size()) {
                moe_block_state& state = engine->expert_states[slot];
                if (state.active && state.tier == moe_TIER_VRAM) {
                    state.pending_evict = true;
                }
            }
        }
    }

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
                // FIX 8: Release the RAM reservation since we are upgrading to VRAM.
                // Do NOT add to pending_vram_reservation: try_add_tier_usage already
                // committed directly to vram_used_bytes. Adding to pending here would
                // cause a transient double-count between now and the cleanup loop,
                // incorrectly blocking other router-selected experts from VRAM admission.
                prefetch_release_pending_ram_reservation(engine, items[i].bytes);
            }
        }
    }

    // Pass 2: Enqueue all items
    // BUGFIX 127: Issue 32 - Re-check expert state between Pass 1 and Pass 2 ★★★
    // Problem: Pass 1 releases map_mutex after admission decisions.
    // Between Pass 1 and Pass 2, another worker can move same expert to different tier.
    // Pass 1 decides "upload to VRAM" → Pass 2 finds it already in VRAM → duplicate upload.
    // Solution: Re-acquire map_mutex at Pass 2 start, re-check tier and pending_evict.
    // Skip if already in target tier, wait if pending_evict.
    int ok_count = 0;
    for (uint32_t i = 0; i < item_count; ++i) {
        if (stream_inflight[i]) {
            ++ok_count;
            continue;
        }

        // BUGFIX 127: Re-check state before enqueue
        bool skip_item = false;
        {
            std::lock_guard<std::mutex> lock(engine->map_mutex);
            const size_t slot = expert_dense_slot(engine, (int)items[i].layer, (int)items[i].expert);
            if (slot < engine->expert_states.size()) {
                const moe_block_state& state = engine->expert_states[slot];
                // Already in target tier → skip
                if (state.active && state.tier == admitted_tiers[i]) {
                    skip_item = true;
                    ++ok_count;
                }
                // Pending eviction → skip (will be retried by eviction completion handler)
                else if (state.pending_evict) {
                    skip_item = true;
                }
            }
        }
        if (skip_item) {
            // BUGFIX 716: Release pending VRAM reservation correctly on skip_item ★★★
            // Problem: If skip_item=true, the reservation wasn't released because of an extra
            // condition `&& items[i].target_tier == moe_TIER_VRAM`. If admitted_tiers[i]
            // is VRAM, VRAM reservation was taken, regardless of the original target.
            // Solution: Release based solely on admitted_tiers.
            if (vram_pending_reserved[i] && items[i].bytes > 0) {
                prefetch_release_pending_vram_reservation(engine, items[i].bytes);
            } else if (admitted_tiers[i] == moe_TIER_RAM && items[i].bytes > 0) {
                prefetch_release_pending_ram_reservation(engine, items[i].bytes);
            }
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
                const size_t slot = expert_dense_slot(engine, (int)items[i].layer, (int)items[i].expert);
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
        // BUGFIX 275: Fix budget release logic for tier transitions ★★
        // Release only reservations actually taken by prefetch_admitted_tier.
        // RAM->VRAM upgrades commit directly to tier usage, so they must not
        // subtract pending_vram_reservation that belongs to another item.
        if (admitted_tiers[i] == moe_TIER_RAM && items[i].bytes > 0) {
            // Item stayed in RAM tier → release RAM reservation
            prefetch_release_pending_ram_reservation(engine, items[i].bytes);
        } else if (vram_pending_reserved[i] && items[i].bytes > 0) {
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



