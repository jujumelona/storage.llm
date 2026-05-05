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
                std::max(64u, pinned_workers * dyn_batch_max * 6u));
            if (snap_pinned_load >= pinned_threshold) {
                under_pressure = true;
            }
        }
        moe_expert_prefetch_item_t adjusted_item = items[i];
        if (under_pressure && !prefetch_is_router_selected(items[i])) {
            adjusted_item.target_tier = moe_TIER_RAM;
        }
        admitted_tiers[i] = prefetch_admitted_tier(engine, adjusted_item, all_to_free[i]);
        vram_pending_reserved[i] = (admitted_tiers[i] == moe_TIER_VRAM) ? 1u : 0u;
    }

    // BUGFIX 60: Issue 3 - TOCTOU deferred free pattern
    // BUGFIX Problem 11: Reserve flat_to_free capacity to avoid O(n²) reallocation ★
    // Problem: insert() without reserve causes log(n) reallocations for large plans
    // Solution: Calculate total size and reserve once
    std::vector<std::pair<int, int>> flat_to_free;
    size_t total_free_count = 0;
    for (const auto& v : all_to_free) total_free_count += v.size();
    flat_to_free.reserve(total_free_count);
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
            // BUGFIX 947: Consolidated reservation release with bytes==0 diagnostic ★★ DEFENSIVE
            // Problem: When items[i].bytes==0, all rollback paths are silently skipped.
            //   If a zero-byte item somehow obtained a reservation, the reservation leaks.
            // Solution: Add diagnostic warning for zero-byte items with reservations.
            // Additionally documents all 4 cases for future maintenance.
            if (items[i].bytes == 0) {
                // Zero-byte items should never have reservations; warn if they do
                if (vram_pending_reserved[i] || admitted_tiers[i] != moe_TIER_NONE) {
                    fprintf(stderr,
                        "[storageLLM WARNING] skip_item with bytes=0 but reservation active: "
                        "tier=%d vram_pending=%d target=%d layer=%u expert=%u\n",
                        (int)admitted_tiers[i], (int)vram_pending_reserved[i],
                        (int)items[i].target_tier, items[i].layer, items[i].expert);
                }
            } else if (admitted_tiers[i] == moe_TIER_VRAM && !vram_pending_reserved[i]) {
                // Case 1: RAM→VRAM upgrade succeeded (try_add_tier_usage committed)
                subtract_tier_usage(engine, moe_TIER_VRAM, items[i].bytes);
            } else if (vram_pending_reserved[i]) {
                // Case 2: Normal VRAM admission (pending reservation held)
                prefetch_release_pending_vram_reservation(engine, items[i].bytes);
            } else if (admitted_tiers[i] == moe_TIER_RAM && items[i].target_tier == moe_TIER_VRAM) {
                // Case 3: RAM→VRAM upgrade failed, stayed in RAM
                prefetch_release_pending_ram_reservation(engine, items[i].bytes);
            } else if (admitted_tiers[i] == moe_TIER_RAM) {
                // Case 4: Normal RAM admission
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
            // BUGFIX 930: Fix RAM→VRAM upgrade rollback on enqueue failure ★★★ CRITICAL
            // Problem: Pass 1 RAM→VRAM upgrade commits vram_used_bytes via try_add_tier_usage
            // → Pass 2 enqueue fails → No rollback → vram_used_bytes leaks
            // → Repeated failures accumulate → VRAM appears full → eviction storm → OOM
            // Solution: Add rollback case for RAM→VRAM upgrade path
            // Impact: Prevents VRAM budget leak on enqueue failure
            if (admitted_tiers[i] == moe_TIER_VRAM && !vram_pending_reserved[i]) {
                // RAM→VRAM upgrade succeeded in Pass 1, enqueue failed in Pass 2
                // try_add_tier_usage committed to vram_used_bytes, must rollback
                subtract_tier_usage(engine, moe_TIER_VRAM, items[i].bytes);
            } else if (vram_pending_reserved[i]) {
                // Normal VRAM admission path
                prefetch_release_pending_vram_reservation(engine, items[i].bytes);
            } else if (admitted_tiers[i] == moe_TIER_RAM) {
                // Normal RAM admission path
                prefetch_release_pending_ram_reservation(engine, items[i].bytes);
            }
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
