// ============================================================
// Mutex ordering convention:
//   device_mutex -> map_mutex for device allocation cleanup.
//   evict heap selection uses heap_mutex -> map_mutex and does not hold device_mutex.
//   request_expert uses map_mutex, releases it, then updates eviction heaps.
// ============================================================

static int pop_valid_evict_candidate(moe_pc_engine* engine, moe_tier_t tier, moe_evict_candidate* out) {
    if (!engine || !out) {
        return 0;
    }
    const int idx = tier_heap_index(tier);
    if (idx < 0) {
        return 0;
    }
    const uint32_t heap_size = (uint32_t)engine->evict_heaps[idx].size();
    // Bug 21 Fix: Reduce stale_limit from 128 to 64 to reduce map_mutex contention
    const uint32_t stale_limit = std::max<uint32_t>(64u, std::min<uint32_t>(256u, heap_size / 8u));
    uint32_t stale_count = 0;
    std::unique_lock<std::mutex> heap_lock(evict_heap_mutex(engine, idx));
    while (!engine->evict_heaps[idx].empty() && stale_count < stale_limit) {
        moe_evict_candidate victim = engine->evict_heaps[idx].top();
        engine->evict_heaps[idx].pop();
        engine->evict_heap_versions[idx].fetch_add(1, std::memory_order_release);
        if (victim.slot >= engine->expert_states.size()) {
            ++stale_count;
            // Bug 21 Fix: Reduce compaction threshold from 128 to 64
            if (stale_count >= 64u) {
                heap_lock.unlock();
                compact_evict_heap(engine, idx, true);
                heap_lock.lock();
                stale_count = 0;
            }
            continue;
        }
        {
            std::lock_guard<std::mutex> map_lock(engine->map_mutex);
            const moe_block_state& state = engine->expert_states[victim.slot];
            if (state.active && !state.pending_evict && state.tier == tier &&
                state.bytes == victim.bytes && state.hot_score == victim.hot_score &&
                state.last_touch_epoch == victim.last_touch_epoch) {
                *out = victim;
                return 1;
            }
        }
        ++stale_count;
        // Bug 21 Fix: Reduce compaction threshold from 128 to 64
        if (stale_count >= 64u) {
            heap_lock.unlock();
            compact_evict_heap(engine, idx, true);
            heap_lock.lock();
            stale_count = 0;
        }
    }
    return 0;
}

static int demote_one_victim_to(
    moe_pc_engine* engine,
    const moe_evict_candidate& victim,
    moe_tier_t from,
    moe_tier_t to,
    std::vector<std::pair<int, int>>& to_free
) {
    // BUGFIX 49: Issue 16 - Lock order verification ??    // Lock order: heap_mutex ??map_mutex (documented at file top)
    // This function is called from demote_cold_until, which is called from
    // prefetch_plan_execute Pass 1. Pass 1 collects to_free entries while
    // holding map_mutex, but device_free_experts_batched is called AFTER
    // Pass 1 completes and all locks are released.
    // Lock order is safe: map_mutex ??to_free.push_back (no device_mutex held)
    // device_free_experts_batched acquires device_mutex ??map_mutex AFTER Pass 1.
    // No deadlock risk because device_mutex is never held during Pass 1.
    moe_block_state evict_snap{};
    {
        std::lock_guard<std::mutex> map_lock(engine->map_mutex);
        if (victim.slot >= engine->expert_states.size()) {
            return 0;
        }
        moe_block_state& state = engine->expert_states[victim.slot];
        if (!state.active || state.pending_evict || state.tier != from ||
            state.bytes != victim.bytes || state.hot_score != victim.hot_score ||
            state.last_touch_epoch != victim.last_touch_epoch) {
            return 0;
        }
        if (to == moe_TIER_RAM &&
            tier_used(engine, moe_TIER_RAM) + state.bytes > tier_budget(engine, moe_TIER_RAM)) {
            return 0;
        }
        subtract_tier_usage(engine, from, state.bytes);
        if (from == moe_TIER_VRAM) {
            engine->vram_expert_count.fetch_sub(1);
            int layer = 0;
            int expert = 0;
            expert_dense_slot_decode(victim.slot, &layer, &expert);
            // Safe: to_free.push_back while holding map_lock
            // device_free_experts_batched called later without any locks held
            to_free.push_back({layer, expert});
            state.pending_evict = true;
        } else if (from == moe_TIER_RAM) {
            engine->ram_expert_count.fetch_sub(1);
        } else if (from == moe_TIER_DB) {
            engine->db_expert_count.fetch_sub(1);
        }
        state.tier = to;
        if (to == moe_TIER_VRAM) engine->vram_expert_count.fetch_add(1);
        else if (to == moe_TIER_RAM) engine->ram_expert_count.fetch_add(1);
        else if (to == moe_TIER_DB) engine->db_expert_count.fetch_add(1);
        add_tier_usage(engine, to, state.bytes);
        evict_snap = state;
    }
    replace_evict_candidate(engine, victim.slot, evict_snap);
    return 1;
}

static void demote_cold_until(moe_pc_engine* engine, moe_tier_t tier, uint64_t needed, std::vector<std::pair<int, int>>& to_free);

static int demote_one_vram_for_device_room(moe_pc_engine* engine, std::vector<std::pair<int, int>>& to_free) {
    moe_evict_candidate victim{};
    if (!pop_valid_evict_candidate(engine, moe_TIER_VRAM, &victim)) {
        return 0;
    }
    moe_tier_t next_tier = moe_TIER_RAM;
    const uint64_t ram_budget = tier_budget(engine, moe_TIER_RAM);
    // Use demote_cold_until instead of an O(N) one-by-one RAM eviction loop.
    if (ram_budget > 0 && tier_used(engine, moe_TIER_RAM) + victim.bytes > ram_budget) {
        std::vector<std::pair<int, int>> ignored;
        demote_cold_until(engine, moe_TIER_RAM, victim.bytes, ignored);
    }
    if (ram_budget == 0 || tier_used(engine, moe_TIER_RAM) + victim.bytes > ram_budget) {
        next_tier = moe_TIER_DB;
    }
    return demote_one_victim_to(engine, victim, moe_TIER_VRAM, next_tier, to_free);
}

static void demote_cold_until(moe_pc_engine* engine, moe_tier_t tier, uint64_t needed, std::vector<std::pair<int, int>>& to_free) {
    if (!engine) {
        return;
    }
    const uint64_t budget = tier_budget(engine, tier);
    const int idx = tier_heap_index(tier);
    if (idx < 0 || budget == 0 || (tier != moe_TIER_VRAM && tier != moe_TIER_RAM)) {
        return;
    }

    compact_evict_heap(engine, idx);

    while (tier_used(engine, tier) + needed > budget) {
        moe_evict_candidate victim{};
        if (!pop_valid_evict_candidate(engine, tier, &victim)) {
            return;
        }
        moe_tier_t next_tier = moe_TIER_DB;
        if (tier == moe_TIER_VRAM) {
            next_tier = moe_TIER_RAM;
            const uint64_t ram_budget = tier_budget(engine, moe_TIER_RAM);
            // Use a single cold-demotion pass instead of nested one-by-one RAM evictions.
            if (ram_budget > 0 && tier_used(engine, moe_TIER_RAM) + victim.bytes > ram_budget) {
                std::vector<std::pair<int, int>> ignored;
                demote_cold_until(engine, moe_TIER_RAM, victim.bytes, ignored);
            }
            if (ram_budget == 0 || tier_used(engine, moe_TIER_RAM) + victim.bytes > ram_budget) {
                next_tier = moe_TIER_DB;
            }
        }
        // If RAM demotion fails, fall back to DB so cold experts do not stay pinned.
        if (!demote_one_victim_to(engine, victim, tier, next_tier, to_free)) {
            if (next_tier == moe_TIER_RAM) {
                // RAM saturation prevented demotion; try DB directly.
                demote_one_victim_to(engine, victim, tier, moe_TIER_DB, to_free);
            }
            continue;
        }
    }
}
