int moe_pc_engine_set_kv_mode(moe_pc_engine_t* engine, moe_kv_mode_t mode) {
    if (!engine) {
        return 0;
    }
    if (mode != moe_KV_MODE_PLAIN && mode != moe_KV_MODE_QKV) {
        return 0;
    }
    if (engine->qkv_forced_by_format.load(std::memory_order_relaxed) && mode != moe_KV_MODE_QKV) {
        return 0;
    }
    engine->config.kv_mode = mode;
    qkv_set_mode(mode == moe_KV_MODE_QKV);
    return 1;
}

moe_kv_mode_t moe_pc_engine_get_kv_mode(const moe_pc_engine_t* engine) {
    return engine ? engine->config.kv_mode : moe_KV_MODE_PLAIN;
}

// Bug #20: Restructured to release map_mutex before demote_cold_until
// Phase 1: compute target under lock ??Phase 2: evict without lock ??Phase 3: finalize under lock
int moe_pc_engine_request_expert(
    moe_pc_engine_t* engine,
    int layer,
    int expert,
    uint64_t bytes,
    moe_tier_t preferred_tier
) {
    if (!engine || layer < 0 || expert < 0 || bytes == 0) {
        return 0;
    }
    const size_t slot = expert_dense_slot(engine, layer, expert);
    if (slot == tensor_slot_invalid() || slot >= engine->expert_states.size()) {
        return 0;
    }
    std::vector<std::pair<int, int>> to_free;
    moe_tier_t target = moe_TIER_DB;
    bool need_demote = false;
    uint64_t demote_needed_bytes = bytes;

    // Phase 1: compute target under map_mutex
    moe_tier_t saved_old_tier = moe_TIER_DB;
    uint64_t saved_old_bytes = 0;
    bool saved_was_active = false;

    {
        std::unique_lock<std::mutex> lock(engine->map_mutex);
        moe_block_state& state = engine->expert_states[slot];
        const double boost = preferred_tier == moe_TIER_VRAM ? 4.0 :
            preferred_tier == moe_TIER_RAM ? 1.0 : 0.25;

        saved_was_active = state.active;
        saved_old_tier = state.tier;
        saved_old_bytes = state.bytes;

        if (state.active) {
            if (state.pending_evict) {
                return 0;
            }
            // Bug 6: STREAM state expert should return immediately (already transferring)
            if (state.tier == moe_TIER_STREAM) {
                return 1;
            }
            if (tier_speed_rank(state.tier) <= tier_speed_rank(preferred_tier)) {
                if (state.bytes != bytes) {
                    subtract_tier_usage(engine, state.tier, state.bytes);
                    state.bytes = bytes;
                    add_tier_usage(engine, state.tier, bytes);
                }
                touch_expert_heat(engine, &state, boost);
                moe_block_state state_snap = state;
                lock.unlock();
                replace_evict_candidate(engine, slot, state_snap);
                return 1;  // early exit: already at good tier
            }
            subtract_tier_usage(engine, state.tier, state.bytes);
            // BUGFIX 609-611: Check expert counts > 0 before fetch_sub ★
            // Problem: Expert counts can underflow if fetch_sub called when count is 0
            // Solution: Check before decrement
            if (state.tier == moe_TIER_VRAM && engine->vram_expert_count.load(std::memory_order_relaxed) > 0) {
                engine->vram_expert_count.fetch_sub(1);
            } else if (state.tier == moe_TIER_RAM && engine->ram_expert_count.load(std::memory_order_relaxed) > 0) {
                engine->ram_expert_count.fetch_sub(1);
            } else if (state.tier == moe_TIER_DB && engine->db_expert_count.load(std::memory_order_relaxed) > 0) {
                engine->db_expert_count.fetch_sub(1);
            }
        }

        state.bytes = bytes;
        state.tier = moe_TIER_STREAM;
        state.active = true;
        state.pending_evict = false;
        touch_expert_heat(engine, &state, boost);

        target = preferred_tier;
        const uint64_t vram_expert_budget = vram_expert_budget_bytes(engine);
        if (target == moe_TIER_VRAM && (vram_expert_budget == 0 || bytes > vram_expert_budget)) {
            target = moe_TIER_RAM;
        }
        if (target == moe_TIER_RAM) {
            const uint64_t ram_budget = tier_budget(engine, moe_TIER_RAM);
            // Bug 1 Fix: RAM threshold matches prefetch_admitted_tier (93%). Critical Fix 4 raised
            // prefetch_admitted_tier from 87% →93%. This must match to prevent synchronous
            // demote_cold_until on the token forward path when RAM is 88~92% full.
            // BUGFIX 99: Precompute percentage threshold
            const uint64_t ram_limit = (ram_budget * 93ull) / 100ull;
            const bool too_big = ram_budget > 0 && bytes > ram_budget;
            const bool over_limit = ram_budget > 0 &&
                tier_used(engine, moe_TIER_RAM) + bytes > ram_limit;
            if (too_big || ram_budget == 0) {
                target = engine->config.allow_db_streaming ? moe_TIER_STREAM : moe_TIER_DB;
            } else if (over_limit) {
                const uint64_t headroom = ram_budget > ram_limit ? ram_budget - ram_limit : 0;
                demote_needed_bytes = bytes > UINT64_MAX - headroom ? UINT64_MAX : bytes + headroom;
            }
        }
        if (target == moe_TIER_STREAM) {
            state.tier = moe_TIER_DB;
            engine->db_expert_count.fetch_add(1);
            add_tier_usage(engine, moe_TIER_DB, state.bytes);
            moe_block_state state_snap = state;
            // Fix 2: Release map_mutex before heap_mutex acquisition
            // (replace_evict_candidate acquires heap_mutex)
            lock.unlock(); // explicitly release map_mutex
            replace_evict_candidate(engine, slot, state_snap);
            return 1;
        }
        need_demote = true;
    } // map_mutex released

    // Phase 2: evict WITHOUT holding map_mutex (may acquire heap_mutex)
    if (need_demote) {
        demote_cold_until(engine, target, demote_needed_bytes, to_free);
    }

    // Phase 3: finalize state under map_mutex
    // Fix 2: Separate map_mutex and heap_mutex to enforce lock ordering.
    moe_block_state evict_snap{};
    bool need_evict_update = false;
    std::vector<std::pair<size_t, moe_block_state>> rollback_evict_updates;
    rollback_evict_updates.reserve(to_free.size());
    const int result = [&]() {
        std::lock_guard<std::mutex> lock(engine->map_mutex);
        moe_block_state& state = engine->expert_states[slot];
        
        // BUGFIX: Verify state hasn't been changed by another thread during Phase 2
        // Phase 1 guarantees this expert was set to moe_TIER_STREAM.
        if (state.tier != moe_TIER_STREAM) {
            return 0; // State was altered (e.g. evicted) while map_mutex was released
        }

        moe_tier_t final_target = target;
        const uint64_t ram_budget = tier_budget(engine, moe_TIER_RAM);
        // Bug 1 Fix: RAM threshold matches prefetch_admitted_tier (93%). Critical Fix 4 raised
        // prefetch_admitted_tier from 87% →93%. This must match to prevent synchronous
        // demote_cold_until on the token forward path when RAM is 88~92% full.
        // BUGFIX 99: Precompute percentage threshold
        const uint64_t ram_threshold_93 = (ram_budget * 93ull) / 100ull;
        if (final_target == moe_TIER_RAM && ram_budget > 0 &&
            tier_used(engine, moe_TIER_RAM) + bytes > ram_threshold_93) {
            final_target = moe_TIER_DB;
        }
        if (try_add_tier_usage(engine, final_target, bytes, tier_budget(engine, final_target))) {
            state.tier = final_target;
            state.pending_evict = false;
            if (final_target == moe_TIER_VRAM) engine->vram_expert_count.fetch_add(1);
            else if (final_target == moe_TIER_RAM) engine->ram_expert_count.fetch_add(1);
            else if (final_target == moe_TIER_DB) engine->db_expert_count.fetch_add(1);
            evict_snap = state; need_evict_update = true;
        } else if (engine->config.allow_db_streaming) {
            state.tier = moe_TIER_DB;
            state.pending_evict = false;
            engine->db_expert_count.fetch_add(1);
            add_tier_usage(engine, moe_TIER_DB, state.bytes);
            evict_snap = state; need_evict_update = true;
        } else {
            // Bug 7: Rollback demoted experts when Phase 3 fails
            // to_free experts were demoted to RAM but new expert can't be added
            // Restore their tier to prevent orphaned state
            if (saved_was_active) {
                add_tier_usage(engine, saved_old_tier, saved_old_bytes);
                if (saved_old_tier == moe_TIER_VRAM) engine->vram_expert_count.fetch_add(1);
                else if (saved_old_tier == moe_TIER_RAM) engine->ram_expert_count.fetch_add(1);
                else if (saved_old_tier == moe_TIER_DB) engine->db_expert_count.fetch_add(1);
                state.tier = saved_old_tier;
                state.bytes = saved_old_bytes;
                state.active = true;
                state.pending_evict = false;
                evict_snap = state; need_evict_update = true;
            } else {
                state = moe_block_state{};
            }
            // Bug 7: Rollback demoted experts in to_free list
            for (auto& pair : to_free) {
                const size_t victim_slot = expert_dense_slot(engine, pair.first, pair.second);
                if (victim_slot < engine->expert_states.size()) {
                    moe_block_state& victim_state = engine->expert_states[victim_slot];
                    // Restore to VRAM (they were demoted from VRAM to RAM/DB in Phase 2)
                    if (victim_state.active) {
                        if (victim_state.tier == moe_TIER_RAM) {
                            subtract_tier_usage(engine, moe_TIER_RAM, victim_state.bytes);
                            victim_state.tier = moe_TIER_VRAM;
                            victim_state.pending_evict = false;
                            add_tier_usage(engine, moe_TIER_VRAM, victim_state.bytes);
                            // BUGFIX 612: Check ram_expert_count > 0 before fetch_sub ★
                            // Problem: ram_expert_count can underflow
                            // Solution: Check before decrement
                            if (engine->ram_expert_count.load(std::memory_order_relaxed) > 0) {
                                engine->ram_expert_count.fetch_sub(1);
                            }
                            engine->vram_expert_count.fetch_add(1);
                            rollback_evict_updates.push_back({victim_slot, victim_state});
                        } else if (victim_state.tier == moe_TIER_DB) {
                            subtract_tier_usage(engine, moe_TIER_DB, victim_state.bytes);
                            victim_state.tier = moe_TIER_VRAM;
                            victim_state.pending_evict = false;
                            add_tier_usage(engine, moe_TIER_VRAM, victim_state.bytes);
                            // BUGFIX 613: Check db_expert_count > 0 before fetch_sub ★
                            // Problem: db_expert_count can underflow
                            // Solution: Check before decrement
                            if (engine->db_expert_count.load(std::memory_order_relaxed) > 0) {
                                engine->db_expert_count.fetch_sub(1);
                            }
                            engine->vram_expert_count.fetch_add(1);
                            rollback_evict_updates.push_back({victim_slot, victim_state});
                        } else if (victim_state.tier == moe_TIER_VRAM) {
                            victim_state.pending_evict = false;
                            rollback_evict_updates.push_back({victim_slot, victim_state});
                        }
                    }
                }
            }
            to_free.clear();  // Don't free them since we rolled back
            return 0;
        }
        return 1;
    }();
    // heap_mutex acquired AFTER map_mutex released - correct ordering
    if (need_evict_update) {
        replace_evict_candidate(engine, slot, evict_snap);
    }
    for (const auto& item : rollback_evict_updates) {
        replace_evict_candidate(engine, item.first, item.second);
    }
    for (auto& pair : to_free) {
        device_free_expert(engine, pair.first, pair.second);
    }
    return result;
}
