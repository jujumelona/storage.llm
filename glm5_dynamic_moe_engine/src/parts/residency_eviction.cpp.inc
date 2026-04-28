// ============================================================
// Mutex Ordering Convention (Bug #6):
//   device_mutex → heap_mutex → map_mutex
// All code paths MUST acquire in this order to prevent deadlocks.
// demote_cold_until: heap_mutex → map_mutex (OK, no device_mutex)
// device_make_room_locked: device_mutex → heap_mutex → map_mutex (OK)
// request_expert: map_mutex → (release) → heap_mutex (OK, 3-phase)
// ============================================================
static void demote_cold_until(glm5_pc_engine* engine, glm5_tier_t tier, uint64_t needed, std::vector<std::pair<int, int>>& to_free) {
    if (!engine) {
        return;
    }
    const uint64_t budget = tier_budget(engine, tier);
    const int idx = tier_heap_index(tier);
    if (idx < 0 || budget == 0 || (tier != GLM5_TIER_VRAM && tier != GLM5_TIER_RAM)) {
        return;
    }

    // Fix 53: Use shared compaction function (replaces inline Fix 48)
    {
        std::lock_guard<std::mutex> heap_lock(engine->heap_mutex);
        compact_evict_heap_locked(engine, idx);
    }

    // Fix 48: tighter stale_limit (consistent with device_make_room_locked)
    const uint32_t stale_limit = engine->expert_states.size() > 0
        ? (uint32_t)std::min<size_t>(engine->expert_states.size(), expert_dense_slot_width()) : 128u;
    uint32_t stale_count = 0;
    while (tier_used(engine, tier) + needed > budget) {
        glm5_evict_candidate victim{};
        bool victim_valid = false;
        {
            std::lock_guard<std::mutex> heap_lock(engine->heap_mutex);
            while (!engine->evict_heaps[idx].empty() && stale_count < stale_limit) {
                victim = engine->evict_heaps[idx].top();
                engine->evict_heaps[idx].pop();
                if (victim.slot >= engine->expert_states.size()) {
                    ++stale_count;
                    if (stale_count >= 64u) {
                        compact_evict_heap_locked(engine, idx, true);
                        stale_count = 0;
                    }
                    continue;
                }
                glm5_block_state& state = engine->expert_states[victim.slot];
                if (!state.active || state.tier != tier || state.bytes != victim.bytes ||
                    state.hot_score != victim.hot_score || state.last_touch_epoch != victim.last_touch_epoch) {
                    ++stale_count;
                    if (stale_count >= 64u) {
                        compact_evict_heap_locked(engine, idx, true);
                        stale_count = 0;
                    }
                    continue;
                }
                victim_valid = true;  // Bug 9: Removed duplicate line
                break;
            }
        }
        if (!victim_valid) return;

        stale_count = 0;
        // Fix 62 + Bug #8: Cascade eviction without holding map_mutex
        // TOCTOU fix: read victim_bytes atomically under map_mutex,
        // then re-validate after re-acquire to catch concurrent mutations.
        glm5_tier_t next_tier = GLM5_TIER_DB;
        uint64_t victim_bytes = 0;
        {
            std::lock_guard<std::mutex> map_lock(engine->map_mutex);
            glm5_block_state& state = engine->expert_states[victim.slot];
            // Re-validate victim is still in expected tier
            if (!state.active || state.tier != tier) {
                continue;  // Victim was mutated by another thread — skip
            }
            victim_bytes = state.bytes;
        }
        // map_mutex released — safe to recurse for RAM cascade
        if (tier == GLM5_TIER_VRAM) {
            next_tier = GLM5_TIER_RAM;
            demote_cold_until(engine, GLM5_TIER_RAM, victim_bytes, to_free);
            if (tier_used(engine, GLM5_TIER_RAM) + victim_bytes > tier_budget(engine, GLM5_TIER_RAM)) {
                next_tier = GLM5_TIER_DB;
            }
        }
        // Re-acquire map_mutex for state mutation
        glm5_block_state evict_snap_for_requeue{};
        bool need_requeue = false;
        {
            std::lock_guard<std::mutex> map_lock(engine->map_mutex);
            glm5_block_state& state = engine->expert_states[victim.slot];
            // Re-validate AGAIN after recursive call — another thread may have touched this slot
            if (!state.active || state.tier != tier) {
                continue;  // Stale — skip this victim entirely
            }
            subtract_tier_usage(engine, tier, state.bytes);
            if (tier == GLM5_TIER_VRAM) {
                engine->vram_expert_count.fetch_sub(1);
                int layer = 0;
                int expert = 0;
                expert_dense_slot_decode(victim.slot, &layer, &expert);
                to_free.push_back({layer, expert});
            } else if (tier == GLM5_TIER_RAM) {
                engine->ram_expert_count.fetch_sub(1);
            } else if (tier == GLM5_TIER_DB) {
                engine->db_expert_count.fetch_sub(1);
            }
            state.tier = next_tier;
            if (next_tier == GLM5_TIER_VRAM) engine->vram_expert_count.fetch_add(1);
            else if (next_tier == GLM5_TIER_RAM) engine->ram_expert_count.fetch_add(1);
            else if (next_tier == GLM5_TIER_DB) engine->db_expert_count.fetch_add(1);
            add_tier_usage(engine, next_tier, state.bytes);
            evict_snap_for_requeue = state;
            need_requeue = true;
        }
        // Fix 2: heap_mutex acquired AFTER map_mutex released
        if (need_requeue) {
            replace_evict_candidate(engine, victim.slot, evict_snap_for_requeue);
        }
    }
}
