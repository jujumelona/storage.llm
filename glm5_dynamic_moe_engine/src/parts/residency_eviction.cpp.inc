// ============================================================
// Mutex ordering convention:
//   device_mutex -> map_mutex for device allocation cleanup.
//   evict heap selection uses heap_mutex -> map_mutex and does not hold device_mutex.
//   request_expert uses map_mutex, releases it, then updates eviction heaps.
// ============================================================

static int pop_valid_evict_candidate(glm5_pc_engine* engine, glm5_tier_t tier, glm5_evict_candidate* out) {
    if (!engine || !out) {
        return 0;
    }
    const int idx = tier_heap_index(tier);
    if (idx < 0) {
        return 0;
    }
    const uint32_t stale_limit = 256u;
    uint32_t stale_count = 0;
    std::unique_lock<std::mutex> heap_lock(evict_heap_mutex(engine, idx));
    while (!engine->evict_heaps[idx].empty() && stale_count < stale_limit) {
        glm5_evict_candidate victim = engine->evict_heaps[idx].top();
        engine->evict_heaps[idx].pop();
        engine->evict_heap_versions[idx].fetch_add(1, std::memory_order_release);
        if (victim.slot >= engine->expert_states.size()) {
            ++stale_count;
            // Bottleneck 1: Increase compaction threshold to reduce heap_mutex hold time
            // 64 → 128 reduces compaction frequency by 2x, trading heap memory for latency
            if (stale_count >= 128u) {
                heap_lock.unlock();
                compact_evict_heap(engine, idx, true);
                heap_lock.lock();
                stale_count = 0;
            }
            continue;
        }
        {
            std::lock_guard<std::mutex> map_lock(engine->map_mutex);
            const glm5_block_state& state = engine->expert_states[victim.slot];
            if (state.active && !state.pending_evict && state.tier == tier &&
                state.bytes == victim.bytes && state.hot_score == victim.hot_score &&
                state.last_touch_epoch == victim.last_touch_epoch) {
                *out = victim;
                return 1;
            }
        }
        ++stale_count;
        // Bottleneck 1: Increase compaction threshold (64 → 128)
        if (stale_count >= 128u) {
            heap_lock.unlock();
            compact_evict_heap(engine, idx, true);
            heap_lock.lock();
            stale_count = 0;
        }
    }
    return 0;
}

static int demote_one_victim_to(
    glm5_pc_engine* engine,
    const glm5_evict_candidate& victim,
    glm5_tier_t from,
    glm5_tier_t to,
    std::vector<std::pair<int, int>>& to_free
) {
    glm5_block_state evict_snap{};
    {
        std::lock_guard<std::mutex> map_lock(engine->map_mutex);
        if (victim.slot >= engine->expert_states.size()) {
            return 0;
        }
        glm5_block_state& state = engine->expert_states[victim.slot];
        if (!state.active || state.pending_evict || state.tier != from ||
            state.bytes != victim.bytes || state.hot_score != victim.hot_score ||
            state.last_touch_epoch != victim.last_touch_epoch) {
            return 0;
        }
        if (to == GLM5_TIER_RAM &&
            tier_used(engine, GLM5_TIER_RAM) + state.bytes > tier_budget(engine, GLM5_TIER_RAM)) {
            return 0;
        }
        subtract_tier_usage(engine, from, state.bytes);
        if (from == GLM5_TIER_VRAM) {
            engine->vram_expert_count.fetch_sub(1);
            int layer = 0;
            int expert = 0;
            expert_dense_slot_decode(victim.slot, &layer, &expert);
            to_free.push_back({layer, expert});
            state.pending_evict = true;
        } else if (from == GLM5_TIER_RAM) {
            engine->ram_expert_count.fetch_sub(1);
        } else if (from == GLM5_TIER_DB) {
            engine->db_expert_count.fetch_sub(1);
        }
        state.tier = to;
        if (to == GLM5_TIER_VRAM) engine->vram_expert_count.fetch_add(1);
        else if (to == GLM5_TIER_RAM) engine->ram_expert_count.fetch_add(1);
        else if (to == GLM5_TIER_DB) engine->db_expert_count.fetch_add(1);
        add_tier_usage(engine, to, state.bytes);
        evict_snap = state;
    }
    replace_evict_candidate(engine, victim.slot, evict_snap);
    return 1;
}

static int demote_one_ram_to_db(glm5_pc_engine* engine) {
    glm5_evict_candidate victim{};
    if (!pop_valid_evict_candidate(engine, GLM5_TIER_RAM, &victim)) {
        return 0;
    }
    std::vector<std::pair<int, int>> ignored;
    return demote_one_victim_to(engine, victim, GLM5_TIER_RAM, GLM5_TIER_DB, ignored);
}

static int demote_one_vram_for_device_room(glm5_pc_engine* engine, std::vector<std::pair<int, int>>& to_free) {
    glm5_evict_candidate victim{};
    if (!pop_valid_evict_candidate(engine, GLM5_TIER_VRAM, &victim)) {
        return 0;
    }
    glm5_tier_t next_tier = GLM5_TIER_RAM;
    const uint64_t ram_budget = tier_budget(engine, GLM5_TIER_RAM);
    while (ram_budget > 0 && tier_used(engine, GLM5_TIER_RAM) + victim.bytes > ram_budget) {
        if (!demote_one_ram_to_db(engine)) {
            break;
        }
    }
    if (ram_budget == 0 || tier_used(engine, GLM5_TIER_RAM) + victim.bytes > ram_budget) {
        next_tier = GLM5_TIER_DB;
    }
    return demote_one_victim_to(engine, victim, GLM5_TIER_VRAM, next_tier, to_free);
}

static void demote_cold_until(glm5_pc_engine* engine, glm5_tier_t tier, uint64_t needed, std::vector<std::pair<int, int>>& to_free) {
    if (!engine) {
        return;
    }
    const uint64_t budget = tier_budget(engine, tier);
    const int idx = tier_heap_index(tier);
    if (idx < 0 || budget == 0 || (tier != GLM5_TIER_VRAM && tier != GLM5_TIER_RAM)) {
        return;
    }

    compact_evict_heap(engine, idx);

    while (tier_used(engine, tier) + needed > budget) {
        glm5_evict_candidate victim{};
        if (!pop_valid_evict_candidate(engine, tier, &victim)) {
            return;
        }
        glm5_tier_t next_tier = GLM5_TIER_DB;
        if (tier == GLM5_TIER_VRAM) {
            next_tier = GLM5_TIER_RAM;
            const uint64_t ram_budget = tier_budget(engine, GLM5_TIER_RAM);
            while (ram_budget > 0 && tier_used(engine, GLM5_TIER_RAM) + victim.bytes > ram_budget) {
                if (!demote_one_ram_to_db(engine)) {
                    break;
                }
            }
            if (ram_budget == 0 || tier_used(engine, GLM5_TIER_RAM) + victim.bytes > ram_budget) {
                next_tier = GLM5_TIER_DB;
            }
        }
        if (!demote_one_victim_to(engine, victim, tier, next_tier, to_free)) {
            continue;
        }
    }
}
