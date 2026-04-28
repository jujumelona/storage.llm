int glm5_pc_engine_set_kv_mode(glm5_pc_engine_t* engine, glm5_kv_mode_t mode) {
    if (!engine) {
        return 0;
    }
    if (mode != GLM5_KV_MODE_PLAIN && mode != GLM5_KV_MODE_QKV) {
        return 0;
    }
    engine->config.kv_mode = mode;
    qkv_set_mode(mode == GLM5_KV_MODE_QKV);
    return 1;
}

glm5_kv_mode_t glm5_pc_engine_get_kv_mode(const glm5_pc_engine_t* engine) {
    return engine ? engine->config.kv_mode : GLM5_KV_MODE_PLAIN;
}

// Bug #20: Restructured to release map_mutex before demote_cold_until
// Phase 1: compute target under lock → Phase 2: evict without lock → Phase 3: finalize under lock
int glm5_pc_engine_request_expert(
    glm5_pc_engine_t* engine,
    int layer,
    int expert,
    uint64_t bytes,
    glm5_tier_t preferred_tier
) {
    if (!engine || layer < 0 || expert < 0 || bytes == 0) {
        return 0;
    }
    const size_t slot = expert_dense_slot(layer, expert);
    if (slot == tensor_slot_invalid() || slot >= engine->expert_states.size()) {
        return 0;
    }
    std::vector<std::pair<int, int>> to_free;
    glm5_tier_t target = GLM5_TIER_DB;
    bool need_demote = false;

    // Phase 1: compute target under map_mutex
    {
        std::unique_lock<std::mutex> lock(engine->map_mutex);
        glm5_block_state& state = engine->expert_states[slot];
        const double boost = preferred_tier == GLM5_TIER_VRAM ? 4.0 :
            preferred_tier == GLM5_TIER_RAM ? 1.0 : 0.25;

        if (state.active) {
            if (tier_speed_rank(state.tier) <= tier_speed_rank(preferred_tier)) {
                if (state.bytes != bytes) {
                    subtract_tier_usage(engine, state.tier, state.bytes);
                    state.bytes = bytes;
                    add_tier_usage(engine, state.tier, bytes);
                }
                touch_expert_heat(engine, &state, boost);
                push_evict_candidate(engine, slot, state);
                return 1;  // early exit — already at good tier
            }
            subtract_tier_usage(engine, state.tier, state.bytes);
            if (state.tier == GLM5_TIER_VRAM) engine->vram_expert_count.fetch_sub(1);
            else if (state.tier == GLM5_TIER_RAM) engine->ram_expert_count.fetch_sub(1);
            else if (state.tier == GLM5_TIER_DB) engine->db_expert_count.fetch_sub(1);
        }

        state.bytes = bytes;
        state.tier = GLM5_TIER_STREAM;
        state.active = true;
        touch_expert_heat(engine, &state, boost);

        target = preferred_tier;
        const uint64_t vram_expert_budget = vram_expert_budget_bytes(engine);
        if (target == GLM5_TIER_VRAM && (vram_expert_budget == 0 || bytes > vram_expert_budget)) {
            target = GLM5_TIER_RAM;
        }
        if (target == GLM5_TIER_RAM && bytes > engine->config.ram_budget_bytes) {
            target = engine->config.allow_db_streaming ? GLM5_TIER_STREAM : GLM5_TIER_DB;
        }
        if (target == GLM5_TIER_STREAM) {
            state.tier = GLM5_TIER_DB;
            engine->db_expert_count.fetch_add(1);
            add_tier_usage(engine, GLM5_TIER_DB, state.bytes);
            glm5_block_state state_snap = state;
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
        demote_cold_until(engine, target, bytes, to_free);
    }

    // Phase 3: finalize state under map_mutex
    // Fix 2: Separate map_mutex and heap_mutex to enforce lock ordering.
    glm5_block_state evict_snap{};
    bool need_evict_update = false;
    const int result = [&]() {
        std::lock_guard<std::mutex> lock(engine->map_mutex);
        glm5_block_state& state = engine->expert_states[slot];
        if (try_add_tier_usage(engine, target, bytes, tier_budget(engine, target))) {
            state.tier = target;
            if (target == GLM5_TIER_VRAM) engine->vram_expert_count.fetch_add(1);
            else if (target == GLM5_TIER_RAM) engine->ram_expert_count.fetch_add(1);
            else if (target == GLM5_TIER_DB) engine->db_expert_count.fetch_add(1);
            evict_snap = state; need_evict_update = true;
        } else if (engine->config.allow_db_streaming) {
            state.tier = GLM5_TIER_DB;
            engine->db_expert_count.fetch_add(1);
            add_tier_usage(engine, GLM5_TIER_DB, state.bytes);
            evict_snap = state; need_evict_update = true;
        } else {
            state = glm5_block_state{};
            return 0;
        }
        return 1;
    }();
    // heap_mutex acquired AFTER map_mutex released - correct ordering
    if (need_evict_update) {
        replace_evict_candidate(engine, slot, evict_snap);
    }
    for (auto& pair : to_free) {
        device_free_expert(engine, pair.first, pair.second);
    }
    return result;
}
