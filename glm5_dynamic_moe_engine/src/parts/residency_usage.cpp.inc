static int tier_heap_index(glm5_tier_t tier) {
    switch (tier) {
        case GLM5_TIER_VRAM:
            return 0;
        case GLM5_TIER_RAM:
            return 1;
        case GLM5_TIER_DB:
            return 2;
        case GLM5_TIER_STREAM:
        default:
            return -1;
    }
}

static std::atomic<uint64_t>* tier_counter(glm5_pc_engine* engine, glm5_tier_t tier) {
    if (!engine) {
        return nullptr;
    }
    switch (tier) {
        case GLM5_TIER_VRAM:
            return &engine->vram_used_bytes;
        case GLM5_TIER_RAM:
            return &engine->ram_used_bytes;
        case GLM5_TIER_DB:
            return &engine->db_used_bytes;
        case GLM5_TIER_STREAM:
        default:
            return nullptr;
    }
}

static void add_tier_usage(glm5_pc_engine* engine, glm5_tier_t tier, uint64_t bytes) {
    std::atomic<uint64_t>* counter = tier_counter(engine, tier);
    if (counter && bytes) {
        counter->fetch_add(bytes);
    }
}

static void subtract_tier_usage(glm5_pc_engine* engine, glm5_tier_t tier, uint64_t bytes) {
    std::atomic<uint64_t>* counter = tier_counter(engine, tier);
    if (!counter || bytes == 0) {
        return;
    }
    uint64_t current = counter->load();
    for (;;) {
        const uint64_t next = current > bytes ? current - bytes : 0;
        if (counter->compare_exchange_weak(current, next)) {
            return;
        }
    }
}

// Fix 53: Single compaction function — caller must hold heap_mutex AND map_mutex
static constexpr size_t EVICT_HEAP_COMPACTION_NUMERATOR = 3;
static constexpr size_t EVICT_HEAP_COMPACTION_DENOMINATOR = 2;

static int try_add_tier_usage(glm5_pc_engine* engine, glm5_tier_t tier, uint64_t bytes, uint64_t budget) {
    std::atomic<uint64_t>* counter = tier_counter(engine, tier);
    if (!counter || bytes == 0) {
        return 0;
    }
    uint64_t current = counter->load(std::memory_order_relaxed);
    for (;;) {
        if (budget != 0 && (current > budget || bytes > budget - current)) {
            return 0;
        }
        if (counter->compare_exchange_weak(
                current,
                current + bytes,
                std::memory_order_relaxed,
                std::memory_order_relaxed)) {
            return 1;
        }
    }
}

// Caller must hold heap_mutex and must not already hold map_mutex.
static void compact_evict_heap_locked(glm5_pc_engine* engine, int idx, bool force = false) {
    if (!engine || idx < 0 || idx >= 3) return;
    const size_t heap_size = engine->evict_heaps[idx].size();
    if (heap_size == 0) {
        return;
    }
    std::vector<glm5_evict_candidate> valid;
    valid.reserve(heap_size);
    {
        std::lock_guard<std::mutex> map_lock(engine->map_mutex);
        while (!engine->evict_heaps[idx].empty()) {
            const auto top = engine->evict_heaps[idx].top();
            engine->evict_heaps[idx].pop();
            if (top.slot >= engine->expert_states.size()) {
                continue;
            }
            const glm5_block_state& state = engine->expert_states[top.slot];
            if (state.active && state.tier == top.tier && state.bytes == top.bytes &&
                state.hot_score == top.hot_score && state.last_touch_epoch == top.last_touch_epoch) {
                valid.push_back(top);
            }
        }
    }
    using heap_t = std::priority_queue<glm5_evict_candidate, std::vector<glm5_evict_candidate>, glm5_evict_colder_first>;
    engine->evict_heaps[idx] = heap_t(glm5_evict_colder_first{}, std::move(valid));
}

static void push_evict_candidate(glm5_pc_engine* engine, size_t slot, const glm5_block_state& state) {
    const int idx = tier_heap_index(state.tier);
    if (!engine || idx < 0 || state.bytes == 0) {
        return;
    }
    glm5_evict_candidate candidate;
    candidate.slot = slot;
    candidate.tier = state.tier;
    candidate.bytes = state.bytes;
    candidate.hot_score = state.hot_score;
    candidate.last_touch_epoch = state.last_touch_epoch;

    // Fix 42: Dedicated heap_mutex protects evict_heaps from device/map mutex races
    std::lock_guard<std::mutex> heap_lock(engine->heap_mutex);
    engine->evict_heaps[idx].push(candidate);
}

static void replace_evict_candidate(glm5_pc_engine* engine, size_t slot, const glm5_block_state& state) {
    if (!engine) {
        return;
    }
    const int target_idx = tier_heap_index(state.tier);
    std::lock_guard<std::mutex> heap_lock(engine->heap_mutex);
    using heap_t = std::priority_queue<glm5_evict_candidate, std::vector<glm5_evict_candidate>, glm5_evict_colder_first>;
    for (int idx = 0; idx < 3; ++idx) {
        std::vector<glm5_evict_candidate> kept;
        kept.reserve(engine->evict_heaps[idx].size());
        while (!engine->evict_heaps[idx].empty()) {
            const glm5_evict_candidate top = engine->evict_heaps[idx].top();
            engine->evict_heaps[idx].pop();
            if (top.slot != slot) {
                kept.push_back(top);
            }
        }
        engine->evict_heaps[idx] = heap_t(glm5_evict_colder_first{}, std::move(kept));
    }
    if (target_idx >= 0 && state.bytes != 0) {
        glm5_evict_candidate candidate;
        candidate.slot = slot;
        candidate.tier = state.tier;
        candidate.bytes = state.bytes;
        candidate.hot_score = state.hot_score;
        candidate.last_touch_epoch = state.last_touch_epoch;
        engine->evict_heaps[target_idx].push(candidate);
    }
}

static void touch_expert_heat(glm5_pc_engine* engine, glm5_block_state* state, double boost) {
    if (!engine || !state) {
        return;
    }
    const uint64_t now = engine->expert_touch_epoch.fetch_add(1) + 1;
    const uint64_t gap = state->last_touch_epoch ? now - state->last_touch_epoch : 0;
    if (gap > 0) {
        // Fix 54: >>14 (gap/16384) instead of >>10 (gap/1024) to prevent
        // premature eviction of recently-touched experts in high-touch MoE
        const int shift = (int)(gap >> 14);
        const double factor = shift >= 52 ? 0.0 : std::ldexp(1.0, -shift);
        state->hot_score *= factor;
    }
    state->hot_score += boost;
    state->last_touch_epoch = now;
}
