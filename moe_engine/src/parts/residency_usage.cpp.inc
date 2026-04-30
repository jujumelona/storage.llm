static int tier_heap_index(moe_tier_t tier) {
    switch (tier) {
        case moe_TIER_VRAM:
            return 0;
        case moe_TIER_RAM:
            return 1;
        case moe_TIER_DB:
            return 2;
        case moe_TIER_STREAM:
        default:
            return -1;
    }
}

static std::atomic<uint64_t>* tier_counter(moe_pc_engine* engine, moe_tier_t tier) {
    if (!engine) {
        return nullptr;
    }
    switch (tier) {
        case moe_TIER_VRAM:
            return &engine->vram_used_bytes;
        case moe_TIER_RAM:
            return &engine->ram_used_bytes;
        case moe_TIER_DB:
            return &engine->db_used_bytes;
        case moe_TIER_STREAM:
        default:
            return nullptr;
    }
}

static void add_tier_usage(moe_pc_engine* engine, moe_tier_t tier, uint64_t bytes) {
    std::atomic<uint64_t>* counter = tier_counter(engine, tier);
    if (counter && bytes) {
        counter->fetch_add(bytes);
    }
}

static void subtract_tier_usage(moe_pc_engine* engine, moe_tier_t tier, uint64_t bytes) {
    std::atomic<uint64_t>* counter = tier_counter(engine, tier);
    if (!counter || bytes == 0) {
        return;
    }
    // BUGFIX 87: Issue 11 - Replace CAS loop with fetch_sub ★★
    // Problem: CAS loop on every tier usage subtract (hot path)
    // High contention when multiple workers free experts simultaneously
    // Solution: Use fetch_sub with saturating correction (same as device_free_bytes)
    // Temporary underflow acceptable, tier_used() reads are approximate anyway
    const uint64_t old_val = counter->fetch_sub(bytes, std::memory_order_relaxed);

    // Saturating correction: if underflow occurred, clamp to 0
    if (old_val < bytes) {
        counter->store(0, std::memory_order_relaxed);
    }
}

// Eviction heap compaction snapshots under heap_mutex, then validates/rebuilds outside it.
static constexpr size_t EVICT_HEAP_COMPACTION_NUMERATOR = 3;
static constexpr size_t EVICT_HEAP_COMPACTION_DENOMINATOR = 2;

static std::mutex& evict_heap_mutex(moe_pc_engine* engine, int idx) {
    const int safe_idx = (idx >= 0 && idx < 3) ? idx : 0;
    return engine->heap_mutexes[safe_idx];
}

static int try_add_tier_usage(moe_pc_engine* engine, moe_tier_t tier, uint64_t bytes, uint64_t budget) {
    std::atomic<uint64_t>* counter = tier_counter(engine, tier);
    if (!counter || bytes == 0) {
        return 0;
    }
    // BUGFIX 88: Issue 12 - Optimize try_add_tier_usage CAS loop ★★
    // Problem: CAS loop retries on every contention (hot path for expert allocation)
    // Multiple workers allocating experts → high CAS contention
    // Solution: Early exit if budget exceeded (no CAS needed)
    // Only CAS when budget check passes
    uint64_t current = counter->load(std::memory_order_relaxed);

    // Fast path: check budget before CAS
    if (budget != 0 && (current > budget || bytes > budget - current)) {
        return 0;  // Budget exceeded, no CAS needed
    }

    // Slow path: CAS loop (but only when budget allows)
    // BUGFIX 99: Issue 20 - Precompute percentage threshold ★
    // Problem: (budget * 85ull) / 100ull computed on every successful CAS
    // Solution: Precompute threshold once before loop
    const uint64_t threshold_85 = budget > 0 ? (budget * 85ull) / 100ull : UINT64_MAX;
    for (;;) {
        // Re-check budget in loop (current may have changed)
        if (budget != 0 && (current > budget || bytes > budget - current)) {
            return 0;
        }
        if (counter->compare_exchange_weak(
                current,
                current + bytes,
                std::memory_order_relaxed,
                std::memory_order_relaxed)) {
            // Trigger eviction if threshold exceeded
            if (tier == moe_TIER_VRAM && (current + bytes) > threshold_85) {
                engine->vram_eviction_needed.store(true, std::memory_order_relaxed);
                engine->vram_eviction_cv.notify_one();
            } else if (tier == moe_TIER_RAM && (current + bytes) > threshold_85) {
                engine->ram_eviction_needed.store(true, std::memory_order_relaxed);
                engine->ram_eviction_cv.notify_one();
            }
            return 1;
        }
    }
}

static uint64_t evict_heap_active_count(const moe_pc_engine* engine, int idx) {
    if (!engine) {
        return 0;
    }
    if (idx == 0) {
        return engine->vram_expert_count.load(std::memory_order_relaxed);
    }
    if (idx == 1) {
        return engine->ram_expert_count.load(std::memory_order_relaxed);
    }
    if (idx == 2) {
        return engine->db_expert_count.load(std::memory_order_relaxed);
    }
    return 0;
}

static void compact_evict_heap(moe_pc_engine* engine, int idx, bool force = false) {
    if (!engine || idx < 0 || idx >= 3) return;
    using heap_t = std::priority_queue<moe_evict_candidate, std::vector<moe_evict_candidate>, moe_evict_colder_first>;
    heap_t snapshot;
    size_t heap_size = 0;
    {
        std::lock_guard<std::mutex> heap_lock(evict_heap_mutex(engine, idx));
        heap_size = engine->evict_heaps[idx].size();
        if (heap_size == 0) {
            return;
        }
        const uint64_t active_count = evict_heap_active_count(engine, idx);
        if (!force && heap_size <= active_count * 4u + 512u) {
            return;
        }
        snapshot = std::move(engine->evict_heaps[idx]);
        engine->evict_heaps[idx] = heap_t();
    }

    std::vector<moe_evict_candidate> entries;
    entries.reserve(heap_size);
    while (!snapshot.empty()) {
        entries.push_back(snapshot.top());
        snapshot.pop();
    }

    std::vector<moe_evict_candidate> valid;
    valid.reserve(entries.size());
    {
        std::lock_guard<std::mutex> map_lock(engine->map_mutex);
        for (const auto& top : entries) {
            if (top.slot >= engine->expert_states.size()) {
                continue;
            }
            const moe_block_state& state = engine->expert_states[top.slot];
            if (state.active && state.tier == top.tier && state.bytes == top.bytes &&
                state.hot_score == top.hot_score && state.last_touch_epoch == top.last_touch_epoch) {
                valid.push_back(top);
            }
        }
    }
    heap_t compacted(moe_evict_colder_first{}, std::move(valid));
    {
        std::lock_guard<std::mutex> heap_lock(evict_heap_mutex(engine, idx));
        while (!engine->evict_heaps[idx].empty()) {
            compacted.push(engine->evict_heaps[idx].top());
            engine->evict_heaps[idx].pop();
        }
        engine->evict_heaps[idx] = std::move(compacted);
        engine->evict_heap_versions[idx].fetch_add(1, std::memory_order_release);
    }
}

static void push_evict_candidate(moe_pc_engine* engine, size_t slot, const moe_block_state& state) {
    const int idx = tier_heap_index(state.tier);
    if (!engine || idx < 0 || state.bytes == 0) {
        return;
    }
    moe_evict_candidate candidate;
    candidate.slot = slot;
    candidate.tier = state.tier;
    candidate.bytes = state.bytes;
    candidate.hot_score = state.hot_score;
    candidate.last_touch_epoch = state.last_touch_epoch;

    std::lock_guard<std::mutex> heap_lock(evict_heap_mutex(engine, idx));
    engine->evict_heaps[idx].push(candidate);
    engine->evict_heap_versions[idx].fetch_add(1, std::memory_order_release);
}

static void replace_evict_candidate(moe_pc_engine* engine, size_t slot, const moe_block_state& state) {
    if (!engine) {
        return;
    }
    const int target_idx = tier_heap_index(state.tier);
    if (target_idx >= 0 && state.bytes != 0) {
        moe_evict_candidate candidate;
        candidate.slot = slot;
        candidate.tier = state.tier;
        candidate.bytes = state.bytes;
        candidate.hot_score = state.hot_score;
        candidate.last_touch_epoch = state.last_touch_epoch;

        std::lock_guard<std::mutex> heap_lock(evict_heap_mutex(engine, target_idx));
        engine->evict_heaps[target_idx].push(candidate);
        engine->evict_heap_versions[target_idx].fetch_add(1, std::memory_order_release);
    }
}

static void touch_expert_heat(moe_pc_engine* engine, moe_block_state* state, double boost) {
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
