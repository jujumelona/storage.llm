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

    // BUGFIX 115: Fix wait-free saturating subtract data race ★★
    // Problem: store(0) overwrites concurrent fetch_add, permanently leaking tier budget.
    // Solution: Add back the exact over-subtracted amount instead of forcing to 0.
    // This preserves concurrent additions while correctly saturating at 0.
    if (old_val < bytes) {
        counter->fetch_add(bytes - old_val, std::memory_order_relaxed);
    }
}

// Eviction heap compaction snapshots under heap_mutex, then validates/rebuilds outside it.
static constexpr size_t EVICT_HEAP_COMPACTION_NUMERATOR = 3;
static constexpr size_t EVICT_HEAP_COMPACTION_DENOMINATOR = 2;

#include <cassert>

static std::mutex& evict_heap_mutex(moe_pc_engine* engine, int idx) {
    const int safe_idx = (idx >= 0 && idx < 3) ? idx : 0;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    assert(&engine->heap_mutexes[safe_idx] != &engine->heap_mutex_abi_compat_unused);
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
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
        // BUGFIX 161: Strict budget check BEFORE CAS to prevent TOCTOU
        // Re-check budget in loop (current may have changed)
        // Use strict inequality to prevent any possibility of exceeding budget
        if (budget != 0 && (current > budget || bytes > budget - current)) {
            return 0;
        }

        // CAS attempt - current is re-loaded on failure
        if (counter->compare_exchange_weak(
                current,
                current + bytes,
                std::memory_order_acq_rel,  // BUGFIX 161: Use acq_rel for proper synchronization
                std::memory_order_acquire)) {  // BUGFIX 161: Use acquire on failure
            // Trigger eviction if threshold exceeded
            if (tier == moe_TIER_VRAM && (current + bytes) > threshold_85) {
                engine->vram_eviction_needed.store(true, std::memory_order_relaxed);
                engine->vram_eviction_cv.notify_one();
            } else if (tier == moe_TIER_RAM && (current + bytes) > threshold_85) {
                engine->ram_eviction_needed.store(true, std::memory_order_relaxed);
                // BUGFIX 110: Notify vram_eviction_cv for RAM eviction too
                // proactive_eviction_worker_loop waits ONLY on vram_eviction_cv for BOTH VRAM and RAM evictions.
                // Notifying ram_eviction_cv here resulted in a Lost Wakeup, delaying RAM eviction by up to 25ms (the wait timeout).
                engine->vram_eviction_cv.notify_one();
            }
            return 1;
        }
        // CAS failed - current is updated with latest value, loop continues
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

// Lock order: heap_mutex -> map_mutex (device_mutex MUST NOT be held here)
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
        // BUGFIX 603: Check active_count * 4u overflow ★
        // Problem: active_count * 4u can overflow uint64_t
        // Solution: Check before multiplication
        if (!force && active_count <= UINT64_MAX / 4u && heap_size <= active_count * 4u + 512u) {
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
            // BUGFIX 604: Check top.slot bounds before accessing expert_states ★
            // Problem: top.slot can be corrupted or out of bounds
            // Solution: Check bounds before access
            if (top.slot >= engine->expert_states.size()) {
                continue;
            }
            const moe_block_state& state = engine->expert_states[top.slot];
            if (state.active && state.tier == top.tier && state.bytes == top.bytes &&
                state.hot_score == top.hot_score &&
                state.last_touch_epoch == top.last_touch_epoch &&
                state.predicted_next_use_epoch == top.predicted_next_use_epoch) {
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
    candidate.predicted_next_use_epoch = state.predicted_next_use_epoch;

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
        candidate.predicted_next_use_epoch = state.predicted_next_use_epoch;

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

    // ARC frequency tracking (Megiddo & Modha 2003, adapted for MoE)
    state->access_count = (state->access_count < UINT16_MAX) ?
        state->access_count + 1u : UINT16_MAX;

    if (gap > 0) {
        // BUGFIX 196: Update avg_gap_ema for better prediction ★★
        // EMA with alpha=0.25 (weight 3:1 for old:new)
        if (state->avg_gap_ema > 0) {
            state->avg_gap_ema = (state->avg_gap_ema * 3 + gap) / 4;
        } else {
            state->avg_gap_ema = gap;  // First gap
        }

        // ARC-adaptive decay: T2 (frequent, >=3 hits) decays slower than T1
        const bool is_frequent = state->access_count >= 3u;
        const int decay_shift_base = is_frequent ? 16 : 12;
        // BUGFIX 551: Check gap before right shift to prevent overflow ★
        // Problem: gap >> decay_shift_base can overflow if gap is very large
        // If gap > 2^63, shift result is undefined behavior
        // Solution: Clamp gap to reasonable range before shift
        const uint64_t clamped_gap = std::min<uint64_t>(gap, (uint64_t{1} << 60));
        const int shift = (int)(clamped_gap >> decay_shift_base);
        const double factor = shift >= 52 ? 0.0 : std::ldexp(1.0, -shift);
        state->hot_score *= factor;
    }

    // Frequency-weighted boost: heavy hitters get higher heat
    const double freq_mult =
        state->access_count >= 5u ? 2.0 :
        state->access_count >= 3u ? 1.5 :
        state->access_count >= 2u ? 1.2 : 1.0;
    state->hot_score += boost * freq_mult;
    state->last_touch_epoch = now;

    // BUGFIX 196: Use avg_gap_ema for horizon prediction ★★
    // Old: horizon = 8192.0 / heat (inverse of hot_score)
    // New: Use actual measured gap EMA when available
    // More accurate for periodic access patterns
    uint64_t horizon = 0;
    if (state->avg_gap_ema > 0) {
        horizon = state->avg_gap_ema;
        // Frequent experts: predict sooner re-use (reduce horizon by 25%)
        if (state->access_count >= 3u) {
            horizon = (horizon * 3) / 4;
        }
    } else {
        // Fallback to hot_score-based prediction for new experts
        const double heat = state->hot_score > 0.001 ? state->hot_score : 0.001;
        horizon = (uint64_t)(8192.0 / heat);
        // Frequent experts: predict sooner re-use
        if (state->access_count >= 3u) {
            horizon = horizon > 1ull ? horizon / 2ull : 1ull;
        }
    }

    if (horizon < 1ull) {
        horizon = 1ull;
    } else if (horizon > (1ull << 30)) {
        horizon = (1ull << 30);
    }
    state->predicted_next_use_epoch = now > UINT64_MAX - horizon ? UINT64_MAX : now + horizon;
}

// BUGFIX 197: Preserve access_count on VRAM→RAM demotion ★★
// Problem: Memory pressure forces VRAM→RAM, but access_count reset to 0
// Hot expert treated as T1 (first-time) when re-promoted → fast decay
// Solution: Only reset on RAM→DISK demotion (actual cold data)
// ARC: reset access_count on demotion so re-admitted experts start in T1
static void moe_arc_reset_on_demotion(moe_block_state* state, moe_tier_t from, moe_tier_t to) {
    if (!state) return;
    // BUGFIX 197: Only reset for RAM→DISK, preserve for VRAM→RAM
    if (from == moe_TIER_VRAM && to == moe_TIER_RAM) {
        // Memory pressure demotion - keep access_count
        // Expert likely to be re-promoted soon
    } else {
        // Actual cold data demotion - reset to T1
        state->access_count = 0;
    }
}
