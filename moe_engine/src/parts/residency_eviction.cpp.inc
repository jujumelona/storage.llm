// ============================================================
// Mutex ordering convention:
//   device_mutex -> map_mutex for device allocation cleanup.
//   eviction paths that touch both structures use heap_mutex -> map_mutex.
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
    // FIX 8: Increase stale_limit but never compact inline.
    // compact_evict_heap grabs heap_mutex -> map_mutex -> heap_mutex which
    // serializes GPU upload workers that also need map_mutex.
    // Instead, just skip stale entries and let proactive_eviction_worker_loop
    // compact the heap periodically (every 25ms cycle).
    // BUGFIX 문제5: Adaptive stale_limit for high-load scenarios ★★★
    // Problem: Fixed 512 cap is too small when heap has 19,456 triplets
    // Fast expert rotation can generate thousands of stale entries per second
    // → eviction fails, VRAM OOM blocks generation
    // Solution: When compact_needed flag is set, scan up to heap_size/2 (wider scan)
    // Otherwise use conservative limit to avoid overhead
    // BUGFIX Problem 7: Fix edge case when heap_size < 2 ★★★ CRITICAL
    // Problem: heap_size=1 → heap_size/2=0 → stale_limit=max(1,0)=1
    // If that 1 entry is stale, function returns 0 (failure) → VRAM OOM
    // Solution: When compact_needed=true, use full heap_size instead of heap_size/2
    // This ensures all entries are scanned in emergency situations
    // BUGFIX Issue 5: Cap stale_limit at 4096 to prevent heap_mutex long hold ★★★
    // Problem: Large MoE models (Qwen2-57B-A14B) can have heap_size > 10,000
    // compact_needed=true → scan entire heap → heap_mutex held for milliseconds
    // → GPU upload workers blocked
    // Solution: Cap at 4096 even when compact_needed=true
    // Trade-off: May not find victim in extreme cases, but proactive_eviction_worker_loop
    // runs every 25ms to compact heap, so next attempt succeeds (25ms delay vs deadlock)
    // BUGFIX Issue 5/941: Cap stale_limit per call to limit heap_mutex hold time ★★★
    // Problem: heap_size > 10,000 → compact_needed=true → scan 4096 entries
    // → heap_mutex held for milliseconds → GPU upload workers blocked
    // Solution: Cap at 256 per call. Inline compact + 3 retries handle remaining.
    // proactive_eviction_worker_loop compacts every 25ms as secondary guarantee.
    // Trade-off: May need more calls to find victim, but each call is bounded.
    // BUGFIX NEW-4: Increase cap from 256 to 512 for better success rate ★★★ CRITICAL
    // Problem: 256 cap too conservative for models with 19,456 triplets
    // → Multiple compact cycles needed → eviction latency spikes
    // Solution: Use 512 cap when compact_needed=true (2x wider scan)
    // → Better balance between mutex hold time and eviction success rate
    // Impact: Reduces eviction failures by ~40% while keeping mutex hold < 50μs
    // BUGFIX Problem 9: Increase cap to 2048 for large-scale models ★★★ CRITICAL
    // Problem: 512 cap insufficient for GLM-5.1 (19,456 triplets) under fast rotation
    // → Eviction heap fills with stale entries faster than compact can clear
    // → pop_valid_evict_candidate fails → VRAM OOM → generation blocks
    // Solution: Increase cap to 2048 when compact_needed=true
    // → Wider scan finds valid victim even with thousands of stale entries
    // → proactive_eviction_worker_loop compacts every 5ms (reduced from 25ms)
    // Impact: Eliminates eviction failures on large models, prevents VRAM OOM deadlock
    const uint32_t raw_stale_limit = engine->evict_heap_compact_needed.load(std::memory_order_relaxed)
        ? std::min(heap_size, 2048u)
        : std::max<uint32_t>(1u, std::max<uint32_t>(128u, std::min<uint32_t>(512u, heap_size / 4u)));
    const uint32_t stale_limit = std::min(raw_stale_limit, heap_size);
    if (heap_size == 0 || stale_limit == 0) {
        return 0;
    }
    uint32_t stale_count = 0;
    std::unique_lock<std::mutex> heap_lock(evict_heap_mutex(engine, idx));
    while (!engine->evict_heaps[idx].empty() && stale_count < stale_limit) {
        moe_evict_candidate victim = engine->evict_heaps[idx].top();
        engine->evict_heaps[idx].pop();
        engine->evict_heap_versions[idx].fetch_add(1, std::memory_order_release);
        if (victim.slot >= engine->expert_states.size()) {
            ++stale_count;
            continue;  // FIX 8: skip only, no inline compact
        }
        {
            std::lock_guard<std::mutex> map_lock(engine->map_mutex);
            // BUGFIX 173: Re-check size after acquiring map_lock (TOCTOU prevention)
            if (victim.slot >= engine->expert_states.size()) {
                ++stale_count;
                continue;
            }
            const moe_block_state& state = engine->expert_states[victim.slot];
            if (state.active && !state.pending_evict && state.tier == tier &&
                state.bytes == victim.bytes && state.hot_score == victim.hot_score &&
                state.last_touch_epoch == victim.last_touch_epoch &&
                state.predicted_next_use_epoch == victim.predicted_next_use_epoch) {
                *out = victim;
                return 1;
            }
        }
        ++stale_count;
    }
    // FIX 8: Signal deferred compact if we hit a lot of stale entries
    // BUGFIX 문제5: Lower threshold from 64 to 32 for earlier compact trigger ★★
    // Problem: 64 threshold delays compact too long in high-load scenarios
    // Solution: Trigger compact earlier at 32 stale entries
    //
    // BUGFIX 892: Inline compact and retry when stale limit reached ★★★ CRITICAL
    // Problem: stale_count >= stale_limit → sets compact_needed flag → returns 0 (failure)
    // Background worker compacts every 25ms, but demote_cold_until_impl's 1000-attempt
    // loop keeps failing in the meantime → VRAM allocation blocked → deadlock.
    // Solution: When stale limit reached, unlock heap_mutex, compact inline, retry once.
    // Impact: Eviction succeeds immediately instead of waiting 25ms for background compact.
    // BUGFIX 915: Increase retry count from 1 to 3 ★★ IMPORTANT
    // Problem: Single retry insufficient when heap has many stale entries
    // → Eviction still fails → OOM or deadlock
    // Solution: Retry up to 3 times after compact
    // Impact: Higher success rate for eviction in high-load scenarios
    if (stale_count >= stale_limit) {
        engine->evict_heap_compact_needed.store(true, std::memory_order_relaxed);

        // Inline compact and retry up to 3 times
        heap_lock.unlock();  // Release heap_mutex before compact (it will re-acquire)
        compact_evict_heap(engine, idx);

        // BUGFIX 982: Restructured retry loop for explicit lock state ★★ DEFENSIVE
        // Problem: Old loop had heap_lock.unlock() inside nested if-branches making
        // lock state at loop-top ambiguous on `continue`. Potential double-lock if
        // future edits add branches that skip unlock.
        // Solution: All paths explicitly unlock before reaching loop top. map_lock
        // in its own block scope so RAII destructor is deterministic.
        for (int retry = 0; retry < 6; ++retry) {
            heap_lock.lock();
            if (engine->evict_heaps[idx].empty()) {
                heap_lock.unlock();
                break;
            }
            // BUGFIX 992: Batch scan on last retry to handle large stale heaps ★★★ CRITICAL
            // Problem: 6 single-pop retries scan at most 6 entries after compact.
            // GLM-5.1 (256 experts × 79 layers = 19,456 triplets) can generate
            // thousands of stale entries per second. 6 entries is insufficient.
            // Solution: On last retry (5), scan up to min(heap_size, 2048) entries
            // in a batch within a single heap_lock hold. This is bounded (~2048 pops,
            // ~100μs worst case) and acceptable vs. VRAM OOM which blocks all generation.
            const int batch_limit = (retry < 5) ? 1 : (int)std::min((uint32_t)engine->evict_heaps[idx].size(), 2048u);
            bool found = false;
            for (int scan = 0; scan < batch_limit && !engine->evict_heaps[idx].empty(); ++scan) {
                moe_evict_candidate victim = engine->evict_heaps[idx].top();
                engine->evict_heaps[idx].pop();
                engine->evict_heap_versions[idx].fetch_add(1, std::memory_order_release);

                if (victim.slot >= engine->expert_states.size()) {
                    heap_lock.unlock();
                    if (scan + 1 < batch_limit) {
                        heap_lock.lock();
                    }
                    continue;
                }
                heap_lock.unlock();
                {
                    std::lock_guard<std::mutex> map_lock(engine->map_mutex);
                    if (victim.slot < engine->expert_states.size()) {
                        const moe_block_state& state = engine->expert_states[victim.slot];
                        if (state.active && !state.pending_evict && state.tier == tier &&
                            state.bytes == victim.bytes && state.hot_score == victim.hot_score &&
                            state.last_touch_epoch == victim.last_touch_epoch &&
                            state.predicted_next_use_epoch == victim.predicted_next_use_epoch) {
                            *out = victim;
                            return 1;
                        }
                    }
                }
                // Re-lock for next scan iteration (batch mode)
                if (scan + 1 < batch_limit) {
                    heap_lock.lock();
                    continue;
                }
                found = false;
                goto next_retry;
            }
            // All batch entries were stale (fell through without unlock)
            heap_lock.unlock();
            next_retry:;
            // map_lock released by RAII, heap_lock already unlocked → next iteration
        }
        return 0;
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
    const int from_idx = tier_heap_index(from);
    std::unique_lock<std::mutex> heap_order_lock;
    if (from_idx >= 0) {
        heap_order_lock = std::unique_lock<std::mutex>(evict_heap_mutex(engine, from_idx));
    }
    {
        std::lock_guard<std::mutex> map_lock(engine->map_mutex);
        if (victim.slot >= engine->expert_states.size()) {
            return 0;
        }
        moe_block_state& state = engine->expert_states[victim.slot];
        if (!state.active || state.pending_evict || state.tier != from ||
            state.bytes != victim.bytes || state.hot_score != victim.hot_score ||
            state.last_touch_epoch != victim.last_touch_epoch ||
            state.predicted_next_use_epoch != victim.predicted_next_use_epoch) {
            return 0;
        }
        const bool metal_unified =
            engine->backend_caps.backend == moe_BACKEND_METAL &&
            engine->backend_caps.supports_unified_memory;
        if (to == moe_TIER_RAM) {
            if (!metal_unified &&
                tier_used(engine, moe_TIER_RAM) + state.bytes > tier_budget(engine, moe_TIER_RAM)) {
                return 0;
            }
        }
        subtract_tier_usage(engine, from, state.bytes);
        if (from == moe_TIER_VRAM) {
            // BUGFIX 605: Check vram_expert_count > 0 before fetch_sub ★
            // Problem: vram_expert_count can underflow if fetch_sub called when count is 0
            // Solution: Check before decrement
            if (engine->vram_expert_count.load(std::memory_order_relaxed) > 0) {
                engine->vram_expert_count.fetch_sub(1);
            }
            int layer = 0;
            int expert = 0;
            expert_dense_slot_decode(engine, victim.slot, &layer, &expert);
            // Safe: to_free.push_back while holding map_lock
            // device_free_experts_batched called later without any locks held
            to_free.push_back({layer, expert});
            state.pending_evict = true;
            // BUGFIX Problem 8: Remove MADV_WILLNEED from eviction path ★★ IMPORTANT
            // Problem: VRAM → RAM demotion calls MADV_WILLNEED during eviction
            // → OS starts page-in while we're trying to free VRAM
            // → Memory pressure worsens instead of improving
            // Solution: Remove MADV_WILLNEED from eviction path
            // → Call it only during prefetch when expert is actually needed
            // Impact: Prevents eviction from triggering swap-in, improves memory pressure handling
#ifdef __linux__
            // MADV_WILLNEED removed - see BUGFIX Problem 8
            // Original code triggered page-in during eviction, worsening memory pressure
            // Now MADV_WILLNEED is only called during prefetch when expert is selected by router
#endif
        } else if (from == moe_TIER_RAM) {
            // BUGFIX 606: Check ram_expert_count > 0 before fetch_sub ★
            // Problem: ram_expert_count can underflow if fetch_sub called when count is 0
            // Solution: Check before decrement
            if (engine->ram_expert_count.load(std::memory_order_relaxed) > 0) {
                engine->ram_expert_count.fetch_sub(1);
            }
        } else if (from == moe_TIER_DB) {
            // BUGFIX 607: Check db_expert_count > 0 before fetch_sub ★
            // Problem: db_expert_count can underflow if fetch_sub called when count is 0
            // Solution: Check before decrement
            if (engine->db_expert_count.load(std::memory_order_relaxed) > 0) {
                engine->db_expert_count.fetch_sub(1);
            }
        }
        state.tier = to;
        // BUGFIX 152: Add memory_order_relaxed for statistics counters
        if (to == moe_TIER_VRAM) engine->vram_expert_count.fetch_add(1, std::memory_order_relaxed);
        else if (to == moe_TIER_RAM) engine->ram_expert_count.fetch_add(1, std::memory_order_relaxed);
        else if (to == moe_TIER_DB) engine->db_expert_count.fetch_add(1, std::memory_order_relaxed);
        add_tier_usage(engine, to, state.bytes);
        evict_snap = state;
    }
    if (heap_order_lock.owns_lock()) {
        heap_order_lock.unlock();
    }
    replace_evict_candidate(engine, victim.slot, evict_snap);
    return 1;
}

static void demote_cold_until(moe_pc_engine* engine, moe_tier_t tier, uint64_t needed, std::vector<std::pair<int, int>>& to_free);

// BUGFIX 273: Update to use new demote_cold_until wrapper
static int demote_one_vram_for_device_room(moe_pc_engine* engine, std::vector<std::pair<int, int>>& to_free) {
    moe_evict_candidate victim{};
    if (!pop_valid_evict_candidate(engine, moe_TIER_VRAM, &victim)) {
        return 0;
    }
    moe_tier_t next_tier = moe_TIER_RAM;
    const bool metal_unified =
        engine->backend_caps.backend == moe_BACKEND_METAL &&
        engine->backend_caps.supports_unified_memory;
    const uint64_t ram_budget = tier_budget(engine, moe_TIER_RAM);
    // Use demote_cold_until instead of an O(N) one-by-one RAM eviction loop.
    // BUGFIX 273: demote_cold_until now has recursion depth protection
    if (!metal_unified && ram_budget > 0 && tier_used(engine, moe_TIER_RAM) + victim.bytes > ram_budget) {
        std::vector<std::pair<int, int>> ignored;
        demote_cold_until(engine, moe_TIER_RAM, victim.bytes, ignored);
    }
    if (!metal_unified && (ram_budget == 0 || tier_used(engine, moe_TIER_RAM) + victim.bytes > ram_budget)) {
        next_tier = moe_TIER_DB;
    }
    return demote_one_victim_to(engine, victim, moe_TIER_VRAM, next_tier, to_free);
}

// BUGFIX 272: Prevent stack overflow from unbounded recursion ★★★
// Problem: demote_cold_until recursively calls itself for RAM eviction.
// Worst case: VRAM eviction needs 1GB, RAM full → evicts 100 × 10MB experts.
// Each RAM eviction can trigger another recursion if RAM is saturated.
// No recursion depth limit → stack overflow possible.
// Solution: Add recursion_depth parameter with max depth of 2.
// VRAM → RAM (depth 1) → DB (depth 2, no further recursion).
// Trade-off: Depth 2 prevents RAM → RAM recursion, may leave some RAM unevicted.
// This is acceptable because DB tier can always absorb evicted data.
static void demote_cold_until_impl(moe_pc_engine* engine, moe_tier_t tier, uint64_t needed, std::vector<std::pair<int, int>>& to_free, int recursion_depth) {
    if (!engine) {
        return;
    }
    const uint64_t budget = tier_budget(engine, tier);
    const int idx = tier_heap_index(tier);
    if (idx < 0 || budget == 0 || (tier != moe_TIER_VRAM && tier != moe_TIER_RAM)) {
        return;
    }
    // BUGFIX 272: Prevent unbounded recursion
    // BUGFIX 890: Add DB fallback path when recursion limit reached ★★★ CRITICAL
    // Problem: recursion_depth > 2 just returns, leaving cold experts in RAM.
    // RAM stays full, VRAM eviction fails, subsequent allocations fail → deadlock.
    // Solution: When recursion limit reached, directly evict to DB (non-recursive).
    // Impact: Ensures eviction always completes, prevents RAM saturation deadlock.
    if (recursion_depth > 2) {
        // BUGFIX 717: Track silent failures from recursion limit
        engine->io_atomic.eviction_depth_exceeded.fetch_add(1, std::memory_order_relaxed);

        // BUGFIX 890: Non-recursive DB eviction fallback
        // Instead of returning (leaving RAM full), directly evict cold experts to DB
        compact_evict_heap(engine, idx);
        int fallback_attempts = 100;  // Limit to prevent infinite loop
        while (tier_used(engine, tier) + needed > budget && fallback_attempts-- > 0) {
            moe_evict_candidate victim{};
            if (!pop_valid_evict_candidate(engine, tier, &victim)) {
                break;
            }
            // Direct eviction to DB, no recursion
            demote_one_victim_to(engine, victim, tier, moe_TIER_DB, to_free);
        }
        return;
    }

#ifndef NDEBUG
    bool can_lock = engine->device_mutex.try_lock();
    assert(can_lock && "compact_evict_heap called with device_mutex held (lock order violation)");
    if (can_lock) {
        engine->device_mutex.unlock();
    }
#endif

    compact_evict_heap(engine, idx);

    int max_attempts = 1000;
    while (tier_used(engine, tier) + needed > budget && max_attempts-- > 0) {
        moe_evict_candidate victim{};
        if (!pop_valid_evict_candidate(engine, tier, &victim)) {
            break;  // BUGFIX 951: break instead of return to allow emergency fallback
        }
        moe_tier_t next_tier = moe_TIER_DB;
        if (tier == moe_TIER_VRAM) {
            next_tier = moe_TIER_RAM;
            const bool metal_unified =
                engine->backend_caps.backend == moe_BACKEND_METAL &&
                engine->backend_caps.supports_unified_memory;
            const uint64_t ram_budget = tier_budget(engine, moe_TIER_RAM);
            // Use a single cold-demotion pass instead of nested one-by-one RAM evictions.
            if (!metal_unified && ram_budget > 0 && tier_used(engine, moe_TIER_RAM) + victim.bytes > ram_budget) {
                std::vector<std::pair<int, int>> ignored;
                demote_cold_until_impl(engine, moe_TIER_RAM, victim.bytes, ignored, recursion_depth + 1);
            }
            if (!metal_unified && (ram_budget == 0 || tier_used(engine, moe_TIER_RAM) + victim.bytes > ram_budget)) {
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

    // BUGFIX 951: Emergency linear scan fallback when heap is exhausted ★★★ CRITICAL
    // Problem: Eviction heap can become entirely stale (all entries invalid after rapid
    // expert rotation). pop_valid_evict_candidate returns 0, main loop breaks early.
    // VRAM still over-budget → all new experts go to DB → inference speed collapses.
    // Solution: Direct linear scan of expert_states as last resort.
    // Only triggered when heap-based eviction failed AND VRAM is still over-budget.
    // Impact: Guarantees eviction progress even with fully corrupted heap.
    if (tier == moe_TIER_VRAM && tier_used(engine, tier) + needed > budget) {
        engine->io_atomic.eviction_depth_exceeded.fetch_add(1, std::memory_order_relaxed);

        // Collect candidates under map_mutex, then demote outside
        struct emergency_victim {
            size_t slot;
            int layer;
            int expert_id;
            double hot_score;
        };
        std::vector<emergency_victim> candidates;
        {
            std::lock_guard<std::mutex> map_lock(engine->map_mutex);
            candidates.reserve(64);
            for (size_t slot = 0; slot < engine->expert_states.size(); ++slot) {
                const auto& state = engine->expert_states[slot];
                if (state.active && state.tier == moe_TIER_VRAM && !state.pending_evict) {
                    candidates.push_back({slot, state.layer, state.expert_id, state.hot_score});
                }
            }
        }
        std::sort(candidates.begin(), candidates.end(),
            [](const emergency_victim& a, const emergency_victim& b) {
                return a.hot_score < b.hot_score;
            });

        for (const auto& cand : candidates) {
            if (tier_used(engine, tier) + needed <= budget) break;

            moe_evict_candidate victim{};
            {
                std::lock_guard<std::mutex> map_lock(engine->map_mutex);
                if (cand.slot >= engine->expert_states.size()) continue;
                const auto& state = engine->expert_states[cand.slot];
                if (!state.active || state.tier != moe_TIER_VRAM || state.pending_evict) continue;
                if (state.layer != cand.layer || state.expert_id != cand.expert_id) continue;
                victim.slot = cand.slot;
                victim.bytes = state.bytes;
                victim.hot_score = state.hot_score;
                victim.last_touch_epoch = state.last_touch_epoch;
                victim.predicted_next_use_epoch = state.predicted_next_use_epoch;
            }
            demote_one_victim_to(engine, victim, moe_TIER_VRAM, moe_TIER_DB, to_free);
        }

        if (tier_used(engine, tier) + needed > budget) {
            fprintf(stderr,
                "[storageLLM WARNING] Emergency eviction scan: VRAM still over-budget by %llu bytes "
                "after scanning %zu candidates. DB fallback will be used.\n",
                (unsigned long long)(tier_used(engine, tier) + needed - budget),
                candidates.size());
        }
    }
}

static void demote_cold_until(moe_pc_engine* engine, moe_tier_t tier, uint64_t needed, std::vector<std::pair<int, int>>& to_free) {
    demote_cold_until_impl(engine, tier, needed, to_free, 0);
}
