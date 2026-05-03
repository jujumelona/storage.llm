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
    const uint32_t stale_limit = engine->evict_heap_compact_needed.load(std::memory_order_relaxed)
        ? (heap_size > 0 ? std::min(heap_size, std::max(1u, heap_size / 2u)) : 0u)
        : std::max<uint32_t>(128u, std::min<uint32_t>(512u, heap_size / 4u));
    if (stale_limit == 0) {
        engine->evict_heap_compact_needed.store(false, std::memory_order_relaxed);
        return 0;  // heap 자체가 비어있음, 정상 종료
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
    if (stale_count >= 32u) {
        engine->evict_heap_compact_needed.store(true, std::memory_order_relaxed);
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
        if (to == moe_TIER_RAM && !metal_unified &&
            tier_used(engine, moe_TIER_RAM) + state.bytes > tier_budget(engine, moe_TIER_RAM)) {
            return 0;
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
#ifdef __linux__
            if (to == moe_TIER_RAM) {
                const moe_projection_t projs[3] = {moe_PROJ_GATE, moe_PROJ_UP, moe_PROJ_DOWN};
                for (moe_projection_t proj : projs) {
                    const moe_tensor_record* r = find_tensor_record(engine, layer, expert, proj);
                    if (r) {
                        mmap_context_t* shard = get_or_map_tensor_shard(engine, *r);
                        if (shard) {
                            const uint64_t ranges[][2] = {
                                {r->info.weight_byte_offset, r->info.weight_byte_length},
                                {r->info.scale_byte_offset, r->info.scale_byte_length},
                                {r->info.scale2_byte_offset, r->info.scale2_byte_length},
                            };
                            for (const auto& range : ranges) {
                                if (range[1] > 0) {
                                    void* ptr = mmap_get_ptr(shard, static_cast<size_t>(range[0]));
                                    if (ptr) madvise(ptr, static_cast<size_t>(range[1]), MADV_WILLNEED);
                                }
                            }
                        }
                    }
                }
            }
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
    if (recursion_depth > 2) {
        // BUGFIX 717: Track silent failures from recursion limit
        engine->io_atomic.eviction_depth_exceeded.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    compact_evict_heap(engine, idx);

    int max_attempts = 1000;
    while (tier_used(engine, tier) + needed > budget && max_attempts-- > 0) {
        moe_evict_candidate victim{};
        if (!pop_valid_evict_candidate(engine, tier, &victim)) {
            return;
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
}

static void demote_cold_until(moe_pc_engine* engine, moe_tier_t tier, uint64_t needed, std::vector<std::pair<int, int>>& to_free) {
    demote_cold_until_impl(engine, tier, needed, to_free, 0);
}


