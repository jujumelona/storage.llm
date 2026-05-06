static uint32_t prefetch_stage_load(const moe_stage_queue& stage) {
    // Intentional approximation: depth and inflight are read separately without lock.
    // TOCTOU race possible but acceptable for admission control heuristic.
    // Actual value may be ±1-2 from true sum, threshold checks account for this.
    return stage.depth.load(std::memory_order_relaxed) + stage.inflight.load(std::memory_order_relaxed);
}

static int prefetch_is_router_selected(const moe_expert_prefetch_item_t& item) {
    return item.reason == moe_PREFETCH_REASON_ROUTER_SELECTED;
}

static int prefetch_under_queue_pressure(
    const moe_pc_engine* engine,
    const moe_expert_prefetch_item_t& item
) {
    if (!engine) {
        return 0;
    }
    if (engine->disk_stage_depth_cap > 0 &&
        prefetch_stage_load(engine->disk_stage) * 4u >= engine->disk_stage_depth_cap * 3u) { return 1; }
    if (item.target_tier == moe_TIER_VRAM &&
        ((engine->pinned_stage_depth_cap > 0 &&
          prefetch_stage_load(engine->pinned_stage) * 4u >= engine->pinned_stage_depth_cap * 3u) ||
         (engine->gpu_stage_depth_cap > 0 &&
          prefetch_stage_load(engine->gpu_stage) * 4u >= engine->gpu_stage_depth_cap * 3u))) { return 1; }
    // BUG 5 FIX: RAM target should also check pinned_stage load
    // When VRAM targets congest pinned_stage, RAM targets can flood disk_stage
    // and push it to 100%, even though RAM targets don't use pinned_stage directly.
    // Reject RAM targets when pinned_stage is >75% to maintain pipeline balance.
    // BUGFIX 25: Use realistic threshold based on worker count ★★
    // Old: max_queue * 75% (9216) → never reached with 1-2 pinned workers
    // New: pinned_workers * batch_max * multiplier → realistic congestion detection
    // BUGFIX Problem 5: Reduce multiplier from 12 to 6 ★★ IMPORTANT
    // Problem: 12x multiplier allows pinned_stage to fill almost completely before blocking
    // This causes disk_stage overload and GPU stream starvation
    // Solution: Use 6x multiplier for earlier RAM prefetch blocking
    // Impact: Better pipeline balance, higher GPU utilization
    if (item.target_tier == moe_TIER_RAM) {
        const uint32_t pinned_workers = runtime_pinned_workers(engine);
        const uint32_t dyn_batch_max = moe_gpu_stage_batch_max(engine);
        // Fix 4: Scale pinned_pressure_threshold properly using max_prefetch_queue
        const uint32_t pinned_pressure_threshold = std::min(
            engine->io_config.max_prefetch_queue / 4u,
            std::max(64u, pinned_workers * dyn_batch_max * 6u)  // Changed from 12 to 6
        );
        if (prefetch_stage_load(engine->pinned_stage) >= pinned_pressure_threshold) {
            return 1;
        }
    }
    return 0;
}

static moe_tier_t prefetch_admitted_tier(
    moe_pc_engine* engine,
    const moe_expert_prefetch_item_t& item,
    std::vector<std::pair<int, int>>& to_free
) {
    if (!engine) {
        return item.target_tier;
    }
    const bool waits_for_common_vram =
        engine->io_config.enable_common_raw_vram_pin &&
        engine->common_vram_reserved_bytes.load(std::memory_order_relaxed) > 0;
    if (item.target_tier == moe_TIER_VRAM &&
        waits_for_common_vram &&
        !engine->common_vram_upload_complete.load(std::memory_order_acquire)) {
        return moe_TIER_RAM;
    }

    moe_tier_t target = item.target_tier;
    if (!prefetch_is_router_selected(item) && prefetch_under_queue_pressure(engine, item)) {
        target = moe_TIER_RAM;
    }
    if (target == moe_TIER_VRAM) {
        uint64_t pending = engine->pending_vram_reservation.load(std::memory_order_relaxed);
        // BUGFIX 996: CAS backoff for high-contention admission ★★ PERFORMANCE
        uint32_t spin = 0;
        for (;;) {
            const uint64_t vram_budget = device_budget_bytes(engine);

            if (vram_budget == 0 || item.bytes > vram_budget) {
                target = moe_TIER_RAM;
                break;
            }
            // BUGFIX 899 follow-up: Remove device_allocated (used) double subtraction ★★
            if (pending > vram_budget ||
                item.bytes > vram_budget - pending) {
                target = moe_TIER_RAM;
                device_refresh_mem_info(engine, 1);
                break;
            }
            if (engine->pending_vram_reservation.compare_exchange_weak(
                    pending,
                    pending + item.bytes,
                    std::memory_order_relaxed,
                    std::memory_order_relaxed)) {
                break;
            }
            // Backoff: yield after 8 spins to reduce cache-line contention
            if (++spin > 8) { std::this_thread::yield(); spin = 0; }
        }
    }
    const uint64_t ram_budget = tier_budget(engine, moe_TIER_RAM);
    // BUGFIX 문제 1: Remove 85% threshold - use single 93% threshold ★★★
    // Problem: 85% and 93% dual thresholds create dead zone
    // io_worker proactive eviction at 75% already handles RAM pressure
    // 85% threshold triggers unnecessary eviction, causing latency spikes
    // Solution: Remove 85% branch, keep only 93% for DB routing
    // Impact: Cleaner admission logic, eviction delegated to background io_worker

    // BUGFIX Problem 6: Dynamic RAM threshold based on eviction target ★★ IMPORTANT
    // Problem: Fixed 93% threshold with 75% eviction target creates 18%p gap
    // Fast prefetch can fill gap before eviction catches up → sudden DB routing
    // Solution: Calculate threshold as eviction_target + 10%p for consistent headroom
    // Impact: Better pipeline balance, smoother RAM→DB transition
        // BUGFIX 980: Clamp eviction_pct to 85% max for admission headroom ★★★
        // Problem: ram_eviction_target_fill_pct ≥ 86 → admission_pct clamped to 95
        // → only 5-9%p headroom → eviction can't catch up → DB fallback thrashing
        // Solution: Cap eviction_pct at 85 so admission is always eviction + 10
        const uint64_t eviction_pct = std::min(
            engine->io_config.ram_eviction_target_fill_pct > 0
                ? engine->io_config.ram_eviction_target_fill_pct : 75ull,
            85ull);
        const uint64_t admission_pct = eviction_pct + 10ull;  // always ≤ 95
        const uint64_t ram_threshold = (ram_budget * admission_pct) / 100ull;

        uint64_t ram_pending = engine->pending_ram_reservation.load(std::memory_order_relaxed);
        // BUGFIX 996: CAS backoff for RAM admission ★★ PERFORMANCE
        uint32_t ram_spin = 0;
        for (;;) {
            const uint64_t ram_used_with_pending =
                tier_used(engine, moe_TIER_RAM) + ram_pending + item.bytes;
            if (ram_used_with_pending > ram_threshold) {
                target = moe_TIER_DB;
                break;
            }
            if (engine->pending_ram_reservation.compare_exchange_weak(
                    ram_pending, ram_pending + item.bytes,
                    std::memory_order_relaxed, std::memory_order_relaxed)) {
                break;
            }
            if (++ram_spin > 8) { std::this_thread::yield(); ram_spin = 0; }
        }
    }
    return target;
}

// BUGFIX 256: Fix underflow recovery with CAS loop ★★
// Problem: fetch_sub then fetch_add for recovery is not atomic
// If old_val < bytes, underflow already occurred. Another thread may have
// modified pending_vram_reservation between fetch_sub and fetch_add.
// Solution: Use CAS loop to atomically prevent underflow
static void prefetch_release_pending_vram_reservation(moe_pc_engine* engine, uint64_t bytes) {
    if (!engine || bytes == 0) {
        return;
    }
    uint64_t expected = engine->pending_vram_reservation.load(std::memory_order_relaxed);
    // BUGFIX 996: CAS backoff + underflow clamp-once ★★ PERFORMANCE
    // Problem: Underflow path (expected < bytes) re-enters CAS loop instead of
    // breaking. If another thread concurrently modifies the counter, the loop
    // retries indefinitely with bytes > expected.
    // Solution: On underflow, clamp bytes = expected and CAS once to 0.
    uint32_t rel_spin = 0;
    while (true) {
        if (expected < bytes) {
            // Underflow: clamp to 0 in one attempt, then break regardless
            engine->pending_vram_reservation.compare_exchange_strong(
                    expected, 0,
                    std::memory_order_relaxed,
                    std::memory_order_relaxed);
            break;  // Best-effort clamp — don't re-spin on underflow
        } else {
            // Normal case - subtract bytes
            if (engine->pending_vram_reservation.compare_exchange_weak(
                    expected, expected - bytes,
                    std::memory_order_relaxed,
                    std::memory_order_relaxed)) {
                break;
            }
            if (++rel_spin > 8) { std::this_thread::yield(); rel_spin = 0; }
        }
    }
}

// BUGFIX 256: Apply same fix to RAM reservation
static void prefetch_release_pending_ram_reservation(moe_pc_engine* engine, uint64_t bytes) {
    if (!engine || bytes == 0) {
        return;
    }
    uint64_t expected = engine->pending_ram_reservation.load(std::memory_order_relaxed);
    while (true) {
        if (expected < bytes) {
            // Underflow would occur - clamp to 0
            if (engine->pending_ram_reservation.compare_exchange_weak(
                    expected, 0,
                    std::memory_order_relaxed,
                    std::memory_order_relaxed)) {
                break;
            }
        } else {
            // Normal case - subtract bytes
            if (engine->pending_ram_reservation.compare_exchange_weak(
                    expected, expected - bytes,
                    std::memory_order_relaxed,
                    std::memory_order_relaxed)) {
                break;
            }
        }
    }
}
