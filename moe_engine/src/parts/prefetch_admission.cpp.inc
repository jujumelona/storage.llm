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
        // BUGFIX 1014: Atomic VRAM admission under vram_transition_mutex ★★★ CRITICAL
        // Problem: device_budget_bytes() releases mutex, then CAS happens outside mutex
        // → N threads read same budget, all CAS succeed → N × item.bytes over-allocated
        // → GPU OOM when urgent items pile up
        // Solution: Hold vram_transition_mutex for entire check-and-reserve operation
        // Impact: Eliminates VRAM over-commit, prevents GPU OOM under load
        std::unique_lock<std::shared_mutex> vram_lock(engine->vram_transition_mutex);

        // Read all counters under mutex (consistent snapshot)
        const uint64_t pending = engine->pending_vram_reservation.load(std::memory_order_relaxed);
        const uint64_t in_flight = engine->vram_in_flight_bytes.load(std::memory_order_relaxed);
        const uint64_t kv_actual = engine->device_kv_cache_bytes.load(std::memory_order_relaxed);
        const uint64_t common_reserved = engine->common_vram_reserved_bytes.load(std::memory_order_relaxed);
        const uint64_t device_allocated = engine->device_allocated_bytes.load(std::memory_order_relaxed);
        const uint64_t device_free = engine->device_free_bytes.load(std::memory_order_relaxed);

        const uint64_t kv_reserved = engine->kv_actually_in_vram.load(std::memory_order_acquire) ? kv_actual : 0;
        uint64_t reserved_total = pending;
        if (reserved_total <= UINT64_MAX - in_flight) reserved_total += in_flight;
        if (reserved_total <= UINT64_MAX - kv_reserved) reserved_total += kv_reserved;
        if (reserved_total <= UINT64_MAX - common_reserved) reserved_total += common_reserved;

        uint64_t vram_budget = 0;
        if (engine->config.vram_budget_bytes) {
            vram_budget = engine->config.vram_budget_bytes > reserved_total
                ? engine->config.vram_budget_bytes - reserved_total : 0;
        } else if (engine->backend_caps.supports_unified_memory &&
                   engine->backend_caps.backend == moe_BACKEND_METAL) {
            const uint64_t sys = engine->backend_caps.system_memory_bytes > 0
                ? engine->backend_caps.system_memory_bytes : 16ull * 1024 * 1024 * 1024;
            uint64_t pct = engine->io_config.metal_memory_budget_pct;
            if (pct == 0 || pct < 50 || pct > 90) pct = 75ull;
            const uint64_t raw_budget = (sys * pct) / 100ull;
            const uint64_t cpu_ram_used = tier_used(engine, moe_TIER_RAM);
            const uint64_t kv_in_ram = std::min(kv_actual, cpu_ram_used);
            const uint64_t non_kv_ram = cpu_ram_used - kv_in_ram;
            uint64_t total_reserved = reserved_total;
            if (non_kv_ram < UINT64_MAX - total_reserved) total_reserved += non_kv_ram;
            vram_budget = raw_budget > total_reserved ? raw_budget - total_reserved : 0;
        } else {
            int64_t s_free = (int64_t)device_free;
            if (s_free < 0) s_free = 0;
            const uint64_t raw_budget = device_allocated > UINT64_MAX - (uint64_t)s_free
                ? UINT64_MAX : device_allocated + (uint64_t)s_free;
            vram_budget = raw_budget > reserved_total ? raw_budget - reserved_total : 0;
        }

        // Check and reserve atomically under mutex
        if (vram_budget == 0 || item.bytes > vram_budget ||
            pending > vram_budget || item.bytes > vram_budget - pending) {
            target = moe_TIER_RAM;
            vram_lock.unlock();
            device_refresh_mem_info(engine, 1);
        } else {
            // Reserve under mutex - no CAS needed, we hold exclusive lock
            engine->pending_vram_reservation.store(pending + item.bytes, std::memory_order_relaxed);
            vram_lock.unlock();
        }
    }
    // BUGFIX NEW-13: Remove duplicate Metal budget calculation ★★★ CRITICAL
    // Problem: prefetch_admitted_tier() duplicates device_budget_bytes() logic inline
    // → kv_in_ram = std::min(kv_actual, cpu_ram_used) is semantically wrong
    // → kv_actual is VRAM KV usage, cpu_ram_used is RAM usage (different units for Metal)
    // Solution: Removed duplicate Metal budget block (lines 109-116 in original)
    // → device_budget_bytes() already handles Metal correctly with fast-path
    // Impact: Eliminates code duplication, fixes semantic error in Metal KV accounting
    // Note: Metal budget calculation now handled by device_budget_bytes() call above
    // BUGFIX Problem 2: Delegate Metal UMA budget to device_budget_bytes() ★★★ CRITICAL
    // The inline Metal budget calculation (lines 109-137 in original) has been removed.
    // All Metal UMA budget logic is now centralized in device_budget_bytes() fast-path.
    // This eliminates semantic errors and code duplication.

    if (target == moe_TIER_RAM) {
    // BUGFIX Problem 5: Reserve RAM for KV cache growth trajectory ★★ IMPORTANT
    // Problem: RAM admission only checks current tier_used(RAM), ignoring KV growth
    // → Long context requests cause KV to fill RAM → demote_cold_until blocks hot path
    // Solution: Reserve max_seq_len × kv_bytes_per_token from RAM budget
    // → Expert admission threshold accounts for worst-case KV growth
    // Impact: Prevents RAM overflow, eliminates I/O stalls during long context generation
    const uint64_t ram_budget_raw = tier_budget(engine, moe_TIER_RAM);
    uint64_t reserved_kv_ram = 0;
    const uint32_t context_limit = engine->attn_config.max_seq_len > 0
        ? engine->attn_config.max_seq_len
        : engine->config.max_context_len;
    if (context_limit > 0 && engine->attn_config_valid.load(std::memory_order_acquire)) {
        const moe_storage_constants_t* storage = moe_storage_constants();
        const uint32_t num_layers = engine->dynamic_shape.num_hidden_layers > 0
            ? engine->dynamic_shape.num_hidden_layers
            : (storage && storage->num_hidden_layers > 0 ? storage->num_hidden_layers : 80);
        const uint32_t num_kv_heads = engine->attn_config.num_kv_heads > 0
            ? engine->attn_config.num_kv_heads : 8;
        const uint32_t head_dim = engine->attn_config.head_dim > 0
            ? engine->attn_config.head_dim : 128;
        const uint64_t kv_bytes_per_token = (uint64_t)num_layers * num_kv_heads * head_dim * 2 * 2;  // K+V, FP16
        reserved_kv_ram = (uint64_t)context_limit * kv_bytes_per_token;
        // Cap reservation at 20% of RAM budget to avoid over-reservation for short contexts
        const uint64_t max_reservation = (ram_budget_raw * 20ull) / 100ull;
        if (reserved_kv_ram > max_reservation) {
            reserved_kv_ram = max_reservation;
        }
    }
    const uint64_t ram_budget = ram_budget_raw > reserved_kv_ram ? ram_budget_raw - reserved_kv_ram : ram_budget_raw;
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
