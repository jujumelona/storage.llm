static uint32_t prefetch_stage_load(const moe_stage_queue& stage) {
    // Intentional approximation: depth and inflight are read separately without lock.
    // TOCTOU race possible but acceptable for admission control heuristic.
    // Actual value may be 吏?-2 from true sum, threshold checks account for this.
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
    if (moe_storage_runtime_cpu_ram_path(engine)) {
        return 0;
    }
    if (item.target_tier == moe_TIER_VRAM &&
        ((engine->pinned_stage_depth_cap > 0 &&
          prefetch_stage_load(engine->pinned_stage) * 4u >= engine->pinned_stage_depth_cap * 3u) ||
         (engine->gpu_stage_depth_cap > 0 &&
          prefetch_stage_load(engine->gpu_stage) * 4u >= engine->gpu_stage_depth_cap * 3u))) { return 1; }
    // BUG 5 FIX: RAM target should also check pinned_stage load
    // When VRAM targets congest pinned_stage, RAM targets can flood disk_stage
    // and push it to 100%, even though RAM targets don't use pinned_stage directly.
    // Reject RAM targets when pinned_stage is >75% to maintain pipeline balance.
    // BUGFIX 25: Use realistic threshold based on worker count ??몄겲
    // Old: max_queue * 75% (9216) ??never reached with 1-2 pinned workers
    // New: pinned_workers * batch_max * multiplier ??realistic congestion detection
    // BUGFIX Problem 5: Reduce multiplier from 12 to 6 ??몄겲 IMPORTANT
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
    std::vector<std::pair<int, int>>& to_free,
    vram_reservation_token* out_vram_token = nullptr
) {
    if (out_vram_token) {
        *out_vram_token = vram_reservation_token{};
    }
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
    if (moe_storage_runtime_cpu_ram_path(engine) && target == moe_TIER_VRAM) {
        target = moe_TIER_RAM;
    } else if (!prefetch_is_router_selected(item) && prefetch_under_queue_pressure(engine, item)) {
        target = moe_TIER_RAM;
    }
    if (target == moe_TIER_VRAM) {
        vram_reservation_token token{};
        if (!device_vram_try_reserve_token(engine, item.bytes, moe_DEVICE_POOL_EXPERT, &token)) {
            target = moe_TIER_RAM;
            device_refresh_mem_info(engine, 1);
        } else if (out_vram_token) {
            *out_vram_token = token;
        } else {
            device_vram_release_reservation_token(engine, token);
            target = moe_TIER_RAM;
        }
    }
    if (target == moe_TIER_RAM) {
    // BUGFIX Problem 5: Reserve RAM for KV cache growth trajectory ??몄겲 IMPORTANT
    // Problem: RAM admission only checks current tier_used(RAM), ignoring KV growth
    // ??Long context requests cause KV to fill RAM ??demote_cold_until blocks hot path
    // Solution: Reserve max_seq_len ??kv_bytes_per_token from RAM budget
    // ??Expert admission threshold accounts for worst-case KV growth
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
    // BUGFIX ?얜챷??1: Remove 85% threshold - use single 93% threshold ??몄겲??    // Problem: 85% and 93% dual thresholds create dead zone
    // io_worker proactive eviction at 75% already handles RAM pressure
    // 85% threshold triggers unnecessary eviction, causing latency spikes
    // Solution: Remove 85% branch, keep only 93% for DB routing
    // Impact: Cleaner admission logic, eviction delegated to background io_worker

    // BUGFIX Problem 6: Dynamic RAM threshold based on eviction target ??몄겲 IMPORTANT
    // Problem: Fixed 93% threshold with 75% eviction target creates 18%p gap
    // Fast prefetch can fill gap before eviction catches up ??sudden DB routing
    // Solution: Calculate threshold as eviction_target + 10%p for consistent headroom
    // Impact: Better pipeline balance, smoother RAM?臾쵢 transition
        // BUGFIX 980: Clamp eviction_pct to 85% max for admission headroom ??몄겲??        // Problem: ram_eviction_target_fill_pct ??86 ??admission_pct clamped to 95
        // ??only 5-9%p headroom ??eviction can't catch up ??DB fallback thrashing
        // Solution: Cap eviction_pct at 85 so admission is always eviction + 10
        const uint64_t eviction_pct = std::min(
            engine->io_config.ram_eviction_target_fill_pct > 0
                ? engine->io_config.ram_eviction_target_fill_pct : 75ull,
            85ull);
        const uint64_t admission_pct = eviction_pct + 10ull;  // always ??95
        const uint64_t ram_threshold = (ram_budget * admission_pct) / 100ull;

        uint64_t ram_pending = engine->pending_ram_reservation.load(std::memory_order_relaxed);
        // BUGFIX 996: CAS backoff for RAM admission ??몄겲 PERFORMANCE
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
