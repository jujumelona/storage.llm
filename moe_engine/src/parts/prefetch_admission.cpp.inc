static uint32_t prefetch_stage_load(const moe_stage_queue& stage) {
    return stage.depth.load(std::memory_order_relaxed) + stage.inflight.load(std::memory_order_relaxed);
}

static int prefetch_is_router_selected(const moe_expert_prefetch_item_t& item) {
    return item.reason == moe_PREFETCH_REASON_ROUTER_SELECTED;
}

static int prefetch_under_queue_pressure(
    const moe_pc_engine* engine,
    const moe_expert_prefetch_item_t& item
) {
    if (!engine || engine->io_config.max_prefetch_queue == 0) {
        return 0;
    }
    const uint32_t max_queue = engine->io_config.max_prefetch_queue;
    if (prefetch_stage_load(engine->disk_stage) * 4u >= max_queue * 3u) {
        return 1;
    }
    if (item.target_tier == moe_TIER_VRAM &&
        (prefetch_stage_load(engine->pinned_stage) * 4u >= max_queue * 3u ||
         prefetch_stage_load(engine->gpu_stage) * 4u >= max_queue * 3u)) {
        return 1;
    }
    // BUG 5 FIX: RAM target should also check pinned_stage load
    // When VRAM targets congest pinned_stage, RAM targets can flood disk_stage
    // and push it to 100%, even though RAM targets don't use pinned_stage directly.
    // Reject RAM targets when pinned_stage is >75% to maintain pipeline balance.
    // BUGFIX 25: Use realistic threshold based on worker count ?�★
    // Old: max_queue * 75% (9216) ??never reached with 1-2 pinned workers
    // New: pinned_workers * batch_max * 4 ??realistic congestion detection
    if (item.target_tier == moe_TIER_RAM) {
        const uint32_t pinned_workers = runtime_pinned_workers(engine);
        const uint32_t dyn_batch_max = moe_gpu_stage_batch_max(engine);
        const uint32_t pinned_pressure_threshold = std::max(32u, pinned_workers * dyn_batch_max * 8u);
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
    if (!engine || prefetch_is_router_selected(item)) {
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
    if (prefetch_under_queue_pressure(engine, item)) {
        target = moe_TIER_RAM;
    }
    const uint64_t ram_budget = tier_budget(engine, moe_TIER_RAM);
    // Critical Fix 4: Raise RAM threshold from 87% to 93% to reduce hot-path eviction
    // io_worker proactive eviction at 75% provides sufficient headroom.
    // Admission-triggered demote_cold_until causes token generation latency spikes.
    // By raising threshold, we delegate eviction to background io_worker and only
    // route to DB when RAM is critically full.
    if (target == moe_TIER_RAM && ram_budget > 0) {
        // BUGFIX 99: Precompute percentage threshold
        const uint64_t ram_threshold_93 = (ram_budget * 93ull) / 100ull;
        if (tier_used(engine, moe_TIER_RAM) + engine->pending_ram_reservation.load(std::memory_order_relaxed) + item.bytes > ram_threshold_93) {
            // Route to DB instead of synchronous eviction
            target = moe_TIER_DB;
        } else {
            engine->pending_ram_reservation.fetch_add(item.bytes, std::memory_order_relaxed);
        }
    }
    return target;
}

static void prefetch_release_pending_ram_reservation(moe_pc_engine* engine, uint64_t bytes) {
    if (!engine || bytes == 0) {
        return;
    }
    const uint64_t old_val = engine->pending_ram_reservation.fetch_sub(bytes, std::memory_order_relaxed);
    if (old_val < bytes) {
        engine->pending_ram_reservation.store(0, std::memory_order_relaxed);
    }
}
