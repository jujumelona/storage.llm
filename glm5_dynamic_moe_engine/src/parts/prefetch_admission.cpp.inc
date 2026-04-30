static uint32_t prefetch_stage_load(const glm5_stage_queue& stage) {
    return stage.depth.load(std::memory_order_relaxed) + stage.inflight.load(std::memory_order_relaxed);
}

static int prefetch_is_router_selected(const glm5_expert_prefetch_item_t& item) {
    return item.reason == GLM5_PREFETCH_REASON_ROUTER_SELECTED;
}

static int prefetch_under_queue_pressure(
    const glm5_pc_engine* engine,
    const glm5_expert_prefetch_item_t& item
) {
    if (!engine || engine->io_config.max_prefetch_queue == 0) {
        return 0;
    }
    const uint32_t max_queue = engine->io_config.max_prefetch_queue;
    if (prefetch_stage_load(engine->disk_stage) * 4u >= max_queue * 3u) {
        return 1;
    }
    if (item.target_tier == GLM5_TIER_VRAM &&
        (prefetch_stage_load(engine->pinned_stage) * 4u >= max_queue * 3u ||
         prefetch_stage_load(engine->gpu_stage) * 4u >= max_queue * 3u)) {
        return 1;
    }
    // BUG 5 FIX: RAM target should also check pinned_stage load
    // When VRAM targets congest pinned_stage, RAM targets can flood disk_stage
    // and push it to 100%, even though RAM targets don't use pinned_stage directly.
    // Reject RAM targets when pinned_stage is >75% to maintain pipeline balance.
    if (item.target_tier == GLM5_TIER_RAM &&
        prefetch_stage_load(engine->pinned_stage) * 4u >= max_queue * 3u) {
        return 1;
    }
    return 0;
}

static glm5_tier_t prefetch_admitted_tier(
    glm5_pc_engine* engine,
    const glm5_expert_prefetch_item_t& item,
    std::vector<std::pair<int, int>>& to_free,
    uint64_t extra_reserved = 0
) {
    if (!engine || prefetch_is_router_selected(item)) {
        return item.target_tier;
    }
    const bool waits_for_common_vram =
        engine->io_config.enable_common_raw_vram_pin &&
        engine->common_vram_reserved_bytes.load(std::memory_order_relaxed) > 0;
    if (item.target_tier == GLM5_TIER_VRAM &&
        waits_for_common_vram &&
        !engine->common_vram_upload_complete.load(std::memory_order_acquire)) {
        return GLM5_TIER_RAM;
    }

    glm5_tier_t target = item.target_tier;
    if (prefetch_under_queue_pressure(engine, item)) {
        target = GLM5_TIER_RAM;
    }
    const uint64_t ram_budget = tier_budget(engine, GLM5_TIER_RAM);
    // Critical Fix 4: Raise RAM threshold from 87% to 93% to reduce hot-path eviction
    // io_worker proactive eviction at 75% provides sufficient headroom.
    // Admission-triggered demote_cold_until causes token generation latency spikes.
    // By raising threshold, we delegate eviction to background io_worker and only
    // route to DB when RAM is critically full.
    if (target == GLM5_TIER_RAM && ram_budget > 0 &&
        tier_used(engine, GLM5_TIER_RAM) + extra_reserved + item.bytes > (ram_budget * 93ull) / 100ull) {
        // Route to DB instead of synchronous eviction
        target = GLM5_TIER_DB;
    }
    return target;
}
