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
    // Bug 3 Fix: Block VRAM admission if common tensors are not fully uploaded
    if (item.target_tier == GLM5_TIER_VRAM && !engine->common_vram_upload_complete.load(std::memory_order_acquire)) {
        return GLM5_TIER_RAM;
    }

    glm5_tier_t target = item.target_tier;
    if (prefetch_under_queue_pressure(engine, item)) {
        target = GLM5_TIER_RAM;
    }
    const uint64_t ram_budget = tier_budget(engine, GLM5_TIER_RAM);
    if (target == GLM5_TIER_RAM && ram_budget > 0 &&
        tier_used(engine, GLM5_TIER_RAM) + extra_reserved + item.bytes > (ram_budget * 87ull) / 100ull) {
        demote_cold_until(engine, GLM5_TIER_RAM, item.bytes, to_free);
        if (tier_used(engine, GLM5_TIER_RAM) + extra_reserved + item.bytes > (ram_budget * 87ull) / 100ull) {
            target = GLM5_TIER_DB;
        }
    }
    return target;
}
