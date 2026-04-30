static uint64_t orchestrator_expert_bytes(const moe_pc_engine* engine, int layer, int expert) {
    return expert_triplet_or_hardcoded_bytes(engine, layer, expert);
}

static void orchestrator_append_item(
    moe_pc_engine* engine,
    moe_expert_prefetch_item_t* items,
    uint32_t max_items,
    uint32_t* count,
    int layer,
    int expert,
    moe_tier_t tier,
    moe_prefetch_reason_t reason,
    float priority
) {
    if (!items || !count || *count >= max_items || layer < 0 || expert < 0) {
        return;
    }
    if (prefetch_plan_has_item(engine, layer, expert)) {
        return;
    }
    const uint64_t bytes = orchestrator_expert_bytes(engine, layer, expert);
    if (bytes == 0) {
        return;
    }
    moe_expert_prefetch_item_t item{};
    item.layer = (uint32_t)layer;
    item.expert = (uint32_t)expert;
    item.bytes = bytes;
    item.target_tier = tier;
    item.reason = reason;
    item.priority = priority;
    items[(*count)++] = item;
    prefetch_plan_mark_seen(engine, layer, expert);
}

static void orchestrator_append_topology(
    moe_pc_engine* engine,
    int layer,
    const int* selected,
    uint32_t selected_count,
    moe_tier_t tier,
    moe_expert_prefetch_item_t* items,
    uint32_t max_items,
    uint32_t* count
) {
    if (!engine || !selected || !items || !count) {
        return;
    }
    static thread_local std::vector<std::tuple<uint64_t, int, int>> candidates;
    prefetch_plan_topology_candidates(engine, layer, selected, selected_count, &candidates);
    prefetch_plan_sort_topology(candidates);
    uint32_t added = 0;
    for (const auto& candidate : candidates) {
        const uint32_t before = *count;
        orchestrator_append_item(engine, items, max_items, count, std::get<1>(candidate),
            std::get<2>(candidate), tier, moe_PREFETCH_REASON_TOPOLOGY, 50.0f + (float)std::get<0>(candidate));
        added += *count != before ? 1u : 0u;
        if (added >= engine->io_config.topology_max_candidates) {
            return;
        }
    }
}
