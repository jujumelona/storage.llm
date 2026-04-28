int glm5_pc_engine_plan_moe_prefetch(
    glm5_pc_engine_t* engine,
    int current_layer,
    const int* selected_experts,
    uint32_t selected_count,
    const int* next_layer_candidates,
    uint32_t next_count,
    glm5_expert_prefetch_item_t* out_items,
    uint32_t max_items,
    uint32_t* out_item_count
) {
    if (!engine || current_layer < 0 || !out_item_count) {
        return 0;
    }

    std::vector<glm5_expert_prefetch_item_t> items;
    prefetch_plan_clear_seen(engine);
    items.reserve((size_t)selected_count + (size_t)next_count +
        (size_t)engine->io_config.topology_max_candidates);
    prefetch_plan_add_selected(engine, items, current_layer, selected_experts, selected_count);
    prefetch_plan_add_lookahead(engine, items, current_layer + 1, next_layer_candidates, next_count);
    prefetch_plan_add_topology(engine, items, current_layer, selected_experts, selected_count);
    prefetch_plan_sort_items(items);

    const size_t item_count = items.size();
    *out_item_count = (uint32_t)item_count;
    if (out_items && max_items > 0) {
        // Fix 13: Use size_t for comparison to handle large item counts
        const size_t copy_count = (size_t)max_items < item_count ? (size_t)max_items : item_count;
        std::memcpy(out_items, items.data(), sizeof(glm5_expert_prefetch_item_t) * copy_count);
    }
    // Return 1 if all items fit, 0 if partial copy (caller can check out_item_count)
    return (size_t)max_items >= item_count ? 1 : 0;
}
