int moe_pc_engine_plan_moe_prefetch(
    moe_pc_engine_t* engine,
    int current_layer,
    const int* selected_experts,
    uint32_t selected_count,
    const int* next_layer_candidates,
    uint32_t next_count,
    moe_expert_prefetch_item_t* out_items,
    uint32_t max_items,
    uint32_t* out_item_count
) {
    if (!engine || current_layer < 0 || !out_item_count) {
        return 0;
    }

    std::vector<moe_expert_prefetch_item_t> items;
    // prefetch_plan_clear_seen/add_* lock the generation map internally.
    // Do not hold prefetch_plan_mutex around them; the helpers re-enter it.
    prefetch_plan_clear_seen(engine);
    // BUGFIX 577: Check reserve size for overflow ★
    // Problem: selected_count + next_count + topology_max_candidates can overflow
    // Solution: Check before reserve
    const size_t reserve_size = (size_t)selected_count + (size_t)next_count +
        (size_t)engine->io_config.topology_max_candidates;
    if (reserve_size > SIZE_MAX / sizeof(moe_expert_prefetch_item_t)) {
        *out_item_count = 0;
        return 0;
    }
    items.reserve(reserve_size);
    prefetch_plan_add_selected(engine, items, current_layer, selected_experts, selected_count);
    prefetch_plan_add_lookahead(engine, items, current_layer + 1, next_layer_candidates, next_count);
    prefetch_plan_add_topology(engine, items, current_layer, selected_experts, selected_count);
    prefetch_plan_sort_items(items);

    // BUGFIX 문제4: Trim prefetch plan to VRAM capacity ★★★
    // Problem: prefetch_plan_add_topology() adds topology_max_candidates items
    // without checking VRAM free space. When VRAM is full, GPU workers repeatedly
    // fail VRAM reservation and requeue, wasting CPU cycles.
    // Solution: After sorting by priority, calculate cumulative bytes and trim
    // items that exceed VRAM budget (with headroom). High-priority items remain.
    // Trade-off: Slightly reduces prefetch breadth, but proactive eviction worker
    // continuously frees VRAM, so no real throughput loss. Stabilizes token latency.
    {
        const uint64_t device_budget = device_budget_bytes(engine);
        const uint64_t device_allocated = engine->device_allocated_bytes.load(std::memory_order_relaxed);
        const uint64_t device_free = device_budget > device_allocated ? device_budget - device_allocated : 0;

        // Use headroom percentage (default 10%) to avoid filling VRAM completely
        const uint64_t headroom_pct = engine->io_config.device_memory_headroom_pct > 0
            ? engine->io_config.device_memory_headroom_pct : 10u;
        const uint64_t usable_free = (device_free * (100u - headroom_pct)) / 100u;

        uint64_t cumulative_bytes = 0;
        size_t trim_index = items.size();

        // BUGFIX 2-1: Use normal_bundle_max_bytes instead of max_tensor_stream_bytes * 3
        // Problem: max_tensor_stream_bytes is the largest single tensor streamed so far,
        // but gate/up/down projections can have different sizes. Multiplying by 3 overestimates
        // expert size, causing excessive trimming even when VRAM is sufficient.
        // Solution: Use normal_bundle_max_bytes from hardcode_summary, which is the actual
        // maximum bytes for a gate+up+down bundle, already aggregated during model loading.
        const moe_storage_constants_t* storage = moe_storage_constants();
        const uint64_t expert_bundle_bytes = storage && storage->normal_bundle_max_bytes
            ? storage->normal_bundle_max_bytes
            : 64ull * 1024 * 1024;  // Fallback: 64MB estimate

        for (size_t i = 0; i < items.size(); ++i) {
            if (cumulative_bytes + expert_bundle_bytes > usable_free) {
                trim_index = i;
                break;
            }
            cumulative_bytes += expert_bundle_bytes;
        }

        if (trim_index < items.size()) {
            items.resize(trim_index);
        }
    }

    const size_t item_count = items.size();
    *out_item_count = (uint32_t)item_count;
    if (out_items && max_items > 0) {
        // Fix 13: Use size_t for comparison to handle large item counts
        const size_t copy_count = (size_t)max_items < item_count ? (size_t)max_items : item_count;
        std::memcpy(out_items, items.data(), sizeof(moe_expert_prefetch_item_t) * copy_count);
    }
    // Return 1 if all items fit, 0 if partial copy (caller can check out_item_count)
    return (size_t)max_items >= item_count ? 1 : 0;
}
