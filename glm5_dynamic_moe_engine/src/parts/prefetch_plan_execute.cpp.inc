int glm5_pc_engine_execute_prefetch_plan(
    glm5_pc_engine_t* engine,
    const glm5_expert_prefetch_item_t* items,
    uint32_t item_count
) {
    if (!engine || !items || item_count == 0) {
        return 0;
    }
    std::vector<uint8_t> stream_inflight(item_count, 0);
    for (uint32_t i = 0; i < item_count; ++i) {
        std::lock_guard<std::mutex> lock(engine->map_mutex);
        const size_t slot = expert_dense_slot((int)items[i].layer, (int)items[i].expert);
        if (slot < engine->expert_states.size()) {
            const glm5_block_state& state = engine->expert_states[slot];
            stream_inflight[i] = state.active && state.tier == GLM5_TIER_STREAM ? 1u : 0u;
        }
    }

    // Problem 5 Fix: Split into two passes to reduce per-item mutex chaining.
    // Pass 1: Admission + eviction decisions (heap_mutex + map_mutex per item).
    // Pass 2: Enqueue all items (map_mutex once per item).
    // This prevents worst-case heap_mutex → (map_mutex × n evictions) → device_mutex → map_mutex
    // from happening item_count times in a tight loop.

    std::vector<glm5_tier_t> admitted_tiers(item_count, GLM5_TIER_DB);
    std::vector<std::vector<std::pair<int, int>>> all_to_free(item_count);

    // Pass 1: Admission and eviction decisions
    uint64_t batch_ram_reserved = 0;
    for (uint32_t i = 0; i < item_count; ++i) {
        if (stream_inflight[i]) {
            continue;
        }
        admitted_tiers[i] = prefetch_admitted_tier(engine, items[i], all_to_free[i], batch_ram_reserved);
        if (admitted_tiers[i] == GLM5_TIER_RAM) {
            batch_ram_reserved += items[i].bytes;
        }
    }

    std::vector<std::pair<int, int>> flat_to_free;
    for (auto& to_free : all_to_free) {
        flat_to_free.insert(flat_to_free.end(), to_free.begin(), to_free.end());
    }
    if (!flat_to_free.empty()) {
        device_free_experts_batched(engine, flat_to_free, 8);
    }

    // Re-check RAM-admitted speculative items after freeing victims. Reserve
    // with the tier counter itself; a stale remaining-bytes snapshot can race
    // with other workers and over-admit VRAM work.
    for (uint32_t i = 0; i < item_count; ++i) {
        if (admitted_tiers[i] == GLM5_TIER_RAM
            && items[i].target_tier == GLM5_TIER_VRAM
            && !prefetch_is_router_selected(items[i])) {
            if (try_add_tier_usage(engine, GLM5_TIER_VRAM, items[i].bytes,
                                   tier_budget(engine, GLM5_TIER_VRAM))) {
                admitted_tiers[i] = GLM5_TIER_VRAM;
            }
        }
    }

    // Pass 2: Enqueue all items
    int ok_count = 0;
    for (uint32_t i = 0; i < item_count; ++i) {
        if (stream_inflight[i]) {
            ++ok_count;
            continue;
        }
        const int queued = glm5_pc_engine_prefetch_expert_triplet_impl(
            engine,
            (int)items[i].layer,
            (int)items[i].expert,
            admitted_tiers[i],
            0,
            1
        );
        if (queued) {
            ++ok_count;
        } else if (items[i].bytes > 0) {
            // Bug 10: Re-check STREAM state before fallback to avoid tier corruption
            // Between snapshot and here, another IO worker may have started streaming this expert
            bool is_streaming = false;
            {
                std::lock_guard<std::mutex> lock(engine->map_mutex);
                const size_t slot = expert_dense_slot((int)items[i].layer, (int)items[i].expert);
                if (slot < engine->expert_states.size()) {
                    const glm5_block_state& state = engine->expert_states[slot];
                    is_streaming = state.active && state.tier == GLM5_TIER_STREAM;
                }
            }
            if (is_streaming) {
                ++ok_count;
            } else {
                ok_count += glm5_pc_engine_request_expert(
                    engine,
                    (int)items[i].layer,
                    (int)items[i].expert,
                    items[i].bytes,
                    admitted_tiers[i]
                ) ? 1 : 0;
            }
        }
    }
    if (ok_count > 0) {
        const uint32_t wake_count = std::min(static_cast<uint32_t>(ok_count), runtime_disk_workers(engine));
        for (uint32_t i = 0; i < wake_count; ++i) {
            engine->disk_stage.cv.notify_one();
        }
    }
    return ok_count > 0 ? 1 : 0;
}

int glm5_pc_engine_set_topology_path(glm5_pc_engine_t* engine, const char* path) {
    if (!engine) {
        return 0;
    }
    std::unique_lock<std::shared_mutex> lock(engine->topology_mutex);
    engine->topology_path = path ? path : "";
    return 1;
}

