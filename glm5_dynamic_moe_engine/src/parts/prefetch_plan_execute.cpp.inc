int glm5_pc_engine_execute_prefetch_plan(
    glm5_pc_engine_t* engine,
    const glm5_expert_prefetch_item_t* items,
    uint32_t item_count
) {
    if (!engine || !items || item_count == 0) {
        return 0;
    }
    std::vector<uint8_t> stream_inflight(item_count, 0);
    {
        std::lock_guard<std::mutex> lock(engine->map_mutex);
        for (uint32_t i = 0; i < item_count; ++i) {
            const size_t slot = expert_dense_slot((int)items[i].layer, (int)items[i].expert);
            if (slot < engine->expert_states.size()) {
                const glm5_block_state& state = engine->expert_states[slot];
                stream_inflight[i] = state.active && state.tier == GLM5_TIER_STREAM ? 1u : 0u;
            }
        }
    }
    int ok_count = 0;
    for (uint32_t i = 0; i < item_count; ++i) {
        if (stream_inflight[i]) {
            ++ok_count;
            continue;
        }
        const glm5_tier_t tier = prefetch_admitted_tier(engine, items[i]);
        const int queued = glm5_pc_engine_prefetch_expert_triplet_impl(
            engine,
            (int)items[i].layer,
            (int)items[i].expert,
            tier,
            0,
            1
        );
        if (queued) {
            ++ok_count;
        } else if (items[i].bytes > 0) {
            ok_count += glm5_pc_engine_request_expert(
                engine,
                (int)items[i].layer,
                (int)items[i].expert,
                items[i].bytes,
                tier
            ) ? 1 : 0;
        }
    }
    if (ok_count > 0) {
        engine->disk_stage.cv.notify_all();
        engine->pinned_stage.cv.notify_all();
        engine->gpu_stage.cv.notify_all();
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

