static int glm5_pc_engine_prefetch_expert_triplet_impl(
    glm5_pc_engine_t* engine,
    int layer,
    int expert,
    glm5_tier_t preferred_tier,
    int notify_workers,
    int stream_state_prechecked = 0
) {
    if (!engine || layer < 0 || expert < 0) {
        return 0;
    }
    const uint64_t bytes = expert_triplet_or_hardcoded_bytes(engine, layer, expert);
    if (bytes == 0) {
        return 0;
    }
    const int requested = glm5_pc_engine_request_expert(engine, layer, expert, bytes, preferred_tier);
    if (!requested) {
        return 0;
    }
    if (preferred_tier == GLM5_TIER_DB || preferred_tier == GLM5_TIER_STREAM) {
        return requested;
    }

    const glm5_projection_t projs[3] = {GLM5_PROJ_GATE, GLM5_PROJ_UP, GLM5_PROJ_DOWN};
    int queued = 0;
    for (auto proj : projs) {
        queued += enqueue_tensor_prefetch(engine, layer, expert, proj, preferred_tier, stream_state_prechecked);
    }
    if (notify_workers) {
        for (int i = 0; i < queued; ++i) {
            engine->disk_stage.cv.notify_one();
        }
    }
    return queued > 0 ? 1 : requested;
}

int glm5_pc_engine_prefetch_expert_triplet(
    glm5_pc_engine_t* engine,
    int layer,
    int expert,
    glm5_tier_t preferred_tier
) {
    return glm5_pc_engine_prefetch_expert_triplet_impl(
        engine,
        layer,
        expert,
        preferred_tier,
        1
    );
}

static int prefetch_expert_batch_raw(
    glm5_pc_engine_t* engine,
    int layer,
    const int* experts,
    uint32_t expert_count,
    glm5_tier_t preferred_tier
) {
    if (!engine || !experts || expert_count == 0) {
        return 0;
    }
    int ok_count = 0;
    for (uint32_t i = 0; i < expert_count; ++i) {
        ok_count += glm5_pc_engine_prefetch_expert_triplet(engine, layer, experts[i], preferred_tier) ? 1 : 0;
    }
    return ok_count > 0 ? 1 : 0;
}

int glm5_pc_engine_prefetch_moe_lookahead(
    glm5_pc_engine_t* engine,
    int current_layer,
    const int* selected_experts,
    uint32_t selected_count,
    const int* next_layer_candidates,
    uint32_t next_count
) {
    if (!engine || current_layer < 0) {
        return 0;
    }
    prefetch_plan_clear_seen(engine);

    int ok_count = 0;
    release_moe_activation_raw(engine);
    prefetch_moe_shared_compute_raw(engine, current_layer);
    if (selected_experts && selected_count > 0) {
        ok_count += prefetch_expert_batch_raw(
            engine,
            current_layer,
            selected_experts,
            selected_count,
            vram_expert_budget_bytes(engine) > 0 ? GLM5_TIER_VRAM : GLM5_TIER_RAM
        ) ? 1 : 0;
    }
    if (engine->io_config.enable_lookahead_prefetch && next_layer_candidates && next_count > 0) {
        ok_count += prefetch_expert_batch_raw(
            engine,
            current_layer + 1,
            next_layer_candidates,
            next_count,
            GLM5_TIER_RAM
        ) ? 1 : 0;
        prefetch_moe_activation_raw(engine, current_layer + 1);
    }
    return ok_count > 0 ? 1 : 0;
}
