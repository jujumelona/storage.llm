int glm5_pc_engine_wait_io(glm5_pc_engine_t* engine);

int glm5_pc_engine_prefetch_expert_triplet(
    glm5_pc_engine_t* engine,
    int layer,
    int expert,
    glm5_tier_t preferred_tier
);

int glm5_pc_engine_startup_warm_model(glm5_pc_engine_t* engine);

int glm5_pc_engine_prepare_moe_activation(glm5_pc_engine_t* engine, int layer);

int glm5_pc_engine_prefetch_moe_lookahead(
    glm5_pc_engine_t* engine,
    int current_layer,
    const int* selected_experts,
    uint32_t selected_count,
    const int* next_layer_candidates,
    uint32_t next_count
);

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
);

int glm5_pc_engine_execute_prefetch_plan(
    glm5_pc_engine_t* engine,
    const glm5_expert_prefetch_item_t* items,
    uint32_t item_count
);

int glm5_pc_engine_orchestrate_moe_step(
    glm5_pc_engine_t* engine,
    int current_layer,
    const int* selected_experts,
    uint32_t selected_count,
    const int* next_layer_candidates,
    uint32_t next_count,
    const glm5_runtime_orchestrator_state_t* state,
    glm5_runtime_orchestrator_plan_t* out_plan,
    glm5_expert_prefetch_item_t* out_items,
    uint32_t max_items,
    uint32_t* out_item_count
);
