int moe_pc_engine_wait_io(moe_pc_engine_t* engine);

int moe_pc_engine_prefetch_expert_triplet(
    moe_pc_engine_t* engine,
    int layer,
    int expert,
    moe_tier_t preferred_tier
);

int moe_pc_engine_startup_warm_model(moe_pc_engine_t* engine);

int moe_pc_engine_prepare_moe_activation(moe_pc_engine_t* engine, int layer);

int moe_pc_engine_prefetch_moe_lookahead(
    moe_pc_engine_t* engine,
    int current_layer,
    const int* selected_experts,
    uint32_t selected_count,
    const int* next_layer_candidates,
    uint32_t next_count
);

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
);

int moe_pc_engine_execute_prefetch_plan(
    moe_pc_engine_t* engine,
    const moe_expert_prefetch_item_t* items,
    uint32_t item_count
);

int moe_pc_engine_orchestrate_moe_step(
    moe_pc_engine_t* engine,
    int current_layer,
    const int* selected_experts,
    uint32_t selected_count,
    const int* next_layer_candidates,
    uint32_t next_count,
    const moe_runtime_orchestrator_state_t* state,
    moe_runtime_orchestrator_plan_t* out_plan,
    moe_expert_prefetch_item_t* out_items,
    uint32_t max_items,
    uint32_t* out_item_count
);
