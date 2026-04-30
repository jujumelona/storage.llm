int moe_pc_engine_set_topology_path(moe_pc_engine_t* engine, const char* path);

int moe_pc_engine_load_topology(moe_pc_engine_t* engine, const char* path);

int moe_pc_engine_save_topology(moe_pc_engine_t* engine, const char* path);

int moe_pc_engine_record_expert_transition(
    moe_pc_engine_t* engine,
    int layer,
    int expert,
    int next_layer,
    int next_expert
);
