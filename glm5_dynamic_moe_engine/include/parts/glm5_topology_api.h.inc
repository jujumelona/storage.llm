int glm5_pc_engine_set_topology_path(glm5_pc_engine_t* engine, const char* path);

int glm5_pc_engine_load_topology(glm5_pc_engine_t* engine, const char* path);

int glm5_pc_engine_save_topology(glm5_pc_engine_t* engine, const char* path);

int glm5_pc_engine_record_expert_transition(
    glm5_pc_engine_t* engine,
    int layer,
    int expert,
    int next_layer,
    int next_expert
);
