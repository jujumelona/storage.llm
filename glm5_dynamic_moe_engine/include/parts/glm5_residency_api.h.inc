glm5_kv_mode_t glm5_pc_engine_get_kv_mode(const glm5_pc_engine_t* engine);

int glm5_pc_engine_set_kv_mode(glm5_pc_engine_t* engine, glm5_kv_mode_t mode);

int glm5_pc_engine_request_expert(
    glm5_pc_engine_t* engine,
    int layer,
    int expert,
    uint64_t bytes,
    glm5_tier_t preferred_tier
);

int glm5_pc_engine_get_stats(const glm5_pc_engine_t* engine, glm5_pc_engine_stats_t* out_stats);

int glm5_pc_engine_configure_io(glm5_pc_engine_t* engine, const glm5_io_config_t* config);

int glm5_pc_engine_get_io_stats(const glm5_pc_engine_t* engine, glm5_io_stats_t* out_stats);
