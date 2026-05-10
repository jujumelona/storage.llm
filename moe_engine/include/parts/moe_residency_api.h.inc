moe_kv_mode_t moe_pc_engine_get_kv_mode(const moe_pc_engine_t* engine);

int moe_pc_engine_set_kv_mode(moe_pc_engine_t* engine, moe_kv_mode_t mode);

int moe_pc_engine_request_expert(
    moe_pc_engine_t* engine,
    int layer,
    int expert,
    uint64_t bytes,
    moe_tier_t preferred_tier
);

int moe_pc_engine_get_stats(const moe_pc_engine_t* engine, moe_pc_engine_stats_t* out_stats);

int moe_pc_engine_configure_io(moe_pc_engine_t* engine, const moe_io_config_t* config);

int moe_pc_engine_get_io_stats(const moe_pc_engine_t* engine, moe_io_stats_t* out_stats);
