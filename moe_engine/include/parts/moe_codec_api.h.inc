int moe_pc_engine_set_model_root(
    moe_pc_engine_t* engine,
    const char* model_root
);

int moe_pc_engine_load_codec_table(
    moe_pc_engine_t* engine,
    const char* table_path,
    const char* model_root,
    const char* scale4_path
);

int moe_pc_engine_prefetch_common_raw(moe_pc_engine_t* engine);

uint64_t moe_pc_engine_tensor_count(const moe_pc_engine_t* engine);

int moe_pc_engine_find_tensor(
    const moe_pc_engine_t* engine,
    int layer,
    int expert,
    moe_projection_t proj,
    moe_tensor_info_t* out_info
);
