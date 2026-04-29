int glm5_pc_engine_set_model_root(
    glm5_pc_engine_t* engine,
    const char* model_root
);

int glm5_pc_engine_load_codec_table(
    glm5_pc_engine_t* engine,
    const char* table_path,
    const char* model_root,
    const char* scale4_path
);

int glm5_pc_engine_prefetch_common_raw(glm5_pc_engine_t* engine);

uint64_t glm5_pc_engine_tensor_count(const glm5_pc_engine_t* engine);

int glm5_pc_engine_find_tensor(
    const glm5_pc_engine_t* engine,
    int layer,
    int expert,
    glm5_projection_t proj,
    glm5_tensor_info_t* out_info
);
