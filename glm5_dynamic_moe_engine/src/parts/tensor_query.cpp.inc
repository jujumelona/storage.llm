uint64_t glm5_pc_engine_tensor_count(const glm5_pc_engine_t* engine) {
    return engine ? (uint64_t)engine->tensors.size() : 0;
}

int glm5_pc_engine_find_tensor(
    const glm5_pc_engine_t* engine,
    int layer,
    int expert,
    glm5_projection_t proj,
    glm5_tensor_info_t* out_info
) {
    const glm5_tensor_record* rec = find_tensor_record(engine, layer, expert, proj);
    if (!rec || !out_info) {
        return 0;
    }
    *out_info = rec->info;
    return 1;
}
