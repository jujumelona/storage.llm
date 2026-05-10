uint64_t moe_pc_engine_tensor_count(const moe_pc_engine_t* engine) {
    if (!engine) {
        return 0;
    }
    if (!engine->tensors.empty()) {
        return (uint64_t)engine->tensors.size();
    }
    return engine->offload_gguf_tensor_count.load(std::memory_order_relaxed);
}

int moe_pc_engine_find_tensor(
    const moe_pc_engine_t* engine,
    int layer,
    int expert,
    moe_projection_t proj,
    moe_tensor_info_t* out_info
) {
    const moe_tensor_record* rec = find_tensor_record(engine, layer, expert, proj);
    if (!rec || !out_info) {
        return 0;
    }
    *out_info = rec->info;
    return 1;
}
