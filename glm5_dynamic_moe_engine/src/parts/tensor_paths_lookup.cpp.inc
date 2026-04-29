// Tensor record lookup and indexing

static size_t find_tensor_record_index(
    const glm5_pc_engine* engine,
    int layer,
    int expert,
    glm5_projection_t proj
) {
    if (!engine) {
        return tensor_slot_invalid();
    }
    const size_t slot_index = tensor_slots_get(engine, layer, expert, proj);
    if (slot_index != tensor_slot_invalid() && slot_index < engine->tensors.size()) {
        return slot_index;
    }
    return tensor_slot_invalid();
}

static const glm5_tensor_record* find_tensor_record(
    const glm5_pc_engine* engine,
    int layer,
    int expert,
    glm5_projection_t proj
) {
    const size_t index = find_tensor_record_index(engine, layer, expert, proj);
    if (index == tensor_slot_invalid()) {
        return nullptr;
    }
    return &engine->tensors[index];
}
