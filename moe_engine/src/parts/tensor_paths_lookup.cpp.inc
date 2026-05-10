// Tensor record lookup and indexing

static size_t find_tensor_record_index(
    const moe_pc_engine* engine,
    int layer,
    int expert,
    moe_projection_t proj
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

static const moe_tensor_record* find_tensor_record(
    const moe_pc_engine* engine,
    int layer,
    int expert,
    moe_projection_t proj
) {
    const size_t index = find_tensor_record_index(engine, layer, expert, proj);
    if (index == tensor_slot_invalid()) {
        return nullptr;
    }
    // BUGFIX 175: Check index is within bounds before accessing vector
    if (index >= engine->tensors.size()) {
        return nullptr;
    }
    return &engine->tensors[index];
}
