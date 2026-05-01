static size_t tensor_slot_invalid() {
    return SIZE_MAX;
}

static size_t tensor_slot_count() {
    const moe_storage_constants_t* storage = moe_storage_constants();
    if (!storage || !storage->num_hidden_layers || !storage->experts_per_moe_layer ||
        !storage->expert_projection_count) {
        return 79ull * 256ull * 3ull;
    }
    return (size_t)storage->num_hidden_layers *
        (size_t)storage->experts_per_moe_layer *
        (size_t)storage->expert_projection_count;
}

static size_t tensor_record_capacity() {
    const moe_storage_constants_t* storage = moe_storage_constants();
    if (!storage || !storage->total_expert_count || !storage->expert_projection_count) {
        return tensor_slot_count();
    }
    return (size_t)storage->total_expert_count * (size_t)storage->expert_projection_count;
}

static size_t expert_slot_count() {
    const moe_storage_constants_t* storage = moe_storage_constants();
    if (!storage || !storage->num_hidden_layers || !storage->experts_per_moe_layer) {
        return 79ull * 256ull;
    }
    return (size_t)storage->num_hidden_layers * (size_t)storage->experts_per_moe_layer;
}

static size_t tensor_dense_slot(int layer, int expert, moe_projection_t proj) {
    const moe_storage_constants_t* storage = moe_storage_constants();
    const uint32_t layers = storage && storage->num_hidden_layers ? storage->num_hidden_layers : 79u;
    const uint32_t experts = storage && storage->experts_per_moe_layer ? storage->experts_per_moe_layer : 256u;
    const uint32_t projs = storage && storage->expert_projection_count ? storage->expert_projection_count : 3u;
    if (layer < 0 || (uint32_t)layer >= layers || expert < 0 || (uint32_t)expert >= experts ||
        (uint32_t)proj >= projs) {
        return tensor_slot_invalid();
    }
    return ((size_t)layer * (size_t)experts + (size_t)expert) * (size_t)projs + (size_t)proj;
}

static size_t expert_dense_slot(int layer, int expert) {
    const moe_storage_constants_t* storage = moe_storage_constants();
    const uint32_t layers = storage && storage->num_hidden_layers ? storage->num_hidden_layers : 79u;
    const uint32_t experts = storage && storage->experts_per_moe_layer ? storage->experts_per_moe_layer : 256u;
    if (layer < 0 || (uint32_t)layer >= layers || expert < 0 || (uint32_t)expert >= experts) {
        return tensor_slot_invalid();
    }
    return (size_t)layer * (size_t)experts + (size_t)expert;
}

static uint32_t expert_dense_slot_width() {
    const moe_storage_constants_t* storage = moe_storage_constants();
    return storage && storage->experts_per_moe_layer ? storage->experts_per_moe_layer : 256u;
}

static void expert_dense_slot_decode(size_t slot, int* layer, int* expert) {
    const uint32_t experts = expert_dense_slot_width();
    if (layer) {
        *layer = experts ? (int)(slot / experts) : 0;
    }
    if (expert) {
        *expert = experts ? (int)(slot % experts) : 0;
    }
}

static void tensor_slots_reset(moe_pc_engine* engine) {
    if (!engine) {
        return;
    }
    engine->tensor_slots.assign(tensor_slot_count(), tensor_slot_invalid());
    engine->expert_bytes.assign(expert_slot_count(), 0);
    engine->expert_runtime_priorities.assign(expert_slot_count(), 0);
    engine->expert_prefetch_priorities.assign(expert_slot_count(), 0);
    engine->expert_residency_hints.assign(expert_slot_count(), moe_JUJU_RESIDENCY_AUTO);
}

static void tensor_slots_set(moe_pc_engine* engine, int layer, int expert, moe_projection_t proj, size_t index) {
    const size_t slot = tensor_dense_slot(layer, expert, proj);
    if (engine && slot != tensor_slot_invalid() && slot < engine->tensor_slots.size()) {
        engine->tensor_slots[slot] = index;
    }
}

static size_t tensor_slots_get(const moe_pc_engine* engine, int layer, int expert, moe_projection_t proj) {
    const size_t slot = tensor_dense_slot(layer, expert, proj);
    if (!engine || slot == tensor_slot_invalid() || slot >= engine->tensor_slots.size()) {
        return tensor_slot_invalid();
    }
    return engine->tensor_slots[slot];
}

static void expert_bytes_add(moe_pc_engine* engine, int layer, int expert, uint64_t bytes) {
    const size_t slot = expert_dense_slot(layer, expert);
    if (engine && slot != tensor_slot_invalid() && slot < engine->expert_bytes.size()) {
        engine->expert_bytes[slot] += bytes;
    }
}

static void expert_runtime_hints_update(
    moe_pc_engine* engine,
    int layer,
    int expert,
    uint16_t runtime_priority,
    uint16_t prefetch_priority,
    uint8_t residency_hint
) {
    const size_t slot = expert_dense_slot(layer, expert);
    if (!engine || slot == tensor_slot_invalid()) {
        return;
    }
    if (slot < engine->expert_runtime_priorities.size()) {
        engine->expert_runtime_priorities[slot] =
            std::max<uint16_t>(engine->expert_runtime_priorities[slot], runtime_priority);
    }
    if (slot < engine->expert_prefetch_priorities.size()) {
        engine->expert_prefetch_priorities[slot] =
            std::max<uint16_t>(engine->expert_prefetch_priorities[slot], prefetch_priority);
    }
    if (slot < engine->expert_residency_hints.size() &&
        residency_hint != moe_JUJU_RESIDENCY_AUTO) {
        const uint8_t current = engine->expert_residency_hints[slot];
        if (current == moe_JUJU_RESIDENCY_AUTO ||
            residency_hint < current ||
            residency_hint == moe_JUJU_RESIDENCY_FAST_MEM) {
            engine->expert_residency_hints[slot] = residency_hint;
        }
    }
}

static uint16_t expert_prefetch_priority_hint(const moe_pc_engine* engine, int layer, int expert) {
    const size_t slot = expert_dense_slot(layer, expert);
    if (!engine || slot == tensor_slot_invalid() || slot >= engine->expert_prefetch_priorities.size()) {
        return 0;
    }
    return engine->expert_prefetch_priorities[slot];
}

static uint8_t expert_residency_hint(const moe_pc_engine* engine, int layer, int expert) {
    const size_t slot = expert_dense_slot(layer, expert);
    if (!engine || slot == tensor_slot_invalid() || slot >= engine->expert_residency_hints.size()) {
        return moe_JUJU_RESIDENCY_AUTO;
    }
    return engine->expert_residency_hints[slot];
}

static uint64_t expert_bytes_get(const moe_pc_engine* engine, int layer, int expert) {
    const size_t slot = expert_dense_slot(layer, expert);
    if (!engine || slot == tensor_slot_invalid() || slot >= engine->expert_bytes.size()) {
        return 0;
    }
    return engine->expert_bytes[slot];
}
