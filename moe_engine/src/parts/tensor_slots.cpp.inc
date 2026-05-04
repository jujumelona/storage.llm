static size_t tensor_slot_invalid() {
    return SIZE_MAX;
}

static void tensor_slot_params(
    const moe_pc_engine* engine,
    uint32_t* out_layers,
    uint32_t* out_experts,
    uint32_t* out_projs
) {
    const moe_model_shape_t shape = moe_engine_model_shape_or_static(engine);
    const uint32_t layers = shape.num_hidden_layers ? shape.num_hidden_layers : 79u;
    const uint32_t experts = shape.experts_per_moe_layer ? shape.experts_per_moe_layer : 256u;
    const uint32_t projs = shape.projection_count ? shape.projection_count : 3u;
    if (out_layers) {
        *out_layers = layers;
    }
    if (out_experts) {
        *out_experts = experts;
    }
    if (out_projs) {
        *out_projs = projs;
    }
}

static size_t tensor_slot_count(const moe_pc_engine* engine = nullptr) {
    uint32_t layers = 79u, experts = 256u, projs = 3u;
    tensor_slot_params(engine, &layers, &experts, &projs);
    // BUGFIX 334: Check for overflow in multiplication ★
    // Problem: layers * experts * projs can overflow size_t
    // Solution: Check each multiplication step for overflow
    if (layers > SIZE_MAX / experts) {
        return SIZE_MAX;  // Overflow would occur
    }
    const size_t layer_expert = (size_t)layers * (size_t)experts;
    if (layer_expert > SIZE_MAX / projs) {
        return SIZE_MAX;  // Overflow would occur
    }
    return layer_expert * (size_t)projs;
}

static size_t tensor_record_capacity(const moe_pc_engine* engine = nullptr) {
    const moe_storage_constants_t storage = moe_engine_storage_constants_or_static(engine);
    if (!storage.total_expert_count || !storage.expert_projection_count) {
        return tensor_slot_count(engine);
    }
    return (size_t)storage.total_expert_count * (size_t)storage.expert_projection_count;
}

static size_t expert_slot_count(const moe_pc_engine* engine = nullptr) {
    uint32_t layers = 79u, experts = 256u, projs = 3u;
    (void)projs;
    tensor_slot_params(engine, &layers, &experts, nullptr);
    // BUGFIX 335: Check for overflow in multiplication ★
    // Problem: layers * experts can overflow size_t
    // Solution: Check multiplication for overflow
    if (layers > SIZE_MAX / experts) {
        return SIZE_MAX;  // Overflow would occur
    }
    return (size_t)layers * (size_t)experts;
}

static size_t tensor_dense_slot(const moe_pc_engine* engine, int layer, int expert, moe_projection_t proj) {
    uint32_t layers = 79u, experts = 256u, projs = 3u;
    tensor_slot_params(engine, &layers, &experts, &projs);
    if (layer < 0 || (uint32_t)layer >= layers || expert < 0 || (uint32_t)expert >= experts ||
        (uint32_t)proj >= projs) {
        return tensor_slot_invalid();
    }
    return ((size_t)layer * (size_t)experts + (size_t)expert) * (size_t)projs + (size_t)proj;
}

static size_t tensor_dense_slot(int layer, int expert, moe_projection_t proj) {
    return tensor_dense_slot(nullptr, layer, expert, proj);
}

static size_t expert_dense_slot(const moe_pc_engine* engine, int layer, int expert) {
    uint32_t layers = 79u, experts = 256u;
    tensor_slot_params(engine, &layers, &experts, nullptr);
    if (layer < 0 || (uint32_t)layer >= layers || expert < 0 || (uint32_t)expert >= experts) {
        return tensor_slot_invalid();
    }
    return (size_t)layer * (size_t)experts + (size_t)expert;
}

static size_t expert_dense_slot(int layer, int expert) {
    return expert_dense_slot(nullptr, layer, expert);
}

static uint32_t expert_dense_slot_width(const moe_pc_engine* engine = nullptr) {
    uint32_t experts = 256u;
    tensor_slot_params(engine, nullptr, &experts, nullptr);
    return experts;
}

static void expert_dense_slot_decode(const moe_pc_engine* engine, size_t slot, int* layer, int* expert) {
    const uint32_t experts = expert_dense_slot_width(engine);
    // BUGFIX 227: Validate experts > 0 to prevent division by zero ★★
    // Problem: experts=0 causes division by zero in slot / experts and slot % experts
    // Solution: Return early with 0 values if experts == 0
    // Impact: Prevents crash in slot decoding with invalid config
    if (experts == 0) {
        if (layer) *layer = 0;
        if (expert) *expert = 0;
        return;
    }
    if (layer) {
        *layer = (int)(slot / experts);
    }
    if (expert) {
        *expert = (int)(slot % experts);
    }
}

static void expert_dense_slot_decode(size_t slot, int* layer, int* expert) {
    expert_dense_slot_decode(nullptr, slot, layer, expert);
}

static void tensor_slots_reset(moe_pc_engine* engine) {
    if (!engine) {
        return;
    }
    engine->tensor_slots.assign(tensor_slot_count(engine), tensor_slot_invalid());
    engine->expert_bytes.assign(expert_slot_count(engine), 0);
    engine->expert_prefetch_priorities.assign(expert_slot_count(engine), 0);
    engine->expert_residency_hints.assign(expert_slot_count(engine), moe_JUJU_RESIDENCY_AUTO);
    engine->expert_segment_policies.assign(expert_slot_count(engine), moe_JUJU_SEGMENT_POLICY_AUTO);
    engine->expert_partial_exec_flags.assign(expert_slot_count(engine), 0);
    engine->expert_partial_accuracy_milli.assign(expert_slot_count(engine), 0);
    engine->juju_buddy_expert_ids.assign(expert_slot_count(engine), std::vector<int>{});
}

static void tensor_slots_set(moe_pc_engine* engine, int layer, int expert, moe_projection_t proj, size_t index) {
    const size_t slot = tensor_dense_slot(engine, layer, expert, proj);
    if (engine && slot != tensor_slot_invalid() && slot < engine->tensor_slots.size()) {
        engine->tensor_slots[slot] = index;
    }
}

static size_t tensor_slots_get(const moe_pc_engine* engine, int layer, int expert, moe_projection_t proj) {
    const size_t slot = tensor_dense_slot(engine, layer, expert, proj);
    if (!engine || slot == tensor_slot_invalid() || slot >= engine->tensor_slots.size()) {
        return tensor_slot_invalid();
    }
    return engine->tensor_slots[slot];
}

static void expert_bytes_add(moe_pc_engine* engine, int layer, int expert, uint64_t bytes) {
    const size_t slot = expert_dense_slot(engine, layer, expert);
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
    const size_t slot = expert_dense_slot(engine, layer, expert);
    if (!engine || slot == tensor_slot_invalid()) {
        return;
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

static void expert_segment_hints_update(
    moe_pc_engine* engine,
    int layer,
    int expert,
    uint8_t segment_policy,
    uint8_t can_partial_exec,
    uint16_t partial_accuracy_milli
) {
    const size_t slot = expert_dense_slot(engine, layer, expert);
    if (!engine || slot == tensor_slot_invalid()) {
        return;
    }
    if (slot < engine->expert_segment_policies.size() &&
        segment_policy != moe_JUJU_SEGMENT_POLICY_AUTO) {
        engine->expert_segment_policies[slot] = segment_policy;
    }
    if (slot < engine->expert_partial_exec_flags.size() && can_partial_exec) {
        engine->expert_partial_exec_flags[slot] = 1;
    }
    if (slot < engine->expert_partial_accuracy_milli.size() && partial_accuracy_milli > 0) {
        engine->expert_partial_accuracy_milli[slot] =
            std::max<uint16_t>(engine->expert_partial_accuracy_milli[slot], partial_accuracy_milli);
    }
}

static void expert_buddy_hints_update(
    moe_pc_engine* engine,
    int layer,
    int expert,
    const std::vector<int>& buddies
) {
    const size_t slot = expert_dense_slot(engine, layer, expert);
    if (!engine || buddies.empty() || slot == tensor_slot_invalid() ||
        slot >= engine->juju_buddy_expert_ids.size()) {
        return;
    }
    std::vector<int>& dst = engine->juju_buddy_expert_ids[slot];
    for (int buddy : buddies) {
        if (buddy < 0 || buddy == expert) {
            continue;
        }
        if (std::find(dst.begin(), dst.end(), buddy) == dst.end()) {
            dst.push_back(buddy);
            engine->juju_buddy_explicit_edge_count.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

static uint16_t expert_prefetch_priority_hint(const moe_pc_engine* engine, int layer, int expert) {
    const size_t slot = expert_dense_slot(engine, layer, expert);
    if (!engine || slot == tensor_slot_invalid() || slot >= engine->expert_prefetch_priorities.size()) {
        return 0;
    }
    return engine->expert_prefetch_priorities[slot];
}

static uint8_t expert_residency_hint(const moe_pc_engine* engine, int layer, int expert) {
    const size_t slot = expert_dense_slot(engine, layer, expert);
    if (!engine || slot == tensor_slot_invalid() || slot >= engine->expert_residency_hints.size()) {
        return moe_JUJU_RESIDENCY_AUTO;
    }
    return engine->expert_residency_hints[slot];
}

static uint8_t expert_partial_exec_hint(const moe_pc_engine* engine, int layer, int expert) {
    const size_t slot = expert_dense_slot(engine, layer, expert);
    if (!engine || slot == tensor_slot_invalid() || slot >= engine->expert_partial_exec_flags.size()) {
        return 0;
    }
    return engine->expert_partial_exec_flags[slot];
}

static uint16_t expert_partial_accuracy_hint_milli(const moe_pc_engine* engine, int layer, int expert) {
    const size_t slot = expert_dense_slot(engine, layer, expert);
    if (!engine || slot == tensor_slot_invalid() || slot >= engine->expert_partial_accuracy_milli.size()) {
        return 0;
    }
    return engine->expert_partial_accuracy_milli[slot];
}

static uint64_t expert_bytes_get(const moe_pc_engine* engine, int layer, int expert) {
    const size_t slot = expert_dense_slot(engine, layer, expert);
    if (!engine || slot == tensor_slot_invalid() || slot >= engine->expert_bytes.size()) {
        return 0;
    }
    return engine->expert_bytes[slot];
}
