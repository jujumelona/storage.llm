static uint64_t fixed_output_core_bytes() {
    const moe_storage_constants_t* s = moe_storage_constants();
    return (s ? s->raw_tensor_lm_head_bytes : 1903165440ull) + 12288ull;
}

static uint64_t max_raw_role_layer_bytes(uint32_t roles) {
    const moe_storage_constants_t* storage = moe_storage_constants();
    const int32_t max_layer = storage ? (int32_t)storage->num_hidden_layers + 1 : 80;
    std::vector<uint64_t> layer_bytes((size_t)max_layer + 2u, 0);
    const uint32_t count = moe_storage_raw_span_count();
    for (uint32_t i = 0; i < count; ++i) {
        const moe_storage_raw_span_spec_t* s = moe_storage_raw_span_at(i);
        if (!s || (s->role & roles) == 0 || s->layer < -1 || s->layer > max_layer) {
            continue;
        }
        layer_bytes[s->layer + 1] += s->bytes;
    }
    uint64_t best = 0;
    for (uint64_t bytes : layer_bytes) {
        best = std::max<uint64_t>(best, bytes);
    }
    return best;
}

static uint64_t working_common_cache_bytes() {
    // Bug #14: memoize ??raw spans are hardcoded constants, result never changes
    static uint64_t cached = 0;
    if (cached) return cached;
    const uint32_t activation_roles = 1u | 2u | 4u;
    const uint32_t shared_compute_roles = 8u;
    cached = fixed_output_core_bytes() +
        max_raw_role_layer_bytes(activation_roles) +
        max_raw_role_layer_bytes(shared_compute_roles);
    return cached;
}
