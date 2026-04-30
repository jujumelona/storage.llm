static uint64_t hardcoded_expert_bundle_bytes(int layer) {
    const moe_storage_layer_spec_t* spec = layer >= 0 ? moe_storage_layer_at((uint32_t)layer) : nullptr;
    if (!spec || !spec->has_moe_experts || spec->expert_count == 0) {
        return 0;
    }
    return spec->bundle_total_bytes / spec->expert_count;
}

static uint64_t expert_triplet_or_hardcoded_bytes(const moe_pc_engine* engine, int layer, int expert) {
    const moe_storage_constants_t* c = moe_storage_constants();
    if (!c || layer < (int)c->first_moe_layer || layer > (int)c->last_moe_layer ||
        expert < 0 || expert >= (int)c->experts_per_moe_layer) {
        return 0;
    }
    const uint64_t bytes = expert_triplet_bytes(engine, layer, expert);
    return bytes ? bytes : hardcoded_expert_bundle_bytes(layer);
}
