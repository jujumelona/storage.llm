static uint64_t tier_used(const moe_pc_engine* engine, moe_tier_t tier) {
    if (!engine) {
        return 0;
    }
    switch (tier) {
        case moe_TIER_VRAM:
            return engine->vram_used_bytes.load();
        case moe_TIER_RAM:
            return engine->ram_used_bytes.load();
        case moe_TIER_DB:
            return engine->db_used_bytes.load();
        case moe_TIER_STREAM:
        default:
            return 0;
    }
}

static uint64_t tier_budget(const moe_pc_engine* engine, moe_tier_t tier) {
    switch (tier) {
        case moe_TIER_VRAM:
            return engine->config.vram_budget_bytes;
        case moe_TIER_RAM:
            return engine->config.ram_budget_bytes;
        case moe_TIER_DB:
            return engine->config.db_budget_bytes;
        case moe_TIER_STREAM:
        default:
            return 0;
    }
}

static uint32_t tier_speed_rank(moe_tier_t tier) {
    switch (tier) {
        case moe_TIER_VRAM:
            return 0;
        case moe_TIER_RAM:
            return 1;
        case moe_TIER_DB:
        case moe_TIER_STREAM:
        default:
            return 2;
    }
}

static uint64_t topology_key(int layer, int expert) {
    const moe_storage_constants_t* storage = moe_storage_constants();
    const uint32_t layers = storage && storage->num_hidden_layers ? storage->num_hidden_layers : 79u;
    const uint32_t experts = storage && storage->experts_per_moe_layer ? storage->experts_per_moe_layer : 256u;
    if (layer < 0 || (uint32_t)layer >= layers || expert < 0 || (uint32_t)expert >= experts) {
        return UINT64_MAX;
    }
    return (static_cast<uint64_t>(static_cast<uint32_t>(layer)) << 32) |
        static_cast<uint32_t>(expert);
}

static int topology_key_layer(uint64_t key) {
    return static_cast<int>(key >> 32);
}

static int topology_key_expert(uint64_t key) {
    return static_cast<int>(key & 0xffffffffu);
}
