static uint64_t moe_common_raw_tensor_bytes() {
    const moe_storage_constants_t* storage = moe_storage_constants();
    return storage ? storage->raw_tensor_total_bytes : 0;
}

static uint64_t moe_common_raw_tensor_count() {
    const moe_storage_constants_t* storage = moe_storage_constants();
    return storage ? storage->raw_tensor_count : 0;
}

static uint64_t common_vram_wanted_bytes(const moe_pc_engine* engine) {
    const uint64_t common_bytes = moe_common_raw_tensor_bytes();
    if (!engine || common_bytes == 0) {
        return 0;
    }
    if (engine->config.common_vram_reserve_bytes) {
        return std::min<uint64_t>(engine->config.common_vram_reserve_bytes, common_bytes);
    }
    const moe_storage_constants_t* s = moe_storage_constants();
    // Fix: Use distinct fallback values for lm_head and embedding
    // lm_head is typically smaller than embedding in current offload MoE roots.
    // Bug Fix: Router bytes removed from common VRAM reservation
    // Router weights are already prefetched via moe_enqueue_attention_prefetch (activation pool)
    // Including them here double-counts ~600MB, reducing expert VRAM budget unnecessarily
    const uint64_t core = (s ? s->raw_tensor_lm_head_bytes : 0ull) + 12288ull;
    const uint64_t embed = s ? s->raw_tensor_embedding_bytes : 0ull;
    const uint64_t bundle = s && s->normal_bundle_max_bytes ? s->normal_bundle_max_bytes : 32ull * 1024ull * 1024ull;
    const uint64_t expert_floor = bundle * 8ull;
    const uint64_t budget = engine->config.vram_budget_bytes;
    if (budget <= expert_floor) {
        return 0;
    }
    uint64_t reserve = budget >= core + expert_floor ? core : 12288ull;
    if (budget >= core + embed + expert_floor) {
        reserve += embed;
    }
    return std::min<uint64_t>(reserve, common_bytes);
}

static uint64_t vram_expert_budget_bytes(const moe_pc_engine* engine) {
    if (!engine || engine->config.vram_budget_bytes == 0) {
        return 0;
    }
    const uint64_t common = engine->common_vram_reserved_bytes.load();
    return engine->config.vram_budget_bytes > common ?
        engine->config.vram_budget_bytes - common : 0;
}

static void apply_common_vram_reservation(moe_pc_engine* engine) {
    if (!engine) {
        return;
    }
    engine->common_vram_reserved_bytes.store(0);
    engine->vram_used_bytes.store(0);
    if (!engine->config.prefer_gpu || !engine_has_device_upload_workers(engine) ||
        !engine->io_config.enable_common_raw_vram_pin) {
        engine->common_vram_upload_complete.store(true, std::memory_order_release);
        return;
    }
    const uint64_t reserve = std::min<uint64_t>(
        common_vram_wanted_bytes(engine),
        engine->config.vram_budget_bytes
    );
    engine->common_vram_reserved_bytes.store(reserve);
    engine->vram_used_bytes.store(0);
    engine->common_vram_upload_complete.store(reserve == 0, std::memory_order_release);
}
