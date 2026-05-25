static int moe_layer_post_ffw_norm_suffixes_f32(
    moe_pc_engine* engine,
    int32_t layer,
    const char* const* suffixes,
    size_t suffix_count,
    const float* input,
    uint32_t hidden_size,
    float* output,
    float eps
) {
    return moe_layer_rmsnorm_suffixes_f32(
        engine, layer, suffixes, suffix_count, input, hidden_size, output, eps);
}

struct moe_post_ffw_norm_role_cache_entry {
    const moe_pc_engine* engine = nullptr;
    uint64_t generation = 0;
    int32_t layer = -1;
    uint32_t role_id = 0;
    int ready = 0;
    int mapped = 0;
    int source = 0;
    moe_raw_tensor_cache_entry raw{};
};

static int moe_map_post_ffw_norm_role_cached_f32(
    moe_pc_engine* engine,
    int32_t layer,
    uint32_t role_id,
    const char* role,
    moe_raw_tensor_cache_entry* out,
    int* out_source
) {
    if (out_source) {
        *out_source = 0;
    }
    if (!engine || !role || !out || !moe_graph_ir_math_required(engine)) {
        return 0;
    }
    thread_local std::vector<moe_post_ffw_norm_role_cache_entry> cache;
    const uint64_t generation = engine->engine_generation;
    for (moe_post_ffw_norm_role_cache_entry& entry : cache) {
        if (entry.ready && entry.engine == engine && entry.generation == generation &&
            entry.layer == layer && entry.role_id == role_id) {
            if (!entry.mapped) {
                return 0;
            }
            *out = entry.raw;
            if (out_source) {
                *out_source = entry.source;
            }
            return 1;
        }
    }
    moe_post_ffw_norm_role_cache_entry entry{};
    entry.engine = engine;
    entry.generation = generation;
    entry.layer = layer;
    entry.role_id = role_id;
    entry.ready = 1;
    static const char* const section = "norms";
    const char* const roles[] = { role };
    if (moe_graph_ir_map_layer_op_role_any(engine, layer, roles, 1, &entry.raw)) {
        entry.mapped = 1;
        entry.source = 1;
    } else if (moe_graph_ir_map_layer_contract_tensor_role_f32(
                   engine, layer, section, role, &entry.raw)) {
        entry.mapped = 1;
        entry.source = 2;
    }
    if (cache.size() >= 256u) {
        cache.erase(cache.begin());
    }
    cache.push_back(entry);
    if (!entry.mapped) {
        return 0;
    }
    *out = entry.raw;
    if (out_source) {
        *out_source = entry.source;
    }
    return 1;
}

static int moe_apply_post_ffw_norm_role_cached_f32(
    moe_pc_engine* engine,
    int32_t layer,
    uint32_t role_id,
    const char* role,
    const float* input,
    uint32_t hidden_size,
    float* output,
    float eps
) {
    moe_raw_tensor_cache_entry raw{};
    int source = 0;
    if (!moe_map_post_ffw_norm_role_cached_f32(engine, layer, role_id, role, &raw, &source)) {
        return 0;
    }
    const int ok = moe_apply_rmsnorm_common_weight_f32(engine, raw, input, hidden_size, output, eps);
    moe_profile_log(
        "rmsnorm_common_apply",
        "layer=%d suffix0=%s ok=%d hidden=%u source=%s",
        layer, role, ok, hidden_size, source == 1 ? "op_role_cached" : "contract_cached");
    return ok;
}

static int moe_layer_post_ffw_norm_f32(
    moe_pc_engine* engine,
    int32_t layer,
    const float* input,
    uint32_t hidden_size,
    float* output,
    float eps
) {
    if (moe_graph_ir_math_required(engine) &&
        moe_apply_post_ffw_norm_role_cached_f32(
            engine, layer, 0, "post_ffw_norm", input, hidden_size, output, eps)) {
        return 1;
    }
    static const char* const suffixes[] = {
        "post_ffw_norm.weight",
        "ffn_post_norm.weight"
    };
    return moe_layer_post_ffw_norm_suffixes_f32(
        engine, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]),
        input, hidden_size, output, eps);
}

static int moe_layer_post_ffw_norm_add_residual_f32(
    moe_pc_engine* engine,
    int32_t layer,
    const float* input,
    uint32_t hidden_size,
    float* residual,
    float eps,
    int* attempted
) {
    if (attempted) {
        *attempted = 0;
    }
    if (!engine || !input || !residual || hidden_size == 0) {
        return 0;
    }
    static const char* const roles[] = { "post_ffw_norm" };
    moe_raw_tensor_cache_entry raw{};
    int source = 0;
    if (moe_graph_ir_math_required(engine) &&
        moe_map_post_ffw_norm_role_cached_f32(engine, layer, 0, roles[0], &raw, &source)) {
        if (attempted) {
            *attempted = 1;
        }
        const int ok = moe_apply_rmsnorm_common_weight_add_f32(
            engine, raw, input, hidden_size, residual, eps);
        moe_profile_log(
            "rmsnorm_common_add",
            "layer=%d suffix0=post_ffw_norm ok=%d hidden=%u source=%s",
            layer, ok, hidden_size, source == 1 ? "op_role_cached" : "contract_cached");
        return ok;
    }
    return 0;
}

static int moe_layer_has_post_ffw_norm_weight(
    const moe_pc_engine* engine,
    int32_t layer
) {
    static const char* const roles[] = { "post_ffw_norm" };
    if (moe_graph_ir_math_required(engine) &&
        moe_graph_ir_layer_has_op_role_any(
            engine, layer, roles, sizeof(roles) / sizeof(roles[0]))) {
        return 1;
    }
    if (moe_graph_ir_math_required(engine) &&
        moe_graph_ir_layer_contract_declares_tensor_role(engine, layer, "norms", "post_ffw_norm")) {
        return 1;
    }
    return
        moe_engine_has_common_tensor_suffix(engine, layer, "post_ffw_norm.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "ffn_post_norm.weight");
}

static int moe_layer_post_ffw_norm1_f32(
    moe_pc_engine* engine,
    int32_t layer,
    const float* input,
    uint32_t hidden_size,
    float* output,
    float eps
) {
    if (moe_graph_ir_math_required(engine) &&
        moe_apply_post_ffw_norm_role_cached_f32(
            engine, layer, 1, "post_ffw_norm_1", input, hidden_size, output, eps)) {
        return 1;
    }
    static const char* const suffixes[] = {
        "post_ffw_norm_1.weight",
        "ffn_post_norm_1.weight"
    };
    return moe_layer_post_ffw_norm_suffixes_f32(
        engine, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]),
        input, hidden_size, output, eps);
}

static int moe_layer_has_post_ffw_norm1_weight(
    const moe_pc_engine* engine,
    int32_t layer
) {
    static const char* const roles[] = { "post_ffw_norm_1" };
    if (moe_graph_ir_math_required(engine) &&
        moe_graph_ir_layer_has_op_role_any(
            engine, layer, roles, sizeof(roles) / sizeof(roles[0]))) {
        return 1;
    }
    if (moe_graph_ir_math_required(engine) &&
        moe_graph_ir_layer_contract_declares_tensor_role(engine, layer, "norms", "post_ffw_norm_1")) {
        return 1;
    }
    return
        moe_engine_has_common_tensor_suffix(engine, layer, "post_ffw_norm_1.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "ffn_post_norm_1.weight");
}

static int moe_layer_post_ffw_norm2_f32(
    moe_pc_engine* engine,
    int32_t layer,
    const float* input,
    uint32_t hidden_size,
    float* output,
    float eps
) {
    if (moe_graph_ir_math_required(engine) &&
        moe_apply_post_ffw_norm_role_cached_f32(
            engine, layer, 2, "post_ffw_norm_2", input, hidden_size, output, eps)) {
        return 1;
    }
    static const char* const suffixes[] = {
        "post_ffw_norm_2.weight",
        "ffn_post_norm_2.weight"
    };
    return moe_layer_post_ffw_norm_suffixes_f32(
        engine, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]),
        input, hidden_size, output, eps);
}

static int moe_layer_has_post_ffw_norm2_weight(
    const moe_pc_engine* engine,
    int32_t layer
) {
    static const char* const roles[] = { "post_ffw_norm_2" };
    if (moe_graph_ir_math_required(engine) &&
        moe_graph_ir_layer_has_op_role_any(
            engine, layer, roles, sizeof(roles) / sizeof(roles[0]))) {
        return 1;
    }
    if (moe_graph_ir_math_required(engine) &&
        moe_graph_ir_layer_contract_declares_tensor_role(engine, layer, "norms", "post_ffw_norm_2")) {
        return 1;
    }
    return
        moe_engine_has_common_tensor_suffix(engine, layer, "post_ffw_norm_2.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "ffn_post_norm_2.weight");
}

static int moe_layer_output_scale_f32(
    moe_pc_engine* engine,
    int32_t layer,
    float* hidden,
    uint32_t hidden_size
) {
    if (!engine || !hidden || hidden_size == 0 || hidden_size > 100000) {
        return 0;
    }
    moe_raw_tensor_cache_entry raw{};
    int mapped = 0;
    if (moe_graph_ir_math_required(engine)) {
        static const char* const roles[] = { "layer_output_scale" };
        mapped = moe_graph_ir_map_layer_op_role_any(
            engine, layer, roles, sizeof(roles) / sizeof(roles[0]), &raw);
        if (!mapped) {
            mapped = moe_graph_ir_map_layer_contract_tensor_role_f32(
                engine, layer, "tail", "layer_output_scale", &raw);
        }
    }
    if (!mapped) {
        static const char* const suffixes[] = {
            "layer_output_scale.weight",
            "layer_scalar.weight",
            "layer_scalar"
        };
        if (!moe_map_layer_common_tensor(engine, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]), &raw)) {
            return 1;
        }
    }
    const uint64_t count = moe_raw_tensor_scalar_count(raw);
    if (count == 1u) {
        float scale = 1.0f;
        if (!moe_raw_tensor_scalar_at_f32(raw, 0, &scale) || !std::isfinite(scale)) {
            return 0;
        }
        moe_scale_f32(hidden, scale, hidden_size);
        return 1;
    }
    if (count < hidden_size) {
        return 0;
    }
    for (uint32_t i = 0; i < hidden_size; ++i) {
        float scale = 1.0f;
        if (!moe_raw_tensor_scalar_at_f32(raw, i, &scale) || !std::isfinite(scale)) {
            return 0;
        }
        hidden[i] *= scale;
    }
    return 1;
}
