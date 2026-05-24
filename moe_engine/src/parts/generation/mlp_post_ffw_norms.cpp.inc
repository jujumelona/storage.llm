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

static int moe_layer_post_ffw_norm_f32(
    moe_pc_engine* engine,
    int32_t layer,
    const float* input,
    uint32_t hidden_size,
    float* output,
    float eps
) {
    static const char* const roles[] = { "post_ffw_norm" };
    if (moe_graph_ir_math_required(engine) &&
        moe_graph_ir_apply_rmsnorm_role_any_f32(
            engine, layer, roles, sizeof(roles) / sizeof(roles[0]),
            input, hidden_size, output, eps)) {
        return 1;
    }
    if (moe_graph_ir_math_required(engine)) {
        moe_raw_tensor_cache_entry raw{};
        if (moe_graph_ir_map_layer_contract_tensor_role_f32(
                engine, layer, "norms", "post_ffw_norm", &raw)) {
            const int ok = moe_apply_rmsnorm_common_weight_f32(engine, raw, input, hidden_size, output, eps);
            moe_profile_log(
                "rmsnorm_common_apply",
                "layer=%d suffix0=post_ffw_norm ok=%d hidden=%u source=contract",
                layer, ok, hidden_size);
            return ok;
        }
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
    if (moe_graph_ir_math_required(engine) &&
        moe_graph_ir_map_layer_op_role_any(
            engine, layer, roles, sizeof(roles) / sizeof(roles[0]), &raw)) {
        if (attempted) {
            *attempted = 1;
        }
        const int ok = moe_apply_rmsnorm_common_weight_add_f32(
            engine, raw, input, hidden_size, residual, eps);
        moe_profile_log(
            "rmsnorm_common_add",
            "layer=%d suffix0=post_ffw_norm ok=%d hidden=%u source=op_role",
            layer, ok, hidden_size);
        return ok;
    }
    if (moe_graph_ir_math_required(engine) &&
        moe_graph_ir_map_layer_contract_tensor_role_f32(
            engine, layer, "norms", "post_ffw_norm", &raw)) {
        if (attempted) {
            *attempted = 1;
        }
        const int ok = moe_apply_rmsnorm_common_weight_add_f32(
            engine, raw, input, hidden_size, residual, eps);
        moe_profile_log(
            "rmsnorm_common_add",
            "layer=%d suffix0=post_ffw_norm ok=%d hidden=%u source=contract",
            layer, ok, hidden_size);
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
    static const char* const roles[] = { "post_ffw_norm_1" };
    if (moe_graph_ir_math_required(engine) &&
        moe_graph_ir_apply_rmsnorm_role_any_f32(
            engine, layer, roles, sizeof(roles) / sizeof(roles[0]),
            input, hidden_size, output, eps)) {
        return 1;
    }
    if (moe_graph_ir_math_required(engine)) {
        moe_raw_tensor_cache_entry raw{};
        if (moe_graph_ir_map_layer_contract_tensor_role_f32(
                engine, layer, "norms", "post_ffw_norm_1", &raw)) {
            const int ok = moe_apply_rmsnorm_common_weight_f32(engine, raw, input, hidden_size, output, eps);
            moe_profile_log(
                "rmsnorm_common_apply",
                "layer=%d suffix0=post_ffw_norm_1 ok=%d hidden=%u source=contract",
                layer, ok, hidden_size);
            return ok;
        }
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
    static const char* const roles[] = { "post_ffw_norm_2" };
    if (moe_graph_ir_math_required(engine) &&
        moe_graph_ir_apply_rmsnorm_role_any_f32(
            engine, layer, roles, sizeof(roles) / sizeof(roles[0]),
            input, hidden_size, output, eps)) {
        return 1;
    }
    if (moe_graph_ir_math_required(engine)) {
        moe_raw_tensor_cache_entry raw{};
        if (moe_graph_ir_map_layer_contract_tensor_role_f32(
                engine, layer, "norms", "post_ffw_norm_2", &raw)) {
            const int ok = moe_apply_rmsnorm_common_weight_f32(engine, raw, input, hidden_size, output, eps);
            moe_profile_log(
                "rmsnorm_common_apply",
                "layer=%d suffix0=post_ffw_norm_2 ok=%d hidden=%u source=contract",
                layer, ok, hidden_size);
            return ok;
        }
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
