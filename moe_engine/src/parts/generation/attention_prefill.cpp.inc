static int moe_apply_rmsnorm_weight_entry_f32(
    const moe_pc_engine* engine,
    const moe_raw_tensor_cache_entry& raw,
    const float* input,
    uint32_t count,
    float* output,
    float eps
) {
    // BUGFIX 214: Division by zero clarification (same pattern as BUGFIX 208)
    // Problem: count == 0 check exists but no comment explaining division safety
    // Solution: Add explicit comment for code clarity
    if (!raw.base || !input || !output || count == 0 || raw.cols != 0 || raw.rows != count) {
        return 0;
    }
    const uint32_t elem = moe_raw_scalar_size(raw.scalar_encoding);
    if (raw.bytes < (uint64_t)count * (uint64_t)elem) {
        return 0;
    }
    double ss = 0.0;
    for (uint32_t i = 0; i < count; ++i) {
        const double val = (double)input[i];
        if (!std::isfinite(val)) {
            return 0;
        }
        ss += val * val;
    }
    // Safe: count > 0 guaranteed by check above
    const float norm = 1.0f / std::sqrt((float)(ss / (double)count) + (eps > 0.0f ? eps : 1.0e-6f));
    const int unit_offset = moe_engine_rmsnorm_unit_offset(engine, raw);
    for (uint32_t i = 0; i < count; ++i) {
        const float w = moe_raw_scalar_to_f32(raw.base + (uint64_t)i * (uint64_t)elem, raw.scalar_encoding);
        if (!std::isfinite(w)) {
            return 0;
        }
        const float result = input[i] * norm * (unit_offset ? (1.0f + w) : w);
        if (!std::isfinite(result)) {
            return 0;
        }
        output[i] = result;
    }
    return 1;
}

static int moe_apply_optional_head_rmsnorm_f32(
    moe_pc_engine* engine,
    int32_t layer,
    moe_raw_tensor_kind_t kind,
    float* values,
    uint32_t head_count,
    uint32_t head_dim,
    float eps
) {
    if (!engine || !values || head_count == 0 || head_dim == 0) {
        return 0;
    }
    moe_raw_tensor_cache_entry raw{};
    if (!get_cached_raw_tensor(engine, layer, kind, &raw)) {
        return 1;
    }
    if (raw.rows != head_dim || raw.cols != 0) {
        return 0;
    }
    std::vector<float> tmp(head_dim);
    for (uint32_t h = 0; h < head_count; ++h) {
        float* head = values + (uint64_t)h * (uint64_t)head_dim;
        if (!moe_apply_rmsnorm_weight_entry_f32(engine, raw, head, head_dim, tmp.data(), eps)) {
            return 0;
        }
        std::memcpy(head, tmp.data(), (size_t)head_dim * sizeof(float));
    }
    return 1;
}

static int moe_apply_unweighted_head_rmsnorm_f32(
    float* values,
    uint32_t head_count,
    uint32_t head_dim,
    float eps
) {
    // BUGFIX 215: Division by zero clarification
    // Problem: head_dim == 0 check exists but no comment explaining division safety
    // Solution: Add explicit comment for code clarity
    if (!values || head_count == 0 || head_dim == 0) {
        return 0;
    }
    const float norm_eps = eps > 0.0f ? eps : 1.0e-6f;
    for (uint32_t h = 0; h < head_count; ++h) {
        float* head = values + (uint64_t)h * (uint64_t)head_dim;
        double ss = 0.0;
        for (uint32_t i = 0; i < head_dim; ++i) {
            const double val = (double)head[i];
            if (!std::isfinite(val)) {
                return 0;
            }
            ss += val * val;
        }
        // Safe: head_dim > 0 guaranteed by check above
        const float norm = 1.0f / std::sqrt((float)(ss / (double)head_dim) + norm_eps);
        for (uint32_t i = 0; i < head_dim; ++i) {
            const float result = head[i] * norm;
            if (!std::isfinite(result)) {
                return 0;
            }
            head[i] = result;
        }
    }
    return 1;
}

static int moe_layer_uses_unweighted_v_norm(
    const moe_pc_engine* engine,
    int32_t layer
) {
    if (!engine || layer < 0) {
        return 0;
    }
    if (moe_engine_graph_ir_mentions(engine, "attn_v_norm") ||
        moe_engine_graph_ir_mentions(engine, "value_norm") ||
        moe_engine_graph_ir_mentions(engine, "v_norm")) {
        return 1;
    }
    if (moe_engine_has_common_tensor_suffix(engine, layer, "attn_v_norm.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "value_norm.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "v_norm.weight")) {
        return 1;
    }
    return moe_engine_has_common_tensor_suffix(engine, layer, "ffn_gate_inp.scale") &&
        (moe_engine_has_common_tensor_suffix(engine, layer, "post_ffw_norm.weight") ||
         moe_engine_has_common_tensor_suffix(engine, layer, "post_ffw_norm_1.weight") ||
         moe_engine_has_common_tensor_suffix(engine, layer, "post_ffw_norm_2.weight") ||
         moe_engine_has_common_tensor_suffix(engine, layer, "layer_output_scale.weight")) ? 1 : 0;
}

static int moe_layer_has_qk_norm_scaled_standard_layout(
    const moe_pc_engine* engine,
    int32_t layer
) {
    if (!engine || layer < 0) {
        return 0;
    }
    const int has_q_norm =
        moe_engine_has_common_tensor_suffix(engine, layer, "attn_q_norm.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "q_norm.weight");
    const int has_k_norm =
        moe_engine_has_common_tensor_suffix(engine, layer, "attn_k_norm.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "k_norm.weight");
    const int has_scaled_block_norms =
        moe_engine_has_common_tensor_suffix(engine, layer, "post_attention_norm.weight") &&
        moe_engine_has_common_tensor_suffix(engine, layer, "post_ffw_norm.weight") &&
        moe_engine_has_common_tensor_suffix(engine, layer, "layer_output_scale.weight");
    return has_q_norm && has_k_norm && has_scaled_block_norms;
}

static float moe_standard_attention_score_scale(
    const moe_pc_engine* engine,
    int32_t layer,
    uint32_t head_dim,
    const char** source_out = nullptr
) {
    if (source_out) {
        *source_out = "head_dim";
    }
    if (head_dim == 0) {
        if (source_out) {
            *source_out = "fallback";
        }
        return 1.0f;
    }
    if (engine && !engine->offload_gguf_metadata_json.empty()) {
        double direct_scale = 0.0;
        if ((moe_json_get_double_local(engine->offload_gguf_metadata_json, "attention_scale", &direct_scale) ||
             moe_json_get_double_local(engine->offload_gguf_metadata_json, "attn_scale", &direct_scale) ||
             moe_json_get_double_local(engine->offload_gguf_metadata_json, "attention_score_scale", &direct_scale) ||
             moe_json_get_double_local(engine->offload_gguf_metadata_json, "qk_scale", &direct_scale)) &&
            direct_scale > 0.0 && direct_scale <= 1024.0 && std::isfinite(direct_scale)) {
            if (source_out) {
                *source_out = "metadata_scale";
            }
            return (float)direct_scale;
        }
    }
    float query_pre_attn_scalar = engine ? engine->model_config_query_pre_attn_scalar : 0.0f;
    if (!(query_pre_attn_scalar > 0.0f) && engine && !engine->offload_gguf_metadata_json.empty()) {
        double value = 0.0;
        if (moe_json_get_double_local(engine->offload_gguf_metadata_json, "query_pre_attn_scalar", &value) ||
            moe_json_get_double_local(engine->offload_gguf_metadata_json, "attention_query_pre_attn_scalar", &value) ||
            moe_json_get_double_local(engine->offload_gguf_metadata_json, "attn_query_pre_attn_scalar", &value)) {
            if (value > 0.0 && value <= 65536.0 && std::isfinite(value)) {
                query_pre_attn_scalar = (float)value;
            }
        }
    }
    if (query_pre_attn_scalar > 0.0f && std::isfinite(query_pre_attn_scalar)) {
        float result = 1.0f / std::sqrt(query_pre_attn_scalar);
        if (!std::isfinite(result)) {
            if (source_out) {
                *source_out = "fallback";
            }
            return 1.0f;
        }
        if (source_out) {
            *source_out = "query_pre_attn_scalar";
        }
        return result;
    }
    if (moe_layer_has_qk_norm_scaled_standard_layout(engine, layer)) {
        if (source_out) {
            *source_out = "qk_norm_layout";
        }
        return 1.0f;
    }
    float result = 1.0f / std::sqrt((float)head_dim);
    if (!std::isfinite(result)) {
        if (source_out) {
            *source_out = "fallback";
        }
        return 1.0f;
    }
    return result;
}

static int moe_attention_score_scale_source_is_explicit(
    const char* source
) {
    return source &&
        (std::strcmp(source, "metadata_scale") == 0 ||
         std::strcmp(source, "query_pre_attn_scalar") == 0);
}

static int moe_attention_score_scale_violates_qk_norm_contract(
    const moe_pc_engine* engine,
    int32_t layer,
    float scale,
    const char* source
) {
    if (!moe_layer_has_qk_norm_scaled_standard_layout(engine, layer) ||
        moe_attention_score_scale_source_is_explicit(source)) {
        return 0;
    }
    return !(std::isfinite(scale) && std::fabs(scale - 1.0f) <= 1.0e-6f);
}

static void moe_standard_attention_rope_params(
    const moe_pc_engine* engine,
    int32_t layer,
    int has_v,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t* io_rope_dim,
    float* io_rope_theta
) {
    if (!io_rope_dim || !io_rope_theta || head_dim == 0) {
        return;
    }
    if (engine &&
        engine->model_config_attention_valid.load(std::memory_order_relaxed) &&
        *io_rope_dim > 0 && *io_rope_dim <= head_dim &&
        *io_rope_theta > 0.0f) {
        return;
    }
    if (!moe_layer_has_qk_norm_scaled_standard_layout(engine, layer)) {
        return;
    }
    double partial_value = 0.0;
    if (!engine || engine->offload_gguf_metadata_json.empty() ||
        !moe_json_get_double_local(engine->offload_gguf_metadata_json, "partial_rotary_factor", &partial_value) ||
        partial_value <= 0.0 || partial_value > 1.0) {
        return;
    }
    double full_theta = 0.0;
    double sliding_theta = 0.0;
    const int has_full_theta =
        moe_json_get_double_local(engine->offload_gguf_metadata_json, "full_rope_theta", &full_theta) ||
        moe_json_get_double_local(engine->offload_gguf_metadata_json, "full_attention_rope_theta", &full_theta);
    const int has_sliding_theta =
        moe_json_get_double_local(engine->offload_gguf_metadata_json, "sliding_rope_theta", &sliding_theta) ||
        moe_json_get_double_local(engine->offload_gguf_metadata_json, "sliding_attention_rope_theta", &sliding_theta);
    const int full_attention_layout =
        !has_v ||
        (head_dim > 256u) ||
        (num_heads > 0 && num_kv_heads > 0 && num_kv_heads * 4u <= num_heads);
    if (full_attention_layout) {
        uint32_t partial_dim = (uint32_t)std::round((float)head_dim * (float)partial_value);
        if (partial_dim < 2u) partial_dim = 2u;
        if (partial_dim > head_dim) partial_dim = head_dim;
        *io_rope_dim = partial_dim;
        if (has_full_theta && full_theta > 0.0) {
            *io_rope_theta = (float)full_theta;
        }
    } else if (has_sliding_theta && sliding_theta > 0.0) {
        *io_rope_dim = head_dim;
        *io_rope_theta = (float)sliding_theta;
    }
}

static int moe_standard_kv_cache_ensure(
    moe_pc_engine* engine,
    moe_decode_scratch* s,
    uint32_t position,
    uint32_t kv_entry_dim
) {
    if (!engine || !s || kv_entry_dim == 0) {
        return 0;
    }
    // BUGFIX 778: Check position for overflow ★★★
    if (position > UINT32_MAX / 2) {
        return 0;
    }
    if (position < s->kv_cache.max_seq_len) {
        return 1;
    }
    if (!s->kv_cache.ensure_capacity(position)) {
        return 0;
    }
    moe_kv_cache_track_ram(engine, &s->kv_cache);
    return position < s->kv_cache.max_seq_len ? 1 : 0;
}

static int moe_layer_standard_attention_f32(
    moe_pc_engine* engine,
    int32_t layer,
    moe_decode_scratch* s,
    float* hidden,
    uint32_t hidden_size,
    uint32_t position,
    const moe_attn_config_t& cfg,
    const float* norm_input = nullptr,
    int add_residual = 1
) {
    const uint64_t total_start_us = moe_profile_now_us();
    const uint64_t map_start_us = moe_profile_now_us();
    uint32_t num_heads = cfg.num_heads;
    uint32_t num_kv_heads = cfg.num_kv_heads;
    uint32_t head_dim = cfg.head_dim;
    uint32_t v_head_dim = cfg.v_head_dim ? cfg.v_head_dim : cfg.head_dim;
    moe_raw_tensor_cache_entry q_shape{};
    moe_raw_tensor_cache_entry k_shape{};
    moe_raw_tensor_cache_entry v_shape{};
    moe_raw_tensor_cache_entry o_shape{};
    moe_raw_tensor_cache_entry q_norm_shape{};
    moe_raw_tensor_cache_entry k_norm_shape{};
    const int has_q = get_cached_raw_tensor(engine, layer, moe_RAW_TENSOR_Q_A_PROJ, &q_shape);
    const int has_k = get_cached_raw_tensor(engine, layer, moe_RAW_TENSOR_KV_A_PROJ_WITH_MQA, &k_shape);
    const int has_v = get_cached_raw_tensor(engine, layer, moe_RAW_TENSOR_KV_B_PROJ, &v_shape);
    const int has_o = get_cached_raw_tensor(engine, layer, moe_RAW_TENSOR_O_PROJ, &o_shape);
    const int has_q_norm = get_cached_raw_tensor(engine, layer, moe_RAW_TENSOR_Q_A_LAYER_NORM, &q_norm_shape);
    const int has_k_norm = get_cached_raw_tensor(engine, layer, moe_RAW_TENSOR_KV_A_LAYER_NORM, &k_norm_shape);
    if (!has_q || !has_k || !has_o) {
        std::fprintf(stderr, "[storagellm attn] missing q/k/o layer=%d q=%d k=%d v=%d o=%d\n",
            layer, has_q, has_k, has_v, has_o);
        return 0;
    }
    if (has_q_norm && has_k_norm &&
        q_norm_shape.rows && q_norm_shape.rows == k_norm_shape.rows &&
        q_shape.rows % q_norm_shape.rows == 0 &&
        k_shape.rows % q_norm_shape.rows == 0) {
        head_dim = q_norm_shape.rows;
    } else if (head_dim == 0 && num_heads && q_shape.rows % num_heads == 0) {
        head_dim = q_shape.rows / num_heads;
    }
    if (head_dim && q_shape.rows % head_dim == 0) {
        num_heads = q_shape.rows / head_dim;
    }
    if (head_dim && k_shape.rows % head_dim == 0) {
        num_kv_heads = k_shape.rows / head_dim;
    }
    if (has_v && num_kv_heads && v_shape.rows % num_kv_heads == 0) {
        v_head_dim = v_shape.rows / num_kv_heads;
    } else if (num_heads && o_shape.cols % num_heads == 0) {
        v_head_dim = o_shape.cols / num_heads;
    }
    const uint32_t q_dim = q_shape.rows;
    const uint32_t k_dim = k_shape.rows;
    const uint32_t kv_v_dim = has_v ? v_shape.rows : num_kv_heads * v_head_dim;
    const uint32_t v_total_dim = o_shape.cols ? o_shape.cols : num_heads * v_head_dim;
    const uint32_t kv_entry_dim = k_dim + kv_v_dim;
    const uint32_t base_rope_dim = std::min<uint32_t>(cfg.qk_rope_head_dim ? cfg.qk_rope_head_dim : head_dim, head_dim);
    uint32_t rope_dim = base_rope_dim;
    float rope_theta = cfg.rope_theta;
    moe_standard_attention_rope_params(
        engine, layer, has_v, num_heads, num_kv_heads, head_dim, &rope_dim, &rope_theta);
    const int use_dynamic_rope =
        moe_engine_uses_hybrid_local_global_attention(engine) ||
        rope_dim != base_rope_dim ||
        std::fabs(rope_theta - cfg.rope_theta) > 1.0e-6f;
    moe_profile_log(
        "attn_standard_cfg",
        "layer=%d pos=%u heads=%u kv_heads=%u head_dim=%u v_head_dim=%u q_dim=%u k_dim=%u kv_v_dim=%u v_total=%u rope_dim=%u theta=%.6g has_v=%d dynamic_rope=%d",
        layer,
        position,
        num_heads,
        num_kv_heads,
        head_dim,
        v_head_dim,
        q_dim,
        k_dim,
