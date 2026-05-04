    {
        std::unique_lock<std::shared_mutex> lock(engine->text_mention_cache_mutex);
        engine->metadata_mention_cache.emplace(key, result);
    }
    return result;
}

static float moe_embedding_scale_value(const moe_pc_engine* engine, uint32_t hidden_size) {
    if (!engine || hidden_size == 0 || hidden_size > 100000u) {
        return 1.0f;
    }
    const uint32_t cached_hidden = engine->cached_embedding_scale_hidden.load(std::memory_order_acquire);
    const uint32_t cached_bits = engine->cached_embedding_scale_bits.load(std::memory_order_acquire);
    if (cached_hidden == hidden_size && cached_bits != 0u) {
        return moe_f32_from_bits(cached_bits);
    }

    double value = 0.0;
    float scale = 1.0f;
    if (!engine->offload_gguf_metadata_json.empty() &&
        (moe_json_get_double_local(engine->offload_gguf_metadata_json, "embedding_scale", &value) ||
         moe_json_get_double_local(engine->offload_gguf_metadata_json, "embed_scale", &value) ||
         moe_json_get_double_local(engine->offload_gguf_metadata_json, "scale_emb", &value) ||
         moe_json_get_double_local(engine->offload_gguf_metadata_json, "scale_embedding", &value)) &&
        value > 0.0) {
        scale = (float)value;
    }
    else if (moe_engine_graph_ir_mentions(engine, "sqrt_hidden_size") ||
        moe_engine_graph_ir_mentions(engine, "scale_embedding") ||
        moe_engine_metadata_mentions(engine, "sqrt_hidden_size") ||
        moe_engine_metadata_mentions(engine, "scale_embedding")) {
        scale = std::sqrt((float)hidden_size);
    }
    else if (moe_engine_has_common_tensor_suffix(engine, 0, "layer_output_scale.weight") &&
        moe_engine_has_common_tensor_suffix(engine, 0, "ffn_gate_inp.scale")) {
        scale = std::sqrt((float)hidden_size);
    }

    engine->cached_embedding_scale_hidden.store(hidden_size, std::memory_order_release);
    engine->cached_embedding_scale_bits.store(moe_f32_bits(scale), std::memory_order_release);
    return scale;
}

static void moe_apply_embedding_scale_f32(
    const moe_pc_engine* engine,
    const moe_storage_constants_t* c,
    float* hidden
) {
    if (!engine || !c || !hidden || c->hidden_size == 0) {
        return;
    }
    const float scale = moe_embedding_scale_value(engine, c->hidden_size);
    if (scale == 1.0f) {
        return;
    }
    for (uint32_t i = 0; i < c->hidden_size; ++i) {
        hidden[i] *= scale;
    }
}

static int moe_engine_uses_gemma_rmsnorm_unit_offset(const moe_pc_engine* engine) {
    if (!engine) {
        return 0;
    }
    if (moe_engine_metadata_mentions(engine, "gemma4") ||
        moe_engine_metadata_mentions(engine, "gemma-4") ||
        moe_engine_metadata_mentions(engine, "gemma_4") ||
        moe_engine_graph_ir_mentions(engine, "gemma4") ||
        moe_engine_graph_ir_mentions(engine, "gemma-4") ||
        moe_engine_graph_ir_mentions(engine, "gemma_4")) {
        return 0;
    }
    return moe_engine_metadata_mentions(engine, "gemma") ||
        moe_engine_graph_ir_mentions(engine, "gemma");
}

static int moe_engine_gelu_activation_mode(const moe_pc_engine* engine) {
    if (!engine) {
        return 0;
    }
    const int cached = engine->cached_gelu_activation_mode.load(std::memory_order_acquire);
    if (cached >= 0) {
        return cached;
    }
    int mode = 0;
    if (moe_engine_graph_ir_mentions(engine, "gelu_pytorch_tanh") ||
        moe_engine_graph_ir_mentions(engine, "gelu_tanh") ||
        moe_engine_metadata_mentions(engine, "gelu_pytorch_tanh") ||
        moe_engine_metadata_mentions(engine, "gelu_tanh")) {
        mode = 2;
    } else if (moe_engine_graph_ir_mentions(engine, "gelu") ||
        moe_engine_metadata_mentions(engine, "gelu")) {
        mode = 1;
    }
    engine->cached_gelu_activation_mode.store(mode, std::memory_order_release);
    return mode;
}

static int moe_engine_prefers_gelu_activation(const moe_pc_engine* engine) {
    return moe_engine_gelu_activation_mode(engine) != 0 ? 1 : 0;
}

static float moe_final_logit_softcap_value(const moe_pc_engine* engine) {
    double value = 0.0;
    if (engine &&
        !engine->offload_gguf_metadata_json.empty() &&
        (moe_json_get_double_local(engine->offload_gguf_metadata_json, "final_logit_softcapping", &value) ||
         moe_json_get_double_local(engine->offload_gguf_metadata_json, "final_logit_softcap", &value) ||
         moe_json_get_double_local(engine->offload_gguf_metadata_json, "logit_softcap", &value)) &&
        value > 0.0) {
        return (float)value;
    }
    return 0.0f;
}

static float moe_apply_final_logit_softcap(const moe_pc_engine* engine, float score) {
    const float cap = moe_final_logit_softcap_value(engine);
    if (!(cap > 0.0f) || !std::isfinite(score)) {
        return score;
    }
    return std::tanh(score / cap) * cap;
}

static uint64_t moe_weight_format_row_bytes_for_cols(
    uint8_t weight_format,
    uint32_t cols
) {
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q8_0) {
        return ((uint64_t)cols + 31ull) / 32ull * 34ull;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q8_1) {
        return ((uint64_t)cols + 31ull) / 32ull * 36ull;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q8_K) {
        return ((uint64_t)cols + 255ull) / 256ull * 292ull;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q4_0) {
        return ((uint64_t)cols + 31ull) / 32ull * 18ull;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q4_1) {
        return ((uint64_t)cols + 31ull) / 32ull * 20ull;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q5_0) {
        return ((uint64_t)cols + 31ull) / 32ull * 22ull;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q5_1) {
        return ((uint64_t)cols + 31ull) / 32ull * 24ull;
    }
    if (weight_format == moe_WEIGHT_ENCODING_MXFP4) {
        return ((uint64_t)cols + 31ull) / 32ull * 17ull;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q2_K) {
        return ((uint64_t)cols + 255ull) / 256ull * 84ull;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q3_K) {
        return ((uint64_t)cols + 255ull) / 256ull * 110ull;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q4_K) {
        return ((uint64_t)cols + 255ull) / 256ull * 144ull;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q5_K) {
        return ((uint64_t)cols + 255ull) / 256ull * 176ull;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q6_K) {
        return ((uint64_t)cols + 255ull) / 256ull * 210ull;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_IQ2_XXS) {
        return ((uint64_t)cols + 255ull) / 256ull * 66ull;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_IQ2_XS) {
        return ((uint64_t)cols + 255ull) / 256ull * 74ull;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_IQ2_S) {
        return ((uint64_t)cols + 255ull) / 256ull * 82ull;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_IQ3_XXS) {
        return ((uint64_t)cols + 255ull) / 256ull * 98ull;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_IQ3_S) {
        return ((uint64_t)cols + 255ull) / 256ull * 110ull;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_IQ4_NL) {
        return ((uint64_t)cols + 31ull) / 32ull * 18ull;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_IQ4_XS) {
        return ((uint64_t)cols + 255ull) / 256ull * 136ull;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_IQ1_S) {
        return ((uint64_t)cols + 255ull) / 256ull * 50ull;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_IQ1_M) {
        return ((uint64_t)cols + 255ull) / 256ull * 56ull;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_TQ1_0) {
        return ((uint64_t)cols + 255ull) / 256ull * 54ull;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_TQ2_0) {
        return ((uint64_t)cols + 255ull) / 256ull * 66ull;
    }
    return 0;
}

static uint8_t moe_effective_weight_format_for_row_bytes(
    const moe_raw_tensor_cache_entry& raw,
    uint32_t cols
) {
    if (!raw.weight_row_bytes || cols == 0 ||
        raw.weight_format != moe_WEIGHT_ENCODING_MXFP4) {
        return raw.weight_format;
    }
    const uint64_t actual = raw.weight_row_bytes;
    if (actual == moe_weight_format_row_bytes_for_cols(raw.weight_format, cols)) {
        return raw.weight_format;
    }
    static const uint8_t candidates[] = {
        moe_WEIGHT_ENCODING_GGUF_Q8_0,
        moe_WEIGHT_ENCODING_GGUF_Q8_1,
        moe_WEIGHT_ENCODING_GGUF_Q8_K,
        moe_WEIGHT_ENCODING_GGUF_Q4_0,
        moe_WEIGHT_ENCODING_GGUF_Q4_1,
        moe_WEIGHT_ENCODING_GGUF_Q5_0,
        moe_WEIGHT_ENCODING_GGUF_Q5_1,
        moe_WEIGHT_ENCODING_GGUF_Q2_K,
        moe_WEIGHT_ENCODING_GGUF_Q3_K,
        moe_WEIGHT_ENCODING_GGUF_Q4_K,
        moe_WEIGHT_ENCODING_GGUF_Q5_K,
        moe_WEIGHT_ENCODING_GGUF_Q6_K,
        moe_WEIGHT_ENCODING_GGUF_IQ2_XXS,
        moe_WEIGHT_ENCODING_GGUF_IQ2_XS,
        moe_WEIGHT_ENCODING_GGUF_IQ2_S,
        moe_WEIGHT_ENCODING_GGUF_IQ3_XXS,
        moe_WEIGHT_ENCODING_GGUF_IQ3_S,
        moe_WEIGHT_ENCODING_GGUF_IQ4_NL,
        moe_WEIGHT_ENCODING_GGUF_IQ4_XS,
        moe_WEIGHT_ENCODING_GGUF_IQ1_S,
        moe_WEIGHT_ENCODING_GGUF_IQ1_M,
        moe_WEIGHT_ENCODING_GGUF_TQ1_0,
        moe_WEIGHT_ENCODING_GGUF_TQ2_0
    };
    uint8_t match = raw.weight_format;
    uint32_t matches = 0;
    for (uint8_t candidate : candidates) {
        if (actual == moe_weight_format_row_bytes_for_cols(candidate, cols)) {
            match = candidate;
            ++matches;
        }
    }
    if (matches == 1u) {
        return match;
    }
    return raw.weight_format;
}

static uint64_t moe_raw_tensor_row_bytes(
    const moe_raw_tensor_cache_entry& raw,
    uint32_t cols
) {
    if (raw.weight_row_bytes) {
        return raw.weight_row_bytes;
    }
    const uint64_t quant_bytes = moe_weight_format_row_bytes_for_cols(raw.weight_format, cols);
    if (quant_bytes) {
        return quant_bytes;
    }
    const uint32_t elem = moe_raw_scalar_size(raw.scalar_encoding);
    return (uint64_t)cols * (uint64_t)elem;
}

static float moe_raw_tensor_dot_row(
    const moe_raw_tensor_cache_entry& raw,
    uint32_t row,
    const float* hidden,
    uint32_t hidden_count
) {
    const uint32_t stride_cols = raw.cols ? raw.cols : hidden_count;
    const uint64_t row_bytes = moe_raw_tensor_row_bytes(raw, stride_cols);
    // BUGFIX 494: row * row_bytes overflow 泥댄겕
    if (!raw.base || !hidden || row >= raw.rows || row_bytes == 0 ||
        row > UINT64_MAX / row_bytes ||
        (uint64_t)row * row_bytes > raw.bytes ||
        row_bytes > raw.bytes - (uint64_t)row * row_bytes) {
        return 0.0f;
    }
    const uint8_t* row_ptr = raw.base + (uint64_t)row * row_bytes;
    const uint8_t effective_weight_format = moe_effective_weight_format_for_row_bytes(raw, stride_cols);
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_Q8_0) {
        return moe_dot_gguf_q8_0_row(row_ptr, hidden, hidden_count);
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_Q8_1) {
        return moe_dot_gguf_q8_1_row(row_ptr, hidden, hidden_count);
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_Q8_K) {
        return moe_dot_gguf_q8_k_row(row_ptr, hidden, hidden_count);
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_Q4_0) {
        return moe_dot_gguf_q4_0_row(row_ptr, hidden, hidden_count);
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_Q4_1) {
        return moe_dot_gguf_q4_1_row(row_ptr, hidden, hidden_count);
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_Q5_0) {
        return moe_dot_gguf_q5_0_row(row_ptr, hidden, hidden_count);
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_Q5_1) {
        return moe_dot_gguf_q5_1_row(row_ptr, hidden, hidden_count);
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_MXFP4) {
        return moe_dot_gguf_mxfp4_row(row_ptr, hidden, hidden_count);
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_Q2_K) {
        return moe_dot_gguf_q2_k_row(row_ptr, hidden, hidden_count);
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_Q3_K) {
