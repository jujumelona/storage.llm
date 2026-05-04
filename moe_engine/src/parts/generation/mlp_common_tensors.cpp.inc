static const moe_gguf_common_tensor_record* moe_find_layer_common_tensor(
    moe_pc_engine* engine,
    int32_t layer,
    const char* const* suffixes,
    size_t suffix_count,
    moe_raw_tensor_desc* out_desc
) {
    if (!engine || !suffixes || layer < 0) {
        return nullptr;
    }
    std::vector<std::string> names;
    moe_push_layer_tensor_candidates(&names, layer, suffixes, suffix_count);
    return moe_common_tensor_find_first_raw_compatible(engine, names, out_desc);
}

static uint32_t moe_fnv1a_hash_string(const char* str) {
    uint32_t hash = 2166136261u;
    while (*str) {
        hash ^= (uint8_t)*str++;
        hash *= 16777619u;
    }
    return hash;
}

static int moe_map_layer_common_tensor(
    moe_pc_engine* engine,
    int32_t layer,
    const char* const* suffixes,
    size_t suffix_count,
    moe_raw_tensor_cache_entry* out
) {
    if (!engine || !out || suffix_count == 0) {
        return 0;
    }
    // Fix 18: Cache tensor mapping to avoid std::string vector allocation per token
    const uint32_t suffix_hash = moe_fnv1a_hash_string(suffixes[0]);
    const uint64_t key = ((uint64_t)(uint32_t)layer << 32) | suffix_hash | 0x80000000ull;
    {
        std::shared_lock<std::shared_mutex> lock(engine->raw_tensor_cache_mutex);
        auto found = engine->raw_tensor_cache.find(key);
        if (found != engine->raw_tensor_cache.end() && found->second.base) {
            *out = found->second;
            return 1;
        }
    }
    std::vector<std::string> names;
    moe_push_layer_tensor_candidates(&names, layer, suffixes, suffix_count);
    moe_raw_tensor_desc desc{};
    const moe_gguf_common_tensor_record* rec =
        moe_common_tensor_find_first_raw_compatible(engine, names, &desc);
    if (!rec) {
        return 0;
    }
    if (!map_common_tensor_record(engine, *rec, desc, out)) {
        return 0;
    }
    {
        std::unique_lock<std::shared_mutex> lock(engine->raw_tensor_cache_mutex);
        engine->raw_tensor_cache[key] = *out;
    }
    return 1;
}

static int moe_common_linear_f32(
    moe_pc_engine* engine,
    const moe_raw_tensor_cache_entry& raw,
    const float* input,
    uint32_t input_count,
    float* output,
    uint32_t output_count
) {
    if (!engine || !input || !output ||
        raw.cols != input_count || raw.rows != output_count) {
        return 0;
    }
    // BUGFIX 506: Check raw.base validity before computing row_bytes ??
    // Problem: raw.base can be null if tensor mapping failed
    // moe_raw_tensor_row_bytes may dereference raw.base internally
    // Solution: Check raw.base before use
    if (!raw.base) {
        return 0;
    }
    const uint64_t row_bytes = moe_raw_tensor_row_bytes(raw, raw.cols);
    if (row_bytes == 0 || raw.bytes < row_bytes * (uint64_t)raw.rows) {
        return 0;
    }
    moe_profile_parallel_label_scope rows_label("common_linear");
    return moe_parallel_rows(engine, raw.rows, 0, [&](uint32_t row) {
        output[row] = moe_raw_tensor_dot_row(raw, row, input, input_count);
        return 1;
    });
}

static inline float moe_generation_gelu_tanh(float x) {
    const float k = 0.7978845608028654f;
    return 0.5f * x * (1.0f + std::tanh(k * (x + 0.044715f * x * x * x)));
}

static inline float moe_generation_gated_activation_cached(int activation_mode, float gate, float up) {
    if (activation_mode == 2) {
        return moe_generation_gelu_tanh(gate) * up;
    }
    return (activation_mode == 1 ? moe_generation_gelu(gate) : moe_generation_silu(gate)) * up;
}

static int moe_raw_tensor_scalar_at_f32(
    const moe_raw_tensor_cache_entry& raw,
    uint64_t offset,
    float* out
) {
    if (!raw.base || !out) {
        return 0;
    }
    const uint32_t elem = moe_raw_scalar_size(raw.scalar_encoding);
    if (elem == 0 || offset > UINT64_MAX / (uint64_t)elem) {
        return 0;
    }
    const uint64_t byte_offset = offset * (uint64_t)elem;
    if (byte_offset > raw.bytes || (uint64_t)elem > raw.bytes - byte_offset) {
        return 0;
    }
    *out = moe_raw_scalar_to_f32(raw.base + byte_offset, raw.scalar_encoding);
    return 1;
}

static uint64_t moe_raw_tensor_scalar_count(const moe_raw_tensor_cache_entry& raw) {
    const uint32_t elem = moe_raw_scalar_size(raw.scalar_encoding);
    if (!elem) {
        return 0;
    }
    uint64_t logical_count = 0;
    if (raw.cols) {
        if (raw.rows > UINT64_MAX / (uint64_t)raw.cols) {
            return raw.bytes / (uint64_t)elem;
        }
        logical_count = (uint64_t)raw.rows * (uint64_t)raw.cols;
    } else {
        logical_count = raw.rows;
    }
    if (logical_count && logical_count <= raw.bytes / (uint64_t)elem) {
        return logical_count;
    }
    return raw.bytes / (uint64_t)elem;
}

static int moe_apply_rmsnorm_common_weight_f32(
    const moe_pc_engine* engine,
    const moe_raw_tensor_cache_entry& raw,
    const float* input,
    uint32_t count,
    float* output,
    float eps
) {
    // BUGFIX 208: Division by zero when count == 0
    // Problem: ss / (double)count crashes if count == 0
    // Solution: Check count == 0 BEFORE division (already done), but clarify comment
    if (!input || !output || count == 0 || moe_raw_tensor_scalar_count(raw) < count) {
        return 0;
    }
    double ss = 0.0;
    for (uint32_t i = 0; i < count; ++i) {
        ss += (double)input[i] * (double)input[i];
    }
    // Safe: count > 0 guaranteed by check above
    const float norm = 1.0f / std::sqrt((float)(ss / (double)count) + (eps > 0.0f ? eps : 1.0e-6f));
    const int unit_offset = moe_engine_uses_gemma_rmsnorm_unit_offset(engine);
    for (uint32_t i = 0; i < count; ++i) {
        float w = 0.0f;
        if (!moe_raw_tensor_scalar_at_f32(raw, i, &w)) {
            return 0;
        }
        output[i] = input[i] * norm * (unit_offset ? (1.0f + w) : w);
    }
    return 1;
}

static int moe_layer_rmsnorm_suffixes_f32(
    moe_pc_engine* engine,
    int32_t layer,
    const char* const* suffixes,
    size_t suffix_count,
    const float* input,
    uint32_t hidden_size,
    float* output,
    float eps
) {
    moe_raw_tensor_cache_entry raw{};
    const int mapped = moe_map_layer_common_tensor(engine, layer, suffixes, suffix_count, &raw);
    moe_profile_log(
        "rmsnorm_common_map",
        "layer=%d suffix0=%s mapped=%d rows=%u cols=%u bytes=%llu hidden=%u input=%p output=%p",
        layer,
        (suffixes && suffix_count > 0 && suffixes[0]) ? suffixes[0] : "",
        mapped,
        raw.rows,
        raw.cols,
        (unsigned long long)raw.bytes,
        hidden_size,
        (const void*)input,
        (void*)output);
    if (!mapped) {
        return 0;
    }
    const int ok = moe_apply_rmsnorm_common_weight_f32(engine, raw, input, hidden_size, output, eps);
    moe_profile_log(
        "rmsnorm_common_apply",
        "layer=%d suffix0=%s ok=%d hidden=%u scalar=%u",
        layer,
        (suffixes && suffix_count > 0 && suffixes[0]) ? suffixes[0] : "",
        ok,
        hidden_size,
        (unsigned)raw.scalar_encoding);
    return ok;
}

static int moe_layer_ffn_norm_f32(
    moe_pc_engine* engine,
    int32_t layer,
    const float* input,
    uint32_t hidden_size,
    float* output,
    float eps
) {
    static const char* const suffixes[] = {
        "ffn_norm.weight",
        "ffn_pre_norm.weight",
        "mlp_norm.weight"
    };
    return moe_layer_rmsnorm_suffixes_f32(
        engine, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]),
        input, hidden_size, output, eps);
}

static int moe_layer_moe_norm_f32(
    moe_pc_engine* engine,
    int32_t layer,
    const float* input,
    uint32_t hidden_size,
    float* output,
    float eps
) {
    static const char* const suffixes[] = {
        "pre_ffw_norm_2.weight",
        "ffn_pre_norm_2.weight",
        "moe_norm.weight"
    };
    return moe_layer_rmsnorm_suffixes_f32(
        engine, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]),
        input, hidden_size, output, eps);
}

static int moe_layer_post_attention_output_norm_f32(
    moe_pc_engine* engine,
    int32_t layer,
    const float* input,
    uint32_t hidden_size,
    float* output,
    float eps
) {
    static const char* const suffixes[] = {
        "post_attention_norm.weight",
        "post_attention_layernorm.weight",
        "post_attention_layer_norm.weight",
        "post_attn_norm.weight"
    };
    return moe_layer_rmsnorm_suffixes_f32(
        engine, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]),
        input, hidden_size, output, eps);
}

// Cache split/post-norm detection per layer; layer 0 must not decide every layer.
static int moe_layer_uses_split_post_norm_block(const moe_pc_engine* engine, int32_t layer) {
    if (!engine || layer < 0) {
        return 0;
    }
    static thread_local const moe_pc_engine* tls_cached_engine = nullptr;
    static thread_local uint64_t tls_cached_generation = 0;
    static thread_local std::vector<int8_t> tls_cached_results;
    const uint64_t generation = engine->engine_generation;
    const size_t layer_index = (size_t)layer;
    int8_t* cached = nullptr;
    if (layer_index < 10000u) {
        if (tls_cached_engine != engine || tls_cached_generation != generation) {
            tls_cached_engine = engine;
            tls_cached_generation = generation;
            tls_cached_results.clear();
        }
        if (tls_cached_results.size() <= layer_index) {
            tls_cached_results.resize(layer_index + 1u, -1);
        }
        cached = &tls_cached_results[layer_index];
        if (*cached >= 0) {
            return *cached;
        }
    }
    int result = 0;
    if (moe_engine_has_common_tensor_suffix(engine, layer, "post_ffw_norm.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "post_ffw_norm_1.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "post_ffw_norm_2.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "layer_output_scale.weight")) {
        result = 1;
    } else {
        result = (moe_engine_graph_ir_mentions(engine, "post_ffw_norm") ||
            moe_engine_graph_ir_mentions(engine, "layer_output_scale")) ? 1 : 0;
    }
    if (cached) {
        *cached = result ? 1 : 0;
    }
    return result;
}

static size_t moe_graph_ir_first_pos_any(
    const moe_pc_engine* engine,
    const char* const* needles,
    size_t needle_count,
    size_t begin = 0
) {
    if (!engine || !needles || engine->offload_graph_ir_json.empty()) {
        return std::string::npos;
    }
    size_t best = std::string::npos;
    for (size_t i = 0; i < needle_count; ++i) {
        if (!needles[i] || !needles[i][0]) {
            continue;
        }
        const size_t pos = moe_ascii_find_ci(engine->offload_graph_ir_json, needles[i], begin);
        if (pos != std::string::npos && (best == std::string::npos || pos < best)) {
            best = pos;
        }
    }
    return best;
}

static int moe_graph_ir_attention_residual_before_ffn_norm(const moe_pc_engine* engine, int32_t layer) {
    if (!engine || !engine->offload_graph_ir_ready.load(std::memory_order_relaxed)) {
        return 0;
    }
    if (moe_engine_has_common_tensor_suffix(engine, layer, "post_attention_norm.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "post_attention_layernorm.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "post_attention_layer_norm.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "post_attn_norm.weight")) {
        return 0;
    }
    static const char* const post_attention_norm_ops[] = {
        "\"name\":\"post_attention_norm\"",
        "\"name\": \"post_attention_norm\"",
        "\"name\":\"post_attention_layernorm\"",
        "\"name\": \"post_attention_layernorm\"",
        "\"name\":\"post_attention_layer_norm\"",
        "\"name\": \"post_attention_layer_norm\""
    };
    static const char* const attention_residual_ops[] = {
        "\"name\":\"attention_residual\"",
        "\"name\": \"attention_residual\"",
        "attention_residual"
    };
    static const char* const ffn_norm_ops[] = {
        "\"name\":\"ffn_norm\"",
        "\"name\": \"ffn_norm\"",
        "ffn_norm"
    };
    const size_t residual_pos = moe_graph_ir_first_pos_any(
        engine, attention_residual_ops, sizeof(attention_residual_ops) / sizeof(attention_residual_ops[0]));
    if (residual_pos == std::string::npos) {
        return 0;
    }
    const size_t ffn_norm_pos = moe_graph_ir_first_pos_any(
        engine, ffn_norm_ops, sizeof(ffn_norm_ops) / sizeof(ffn_norm_ops[0]), residual_pos);
    if (ffn_norm_pos == std::string::npos) {
        return 0;
    }
    const size_t explicit_post_norm_pos = moe_graph_ir_first_pos_any(
        engine, post_attention_norm_ops, sizeof(post_attention_norm_ops) / sizeof(post_attention_norm_ops[0]));
    if (explicit_post_norm_pos != std::string::npos && explicit_post_norm_pos < residual_pos) {
        return 0;
    }
    return 1;
}

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
    static const char* const suffixes[] = {
        "post_ffw_norm.weight",
        "ffn_post_norm.weight"
    };
    return moe_layer_post_ffw_norm_suffixes_f32(
        engine, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]),
        input, hidden_size, output, eps);
}

static int moe_layer_post_ffw_norm1_f32(
    moe_pc_engine* engine,
    int32_t layer,
    const float* input,
    uint32_t hidden_size,
    float* output,
    float eps
) {
    static const char* const suffixes[] = {
        "post_ffw_norm_1.weight",
        "ffn_post_norm_1.weight"
    };
    return moe_layer_post_ffw_norm_suffixes_f32(
        engine, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]),
        input, hidden_size, output, eps);
}

static int moe_layer_post_ffw_norm2_f32(
    moe_pc_engine* engine,
    int32_t layer,
    const float* input,
    uint32_t hidden_size,
    float* output,
    float eps
) {
    static const char* const suffixes[] = {
        "post_ffw_norm_2.weight",
        "ffn_post_norm_2.weight"
    };
    return moe_layer_post_ffw_norm_suffixes_f32(
        engine, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]),
        input, hidden_size, output, eps);
}

static int moe_layer_output_scale_f32(
    moe_pc_engine* engine,
    int32_t layer,
    float* hidden,
    uint32_t hidden_size
) {
    if (!engine || !hidden || hidden_size == 0) {
        return 0;
    }
    // BUGFIX 507: Check hidden_size upper bound to prevent overflow ??
    // Problem: hidden_size can be arbitrarily large from config
    // Loop for (i = 0; i < hidden_size; ++i) can run billions of iterations
    // Solution: Add reasonable upper bound check (e.g., 100000)
    if (hidden_size > 100000) {
        return 0;
    }
    static const char* const suffixes[] = {
        "layer_output_scale.weight",
        "layer_scalar.weight",
        "layer_scalar"
    };
    moe_raw_tensor_cache_entry raw{};
    if (!moe_map_layer_common_tensor(engine, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]), &raw)) {
        return 1;
    }
    const uint64_t count = moe_raw_tensor_scalar_count(raw);
    if (count == 1u) {
        float scale = 1.0f;
        if (!moe_raw_tensor_scalar_at_f32(raw, 0, &scale)) {
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
        if (!moe_raw_tensor_scalar_at_f32(raw, i, &scale)) {
            return 0;
        }
        hidden[i] *= scale;
    }
    return 1;
