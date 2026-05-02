// Bug #1 Fix: Full MLA attention with Q projection, RoPE, scaled dot-product.
// GLM4/Moe Multi-head Latent Attention (MLA):
//   KV path: hidden -> kv_a_proj(576) -> [kv_a_norm(512), mqa_rope(64)]
//            kv_a_norm -> kv_b_proj(28672) -> [K_nope(12288), V(16384)]
//   Q path:  hidden -> q_a_proj(2048) -> q_a_norm(2048) -> q_b_proj(12288+4096=16384)
//            -> [Q_nope(12288), Q_rope(4096)] split by heads
//   Score = softmax(Q_nope*K_nope + Q_rope*K_rope) * V -> o_proj -> attn output

// Keep this tied to the converted model metadata; tensor shapes alone do not determine theta.
struct moe_rope_runtime_row_cache {
    uint32_t dim = 0;
    uint32_t position = 0;
    uint32_t base_bits = 0;
    std::vector<float> cos_row;
    std::vector<float> sin_row;
};

static uint32_t moe_rope_float_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static int moe_rope_runtime_row(
    uint32_t dim,
    uint32_t position,
    float base,
    const float** out_cos,
    const float** out_sin
) {
    if (!out_cos || !out_sin || dim < 2) {
        return 0;
    }
    *out_cos = nullptr;
    *out_sin = nullptr;
    if (!(base > 0.0f)) {
        base = moe_ROPE_THETA;
    }
    const uint32_t half_dim = dim / 2u;
    const uint32_t base_bits = moe_rope_float_bits(base);
    static thread_local std::vector<moe_rope_runtime_row_cache> caches;
    static thread_local size_t next_cache_slot = 0;
    for (moe_rope_runtime_row_cache& cached : caches) {
        if (cached.dim == dim &&
            cached.position == position &&
            cached.base_bits == base_bits &&
            cached.cos_row.size() == half_dim &&
            cached.sin_row.size() == half_dim) {
            *out_cos = cached.cos_row.data();
            *out_sin = cached.sin_row.data();
            return 1;
        }
    }
    moe_rope_runtime_row_cache* slot = nullptr;
    if (caches.size() < 8u) {
        try {
            caches.push_back(moe_rope_runtime_row_cache{});
        } catch (...) {
            return 0;
        }
        slot = &caches.back();
    } else {
        slot = &caches[next_cache_slot++ % caches.size()];
    }
    try {
        slot->cos_row.resize(half_dim);
        slot->sin_row.resize(half_dim);
    } catch (...) {
        return 0;
    }
    slot->dim = dim;
    slot->position = position;
    slot->base_bits = base_bits;
    for (uint32_t j = 0; j < half_dim; ++j) {
        const uint32_t i = j * 2u;
        const float freq = 1.0f / std::pow(base, (float)i / (float)dim);
        const float theta = (float)position * freq;
        slot->cos_row[j] = std::cos(theta);
        slot->sin_row[j] = std::sin(theta);
    }
    *out_cos = slot->cos_row.data();
    *out_sin = slot->sin_row.data();
    return 1;
}

static void moe_rope_inplace(
    float* data,
    uint32_t dim,
    uint32_t position,
    const std::vector<float>& cos_table,
    const std::vector<float>& sin_table,
    float base = moe_ROPE_THETA
) {
    // BUGFIX 224: Validate dim > 0 to prevent division by zero in frequency calculation ★★
    // Problem: dim=0 causes division by zero in freq = 1.0f / std::pow(base, (float)i / (float)dim)
    // Solution: Return early if dim == 0 or data == nullptr
    // Impact: Prevents crash in RoPE application with invalid config
    if (!data || dim == 0) {
        return;
    }
    if (position == 0) {
        return;
    }
    const uint32_t half_dim = dim / 2u;
    const size_t row_offset = (size_t)position * (size_t)half_dim;
    const size_t table_len = half_dim ? cos_table.size() / half_dim : 0;
    if ((size_t)position < table_len &&
        cos_table.size() >= row_offset + half_dim &&
        sin_table.size() >= row_offset + half_dim) {
        const float* cos_row = cos_table.data() + row_offset;
        const float* sin_row = sin_table.data() + row_offset;
        for (uint32_t j = 0; j < half_dim; ++j) {
            const uint32_t i = j * 2u;
            const float cos_t = cos_row[j];
            const float sin_t = sin_row[j];
            const float x0 = data[i];
            const float x1 = data[i + 1u];
            data[i] = x0 * cos_t - x1 * sin_t;
            data[i + 1u] = x0 * sin_t + x1 * cos_t;
        }
        return;
    }
    const float* cached_cos = nullptr;
    const float* cached_sin = nullptr;
    if (moe_rope_runtime_row(dim, position, base, &cached_cos, &cached_sin)) {
        for (uint32_t j = 0; j < half_dim; ++j) {
            const uint32_t i = j * 2u;
            const float cos_t = cached_cos[j];
            const float sin_t = cached_sin[j];
            const float x0 = data[i];
            const float x1 = data[i + 1u];
            data[i] = x0 * cos_t - x1 * sin_t;
            data[i + 1u] = x0 * sin_t + x1 * cos_t;
        }
        return;
    }
    for (uint32_t i = 0; i + 1u < dim; i += 2u) {
        const float freq = 1.0f / std::pow(base, (float)i / (float)dim);
        const float theta = (float)position * freq;
        const float cos_t = std::cos(theta);
        const float sin_t = std::sin(theta);
        const float x0 = data[i];
        const float x1 = data[i + 1];
        data[i]     = x0 * cos_t - x1 * sin_t;
        data[i + 1] = x0 * sin_t + x1 * cos_t;
    }
}

static void moe_rope_rotate_half_inplace(
    float* data,
    uint32_t dim,
    uint32_t position,
    const std::vector<float>& cos_table,
    const std::vector<float>& sin_table,
    float base = moe_ROPE_THETA
) {
    if (!data || dim < 2) {
        return;
    }
    if (position == 0) {
        return;
    }
    const uint32_t half_dim = dim / 2u;
    const size_t row_offset = (size_t)position * (size_t)half_dim;
    const size_t table_len = half_dim ? cos_table.size() / half_dim : 0;
    if ((size_t)position < table_len &&
        cos_table.size() >= row_offset + half_dim &&
        sin_table.size() >= row_offset + half_dim) {
        const float* cos_row = cos_table.data() + row_offset;
        const float* sin_row = sin_table.data() + row_offset;
        for (uint32_t j = 0; j < half_dim; ++j) {
            const float cos_t = cos_row[j];
            const float sin_t = sin_row[j];
            const float x0 = data[j];
            const float x1 = data[j + half_dim];
            data[j] = x0 * cos_t - x1 * sin_t;
            data[j + half_dim] = x1 * cos_t + x0 * sin_t;
        }
        return;
    }
    const float* cached_cos = nullptr;
    const float* cached_sin = nullptr;
    if (moe_rope_runtime_row(dim, position, base, &cached_cos, &cached_sin)) {
        for (uint32_t j = 0; j < half_dim; ++j) {
            const float cos_t = cached_cos[j];
            const float sin_t = cached_sin[j];
            const float x0 = data[j];
            const float x1 = data[j + half_dim];
            data[j] = x0 * cos_t - x1 * sin_t;
            data[j + half_dim] = x1 * cos_t + x0 * sin_t;
        }
        return;
    }
    for (uint32_t j = 0; j < half_dim; ++j) {
        const uint32_t i = j * 2u;
        const float freq = 1.0f / std::pow(base, (float)i / (float)dim);
        const float theta = (float)position * freq;
        const float cos_t = std::cos(theta);
        const float sin_t = std::sin(theta);
        const float x0 = data[j];
        const float x1 = data[j + half_dim];
        data[j] = x0 * cos_t - x1 * sin_t;
        data[j + half_dim] = x1 * cos_t + x0 * sin_t;
    }
}

static void moe_standard_rope_apply_inplace_with_layout(
    int rotate_half,
    float* data,
    uint32_t dim,
    uint32_t position,
    const std::vector<float>& cos_table,
    const std::vector<float>& sin_table,
    float base
) {
    if (rotate_half) {
        moe_rope_rotate_half_inplace(data, dim, position, cos_table, sin_table, base);
    } else {
        moe_rope_inplace(data, dim, position, cos_table, sin_table, base);
    }
}

static int moe_rope_bool_metadata(
    const moe_pc_engine* engine,
    const char* const* keys,
    size_t key_count,
    int* out_value
) {
    if (!engine || !out_value || engine->offload_gguf_metadata_json.empty()) {
        return 0;
    }
    for (size_t i = 0; i < key_count; ++i) {
        int value = 0;
        if (keys[i] && moe_json_get_bool_local(engine->offload_gguf_metadata_json, keys[i], &value)) {
            *out_value = value ? 1 : 0;
            return 1;
        }
    }
    return 0;
}

static int moe_rope_string_metadata_mentions(
    const moe_pc_engine* engine,
    const char* const* keys,
    size_t key_count,
    const char* const* needles,
    size_t needle_count
) {
    if (!engine || engine->offload_gguf_metadata_json.empty()) {
        return 0;
    }
    for (size_t i = 0; i < key_count; ++i) {
        std::string value;
        if (!keys[i] || !moe_json_get_string_local(engine->offload_gguf_metadata_json, keys[i], &value)) {
            continue;
        }
        value = moe_lower_ascii_copy(value);
        for (size_t n = 0; n < needle_count; ++n) {
            if (needles[n] && value.find(needles[n]) != std::string::npos) {
                return 1;
            }
        }
    }
    return 0;
}

static int moe_standard_rope_uses_rotate_half(
    const moe_pc_engine* engine,
    const moe_attn_config_t& cfg
) {
    static const char* const interleaved_bool_keys[] = {
        "rope_interleaved",
        "rotary_interleaved",
        "rotary_emb_interleaved",
        "interleaved_rope",
        "use_interleaved_rope"
    };
    int interleaved = 0;
    if (moe_rope_bool_metadata(
            engine, interleaved_bool_keys,
            sizeof(interleaved_bool_keys) / sizeof(interleaved_bool_keys[0]),
            &interleaved)) {
        return interleaved ? 0 : 1;
    }

    static const char* const layout_keys[] = {
        "rope_layout",
        "rotary_layout",
        "rope_type",
        "rotary_type",
        "position_embedding_type"
    };
    static const char* const rotate_half_names[] = {
        "rotate_half",
        "half_split",
        "split_half"
    };
    static const char* const adjacent_names[] = {
        "interleaved",
        "adjacent",
        "pairwise",
        "pair"
    };
    if (moe_rope_string_metadata_mentions(
            engine, layout_keys, sizeof(layout_keys) / sizeof(layout_keys[0]),
            adjacent_names, sizeof(adjacent_names) / sizeof(adjacent_names[0])) ||
        moe_engine_graph_ir_mentions(engine, "rope_interleaved") ||
        moe_engine_graph_ir_mentions(engine, "rotary_interleaved") ||
        moe_engine_graph_ir_mentions(engine, "adjacent_rope") ||
        moe_engine_graph_ir_mentions(engine, "pairwise_rope")) {
        return 0;
    }
    if (moe_rope_string_metadata_mentions(
            engine, layout_keys, sizeof(layout_keys) / sizeof(layout_keys[0]),
            rotate_half_names, sizeof(rotate_half_names) / sizeof(rotate_half_names[0])) ||
        moe_engine_graph_ir_mentions(engine, "rotate_half") ||
        moe_engine_graph_ir_mentions(engine, "half_split_rope") ||
        moe_engine_graph_ir_mentions(engine, "split_half_rope")) {
        return 1;
    }

    if (engine &&
        !engine->offload_gguf_metadata_json.empty() &&
        (moe_engine_metadata_mentions(engine, "partial_rotary_factor") ||
         moe_engine_metadata_mentions(engine, "rope_scaling"))) {
        return 1;
    }

    return cfg.type != moe_ATTN_TYPE_MLA ? 1 : 0;
}

static void moe_standard_rope_apply_inplace(
    const moe_pc_engine* engine,
    const moe_attn_config_t& cfg,
    float* data,
    uint32_t dim,
    uint32_t position,
    const std::vector<float>& cos_table,
    const std::vector<float>& sin_table,
    float base
) {
    moe_standard_rope_apply_inplace_with_layout(
        moe_standard_rope_uses_rotate_half(engine, cfg),
        data,
        dim,
        position,
        cos_table,
        sin_table,
        base);
}
