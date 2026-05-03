// Bug #1 Fix: Full MLA attention with Q projection, RoPE, scaled dot-product.
// GLM4/Moe Multi-head Latent Attention (MLA):
//   KV path: hidden -> kv_a_proj(576) -> [kv_a_norm(512), mqa_rope(64)]
//            kv_a_norm -> kv_b_proj(28672) -> [K_nope(12288), V(16384)]
//   Q path:  hidden -> q_a_proj(2048) -> q_a_norm(2048) -> q_b_proj(12288+4096=16384)
//            -> [Q_nope(12288), Q_rope(4096)] split by heads
//   Score = softmax(Q_nope*K_nope + Q_rope*K_rope) * V -> o_proj -> attn output

// Keep this tied to the converted model metadata; tensor shapes alone do not determine theta.
// Fix 31+32: Flat contiguous RoPE cache with position-keyed LRU
// - Eliminated per-slot heap vectors (16 mallocs -> 0)
// - Expanded from 8 round-robin to 32 LRU slots for better long-context hit rate
// - cos/sin data is contiguous for SIMD-friendly access
static constexpr uint32_t MOE_ROPE_CACHE_SLOTS = 32u;
static constexpr uint32_t MOE_ROPE_MAX_HALF_DIM = 4096u;

struct moe_rope_flat_cache {
    uint32_t dim[MOE_ROPE_CACHE_SLOTS] = {};
    uint32_t position[MOE_ROPE_CACHE_SLOTS] = {};
    uint32_t base_bits[MOE_ROPE_CACHE_SLOTS] = {};
    uint32_t lru_tick[MOE_ROPE_CACHE_SLOTS] = {};
    alignas(64) float cos_data[MOE_ROPE_CACHE_SLOTS][MOE_ROPE_MAX_HALF_DIM];
    alignas(64) float sin_data[MOE_ROPE_CACHE_SLOTS][MOE_ROPE_MAX_HALF_DIM];
    uint32_t used = 0;
    uint32_t tick = 0;
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
    if (half_dim > MOE_ROPE_MAX_HALF_DIM) {
        return 0;
    }
    const uint32_t bbits = moe_rope_float_bits(base);
    static thread_local moe_rope_flat_cache cache;
    const uint32_t current_tick = ++cache.tick;

    // Lookup in flat cache
    for (uint32_t s = 0; s < cache.used; ++s) {
        if (cache.dim[s] == dim &&
            cache.position[s] == position &&
            cache.base_bits[s] == bbits) {
            cache.lru_tick[s] = current_tick;
            *out_cos = cache.cos_data[s];
            *out_sin = cache.sin_data[s];
            return 1;
        }
    }

    // Miss: find or evict a slot
    uint32_t slot;
    if (cache.used < MOE_ROPE_CACHE_SLOTS) {
        slot = cache.used++;
    } else {
        // Evict LRU
        slot = 0;
        uint32_t oldest = cache.lru_tick[0];
        for (uint32_t s = 1; s < MOE_ROPE_CACHE_SLOTS; ++s) {
            if (cache.lru_tick[s] < oldest) {
                oldest = cache.lru_tick[s];
                slot = s;
            }
        }
    }

    cache.dim[slot] = dim;
    cache.position[slot] = position;
    cache.base_bits[slot] = bbits;
    cache.lru_tick[slot] = current_tick;

    // Fix 31: Use exp instead of pow for freq computation
    // log(base) computed once, then exp(-(j*2/dim)*log_base) = base^(-j*2/dim)
    const float multiplier = std::pow(base, -2.0f / (float)dim);
    float freq = 1.0f;
    for (uint32_t j = 0; j < half_dim; ++j) {
        const float theta = (float)position * freq;
        cache.cos_data[slot][j] = std::cos(theta);
        cache.sin_data[slot][j] = std::sin(theta);
        freq *= multiplier;
    }
    *out_cos = cache.cos_data[slot];
    *out_sin = cache.sin_data[slot];
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
    // Fix 14: Use geometric multiplier
    const float multiplier = std::pow(base, -2.0f / (float)dim);
    float freq = 1.0f;
    for (uint32_t i = 0; i + 1u < dim; i += 2u) {
        const float theta = (float)position * freq;
        const float cos_t = std::cos(theta);
        const float sin_t = std::sin(theta);
        const float x0 = data[i];
        const float x1 = data[i + 1];
        data[i]     = x0 * cos_t - x1 * sin_t;
        data[i + 1] = x0 * sin_t + x1 * cos_t;
        freq *= multiplier;
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
    // Fix 14: Use geometric multiplier
    const float multiplier = std::pow(base, -2.0f / (float)dim);
    float freq = 1.0f;
    for (uint32_t j = 0; j < half_dim; ++j) {
        const float theta = (float)position * freq;
        const float cos_t = std::cos(theta);
        const float sin_t = std::sin(theta);
        const float x0 = data[j];
        const float x1 = data[j + half_dim];
        data[j] = x0 * cos_t - x1 * sin_t;
        data[j + half_dim] = x1 * cos_t + x0 * sin_t;
        freq *= multiplier;
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

struct moe_rope_layout_cache_entry {
    const moe_pc_engine* engine = nullptr;
    int type = 0;
    int rotate_half = 0;
};

static int moe_standard_rope_uses_rotate_half_cached(
    const moe_pc_engine* engine,
    const moe_attn_config_t& cfg
) {
    static thread_local std::vector<moe_rope_layout_cache_entry> cache;
    for (const moe_rope_layout_cache_entry& entry : cache) {
        if (entry.engine == engine && entry.type == (int)cfg.type) {
            return entry.rotate_half;
        }
    }
    const int rotate_half = moe_standard_rope_uses_rotate_half(engine, cfg);
    if (cache.size() < 16u) {
        try {
            cache.push_back(moe_rope_layout_cache_entry{engine, (int)cfg.type, rotate_half});
        } catch (...) {
        }
    }
    return rotate_half;
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
        moe_standard_rope_uses_rotate_half_cached(engine, cfg),
        data,
        dim,
        position,
        cos_table,
        sin_table,
        base);
}
