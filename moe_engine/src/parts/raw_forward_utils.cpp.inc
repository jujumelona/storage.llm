// Raw forward - Utility functions

static float moe_raw_bf16_to_f32(uint16_t v) {
    const uint32_t bits = (uint32_t)v << 16u;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

static uint32_t moe_raw_scalar_size(uint8_t encoding) {
    return encoding == moe_RAW_SCALAR_F32 ? 4u : 2u;
}

static float moe_raw_scalar_to_f32(const uint8_t* ptr, uint8_t encoding) {
    if (!ptr) {
        return 0.0f;
    }
    if (encoding == moe_RAW_SCALAR_F32) {
        float v = 0.0f;
        std::memcpy(&v, ptr, sizeof(v));
        return v;
    }
    uint16_t v = 0;
    std::memcpy(&v, ptr, sizeof(v));
    return encoding == moe_RAW_SCALAR_F16 ? moe_raw_fp16_to_f32(v) : moe_raw_bf16_to_f32(v);
}

static const moe_storage_raw_span_spec_t* find_raw_role_span(int32_t layer, uint32_t role) {
    const uint32_t begin = moe_storage_raw_span_find_first(layer);
    const uint32_t end = moe_storage_raw_span_find_end(layer);
    for (uint32_t i = begin; i < end; ++i) {
        const moe_storage_raw_span_spec_t* s = moe_storage_raw_span_at(i);
        if (s && (s->role & role) != 0 && s->bytes > 0) {
            return s;
        }
    }
    return nullptr;
}

static mmap_context_t* map_raw_span_part(moe_pc_engine* engine, const moe_storage_raw_span_spec_t* span) {
    if (!engine || !span || engine->model_root.empty()) {
        return nullptr;
    }
    const moe_storage_part_spec_t* part = moe_storage_part_by_id(span->part);
    return part && part->path ? get_or_map_shard_partial_by_id(
        engine,
        span->part,
        resolve_model_file_path(engine->model_root, part->path, ""),
        span->offset,
        span->bytes
    ) : nullptr;
}

static int raw_span_bounds_ok(const moe_storage_raw_span_spec_t* span, const mmap_context_t* mapped) {
    return span && mapped && mmap_get_ptr(mapped, 0) &&
        span->bytes <= (uint64_t)mmap_get_size(mapped);
}

static const uint8_t* raw_span_ptr(
    const moe_storage_raw_span_spec_t* span,
    const mmap_context_t* mapped,
    uint64_t rel_offset,
    uint64_t bytes
) {
    if (!raw_span_bounds_ok(span, mapped)) {
        return nullptr;
    }
    const uint64_t mapped_size = (uint64_t)mmap_get_size(mapped);
    if (rel_offset > mapped_size || bytes > mapped_size - rel_offset || rel_offset > (uint64_t)SIZE_MAX) {
        return nullptr;
    }
    return static_cast<const uint8_t*>(mmap_get_ptr(mapped, (size_t)rel_offset));
}

static const moe_storage_raw_span_spec_t* find_raw_role_span_bytes(
    int32_t layer,
    uint32_t role,
    uint64_t bytes,
    uint32_t occurrence
) {
    const uint32_t begin = moe_storage_raw_span_find_first(layer);
    const uint32_t end = moe_storage_raw_span_find_end(layer);
    uint32_t seen = 0;
    for (uint32_t i = begin; i < end; ++i) {
        const moe_storage_raw_span_spec_t* s = moe_storage_raw_span_at(i);
        if (s && (s->role & role) != 0 && s->bytes == bytes) {
            if (seen == occurrence) {
                return s;
            }
            ++seen;
        }
    }
    return nullptr;
}

static float moe_dot_bf16_row(const uint8_t* row, const float* hidden, uint32_t hidden_count) {
    uint32_t i = 0;
    float acc = 0.0f;
#if defined(__AVX2__)
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    for (; i + 15 < hidden_count; i += 16) {
        __m128i h0 = _mm_loadu_si128((const __m128i*)(row + (uint64_t)i * 2ull));
        __m128i h1 = _mm_loadu_si128((const __m128i*)(row + (uint64_t)(i + 8) * 2ull));
        __m256i exp0 = _mm256_cvtepu16_epi32(h0);
        __m256i exp1 = _mm256_cvtepu16_epi32(h1);
        __m256 f0 = _mm256_castsi256_ps(_mm256_slli_epi32(exp0, 16));
        __m256 f1 = _mm256_castsi256_ps(_mm256_slli_epi32(exp1, 16));
        __m256 x0 = _mm256_loadu_ps(hidden + i);
        __m256 x1 = _mm256_loadu_ps(hidden + i + 8);
        sum0 = moe_madd_ps(f0, x0, sum0);
        sum1 = moe_madd_ps(f1, x1, sum1);
    }
    sum0 = _mm256_add_ps(sum0, sum1);
    float buf[8];
    _mm256_storeu_ps(buf, sum0);
    acc += buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];
#endif
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (; i + 3u < hidden_count; i += 4u) {
        uint16_t v0, v1, v2, v3;
        std::memcpy(&v0, row + (uint64_t)(i + 0u) * 2ull, sizeof(v0));
        std::memcpy(&v1, row + (uint64_t)(i + 1u) * 2ull, sizeof(v1));
        std::memcpy(&v2, row + (uint64_t)(i + 2u) * 2ull, sizeof(v2));
        std::memcpy(&v3, row + (uint64_t)(i + 3u) * 2ull, sizeof(v3));
        a0 += moe_raw_bf16_to_f32(v0) * hidden[i + 0u];
        a1 += moe_raw_bf16_to_f32(v1) * hidden[i + 1u];
        a2 += moe_raw_bf16_to_f32(v2) * hidden[i + 2u];
        a3 += moe_raw_bf16_to_f32(v3) * hidden[i + 3u];
    }
    acc += (a0 + a1) + (a2 + a3);
    for (; i < hidden_count; ++i) {
        uint16_t v;
        std::memcpy(&v, row + (uint64_t)i * 2ull, sizeof(v));
        acc += moe_raw_bf16_to_f32(v) * hidden[i];
    }
    return acc;
}

static float moe_dot_raw_row(
    const uint8_t* row,
    uint8_t encoding,
    const float* hidden,
    uint32_t hidden_count
) {
    if (encoding == moe_RAW_SCALAR_BF16) {
        return moe_dot_bf16_row(row, hidden, hidden_count);
    }
    const uint32_t elem = moe_raw_scalar_size(encoding);
    float acc = 0.0f;
    for (uint32_t i = 0; i < hidden_count; ++i) {
        acc += moe_raw_scalar_to_f32(row + (uint64_t)i * elem, encoding) * hidden[i];
    }
    return acc;
}

static int moe_engine_has_common_tensor_suffix(
    const moe_pc_engine* engine,
    int32_t layer,
    const char* suffix
) {
    if (!engine || !suffix || layer < 0) {
        return 0;
    }
    const std::string layer_prefix = "blk." + std::to_string(layer) + ".";
    const std::string suffix_lower = moe_lower_ascii_copy(std::string(suffix));
    for (const auto& it : engine->gguf_common_tensor_slots) {
        const std::string& key = it.first;
        if (key.size() >= layer_prefix.size() + suffix_lower.size() &&
            key.compare(0, layer_prefix.size(), layer_prefix) == 0 &&
            key.compare(key.size() - suffix_lower.size(), suffix_lower.size(), suffix_lower) == 0) {
            return 1;
        }
    }
    return 0;
}

static int moe_engine_graph_ir_mentions(const moe_pc_engine* engine, const char* needle) {
    if (!engine || !needle || !needle[0] || engine->offload_graph_ir_json.empty()) {
        return 0;
    }
    const std::string haystack = moe_lower_ascii_copy(engine->offload_graph_ir_json);
    const std::string target = moe_lower_ascii_copy(std::string(needle));
    return haystack.find(target) != std::string::npos ? 1 : 0;
}

static int moe_engine_metadata_mentions(const moe_pc_engine* engine, const char* needle) {
    if (!engine || !needle || !needle[0] || engine->offload_gguf_metadata_json.empty()) {
        return 0;
    }
    const std::string haystack = moe_lower_ascii_copy(engine->offload_gguf_metadata_json);
    const std::string target = moe_lower_ascii_copy(std::string(needle));
    return haystack.find(target) != std::string::npos ? 1 : 0;
}

static int moe_engine_prefers_gelu_activation(const moe_pc_engine* engine) {
    if (!engine) {
        return 0;
    }
    if (moe_engine_graph_ir_mentions(engine, "gelu") ||
        moe_engine_metadata_mentions(engine, "gelu")) {
        return 1;
    }
    return 0;
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

static uint64_t moe_raw_tensor_row_bytes(
    const moe_raw_tensor_cache_entry& raw,
    uint32_t cols
) {
    if (raw.weight_row_bytes) {
        return raw.weight_row_bytes;
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_Q8_0) {
        return ((uint64_t)cols + 31ull) / 32ull * 34ull;
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_Q5_1) {
        return ((uint64_t)cols + 31ull) / 32ull * 24ull;
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_MXFP4) {
        return ((uint64_t)cols + 31ull) / 32ull * 17ull;
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
    if (!raw.base || !hidden || row >= raw.rows || row_bytes == 0 ||
        (uint64_t)row * row_bytes > raw.bytes ||
        row_bytes > raw.bytes - (uint64_t)row * row_bytes) {
        return 0.0f;
    }
    const uint8_t* row_ptr = raw.base + (uint64_t)row * row_bytes;
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_Q8_0) {
        return moe_dot_gguf_q8_0_row(row_ptr, hidden, hidden_count);
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_Q5_1) {
        return moe_dot_gguf_q5_1_row(row_ptr, hidden, hidden_count);
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_MXFP4) {
        return moe_dot_gguf_mxfp4_row(row_ptr, hidden, hidden_count);
    }
    return moe_dot_raw_row(row_ptr, raw.scalar_encoding, hidden, hidden_count);
}

static int moe_raw_tensor_copy_row_f32(
    const moe_raw_tensor_cache_entry& raw,
    uint32_t row,
    float* out,
    uint32_t out_count
) {
    const uint32_t stride_cols = raw.cols ? raw.cols : out_count;
    const uint64_t row_bytes = moe_raw_tensor_row_bytes(raw, stride_cols);
    if (!raw.base || !out || row >= raw.rows || row_bytes == 0 ||
        (uint64_t)row * row_bytes > raw.bytes ||
        row_bytes > raw.bytes - (uint64_t)row * row_bytes) {
        return 0;
    }
    const uint8_t* row_ptr = raw.base + (uint64_t)row * row_bytes;
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_Q8_0) {
        const uint32_t block_cols = 32u;
        const uint32_t block_bytes = 34u;
        uint32_t dst = 0;
        const uint8_t* block = row_ptr;
        while (dst < out_count) {
            uint16_t d16 = 0;
            std::memcpy(&d16, block, sizeof(d16));
            const float d = moe_raw_fp16_to_f32(d16);
            const int8_t* q = reinterpret_cast<const int8_t*>(block + 2u);
            const uint32_t limit = std::min<uint32_t>(block_cols, out_count - dst);
            for (uint32_t i = 0; i < limit; ++i) {
                out[dst + i] = d * (float)q[i];
            }
            dst += limit;
            block += block_bytes;
        }
        return 1;
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_Q5_1) {
        const uint32_t block_cols = 32u;
        const uint32_t block_bytes = 24u;
        uint32_t dst = 0;
        const uint8_t* block = row_ptr;
        while (dst < out_count) {
            uint16_t d16 = 0;
            uint16_t m16 = 0;
            uint32_t qh = 0;
            std::memcpy(&d16, block, sizeof(d16));
            std::memcpy(&m16, block + 2u, sizeof(m16));
            std::memcpy(&qh, block + 4u, sizeof(qh));
            const float d = moe_raw_fp16_to_f32(d16);
            const float m = moe_raw_fp16_to_f32(m16);
            const uint8_t* qs = block + 8u;
            const uint32_t limit = std::min<uint32_t>(block_cols, out_count - dst);
            for (uint32_t i = 0; i < limit; ++i) {
                const uint8_t byte = qs[i & 15u];
                const uint32_t low = i < 16u ? (byte & 0x0fu) : (byte >> 4);
                const uint32_t high = (qh >> i) & 1u;
                out[dst + i] = d * (float)(low | (high << 4u)) + m;
            }
            dst += limit;
            block += block_bytes;
        }
        return 1;
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_MXFP4) {
        const uint32_t block_cols = 32u;
        const uint32_t block_bytes = 17u;
        uint32_t dst = 0;
        const uint8_t* block = row_ptr;
        while (dst < out_count) {
            const float scale = moe_e8m0_to_f32_half(block[0]);
            const uint8_t* qs = block + 1u;
            const uint32_t limit = std::min<uint32_t>(block_cols, out_count - dst);
            for (uint32_t i = 0; i < limit; ++i) {
                out[dst + i] = moe_mxfp4_value_from_code(moe_mxfp4_code_at(qs, i)) * scale;
            }
            dst += limit;
            block += block_bytes;
        }
        return 1;
    }
    const uint32_t elem = moe_raw_scalar_size(raw.scalar_encoding);
    for (uint32_t i = 0; i < out_count; ++i) {
        out[i] = moe_raw_scalar_to_f32(row_ptr + (uint64_t)i * (uint64_t)elem, raw.scalar_encoding);
    }
    return 1;
}
