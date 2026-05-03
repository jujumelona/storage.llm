// Raw forward - Utility functions

static float moe_raw_bf16_to_f32(uint16_t v) {
    const uint32_t bits = (uint32_t)v << 16u;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

static uint32_t moe_f32_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float moe_f32_from_bits(uint32_t bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

static size_t moe_ascii_find_ci(const std::string& haystack, const char* needle, size_t begin = 0) {
    if (!needle || !needle[0]) {
        return std::string::npos;
    }
    const size_t n = std::strlen(needle);
    if (n == 0 || begin >= haystack.size() || haystack.size() < n) {
        return std::string::npos;
    }
    for (size_t i = begin; i + n <= haystack.size(); ++i) {
        size_t j = 0;
        for (; j < n; ++j) {
            const unsigned char a = static_cast<unsigned char>(haystack[i + j]);
            const unsigned char b = static_cast<unsigned char>(needle[j]);
            if (std::tolower(a) != std::tolower(b)) {
                break;
            }
        }
        if (j == n) {
            return i;
        }
    }
    return std::string::npos;
}

static int moe_ascii_contains_ci(const std::string& haystack, const char* needle) {
    if (!needle || !needle[0]) {
        return 0;
    }
    if (moe_ascii_find_ci(haystack, needle) != std::string::npos) {
        return 1;
    }
    return 0;
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

static void moe_trace_raw_dot_unsupported(const moe_raw_tensor_cache_entry& raw) {
    static std::atomic<uint32_t> logged{0};
    const uint32_t n = logged.fetch_add(1, std::memory_order_relaxed);
    if (n >= 32u) {
        return;
    }
    moe_profile_log(
        "raw_dot_unsupported",
        "format=%u scalar=%u rows=%u cols=%u bytes=%llu row_bytes=%llu",
        (uint32_t)raw.weight_format,
        (uint32_t)raw.scalar_encoding,
        raw.rows,
        raw.cols,
        (unsigned long long)raw.bytes,
        (unsigned long long)raw.weight_row_bytes);
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
    // BUGFIX 495: rel_offset + bytes overflow 체크
    if (rel_offset > mapped_size || bytes > mapped_size - rel_offset ||
        rel_offset > (uint64_t)SIZE_MAX ||
        (bytes > 0 && rel_offset > UINT64_MAX - bytes)) {
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
    // BUGFIX 529: Check hidden_count upper bound to prevent excessive iteration ★
    // Problem: hidden_count can be arbitrarily large from config
    // Loop can run billions of iterations
    // Solution: Add reasonable upper bound check (e.g., 100000)
    if (!row || !hidden || hidden_count > 100000) {
        return 0.0f;
    }
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
    std::string cache_key = std::to_string(layer);
    cache_key.push_back('\n');
    cache_key.append(suffix_lower);
    {
        std::shared_lock<std::shared_mutex> lock(engine->common_tensor_suffix_cache_mutex);
        auto found = engine->common_tensor_suffix_cache.find(cache_key);
        if (found != engine->common_tensor_suffix_cache.end()) {
            return found->second;
        }
    }
    int result = 0;
    for (const auto& it : engine->gguf_common_tensor_slots) {
        const std::string& key = it.first;
        if (key.size() >= layer_prefix.size() + suffix_lower.size() &&
            key.compare(0, layer_prefix.size(), layer_prefix) == 0 &&
            key.compare(key.size() - suffix_lower.size(), suffix_lower.size(), suffix_lower) == 0) {
            result = 1;
            break;
        }
    }
    {
        std::unique_lock<std::shared_mutex> lock(engine->common_tensor_suffix_cache_mutex);
        engine->common_tensor_suffix_cache.emplace(std::move(cache_key), result);
    }
    return result;
}

static int moe_engine_graph_ir_mentions(const moe_pc_engine* engine, const char* needle) {
    if (!engine || !needle || !needle[0] || engine->offload_graph_ir_json.empty()) {
        return 0;
    }
    const std::string key = moe_lower_ascii_copy(std::string(needle));
    {
        std::shared_lock<std::shared_mutex> lock(engine->text_mention_cache_mutex);
        auto found = engine->graph_ir_mention_cache.find(key);
        if (found != engine->graph_ir_mention_cache.end()) {
            return found->second;
        }
    }
    const int result = moe_ascii_contains_ci(engine->offload_graph_ir_json, needle);
    {
        std::unique_lock<std::shared_mutex> lock(engine->text_mention_cache_mutex);
        engine->graph_ir_mention_cache.emplace(key, result);
    }
    return result;
}

static int moe_engine_metadata_mentions(const moe_pc_engine* engine, const char* needle) {
    if (!engine || !needle || !needle[0] || engine->offload_gguf_metadata_json.empty()) {
        return 0;
    }
    const std::string key = moe_lower_ascii_copy(std::string(needle));
    {
        std::shared_lock<std::shared_mutex> lock(engine->text_mention_cache_mutex);
        auto found = engine->metadata_mention_cache.find(key);
        if (found != engine->metadata_mention_cache.end()) {
            return found->second;
        }
    }
    const int result = moe_ascii_contains_ci(engine->offload_gguf_metadata_json, needle);
    {
        std::unique_lock<std::shared_mutex> lock(engine->text_mention_cache_mutex);
        engine->metadata_mention_cache.emplace(key, result);
    }
    return result;
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
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_Q8_1) {
        return ((uint64_t)cols + 31ull) / 32ull * 36ull;
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_Q8_K) {
        return ((uint64_t)cols + 255ull) / 256ull * 292ull;
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_Q4_0) {
        return ((uint64_t)cols + 31ull) / 32ull * 18ull;
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_Q4_1) {
        return ((uint64_t)cols + 31ull) / 32ull * 20ull;
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_Q5_0) {
        return ((uint64_t)cols + 31ull) / 32ull * 22ull;
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_Q5_1) {
        return ((uint64_t)cols + 31ull) / 32ull * 24ull;
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_MXFP4) {
        return ((uint64_t)cols + 31ull) / 32ull * 17ull;
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_Q2_K) {
        return ((uint64_t)cols + 255ull) / 256ull * 84ull;
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_Q3_K) {
        return ((uint64_t)cols + 255ull) / 256ull * 110ull;
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_Q4_K) {
        return ((uint64_t)cols + 255ull) / 256ull * 144ull;
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_Q5_K) {
        return ((uint64_t)cols + 255ull) / 256ull * 176ull;
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_Q6_K) {
        return ((uint64_t)cols + 255ull) / 256ull * 210ull;
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_IQ2_XXS) {
        return ((uint64_t)cols + 255ull) / 256ull * 66ull;
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_IQ2_XS) {
        return ((uint64_t)cols + 255ull) / 256ull * 74ull;
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_IQ2_S) {
        return ((uint64_t)cols + 255ull) / 256ull * 82ull;
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_IQ3_XXS) {
        return ((uint64_t)cols + 255ull) / 256ull * 98ull;
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_IQ3_S) {
        return ((uint64_t)cols + 255ull) / 256ull * 110ull;
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_IQ4_NL) {
        return ((uint64_t)cols + 31ull) / 32ull * 18ull;
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_IQ4_XS) {
        return ((uint64_t)cols + 255ull) / 256ull * 136ull;
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_IQ1_S) {
        return ((uint64_t)cols + 255ull) / 256ull * 50ull;
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_IQ1_M) {
        return ((uint64_t)cols + 255ull) / 256ull * 56ull;
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
    // BUGFIX 494: row * row_bytes overflow 체크
    if (!raw.base || !hidden || row >= raw.rows || row_bytes == 0 ||
        row > UINT64_MAX / row_bytes ||
        (uint64_t)row * row_bytes > raw.bytes ||
        row_bytes > raw.bytes - (uint64_t)row * row_bytes) {
        return 0.0f;
    }
    const uint8_t* row_ptr = raw.base + (uint64_t)row * row_bytes;
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_Q8_0) {
        return moe_dot_gguf_q8_0_row(row_ptr, hidden, hidden_count);
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_Q8_1) {
        return moe_dot_gguf_q8_1_row(row_ptr, hidden, hidden_count);
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_Q8_K) {
        return moe_dot_gguf_q8_k_row(row_ptr, hidden, hidden_count);
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_Q4_0) {
        return moe_dot_gguf_q4_0_row(row_ptr, hidden, hidden_count);
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_Q4_1) {
        return moe_dot_gguf_q4_1_row(row_ptr, hidden, hidden_count);
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_Q5_0) {
        return moe_dot_gguf_q5_0_row(row_ptr, hidden, hidden_count);
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_Q5_1) {
        return moe_dot_gguf_q5_1_row(row_ptr, hidden, hidden_count);
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_MXFP4) {
        return moe_dot_gguf_mxfp4_row(row_ptr, hidden, hidden_count);
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_Q2_K) {
        return moe_dot_gguf_q2_k_row(row_ptr, hidden, hidden_count);
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_Q3_K) {
        return moe_dot_gguf_q3_k_row(row_ptr, hidden, hidden_count);
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_Q4_K) {
        return moe_dot_gguf_q4_k_row(row_ptr, hidden, hidden_count);
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_Q5_K) {
        return moe_dot_gguf_q5_k_row(row_ptr, hidden, hidden_count);
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_Q6_K) {
        return moe_dot_gguf_q6_k_row(row_ptr, hidden, hidden_count);
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_IQ2_XXS) {
        return moe_dot_gguf_iq2_xxs_row(row_ptr, hidden, hidden_count);
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_IQ2_XS) {
        return moe_dot_gguf_iq2_xs_row(row_ptr, hidden, hidden_count);
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_IQ2_S) {
        return moe_dot_gguf_iq2_s_row(row_ptr, hidden, hidden_count);
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_IQ3_XXS) {
        return moe_dot_gguf_iq3_xxs_row(row_ptr, hidden, hidden_count);
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_IQ3_S) {
        return moe_dot_gguf_iq3_s_row(row_ptr, hidden, hidden_count);
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_IQ4_NL) {
        return moe_dot_gguf_iq4_nl_row(row_ptr, hidden, hidden_count);
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_IQ4_XS) {
        return moe_dot_gguf_iq4_xs_row(row_ptr, hidden, hidden_count);
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_IQ1_S) {
        return moe_dot_gguf_iq1_s_row(row_ptr, hidden, hidden_count);
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_IQ1_M) {
        return moe_dot_gguf_iq1_m_row(row_ptr, hidden, hidden_count);
    }
    if (raw.weight_format != moe_WEIGHT_ENCODING_UNKNOWN &&
        raw.weight_format != moe_WEIGHT_ENCODING_RAW_FP16 &&
        raw.weight_format != moe_WEIGHT_ENCODING_RAW_BF16 &&
        raw.weight_format != moe_WEIGHT_ENCODING_RAW_FP32) {
        moe_trace_raw_dot_unsupported(raw);
        return 0.0f;
    }
    return moe_dot_raw_row(row_ptr, raw.scalar_encoding, hidden, hidden_count);
}

static int moe_raw_tensor_copy_row_f32(
    const moe_raw_tensor_cache_entry& raw,
    uint32_t row,
    float* out,
    uint32_t out_count
) {
    // BUGFIX 530: Check out_count upper bound to prevent excessive iteration ★
    // Problem: out_count can be arbitrarily large from config
    // Loop can run billions of iterations
    // Solution: Add reasonable upper bound check (e.g., 100000)
    if (!out || out_count > 100000) {
        return 0;
    }
    const uint32_t stride_cols = raw.cols ? raw.cols : out_count;
    const uint64_t row_bytes = moe_raw_tensor_row_bytes(raw, stride_cols);
    // BUGFIX 493: row * row_bytes overflow 체크
    if (!raw.base || !out || row >= raw.rows || row_bytes == 0 ||
        row > UINT64_MAX / row_bytes ||
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
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_Q8_1) {
        const uint32_t block_cols = 32u;
        const uint32_t block_bytes = 36u;
        uint32_t dst = 0;
        const uint8_t* block = row_ptr;
        while (dst < out_count) {
            uint16_t d16 = 0;
            std::memcpy(&d16, block, sizeof(d16));
            const float d = moe_raw_fp16_to_f32(d16);
            const int8_t* q = reinterpret_cast<const int8_t*>(block + 4u);
            const uint32_t limit = std::min<uint32_t>(block_cols, out_count - dst);
            for (uint32_t i = 0; i < limit; ++i) {
                out[dst + i] = d * (float)q[i];
            }
            dst += limit;
            block += block_bytes;
        }
        return 1;
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_Q8_K) {
        const uint32_t block_cols = 256u;
        const uint32_t block_bytes = 292u;
        uint32_t dst = 0;
        const uint8_t* block = row_ptr;
        while (dst < out_count) {
            float d = 0.0f;
            std::memcpy(&d, block, sizeof(d));
            const int8_t* q = reinterpret_cast<const int8_t*>(block + 4u);
            const uint32_t limit = std::min<uint32_t>(block_cols, out_count - dst);
            for (uint32_t i = 0; i < limit; ++i) {
                out[dst + i] = d * (float)q[i];
            }
            dst += limit;
            block += block_bytes;
        }
        return 1;
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_Q4_0) {
        const uint32_t block_cols = 32u;
        const uint32_t block_bytes = 18u;
        uint32_t dst = 0;
        const uint8_t* block = row_ptr;
        while (dst < out_count) {
            uint16_t d16 = 0;
            std::memcpy(&d16, block, sizeof(d16));
            const float d = moe_raw_fp16_to_f32(d16);
            const uint8_t* qs = block + 2u;
            const uint32_t limit = std::min<uint32_t>(block_cols, out_count - dst);
            for (uint32_t i = 0; i < limit; ++i) {
                const uint8_t byte = qs[i & 15u];
                const uint8_t q = (i < 16u) ? (byte & 0x0fu) : (byte >> 4);
                out[dst + i] = d * (float)((int)q - 8);
            }
            dst += limit;
            block += block_bytes;
        }
        return 1;
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_Q4_1) {
        const uint32_t block_cols = 32u;
        const uint32_t block_bytes = 20u;
        uint32_t dst = 0;
        const uint8_t* block = row_ptr;
        while (dst < out_count) {
            uint16_t d16 = 0;
            uint16_t m16 = 0;
            std::memcpy(&d16, block, sizeof(d16));
            std::memcpy(&m16, block + 2u, sizeof(m16));
            const float d = moe_raw_fp16_to_f32(d16);
            const float m = moe_raw_fp16_to_f32(m16);
            const uint8_t* qs = block + 4u;
            const uint32_t limit = std::min<uint32_t>(block_cols, out_count - dst);
            for (uint32_t i = 0; i < limit; ++i) {
                const uint8_t byte = qs[i & 15u];
                const uint8_t q = (i < 16u) ? (byte & 0x0fu) : (byte >> 4);
                out[dst + i] = d * (float)q + m;
            }
            dst += limit;
            block += block_bytes;
        }
        return 1;
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_Q5_0) {
        const uint32_t block_cols = 32u;
        const uint32_t block_bytes = 22u;
        uint32_t dst = 0;
        const uint8_t* block = row_ptr;
        while (dst < out_count) {
            uint16_t d16 = 0;
            uint32_t qh = 0;
            std::memcpy(&d16, block, sizeof(d16));
            std::memcpy(&qh, block + 2u, sizeof(qh));
            const float d = moe_raw_fp16_to_f32(d16);
            const uint8_t* qs = block + 6u;
            const uint32_t limit = std::min<uint32_t>(block_cols, out_count - dst);
            for (uint32_t i = 0; i < limit; ++i) {
                const uint8_t byte = qs[i & 15u];
                const uint32_t low = (i < 16u) ? (byte & 0x0fu) : (byte >> 4);
                const uint32_t high = (qh >> i) & 1u;
                out[dst + i] = d * (float)((int)(low | (high << 4u)) - 16);
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
                const uint32_t low = (i < 16u) ? (byte & 0x0fu) : (byte >> 4);
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
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_Q2_K) {
        moe_dequant_gguf_q2_k_row(row_ptr, out, out_count);
        return 1;
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_Q3_K) {
        moe_dequant_gguf_q3_k_row(row_ptr, out, out_count);
        return 1;
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_Q4_K) {
        moe_dequant_gguf_q4_k_row(row_ptr, out, out_count);
        return 1;
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_Q5_K) {
        moe_dequant_gguf_q5_k_row(row_ptr, out, out_count);
        return 1;
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_Q6_K) {
        moe_dequant_gguf_q6_k_row(row_ptr, out, out_count);
        return 1;
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_IQ2_XXS) {
        moe_dequant_gguf_iq2_xxs_row(row_ptr, out, out_count);
        return 1;
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_IQ2_XS) {
        moe_dequant_gguf_iq2_xs_row(row_ptr, out, out_count);
        return 1;
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_IQ2_S) {
        moe_dequant_gguf_iq2_s_row(row_ptr, out, out_count);
        return 1;
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_IQ3_XXS) {
        moe_dequant_gguf_iq3_xxs_row(row_ptr, out, out_count);
        return 1;
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_IQ3_S) {
        moe_dequant_gguf_iq3_s_row(row_ptr, out, out_count);
        return 1;
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_IQ4_NL) {
        moe_dequant_gguf_iq4_nl_row(row_ptr, out, out_count);
        return 1;
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_IQ4_XS) {
        moe_dequant_gguf_iq4_xs_row(row_ptr, out, out_count);
        return 1;
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_IQ1_S) {
        moe_dequant_gguf_iq1_s_row(row_ptr, out, out_count);
        return 1;
    }
    if (raw.weight_format == moe_WEIGHT_ENCODING_GGUF_IQ1_M) {
        moe_dequant_gguf_iq1_m_row(row_ptr, out, out_count);
        return 1;
    }
    if (raw.weight_format != moe_WEIGHT_ENCODING_UNKNOWN &&
        raw.weight_format != moe_WEIGHT_ENCODING_RAW_FP16 &&
        raw.weight_format != moe_WEIGHT_ENCODING_RAW_BF16 &&
        raw.weight_format != moe_WEIGHT_ENCODING_RAW_FP32) {
        moe_trace_raw_dot_unsupported(raw);
        return 0;
    }
    const uint32_t elem = moe_raw_scalar_size(raw.scalar_encoding);
    for (uint32_t i = 0; i < out_count; ++i) {
        out[i] = moe_raw_scalar_to_f32(row_ptr + (uint64_t)i * (uint64_t)elem, raw.scalar_encoding);
    }
    return 1;
}
