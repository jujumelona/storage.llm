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

static int moe_graph_ir_tensor_binding_name(
    const moe_pc_engine* engine,
    const char* key,
    std::string* out
) {
    if (out) {
        out->clear();
    }
    if (!engine || !key || !key[0] || !out || engine->offload_graph_ir_json.empty()) {
        return 0;
    }
    size_t begin = 0;
    size_t end = 0;
    if (moe_json_find_object_local(engine->offload_graph_ir_json, "tensor_bindings", &begin, &end) &&
        moe_json_get_string_slice(engine->offload_graph_ir_json, begin, end, key, out) &&
        !out->empty()) {
        return 1;
    }
    return moe_json_get_string_local(engine->offload_graph_ir_json, key, out) && !out->empty();
}

static uint32_t moe_raw_scalar_size(uint8_t encoding) {
    if (encoding == moe_RAW_SCALAR_F32) {
        return 4u;
    }
    if (encoding == moe_RAW_SCALAR_F16 || encoding == moe_RAW_SCALAR_BF16) {
        return 2u;
    }
    return 0u;
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
    if (encoding == moe_RAW_SCALAR_F16) {
        return moe_raw_fp16_to_f32(v);
    }
    if (encoding == moe_RAW_SCALAR_BF16) {
        return moe_raw_bf16_to_f32(v);
    }
    return 0.0f;
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
    // BUGFIX 495: rel_offset + bytes overflow 泥댄겕
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
    // BUGFIX 712: BF16 dot product NaN handling ★★
    // Problem: No validation of row or hidden input, no NaN check on result
    // Impact: NaN in weights or activations propagates to PPL
    // Solution: Check inputs and final result for NaN/Inf
    if (!row || !hidden) {
        return 0.0f;
    }
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
    // BUGFIX 712: Final NaN/Inf check
    if (!std::isfinite(acc)) {
        return 0.0f;
    }
    return acc;
}

static float moe_dot_raw_row(
    const uint8_t* row,
    uint8_t encoding,
    const float* hidden,
    uint32_t hidden_count
) {
    // BUGFIX 529: Check hidden_count upper bound to prevent excessive iteration ??    // Problem: hidden_count can be arbitrarily large from config
    // Loop can run billions of iterations
    // Solution: Add reasonable upper bound check (e.g., 100000)
    if (!row || !hidden || hidden_count > 100000) {
        return 0.0f;
    }
    if (encoding == moe_RAW_SCALAR_BF16) {
        return moe_dot_bf16_row(row, hidden, hidden_count);
    }
    const uint32_t elem = moe_raw_scalar_size(encoding);
    if (!elem) {
        return 0.0f;
    }
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
    const std::string suffix_lower = moe_lower_ascii_copy(std::string(suffix));
    char p0[64], p1[64], p2[64], p3[64], p4[64];
    std::snprintf(p0, sizeof(p0), "blk.%d.", (int)layer);
    std::snprintf(p1, sizeof(p1), "model.layers.%d.", (int)layer);
    std::snprintf(p2, sizeof(p2), "layers.%d.", (int)layer);
    std::snprintf(p3, sizeof(p3), "transformer.h.%d.", (int)layer);
    std::snprintf(p4, sizeof(p4), "h.%d.", (int)layer);
    const char* prefixes[] = {p0, p1, p2, p3, p4};
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
        for (const char* prefix : prefixes) {
            const size_t prefix_len = std::strlen(prefix);
            if (key.size() >= prefix_len + suffix_lower.size() &&
                key.compare(0, prefix_len, prefix) == 0 &&
                key.compare(key.size() - suffix_lower.size(), suffix_lower.size(), suffix_lower) == 0) {
                result = 1;
                break;
            }
        }
        if (result) break;
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
