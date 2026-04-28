static float glm5_raw_bf16_to_f32(uint16_t v) {
    const uint32_t bits = (uint32_t)v << 16u;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

static const glm5_storage_raw_span_spec_t* find_raw_role_span(int32_t layer, uint32_t role) {
    const uint32_t begin = glm5_storage_raw_span_find_first(layer);
    const uint32_t end = glm5_storage_raw_span_find_end(layer);
    for (uint32_t i = begin; i < end; ++i) {
        const glm5_storage_raw_span_spec_t* s = glm5_storage_raw_span_at(i);
        if (s && (s->role & role) != 0 && s->bytes > 0) {
            return s;
        }
    }
    return nullptr;
}

static mmap_context_t* map_raw_span_part(glm5_pc_engine* engine, const glm5_storage_raw_span_spec_t* span) {
    if (!engine || !span || engine->model_root.empty()) {
        return nullptr;
    }
    const glm5_storage_part_spec_t* part = glm5_storage_part_by_id(span->part);
    return part && part->path ? get_or_map_shard_partial_by_id(
        engine,
        span->part,
        resolve_model_file_path(engine->model_root, part->path, ""),
        span->offset,
        span->bytes
    ) : nullptr;
}

static int raw_span_bounds_ok(const glm5_storage_raw_span_spec_t* span, const mmap_context_t* mapped) {
    return span && mapped && mmap_get_ptr(mapped, 0) &&
        span->bytes <= (uint64_t)mmap_get_size(mapped);
}

static const uint8_t* raw_span_ptr(
    const glm5_storage_raw_span_spec_t* span,
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

struct glm5_raw_tensor_desc {
    uint32_t role;
    uint64_t span_bytes;
    uint32_t occurrence;
    uint64_t rel_offset;
    uint64_t bytes;
    uint32_t rows;
    uint32_t cols;
};

static int raw_tensor_desc_for_kind(glm5_raw_tensor_kind_t kind, glm5_raw_tensor_desc* out) {
    if (!out) {
        return 0;
    }
    switch (kind) {
        case GLM5_RAW_TENSOR_INPUT_LAYER_NORM:
            *out = {GLM5_RAW_ROLE_NORM, 12288ull, 0u, 0ull, 12288ull, 6144u, 0u};
            return 1;
        case GLM5_RAW_TENSOR_POST_ATTENTION_LAYER_NORM:
            *out = {GLM5_RAW_ROLE_NORM, 12288ull, 1u, 0ull, 12288ull, 6144u, 0u};
            return 1;
        case GLM5_RAW_TENSOR_Q_A_LAYER_NORM:
            *out = {GLM5_RAW_ROLE_NORM, 4096ull, 0u, 0ull, 4096ull, 2048u, 0u};
            return 1;
        case GLM5_RAW_TENSOR_KV_A_LAYER_NORM:
            *out = {GLM5_RAW_ROLE_NORM, 1024ull, 0u, 0ull, 1024ull, 512u, 0u};
            return 1;
        case GLM5_RAW_TENSOR_Q_A_PROJ:
            *out = {GLM5_RAW_ROLE_ATTENTION, 92274688ull, 0u, 0ull, 25165824ull, 2048u, 6144u};
            return 1;
        case GLM5_RAW_TENSOR_Q_B_PROJ:
            *out = {GLM5_RAW_ROLE_ATTENTION, 92274688ull, 0u, 25165824ull, 67108864ull, 16384u, 2048u};
            return 1;
        case GLM5_RAW_TENSOR_KV_A_PROJ_WITH_MQA:
            *out = {GLM5_RAW_ROLE_ATTENTION, 237764608ull, 0u, 0ull, 7077888ull, 576u, 6144u};
            return 1;
        case GLM5_RAW_TENSOR_KV_B_PROJ:
            *out = {GLM5_RAW_ROLE_ATTENTION, 237764608ull, 0u, 7077888ull, 29360128ull, 28672u, 512u};
            return 1;
        case GLM5_RAW_TENSOR_O_PROJ:
            *out = {GLM5_RAW_ROLE_ATTENTION, 237764608ull, 0u, 36438016ull, 201326592ull, 6144u, 16384u};
            return 1;
        case GLM5_RAW_TENSOR_INDEXER_K_NORM_BIAS:
            *out = {GLM5_RAW_ROLE_ATTENTION, 18743808ull, 0u, 0ull, 256ull, 128u, 0u};
            return 1;
        case GLM5_RAW_TENSOR_INDEXER_K_NORM_WEIGHT:
            *out = {GLM5_RAW_ROLE_ATTENTION, 18743808ull, 0u, 256ull, 256ull, 128u, 0u};
            return 1;
        case GLM5_RAW_TENSOR_INDEXER_WEIGHTS_PROJ:
            *out = {GLM5_RAW_ROLE_ATTENTION, 18743808ull, 0u, 512ull, 393216ull, 32u, 6144u};
            return 1;
        case GLM5_RAW_TENSOR_INDEXER_WK:
            *out = {GLM5_RAW_ROLE_ATTENTION, 18743808ull, 0u, 393728ull, 1572864ull, 128u, 6144u};
            return 1;
        case GLM5_RAW_TENSOR_INDEXER_WQ_B:
            *out = {GLM5_RAW_ROLE_ATTENTION, 18743808ull, 0u, 1966592ull, 16777216ull, 4096u, 2048u};
            return 1;
        default:
            return 0;
    }
}

static const glm5_storage_raw_span_spec_t* find_raw_role_span_bytes(
    int32_t layer,
    uint32_t role,
    uint64_t bytes,
    uint32_t occurrence
) {
    const uint32_t begin = glm5_storage_raw_span_find_first(layer);
    const uint32_t end = glm5_storage_raw_span_find_end(layer);
    uint32_t seen = 0;
    for (uint32_t i = begin; i < end; ++i) {
        const glm5_storage_raw_span_spec_t* s = glm5_storage_raw_span_at(i);
        if (s && (s->role & role) != 0 && s->bytes == bytes) {
            if (seen == occurrence) {
                return s;
            }
            ++seen;
        }
    }
    return nullptr;
}

static int find_raw_tensor_desc_and_span(
    int32_t layer,
    glm5_raw_tensor_kind_t kind,
    glm5_raw_tensor_desc* out_desc,
    const glm5_storage_raw_span_spec_t** out_span
) {
    glm5_raw_tensor_desc desc{};
    if (!raw_tensor_desc_for_kind(kind, &desc)) {
        return 0;
    }
    const glm5_storage_raw_span_spec_t* span =
        find_raw_role_span_bytes(layer, desc.role, desc.span_bytes, desc.occurrence);
    if (!span && desc.role == GLM5_RAW_ROLE_ATTENTION) {
        // Boundary layers can pack the 18,743,808-byte indexer span and the
        // 92,274,688-byte Q span into one 111,018,496-byte attention span.
        const uint64_t combined_indexer_q_bytes = 18743808ull + 92274688ull;
        if (desc.span_bytes == 92274688ull) {
            const glm5_storage_raw_span_spec_t* combined =
                find_raw_role_span_bytes(layer, GLM5_RAW_ROLE_ATTENTION, combined_indexer_q_bytes, 0);
            if (combined) {
                desc.rel_offset += 18743808ull;
                span = combined;
            }
        } else if (desc.span_bytes == 18743808ull) {
            span = find_raw_role_span_bytes(layer, GLM5_RAW_ROLE_ATTENTION, combined_indexer_q_bytes, 0);
        }
    }
    if (!span || desc.rel_offset > span->bytes || desc.bytes > span->bytes - desc.rel_offset) {
        return 0;
    }
    if (out_desc) {
        *out_desc = desc;
    }
    if (out_span) {
        *out_span = span;
    }
    return 1;
}

static int raw_span_backing_file_ready(
    const glm5_pc_engine* engine,
    const glm5_storage_raw_span_spec_t* span
) {
    if (!engine || engine->model_root.empty() || !span) {
        return 0;
    }
    const glm5_storage_part_spec_t* part = glm5_storage_part_by_id(span->part);
    if (!part || !part->path) {
        return 0;
    }
    uint64_t file_bytes = 0;
    {
        std::lock_guard<std::mutex> lock(engine->raw_file_size_cache_mutex);
        auto it = engine->raw_file_size_cache.find(span->part);
        if (it != engine->raw_file_size_cache.end()) {
            file_bytes = it->second;
        } else {
            const std::string path = resolve_model_file_path(engine->model_root, part->path, "");
            if (!file_size_bytes(path, &file_bytes)) {
                return 0;
            }
            engine->raw_file_size_cache[span->part] = file_bytes;
        }
    }
    return span->offset <= file_bytes && span->bytes <= file_bytes - span->offset ? 1 : 0;
}

static int glm5_attention_projectors_ready(const glm5_pc_engine* engine) {
    const glm5_storage_constants_t* c = glm5_storage_constants();
    if (!engine || engine->model_root.empty() || !c || c->num_hidden_layers == 0) {
        return 0;
    }
    const glm5_raw_tensor_kind_t required[] = {
        GLM5_RAW_TENSOR_INPUT_LAYER_NORM,
        GLM5_RAW_TENSOR_POST_ATTENTION_LAYER_NORM,
        GLM5_RAW_TENSOR_Q_A_LAYER_NORM,
        GLM5_RAW_TENSOR_KV_A_LAYER_NORM,
        GLM5_RAW_TENSOR_Q_A_PROJ,
        GLM5_RAW_TENSOR_Q_B_PROJ,
        GLM5_RAW_TENSOR_KV_A_PROJ_WITH_MQA,
        GLM5_RAW_TENSOR_KV_B_PROJ,
        GLM5_RAW_TENSOR_O_PROJ
    };
    for (uint32_t layer = 0; layer < c->num_hidden_layers; ++layer) {
        for (glm5_raw_tensor_kind_t kind : required) {
            const glm5_storage_raw_span_spec_t* span = nullptr;
            if (!find_raw_tensor_desc_and_span((int32_t)layer, kind, nullptr, &span) ||
                !raw_span_backing_file_ready(engine, span)) {
                return 0;
            }
        }
    }
    return 1;
}

static int glm5_raw_embedding_ready(const glm5_pc_engine* engine) {
    const glm5_storage_constants_t* c = glm5_storage_constants();
    const glm5_storage_raw_span_spec_t* span = find_raw_role_span(-1, GLM5_RAW_ROLE_EMBED);
    if (!engine || engine->model_root.empty() || !c || !span ||
        c->hidden_size == 0 || c->vocab_size == 0) {
        return 0;
    }
    return span->bytes >= (uint64_t)c->hidden_size * (uint64_t)c->vocab_size * 2ull &&
        raw_span_backing_file_ready(engine, span) ? 1 : 0;
}

static int glm5_raw_lm_head_ready(const glm5_pc_engine* engine) {
    const glm5_storage_constants_t* c = glm5_storage_constants();
    const int32_t lm_head_layer = c ? (int32_t)c->num_hidden_layers + 1 : 80;
    const glm5_storage_raw_span_spec_t* span = find_raw_role_span(lm_head_layer, GLM5_RAW_ROLE_LM_HEAD);
    if (!engine || engine->model_root.empty() || !c || !span ||
        c->hidden_size == 0 || c->vocab_size == 0) {
        return 0;
    }
    return span->bytes >= (uint64_t)c->hidden_size * (uint64_t)c->vocab_size * 2ull &&
        raw_span_backing_file_ready(engine, span) ? 1 : 0;
}

static int glm5_copy_embedding_row_f32(
    const glm5_storage_constants_t* c,
    const glm5_storage_raw_span_spec_t* span,
    const mmap_context_t* mapped,
    uint32_t token_id,
    float* out_hidden
) {
    if (!c || !span || !mapped || !out_hidden || token_id >= c->vocab_size) {
        return 0;
    }
    const uint64_t row_bytes = (uint64_t)c->hidden_size * 2ull;
    const uint64_t row_offset = row_bytes * (uint64_t)token_id;
    const uint8_t* row = raw_span_ptr(span, mapped, row_offset, row_bytes);
    if (!row) {
        return 0;
    }
    uint32_t i = 0;
#if defined(__AVX2__)
    for (; i + 7 < c->hidden_size; i += 8) {
        __m128i h = _mm_loadu_si128((const __m128i*)(row + (uint64_t)i * 2ull));
        __m256i exp = _mm256_cvtepu16_epi32(h);
        __m256 f = _mm256_castsi256_ps(_mm256_slli_epi32(exp, 16));
        _mm256_storeu_ps(out_hidden + i, f);
    }
#endif
    for (; i < c->hidden_size; ++i) {
        uint16_t v;
        std::memcpy(&v, row + (uint64_t)i * 2ull, sizeof(v));
        out_hidden[i] = glm5_raw_bf16_to_f32(v);
    }
    return 1;
}

int glm5_pc_engine_embed_token_f32(
    glm5_pc_engine_t* engine,
    uint32_t token_id,
    float* out_hidden,
    uint32_t out_count
) {
    const glm5_storage_constants_t* c = glm5_storage_constants();
    if (!engine || !c || !out_hidden || out_count != c->hidden_size || token_id >= c->vocab_size) {
        return 0;
    }
    const glm5_storage_raw_span_spec_t* span = find_raw_role_span(-1, GLM5_RAW_ROLE_EMBED);
    mmap_context_t* mapped = map_raw_span_part(engine, span);
    if (!raw_span_bounds_ok(span, mapped)) {
        return 0;
    }
    return glm5_copy_embedding_row_f32(c, span, mapped, token_id, out_hidden);
}

int glm5_pc_engine_embed_tokens_f32(
    glm5_pc_engine_t* engine,
    const int32_t* token_ids,
    uint32_t token_count,
    float* out_hidden,
    uint32_t out_count
) {
    const glm5_storage_constants_t* c = glm5_storage_constants();
    if (!engine || !c || !token_ids || !out_hidden || token_count == 0 || c->hidden_size == 0) {
        return 0;
    }
    const uint64_t expected = (uint64_t)token_count * (uint64_t)c->hidden_size;
    if (expected > UINT32_MAX || out_count != (uint32_t)expected) {
        return 0;
    }
    const glm5_storage_raw_span_spec_t* span = find_raw_role_span(-1, GLM5_RAW_ROLE_EMBED);
    mmap_context_t* mapped = map_raw_span_part(engine, span);
    if (!raw_span_bounds_ok(span, mapped)) {
        return 0;
    }
    return glm5_parallel_rows(engine, token_count, 0, [&](uint32_t row) {
        const int32_t token = token_ids[row];
        if (token < 0) {
            return 0;
        }
        float* dst = out_hidden + (uint64_t)row * (uint64_t)c->hidden_size;
        return glm5_copy_embedding_row_f32(c, span, mapped, (uint32_t)token, dst);
    });
}

static float glm5_dot_bf16_row(const uint8_t* row, const float* hidden, uint32_t hidden_count) {
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
        sum0 = _mm256_fmadd_ps(f0, x0, sum0);
        sum1 = _mm256_fmadd_ps(f1, x1, sum1);
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
        a0 += glm5_raw_bf16_to_f32(v0) * hidden[i + 0u];
        a1 += glm5_raw_bf16_to_f32(v1) * hidden[i + 1u];
        a2 += glm5_raw_bf16_to_f32(v2) * hidden[i + 2u];
        a3 += glm5_raw_bf16_to_f32(v3) * hidden[i + 3u];
    }
    acc += (a0 + a1) + (a2 + a3);
    for (; i < hidden_count; ++i) {
        uint16_t v;
        std::memcpy(&v, row + (uint64_t)i * 2ull, sizeof(v));
        acc += glm5_raw_bf16_to_f32(v) * hidden[i];
    }
    return acc;
}

int glm5_pc_engine_find_raw_tensor(
    glm5_pc_engine_t* engine,
    int32_t layer,
    glm5_raw_tensor_kind_t kind,
    glm5_raw_tensor_info_t* out_info
) {
    (void)engine;
    if (!out_info) {
        return 0;
    }
    glm5_raw_tensor_desc desc{};
    const glm5_storage_raw_span_spec_t* span = nullptr;
    if (!find_raw_tensor_desc_and_span(layer, kind, &desc, &span)) {
        return 0;
    }
    std::memset(out_info, 0, sizeof(*out_info));
    out_info->layer = layer;
    out_info->part = span->part;
    out_info->offset = span->offset + desc.rel_offset;
    out_info->bytes = desc.bytes;
    out_info->rows = desc.rows;
    out_info->cols = desc.cols;
    out_info->bf16 = 1;
    return 1;
}

int glm5_pc_engine_raw_linear_f32(
    glm5_pc_engine_t* engine,
    int32_t layer,
    glm5_raw_tensor_kind_t kind,
    const float* input,
    uint32_t input_count,
    float* output,
    uint32_t output_count
) {
    if (!engine || !input || !output) {
        return 0;
    }
    glm5_raw_tensor_desc desc{};
    const glm5_storage_raw_span_spec_t* span = nullptr;
    if (!find_raw_tensor_desc_and_span(layer, kind, &desc, &span) ||
        desc.cols == 0 || input_count != desc.cols || output_count != desc.rows) {
        return 0;
    }
    mmap_context_t* mapped = map_raw_span_part(engine, span);
    if (!raw_span_bounds_ok(span, mapped)) {
        return 0;
    }
    const uint8_t* base = raw_span_ptr(span, mapped, desc.rel_offset, desc.bytes);
    if (!base) {
        return 0;
    }
    const uint64_t row_bytes = (uint64_t)desc.cols * 2ull;
    return glm5_parallel_rows(engine, desc.rows, 0, [&](uint32_t row) {
        output[row] = glm5_dot_bf16_row(base + row_bytes * (uint64_t)row, input, input_count);
        return 1;
    });
}

int glm5_pc_engine_raw_rmsnorm_f32(
    glm5_pc_engine_t* engine,
    int32_t layer,
    glm5_raw_tensor_kind_t kind,
    const float* input,
    uint32_t input_count,
    float* output,
    uint32_t output_count,
    float eps
) {
    if (!engine || !input || !output || input_count == 0 || output_count != input_count) {
        return 0;
    }
    glm5_raw_tensor_desc desc{};
    const glm5_storage_raw_span_spec_t* span = nullptr;
    if (!find_raw_tensor_desc_and_span(layer, kind, &desc, &span) ||
        desc.cols != 0 || desc.rows != input_count) {
        return 0;
    }
    mmap_context_t* mapped = map_raw_span_part(engine, span);
    if (!raw_span_bounds_ok(span, mapped)) {
        return 0;
    }
    const uint8_t* weights = raw_span_ptr(span, mapped, desc.rel_offset, desc.bytes);
    if (!weights) {
        return 0;
    }
    double ss = 0.0;
    uint32_t i = 0;
#if defined(__AVX2__)
    // Use double precision accumulator for better accuracy with large hidden_size
    __m256d sum0 = _mm256_setzero_pd();
    __m256d sum1 = _mm256_setzero_pd();
    // Fix: Process 16 elements at a time to avoid OOB read on x1
    for (; i + 15 < input_count; i += 16) {
        __m256 x0 = _mm256_loadu_ps(input + i);
        __m256 x1 = _mm256_loadu_ps(input + i + 8);
        // Convert float to double for higher precision accumulation
        __m256d xd0 = _mm256_cvtps_pd(_mm256_castps256_ps128(x0));
        __m256d xd1 = _mm256_cvtps_pd(_mm256_extractf128_ps(x0, 1));
        __m256d xd2 = _mm256_cvtps_pd(_mm256_castps256_ps128(x1));
        __m256d xd3 = _mm256_cvtps_pd(_mm256_extractf128_ps(x1, 1));
        sum0 = _mm256_fmadd_pd(xd0, xd0, sum0);
        sum1 = _mm256_fmadd_pd(xd1, xd1, sum1);
        sum0 = _mm256_fmadd_pd(xd2, xd2, sum0);
        sum1 = _mm256_fmadd_pd(xd3, xd3, sum1);
    }
    sum0 = _mm256_add_pd(sum0, sum1);
    double buf[4];
    _mm256_storeu_pd(buf, sum0);
    ss = buf[0] + buf[1] + buf[2] + buf[3];
#endif
    for (; i < input_count; ++i) {
        ss += (double)input[i] * (double)input[i];
    }
    const float norm = 1.0f / std::sqrt((float)(ss / (double)input_count) + (eps > 0.0f ? eps : 1.0e-6f));
    return glm5_parallel_rows(engine, input_count, 0, [&](uint32_t i) {
        uint16_t w;
        std::memcpy(&w, weights + (uint64_t)i * 2ull, sizeof(w));
        output[i] = input[i] * norm * glm5_raw_bf16_to_f32(w);
        return 1;
    });
}

// GAP 1A: Quantize plain float K/V into persistent per-head qkv_state_t
// Called once per head when context changes; state persists across decode calls
static qkv_state_t* glm5_qkv_ensure_head_state(
    glm5_pc_engine* engine,
    uint32_t head,
    const float* head_keys,
    const float* head_values,
    uint32_t context_tokens,
    uint32_t head_dim
) {
    if (!engine || !head_keys || !head_values || !context_tokens || !head_dim) return nullptr;
    std::lock_guard<std::mutex> lock(engine->qkv_mutex);

    // Initialize config once
    if (!engine->qkv_cfg_initialized.load(std::memory_order_acquire)) {
        engine->qkv_cfg = qkv_config_default((int)head_dim);
        engine->qkv_cfg_initialized.store(true, std::memory_order_release);
    }

    // Check if state needs (re)allocation — context size changed
    if (engine->qkv_head_state_slots.size() <= head) {
        engine->qkv_head_state_slots.resize((size_t)head + 1u);
        engine->qkv_head_state_valid.resize((size_t)head + 1u, 0);
    }
    qkv_state_t* slot = &engine->qkv_head_state_slots[head];
    if (engine->qkv_head_state_valid[head] && slot->n_tokens == (int)context_tokens) {
        if (!qkv_quantize(slot, &engine->qkv_cfg, head_keys, head_values, (int)context_tokens)) {
            return nullptr;
        }
        return slot;
    }

    if (engine->qkv_head_state_valid[head]) {
        qkv_free(slot);
        *slot = qkv_state_t{};
        engine->qkv_head_state_valid[head] = 0;
    }

/*
    auto it = engine->qkv_head_states.find(head);
    if (it != engine->qkv_head_states.end() && it->second.n_tokens == (int)context_tokens) {
        // Already quantized for this context size — re-quantize with new data
        qkv_state_t* st = &it->second;
        qkv_quantize(st, &engine->qkv_cfg, head_keys, head_values, (int)context_tokens);
        return st;
    }

    // Free old state if exists
    if (it != engine->qkv_head_states.end()) {
        qkv_free(&it->second);
        engine->qkv_head_states.erase(it);
    }

*/

    // Allocate new state
    qkv_state_t st{};
    if (!qkv_init(&st, &engine->qkv_cfg, (int)context_tokens)) {
        return nullptr;
    }
    if (!qkv_quantize(&st, &engine->qkv_cfg, head_keys, head_values, (int)context_tokens)) {
        qkv_free(&st);
        return nullptr;
    }
    *slot = st;
    engine->qkv_head_state_valid[head] = 1;
    return slot;
}

int glm5_pc_engine_attention_decode_f32(
    glm5_pc_engine_t* engine,
    const float* query,
    const float* key_cache,
    const float* value_cache,
    uint32_t context_tokens,
    uint32_t head_count,
    uint32_t head_dim,
    float* output
) {
    if (!engine || !query || !key_cache || !value_cache || !output ||
        context_tokens == 0 || head_count == 0 || head_dim == 0) {
        return 0;
    }
    const float inv_sqrt = 1.0f / std::sqrt((float)head_dim);
    if (context_tokens > (uint32_t)std::numeric_limits<int>::max() ||
        head_dim > (uint32_t)std::numeric_limits<int>::max()) {
        return 0;
    }
    return glm5_parallel_rows(engine, head_count, 0, [&](uint32_t head) {
        const float* q = query + (uint64_t)head * (uint64_t)head_dim;
        float* out = output + (uint64_t)head * (uint64_t)head_dim;
        const float* head_keys = key_cache + (uint64_t)head * (uint64_t)context_tokens * (uint64_t)head_dim;
        const float* head_values = value_cache + (uint64_t)head * (uint64_t)context_tokens * (uint64_t)head_dim;
        // GAP 1A: QKV mode — quantize into persistent state, attention in quantized domain
        if (engine->config.kv_mode == GLM5_KV_MODE_QKV && qkv_is_enabled()) {
            qkv_state_t* st = glm5_qkv_ensure_head_state(
                engine, head, head_keys, head_values, context_tokens, head_dim);
            if (!st) return 0;
            return qkv_attention_decode(q, st, &engine->qkv_cfg, context_tokens, head_dim, out);
        }
        float max_score = -std::numeric_limits<float>::infinity();

        static thread_local std::vector<float> tls_scores;
        if (tls_scores.size() < context_tokens) {
            tls_scores.resize(context_tokens);
        }
        float* scores = tls_scores.data();

        for (uint32_t t = 0; t < context_tokens; ++t) {
            // [head][t][d] layout
            const float* k = head_keys + (uint64_t)t * (uint64_t)head_dim;
            float score = 0.0f;
            for (uint32_t d = 0; d < head_dim; ++d) {
                score += q[d] * k[d];
            }
            score *= inv_sqrt;
            scores[t] = score;
            if (score > max_score) {
                max_score = score;
            }
        }
        for (uint32_t d = 0; d < head_dim; ++d) {
            out[d] = 0.0f;
        }
        float denom = 0.0f;
        for (uint32_t t = 0; t < context_tokens; ++t) {
            // [head][t][d] layout
            const float* v = head_values + (uint64_t)t * (uint64_t)head_dim;
            const float weight = std::exp(scores[t] - max_score);
            denom += weight;
            for (uint32_t d = 0; d < head_dim; ++d) {
                out[d] += weight * v[d];
            }
        }
        if (denom <= 0.0f) {
            return 0;
        }
        const float scale = 1.0f / denom;
        for (uint32_t d = 0; d < head_dim; ++d) {
            out[d] *= scale;
        }
        return 1;
    });
}


int glm5_pc_engine_lm_head_top1_f32(
    glm5_pc_engine_t* engine,
    const float* hidden,
    uint32_t hidden_count,
    uint32_t* out_token_id,
    float* out_logit
) {
    const glm5_storage_constants_t* c = glm5_storage_constants();
    if (!engine || !c || !hidden || !out_token_id || !out_logit ||
        hidden_count != c->hidden_size || c->vocab_size == 0) {
        return 0;
    }
    const int32_t lm_head_layer = (int32_t)c->num_hidden_layers + 1;
    const glm5_storage_raw_span_spec_t* span = find_raw_role_span(lm_head_layer, GLM5_RAW_ROLE_LM_HEAD);
    mmap_context_t* mapped = map_raw_span_part(engine, span);
    if (!raw_span_bounds_ok(span, mapped)) {
        return 0;
    }
    const uint64_t row_bytes = (uint64_t)c->hidden_size * 2ull;
    if (span->bytes < row_bytes * (uint64_t)c->vocab_size) {
        return 0;
    }

    const uint8_t* base = raw_span_ptr(span, mapped, 0, row_bytes * (uint64_t)c->vocab_size);
    if (!base) {
        return 0;
    }

    const uint32_t chunk_size = 1024;
    const uint32_t num_chunks = (c->vocab_size + chunk_size - 1) / chunk_size;

    struct chunk_result_t {
        uint32_t token;
        float score;
    };
    // Bug #27: Use thread_local scratch instead of per-call heap allocation
    static thread_local std::vector<chunk_result_t> chunk_results;
    if (chunk_results.size() < num_chunks) {
        chunk_results.resize(num_chunks);
    }

    if (!glm5_parallel_rows(engine, num_chunks, 0, [&](uint32_t chunk) {
        const uint32_t start = chunk * chunk_size;
        const uint32_t end = std::min(start + chunk_size, c->vocab_size);
        uint32_t local_best = start;
        float local_max = -std::numeric_limits<float>::infinity();
        
        for (uint32_t t = start; t < end; ++t) {
            const uint8_t* row = base + row_bytes * (uint64_t)t;
            float s = glm5_dot_bf16_row(row, hidden, hidden_count);
            if (s > local_max) {
                local_max = s;
                local_best = t;
            }
        }
        
        chunk_results[chunk] = {local_best, local_max};
        return 1;
    })) {
        return 0;
    }

    // Single-thread final reduce over chunk_results
    uint32_t best_token = 0;
    float best_score = -std::numeric_limits<float>::infinity();
    for (uint32_t i = 0; i < num_chunks; ++i) {
        if (chunk_results[i].score > best_score) {
            best_score = chunk_results[i].score;
            best_token = chunk_results[i].token;
        }
    }

    *out_token_id = best_token;
    *out_logit = best_score;
    return 1;
}
