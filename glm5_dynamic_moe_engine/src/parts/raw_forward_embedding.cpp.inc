// Raw forward - Embedding operations

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
