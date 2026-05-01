// Raw forward - Embedding operations

static int moe_copy_embedding_row_f32(
    const moe_storage_constants_t* c,
    const uint8_t* base,
    uint64_t bytes,
    uint8_t encoding,
    uint8_t weight_format,
    uint64_t weight_row_bytes,
    uint32_t token_id,
    float* out_hidden
) {
    if (!c || !base || !out_hidden || token_id >= c->vocab_size) {
        return 0;
    }
    moe_raw_tensor_cache_entry raw{};
    raw.base = base;
    raw.bytes = bytes;
    raw.rows = c->vocab_size;
    raw.cols = c->hidden_size;
    raw.scalar_encoding = encoding;
    raw.weight_format = weight_format;
    raw.weight_row_bytes = weight_row_bytes;
    const uint64_t row_bytes = moe_raw_tensor_row_bytes(raw, c->hidden_size);
    const uint64_t row_offset = row_bytes * (uint64_t)token_id;
    if (row_offset > bytes || row_bytes > bytes - row_offset) {
        return 0;
    }
    const uint8_t* row = base + row_offset;
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q8_0 ||
        weight_format == moe_WEIGHT_ENCODING_GGUF_Q5_1 ||
        weight_format == moe_WEIGHT_ENCODING_MXFP4) {
        return moe_raw_tensor_copy_row_f32(raw, token_id, out_hidden, c->hidden_size);
    }
    const uint32_t elem = moe_raw_scalar_size(encoding);
    uint32_t i = 0;
#if defined(__AVX2__)
    if (encoding == moe_RAW_SCALAR_BF16) {
    for (; i + 7 < c->hidden_size; i += 8) {
        __m128i h = _mm_loadu_si128((const __m128i*)(row + (uint64_t)i * 2ull));
        __m256i exp = _mm256_cvtepu16_epi32(h);
        __m256 f = _mm256_castsi256_ps(_mm256_slli_epi32(exp, 16));
        _mm256_storeu_ps(out_hidden + i, f);
    }
    }
#endif
    for (; i < c->hidden_size; ++i) {
        out_hidden[i] = moe_raw_scalar_to_f32(row + (uint64_t)i * (uint64_t)elem, encoding);
    }
    return 1;
}

static int moe_map_embedding_tensor(
    moe_pc_engine* engine,
    const moe_storage_constants_t* c,
    moe_raw_tensor_cache_entry* out
) {
    if (!engine || !c || !out) {
        return 0;
    }
    static const char* const embedding_names[] = {
        "token_embd.weight",
        "model.embed_tokens.weight",
        "embed_tokens.weight",
        "transformer.wte.weight"
    };
    moe_raw_tensor_desc desc{};
    const moe_gguf_common_tensor_record* rec =
        find_common_tensor_by_names(engine, embedding_names, sizeof(embedding_names) / sizeof(embedding_names[0]), &desc);
    if (!rec || desc.cols != c->hidden_size || desc.rows < c->vocab_size) {
        return 0;
    }
    return map_common_tensor_record(engine, *rec, desc, out);
}

int moe_pc_engine_embed_token_f32(
    moe_pc_engine_t* engine,
    uint32_t token_id,
    float* out_hidden,
    uint32_t out_count
) {
    const moe_storage_constants_t runtime_c = moe_engine_storage_constants_or_static(engine);
    const moe_storage_constants_t* c = &runtime_c;
    if (!engine || !c || !out_hidden || out_count != c->hidden_size || token_id >= c->vocab_size) {
        return 0;
    }
    moe_raw_tensor_cache_entry common{};
    if (moe_map_embedding_tensor(engine, c, &common)) {
        return moe_copy_embedding_row_f32(
            c, common.base, common.bytes, common.scalar_encoding,
            common.weight_format, common.weight_row_bytes, token_id, out_hidden);
    }
    const moe_storage_raw_span_spec_t* span = find_raw_role_span(-1, moe_RAW_ROLE_EMBED);
    mmap_context_t* mapped = map_raw_span_part(engine, span);
    if (!raw_span_bounds_ok(span, mapped)) {
        return 0;
    }
    const uint8_t* base = raw_span_ptr(span, mapped, 0, span->bytes);
    return moe_copy_embedding_row_f32(
        c, base, span->bytes, moe_RAW_SCALAR_BF16,
        moe_WEIGHT_ENCODING_RAW_FP16, (uint64_t)c->hidden_size * 2ull,
        token_id, out_hidden);
}

int moe_pc_engine_embed_tokens_f32(
    moe_pc_engine_t* engine,
    const int32_t* token_ids,
    uint32_t token_count,
    float* out_hidden,
    uint32_t out_count
) {
    const moe_storage_constants_t runtime_c = moe_engine_storage_constants_or_static(engine);
    const moe_storage_constants_t* c = &runtime_c;
    if (!engine || !c || !token_ids || !out_hidden || token_count == 0 || c->hidden_size == 0) {
        return 0;
    }
    const uint64_t expected = (uint64_t)token_count * (uint64_t)c->hidden_size;
    if (expected > UINT32_MAX || out_count != (uint32_t)expected) {
        return 0;
    }
    moe_raw_tensor_cache_entry common{};
    if (moe_map_embedding_tensor(engine, c, &common)) {
        return moe_parallel_rows(engine, token_count, 0, [&](uint32_t row) {
            const int32_t token = token_ids[row];
            if (token < 0) {
                return 0;
            }
            float* dst = out_hidden + (uint64_t)row * (uint64_t)c->hidden_size;
            return moe_copy_embedding_row_f32(
                c, common.base, common.bytes, common.scalar_encoding,
                common.weight_format, common.weight_row_bytes, (uint32_t)token, dst);
        });
    }
    const moe_storage_raw_span_spec_t* span = find_raw_role_span(-1, moe_RAW_ROLE_EMBED);
    mmap_context_t* mapped = map_raw_span_part(engine, span);
    if (!raw_span_bounds_ok(span, mapped)) {
        return 0;
    }
    const uint8_t* base = raw_span_ptr(span, mapped, 0, span->bytes);
    return moe_parallel_rows(engine, token_count, 0, [&](uint32_t row) {
        const int32_t token = token_ids[row];
        if (token < 0) {
            return 0;
        }
        float* dst = out_hidden + (uint64_t)row * (uint64_t)c->hidden_size;
        return moe_copy_embedding_row_f32(
            c, base, span->bytes, moe_RAW_SCALAR_BF16,
            moe_WEIGHT_ENCODING_RAW_FP16, (uint64_t)c->hidden_size * 2ull,
            (uint32_t)token, dst);
    });
}
