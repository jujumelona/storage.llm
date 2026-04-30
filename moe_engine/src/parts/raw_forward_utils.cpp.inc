// Raw forward - Utility functions

static float moe_raw_bf16_to_f32(uint16_t v) {
    const uint32_t bits = (uint32_t)v << 16u;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
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
