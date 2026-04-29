#if defined(__AVX2__)
static inline float glm5_hsum256_ps(__m256 v) {
#if defined(__AVX2__)
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
#else
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, v);
    return lanes[0] + lanes[1] + lanes[2] + lanes[3] +
           lanes[4] + lanes[5] + lanes[6] + lanes[7];
#endif
}
#endif

static inline float glm5_dot_f32(const float* a, const float* b, uint32_t count) {
    float sum = 0.0f;
    uint32_t i = 0;
#if defined(__AVX2__)
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    for (; i + 15u < count; i += 16u) {
        const __m256 av0 = _mm256_loadu_ps(a + i);
        const __m256 bv0 = _mm256_loadu_ps(b + i);
        const __m256 av1 = _mm256_loadu_ps(a + i + 8u);
        const __m256 bv1 = _mm256_loadu_ps(b + i + 8u);
        acc0 = glm5_madd_ps(av0, bv0, acc0);
        acc1 = glm5_madd_ps(av1, bv1, acc1);
    }
    sum = glm5_hsum256_ps(_mm256_add_ps(acc0, acc1));
#endif
    for (; i < count; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

static inline void glm5_scale_f32(float* values, float scale, uint32_t count) {
    uint32_t i = 0;
#if defined(__AVX2__)
    const __m256 vscale = _mm256_set1_ps(scale);
    for (; i + 7u < count; i += 8u) {
        __m256 v = _mm256_loadu_ps(values + i);
        v = _mm256_mul_ps(v, vscale);
        _mm256_storeu_ps(values + i, v);
    }
#endif
    for (; i < count; ++i) {
        values[i] *= scale;
    }
}

static inline void glm5_axpy_f32(float* dst, const float* src, float alpha, uint32_t count) {
    uint32_t i = 0;
#if defined(__AVX2__)
    const __m256 valpha = _mm256_set1_ps(alpha);
    for (; i + 7u < count; i += 8u) {
        const __m256 s = _mm256_loadu_ps(src + i);
        __m256 d = _mm256_loadu_ps(dst + i);
        d = glm5_madd_ps(valpha, s, d);
        _mm256_storeu_ps(dst + i, d);
    }
#endif
    for (; i < count; ++i) {
        dst[i] += alpha * src[i];
    }
}

static inline void glm5_axpy_bf16_row_f32(float* dst, const uint8_t* row, float alpha, uint32_t count) {
    uint32_t i = 0;
#if defined(__AVX2__)
    const __m256 valpha = _mm256_set1_ps(alpha);
    for (; i + 7u < count; i += 8u) {
        const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + (uint64_t)i * 2ull));
        const __m256i expanded = _mm256_cvtepu16_epi32(packed);
        const __m256 values = _mm256_castsi256_ps(_mm256_slli_epi32(expanded, 16));
        __m256 accum = _mm256_loadu_ps(dst + i);
        accum = glm5_madd_ps(valpha, values, accum);
        _mm256_storeu_ps(dst + i, accum);
    }
#endif
    for (; i < count; ++i) {
        uint16_t bf16;
        std::memcpy(&bf16, row + (uint64_t)i * 2ull, sizeof(bf16));
        dst[i] += alpha * glm5_raw_bf16_to_f32(bf16);
    }
}
