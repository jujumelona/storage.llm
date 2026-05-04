static float moe_dot_raw_fp16_row(const uint8_t* packed, const float* x, uint32_t cols) {
#if defined(__AVX512F__) && defined(__F16C__)
    // BUGFIX 708: AVX-512 16-wide FP16 dot product ?�★??
    const uint16_t* fp16 = reinterpret_cast<const uint16_t*>(packed);
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    uint32_t col = 0;
    for (; col + 31u < cols; col += 32u) {
        __m256i h0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(fp16 + col));
        __m256i h1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(fp16 + col + 16u));
        __m512 f0 = _mm512_cvtph_ps(h0);
        __m512 f1 = _mm512_cvtph_ps(h1);
        __m512 x0 = _mm512_loadu_ps(x + col);
        __m512 x1 = _mm512_loadu_ps(x + col + 16u);
        acc0 = _mm512_fmadd_ps(f0, x0, acc0);
        acc1 = _mm512_fmadd_ps(f1, x1, acc1);
    }
    for (; col + 15u < cols; col += 16u) {
        __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(fp16 + col));
        __m512 f = _mm512_cvtph_ps(h);
        __m512 xv = _mm512_loadu_ps(x + col);
        acc0 = _mm512_fmadd_ps(f, xv, acc0);
    }
    acc0 = _mm512_add_ps(acc0, acc1);
    float acc = _mm512_reduce_add_ps(acc0);
    for (; col < cols; ++col) {
        acc += moe_raw_fp16_to_f32(fp16[col]) * x[col];
    }
    return acc;
#elif defined(__F16C__) && defined(__AVX2__)
    const uint16_t* fp16 = reinterpret_cast<const uint16_t*>(packed);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    uint32_t col = 0;
    for (; col + 15u < cols; col += 16u) {
        __m128i h0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(fp16 + col));
        __m128i h1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(fp16 + col + 8u));
        __m256 f0 = _mm256_cvtph_ps(h0);
        __m256 f1 = _mm256_cvtph_ps(h1);
        __m256 x0 = _mm256_loadu_ps(x + col);
        __m256 x1 = _mm256_loadu_ps(x + col + 8u);
        acc0 = moe_madd_ps(f0, x0, acc0);
        acc1 = moe_madd_ps(f1, x1, acc1);
    }
    for (; col + 7u < cols; col += 8u) {
        __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(fp16 + col));
        __m256 f = _mm256_cvtph_ps(h);
        __m256 xv = _mm256_loadu_ps(x + col);
        acc0 = moe_madd_ps(f, xv, acc0);
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    // horizontal sum
    __m128 lo = _mm256_castps256_ps128(acc0);
    __m128 hi = _mm256_extractf128_ps(acc0, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float acc = _mm_cvtss_f32(lo);
    // tail
    for (; col < cols; ++col) {
        acc += moe_raw_fp16_to_f32(fp16[col]) * x[col];
    }
    return acc;
#elif defined(__ARM_NEON) || defined(__aarch64__)
    // BUGFIX 문제9: ARM NEON SIMD for Apple Silicon / ARM64 ?�★??
    // Problem: Only AVX-512/AVX2 paths existed, ARM fell back to scalar loop
    // Impact: 8x slower on Apple Silicon M2/M3 despite Metal zero-copy
    // Solution: Add NEON intrinsics for FP16 dot product
    #include <arm_neon.h>
    const uint16_t* fp16 = reinterpret_cast<const uint16_t*>(packed);
    float32x4_t acc0 = vdupq_n_f32(0.f);
    float32x4_t acc1 = vdupq_n_f32(0.f);
    uint32_t col = 0;
    for (; col + 7u < cols; col += 8u) {
        float16x8_t h = vld1q_f16(reinterpret_cast<const __fp16*>(fp16 + col));
        float32x4_t lo = vcvt_f32_f16(vget_low_f16(h));
        float32x4_t hi = vcvt_f32_f16(vget_high_f16(h));
        float32x4_t x0 = vld1q_f32(x + col);
        float32x4_t x1 = vld1q_f32(x + col + 4u);
        acc0 = vfmaq_f32(acc0, lo, x0);
        acc1 = vfmaq_f32(acc1, hi, x1);
    }
    acc0 = vaddq_f32(acc0, acc1);
    float acc = vaddvq_f32(acc0);
    for (; col < cols; ++col) {
        acc += moe_raw_fp16_to_f32(fp16[col]) * x[col];
    }
    return acc;
#else
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    uint32_t col = 0;
    for (; col + 3u < cols; col += 4u) {
        uint16_t v0, v1, v2, v3;
        std::memcpy(&v0, packed + (uint64_t)(col + 0u) * 2ull, sizeof(v0));
        std::memcpy(&v1, packed + (uint64_t)(col + 1u) * 2ull, sizeof(v1));
        std::memcpy(&v2, packed + (uint64_t)(col + 2u) * 2ull, sizeof(v2));
        std::memcpy(&v3, packed + (uint64_t)(col + 3u) * 2ull, sizeof(v3));
        a0 += moe_raw_fp16_to_f32(v0) * x[col + 0u];
        a1 += moe_raw_fp16_to_f32(v1) * x[col + 1u];
        a2 += moe_raw_fp16_to_f32(v2) * x[col + 2u];
        a3 += moe_raw_fp16_to_f32(v3) * x[col + 3u];
    }
    float acc = (a0 + a1) + (a2 + a3);
    for (; col < cols; ++col) {
        uint16_t v;
        std::memcpy(&v, packed + (uint64_t)col * 2ull, sizeof(v));
        acc += moe_raw_fp16_to_f32(v) * x[col];
    }
    return acc;
#endif
}

static float moe_dot_raw_fp32_row(const uint8_t* packed, const float* x, uint32_t cols) {
#if defined(__AVX512F__)
    // BUGFIX 708: AVX-512 16-wide FP32 dot product ?�★??
    // ~2x throughput over AVX2 on Ice Lake+/Zen4+.
    const float* fp32 = reinterpret_cast<const float*>(packed);
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    uint32_t col = 0;
    for (; col + 31u < cols; col += 32u) {
        __m512 f0 = _mm512_loadu_ps(fp32 + col);
        __m512 f1 = _mm512_loadu_ps(fp32 + col + 16u);
        __m512 x0 = _mm512_loadu_ps(x + col);
        __m512 x1 = _mm512_loadu_ps(x + col + 16u);
        acc0 = _mm512_fmadd_ps(f0, x0, acc0);
        acc1 = _mm512_fmadd_ps(f1, x1, acc1);
    }
    for (; col + 15u < cols; col += 16u) {
        __m512 f = _mm512_loadu_ps(fp32 + col);
        __m512 xv = _mm512_loadu_ps(x + col);
        acc0 = _mm512_fmadd_ps(f, xv, acc0);
    }
    acc0 = _mm512_add_ps(acc0, acc1);
    float acc = _mm512_reduce_add_ps(acc0);
    for (; col < cols; ++col) {
        acc += fp32[col] * x[col];
    }
    return acc;
#elif defined(__AVX2__)
    const float* fp32 = reinterpret_cast<const float*>(packed);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    uint32_t col = 0;
    for (; col + 15u < cols; col += 16u) {
        __m256 f0 = _mm256_loadu_ps(fp32 + col);
        __m256 f1 = _mm256_loadu_ps(fp32 + col + 8u);
        __m256 x0 = _mm256_loadu_ps(x + col);
        __m256 x1 = _mm256_loadu_ps(x + col + 8u);
        acc0 = moe_madd_ps(f0, x0, acc0);
        acc1 = moe_madd_ps(f1, x1, acc1);
    }
    for (; col + 7u < cols; col += 8u) {
        __m256 f = _mm256_loadu_ps(fp32 + col);
        __m256 xv = _mm256_loadu_ps(x + col);
        acc0 = moe_madd_ps(f, xv, acc0);
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    // horizontal sum
    __m128 lo = _mm256_castps256_ps128(acc0);
    __m128 hi = _mm256_extractf128_ps(acc0, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float acc = _mm_cvtss_f32(lo);
    // tail
    for (; col < cols; ++col) {
        acc += fp32[col] * x[col];
    }
    return acc;
#elif defined(__ARM_NEON) || defined(__aarch64__)
    // BUGFIX 문제9: ARM NEON SIMD for Apple Silicon / ARM64 ?�★??
    // Problem: Only AVX-512/AVX2 paths existed, ARM fell back to scalar loop
    // Impact: 8x slower on Apple Silicon for FP32 operations
    // Solution: Add NEON intrinsics for FP32 dot product
    #include <arm_neon.h>
    const float* fp32 = reinterpret_cast<const float*>(packed);
    float32x4_t acc0 = vdupq_n_f32(0.f);
    float32x4_t acc1 = vdupq_n_f32(0.f);
    uint32_t col = 0;
    for (; col + 7u < cols; col += 8u) {
        float32x4_t f0 = vld1q_f32(fp32 + col);
        float32x4_t f1 = vld1q_f32(fp32 + col + 4u);
        float32x4_t x0 = vld1q_f32(x + col);
        float32x4_t x1 = vld1q_f32(x + col + 4u);
        acc0 = vfmaq_f32(acc0, f0, x0);
        acc1 = vfmaq_f32(acc1, f1, x1);
    }
    acc0 = vaddq_f32(acc0, acc1);
    float acc = vaddvq_f32(acc0);
    for (; col < cols; ++col) {
        acc += fp32[col] * x[col];
    }
    return acc;
#else
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    uint32_t col = 0;
    for (; col + 3u < cols; col += 4u) {
        float v0, v1, v2, v3;
        std::memcpy(&v0, packed + (uint64_t)(col + 0u) * 4ull, sizeof(v0));
        std::memcpy(&v1, packed + (uint64_t)(col + 1u) * 4ull, sizeof(v1));
        std::memcpy(&v2, packed + (uint64_t)(col + 2u) * 4ull, sizeof(v2));
        std::memcpy(&v3, packed + (uint64_t)(col + 3u) * 4ull, sizeof(v3));
        a0 += v0 * x[col + 0u];
        a1 += v1 * x[col + 1u];
        a2 += v2 * x[col + 2u];
        a3 += v3 * x[col + 3u];
    }
    float acc = (a0 + a1) + (a2 + a3);
    for (; col < cols; ++col) {
        float v;
        std::memcpy(&v, packed + (uint64_t)col * 4ull, sizeof(v));
        acc += v * x[col];
    }
    return acc;
#endif
}

static inline float moe_tensor_bf16_to_f32(uint16_t v) {
    const uint32_t bits = (uint32_t)v << 16u;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

static float moe_dot_raw_bf16_row(const uint8_t* packed, const float* x, uint32_t cols) {
    if (!packed || !x) {
        return 0.0f;
    }
    float acc = 0.0f;
    for (uint32_t i = 0; i < cols; ++i) {
        uint16_t v = 0;
        std::memcpy(&v, packed + (uint64_t)i * 2ull, sizeof(v));
        acc += moe_tensor_bf16_to_f32(v) * x[i];
    }
    return acc;
}

static uint32_t moe_packed_lsb_code_at(const uint8_t* packed, uint32_t element_index, uint32_t bits) {
    const uint64_t bit_offset = (uint64_t)element_index * bits;
    const uint64_t byte_offset = bit_offset >> 3;
    const uint32_t shift = (uint32_t)(bit_offset & 7u);
    uint32_t window = packed[byte_offset];
    if (shift + bits > 8u) {
        window |= (uint32_t)packed[byte_offset + 1u] << 8u;
    }
    const uint32_t mask = (1u << bits) - 1u;
    return (window >> shift) & mask;
}

#if defined(__AVX2__)
static inline float moe_dot_hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

static inline __m256 moe_i8x8_to_ps(const int8_t* values) {
    const __m128i i8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(values));
    const __m256i i32 = _mm256_cvtepi8_epi32(i8);
    return _mm256_cvtepi32_ps(i32);
}
#endif

static inline uint16_t moe_load_u16le(const uint8_t* p) {
    uint16_t v = 0;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

static inline uint32_t moe_load_u32le(const uint8_t* p) {
    uint32_t v = 0;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

static inline float moe_load_f16le(const uint8_t* p) {
    return moe_raw_fp16_to_f32(moe_load_u16le(p));
}

static inline void moe_accum_or_store(
    float v,
    const float* x,
    float* out,
    uint32_t col,
    uint32_t cols,
    float* acc
) {
    if (col >= cols) {
        return;
    }
    if (out) {
        out[col] = v;
    }
    if (x && acc) {
        *acc += v * x[col];
    }
}

static inline void moe_get_scale_min_k4(uint32_t j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4u) {
        *d = q[j] & 63u;
        *m = q[j + 4u] & 63u;
    } else {
        *d = (q[j + 4u] & 0x0fu) | ((q[j - 4u] >> 6) << 4);
        *m = (q[j + 4u] >> 4) | ((q[j] >> 6) << 4);
    }
}
