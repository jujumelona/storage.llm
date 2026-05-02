static float moe_dot_raw_fp16_row(const uint8_t* packed, const float* x, uint32_t cols) {
#if defined(__AVX512F__) && defined(__F16C__)
    // BUGFIX 708: AVX-512 16-wide FP16 dot product ★★★
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
    // BUGFIX 문제9: ARM NEON SIMD for Apple Silicon / ARM64 ★★★
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
    // BUGFIX 708: AVX-512 16-wide FP32 dot product ★★★
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
    // BUGFIX 문제9: ARM NEON SIMD for Apple Silicon / ARM64 ★★★
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

static float moe_dot_gguf_q8_0_row(
    const uint8_t* packed,
    const float* x,
    uint32_t cols
) {
    if (!packed || !x) {
        return 0.0f;
    }
    float acc = 0.0f;
    const uint32_t block_cols = 32u;
    const uint32_t block_bytes = 34u;
    for (uint32_t start = 0; start < cols; start += block_cols) {
        uint16_t d16 = 0;
        std::memcpy(&d16, packed, sizeof(d16));
        const float d = moe_raw_fp16_to_f32(d16);
        const int8_t* q = reinterpret_cast<const int8_t*>(packed + 2u);
        const uint32_t limit = std::min<uint32_t>(block_cols, cols - start);
#if defined(__AVX2__)
        const __m256 vscale = _mm256_set1_ps(d);
        __m256 vacc = _mm256_setzero_ps();
        uint32_t i = 0;
        for (; i + 7u < limit; i += 8u) {
            const __m256 qv = moe_i8x8_to_ps(q + i);
            const __m256 xv = _mm256_loadu_ps(x + start + i);
            vacc = moe_madd_ps(_mm256_mul_ps(qv, vscale), xv, vacc);
        }
        acc += moe_dot_hsum256_ps(vacc);
        for (; i < limit; ++i) {
            acc += d * (float)q[i] * x[start + i];
        }
#else
        for (uint32_t i = 0; i < limit; ++i) {
            acc += d * (float)q[i] * x[start + i];
        }
#endif
        packed += block_bytes;
    }
    return acc;
}

static float moe_dot_gguf_q8_1_row(
    const uint8_t* packed,
    const float* x,
    uint32_t cols
) {
    if (!packed || !x) {
        return 0.0f;
    }
    float acc = 0.0f;
    const uint32_t block_cols = 32u;
    const uint32_t block_bytes = 36u;
    for (uint32_t start = 0; start < cols; start += block_cols) {
        uint16_t d16 = 0;
        std::memcpy(&d16, packed, sizeof(d16));
        const float d = moe_raw_fp16_to_f32(d16);
        const int8_t* q = reinterpret_cast<const int8_t*>(packed + 4u);
        const uint32_t limit = std::min<uint32_t>(block_cols, cols - start);
        for (uint32_t i = 0; i < limit; ++i) {
            acc += d * (float)q[i] * x[start + i];
        }
        packed += block_bytes;
    }
    return acc;
}

static float moe_dot_gguf_q4_0_row(
    const uint8_t* packed,
    const float* x,
    uint32_t cols
) {
    if (!packed || !x) {
        return 0.0f;
    }
    float acc = 0.0f;
    const uint32_t block_cols = 32u;
    const uint32_t block_bytes = 18u;
    for (uint32_t start = 0; start < cols; start += block_cols) {
        uint16_t d16 = 0;
        std::memcpy(&d16, packed, sizeof(d16));
        const float d = moe_raw_fp16_to_f32(d16);
        const uint8_t* qs = packed + 2u;
        const uint32_t limit = std::min<uint32_t>(block_cols, cols - start);
        for (uint32_t i = 0; i < limit; ++i) {
            const uint8_t byte = qs[i & 15u];
            const uint8_t q = (i < 16u) ? (byte & 0x0fu) : (byte >> 4);
            acc += d * ((float)((int)q - 8)) * x[start + i];
        }
        packed += block_bytes;
    }
    return acc;
}

static float moe_dot_gguf_q4_1_row(
    const uint8_t* packed,
    const float* x,
    uint32_t cols
) {
    if (!packed || !x) {
        return 0.0f;
    }
    float acc = 0.0f;
    const uint32_t block_cols = 32u;
    const uint32_t block_bytes = 20u;
    for (uint32_t start = 0; start < cols; start += block_cols) {
        uint16_t d16 = 0;
        uint16_t m16 = 0;
        std::memcpy(&d16, packed, sizeof(d16));
        std::memcpy(&m16, packed + 2u, sizeof(m16));
        const float d = moe_raw_fp16_to_f32(d16);
        const float m = moe_raw_fp16_to_f32(m16);
        const uint8_t* qs = packed + 4u;
        const uint32_t limit = std::min<uint32_t>(block_cols, cols - start);
        for (uint32_t i = 0; i < limit; ++i) {
            const uint8_t byte = qs[i & 15u];
            const uint8_t q = (i < 16u) ? (byte & 0x0fu) : (byte >> 4);
            acc += (d * (float)q + m) * x[start + i];
        }
        packed += block_bytes;
    }
    return acc;
}

static float moe_dot_gguf_q5_0_row(
    const uint8_t* packed,
    const float* x,
    uint32_t cols
) {
    if (!packed || !x) {
        return 0.0f;
    }
    float acc = 0.0f;
    const uint32_t block_cols = 32u;
    const uint32_t block_bytes = 22u;
    for (uint32_t start = 0; start < cols; start += block_cols) {
        uint16_t d16 = 0;
        uint32_t qh = 0;
        std::memcpy(&d16, packed, sizeof(d16));
        std::memcpy(&qh, packed + 2u, sizeof(qh));
        const float d = moe_raw_fp16_to_f32(d16);
        const uint8_t* qs = packed + 6u;
        const uint32_t limit = std::min<uint32_t>(block_cols, cols - start);
        for (uint32_t i = 0; i < limit; ++i) {
            const uint8_t byte = qs[i & 15u];
            const uint32_t low = (i < 16u) ? (byte & 0x0fu) : (byte >> 4);
            const uint32_t high = (qh >> i) & 1u;
            acc += d * ((float)((int)(low | (high << 4u)) - 16)) * x[start + i];
        }
        packed += block_bytes;
    }
    return acc;
}

static const int8_t* moe_iq4nl_value_table() {
    static constexpr int8_t kValues[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10,
           1,   13,  25,  38,  53,  69,  89, 113,
    };
    return kValues;
}

static float moe_gguf_iq4_nl_row_impl(const uint8_t* packed, const float* x, float* out, uint32_t cols) {
    float acc = 0.0f;
    const uint32_t block_cols = 32u;
    const uint32_t block_bytes = 18u;
    const int8_t* values = moe_iq4nl_value_table();
    for (uint32_t start = 0; start < cols; start += block_cols) {
        const float d = moe_load_f16le(packed);
        const uint8_t* qs = packed + 2u;
        const uint32_t limit = std::min<uint32_t>(block_cols, cols - start);
        for (uint32_t i = 0; i < limit; ++i) {
            const uint8_t byte = qs[i & 15u];
            const uint8_t code = (i < 16u) ? (byte & 0x0fu) : (byte >> 4);
            const float v = d * (float)values[code];
            moe_accum_or_store(v, x, out, start + i, cols, &acc);
        }
        packed += block_bytes;
    }
    return acc;
}

static float moe_dot_gguf_iq4_nl_row(const uint8_t* packed, const float* x, uint32_t cols) {
    return moe_gguf_iq4_nl_row_impl(packed, x, nullptr, cols);
}

static void moe_dequant_gguf_iq4_nl_row(const uint8_t* packed, float* out, uint32_t cols) {
    (void)moe_gguf_iq4_nl_row_impl(packed, nullptr, out, cols);
}

static float moe_gguf_iq4_xs_row_impl(const uint8_t* packed, const float* x, float* out, uint32_t cols) {
    float acc = 0.0f;
    const uint32_t block_cols = 256u;
    const uint32_t block_bytes = 136u;
    const int8_t* values = moe_iq4nl_value_table();
    for (uint32_t base = 0; base < cols; base += block_cols) {
        const float d = moe_load_f16le(packed);
        const uint16_t scales_h = moe_load_u16le(packed + 2u);
        const uint8_t* scales_l = packed + 4u;
        const uint8_t* qs = packed + 8u;
        const uint32_t limit = std::min<uint32_t>(block_cols, cols - base);
        for (uint32_t ib = 0; ib < 8u && ib * 32u < limit; ++ib) {
            const uint32_t scale_l = (scales_l[ib / 2u] >> (4u * (ib & 1u))) & 0x0fu;
            const uint32_t scale_h = (scales_h >> (2u * ib)) & 0x03u;
            const float dl = d * (float)((int)(scale_l | (scale_h << 4u)) - 32);
            const uint8_t* block_qs = qs + ib * 16u;
            const uint32_t local_limit = std::min<uint32_t>(32u, limit - ib * 32u);
            for (uint32_t j = 0; j < local_limit; ++j) {
                const uint8_t byte = block_qs[j & 15u];
                const uint8_t code = (j < 16u) ? (byte & 0x0fu) : (byte >> 4);
                const float v = dl * (float)values[code];
                moe_accum_or_store(v, x, out, base + ib * 32u + j, cols, &acc);
            }
        }
        packed += block_bytes;
    }
    return acc;
}

static float moe_dot_gguf_iq4_xs_row(const uint8_t* packed, const float* x, uint32_t cols) {
    return moe_gguf_iq4_xs_row_impl(packed, x, nullptr, cols);
}

static void moe_dequant_gguf_iq4_xs_row(const uint8_t* packed, float* out, uint32_t cols) {
    (void)moe_gguf_iq4_xs_row_impl(packed, nullptr, out, cols);
}
static float moe_e8m0_to_f32_half(uint8_t v) {
    uint32_t bits = v ? ((uint32_t)v << 23u) : 0x00400000u;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

static const float* moe_mxfp4_value_table() {
    static constexpr float kMxfp4Table[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    return kMxfp4Table;
}

static float moe_mxfp4_value_from_code(uint8_t code) {
    return moe_mxfp4_value_table()[code & 0x0fu];
}

static uint8_t moe_mxfp4_code_at(const uint8_t* qs, uint32_t index) {
    // BUGFIX 217: Fix incorrect index masking for mxfp4 block access
    // Problem: index & 15u masks to 0-15, but mxfp4 blocks have 32 elements (16 bytes)
    //          This causes reading wrong byte for indices 16-31
    // Solution: Use index >> 1 for byte offset, index & 1 for nibble selection
    // Impact: Fixes data corruption in mxfp4 quantized weights
    const uint8_t byte = qs[index >> 1];
    return (index & 1u) ? (byte >> 4) : (byte & 0x0fu);
}

static float moe_dot_gguf_mxfp4_row(
    const uint8_t* packed,
    const float* x,
    uint32_t cols
) {
    if (!packed || !x) {
        return 0.0f;
    }
    float acc = 0.0f;
    const uint32_t block_cols = 32u;
    const uint32_t block_bytes = 17u;
    const float* table = moe_mxfp4_value_table();
    uint32_t start = 0;
    const uint32_t full_cols = cols & ~(block_cols - 1u);
    for (; start < full_cols; start += block_cols) {
        const float scale = moe_e8m0_to_f32_half(packed[0]);
        const uint8_t* qs = packed + 1u;
        const float* xb = x + start;
        float even = 0.0f;
        float odd = 0.0f;
#if defined(__AVX2__) && defined(__FMA__)
        // BUGFIX 705: AVX2 vectorized MXFP4 dot product ★★★
        // 16-entry FP32 table (64B) always in L1. Gather throughput ~5cy on Haswell+.
        const __m256i lo_mask = _mm256_set1_epi32(0x0F);
        __m256 vacc_even = _mm256_setzero_ps();
        __m256 vacc_odd  = _mm256_setzero_ps();
        for (uint32_t j = 0; j < 16u; j += 8u) {
            __m128i raw8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qs + j));
            __m256i bytes8 = _mm256_cvtepu8_epi32(raw8);
            __m256i lo_idx = _mm256_and_si256(bytes8, lo_mask);
            __m256 lo_vals = _mm256_i32gather_ps(table, lo_idx, 4);
            __m256i hi_idx = _mm256_srli_epi32(bytes8, 4);
            __m256 hi_vals = _mm256_i32gather_ps(table, hi_idx, 4);
            const __m256i eidx = _mm256_set_epi32(
                (int)(j*2u+14u),(int)(j*2u+12u),(int)(j*2u+10u),(int)(j*2u+8u),
                (int)(j*2u+6u),(int)(j*2u+4u),(int)(j*2u+2u),(int)(j*2u+0u));
            const __m256i oidx = _mm256_set_epi32(
                (int)(j*2u+15u),(int)(j*2u+13u),(int)(j*2u+11u),(int)(j*2u+9u),
                (int)(j*2u+7u),(int)(j*2u+5u),(int)(j*2u+3u),(int)(j*2u+1u));
            vacc_even = _mm256_fmadd_ps(lo_vals, _mm256_i32gather_ps(xb, eidx, 4), vacc_even);
            vacc_odd  = _mm256_fmadd_ps(hi_vals, _mm256_i32gather_ps(xb, oidx, 4), vacc_odd);
        }
        even = moe_dot_hsum256_ps(vacc_even);
        odd  = moe_dot_hsum256_ps(vacc_odd);
#else
        for (uint32_t j = 0; j < 16u; ++j) {
            const uint8_t byte = qs[j];
            even += table[byte & 0x0fu] * xb[j * 2u];
            odd += table[byte >> 4] * xb[j * 2u + 1u];
        }
#endif
        acc += (even + odd) * scale;
        packed += block_bytes;
    }
    if (start < cols) {
        const float scale = moe_e8m0_to_f32_half(packed[0]);
        const uint8_t* qs = packed + 1u;
        const uint32_t limit = cols - start;
        float block_acc = 0.0f;
        uint32_t i = 0;
        for (; i + 1u < limit; i += 2u) {
            const uint8_t byte = qs[i >> 1];
            block_acc += table[byte & 0x0fu] * x[start + i];
            block_acc += table[byte >> 4] * x[start + i + 1u];
        }
        if (i < limit) {
            block_acc += table[qs[i >> 1] & 0x0fu] * x[start + i];
        }
        acc += block_acc * scale;
        packed += block_bytes;
    }
    return acc;
}

static float moe_dot_gguf_q5_1_row(
    const uint8_t* packed,
    const float* x,
    uint32_t cols
) {
    if (!packed || !x) {
        return 0.0f;
    }
    float acc = 0.0f;
    const uint32_t block_cols = 32u;
    const uint32_t block_bytes = 24u;
    for (uint32_t start = 0; start < cols; start += block_cols) {
        uint16_t d16 = 0;
        uint16_t m16 = 0;
        uint32_t qh = 0;
        std::memcpy(&d16, packed, sizeof(d16));
        std::memcpy(&m16, packed + 2u, sizeof(m16));
        std::memcpy(&qh, packed + 4u, sizeof(qh));
        const float d = moe_raw_fp16_to_f32(d16);
        const float m = moe_raw_fp16_to_f32(m16);
        const uint8_t* qs = packed + 8u;
        const uint32_t limit = std::min<uint32_t>(block_cols, cols - start);
        for (uint32_t i = 0; i < limit; ++i) {
            const uint8_t byte = qs[i & 15u];
            const uint32_t low = (i < 16u) ? (byte & 0x0fu) : (byte >> 4);
            const uint32_t high = (qh >> i) & 1u;
            const float w = d * (float)(low | (high << 4u)) + m;
            acc += w * x[start + i];
        }
        packed += block_bytes;
    }
    return acc;
}

static float moe_gguf_q2_k_row_impl(const uint8_t* packed, const float* x, float* out, uint32_t cols) {
    float acc = 0.0f;
    for (uint32_t base = 0; base < cols; base += 256u, packed += 84u) {
        const uint8_t* scales = packed;
        const uint8_t* q = packed + 16u;
        const float d = moe_load_f16le(packed + 80u);
        const float minv = moe_load_f16le(packed + 82u);
        uint32_t is = 0;
        for (uint32_t n = 0; n < 256u; n += 128u) {
            uint32_t shift = 0;
            for (uint32_t j = 0; j < 4u; ++j) {
                uint8_t sc = scales[is++];
                const float dl = d * (float)(sc & 0x0fu);
                const float ml = minv * (float)(sc >> 4);
                for (uint32_t l = 0; l < 16u; ++l) {
                    const float v = dl * (float)((q[l] >> shift) & 3u) - ml;
                    moe_accum_or_store(v, x, out, base + n + j * 32u + l, cols, &acc);
                }
                sc = scales[is++];
                const float dl2 = d * (float)(sc & 0x0fu);
                const float ml2 = minv * (float)(sc >> 4);
                for (uint32_t l = 0; l < 16u; ++l) {
                    const float v = dl2 * (float)((q[l + 16u] >> shift) & 3u) - ml2;
                    moe_accum_or_store(v, x, out, base + n + j * 32u + 16u + l, cols, &acc);
                }
                shift += 2u;
            }
            q += 32u;
        }
    }
    return acc;
}

static float moe_gguf_q3_k_row_impl(const uint8_t* packed, const float* x, float* out, uint32_t cols) {
    float acc = 0.0f;
    const uint32_t kmask1 = 0x03030303u;
    const uint32_t kmask2 = 0x0f0f0f0fu;
    for (uint32_t base = 0; base < cols; base += 256u, packed += 110u) {
        const uint8_t* hm = packed;
        const uint8_t* q = packed + 32u;
        const uint8_t* scale_bytes = packed + 96u;
        const float d_all = moe_load_f16le(packed + 108u);
        uint32_t aux[4]{};
        std::memcpy(aux, scale_bytes, 12u);
        const uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t* scales = reinterpret_cast<const int8_t*>(aux);
        uint32_t is = 0;
        uint8_t m = 1;
        for (uint32_t n = 0; n < 256u; n += 128u) {
            uint32_t shift = 0;
            for (uint32_t j = 0; j < 4u; ++j) {
                float dl = d_all * (float)(scales[is++] - 32);
                for (uint32_t l = 0; l < 16u; ++l) {
                    const int32_t qv = (int32_t)((q[l] >> shift) & 3u) - ((hm[l] & m) ? 0 : 4);
                    moe_accum_or_store(dl * (float)qv, x, out, base + n + j * 32u + l, cols, &acc);
                }
                dl = d_all * (float)(scales[is++] - 32);
                for (uint32_t l = 0; l < 16u; ++l) {
                    const int32_t qv = (int32_t)((q[l + 16u] >> shift) & 3u) - ((hm[l + 16u] & m) ? 0 : 4);
                    moe_accum_or_store(dl * (float)qv, x, out, base + n + j * 32u + 16u + l, cols, &acc);
                }
                shift += 2u;
                m = (uint8_t)(m << 1);
            }
            q += 32u;
        }
    }
    return acc;
}

static float moe_gguf_q4_k_row_impl(const uint8_t* packed, const float* x, float* out, uint32_t cols) {
    float acc = 0.0f;
    for (uint32_t base = 0; base < cols; base += 256u, packed += 144u) {
        const float d = moe_load_f16le(packed);
        const float minv = moe_load_f16le(packed + 2u);
        const uint8_t* scales = packed + 4u;
        const uint8_t* q = packed + 16u;
        uint32_t is = 0;
        for (uint32_t j = 0; j < 256u; j += 64u) {
            uint8_t sc = 0;
            uint8_t mn = 0;
            moe_get_scale_min_k4(is + 0u, scales, &sc, &mn);
            const float d1 = d * (float)sc;
            const float m1 = minv * (float)mn;
            moe_get_scale_min_k4(is + 1u, scales, &sc, &mn);
            const float d2 = d * (float)sc;
            const float m2 = minv * (float)mn;
            for (uint32_t l = 0; l < 32u; ++l) {
                moe_accum_or_store(d1 * (float)(q[l] & 0x0fu) - m1, x, out, base + j + l, cols, &acc);
            }
            for (uint32_t l = 0; l < 32u; ++l) {
                moe_accum_or_store(d2 * (float)(q[l] >> 4) - m2, x, out, base + j + 32u + l, cols, &acc);
            }
            q += 32u;
            is += 2u;
        }
    }
    return acc;
}

static float moe_gguf_q5_k_row_impl(const uint8_t* packed, const float* x, float* out, uint32_t cols) {
    float acc = 0.0f;
    for (uint32_t base = 0; base < cols; base += 256u, packed += 176u) {
        const float d = moe_load_f16le(packed);
        const float minv = moe_load_f16le(packed + 2u);
        const uint8_t* scales = packed + 4u;
        const uint8_t* qh = packed + 16u;
        const uint8_t* ql = packed + 48u;
        uint32_t is = 0;
        uint8_t u1 = 1;
        uint8_t u2 = 2;
        for (uint32_t j = 0; j < 256u; j += 64u) {
            uint8_t sc = 0;
            uint8_t mn = 0;
            moe_get_scale_min_k4(is + 0u, scales, &sc, &mn);
            const float d1 = d * (float)sc;
            const float m1 = minv * (float)mn;
            moe_get_scale_min_k4(is + 1u, scales, &sc, &mn);
            const float d2 = d * (float)sc;
            const float m2 = minv * (float)mn;
            for (uint32_t l = 0; l < 32u; ++l) {
                const uint32_t qv = (uint32_t)(ql[l] & 0x0fu) + ((qh[l] & u1) ? 16u : 0u);
                moe_accum_or_store(d1 * (float)qv - m1, x, out, base + j + l, cols, &acc);
            }
            for (uint32_t l = 0; l < 32u; ++l) {
                const uint32_t qv = (uint32_t)(ql[l] >> 4) + ((qh[l] & u2) ? 16u : 0u);
                moe_accum_or_store(d2 * (float)qv - m2, x, out, base + j + 32u + l, cols, &acc);
            }
            ql += 32u;
            is += 2u;
            u1 = (uint8_t)(u1 << 2);
            u2 = (uint8_t)(u2 << 2);
        }
    }
    return acc;
}

static float moe_gguf_q6_k_row_impl(const uint8_t* packed, const float* x, float* out, uint32_t cols) {
    float acc = 0.0f;
    for (uint32_t base = 0; base < cols; base += 256u, packed += 210u) {
        const uint8_t* ql = packed;
        const uint8_t* qh = packed + 128u;
        const int8_t* sc = reinterpret_cast<const int8_t*>(packed + 192u);
        const float d = moe_load_f16le(packed + 208u);
        for (uint32_t n = 0; n < 256u; n += 128u) {
            for (uint32_t l = 0; l < 32u; ++l) {
                const uint32_t is = l / 16u;
                const int32_t q1 = (int32_t)((ql[l] & 0x0fu) | (((qh[l] >> 0) & 3u) << 4)) - 32;
                const int32_t q2 = (int32_t)((ql[l + 32u] & 0x0fu) | (((qh[l] >> 2) & 3u) << 4)) - 32;
                const int32_t q3 = (int32_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3u) << 4)) - 32;
                const int32_t q4 = (int32_t)((ql[l + 32u] >> 4) | (((qh[l] >> 6) & 3u) << 4)) - 32;
                moe_accum_or_store(d * (float)sc[is + 0u] * (float)q1, x, out, base + n + l, cols, &acc);
                moe_accum_or_store(d * (float)sc[is + 2u] * (float)q2, x, out, base + n + 32u + l, cols, &acc);
                moe_accum_or_store(d * (float)sc[is + 4u] * (float)q3, x, out, base + n + 64u + l, cols, &acc);
                moe_accum_or_store(d * (float)sc[is + 6u] * (float)q4, x, out, base + n + 96u + l, cols, &acc);
            }
            ql += 64u;
            qh += 32u;
            sc += 8u;
        }
    }
    return acc;
}

static float moe_gguf_iq2_xxs_row_impl(const uint8_t* packed, const float* x, float* out, uint32_t cols) {
    float acc = 0.0f;
    for (uint32_t base = 0; base < cols; base += 256u, packed += 66u) {
        const float d = moe_load_f16le(packed);
        const uint8_t* qs = packed + 2u;
        for (uint32_t ib32 = 0; ib32 < 8u; ++ib32) {
            uint32_t aux32[2] = {
                moe_load_u32le(qs + 8u * ib32),
                moe_load_u32le(qs + 8u * ib32 + 4u)
            };
            const uint8_t* aux8 = reinterpret_cast<const uint8_t*>(aux32);
            const float db = d * (0.5f + (float)(aux32[1] >> 28)) * 0.25f;
            for (uint32_t l = 0; l < 4u; ++l) {
                const uint8_t* grid = reinterpret_cast<const uint8_t*>(&moe_iq2xxs_grid[aux8[l]]);
                const uint8_t signs = moe_iq2xs_signs[(aux32[1] >> (7u * l)) & 127u];
                for (uint32_t j = 0; j < 8u; ++j) {
                    const float sign = (signs & moe_iq2xs_mask[j]) ? -1.0f : 1.0f;
                    moe_accum_or_store(db * (float)grid[j] * sign, x, out, base + ib32 * 32u + l * 8u + j, cols, &acc);
                }
            }
        }
    }
    return acc;
}

static float moe_gguf_iq3_xxs_row_impl(const uint8_t* packed, const float* x, float* out, uint32_t cols) {
    float acc = 0.0f;
    for (uint32_t base = 0; base < cols; base += 256u, packed += 98u) {
        const float d = moe_load_f16le(packed);
        const uint8_t* qs = packed + 2u;
        const uint8_t* scales_and_signs = qs + 64u;
        for (uint32_t ib32 = 0; ib32 < 8u; ++ib32) {
            const uint32_t aux32 = moe_load_u32le(scales_and_signs + 4u * ib32);
            const uint8_t* q = qs + 8u * ib32;
            const float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
            for (uint32_t l = 0; l < 4u; ++l) {
                const uint8_t signs = moe_iq2xs_signs[(aux32 >> (7u * l)) & 127u];
                const uint8_t* grid1 = reinterpret_cast<const uint8_t*>(&moe_iq3xxs_grid[q[2u * l + 0u]]);
                const uint8_t* grid2 = reinterpret_cast<const uint8_t*>(&moe_iq3xxs_grid[q[2u * l + 1u]]);
                for (uint32_t j = 0; j < 4u; ++j) {
                    const float sign1 = (signs & moe_iq2xs_mask[j]) ? -1.0f : 1.0f;
                    const float sign2 = (signs & moe_iq2xs_mask[j + 4u]) ? -1.0f : 1.0f;
                    moe_accum_or_store(db * (float)grid1[j] * sign1, x, out, base + ib32 * 32u + l * 8u + j, cols, &acc);
                    moe_accum_or_store(db * (float)grid2[j] * sign2, x, out, base + ib32 * 32u + l * 8u + 4u + j, cols, &acc);
                }
            }
        }
    }
    return acc;
}

static float moe_gguf_iq2_xs_row_impl(const uint8_t* packed, const float* x, float* out, uint32_t cols) {
    float acc = 0.0f;
    for (uint32_t base = 0; base < cols; base += 256u, packed += 74u) {
        const float d = moe_load_f16le(packed);
        const uint8_t* qs = packed + 2u;
        const uint8_t* scales = packed + 66u;
        for (uint32_t ib32 = 0; ib32 < 8u; ++ib32) {
            const float db0 = d * (0.5f + (float)(scales[ib32] & 0x0fu)) * 0.25f;
            const float db1 = d * (0.5f + (float)(scales[ib32] >> 4)) * 0.25f;
            for (uint32_t l = 0; l < 4u; ++l) {
                const uint16_t qv = moe_load_u16le(qs + (4u * ib32 + l) * 2u);
                const uint8_t* grid = reinterpret_cast<const uint8_t*>(&moe_iq2xs_grid[qv & 511u]);
                const uint8_t signs = moe_iq2xs_signs[qv >> 9];
                const float db = (l < 2u) ? db0 : db1;
                for (uint32_t j = 0; j < 8u; ++j) {
                    const float sign = (signs & moe_iq2xs_mask[j]) ? -1.0f : 1.0f;
                    moe_accum_or_store(db * (float)grid[j] * sign, x, out, base + ib32 * 32u + l * 8u + j, cols, &acc);
                }
            }
        }
    }
    return acc;
}

static float moe_gguf_iq2_s_row_impl(const uint8_t* packed, const float* x, float* out, uint32_t cols) {
    float acc = 0.0f;
    for (uint32_t base = 0; base < cols; base += 256u, packed += 82u) {
        const float d = moe_load_f16le(packed);
        const uint8_t* qs = packed + 2u;
        const uint8_t* qh = packed + 66u;
        const uint8_t* scales = packed + 74u;
        const uint8_t* signs = qs + 32u;
        for (uint32_t ib32 = 0; ib32 < 8u; ++ib32) {
            const float db0 = d * (0.5f + (float)(scales[ib32] & 0x0fu)) * 0.25f;
            const float db1 = d * (0.5f + (float)(scales[ib32] >> 4)) * 0.25f;
            for (uint32_t l = 0; l < 4u; ++l) {
                const uint32_t grid_index = (uint32_t)qs[l] | (((uint32_t)qh[ib32] << (8u - 2u * l)) & 0x300u);
                const uint8_t* grid = reinterpret_cast<const uint8_t*>(&moe_iq2s_grid[grid_index]);
                const float db = (l < 2u) ? db0 : db1;
                for (uint32_t j = 0; j < 8u; ++j) {
                    const float sign = (signs[l] & moe_iq2xs_mask[j]) ? -1.0f : 1.0f;
                    moe_accum_or_store(db * (float)grid[j] * sign, x, out, base + ib32 * 32u + l * 8u + j, cols, &acc);
                }
            }
            qs += 4u;
            signs += 4u;
        }
    }
    return acc;
}

static float moe_gguf_iq3_s_row_impl(const uint8_t* packed, const float* x, float* out, uint32_t cols) {
    float acc = 0.0f;
    for (uint32_t base = 0; base < cols; base += 256u, packed += 110u) {
        const float d = moe_load_f16le(packed);
        const uint8_t* qs = packed + 2u;
        const uint8_t* qh = packed + 66u;
        const uint8_t* signs = packed + 74u;
        const uint8_t* scales = packed + 106u;
        for (uint32_t ib32 = 0; ib32 < 8u; ib32 += 2u) {
            const float db1 = d * (float)(1u + 2u * (uint32_t)(scales[ib32 / 2u] & 0x0fu));
            const float db2 = d * (float)(1u + 2u * (uint32_t)(scales[ib32 / 2u] >> 4));
            for (uint32_t l = 0; l < 4u; ++l) {
                const uint8_t* grid1 = reinterpret_cast<const uint8_t*>(&moe_iq3s_grid[(uint32_t)qs[2u * l + 0u] | (((uint32_t)qh[0] << (8u - 2u * l)) & 256u)]);
                const uint8_t* grid2 = reinterpret_cast<const uint8_t*>(&moe_iq3s_grid[(uint32_t)qs[2u * l + 1u] | (((uint32_t)qh[0] << (7u - 2u * l)) & 256u)]);
                for (uint32_t j = 0; j < 4u; ++j) {
                    const float sign1 = (signs[l] & moe_iq2xs_mask[j]) ? -1.0f : 1.0f;
                    const float sign2 = (signs[l] & moe_iq2xs_mask[j + 4u]) ? -1.0f : 1.0f;
                    moe_accum_or_store(db1 * (float)grid1[j] * sign1, x, out, base + ib32 * 32u + l * 8u + j, cols, &acc);
                    moe_accum_or_store(db1 * (float)grid2[j] * sign2, x, out, base + ib32 * 32u + l * 8u + 4u + j, cols, &acc);
                }
            }
            qs += 8u;
            signs += 4u;
            for (uint32_t l = 0; l < 4u; ++l) {
                const uint8_t* grid1 = reinterpret_cast<const uint8_t*>(&moe_iq3s_grid[(uint32_t)qs[2u * l + 0u] | (((uint32_t)qh[1] << (8u - 2u * l)) & 256u)]);
                const uint8_t* grid2 = reinterpret_cast<const uint8_t*>(&moe_iq3s_grid[(uint32_t)qs[2u * l + 1u] | (((uint32_t)qh[1] << (7u - 2u * l)) & 256u)]);
                for (uint32_t j = 0; j < 4u; ++j) {
                    const float sign1 = (signs[l] & moe_iq2xs_mask[j]) ? -1.0f : 1.0f;
                    const float sign2 = (signs[l] & moe_iq2xs_mask[j + 4u]) ? -1.0f : 1.0f;
                    moe_accum_or_store(db2 * (float)grid1[j] * sign1, x, out, base + (ib32 + 1u) * 32u + l * 8u + j, cols, &acc);
                    moe_accum_or_store(db2 * (float)grid2[j] * sign2, x, out, base + (ib32 + 1u) * 32u + l * 8u + 4u + j, cols, &acc);
                }
            }
            qs += 8u;
            signs += 4u;
            qh += 2u;
        }
    }
    return acc;
}

static float moe_gguf_iq1_s_row_impl(const uint8_t* packed, const float* x, float* out, uint32_t cols) {
    float acc = 0.0f;
    constexpr float delta_value = 0.125f;
    for (uint32_t base = 0; base < cols; base += 256u, packed += 50u) {
        const float d = moe_load_f16le(packed);
        const uint8_t* qs = packed + 2u;
        const uint8_t* qh = packed + 34u;
        for (uint32_t ib = 0; ib < 8u; ++ib) {
            const uint16_t h = moe_load_u16le(qh + ib * 2u);
            const float dl = d * (float)(2u * ((h >> 12) & 7u) + 1u);
            const float delta = (h & 0x8000u) ? -delta_value : delta_value;
            for (uint32_t l = 0; l < 4u; ++l) {
                const uint32_t idx = (uint32_t)qs[l] | (((uint32_t)(h >> (3u * l)) & 7u) << 8u);
                const int8_t* grid = reinterpret_cast<const int8_t*>(&moe_iq1s_grid[idx]);
                for (uint32_t j = 0; j < 8u; ++j) {
                    moe_accum_or_store(dl * ((float)grid[j] + delta), x, out, base + ib * 32u + l * 8u + j, cols, &acc);
                }
            }
            qs += 4u;
        }
    }
    return acc;
}

static float moe_gguf_iq1_m_row_impl(const uint8_t* packed, const float* x, float* out, uint32_t cols) {
    float acc = 0.0f;
    constexpr float delta_value = 0.125f;
    for (uint32_t base = 0; base < cols; base += 256u, packed += 56u) {
        const uint8_t* qs = packed;
        const uint8_t* qh = packed + 32u;
        const uint8_t* scales = packed + 48u;
        const uint16_t scale_u16 =
            (uint16_t)((moe_load_u16le(scales + 0u) >> 12) |
                       ((moe_load_u16le(scales + 2u) >> 8) & 0x00f0u) |
                       ((moe_load_u16le(scales + 4u) >> 4) & 0x0f00u) |
                       (moe_load_u16le(scales + 6u) & 0xf000u));
        const float d = moe_raw_fp16_to_f32(scale_u16);
        for (uint32_t ib = 0; ib < 8u; ++ib) {
            const uint16_t s = moe_load_u16le(scales + (ib / 2u) * 2u);
            const float dl1 = d * (float)(2u * ((s >> (6u * (ib & 1u) + 0u)) & 0x7u) + 1u);
            const float dl2 = d * (float)(2u * ((s >> (6u * (ib & 1u) + 3u)) & 0x7u) + 1u);
            const uint32_t idx0 = (uint32_t)qs[0] | (((uint32_t)qh[0] << 8u) & 0x700u);
            const uint32_t idx1 = (uint32_t)qs[1] | (((uint32_t)qh[0] << 4u) & 0x700u);
            const uint32_t idx2 = (uint32_t)qs[2] | (((uint32_t)qh[1] << 8u) & 0x700u);
            const uint32_t idx3 = (uint32_t)qs[3] | (((uint32_t)qh[1] << 4u) & 0x700u);
            const float delta0 = (qh[0] & 0x08u) ? -delta_value : delta_value;
            const float delta1 = (qh[0] & 0x80u) ? -delta_value : delta_value;
            const float delta2 = (qh[1] & 0x08u) ? -delta_value : delta_value;
            const float delta3 = (qh[1] & 0x80u) ? -delta_value : delta_value;
            const int8_t* grids[4] = {
                reinterpret_cast<const int8_t*>(&moe_iq1s_grid[idx0]),
                reinterpret_cast<const int8_t*>(&moe_iq1s_grid[idx1]),
                reinterpret_cast<const int8_t*>(&moe_iq1s_grid[idx2]),
                reinterpret_cast<const int8_t*>(&moe_iq1s_grid[idx3])
            };
            const float dls[4] = { dl1, dl1, dl2, dl2 };
            const float deltas[4] = { delta0, delta1, delta2, delta3 };
            for (uint32_t l = 0; l < 4u; ++l) {
                for (uint32_t j = 0; j < 8u; ++j) {
                    moe_accum_or_store(dls[l] * ((float)grids[l][j] + deltas[l]), x, out, base + ib * 32u + l * 8u + j, cols, &acc);
                }
            }
            qs += 4u;
            qh += 2u;
        }
    }
    return acc;
}

static float moe_dot_gguf_q2_k_row(const uint8_t* packed, const float* x, uint32_t cols) {
    return moe_gguf_q2_k_row_impl(packed, x, nullptr, cols);
}

static float moe_dot_gguf_q3_k_row(const uint8_t* packed, const float* x, uint32_t cols) {
    return moe_gguf_q3_k_row_impl(packed, x, nullptr, cols);
}

static float moe_dot_gguf_q4_k_row(const uint8_t* packed, const float* x, uint32_t cols) {
    return moe_gguf_q4_k_row_impl(packed, x, nullptr, cols);
}

static float moe_dot_gguf_q5_k_row(const uint8_t* packed, const float* x, uint32_t cols) {
    return moe_gguf_q5_k_row_impl(packed, x, nullptr, cols);
}

static float moe_dot_gguf_q6_k_row(const uint8_t* packed, const float* x, uint32_t cols) {
    return moe_gguf_q6_k_row_impl(packed, x, nullptr, cols);
}

static float moe_dot_gguf_iq2_xxs_row(const uint8_t* packed, const float* x, uint32_t cols) {
    return moe_gguf_iq2_xxs_row_impl(packed, x, nullptr, cols);
}

static float moe_dot_gguf_iq2_xs_row(const uint8_t* packed, const float* x, uint32_t cols) {
    return moe_gguf_iq2_xs_row_impl(packed, x, nullptr, cols);
}

static float moe_dot_gguf_iq2_s_row(const uint8_t* packed, const float* x, uint32_t cols) {
    return moe_gguf_iq2_s_row_impl(packed, x, nullptr, cols);
}

static float moe_dot_gguf_iq3_xxs_row(const uint8_t* packed, const float* x, uint32_t cols) {
    return moe_gguf_iq3_xxs_row_impl(packed, x, nullptr, cols);
}

static float moe_dot_gguf_iq3_s_row(const uint8_t* packed, const float* x, uint32_t cols) {
    return moe_gguf_iq3_s_row_impl(packed, x, nullptr, cols);
}

static float moe_dot_gguf_iq1_s_row(const uint8_t* packed, const float* x, uint32_t cols) {
    return moe_gguf_iq1_s_row_impl(packed, x, nullptr, cols);
}

static float moe_dot_gguf_iq1_m_row(const uint8_t* packed, const float* x, uint32_t cols) {
    return moe_gguf_iq1_m_row_impl(packed, x, nullptr, cols);
}

static void moe_dequant_gguf_q2_k_row(const uint8_t* packed, float* out, uint32_t cols) {
    (void)moe_gguf_q2_k_row_impl(packed, nullptr, out, cols);
}

static void moe_dequant_gguf_q3_k_row(const uint8_t* packed, float* out, uint32_t cols) {
    (void)moe_gguf_q3_k_row_impl(packed, nullptr, out, cols);
}

static void moe_dequant_gguf_q4_k_row(const uint8_t* packed, float* out, uint32_t cols) {
    (void)moe_gguf_q4_k_row_impl(packed, nullptr, out, cols);
}

static void moe_dequant_gguf_q5_k_row(const uint8_t* packed, float* out, uint32_t cols) {
    (void)moe_gguf_q5_k_row_impl(packed, nullptr, out, cols);
}

static void moe_dequant_gguf_q6_k_row(const uint8_t* packed, float* out, uint32_t cols) {
    (void)moe_gguf_q6_k_row_impl(packed, nullptr, out, cols);
}

static void moe_dequant_gguf_iq2_xxs_row(const uint8_t* packed, float* out, uint32_t cols) {
    (void)moe_gguf_iq2_xxs_row_impl(packed, nullptr, out, cols);
}

static void moe_dequant_gguf_iq2_xs_row(const uint8_t* packed, float* out, uint32_t cols) {
    (void)moe_gguf_iq2_xs_row_impl(packed, nullptr, out, cols);
}

static void moe_dequant_gguf_iq2_s_row(const uint8_t* packed, float* out, uint32_t cols) {
    (void)moe_gguf_iq2_s_row_impl(packed, nullptr, out, cols);
}

static void moe_dequant_gguf_iq3_xxs_row(const uint8_t* packed, float* out, uint32_t cols) {
    (void)moe_gguf_iq3_xxs_row_impl(packed, nullptr, out, cols);
}

static void moe_dequant_gguf_iq3_s_row(const uint8_t* packed, float* out, uint32_t cols) {
    (void)moe_gguf_iq3_s_row_impl(packed, nullptr, out, cols);
}

static void moe_dequant_gguf_iq1_s_row(const uint8_t* packed, float* out, uint32_t cols) {
    (void)moe_gguf_iq1_s_row_impl(packed, nullptr, out, cols);
}

static void moe_dequant_gguf_iq1_m_row(const uint8_t* packed, float* out, uint32_t cols) {
    (void)moe_gguf_iq1_m_row_impl(packed, nullptr, out, cols);
}

static float moe_dot_linear_q8_group(
    const uint8_t* packed,
    const float* x,
    float scale,
    uint32_t start,
    uint32_t limit
) {
    float acc = 0.0f;
    uint32_t col = start;
#if defined(__AVX2__)
    const __m256 vscale = _mm256_set1_ps(scale);
    __m256 vacc = _mm256_setzero_ps();
    for (; col + 7u < limit; col += 8u) {
        const __m256 qv = moe_i8x8_to_ps(reinterpret_cast<const int8_t*>(packed + col));
        vacc = moe_madd_ps(_mm256_mul_ps(qv, vscale), _mm256_loadu_ps(x + col), vacc);
    }
    acc += moe_dot_hsum256_ps(vacc);
#endif
    for (; col < limit; ++col) {
        int8_t q = 0;
        std::memcpy(&q, packed + col, sizeof(q));
        acc += (float)q * scale * x[col];
    }
    return acc;
}

static float moe_dot_linear_packed_group(
    const uint8_t* packed,
    const float* x,
    float scale,
    uint32_t bits,
    uint32_t start,
    uint32_t limit
) {
    const float midpoint = ((float)((1u << bits) - 1u)) * 0.5f;
    float a0 = 0.0f;
    float a1 = 0.0f;
    float a2 = 0.0f;
    float a3 = 0.0f;
    uint32_t col = start;
    if (bits == 4u && (col & 1u) == 0) {
        for (; col + 1u < limit; col += 2u) {
            const uint8_t byte = packed[col >> 1];
            a0 += ((float)(byte & 0x0fu) - midpoint) * x[col];
            a1 += ((float)(byte >> 4) - midpoint) * x[col + 1u];
        }
    } else if (bits == 2u && (col & 3u) == 0) {
        for (; col + 3u < limit; col += 4u) {
            const uint8_t byte = packed[col >> 2];
            a0 += ((float)(byte & 0x03u) - midpoint) * x[col];
            a1 += ((float)((byte >> 2) & 0x03u) - midpoint) * x[col + 1u];
            a2 += ((float)((byte >> 4) & 0x03u) - midpoint) * x[col + 2u];
            a3 += ((float)(byte >> 6) - midpoint) * x[col + 3u];
        }
    }
    for (; col + 3u < limit; col += 4u) {
        a0 += ((float)moe_packed_lsb_code_at(packed, col, bits) - midpoint) * x[col];
        a1 += ((float)moe_packed_lsb_code_at(packed, col + 1u, bits) - midpoint) * x[col + 1u];
        a2 += ((float)moe_packed_lsb_code_at(packed, col + 2u, bits) - midpoint) * x[col + 2u];
        a3 += ((float)moe_packed_lsb_code_at(packed, col + 3u, bits) - midpoint) * x[col + 3u];
    }
    float acc = (a0 + a1) + (a2 + a3);
    for (; col < limit; ++col) {
        acc += ((float)moe_packed_lsb_code_at(packed, col, bits) - midpoint) * x[col];
    }
    return acc * scale;
}

static float moe_dot_linear_quant_row(
    const moe_tensor_record& rec,
    const mmap_context_t* shard,
    const uint8_t* packed,
    const float* x,
    uint32_t bits,
    uint32_t row
) {
    const uint32_t ngroups = rec.info.groups ? rec.info.groups : 1u;
    const uint32_t group_size = rec.info.group_size ? rec.info.group_size : rec.info.cols;
    float acc = 0.0f;
    if (bits != 2u && bits != 3u && bits != 4u && bits != 8u) {
        return 0.0f;
    }
    for (uint32_t group = 0; group < ngroups; ++group) {
        const float scale = moe_raw_scale_at(rec, shard, row, group);
        const uint32_t start = group * group_size;
        const uint32_t limit = std::min<uint32_t>(start + group_size, rec.info.cols);
        if (bits == 8u) {
            acc += moe_dot_linear_q8_group(packed, x, scale, start, limit);
        } else {
            acc += moe_dot_linear_packed_group(packed, x, scale, bits, start, limit);
        }
    }
    return acc;
}

// BUGFIX 46: Issue 10 - AVX2 vectorized bf16?�f32 AXPY for Q-side absorption ?�★??
// Problem: moe_axpy_bf16_row_f32 called 12,288 times per token (64 heads × 192 nope_dim)
// in scalar loop. Each call processes 512 elements (cdim). Total: 6.3M scalar ops/token.
// Solution: AVX2 path processes 8 bf16?�f32 conversions + FMA per iteration.
// Pattern: _mm256_cvtepu16_epi32 + _mm256_slli_epi32(16) + _mm256_fmadd_ps
// Symmetric to store_entry AVX2 path (inverse direction).
static void moe_axpy_bf16_row_f32_unused(float* y, const uint8_t* packed_bf16, float alpha, uint32_t cols) {
    const uint16_t* bf16 = reinterpret_cast<const uint16_t*>(packed_bf16);
    uint32_t col = 0;
#if defined(__AVX2__)
    const __m256 valpha = _mm256_set1_ps(alpha);
    for (; col + 7u < cols; col += 8u) {
        // Load 8 bf16 values (16-bit each)
        __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(bf16 + col));
        // Zero-extend to 32-bit integers
        __m256i expanded = _mm256_cvtepu16_epi32(packed);
        // Shift left by 16 to convert bf16 to f32 (bf16 is upper 16 bits of f32)
        __m256i shifted = _mm256_slli_epi32(expanded, 16);
        __m256 values = _mm256_castsi256_ps(shifted);
        // Load current y values
        __m256 vy = _mm256_loadu_ps(y + col);
        // y = y + alpha * values (FMA: fused multiply-add)
        vy = _mm256_fmadd_ps(values, valpha, vy);
        // Store result
        _mm256_storeu_ps(y + col, vy);
    }
#endif
    // Scalar tail
    for (; col < cols; ++col) {
        uint32_t bits = (uint32_t)bf16[col] << 16;
        float value;
        std::memcpy(&value, &bits, sizeof(value));
        y[col] += alpha * value;
    }
}

static float moe_dot_fp4_row(
    const moe_tensor_record& rec,
    const mmap_context_t* shard,
    const uint8_t* packed,
    const float* x,
    moe_pc_engine_t* engine,
    uint32_t row
) {
    const uint32_t ngroups = rec.info.groups;
    // Fix 28: Shrink thread_local buffer if it grew too large (4x current need)
    static thread_local std::vector<float> tls_scales_buf;
    // BUGFIX 501: ngroups 범위 체크
    if (ngroups > 100000) return 0.0f;
    if (tls_scales_buf.size() < ngroups) {
        tls_scales_buf.resize(ngroups);
    } else if (tls_scales_buf.capacity() > ngroups * 4u && ngroups > 0) {
        tls_scales_buf.shrink_to_fit();
        tls_scales_buf.resize(ngroups);
    }
    float* scales_buf = tls_scales_buf.data();
    if (rec.scale4_entry && engine->scale4) {
        for (uint32_t g = 0; g < ngroups; ++g) {
            scales_buf[g] = moe_scale4_get_scale(engine->scale4, rec.scale4_entry, row, g);
        }
    } else {
        for (uint32_t g = 0; g < ngroups; ++g) {
            scales_buf[g] = moe_raw_scale_at(rec, shard, row, g);
        }
    }
    float acc = 0.0f;
    for (uint32_t group = 0; group < ngroups; ++group) {
        const float scale = scales_buf[group];
        const uint32_t start = group * rec.info.group_size;
        const uint32_t limit = std::min<uint32_t>(start + rec.info.group_size, rec.info.cols);
#if defined(__AVX2__)
        // Fix 25: True SIMD nibble extraction; process 16 nibbles (8 bytes) per iteration.
        // Each byte contains 2 nibbles: low nibble at even col, high nibble at odd col
        const __m256 vscale = _mm256_set1_ps(scale);
        __m256 vacc = _mm256_setzero_ps();
        uint32_t col = start;
        // Process 8 nibbles (4 bytes) per iteration. Scalar table expansion avoids
        // AVX2 gather latency on the 16-entry FP4 table.
        for (; col + 7u < limit; col += 8u) {
            uint32_t byte_offset = col / 2;
            const uint8_t b0 = packed[byte_offset + 0u];
            const uint8_t b1 = packed[byte_offset + 1u];
            const uint8_t b2 = packed[byte_offset + 2u];
            const uint8_t b3 = packed[byte_offset + 3u];
            const float t0 = moe_engine::kNvFp4Table[(b0 >> 4) & 0x0F];
            const float t1 = moe_engine::kNvFp4Table[b0 & 0x0F];
            const float t2 = moe_engine::kNvFp4Table[(b1 >> 4) & 0x0F];
            const float t3 = moe_engine::kNvFp4Table[b1 & 0x0F];
            const float t4 = moe_engine::kNvFp4Table[(b2 >> 4) & 0x0F];
            const float t5 = moe_engine::kNvFp4Table[b2 & 0x0F];
            const float t6 = moe_engine::kNvFp4Table[(b3 >> 4) & 0x0F];
            const float t7 = moe_engine::kNvFp4Table[b3 & 0x0F];
            __m256 vals = _mm256_set_ps(t7, t6, t5, t4, t3, t2, t1, t0);
            vals = _mm256_mul_ps(vals, vscale);
            __m256 xv = _mm256_loadu_ps(x + col);
            vacc = moe_madd_ps(vals, xv, vacc);
        }
        // Horizontal sum
        __m128 lo = _mm256_castps256_ps128(vacc);
        __m128 hi = _mm256_extractf128_ps(vacc, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        acc += _mm_cvtss_f32(lo);
        // Scalar tail
        for (; col < limit; ++col) {
            acc += moe_engine::fp4_value_at(packed, col, scale) * x[col];
        }
#else
        for (uint32_t col = start; col < limit; ++col) {
            acc += moe_engine::fp4_value_at(packed, col, scale) * x[col];
        }
#endif
    }
    return acc;
}
