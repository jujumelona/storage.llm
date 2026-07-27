// Q8/Q4/Q5 quantization kernels
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
#if defined(__AVX2__)
        _mm_prefetch(reinterpret_cast<const char*>(packed + 512), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(x + start + 64), _MM_HINT_T0);
#endif
        uint16_t d16 = 0;
        std::memcpy(&d16, packed, sizeof(d16));
        const float d = moe_raw_fp16_to_f32(d16);
        const int8_t* q = reinterpret_cast<const int8_t*>(packed + 2u);
        const uint32_t limit = std::min<uint32_t>(block_cols, cols - start);
#if defined(__AVX2__)
        // PERF: Scale-factor hoisting — accumulate q*x first, multiply by d once.
        // Removes N/8 vector multiplications per block (4 for Q8_0's 32-element blocks).
        __m256 vacc = _mm256_setzero_ps();
        uint32_t i = 0;
        for (; i + 7u < limit; i += 8u) {
            const __m256 qv = moe_i8x8_to_ps(q + i);
            const __m256 xv = _mm256_loadu_ps(x + start + i);
            vacc = moe_madd_ps(qv, xv, vacc);
        }
        acc += moe_dot_hsum256_ps(vacc) * d;
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

static int moe_dot_gguf_q8_0_two_rows(
    const uint8_t* packed0,
    const uint8_t* packed1,
    const float* x,
    uint32_t cols,
    float* out0,
    float* out1
) {
    if (!packed0 || !packed1 || !x || !out0 || !out1) {
        return 0;
    }
    *out0 = moe_dot_gguf_q8_0_row(packed0, x, cols);
    *out1 = moe_dot_gguf_q8_0_row(packed1, x, cols);
    return 1;
}

static int moe_dot_gguf_q8_0_four_rows(
    const uint8_t* packed0,
    const uint8_t* packed1,
    const uint8_t* packed2,
    const uint8_t* packed3,
    const float* x,
    uint32_t cols,
    float* out0,
    float* out1,
    float* out2,
    float* out3
) {
    if (!packed0 || !packed1 || !packed2 || !packed3 || !x || !out0 || !out1 || !out2 || !out3) {
        return 0;
    }
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const uint32_t block_cols = 32u;
    const uint32_t block_bytes = 34u;
    for (uint32_t start = 0; start < cols; start += block_cols) {
#if defined(__AVX2__)
        // PERF: Software prefetch next block weights into L1 cache (~256 bytes ahead)
        _mm_prefetch(reinterpret_cast<const char*>(packed0 + 256), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(packed1 + 256), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(packed2 + 256), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(packed3 + 256), _MM_HINT_T0);
#endif
        uint16_t d0_16 = 0, d1_16 = 0, d2_16 = 0, d3_16 = 0;
        std::memcpy(&d0_16, packed0, 2);
        std::memcpy(&d1_16, packed1, 2);
        std::memcpy(&d2_16, packed2, 2);
        std::memcpy(&d3_16, packed3, 2);
        const float d0 = moe_raw_fp16_to_f32(d0_16);
        const float d1 = moe_raw_fp16_to_f32(d1_16);
        const float d2 = moe_raw_fp16_to_f32(d2_16);
        const float d3 = moe_raw_fp16_to_f32(d3_16);
        const int8_t* q0 = reinterpret_cast<const int8_t*>(packed0 + 2u);
        const int8_t* q1 = reinterpret_cast<const int8_t*>(packed1 + 2u);
        const int8_t* q2 = reinterpret_cast<const int8_t*>(packed2 + 2u);
        const int8_t* q3 = reinterpret_cast<const int8_t*>(packed3 + 2u);
        const uint32_t limit = std::min<uint32_t>(block_cols, cols - start);
#if defined(__AVX2__)
        // PERF: Fused 4-row GEMV — load vector x ONCE into SIMD register for 4 rows.
        // Cuts input vector L1/L2 memory bandwidth demand by 75%.
        __m256 vacc0 = _mm256_setzero_ps();
        __m256 vacc1 = _mm256_setzero_ps();
        __m256 vacc2 = _mm256_setzero_ps();
        __m256 vacc3 = _mm256_setzero_ps();
        uint32_t i = 0;
        for (; i + 7u < limit; i += 8u) {
            const __m256 xv = _mm256_loadu_ps(x + start + i);
            vacc0 = moe_madd_ps(moe_i8x8_to_ps(q0 + i), xv, vacc0);
            vacc1 = moe_madd_ps(moe_i8x8_to_ps(q1 + i), xv, vacc1);
            vacc2 = moe_madd_ps(moe_i8x8_to_ps(q2 + i), xv, vacc2);
            vacc3 = moe_madd_ps(moe_i8x8_to_ps(q3 + i), xv, vacc3);
        }
        acc0 += moe_dot_hsum256_ps(vacc0) * d0;
        acc1 += moe_dot_hsum256_ps(vacc1) * d1;
        acc2 += moe_dot_hsum256_ps(vacc2) * d2;
        acc3 += moe_dot_hsum256_ps(vacc3) * d3;
        for (; i < limit; ++i) {
            const float xi = x[start + i];
            acc0 += d0 * (float)q0[i] * xi;
            acc1 += d1 * (float)q1[i] * xi;
            acc2 += d2 * (float)q2[i] * xi;
            acc3 += d3 * (float)q3[i] * xi;
        }
#else
        for (uint32_t i = 0; i < limit; ++i) {
            const float xi = x[start + i];
            acc0 += d0 * (float)q0[i] * xi;
            acc1 += d1 * (float)q1[i] * xi;
            acc2 += d2 * (float)q2[i] * xi;
            acc3 += d3 * (float)q3[i] * xi;
        }
#endif
        packed0 += block_bytes;
        packed1 += block_bytes;
        packed2 += block_bytes;
        packed3 += block_bytes;
    }
    *out0 = acc0; *out1 = acc1; *out2 = acc2; *out3 = acc3;
    return 1;
}

static int moe_dot_gguf_q8_0_eight_rows(
    const uint8_t* p0, const uint8_t* p1, const uint8_t* p2, const uint8_t* p3,
    const uint8_t* p4, const uint8_t* p5, const uint8_t* p6, const uint8_t* p7,
    const float* x, uint32_t cols,
    float* out0, float* out1, float* out2, float* out3,
    float* out4, float* out5, float* out6, float* out7
) {
    if (!p0 || !p1 || !p2 || !p3 || !p4 || !p5 || !p6 || !p7 || !x ||
        !out0 || !out1 || !out2 || !out3 || !out4 || !out5 || !out6 || !out7) {
        return 0;
    }
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
    const uint32_t block_cols = 32u;
    const uint32_t block_bytes = 34u;
    for (uint32_t start = 0; start < cols; start += block_cols) {
#if defined(__AVX2__)
        _mm_prefetch(reinterpret_cast<const char*>(p0 + 256), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(p1 + 256), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(p2 + 256), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(p3 + 256), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(p4 + 256), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(p5 + 256), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(p6 + 256), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(p7 + 256), _MM_HINT_T0);
#endif
        uint16_t d0_16 = 0, d1_16 = 0, d2_16 = 0, d3_16 = 0;
        uint16_t d4_16 = 0, d5_16 = 0, d6_16 = 0, d7_16 = 0;
        std::memcpy(&d0_16, p0, 2); std::memcpy(&d1_16, p1, 2);
        std::memcpy(&d2_16, p2, 2); std::memcpy(&d3_16, p3, 2);
        std::memcpy(&d4_16, p4, 2); std::memcpy(&d5_16, p5, 2);
        std::memcpy(&d6_16, p6, 2); std::memcpy(&d7_16, p7, 2);
        const float d0 = moe_raw_fp16_to_f32(d0_16), d1 = moe_raw_fp16_to_f32(d1_16);
        const float d2 = moe_raw_fp16_to_f32(d2_16), d3 = moe_raw_fp16_to_f32(d3_16);
        const float d4 = moe_raw_fp16_to_f32(d4_16), d5 = moe_raw_fp16_to_f32(d5_16);
        const float d6 = moe_raw_fp16_to_f32(d6_16), d7 = moe_raw_fp16_to_f32(d7_16);
        const int8_t* q0 = reinterpret_cast<const int8_t*>(p0 + 2u);
        const int8_t* q1 = reinterpret_cast<const int8_t*>(p1 + 2u);
        const int8_t* q2 = reinterpret_cast<const int8_t*>(p2 + 2u);
        const int8_t* q3 = reinterpret_cast<const int8_t*>(p3 + 2u);
        const int8_t* q4 = reinterpret_cast<const int8_t*>(p4 + 2u);
        const int8_t* q5 = reinterpret_cast<const int8_t*>(p5 + 2u);
        const int8_t* q6 = reinterpret_cast<const int8_t*>(p6 + 2u);
        const int8_t* q7 = reinterpret_cast<const int8_t*>(p7 + 2u);
        const uint32_t limit = std::min<uint32_t>(block_cols, cols - start);
#if defined(__AVX2__)
        __m256 vacc0 = _mm256_setzero_ps(), vacc1 = _mm256_setzero_ps();
        __m256 vacc2 = _mm256_setzero_ps(), vacc3 = _mm256_setzero_ps();
        __m256 vacc4 = _mm256_setzero_ps(), vacc5 = _mm256_setzero_ps();
        __m256 vacc6 = _mm256_setzero_ps(), vacc7 = _mm256_setzero_ps();
        uint32_t i = 0;
        for (; i + 7u < limit; i += 8u) {
            const __m256 xv = _mm256_loadu_ps(x + start + i);
            vacc0 = moe_madd_ps(moe_i8x8_to_ps(q0 + i), xv, vacc0);
            vacc1 = moe_madd_ps(moe_i8x8_to_ps(q1 + i), xv, vacc1);
            vacc2 = moe_madd_ps(moe_i8x8_to_ps(q2 + i), xv, vacc2);
            vacc3 = moe_madd_ps(moe_i8x8_to_ps(q3 + i), xv, vacc3);
            vacc4 = moe_madd_ps(moe_i8x8_to_ps(q4 + i), xv, vacc4);
            vacc5 = moe_madd_ps(moe_i8x8_to_ps(q5 + i), xv, vacc5);
            vacc6 = moe_madd_ps(moe_i8x8_to_ps(q6 + i), xv, vacc6);
            vacc7 = moe_madd_ps(moe_i8x8_to_ps(q7 + i), xv, vacc7);
        }
        acc0 += moe_dot_hsum256_ps(vacc0) * d0; acc1 += moe_dot_hsum256_ps(vacc1) * d1;
        acc2 += moe_dot_hsum256_ps(vacc2) * d2; acc3 += moe_dot_hsum256_ps(vacc3) * d3;
        acc4 += moe_dot_hsum256_ps(vacc4) * d4; acc5 += moe_dot_hsum256_ps(vacc5) * d5;
        acc6 += moe_dot_hsum256_ps(vacc6) * d6; acc7 += moe_dot_hsum256_ps(vacc7) * d7;
        for (; i < limit; ++i) {
            const float xi = x[start + i];
            acc0 += d0 * (float)q0[i] * xi; acc1 += d1 * (float)q1[i] * xi;
            acc2 += d2 * (float)q2[i] * xi; acc3 += d3 * (float)q3[i] * xi;
            acc4 += d4 * (float)q4[i] * xi; acc5 += d5 * (float)q5[i] * xi;
            acc6 += d6 * (float)q6[i] * xi; acc7 += d7 * (float)q7[i] * xi;
        }
#else
        for (uint32_t i = 0; i < limit; ++i) {
            const float xi = x[start + i];
            acc0 += d0 * (float)q0[i] * xi; acc1 += d1 * (float)q1[i] * xi;
            acc2 += d2 * (float)q2[i] * xi; acc3 += d3 * (float)q3[i] * xi;
            acc4 += d4 * (float)q4[i] * xi; acc5 += d5 * (float)q5[i] * xi;
            acc6 += d6 * (float)q6[i] * xi; acc7 += d7 * (float)q7[i] * xi;
        }
#endif
        p0 += block_bytes; p1 += block_bytes; p2 += block_bytes; p3 += block_bytes;
        p4 += block_bytes; p5 += block_bytes; p6 += block_bytes; p7 += block_bytes;
    }
    *out0 = acc0; *out1 = acc1; *out2 = acc2; *out3 = acc3;
    *out4 = acc4; *out5 = acc5; *out6 = acc6; *out7 = acc7;
    return 1;
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
#if defined(__AVX2__)
        // PERF: Scale-factor hoisting for Q8_1.
        __m256 vacc = _mm256_setzero_ps();
        uint32_t i = 0;
        for (; i + 7u < limit; i += 8u) {
            const __m256 qv = moe_i8x8_to_ps(q + i);
            const __m256 xv = _mm256_loadu_ps(x + start + i);
            vacc = moe_madd_ps(qv, xv, vacc);
        }
        acc += moe_dot_hsum256_ps(vacc) * d;
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

static float moe_dot_gguf_q8_k_row(
    const uint8_t* packed,
    const float* x,
    uint32_t cols
) {
    if (!packed || !x) {
        return 0.0f;
    }
    float acc = 0.0f;
    const uint32_t block_cols = 256u;
    const uint32_t block_bytes = 292u;
    for (uint32_t start = 0; start < cols; start += block_cols) {
        float d = 0.0f;
        std::memcpy(&d, packed, sizeof(d));
        const int8_t* q = reinterpret_cast<const int8_t*>(packed + 4u);
        const uint32_t limit = std::min<uint32_t>(block_cols, cols - start);
#if defined(__AVX2__)
        // PERF: Scale-factor hoisting for Q8_K (256-element blocks).
        // Eliminates 32 _mm256_mul_ps per block.
        __m256 vacc0 = _mm256_setzero_ps();
        __m256 vacc1 = _mm256_setzero_ps();
        uint32_t i = 0;
        for (; i + 15u < limit; i += 16u) {
            vacc0 = moe_madd_ps(moe_i8x8_to_ps(q + i),      _mm256_loadu_ps(x + start + i),      vacc0);
            vacc1 = moe_madd_ps(moe_i8x8_to_ps(q + i + 8u), _mm256_loadu_ps(x + start + i + 8u), vacc1);
        }
        for (; i + 7u < limit; i += 8u) {
            vacc0 = moe_madd_ps(moe_i8x8_to_ps(q + i), _mm256_loadu_ps(x + start + i), vacc0);
        }
        acc += moe_dot_hsum256_ps(_mm256_add_ps(vacc0, vacc1)) * d;
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
#if defined(__AVX2__)
        _mm_prefetch(reinterpret_cast<const char*>(packed + 256), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(x + start + 64), _MM_HINT_T0);
#endif
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
#if defined(__AVX2__)
        _mm_prefetch(reinterpret_cast<const char*>(packed + 256), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(x + start + 64), _MM_HINT_T0);
#endif
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
    static const std::array<float, 256> kScale = []() {
        std::array<float, 256> values{};
        for (uint32_t i = 0; i < 256u; ++i) {
            const uint32_t bits = i < 2u ? (0x00200000u << i) : ((i - 1u) << 23u);
            float out = 0.0f;
            std::memcpy(&out, &bits, sizeof(out));
            values[i] = out;
        }
        return values;
    }();
    return kScale[v];
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

static int moe_mxfp4_pairwise_nibble_order_enabled() {
    // GGML MXFP4 stores a 32-value block as sixteen payload bytes:
    // low nibbles decode logical columns 0..15 and high nibbles decode
    // columns 16..31.  JUJU materialization preserves source quant rows, so
    // use that split-16 layout unless a repaired artifact explicitly asks for
    // the older pairwise experiment.
    static int mode = -1;
    if (mode < 0) {
        const char* env = std::getenv("STORAGELLM_MXFP4_NIBBLE_ORDER");
        mode = (env && *env &&
                (std::strcmp(env, "pairwise") == 0 ||
                 std::strcmp(env, "juju_pairwise") == 0)) ? 1 : 0;
    }
    return mode != 0;
}

static uint8_t moe_mxfp4_code_at(const uint8_t* qs, uint32_t index) {
    if (moe_mxfp4_pairwise_nibble_order_enabled()) {
        const uint8_t byte = qs[index >> 1u];
        return (index & 1u) ? (byte >> 4) : (byte & 0x0fu);
    }
    const uint8_t byte = qs[index & 15u];
    return (index < 16u) ? (byte & 0x0fu) : (byte >> 4);
}

#if defined(__AVX2__)
static inline __m256 moe_mxfp4_codes8_to_ps(__m128i codes) {
    const __m128i table = _mm_setr_epi8(
        0, 1, 2, 3, 4, 6, 8, 12,
        0, -1, -2, -3, -4, -6, -8, -12);
    const __m128i vals_i8 = _mm_shuffle_epi8(table, codes);
    const __m256 vals = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(vals_i8));
    return _mm256_mul_ps(vals, _mm256_set1_ps(0.5f));
}

static inline __m256 moe_mxfp4_lookup8_fast(const uint8_t* p, int high) {
    const __m128i mask = _mm_set1_epi8(0x0f);
    __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));
    bytes = high ? _mm_and_si128(_mm_srli_epi16(bytes, 4), mask)
                 : _mm_and_si128(bytes, mask);
    return moe_mxfp4_codes8_to_ps(bytes);
}
#endif

struct moe_mxfp4_row_layout_t {
    uint32_t block_cols;
    uint32_t block_bytes;
    uint8_t bytewise_codes;
};

static inline uint64_t moe_mxfp4_row_bytes_for_block_cols(uint32_t cols, uint32_t block_cols) {
    return ((uint64_t)cols + (uint64_t)block_cols - 1ull) / (uint64_t)block_cols * 17ull;
}

static moe_mxfp4_row_layout_t moe_mxfp4_row_layout_for_bytes(uint32_t cols, uint64_t physical_row_bytes) {
    // Standard GGML MXFP4 is a 32-value block: one E8M0 scale byte plus
    // sixteen packed E2M1 bytes.  Some JUJU/GGUF_MXFP4_MOE artifacts carry an
    // expanded physical row layout instead: one scale byte plus sixteen
    // bytewise low-nibble FP4 codes for each 16 logical values.  The tensor
    // index row stride is the only reliable runtime discriminator here; do not
    // force either layout from model names or global feature flags.
    moe_mxfp4_row_layout_t layout{32u, 17u, 0u};
    if (cols == 0 || physical_row_bytes == 0) {
        return layout;
    }
    const uint64_t bytes_16_expanded = moe_mxfp4_row_bytes_for_block_cols(cols, 16u);
    if (physical_row_bytes == bytes_16_expanded) {
        layout.block_cols = 16u;
        layout.bytewise_codes = 1u;
        return layout;
    }
    const uint64_t bytes_32 = moe_mxfp4_row_bytes_for_block_cols(cols, 32u);
    if (physical_row_bytes >= bytes_32) {
        layout.block_cols = 32u;
        layout.bytewise_codes = 0u;
        return layout;
    }
    return layout;
}

static inline uint8_t moe_mxfp4_code_at_layout(
    const uint8_t* qs,
    uint32_t index,
    const moe_mxfp4_row_layout_t& layout
) {
    return layout.bytewise_codes ? (qs[index] & 0x0fu) : moe_mxfp4_code_at(qs, index);
}

static inline float moe_mxfp4_dot_block(
    const uint8_t* qs,
    const float* x,
    uint32_t start,
    uint32_t limit,
    const moe_mxfp4_row_layout_t& layout,
    const float* table,
    int pairwise_order
) {
#if defined(__AVX2__)
    if (!pairwise_order && limit == 32u && !layout.bytewise_codes) {
        (void)table;
        __m256 acc = _mm256_setzero_ps();
        acc = moe_madd_ps(moe_mxfp4_lookup8_fast(qs, 0), _mm256_loadu_ps(x + start), acc);
        acc = moe_madd_ps(moe_mxfp4_lookup8_fast(qs + 8u, 0), _mm256_loadu_ps(x + start + 8u), acc);
        acc = moe_madd_ps(moe_mxfp4_lookup8_fast(qs, 1), _mm256_loadu_ps(x + start + 16u), acc);
        acc = moe_madd_ps(moe_mxfp4_lookup8_fast(qs + 8u, 1), _mm256_loadu_ps(x + start + 24u), acc);
        return moe_dot_hsum256_ps(acc);
    }
    if (!pairwise_order && limit == 16u && layout.bytewise_codes) {
        (void)table;
        __m256 acc = _mm256_setzero_ps();
        acc = moe_madd_ps(moe_mxfp4_lookup8_fast(qs, 0), _mm256_loadu_ps(x + start), acc);
        acc = moe_madd_ps(moe_mxfp4_lookup8_fast(qs + 8u, 0), _mm256_loadu_ps(x + start + 8u), acc);
        return moe_dot_hsum256_ps(acc);
    }
#endif
    float acc = 0.0f;
    if (layout.bytewise_codes) {
        for (uint32_t j = 0; j < limit; ++j) {
            acc += table[qs[j] & 0x0fu] * x[start + j];
        }
        return acc;
    }
    if (pairwise_order) {
        uint32_t j = 0;
        for (; j + 1u < limit; j += 2u) {
            const uint8_t byte = qs[j >> 1u];
            acc += table[byte & 0x0fu] * x[start + j];
            acc += table[byte >> 4] * x[start + j + 1u];
        }
        if (j < limit) {
            acc += table[qs[j >> 1u] & 0x0fu] * x[start + j];
        }
        return acc;
    }
    const uint32_t low_limit = std::min<uint32_t>(limit, 16u);
    for (uint32_t j = 0; j < low_limit; ++j) {
        acc += table[qs[j] & 0x0fu] * x[start + j];
    }
    for (uint32_t j = 16u; j < limit; ++j) {
        acc += table[qs[j - 16u] >> 4] * x[start + j];
    }
    return acc;
}

static inline void moe_mxfp4_dot_block_two_rows(
    const uint8_t* qs0,
    const uint8_t* qs1,
    const float* x,
    uint32_t start,
    uint32_t limit,
    const moe_mxfp4_row_layout_t& layout,
    const float* table,
    int pairwise_order,
    float* out0,
    float* out1
);

#if defined(__AVX2__)
static inline int moe_mxfp4_dot_row_accum_avx2(
    const uint8_t* packed,
    const float* x,
    uint32_t cols,
    const moe_mxfp4_row_layout_t& layout,
    int pairwise_order,
    float* out_value
) {
    if (!packed || !x || !out_value || pairwise_order) {
        return 0;
    }
    const int standard_32 = (layout.block_cols == 32u && !layout.bytewise_codes);
    const int bytewise_16 = (layout.block_cols == 16u && layout.bytewise_codes);
    if (!standard_32 && !bytewise_16) {
        return 0;
    }

    __m256 vacc = _mm256_setzero_ps();
    float scalar_tail = 0.0f;
    const float* table = moe_mxfp4_value_table();
    uint32_t start = 0;
    while (start + layout.block_cols <= cols) {
        const __m256 vscale = _mm256_set1_ps(moe_e8m0_to_f32_half(packed[0]));
        const uint8_t* qs = packed + 1u;
        if (standard_32) {
            vacc = moe_madd_ps(
                _mm256_mul_ps(moe_mxfp4_lookup8_fast(qs, 0), vscale),
                _mm256_loadu_ps(x + start),
                vacc);
            vacc = moe_madd_ps(
                _mm256_mul_ps(moe_mxfp4_lookup8_fast(qs + 8u, 0), vscale),
                _mm256_loadu_ps(x + start + 8u),
                vacc);
            vacc = moe_madd_ps(
                _mm256_mul_ps(moe_mxfp4_lookup8_fast(qs, 1), vscale),
                _mm256_loadu_ps(x + start + 16u),
                vacc);
            vacc = moe_madd_ps(
                _mm256_mul_ps(moe_mxfp4_lookup8_fast(qs + 8u, 1), vscale),
                _mm256_loadu_ps(x + start + 24u),
                vacc);
        } else {
            vacc = moe_madd_ps(
                _mm256_mul_ps(moe_mxfp4_lookup8_fast(qs, 0), vscale),
                _mm256_loadu_ps(x + start),
                vacc);
            vacc = moe_madd_ps(
                _mm256_mul_ps(moe_mxfp4_lookup8_fast(qs + 8u, 0), vscale),
                _mm256_loadu_ps(x + start + 8u),
                vacc);
        }
        start += layout.block_cols;
        packed += layout.block_bytes;
    }
    while (start < cols) {
        const float scale = moe_e8m0_to_f32_half(packed[0]);
        const uint8_t* qs = packed + 1u;
        const uint32_t limit = std::min<uint32_t>(layout.block_cols, cols - start);
        scalar_tail += moe_mxfp4_dot_block(qs, x, start, limit, layout, table, pairwise_order) * scale;
        start += limit;
        packed += layout.block_bytes;
    }
    *out_value = moe_dot_hsum256_ps(vacc) + scalar_tail;
    return 1;
}

static inline int moe_mxfp4_dot_two_rows_accum_avx2(
    const uint8_t* packed0,
    const uint8_t* packed1,
    const float* x,
    uint32_t cols,
    const moe_mxfp4_row_layout_t& layout,
    int pairwise_order,
    float* out0,
    float* out1
) {
    if (!packed0 || !packed1 || !x || !out0 || !out1 || pairwise_order) {
        return 0;
    }
    const int standard_32 = (layout.block_cols == 32u && !layout.bytewise_codes);
    const int bytewise_16 = (layout.block_cols == 16u && layout.bytewise_codes);
    if (!standard_32 && !bytewise_16) {
        return 0;
    }

    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    float tail0 = 0.0f;
    float tail1 = 0.0f;
    const float* table = moe_mxfp4_value_table();
    uint32_t start = 0;
    while (start + layout.block_cols <= cols) {
        const __m256 scale0 = _mm256_set1_ps(moe_e8m0_to_f32_half(packed0[0]));
        const __m256 scale1 = _mm256_set1_ps(moe_e8m0_to_f32_half(packed1[0]));
        const uint8_t* qs0 = packed0 + 1u;
        const uint8_t* qs1 = packed1 + 1u;
        if (standard_32) {
            const __m256 x0 = _mm256_loadu_ps(x + start);
            const __m256 x1 = _mm256_loadu_ps(x + start + 8u);
            const __m256 x2 = _mm256_loadu_ps(x + start + 16u);
            const __m256 x3 = _mm256_loadu_ps(x + start + 24u);
            acc0 = moe_madd_ps(_mm256_mul_ps(moe_mxfp4_lookup8_fast(qs0, 0), scale0), x0, acc0);
            acc1 = moe_madd_ps(_mm256_mul_ps(moe_mxfp4_lookup8_fast(qs1, 0), scale1), x0, acc1);
            acc0 = moe_madd_ps(_mm256_mul_ps(moe_mxfp4_lookup8_fast(qs0 + 8u, 0), scale0), x1, acc0);
            acc1 = moe_madd_ps(_mm256_mul_ps(moe_mxfp4_lookup8_fast(qs1 + 8u, 0), scale1), x1, acc1);
            acc0 = moe_madd_ps(_mm256_mul_ps(moe_mxfp4_lookup8_fast(qs0, 1), scale0), x2, acc0);
            acc1 = moe_madd_ps(_mm256_mul_ps(moe_mxfp4_lookup8_fast(qs1, 1), scale1), x2, acc1);
            acc0 = moe_madd_ps(_mm256_mul_ps(moe_mxfp4_lookup8_fast(qs0 + 8u, 1), scale0), x3, acc0);
            acc1 = moe_madd_ps(_mm256_mul_ps(moe_mxfp4_lookup8_fast(qs1 + 8u, 1), scale1), x3, acc1);
        } else {
            const __m256 x0 = _mm256_loadu_ps(x + start);
            const __m256 x1 = _mm256_loadu_ps(x + start + 8u);
            acc0 = moe_madd_ps(_mm256_mul_ps(moe_mxfp4_lookup8_fast(qs0, 0), scale0), x0, acc0);
            acc1 = moe_madd_ps(_mm256_mul_ps(moe_mxfp4_lookup8_fast(qs1, 0), scale1), x0, acc1);
            acc0 = moe_madd_ps(_mm256_mul_ps(moe_mxfp4_lookup8_fast(qs0 + 8u, 0), scale0), x1, acc0);
            acc1 = moe_madd_ps(_mm256_mul_ps(moe_mxfp4_lookup8_fast(qs1 + 8u, 0), scale1), x1, acc1);
        }
        start += layout.block_cols;
        packed0 += layout.block_bytes;
        packed1 += layout.block_bytes;
    }
    while (start < cols) {
        const float scale0 = moe_e8m0_to_f32_half(packed0[0]);
        const float scale1 = moe_e8m0_to_f32_half(packed1[0]);
        const uint8_t* qs0 = packed0 + 1u;
        const uint8_t* qs1 = packed1 + 1u;
        const uint32_t limit = std::min<uint32_t>(layout.block_cols, cols - start);
        float block0 = 0.0f;
        float block1 = 0.0f;
        moe_mxfp4_dot_block_two_rows(
            qs0, qs1, x, start, limit, layout, table, pairwise_order, &block0, &block1);
        tail0 += block0 * scale0;
        tail1 += block1 * scale1;
        start += limit;
        packed0 += layout.block_bytes;
        packed1 += layout.block_bytes;
    }
    *out0 = moe_dot_hsum256_ps(acc0) + tail0;
    *out1 = moe_dot_hsum256_ps(acc1) + tail1;
    return 1;
}

static inline int moe_mxfp4_dot_four_rows_accum_avx2(
    const uint8_t* packed0,
    const uint8_t* packed1,
    const uint8_t* packed2,
    const uint8_t* packed3,
    const float* x,
    uint32_t cols,
    const moe_mxfp4_row_layout_t& layout,
    int pairwise_order,
    float* out0,
    float* out1,
    float* out2,
    float* out3
) {
    if (!packed0 || !packed1 || !packed2 || !packed3 || !x ||
        !out0 || !out1 || !out2 || !out3 || pairwise_order) {
        return 0;
    }
    const int standard_32 = (layout.block_cols == 32u && !layout.bytewise_codes);
    const int bytewise_16 = (layout.block_cols == 16u && layout.bytewise_codes);
    if (!standard_32 && !bytewise_16) {
        return 0;
    }

    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    float tail0 = 0.0f;
    float tail1 = 0.0f;
    float tail2 = 0.0f;
    float tail3 = 0.0f;
    const float* table = moe_mxfp4_value_table();
    uint32_t start = 0;
    while (start + layout.block_cols <= cols) {
        const __m256 scale0 = _mm256_set1_ps(moe_e8m0_to_f32_half(packed0[0]));
        const __m256 scale1 = _mm256_set1_ps(moe_e8m0_to_f32_half(packed1[0]));
        const __m256 scale2 = _mm256_set1_ps(moe_e8m0_to_f32_half(packed2[0]));
        const __m256 scale3 = _mm256_set1_ps(moe_e8m0_to_f32_half(packed3[0]));
        const uint8_t* qs0 = packed0 + 1u;
        const uint8_t* qs1 = packed1 + 1u;
        const uint8_t* qs2 = packed2 + 1u;
        const uint8_t* qs3 = packed3 + 1u;
        if (standard_32) {
            const __m256 x0 = _mm256_loadu_ps(x + start);
            const __m256 x1 = _mm256_loadu_ps(x + start + 8u);
            const __m256 x2 = _mm256_loadu_ps(x + start + 16u);
            const __m256 x3 = _mm256_loadu_ps(x + start + 24u);
            acc0 = moe_madd_ps(_mm256_mul_ps(moe_mxfp4_lookup8_fast(qs0, 0), scale0), x0, acc0);
            acc1 = moe_madd_ps(_mm256_mul_ps(moe_mxfp4_lookup8_fast(qs1, 0), scale1), x0, acc1);
            acc2 = moe_madd_ps(_mm256_mul_ps(moe_mxfp4_lookup8_fast(qs2, 0), scale2), x0, acc2);
            acc3 = moe_madd_ps(_mm256_mul_ps(moe_mxfp4_lookup8_fast(qs3, 0), scale3), x0, acc3);
            acc0 = moe_madd_ps(_mm256_mul_ps(moe_mxfp4_lookup8_fast(qs0 + 8u, 0), scale0), x1, acc0);
            acc1 = moe_madd_ps(_mm256_mul_ps(moe_mxfp4_lookup8_fast(qs1 + 8u, 0), scale1), x1, acc1);
            acc2 = moe_madd_ps(_mm256_mul_ps(moe_mxfp4_lookup8_fast(qs2 + 8u, 0), scale2), x1, acc2);
            acc3 = moe_madd_ps(_mm256_mul_ps(moe_mxfp4_lookup8_fast(qs3 + 8u, 0), scale3), x1, acc3);
            acc0 = moe_madd_ps(_mm256_mul_ps(moe_mxfp4_lookup8_fast(qs0, 1), scale0), x2, acc0);
            acc1 = moe_madd_ps(_mm256_mul_ps(moe_mxfp4_lookup8_fast(qs1, 1), scale1), x2, acc1);
            acc2 = moe_madd_ps(_mm256_mul_ps(moe_mxfp4_lookup8_fast(qs2, 1), scale2), x2, acc2);
            acc3 = moe_madd_ps(_mm256_mul_ps(moe_mxfp4_lookup8_fast(qs3, 1), scale3), x2, acc3);
            acc0 = moe_madd_ps(_mm256_mul_ps(moe_mxfp4_lookup8_fast(qs0 + 8u, 1), scale0), x3, acc0);
            acc1 = moe_madd_ps(_mm256_mul_ps(moe_mxfp4_lookup8_fast(qs1 + 8u, 1), scale1), x3, acc1);
            acc2 = moe_madd_ps(_mm256_mul_ps(moe_mxfp4_lookup8_fast(qs2 + 8u, 1), scale2), x3, acc2);
            acc3 = moe_madd_ps(_mm256_mul_ps(moe_mxfp4_lookup8_fast(qs3 + 8u, 1), scale3), x3, acc3);
        } else {
            const __m256 x0 = _mm256_loadu_ps(x + start);
            const __m256 x1 = _mm256_loadu_ps(x + start + 8u);
            acc0 = moe_madd_ps(_mm256_mul_ps(moe_mxfp4_lookup8_fast(qs0, 0), scale0), x0, acc0);
            acc1 = moe_madd_ps(_mm256_mul_ps(moe_mxfp4_lookup8_fast(qs1, 0), scale1), x0, acc1);
            acc2 = moe_madd_ps(_mm256_mul_ps(moe_mxfp4_lookup8_fast(qs2, 0), scale2), x0, acc2);
            acc3 = moe_madd_ps(_mm256_mul_ps(moe_mxfp4_lookup8_fast(qs3, 0), scale3), x0, acc3);
            acc0 = moe_madd_ps(_mm256_mul_ps(moe_mxfp4_lookup8_fast(qs0 + 8u, 0), scale0), x1, acc0);
            acc1 = moe_madd_ps(_mm256_mul_ps(moe_mxfp4_lookup8_fast(qs1 + 8u, 0), scale1), x1, acc1);
            acc2 = moe_madd_ps(_mm256_mul_ps(moe_mxfp4_lookup8_fast(qs2 + 8u, 0), scale2), x1, acc2);
            acc3 = moe_madd_ps(_mm256_mul_ps(moe_mxfp4_lookup8_fast(qs3 + 8u, 0), scale3), x1, acc3);
        }
        start += layout.block_cols;
        packed0 += layout.block_bytes;
        packed1 += layout.block_bytes;
        packed2 += layout.block_bytes;
        packed3 += layout.block_bytes;
    }
    while (start < cols) {
        const float scale0 = moe_e8m0_to_f32_half(packed0[0]);
        const float scale1 = moe_e8m0_to_f32_half(packed1[0]);
        const float scale2 = moe_e8m0_to_f32_half(packed2[0]);
        const float scale3 = moe_e8m0_to_f32_half(packed3[0]);
        const uint8_t* qs0 = packed0 + 1u;
        const uint8_t* qs1 = packed1 + 1u;
        const uint8_t* qs2 = packed2 + 1u;
        const uint8_t* qs3 = packed3 + 1u;
        const uint32_t limit = std::min<uint32_t>(layout.block_cols, cols - start);
        float block0 = 0.0f;
        float block1 = 0.0f;
        float block2 = 0.0f;
        float block3 = 0.0f;
        moe_mxfp4_dot_block_two_rows(
            qs0, qs1, x, start, limit, layout, table, pairwise_order, &block0, &block1);
        moe_mxfp4_dot_block_two_rows(
            qs2, qs3, x, start, limit, layout, table, pairwise_order, &block2, &block3);
        tail0 += block0 * scale0;
        tail1 += block1 * scale1;
        tail2 += block2 * scale2;
        tail3 += block3 * scale3;
        start += limit;
        packed0 += layout.block_bytes;
        packed1 += layout.block_bytes;
        packed2 += layout.block_bytes;
        packed3 += layout.block_bytes;
    }
    *out0 = moe_dot_hsum256_ps(acc0) + tail0;
    *out1 = moe_dot_hsum256_ps(acc1) + tail1;
    *out2 = moe_dot_hsum256_ps(acc2) + tail2;
    *out3 = moe_dot_hsum256_ps(acc3) + tail3;
    return 1;
}
#endif

static inline void moe_mxfp4_dot_block_two_rows(
    const uint8_t* qs0,
    const uint8_t* qs1,
    const float* x,
    uint32_t start,
    uint32_t limit,
    const moe_mxfp4_row_layout_t& layout,
    const float* table,
    int pairwise_order,
    float* out0,
    float* out1
) {
#if defined(__AVX2__)
    if (!pairwise_order && limit == 32u && !layout.bytewise_codes) {
        (void)table;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        const __m256 x0 = _mm256_loadu_ps(x + start);
        const __m256 x1 = _mm256_loadu_ps(x + start + 8u);
        const __m256 x2 = _mm256_loadu_ps(x + start + 16u);
        const __m256 x3 = _mm256_loadu_ps(x + start + 24u);
        acc0 = moe_madd_ps(moe_mxfp4_lookup8_fast(qs0, 0), x0, acc0);
        acc1 = moe_madd_ps(moe_mxfp4_lookup8_fast(qs1, 0), x0, acc1);
        acc0 = moe_madd_ps(moe_mxfp4_lookup8_fast(qs0 + 8u, 0), x1, acc0);
        acc1 = moe_madd_ps(moe_mxfp4_lookup8_fast(qs1 + 8u, 0), x1, acc1);
        acc0 = moe_madd_ps(moe_mxfp4_lookup8_fast(qs0, 1), x2, acc0);
        acc1 = moe_madd_ps(moe_mxfp4_lookup8_fast(qs1, 1), x2, acc1);
        acc0 = moe_madd_ps(moe_mxfp4_lookup8_fast(qs0 + 8u, 1), x3, acc0);
        acc1 = moe_madd_ps(moe_mxfp4_lookup8_fast(qs1 + 8u, 1), x3, acc1);
        *out0 = moe_dot_hsum256_ps(acc0);
        *out1 = moe_dot_hsum256_ps(acc1);
        return;
    }
    if (!pairwise_order && limit == 16u && layout.bytewise_codes) {
        (void)table;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        const __m256 x0 = _mm256_loadu_ps(x + start);
        const __m256 x1 = _mm256_loadu_ps(x + start + 8u);
        acc0 = moe_madd_ps(moe_mxfp4_lookup8_fast(qs0, 0), x0, acc0);
        acc1 = moe_madd_ps(moe_mxfp4_lookup8_fast(qs1, 0), x0, acc1);
        acc0 = moe_madd_ps(moe_mxfp4_lookup8_fast(qs0 + 8u, 0), x1, acc0);
        acc1 = moe_madd_ps(moe_mxfp4_lookup8_fast(qs1 + 8u, 0), x1, acc1);
        *out0 = moe_dot_hsum256_ps(acc0);
        *out1 = moe_dot_hsum256_ps(acc1);
        return;
    }
#endif
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    if (layout.bytewise_codes) {
        for (uint32_t j = 0; j < limit; ++j) {
            const float xv = x[start + j];
            acc0 += table[qs0[j] & 0x0fu] * xv;
            acc1 += table[qs1[j] & 0x0fu] * xv;
        }
    } else if (pairwise_order) {
        uint32_t j = 0;
        for (; j + 1u < limit; j += 2u) {
            const uint8_t b0 = qs0[j >> 1u];
            const uint8_t b1 = qs1[j >> 1u];
            const float x0 = x[start + j];
            const float x1 = x[start + j + 1u];
            acc0 += table[b0 & 0x0fu] * x0;
            acc0 += table[b0 >> 4] * x1;
            acc1 += table[b1 & 0x0fu] * x0;
            acc1 += table[b1 >> 4] * x1;
        }
        if (j < limit) {
            const uint8_t b0 = qs0[j >> 1u];
            const uint8_t b1 = qs1[j >> 1u];
            const float xv = x[start + j];
            acc0 += table[b0 & 0x0fu] * xv;
            acc1 += table[b1 & 0x0fu] * xv;
        }
    } else {
        const uint32_t low_limit = std::min<uint32_t>(limit, 16u);
        for (uint32_t j = 0; j < low_limit; ++j) {
            const float xv = x[start + j];
            acc0 += table[qs0[j] & 0x0fu] * xv;
            acc1 += table[qs1[j] & 0x0fu] * xv;
        }
        for (uint32_t j = 16u; j < limit; ++j) {
            const float xv = x[start + j];
            acc0 += table[qs0[j - 16u] >> 4] * xv;
            acc1 += table[qs1[j - 16u] >> 4] * xv;
        }
    }
    *out0 = acc0;
    *out1 = acc1;
}

static float moe_dot_gguf_mxfp4_row_strided(
    const uint8_t* packed,
    const float* x,
    uint32_t cols,
    uint64_t physical_row_bytes
);

static float moe_dot_gguf_mxfp4_row_with_layout(
    const uint8_t* packed,
    const float* x,
    uint32_t cols,
    const moe_mxfp4_row_layout_t& layout,
    int pairwise_order
);

static void moe_dot_gguf_mxfp4_two_rows_with_layout(
    const uint8_t* packed0,
    const uint8_t* packed1,
    const float* x,
    uint32_t cols,
    const moe_mxfp4_row_layout_t& layout,
    int pairwise_order,
    float* out0,
    float* out1
);

static void moe_dot_gguf_mxfp4_four_rows_with_layout(
    const uint8_t* packed0,
    const uint8_t* packed1,
    const uint8_t* packed2,
    const uint8_t* packed3,
    const float* x,
    uint32_t cols,
    const moe_mxfp4_row_layout_t& layout,
    int pairwise_order,
    float* out0,
    float* out1,
    float* out2,
    float* out3
);

static void moe_dot_gguf_mxfp4_two_rows_strided(
    const uint8_t* packed0,
    const uint8_t* packed1,
    const float* x,
    uint32_t cols,
    uint64_t row_bytes0,
    uint64_t row_bytes1,
    float* out0,
    float* out1
) {
    if (!out0 || !out1) {
        return;
    }
    if (!packed0 || !packed1 || !x || cols == 0) {
        *out0 = 0.0f;
        *out1 = 0.0f;
        return;
    }
    const moe_mxfp4_row_layout_t layout0 = moe_mxfp4_row_layout_for_bytes(cols, row_bytes0);
    const moe_mxfp4_row_layout_t layout1 = moe_mxfp4_row_layout_for_bytes(cols, row_bytes1);
    if (layout0.block_cols != layout1.block_cols ||
        layout0.bytewise_codes != layout1.bytewise_codes) {
        *out0 = moe_dot_gguf_mxfp4_row_strided(packed0, x, cols, row_bytes0);
        *out1 = moe_dot_gguf_mxfp4_row_strided(packed1, x, cols, row_bytes1);
        return;
    }
    const int pairwise_order = moe_mxfp4_pairwise_nibble_order_enabled();
    moe_dot_gguf_mxfp4_two_rows_with_layout(
        packed0, packed1, x, cols, layout0, pairwise_order, out0, out1);
}

static void moe_dot_gguf_mxfp4_two_rows_with_layout(
    const uint8_t* packed0,
    const uint8_t* packed1,
    const float* x,
    uint32_t cols,
    const moe_mxfp4_row_layout_t& layout,
    int pairwise_order,
    float* out0,
    float* out1
) {
    if (!out0 || !out1) {
        return;
    }
    if (!packed0 || !packed1 || !x || cols == 0) {
        *out0 = 0.0f;
        *out1 = 0.0f;
        return;
    }
    const moe_mxfp4_row_layout_t active_layout =
        (layout.block_cols != 0u && layout.block_bytes != 0u) ?
        layout : moe_mxfp4_row_layout_for_bytes(cols, 0ull);
    const float* table = moe_mxfp4_value_table();
#if defined(__AVX2__)
    if (moe_mxfp4_dot_two_rows_accum_avx2(
            packed0, packed1, x, cols, active_layout, pairwise_order, out0, out1)) {
        return;
    }
#endif
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    uint32_t start = 0;
    while (start < cols) {
        const uint32_t limit = std::min<uint32_t>(active_layout.block_cols, cols - start);
        const float scale0 = moe_e8m0_to_f32_half(packed0[0]);
        const float scale1 = moe_e8m0_to_f32_half(packed1[0]);
        const uint8_t* qs0 = packed0 + 1u;
        const uint8_t* qs1 = packed1 + 1u;
        float block0 = 0.0f;
        float block1 = 0.0f;
        moe_mxfp4_dot_block_two_rows(
            qs0,
            qs1,
            x,
            start,
            limit,
            active_layout,
            table,
            pairwise_order,
            &block0,
            &block1);
        acc0 += block0 * scale0;
        acc1 += block1 * scale1;
        start += limit;
        packed0 += active_layout.block_bytes;
        packed1 += active_layout.block_bytes;
    }
    *out0 = acc0;
    *out1 = acc1;
}

static void moe_dot_gguf_mxfp4_four_rows_with_layout(
    const uint8_t* packed0,
    const uint8_t* packed1,
    const uint8_t* packed2,
    const uint8_t* packed3,
    const float* x,
    uint32_t cols,
    const moe_mxfp4_row_layout_t& layout,
    int pairwise_order,
    float* out0,
    float* out1,
    float* out2,
    float* out3
) {
    if (!out0 || !out1 || !out2 || !out3) {
        return;
    }
    if (!packed0 || !packed1 || !packed2 || !packed3 || !x || cols == 0) {
        *out0 = 0.0f;
        *out1 = 0.0f;
        *out2 = 0.0f;
        *out3 = 0.0f;
        return;
    }
    const moe_mxfp4_row_layout_t active_layout =
        (layout.block_cols != 0u && layout.block_bytes != 0u) ?
        layout : moe_mxfp4_row_layout_for_bytes(cols, 0ull);
#if defined(__AVX2__)
    if (moe_mxfp4_dot_four_rows_accum_avx2(
            packed0, packed1, packed2, packed3, x, cols, active_layout, pairwise_order,
            out0, out1, out2, out3)) {
        return;
    }
#endif
    moe_dot_gguf_mxfp4_two_rows_with_layout(
        packed0, packed1, x, cols, active_layout, pairwise_order, out0, out1);
    moe_dot_gguf_mxfp4_two_rows_with_layout(
        packed2, packed3, x, cols, active_layout, pairwise_order, out2, out3);
}

static void moe_dot_gguf_mxfp4_two_rows(
    const uint8_t* packed0,
    const uint8_t* packed1,
    const float* x,
    uint32_t cols,
    float* out0,
    float* out1
) {
    moe_dot_gguf_mxfp4_two_rows_strided(packed0, packed1, x, cols, 0ull, 0ull, out0, out1);
}

static float moe_dot_gguf_mxfp4_row_strided(
    const uint8_t* packed,
    const float* x,
    uint32_t cols,
    uint64_t physical_row_bytes
) {
    if (!packed || !x) {
        return 0.0f;
    }
    const moe_mxfp4_row_layout_t layout = moe_mxfp4_row_layout_for_bytes(cols, physical_row_bytes);
    const int pairwise_order = moe_mxfp4_pairwise_nibble_order_enabled();
    return moe_dot_gguf_mxfp4_row_with_layout(packed, x, cols, layout, pairwise_order);
}

static float moe_dot_gguf_mxfp4_row_with_layout(
    const uint8_t* packed,
    const float* x,
    uint32_t cols,
    const moe_mxfp4_row_layout_t& layout,
    int pairwise_order
) {
    if (!packed || !x) {
        return 0.0f;
    }
    const moe_mxfp4_row_layout_t active_layout =
        (layout.block_cols != 0u && layout.block_bytes != 0u) ?
        layout : moe_mxfp4_row_layout_for_bytes(cols, 0ull);
    const float* table = moe_mxfp4_value_table();
#if defined(__AVX2__)
    float avx_acc = 0.0f;
    if (moe_mxfp4_dot_row_accum_avx2(packed, x, cols, active_layout, pairwise_order, &avx_acc)) {
        return avx_acc;
    }
#endif
    float acc = 0.0f;
    uint32_t start = 0;
    while (start < cols) {
        const float scale = moe_e8m0_to_f32_half(packed[0]);
        const uint8_t* qs = packed + 1u;
        const uint32_t limit = std::min<uint32_t>(active_layout.block_cols, cols - start);
        const float block_acc = moe_mxfp4_dot_block(qs, x, start, limit, active_layout, table, pairwise_order);
        acc += block_acc * scale;
        start += limit;
        packed += active_layout.block_bytes;
    }
    return acc;
}

static void moe_dot_gguf_mxfp4_row_batch_strided(
    const uint8_t* packed,
    const float* const* batch_x,
    uint32_t batch_size,
    uint32_t cols,
    uint64_t physical_row_bytes,
    float* out_values
) {
    if (!packed || !batch_x || !out_values || batch_size == 0) {
        return;
    }
    if (batch_size == 1u) {
        out_values[0] = batch_x[0]
            ? moe_dot_gguf_mxfp4_row_strided(packed, batch_x[0], cols, physical_row_bytes)
            : 0.0f;
        return;
    }
    const moe_mxfp4_row_layout_t layout = moe_mxfp4_row_layout_for_bytes(cols, physical_row_bytes);
    const moe_mxfp4_row_layout_t active_layout =
        (layout.block_cols != 0u && layout.block_bytes != 0u) ?
        layout : moe_mxfp4_row_layout_for_bytes(cols, 0ull);
    const float* table = moe_mxfp4_value_table();
    const int pairwise_order = moe_mxfp4_pairwise_nibble_order_enabled();
    for (uint32_t b = 0; b < batch_size; ++b) {
        out_values[b] = 0.0f;
    }

    alignas(32) float values[32];
    const uint8_t* row = packed;
    uint32_t start = 0;
    while (start < cols) {
        const float scale = moe_e8m0_to_f32_half(row[0]);
        const uint8_t* qs = row + 1u;
        const uint32_t limit = std::min<uint32_t>(active_layout.block_cols, cols - start);
        if (active_layout.bytewise_codes) {
            for (uint32_t j = 0; j < limit; ++j) {
                values[j] = table[qs[j] & 0x0fu] * scale;
            }
        } else if (pairwise_order) {
            uint32_t j = 0;
            for (; j + 1u < limit; j += 2u) {
                const uint8_t byte = qs[j >> 1u];
                values[j] = table[byte & 0x0fu] * scale;
                values[j + 1u] = table[byte >> 4] * scale;
            }
            if (j < limit) {
                values[j] = table[qs[j >> 1u] & 0x0fu] * scale;
            }
        } else {
            const uint32_t low_limit = std::min<uint32_t>(limit, 16u);
            for (uint32_t j = 0; j < low_limit; ++j) {
                values[j] = table[qs[j] & 0x0fu] * scale;
            }
            for (uint32_t j = 16u; j < limit; ++j) {
                values[j] = table[qs[j - 16u] >> 4] * scale;
            }
        }

        for (uint32_t b = 0; b < batch_size; ++b) {
            const float* x = batch_x[b];
            if (!x) {
                continue;
            }
#if defined(__AVX2__)
            if (limit == 32u) {
                __m256 acc = _mm256_setzero_ps();
                acc = moe_madd_ps(_mm256_load_ps(values), _mm256_loadu_ps(x + start), acc);
                acc = moe_madd_ps(_mm256_load_ps(values + 8u), _mm256_loadu_ps(x + start + 8u), acc);
                acc = moe_madd_ps(_mm256_load_ps(values + 16u), _mm256_loadu_ps(x + start + 16u), acc);
                acc = moe_madd_ps(_mm256_load_ps(values + 24u), _mm256_loadu_ps(x + start + 24u), acc);
                out_values[b] += moe_dot_hsum256_ps(acc);
                continue;
            }
            if (limit == 16u) {
                __m256 acc = _mm256_setzero_ps();
                acc = moe_madd_ps(_mm256_load_ps(values), _mm256_loadu_ps(x + start), acc);
                acc = moe_madd_ps(_mm256_load_ps(values + 8u), _mm256_loadu_ps(x + start + 8u), acc);
                out_values[b] += moe_dot_hsum256_ps(acc);
                continue;
            }
#endif
            float acc = 0.0f;
            for (uint32_t j = 0; j < limit; ++j) {
                acc += values[j] * x[start + j];
            }
            out_values[b] += acc;
        }
        start += limit;
        row += active_layout.block_bytes;
    }
}

static float moe_dot_gguf_mxfp4_row(
    const uint8_t* packed,
    const float* x,
    uint32_t cols
) {
    return moe_dot_gguf_mxfp4_row_strided(packed, x, cols, 0ull);
}
