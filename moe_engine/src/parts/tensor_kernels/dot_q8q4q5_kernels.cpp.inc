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
        uint16_t d16 = 0;
        std::memcpy(&d16, packed, sizeof(d16));
        const float d = moe_raw_fp16_to_f32(d16);
        const int8_t* q = reinterpret_cast<const int8_t*>(packed + 2u);
        const uint32_t limit = std::min<uint32_t>(block_cols, cols - start);
        if (!std::isfinite(d)) {
            return std::numeric_limits<float>::quiet_NaN();
        }
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
        if (!std::isfinite(d)) {
            return std::numeric_limits<float>::quiet_NaN();
        }
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
        if (!std::isfinite(d)) {
            return std::numeric_limits<float>::quiet_NaN();
        }
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
        if (!std::isfinite(d)) {
            return std::numeric_limits<float>::quiet_NaN();
        }
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
        if (!std::isfinite(d) || !std::isfinite(m)) {
            return std::numeric_limits<float>::quiet_NaN();
        }
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
        if (!std::isfinite(d)) {
            return std::numeric_limits<float>::quiet_NaN();
        }
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
    if (v == 0u) {
        uint32_t bits = 0x00400000u;
        float out = 0.0f;
        std::memcpy(&out, &bits, sizeof(out));
        return out;
    }
    uint32_t bits = (uint32_t)v << 23u;
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
    const uint8_t byte = qs[index & 15u];
    return (index < 16u) ? (byte & 0x0fu) : (byte >> 4);
}



static void moe_dot_gguf_mxfp4_two_rows(
    const uint8_t* packed0,
    const uint8_t* packed1,
    const float* x,
    uint32_t cols,
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
    const uint32_t block_cols = 32u;
    const uint32_t block_bytes = 17u;
    const float* table = moe_mxfp4_value_table();
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    uint32_t start = 0;
    const uint32_t full_cols = cols & ~(block_cols - 1u);
    for (; start < full_cols; start += block_cols) {
        const float scale0 = moe_e8m0_to_f32_half(packed0[0]);
        const float scale1 = moe_e8m0_to_f32_half(packed1[0]);
        const uint8_t* qs0 = packed0 + 1u;
        const uint8_t* qs1 = packed1 + 1u;
        const float* xb = x + start;
        float low0 = 0.0f, high0 = 0.0f;
        float low1 = 0.0f, high1 = 0.0f;
#if defined(__AVX2__) && defined(__FMA__)
        const __m256i lo_mask = _mm256_set1_epi32(0x0F);
        __m256 acc0_low_v = _mm256_setzero_ps();
        __m256 acc0_high_v = _mm256_setzero_ps();
        __m256 acc1_low_v = _mm256_setzero_ps();
        __m256 acc1_high_v = _mm256_setzero_ps();
        for (uint32_t j = 0; j < 16u; j += 8u) {
            const __m256 x_low = _mm256_loadu_ps(xb + j);
            const __m256 x_high = _mm256_loadu_ps(xb + 16u + j);

            __m128i raw0_8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qs0 + j));
            __m256i bytes0_8 = _mm256_cvtepu8_epi32(raw0_8);
            __m256i lo0_idx = _mm256_and_si256(bytes0_8, lo_mask);
            __m256i hi0_idx = _mm256_srli_epi32(bytes0_8, 4);
            acc0_low_v = _mm256_fmadd_ps(_mm256_i32gather_ps(table, lo0_idx, 4), x_low, acc0_low_v);
            acc0_high_v = _mm256_fmadd_ps(_mm256_i32gather_ps(table, hi0_idx, 4), x_high, acc0_high_v);

            __m128i raw1_8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qs1 + j));
            __m256i bytes1_8 = _mm256_cvtepu8_epi32(raw1_8);
            __m256i lo1_idx = _mm256_and_si256(bytes1_8, lo_mask);
            __m256i hi1_idx = _mm256_srli_epi32(bytes1_8, 4);
            acc1_low_v = _mm256_fmadd_ps(_mm256_i32gather_ps(table, lo1_idx, 4), x_low, acc1_low_v);
            acc1_high_v = _mm256_fmadd_ps(_mm256_i32gather_ps(table, hi1_idx, 4), x_high, acc1_high_v);
        }
        low0 = moe_dot_hsum256_ps(acc0_low_v);
        high0 = moe_dot_hsum256_ps(acc0_high_v);
        low1 = moe_dot_hsum256_ps(acc1_low_v);
        high1 = moe_dot_hsum256_ps(acc1_high_v);
#else
        for (uint32_t j = 0; j < 16u; ++j) {
            const uint8_t b0 = qs0[j];
            const uint8_t b1 = qs1[j];
            const float xl = xb[j];
            const float xh = xb[16u + j];
            low0 += table[b0 & 0x0fu] * xl;
            high0 += table[b0 >> 4] * xh;
            low1 += table[b1 & 0x0fu] * xl;
            high1 += table[b1 >> 4] * xh;
        }
#endif
        acc0 += (low0 + high0) * scale0;
        acc1 += (low1 + high1) * scale1;
        packed0 += block_bytes;
        packed1 += block_bytes;
    }
    if (start < cols) {
        const float scale0 = moe_e8m0_to_f32_half(packed0[0]);
        const float scale1 = moe_e8m0_to_f32_half(packed1[0]);
        const uint8_t* qs0 = packed0 + 1u;
        const uint8_t* qs1 = packed1 + 1u;
        const uint32_t limit = cols - start;
        float tail0 = 0.0f;
        float tail1 = 0.0f;
        for (uint32_t i = 0; i < limit; ++i) {
            const float xv = x[start + i];
            tail0 += table[moe_mxfp4_code_at(qs0, i)] * xv;
            tail1 += table[moe_mxfp4_code_at(qs1, i)] * xv;
        }
        acc0 += tail0 * scale0;
        acc1 += tail1 * scale1;
    }
    *out0 = acc0;
    *out1 = acc1;
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
        float low = 0.0f;
        float high = 0.0f;
#if defined(__AVX2__) && defined(__FMA__)
        // GGUF block_mxfp4 stores the first 16 values in low nibbles and the
        // next 16 values in high nibbles of the same 16 payload bytes.
        const __m256i lo_mask = _mm256_set1_epi32(0x0F);
        __m256 vacc_low = _mm256_setzero_ps();
        __m256 vacc_high = _mm256_setzero_ps();
        for (uint32_t j = 0; j < 16u; j += 8u) {
            __m128i raw8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qs + j));
            __m256i bytes8 = _mm256_cvtepu8_epi32(raw8);
            __m256i lo_idx = _mm256_and_si256(bytes8, lo_mask);
            __m256 lo_vals = _mm256_i32gather_ps(table, lo_idx, 4);
            __m256i hi_idx = _mm256_srli_epi32(bytes8, 4);
            __m256 hi_vals = _mm256_i32gather_ps(table, hi_idx, 4);
            const __m256 low_x = _mm256_loadu_ps(xb + j);
            const __m256 high_x = _mm256_loadu_ps(xb + 16u + j);
            vacc_low = _mm256_fmadd_ps(lo_vals, low_x, vacc_low);
            vacc_high = _mm256_fmadd_ps(hi_vals, high_x, vacc_high);
        }
        low = moe_dot_hsum256_ps(vacc_low);
        high = moe_dot_hsum256_ps(vacc_high);
#else
        for (uint32_t j = 0; j < 16u; ++j) {
            const uint8_t byte = qs[j];
            low += table[byte & 0x0fu] * xb[j];
            high += table[byte >> 4] * xb[16u + j];
        }
#endif
        acc += (low + high) * scale;
        packed += block_bytes;
    }
    if (start < cols) {
        const float scale = moe_e8m0_to_f32_half(packed[0]);
        const uint8_t* qs = packed + 1u;
        const uint32_t limit = cols - start;
        float block_acc = 0.0f;
        for (uint32_t i = 0; i < limit; ++i) {
            block_acc += table[moe_mxfp4_code_at(qs, i)] * x[start + i];
        }
        acc += block_acc * scale;
        packed += block_bytes;
    }
    return acc;
}
