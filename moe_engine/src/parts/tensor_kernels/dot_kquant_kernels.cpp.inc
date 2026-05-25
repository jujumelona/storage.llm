// K-quant kernels
struct moe_q5_high_mask_table_t {
    uint8_t masks[256][8];
    moe_q5_high_mask_table_t() : masks{} {
        for (uint32_t m = 0; m < 256u; ++m) {
            for (uint32_t i = 0; i < 8u; ++i) {
                masks[m][i] = (m & (1u << i)) ? 16u : 0u;
            }
        }
    }
};

static const moe_q5_high_mask_table_t& moe_q5_high_mask_table() {
    static const moe_q5_high_mask_table_t table;
    return table;
}

#if defined(__AVX2__)
static inline __m256 moe_q5_1_values8_avx2(
    __m128i nibbles,
    const uint8_t* high_masks,
    __m256 vd,
    __m256 vm
) {
    const __m128i h = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(high_masks));
    const __m128i q = _mm_or_si128(nibbles, h);
    return _mm256_add_ps(
        _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(q)), vd),
        vm);
}
#endif

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
#if defined(__AVX2__)
    if ((cols % block_cols) == 0u) {
        const moe_q5_high_mask_table_t& kQ5HighMask = moe_q5_high_mask_table();
        const __m128i nibble_mask = _mm_set1_epi8(0x0f);
        for (uint32_t start = 0; start < cols; start += block_cols) {
            uint16_t d16 = 0;
            uint16_t m16 = 0;
            uint32_t qh = 0;
            std::memcpy(&d16, packed, sizeof(d16));
            std::memcpy(&m16, packed + 2u, sizeof(m16));
            std::memcpy(&qh, packed + 4u, sizeof(qh));
            const float d = moe_raw_fp16_to_f32(d16);
            const float m = moe_raw_fp16_to_f32(m16);
            if (!std::isfinite(d) || !std::isfinite(m)) {
                return std::numeric_limits<float>::quiet_NaN();
            }
            const __m256 vd = _mm256_set1_ps(d);
            const __m256 vm = _mm256_set1_ps(m);
            const uint8_t* qs = packed + 8u;
            __m256 vacc = _mm256_setzero_ps();

            const __m128i q0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qs));
            const __m256 v0 = moe_q5_1_values8_avx2(
                _mm_and_si128(q0, nibble_mask), kQ5HighMask.masks[(qh >> 0u) & 0xffu], vd, vm);
            vacc = moe_madd_ps(v0, _mm256_loadu_ps(x + start), vacc);

            const __m128i q1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qs + 8u));
            const __m256 v1 = moe_q5_1_values8_avx2(
                _mm_and_si128(q1, nibble_mask), kQ5HighMask.masks[(qh >> 8u) & 0xffu], vd, vm);
            vacc = moe_madd_ps(v1, _mm256_loadu_ps(x + start + 8u), vacc);

            const __m128i hq0 = _mm_and_si128(_mm_srli_epi16(q0, 4), nibble_mask);
            const __m256 v2 = moe_q5_1_values8_avx2(
                hq0, kQ5HighMask.masks[(qh >> 16u) & 0xffu], vd, vm);
            vacc = moe_madd_ps(v2, _mm256_loadu_ps(x + start + 16u), vacc);

            const __m128i hq1 = _mm_and_si128(_mm_srli_epi16(q1, 4), nibble_mask);
            const __m256 v3 = moe_q5_1_values8_avx2(
                hq1, kQ5HighMask.masks[(qh >> 24u) & 0xffu], vd, vm);
            vacc = moe_madd_ps(v3, _mm256_loadu_ps(x + start + 24u), vacc);

            acc += moe_dot_hsum256_ps(vacc);
            packed += block_bytes;
        }
        return acc;
    }
#endif
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

static int moe_dot_gguf_q5_1_two_rows(
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
#if defined(__AVX2__)
    const uint32_t block_cols = 32u;
    const uint32_t block_bytes = 24u;
    if ((cols % block_cols) == 0u) {
        const moe_q5_high_mask_table_t& kQ5HighMask = moe_q5_high_mask_table();
        const __m128i nibble_mask = _mm_set1_epi8(0x0f);
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint32_t start = 0; start < cols; start += block_cols) {
            uint16_t d16_0 = 0;
            uint16_t m16_0 = 0;
            uint32_t qh0 = 0;
            uint16_t d16_1 = 0;
            uint16_t m16_1 = 0;
            uint32_t qh1 = 0;
            std::memcpy(&d16_0, packed0, sizeof(d16_0));
            std::memcpy(&m16_0, packed0 + 2u, sizeof(m16_0));
            std::memcpy(&qh0, packed0 + 4u, sizeof(qh0));
            std::memcpy(&d16_1, packed1, sizeof(d16_1));
            std::memcpy(&m16_1, packed1 + 2u, sizeof(m16_1));
            std::memcpy(&qh1, packed1 + 4u, sizeof(qh1));
            const float d0 = moe_raw_fp16_to_f32(d16_0);
            const float m0 = moe_raw_fp16_to_f32(m16_0);
            const float d1 = moe_raw_fp16_to_f32(d16_1);
            const float m1 = moe_raw_fp16_to_f32(m16_1);
            if (!std::isfinite(d0) || !std::isfinite(m0) ||
                !std::isfinite(d1) || !std::isfinite(m1)) {
                return 0;
            }
            const __m256 vd0 = _mm256_set1_ps(d0);
            const __m256 vm0 = _mm256_set1_ps(m0);
            const __m256 vd1 = _mm256_set1_ps(d1);
            const __m256 vm1 = _mm256_set1_ps(m1);
            const uint8_t* qs0 = packed0 + 8u;
            const uint8_t* qs1 = packed1 + 8u;
            const __m128i q0a = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qs0));
            const __m128i q0b = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qs0 + 8u));
            const __m128i q1a = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qs1));
            const __m128i q1b = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qs1 + 8u));
            __m256 vacc0 = _mm256_setzero_ps();
            __m256 vacc1 = _mm256_setzero_ps();

            const __m256 x0 = _mm256_loadu_ps(x + start);
            const __m256 x1 = _mm256_loadu_ps(x + start + 8u);
            const __m256 x2 = _mm256_loadu_ps(x + start + 16u);
            const __m256 x3 = _mm256_loadu_ps(x + start + 24u);
            vacc0 = moe_madd_ps(
                moe_q5_1_values8_avx2(_mm_and_si128(q0a, nibble_mask), kQ5HighMask.masks[(qh0 >> 0u) & 0xffu], vd0, vm0),
                x0,
                vacc0);
            vacc1 = moe_madd_ps(
                moe_q5_1_values8_avx2(_mm_and_si128(q1a, nibble_mask), kQ5HighMask.masks[(qh1 >> 0u) & 0xffu], vd1, vm1),
                x0,
                vacc1);
            vacc0 = moe_madd_ps(
                moe_q5_1_values8_avx2(_mm_and_si128(q0b, nibble_mask), kQ5HighMask.masks[(qh0 >> 8u) & 0xffu], vd0, vm0),
                x1,
                vacc0);
            vacc1 = moe_madd_ps(
                moe_q5_1_values8_avx2(_mm_and_si128(q1b, nibble_mask), kQ5HighMask.masks[(qh1 >> 8u) & 0xffu], vd1, vm1),
                x1,
                vacc1);
            vacc0 = moe_madd_ps(
                moe_q5_1_values8_avx2(_mm_and_si128(_mm_srli_epi16(q0a, 4), nibble_mask), kQ5HighMask.masks[(qh0 >> 16u) & 0xffu], vd0, vm0),
                x2,
                vacc0);
            vacc1 = moe_madd_ps(
                moe_q5_1_values8_avx2(_mm_and_si128(_mm_srli_epi16(q1a, 4), nibble_mask), kQ5HighMask.masks[(qh1 >> 16u) & 0xffu], vd1, vm1),
                x2,
                vacc1);
            vacc0 = moe_madd_ps(
                moe_q5_1_values8_avx2(_mm_and_si128(_mm_srli_epi16(q0b, 4), nibble_mask), kQ5HighMask.masks[(qh0 >> 24u) & 0xffu], vd0, vm0),
                x3,
                vacc0);
            vacc1 = moe_madd_ps(
                moe_q5_1_values8_avx2(_mm_and_si128(_mm_srli_epi16(q1b, 4), nibble_mask), kQ5HighMask.masks[(qh1 >> 24u) & 0xffu], vd1, vm1),
                x3,
                vacc1);
            acc0 += moe_dot_hsum256_ps(vacc0);
            acc1 += moe_dot_hsum256_ps(vacc1);
            packed0 += block_bytes;
            packed1 += block_bytes;
        }
        *out0 = acc0;
        *out1 = acc1;
        return 1;
    }
#endif
    *out0 = moe_dot_gguf_q5_1_row(packed0, x, cols);
    *out1 = moe_dot_gguf_q5_1_row(packed1, x, cols);
    return std::isfinite(*out0) && std::isfinite(*out1) ? 1 : 0;
}

static float moe_gguf_q2_k_row_impl(const uint8_t* packed, const float* x, float* out, uint32_t cols) {
    // BUGFIX 702: Q2_K NaN handling ★★
    // Problem: No validation of scale/min values or accumulated result
    // Impact: NaN in quantization parameters propagates to PPL
    // Solution: Check for NaN/Inf in final result
    float acc = 0.0f;
    for (uint32_t base = 0; base < cols; base += 256u, packed += 84u) {
        const uint8_t* scales = packed;
        const uint8_t* q = packed + 16u;
        const float d = moe_load_f16le(packed + 80u);
        const float minv = moe_load_f16le(packed + 82u);
        // BUGFIX 702: Check scale validity
        if (!std::isfinite(d) || !std::isfinite(minv)) {
            continue;
        }
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
    // BUGFIX 702: Final NaN/Inf check
    if (!std::isfinite(acc)) {
        return 0.0f;
    }
    return acc;
}

static float moe_gguf_q3_k_row_impl(const uint8_t* packed, const float* x, float* out, uint32_t cols) {
    // BUGFIX 703: Q3_K NaN handling ★★
    // Problem: No validation of scale values or accumulated result
    // Impact: NaN in quantization parameters propagates to PPL
    // Solution: Check for NaN/Inf in scale and final result
    float acc = 0.0f;
    const uint32_t kmask1 = 0x03030303u;
    const uint32_t kmask2 = 0x0f0f0f0fu;
    for (uint32_t base = 0; base < cols; base += 256u, packed += 110u) {
        const uint8_t* hm = packed;
        const uint8_t* q = packed + 32u;
        const uint8_t* scale_bytes = packed + 96u;
        const float d_all = moe_load_f16le(packed + 108u);
        // BUGFIX 703: Check scale validity
        if (!std::isfinite(d_all)) {
            continue;
        }
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
    // BUGFIX 703: Final NaN/Inf check
    if (!std::isfinite(acc)) {
        return 0.0f;
    }
    return acc;
}

static float moe_gguf_q4_k_row_impl(const uint8_t* packed, const float* x, float* out, uint32_t cols) {
    // BUGFIX 704: Q4_K NaN handling ★★
    // Problem: No validation of scale/min values or accumulated result
    // Impact: NaN in quantization parameters propagates to PPL
    // Solution: Check for NaN/Inf in scale and final result
    float acc = 0.0f;
    for (uint32_t base = 0; base < cols; base += 256u, packed += 144u) {
        const float d = moe_load_f16le(packed);
        const float minv = moe_load_f16le(packed + 2u);
        // BUGFIX 704: Check scale validity
        if (!std::isfinite(d) || !std::isfinite(minv)) {
            continue;
        }
        const uint8_t* scales = packed + 4u;
        const uint8_t* q = packed + 16u;
        uint32_t is = 0;
#if defined(__AVX2__)
        if (x && !out && (cols % 256u) == 0u) {
            const __m128i nibble_mask = _mm_set1_epi8(0x0f);
            for (uint32_t j = 0; j < 256u; j += 64u) {
                uint8_t sc = 0;
                uint8_t mn = 0;
                moe_get_scale_min_k4(is + 0u, scales, &sc, &mn);
                const __m256 d1 = _mm256_set1_ps(d * (float)sc);
                const __m256 m1 = _mm256_set1_ps(minv * (float)mn);
                moe_get_scale_min_k4(is + 1u, scales, &sc, &mn);
                const __m256 d2 = _mm256_set1_ps(d * (float)sc);
                const __m256 m2 = _mm256_set1_ps(minv * (float)mn);
                __m256 vacc = _mm256_setzero_ps();
                for (uint32_t l = 0; l < 32u; l += 8u) {
                    const __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q + l));
                    const __m128i lo = _mm_and_si128(bytes, nibble_mask);
                    const __m128i hi = _mm_and_si128(_mm_srli_epi16(bytes, 4), nibble_mask);
                    const __m256 xlo = _mm256_loadu_ps(x + base + j + l);
                    const __m256 xhi = _mm256_loadu_ps(x + base + j + 32u + l);
                    const __m256 vlo = _mm256_sub_ps(
                        _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo)), d1), m1);
                    const __m256 vhi = _mm256_sub_ps(
                        _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi)), d2), m2);
                    vacc = moe_madd_ps(vlo, xlo, vacc);
                    vacc = moe_madd_ps(vhi, xhi, vacc);
                }
                acc += moe_dot_hsum256_ps(vacc);
                q += 32u;
                is += 2u;
            }
            continue;
        }
#endif
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
    // BUGFIX 704: Final NaN/Inf check
    if (!std::isfinite(acc)) {
        return 0.0f;
    }
    return acc;
}

static float moe_gguf_q5_k_row_impl(const uint8_t* packed, const float* x, float* out, uint32_t cols) {
    // BUGFIX 705: Q5_K NaN handling ★★
    // Problem: No validation of scale/min values or accumulated result
    // Impact: NaN in quantization parameters propagates to PPL
    // Solution: Check for NaN/Inf in scale and final result
    float acc = 0.0f;
    for (uint32_t base = 0; base < cols; base += 256u, packed += 176u) {
        const float d = moe_load_f16le(packed);
        const float minv = moe_load_f16le(packed + 2u);
        // BUGFIX 705: Check scale validity
        if (!std::isfinite(d) || !std::isfinite(minv)) {
            continue;
        }
        const uint8_t* scales = packed + 4u;
        const uint8_t* qh = packed + 16u;
        const uint8_t* ql = packed + 48u;
        uint32_t is = 0;
        uint8_t u1 = 1;
        uint8_t u2 = 2;
#if defined(__AVX2__)
        if (x && !out && (cols % 256u) == 0u) {
            const __m128i nibble_mask = _mm_set1_epi8(0x0f);
            const __m128i high_value = _mm_set1_epi8(16);
            const __m128i zero = _mm_setzero_si128();
            for (uint32_t j = 0; j < 256u; j += 64u) {
                uint8_t sc = 0;
                uint8_t mn = 0;
                moe_get_scale_min_k4(is + 0u, scales, &sc, &mn);
                const __m256 d1 = _mm256_set1_ps(d * (float)sc);
                const __m256 m1 = _mm256_set1_ps(minv * (float)mn);
                moe_get_scale_min_k4(is + 1u, scales, &sc, &mn);
                const __m256 d2 = _mm256_set1_ps(d * (float)sc);
                const __m256 m2 = _mm256_set1_ps(minv * (float)mn);
                const __m128i high_mask1 = _mm_set1_epi8((char)u1);
                const __m128i high_mask2 = _mm_set1_epi8((char)u2);
                __m256 vacc = _mm256_setzero_ps();
                for (uint32_t l = 0; l < 32u; l += 8u) {
                    const __m128i qbytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ql + l));
                    const __m128i hbytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qh + l));
                    const __m128i lo_hi = _mm_andnot_si128(
                        _mm_cmpeq_epi8(_mm_and_si128(hbytes, high_mask1), zero), high_value);
                    const __m128i hi_hi = _mm_andnot_si128(
                        _mm_cmpeq_epi8(_mm_and_si128(hbytes, high_mask2), zero), high_value);
                    const __m128i lo = _mm_or_si128(_mm_and_si128(qbytes, nibble_mask), lo_hi);
                    const __m128i hi = _mm_or_si128(
                        _mm_and_si128(_mm_srli_epi16(qbytes, 4), nibble_mask), hi_hi);
                    const __m256 xlo = _mm256_loadu_ps(x + base + j + l);
                    const __m256 xhi = _mm256_loadu_ps(x + base + j + 32u + l);
                    const __m256 vlo = _mm256_sub_ps(
                        _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo)), d1), m1);
                    const __m256 vhi = _mm256_sub_ps(
                        _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi)), d2), m2);
                    vacc = moe_madd_ps(vlo, xlo, vacc);
                    vacc = moe_madd_ps(vhi, xhi, vacc);
                }
                acc += moe_dot_hsum256_ps(vacc);
                ql += 32u;
                is += 2u;
                u1 = (uint8_t)(u1 << 2);
                u2 = (uint8_t)(u2 << 2);
            }
            continue;
        }
#endif
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
    // BUGFIX 705: Final NaN/Inf check
    if (!std::isfinite(acc)) {
        return 0.0f;
    }
    return acc;
}

static float moe_gguf_q6_k_row_impl(const uint8_t* packed, const float* x, float* out, uint32_t cols) {
    // BUGFIX 706: Q6_K NaN handling ★★
    // Problem: No validation of scale values or accumulated result
    // Impact: NaN in quantization parameters propagates to PPL
    // Solution: Check for NaN/Inf in scale and final result
    float acc = 0.0f;
    for (uint32_t base = 0; base < cols; base += 256u, packed += 210u) {
        const uint8_t* ql = packed;
        const uint8_t* qh = packed + 128u;
        const int8_t* sc = reinterpret_cast<const int8_t*>(packed + 192u);
        const float d = moe_load_f16le(packed + 208u);
        // BUGFIX 706: Check scale validity
        if (!std::isfinite(d)) {
            continue;
        }
#if defined(__AVX2__)
        if (x && !out && (cols % 256u) == 0u) {
            const __m128i nibble_mask = _mm_set1_epi8(0x0f);
            const __m128i two_bit_mask = _mm_set1_epi8(0x03);
            const __m256 bias = _mm256_set1_ps(32.0f);
            auto q6_to_ps = [&](const __m128i values) -> __m256 {
                return _mm256_sub_ps(
                    _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(values)),
                    bias);
            };
            for (uint32_t n = 0; n < 256u; n += 128u) {
                for (uint32_t l = 0; l < 32u; l += 8u) {
                    const uint32_t is = l / 16u;
                    const __m128i qlo = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ql + l));
                    const __m128i qhi = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ql + 32u + l));
                    const __m128i h = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qh + l));
                    const __m128i h0 = _mm_slli_epi16(_mm_and_si128(h, two_bit_mask), 4);
                    const __m128i h1 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(h, 2), two_bit_mask), 4);
                    const __m128i h2 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(h, 4), two_bit_mask), 4);
                    const __m128i h3 = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(h, 6), two_bit_mask), 4);
                    const __m256 q1 = q6_to_ps(_mm_or_si128(_mm_and_si128(qlo, nibble_mask), h0));
                    const __m256 q2 = q6_to_ps(_mm_or_si128(_mm_and_si128(qhi, nibble_mask), h1));
                    const __m256 q3 = q6_to_ps(_mm_or_si128(
                        _mm_and_si128(_mm_srli_epi16(qlo, 4), nibble_mask), h2));
                    const __m256 q4 = q6_to_ps(_mm_or_si128(
                        _mm_and_si128(_mm_srli_epi16(qhi, 4), nibble_mask), h3));
                    __m256 vacc = _mm256_setzero_ps();
                    vacc = moe_madd_ps(
                        _mm256_mul_ps(_mm256_set1_ps(d * (float)sc[is + 0u]), q1),
                        _mm256_loadu_ps(x + base + n + l), vacc);
                    vacc = moe_madd_ps(
                        _mm256_mul_ps(_mm256_set1_ps(d * (float)sc[is + 2u]), q2),
                        _mm256_loadu_ps(x + base + n + 32u + l), vacc);
                    vacc = moe_madd_ps(
                        _mm256_mul_ps(_mm256_set1_ps(d * (float)sc[is + 4u]), q3),
                        _mm256_loadu_ps(x + base + n + 64u + l), vacc);
                    vacc = moe_madd_ps(
                        _mm256_mul_ps(_mm256_set1_ps(d * (float)sc[is + 6u]), q4),
                        _mm256_loadu_ps(x + base + n + 96u + l), vacc);
                    acc += moe_dot_hsum256_ps(vacc);
                }
                ql += 64u;
                qh += 32u;
                sc += 8u;
            }
            continue;
        }
#endif
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
    // BUGFIX 706: Final NaN/Inf check
    if (!std::isfinite(acc)) {
        return 0.0f;
    }
    return acc;
}

static float moe_gguf_iq2_xxs_row_impl(const uint8_t* packed, const float* x, float* out, uint32_t cols) {
    // BUGFIX 707: IQ2_XXS NaN handling ★★
    // Problem: No validation of scale values or accumulated result
    // Impact: NaN in quantization parameters propagates to PPL
    // Solution: Check for NaN/Inf in scale and final result
    float acc = 0.0f;
    for (uint32_t base = 0; base < cols; base += 256u, packed += 66u) {
        const float d = moe_load_f16le(packed);
        // BUGFIX 707: Check scale validity
        if (!std::isfinite(d)) {
            continue;
        }
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
    // BUGFIX 707: Final NaN/Inf check
    if (!std::isfinite(acc)) {
        return 0.0f;
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
    // BUGFIX 708: IQ1_S NaN handling ★★
    // Problem: No validation of scale values or accumulated result
    // Impact: NaN in quantization parameters propagates to PPL
    // Solution: Check for NaN/Inf in scale and final result
    float acc = 0.0f;
    constexpr float delta_value = 0.125f;
    for (uint32_t base = 0; base < cols; base += 256u, packed += 50u) {
        const float d = moe_load_f16le(packed);
        // BUGFIX 708: Check scale validity
        if (!std::isfinite(d)) {
            continue;
        }
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
    // BUGFIX 708: Final NaN/Inf check
    if (!std::isfinite(acc)) {
        return 0.0f;
    }
    return acc;
}

static float moe_gguf_iq1_m_row_impl(const uint8_t* packed, const float* x, float* out, uint32_t cols) {
    // BUGFIX 709: IQ1_M NaN handling ★★
    // Problem: No validation of scale values or accumulated result
    // Impact: NaN in quantization parameters propagates to PPL
    // Solution: Check for NaN/Inf in scale and final result
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
        // BUGFIX 709: Check scale validity
        if (!std::isfinite(d)) {
            continue;
        }
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
    // BUGFIX 709: Final NaN/Inf check
    if (!std::isfinite(acc)) {
        return 0.0f;
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
