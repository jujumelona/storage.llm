// Q8 activation preparation shared by integer CPU GEMV kernels.

struct moe_q8_0_activation_block_t {
    float d = 0.0f;
    int8_t qs[32]{};
};

struct moe_q8_1_activation_block_t {
    float d = 0.0f;
    float s = 0.0f;
    int8_t qs[32]{};
};

#if defined(__AVX2__)
static inline float moe_q8_activation_amax_32_avx2(const float* src) {
    const __m256i abs_mask = _mm256_set1_epi32(0x7fffffff);
    __m256 vmax = _mm256_and_ps(_mm256_loadu_ps(src), _mm256_castsi256_ps(abs_mask));
    vmax = _mm256_max_ps(vmax, _mm256_and_ps(_mm256_loadu_ps(src + 8u), _mm256_castsi256_ps(abs_mask)));
    vmax = _mm256_max_ps(vmax, _mm256_and_ps(_mm256_loadu_ps(src + 16u), _mm256_castsi256_ps(abs_mask)));
    vmax = _mm256_max_ps(vmax, _mm256_and_ps(_mm256_loadu_ps(src + 24u), _mm256_castsi256_ps(abs_mask)));
    vmax = _mm256_max_ps(vmax, _mm256_permute2f128_ps(vmax, vmax, 1));
    vmax = _mm256_max_ps(vmax, _mm256_permute_ps(vmax, 78));
    vmax = _mm256_max_ps(vmax, _mm256_permute_ps(vmax, 177));
    return _mm_cvtss_f32(_mm256_castps256_ps128(vmax));
}

static inline int32_t moe_q8_activation_quantize_32_avx2(
    const float* src,
    float id,
    int8_t* dst,
    int32_t* out_sum
) {
    const __m256 vscale = _mm256_set1_ps(id);
    const __m256i vmin = _mm256_set1_epi32(-127);
    const __m256i vmax = _mm256_set1_epi32(127);
    int32_t sum = 0;
    for (uint32_t i = 0; i < 32u; i += 8u) {
        __m256i q = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(src + i), vscale));
        q = _mm256_max_epi32(q, vmin);
        q = _mm256_min_epi32(q, vmax);
        const __m128i q16 = _mm_packs_epi32(
            _mm256_castsi256_si128(q),
            _mm256_extracti128_si256(q, 1));
        const __m128i q8 = _mm_packs_epi16(q16, q16);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + i), q8);
        if (out_sum) {
            __m128i partial = _mm_add_epi32(
                _mm256_castsi256_si128(q),
                _mm256_extracti128_si256(q, 1));
            partial = _mm_hadd_epi32(partial, partial);
            partial = _mm_hadd_epi32(partial, partial);
            sum += _mm_cvtsi128_si32(partial);
        }
    }
    if (out_sum) {
        *out_sum = sum;
    }
    return sum;
}
#endif

static int moe_quantize_activation_q8_0_into(
    const float* x,
    uint32_t cols,
    moe_q8_0_activation_block_t* out
) {
    if (!x || !out || cols == 0 || (cols & 31u) != 0u) {
        return 0;
    }
    const uint32_t blocks = cols >> 5u;
    for (uint32_t b = 0; b < blocks; ++b) {
        moe_q8_0_activation_block_t& dst = out[b];
        const float* src = x + (size_t)b * 32u;
        float amax = 0.0f;
#if defined(__AVX2__)
        amax = moe_q8_activation_amax_32_avx2(src);
#else
        for (uint32_t i = 0; i < 32u; ++i) {
            amax = std::max(amax, std::fabs(src[i]));
        }
#endif
        const float d = amax > 0.0f ? amax / 127.0f : 0.0f;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;
        dst.d = d;
#if defined(__AVX2__)
        (void)moe_q8_activation_quantize_32_avx2(src, id, dst.qs, nullptr);
#else
        for (uint32_t i = 0; i < 32u; ++i) {
            const int q = (int)std::nearbyint(src[i] * id);
            dst.qs[i] = (int8_t)std::max(-127, std::min(127, q));
        }
#endif
    }
    return 1;
}

static int moe_quantize_activation_q8_0(
    const float* x,
    uint32_t cols,
    std::vector<moe_q8_0_activation_block_t>* out
) {
    if (!out || cols == 0 || (cols & 31u) != 0u) {
        return 0;
    }
    out->resize(cols >> 5u);
    return moe_quantize_activation_q8_0_into(x, cols, out->data());
}

static int moe_quantize_activation_q8_1_into(
    const float* x,
    uint32_t cols,
    moe_q8_1_activation_block_t* out
) {
    if (!x || !out || cols == 0 || (cols & 31u) != 0u) {
        return 0;
    }
    const uint32_t blocks = cols >> 5u;
    for (uint32_t b = 0; b < blocks; ++b) {
        moe_q8_1_activation_block_t& dst = out[b];
        const float* src = x + (size_t)b * 32u;
        float amax = 0.0f;
#if defined(__AVX2__)
        amax = moe_q8_activation_amax_32_avx2(src);
#else
        for (uint32_t i = 0; i < 32u; ++i) {
            amax = std::max(amax, std::fabs(src[i]));
        }
#endif
        const float d = amax > 0.0f ? amax / 127.0f : 0.0f;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;
        int32_t sum = 0;
        dst.d = d;
#if defined(__AVX2__)
        sum = moe_q8_activation_quantize_32_avx2(src, id, dst.qs, &sum);
#else
        for (uint32_t i = 0; i < 32u; ++i) {
            const int q = std::max(-127, std::min(127, (int)std::nearbyint(src[i] * id)));
            dst.qs[i] = (int8_t)q;
            sum += q;
        }
#endif
        dst.s = d * (float)sum;
    }
    return 1;
}

#if defined(__AVX2__)
static inline __m256i moe_mul_sum_i8_pairs_acc_i32x8(
    __m256i acc,
    __m256i weights,
    __m256i activations
) {
    const __m256i abs_weights = _mm256_sign_epi8(weights, weights);
    const __m256i signed_activations = _mm256_sign_epi8(activations, weights);
    const __m256i pair_sums = _mm256_maddubs_epi16(abs_weights, signed_activations);
    return _mm256_add_epi32(acc, _mm256_madd_epi16(pair_sums, _mm256_set1_epi16(1)));
}

static inline __m256 moe_mul_sum_i8_pairs_f32x8(
    __m256i weights,
    __m256i activations
) {
    return _mm256_cvtepi32_ps(moe_mul_sum_i8_pairs_acc_i32x8(
        _mm256_setzero_si256(), weights, activations));
}
#endif

#if defined(__AVX2__) && defined(__F16C__)
static inline int moe_q8_0_q8_0_four_row_scales_avx2(
    const uint8_t* packed0,
    const uint8_t* packed1,
    const uint8_t* packed2,
    const uint8_t* packed3,
    float x_scale,
    float* scales
) {
    uint16_t d0 = 0;
    uint16_t d1 = 0;
    uint16_t d2 = 0;
    uint16_t d3 = 0;
    std::memcpy(&d0, packed0, sizeof(d0));
    std::memcpy(&d1, packed1, sizeof(d1));
    std::memcpy(&d2, packed2, sizeof(d2));
    std::memcpy(&d3, packed3, sizeof(d3));
    const __m128 scale_values = _mm_mul_ps(
        _mm_cvtph_ps(_mm_setr_epi16(
            (int16_t)d0, (int16_t)d1, (int16_t)d2, (int16_t)d3,
            0, 0, 0, 0)),
        _mm_set1_ps(x_scale));
    if (1) {
        _mm_storeu_ps(scales, scale_values);
        return 1;
    }
    const __m128 abs_scale_values = _mm_and_ps(
        scale_values, _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff)));
    const __m128 finite_scale_values = _mm_cmp_ps(
        abs_scale_values,
        _mm_set1_ps(std::numeric_limits<float>::max()),
        _CMP_LE_OQ);
    if (_mm_movemask_ps(finite_scale_values) != 0x0f) {
        return 0;
    }
    _mm_storeu_ps(scales, scale_values);
    return 1;
}
#endif

static float moe_dot_gguf_q8_0_q8_0_row(
    const uint8_t* packed,
    const moe_q8_0_activation_block_t* xq,
    uint32_t cols
) {
    if (!packed || !xq || cols == 0 || (cols & 31u) != 0u) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    float acc = 0.0f;
    const uint32_t blocks = cols >> 5u;
#if defined(__AVX2__)
    __m256 vacc = _mm256_setzero_ps();
#endif
    for (uint32_t b = 0; b < blocks; ++b, packed += 34u) {
        uint16_t d16 = 0;
        std::memcpy(&d16, packed, sizeof(d16));
        const float scale = moe_raw_fp16_to_f32(d16) * xq[b].d;
        if (!std::isfinite(scale)) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        const int8_t* weights = reinterpret_cast<const int8_t*>(packed + 2u);
#if defined(__AVX2__)
        const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights));
        const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq[b].qs));
        vacc = moe_madd_ps(_mm256_set1_ps(scale), moe_mul_sum_i8_pairs_f32x8(w, x), vacc);
#else
        int32_t sum = 0;
        for (uint32_t i = 0; i < 32u; ++i) {
            sum += (int32_t)weights[i] * (int32_t)xq[b].qs[i];
        }
        acc += scale * (float)sum;
#endif
    }
#if defined(__AVX2__)
    acc += moe_dot_hsum256_ps(vacc);
#endif
    return acc;
}

static int moe_dot_gguf_q8_0_q8_0_four_rows(
    const uint8_t* packed0,
    const uint8_t* packed1,
    const uint8_t* packed2,
    const uint8_t* packed3,
    const moe_q8_0_activation_block_t* xq,
    uint32_t cols,
    float* out0,
    float* out1,
    float* out2,
    float* out3
) {
    if (!packed0 || !packed1 || !packed2 || !packed3 || !xq ||
        !out0 || !out1 || !out2 || !out3 || cols == 0 || (cols & 31u) != 0u) {
        return 0;
    }
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    const uint32_t blocks = cols >> 5u;
#if defined(__AVX2__)
    __m256 vacc0 = _mm256_setzero_ps();
    __m256 vacc1 = _mm256_setzero_ps();
    __m256 vacc2 = _mm256_setzero_ps();
    __m256 vacc3 = _mm256_setzero_ps();
#endif
    for (uint32_t b = 0; b < blocks; ++b) {
#if defined(__AVX2__) && defined(__F16C__)
        float scales[4];
        if (!moe_q8_0_q8_0_four_row_scales_avx2(
                packed0, packed1, packed2, packed3, xq[b].d, scales)) {
            return 0;
        }
        const float scale0 = scales[0];
        const float scale1 = scales[1];
        const float scale2 = scales[2];
        const float scale3 = scales[3];
#else
        uint16_t d16_0 = 0;
        uint16_t d16_1 = 0;
        uint16_t d16_2 = 0;
        uint16_t d16_3 = 0;
        std::memcpy(&d16_0, packed0, sizeof(d16_0));
        std::memcpy(&d16_1, packed1, sizeof(d16_1));
        std::memcpy(&d16_2, packed2, sizeof(d16_2));
        std::memcpy(&d16_3, packed3, sizeof(d16_3));
        const float x_scale = xq[b].d;
        const float scale0 = moe_raw_fp16_to_f32(d16_0) * x_scale;
        const float scale1 = moe_raw_fp16_to_f32(d16_1) * x_scale;
        const float scale2 = moe_raw_fp16_to_f32(d16_2) * x_scale;
        const float scale3 = moe_raw_fp16_to_f32(d16_3) * x_scale;
        if (!std::isfinite(scale0) || !std::isfinite(scale1) ||
            !std::isfinite(scale2) || !std::isfinite(scale3)) {
            return 0;
        }
#endif
#if defined(__AVX2__)
        const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq[b].qs));
        vacc0 = moe_madd_ps(
            _mm256_set1_ps(scale0),
            moe_mul_sum_i8_pairs_f32x8(
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed0 + 2u)), x),
            vacc0);
        vacc1 = moe_madd_ps(
            _mm256_set1_ps(scale1),
            moe_mul_sum_i8_pairs_f32x8(
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed1 + 2u)), x),
            vacc1);
        vacc2 = moe_madd_ps(
            _mm256_set1_ps(scale2),
            moe_mul_sum_i8_pairs_f32x8(
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed2 + 2u)), x),
            vacc2);
        vacc3 = moe_madd_ps(
            _mm256_set1_ps(scale3),
            moe_mul_sum_i8_pairs_f32x8(
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed3 + 2u)), x),
            vacc3);
#else
        const int8_t* w0 = reinterpret_cast<const int8_t*>(packed0 + 2u);
        const int8_t* w1 = reinterpret_cast<const int8_t*>(packed1 + 2u);
        const int8_t* w2 = reinterpret_cast<const int8_t*>(packed2 + 2u);
        const int8_t* w3 = reinterpret_cast<const int8_t*>(packed3 + 2u);
        int32_t sum0 = 0;
        int32_t sum1 = 0;
        int32_t sum2 = 0;
        int32_t sum3 = 0;
        for (uint32_t i = 0; i < 32u; ++i) {
            const int32_t x = (int32_t)xq[b].qs[i];
            sum0 += (int32_t)w0[i] * x;
            sum1 += (int32_t)w1[i] * x;
            sum2 += (int32_t)w2[i] * x;
            sum3 += (int32_t)w3[i] * x;
        }
        acc0 += scale0 * (float)sum0;
        acc1 += scale1 * (float)sum1;
        acc2 += scale2 * (float)sum2;
        acc3 += scale3 * (float)sum3;
#endif
        packed0 += 34u;
        packed1 += 34u;
        packed2 += 34u;
        packed3 += 34u;
    }
#if defined(__AVX2__)
    acc0 += moe_dot_hsum256_ps(vacc0);
    acc1 += moe_dot_hsum256_ps(vacc1);
    acc2 += moe_dot_hsum256_ps(vacc2);
    acc3 += moe_dot_hsum256_ps(vacc3);
#endif
    *out0 = acc0;
    *out1 = acc1;
    *out2 = acc2;
    *out3 = acc3;
    return 1;
}

static int moe_dot_gguf_q8_0_q8_0_eight_rows(
    const uint8_t* packed0,
    const uint8_t* packed1,
    const uint8_t* packed2,
    const uint8_t* packed3,
    const uint8_t* packed4,
    const uint8_t* packed5,
    const uint8_t* packed6,
    const uint8_t* packed7,
    const moe_q8_0_activation_block_t* xq,
    uint32_t cols,
    float* out
) {
    if (!packed0 || !packed1 || !packed2 || !packed3 ||
        !packed4 || !packed5 || !packed6 || !packed7 ||
        !xq || !out || cols == 0 || (cols & 31u) != 0u) {
        return 0;
    }
    const uint8_t* packed[8] = {
        packed0, packed1, packed2, packed3,
        packed4, packed5, packed6, packed7,
    };
    float acc[8]{};
    const uint32_t blocks = cols >> 5u;
#if defined(__AVX2__)
    __m256 vacc[8] = {
        _mm256_setzero_ps(), _mm256_setzero_ps(),
        _mm256_setzero_ps(), _mm256_setzero_ps(),
        _mm256_setzero_ps(), _mm256_setzero_ps(),
        _mm256_setzero_ps(), _mm256_setzero_ps(),
    };
#endif
    for (uint32_t b = 0; b < blocks; ++b) {
        const float x_scale = xq[b].d;
#if defined(__AVX2__)
        const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq[b].qs));
#endif
#if defined(__AVX2__) && defined(__F16C__)
        float scales[8];
        if (!moe_q8_0_eight_row_scales_avx2(packed, scales)) {
            return 0;
        }
        const __m256 scale_values = _mm256_mul_ps(
            _mm256_loadu_ps(scales), _mm256_set1_ps(x_scale));
        if (0) {
            const __m256 abs_scale_values = _mm256_and_ps(
                scale_values, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)));
            const __m256 finite_scale_values = _mm256_cmp_ps(
                abs_scale_values,
                _mm256_set1_ps(std::numeric_limits<float>::max()),
                _CMP_LE_OQ);
            if (_mm256_movemask_ps(finite_scale_values) != 0xff) {
                return 0;
            }
        }
        _mm256_storeu_ps(scales, scale_values);
#endif
        for (uint32_t r = 0; r < 8u; ++r) {
#if defined(__AVX2__) && defined(__F16C__)
            const float scale = scales[r];
#else
            uint16_t d16 = 0;
            std::memcpy(&d16, packed[r], sizeof(d16));
            const float scale = moe_raw_fp16_to_f32(d16) * x_scale;
            if (!std::isfinite(scale)) {
                return 0;
            }
#endif
#if defined(__AVX2__)
            vacc[r] = moe_madd_ps(
                _mm256_set1_ps(scale),
                moe_mul_sum_i8_pairs_f32x8(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed[r] + 2u)), x),
                vacc[r]);
#else
            const int8_t* weights = reinterpret_cast<const int8_t*>(packed[r] + 2u);
            int32_t sum = 0;
            for (uint32_t i = 0; i < 32u; ++i) {
                sum += (int32_t)weights[i] * (int32_t)xq[b].qs[i];
            }
            acc[r] += scale * (float)sum;
#endif
            packed[r] += 34u;
        }
    }
#if defined(__AVX2__)
    for (uint32_t r = 0; r < 8u; ++r) {
        acc[r] += moe_dot_hsum256_ps(vacc[r]);
    }
#endif
    for (uint32_t r = 0; r < 8u; ++r) {
        out[r] = acc[r];
        if (0 && !std::isfinite(out[r])) {
            return 0;
        }
    }
    return 1;
}
