// SSD-native MXFP4 8x8 tiles. The artifact stores eight source rows
// interleaved once during materialization, so decode never repacks weights.

static uint64_t moe_mxfp4_8x8_group_bytes(uint32_t cols) {
    return ((uint64_t)cols + 31ull) / 32ull * 136ull;
}

static inline int moe_mxfp4_8x8_all_finite(const float* values) {
    if (!values) {
        return 0;
    }
#if defined(__AVX2__)
    const __m256 v = _mm256_loadu_ps(values);
    const __m256 abs_v = _mm256_and_ps(
        v, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)));
    const __m256 finite = _mm256_cmp_ps(
        abs_v, _mm256_set1_ps(std::numeric_limits<float>::max()), _CMP_LE_OQ);
    return _mm256_movemask_ps(finite) == 0xff;
#else
    for (uint32_t i = 0; i < 8u; ++i) {
        if (!std::isfinite(values[i])) {
            return 0;
        }
    }
    return 1;
#endif
}

#if defined(__AVX2__)
static inline __m256 moe_mxfp4_8x8_scales_ps(const uint8_t* tile) {
    const __m128i order = _mm_setr_epi8(
        0, 4, 1, 5, 2, 6, 3, 7,
        -1, -1, -1, -1, -1, -1, -1, -1);
    const __m128i scale_bytes = _mm_shuffle_epi8(
        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(tile)), order);
    const __m256i scale_indices = _mm256_cvtepu8_epi32(scale_bytes);
    const __m256i normal_bits = _mm256_slli_epi32(
        _mm256_sub_epi32(scale_indices, _mm256_set1_epi32(1)), 23);
    const __m256i tiny_bits = _mm256_sllv_epi32(
        _mm256_set1_epi32(0x00200000), scale_indices);
    const __m256i tiny_mask = _mm256_cmpgt_epi32(
        _mm256_set1_epi32(2), scale_indices);
    return _mm256_castsi256_ps(
        _mm256_blendv_epi8(normal_bits, tiny_bits, tiny_mask));
}
#endif

template <bool HasSecond>
static void moe_dot_gguf_mxfp4_8x8_q8_tile_pair_impl(
    const uint8_t* tile0,
    const uint8_t* tile1,
    const moe_q8_0_activation_block_t* xq,
    uint32_t cols,
    float* out0,
    float* out1
) {
    if (!tile0 || !xq || !out0 || (HasSecond && (!tile1 || !out1)) ||
        cols == 0 || (cols & 31u) != 0u) {
        if (out0) {
            std::fill(out0, out0 + 8u, 0.0f);
        }
        if (out1) {
            std::fill(out1, out1 + 8u, 0.0f);
        }
        return;
    }
    const uint32_t blocks = cols >> 5u;
    static constexpr int8_t kValues[16] = {
        0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
    };
#if defined(__AVX2__)
    const __m256i lut = _mm256_setr_epi8(
        0, 1, 2, 3, 4, 6, 8, 12,
        0, -1, -2, -3, -4, -6, -8, -12,
        0, 1, 2, 3, 4, 6, 8, 12,
        0, -1, -2, -3, -4, -6, -8, -12);
    const __m256i mask = _mm256_set1_epi8(0x0f);
    const __m256i final_permute = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
#else
    std::fill(out0, out0 + 8u, 0.0f);
    if constexpr (HasSecond) {
        std::fill(out1, out1 + 8u, 0.0f);
    }
#endif
    for (uint32_t b = 0; b < blocks; ++b, tile0 += 136u) {
        const moe_q8_0_activation_block_t& xb = xq[b];
#if defined(__AVX2__)
        __m256i q0 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(xb.qs)));
        __m256i q1 = _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(xb.qs + 16u)));
        q0 = _mm256_permute2f128_si256(q0, q0, 0);
        q1 = _mm256_permute2f128_si256(q1, q1, 0);
        const __m256i q0_0 = _mm256_shuffle_epi32(q0, 0);
        const __m256i q0_1 = _mm256_shuffle_epi32(q0, 85);
        const __m256i q0_2 = _mm256_shuffle_epi32(q0, 170);
        const __m256i q0_3 = _mm256_shuffle_epi32(q0, 255);
        const __m256i q1_0 = _mm256_shuffle_epi32(q1, 0);
        const __m256i q1_1 = _mm256_shuffle_epi32(q1, 85);
        const __m256i q1_2 = _mm256_shuffle_epi32(q1, 170);
        const __m256i q1_3 = _mm256_shuffle_epi32(q1, 255);
        const __m256 x_scale = _mm256_set1_ps(xb.d * 0.5f);
        auto accumulate_tile = [&](const uint8_t* tile, __m256 acc) -> __m256 {
            const uint8_t* qs = tile + 8u;
            const __m256i raw_0123_0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs));
            const __m256i raw_4567_0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs + 32u));
            const __m256i raw_0123_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs + 64u));
            const __m256i raw_4567_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs + 96u));
            const __m256i rhs_0123_0 = _mm256_shuffle_epi8(lut, _mm256_and_si256(raw_0123_0, mask));
            const __m256i rhs_4567_0 = _mm256_shuffle_epi8(lut, _mm256_and_si256(raw_4567_0, mask));
            const __m256i rhs_0123_1 = _mm256_shuffle_epi8(lut, _mm256_and_si256(raw_0123_1, mask));
            const __m256i rhs_4567_1 = _mm256_shuffle_epi8(lut, _mm256_and_si256(raw_4567_1, mask));
            const __m256i rhs_0123_2 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(raw_0123_0, 4), mask));
            const __m256i rhs_4567_2 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(raw_4567_0, 4), mask));
            const __m256i rhs_0123_3 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(raw_0123_1, 4), mask));
            const __m256i rhs_4567_3 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(raw_4567_1, 4), mask));
            // Keep two independent madd chains so Zen-class AVX2 cores can
            // overlap the integer work without increasing register pressure
            // enough to spill the unpacked tile values.
            __m256i isum0 = _mm256_setzero_si256();
            __m256i isum1 = _mm256_setzero_si256();
            isum0 = moe_mul_sum_i8_pairs_acc_i32x8(
                isum0, _mm256_blend_epi32(rhs_0123_0, _mm256_shuffle_epi32(rhs_4567_0, 177), 170), q0_0);
            isum1 = moe_mul_sum_i8_pairs_acc_i32x8(
                isum1, _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_0123_0, 177), rhs_4567_0, 170), q0_1);
            isum0 = moe_mul_sum_i8_pairs_acc_i32x8(
                isum0, _mm256_blend_epi32(rhs_0123_1, _mm256_shuffle_epi32(rhs_4567_1, 177), 170), q0_2);
            isum1 = moe_mul_sum_i8_pairs_acc_i32x8(
                isum1, _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_0123_1, 177), rhs_4567_1, 170), q0_3);
            isum0 = moe_mul_sum_i8_pairs_acc_i32x8(
                isum0, _mm256_blend_epi32(rhs_0123_2, _mm256_shuffle_epi32(rhs_4567_2, 177), 170), q1_0);
            isum1 = moe_mul_sum_i8_pairs_acc_i32x8(
                isum1, _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_0123_2, 177), rhs_4567_2, 170), q1_1);
            isum0 = moe_mul_sum_i8_pairs_acc_i32x8(
                isum0, _mm256_blend_epi32(rhs_0123_3, _mm256_shuffle_epi32(rhs_4567_3, 177), 170), q1_2);
            isum1 = moe_mul_sum_i8_pairs_acc_i32x8(
                isum1, _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_0123_3, 177), rhs_4567_3, 170), q1_3);
            const __m256i isum = _mm256_add_epi32(isum0, isum1);
            const __m256 scales = moe_mxfp4_8x8_scales_ps(tile);
            return _mm256_add_ps(acc, _mm256_mul_ps(
                _mm256_cvtepi32_ps(isum),
                _mm256_mul_ps(scales, x_scale)));
        };
        acc0 = accumulate_tile(tile0, acc0);
        if constexpr (HasSecond) {
            acc1 = accumulate_tile(tile1, acc1);
            tile1 += 136u;
        }
#else
        const uint8_t* tiles[2] = {tile0, tile1};
        float* outputs[2] = {out0, out1};
        for (uint32_t t = 0; t < (HasSecond ? 2u : 1u); ++t) {
            const uint8_t* qs = tiles[t] + 8u;
            for (uint32_t r = 0; r < 8u; ++r) {
                int32_t isum = 0;
                for (uint32_t i = 0; i < 16u; ++i) {
                    const uint8_t byte = qs[(i < 8u ? r * 8u : 64u + r * 8u) + (i & 7u)];
                    const int lo = (int)kValues[byte & 0x0fu];
                    const int hi = (int)kValues[byte >> 4];
                    isum += lo * (int)xb.qs[i] + hi * (int)xb.qs[i + 16u];
                }
                outputs[t][r] += (float)isum * xb.d * 0.5f * moe_e8m0_to_f32_half(tiles[t][r]);
            }
        }
        if constexpr (HasSecond) {
            tile1 += 136u;
        }
#endif
    }
#if defined(__AVX2__)
    acc0 = _mm256_permutevar8x32_ps(acc0, final_permute);
    _mm256_storeu_ps(out0, acc0);
    if constexpr (HasSecond) {
        acc1 = _mm256_permutevar8x32_ps(acc1, final_permute);
        _mm256_storeu_ps(out1, acc1);
    }
#endif
}

static void moe_dot_gguf_mxfp4_8x8_q8_tile_pair(
    const uint8_t* tile0,
    const uint8_t* tile1,
    const moe_q8_0_activation_block_t* xq,
    uint32_t cols,
    float* out0,
    float* out1
) {
    if ((tile1 != nullptr) != (out1 != nullptr)) {
        if (out0) {
            std::fill(out0, out0 + 8u, 0.0f);
        }
        if (out1) {
            std::fill(out1, out1 + 8u, 0.0f);
        }
        return;
    }
    if (tile1) {
        moe_dot_gguf_mxfp4_8x8_q8_tile_pair_impl<true>(tile0, tile1, xq, cols, out0, out1);
    } else {
        moe_dot_gguf_mxfp4_8x8_q8_tile_pair_impl<false>(tile0, nullptr, xq, cols, out0, nullptr);
    }
}

static void moe_dot_gguf_mxfp4_8x8_q8_rows(
    const uint8_t* tile,
    const moe_q8_0_activation_block_t* xq,
    uint32_t cols,
    float* out8
) {
    moe_dot_gguf_mxfp4_8x8_q8_tile_pair(tile, nullptr, xq, cols, out8, nullptr);
}
