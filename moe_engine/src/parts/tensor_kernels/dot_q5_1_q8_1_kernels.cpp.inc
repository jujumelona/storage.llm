// Q5_1 weights against one Q8_1 activation row. The activation is quantized
// once per selected expert, then reused across all down-projection output rows.

#if defined(__AVX2__)
static inline int moe_q5_1_8x8_all_finite(const float* values) {
    if (!values) {
        return 0;
    }
    const __m256 v = _mm256_loadu_ps(values);
    const __m256 abs_v = _mm256_and_ps(
        v, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)));
    const __m256 finite = _mm256_cmp_ps(
        abs_v, _mm256_set1_ps(std::numeric_limits<float>::max()), _CMP_LE_OQ);
    return _mm256_movemask_ps(finite) == 0xff;
}

static inline __m256i moe_q5_1_unpack_u5x32_avx2(const uint8_t* packed) {
    uint32_t qh = 0;
    std::memcpy(&qh, packed + 4u, sizeof(qh));
    const __m128i nibble_mask = _mm_set1_epi8(0x0f);
    const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed + 8u));
    const __m256i low_nibbles = _mm256_inserti128_si256(
        _mm256_castsi128_si256(_mm_and_si128(raw, nibble_mask)),
        _mm_and_si128(_mm_srli_epi16(raw, 4), nibble_mask),
        1);
    const __m256i qh_shuffle = _mm256_set_epi64x(
        0x0303030303030303ll,
        0x0202020202020202ll,
        0x0101010101010101ll,
        0x0000000000000000ll);
    __m256i high_bits = _mm256_shuffle_epi8(_mm256_set1_epi32((int)qh), qh_shuffle);
    high_bits = _mm256_or_si256(
        high_bits,
        _mm256_set1_epi64x(0x7fbfdfeff7fbfdfeLL));
    high_bits = _mm256_and_si256(
        _mm256_cmpeq_epi8(high_bits, _mm256_set1_epi64x(-1ll)),
        _mm256_set1_epi8(16));
    return _mm256_or_si256(low_nibbles, high_bits);
}

static inline void moe_q5_1_q8_1_accumulate_row_avx2(
    const uint8_t** row,
    __m256i qy,
    __m256i ones,
    float xd,
    float xs,
    __m256* acc,
    float* min_acc
) {
    uint16_t d16 = 0;
    uint16_t m16 = 0;
    std::memcpy(&d16, *row, sizeof(d16));
    std::memcpy(&m16, *row + 2u, sizeof(m16));
    const __m256i qx = moe_q5_1_unpack_u5x32_avx2(*row);
    const __m256i pair = _mm256_maddubs_epi16(qx, qy);
    const __m256i sum = _mm256_madd_epi16(pair, ones);
    *acc = moe_madd_ps(
        _mm256_cvtepi32_ps(sum),
        _mm256_set1_ps(moe_raw_fp16_to_f32(d16) * xd),
        *acc);
    *min_acc += moe_raw_fp16_to_f32(m16) * xs;
    *row += 24u;
}

#if defined(__F16C__)
static inline __m256i moe_q5_1_8x8_load_dm32_avx2(const uint8_t* tile) {
    uint32_t dm0 = 0;
    uint32_t dm1 = 0;
    uint32_t dm2 = 0;
    uint32_t dm3 = 0;
    uint32_t dm4 = 0;
    uint32_t dm5 = 0;
    uint32_t dm6 = 0;
    uint32_t dm7 = 0;
    std::memcpy(&dm0, tile + 0u, sizeof(dm0));
    std::memcpy(&dm1, tile + 24u, sizeof(dm1));
    std::memcpy(&dm2, tile + 48u, sizeof(dm2));
    std::memcpy(&dm3, tile + 72u, sizeof(dm3));
    std::memcpy(&dm4, tile + 96u, sizeof(dm4));
    std::memcpy(&dm5, tile + 120u, sizeof(dm5));
    std::memcpy(&dm6, tile + 144u, sizeof(dm6));
    std::memcpy(&dm7, tile + 168u, sizeof(dm7));
    return _mm256_setr_epi32(
        (int)dm0, (int)dm1, (int)dm2, (int)dm3,
        (int)dm4, (int)dm5, (int)dm6, (int)dm7);
}

static inline __m256 moe_q5_1_8x8_madd_row_avx2(
    const uint8_t* packed,
    __m256i qy,
    __m256i ones,
    __m256 dscale,
    int scale_index,
    __m256 acc
) {
    const __m256i qx = moe_q5_1_unpack_u5x32_avx2(packed);
    const __m256i pair = _mm256_maddubs_epi16(qx, qy);
    const __m256i sum = _mm256_madd_epi16(pair, ones);
    return moe_madd_ps(
        _mm256_cvtepi32_ps(sum),
        _mm256_permutevar8x32_ps(dscale, _mm256_set1_epi32(scale_index)),
        acc);
}

template <uint32_t RowBase>
static inline __m256 moe_q5_1_8x8_load_four_dscales_avx2(
    const uint8_t* tile,
    float x_scale
) {
    uint16_t d0 = 0;
    uint16_t d1 = 0;
    uint16_t d2 = 0;
    uint16_t d3 = 0;
    std::memcpy(&d0, tile + (RowBase + 0u) * 24u, sizeof(d0));
    std::memcpy(&d1, tile + (RowBase + 1u) * 24u, sizeof(d1));
    std::memcpy(&d2, tile + (RowBase + 2u) * 24u, sizeof(d2));
    std::memcpy(&d3, tile + (RowBase + 3u) * 24u, sizeof(d3));
    return _mm256_mul_ps(
        _mm256_cvtph_ps(_mm_setr_epi16(
            (int16_t)d0, (int16_t)d1, (int16_t)d2, (int16_t)d3,
            0, 0, 0, 0)),
        _mm256_set1_ps(x_scale));
}

template <uint32_t RowBase, bool UpdateMin>
static inline void moe_q5_1_8x8_accumulate_four_rows_avx2(
    const uint8_t* tile,
    const moe_q8_1_activation_block_t* xq,
    uint32_t blocks,
    __m256* min_acc_vec,
    float* out
) {
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    for (uint32_t b = 0; b < blocks; ++b, tile += 192u) {
        const moe_q8_1_activation_block_t& xb = xq[b];
        const __m256i qy = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xb.qs));
        __m256 dscale;
        if constexpr (UpdateMin) {
            const __m256i dm32 = moe_q5_1_8x8_load_dm32_avx2(tile);
            const __m256i d32 = _mm256_and_si256(dm32, _mm256_set1_epi32(0xffff));
            const __m256i m32 = _mm256_srli_epi32(dm32, 16);
            const __m128i d16 = _mm_packus_epi32(
                _mm256_castsi256_si128(d32), _mm256_extracti128_si256(d32, 1));
            dscale = _mm256_mul_ps(_mm256_cvtph_ps(d16), _mm256_set1_ps(xb.d));
            const __m128i m16 = _mm_packus_epi32(
                _mm256_castsi256_si128(m32), _mm256_extracti128_si256(m32, 1));
            *min_acc_vec = moe_madd_ps(
                _mm256_cvtph_ps(m16), _mm256_set1_ps(xb.s), *min_acc_vec);
        } else {
            dscale = moe_q5_1_8x8_load_four_dscales_avx2<RowBase>(tile, xb.d);
        }
        const uint8_t* packed = tile + RowBase * 24u;
        acc0 = moe_q5_1_8x8_madd_row_avx2(packed + 0u, qy, ones, dscale, 0u, acc0);
        acc1 = moe_q5_1_8x8_madd_row_avx2(packed + 24u, qy, ones, dscale, 1u, acc1);
        acc2 = moe_q5_1_8x8_madd_row_avx2(packed + 48u, qy, ones, dscale, 2u, acc2);
        acc3 = moe_q5_1_8x8_madd_row_avx2(packed + 72u, qy, ones, dscale, 3u, acc3);
    }
    out[RowBase + 0u] = moe_dot_hsum256_ps(acc0);
    out[RowBase + 1u] = moe_dot_hsum256_ps(acc1);
    out[RowBase + 2u] = moe_dot_hsum256_ps(acc2);
    out[RowBase + 3u] = moe_dot_hsum256_ps(acc3);
}
#endif

#endif

static int moe_dot_gguf_q5_1_q8_1_eight_rows(
    const uint8_t* packed0,
    const uint8_t* packed1,
    const uint8_t* packed2,
    const uint8_t* packed3,
    const uint8_t* packed4,
    const uint8_t* packed5,
    const uint8_t* packed6,
    const uint8_t* packed7,
    const moe_q8_1_activation_block_t* xq,
    uint32_t cols,
    float* out
) {
    if (!packed0 || !packed1 || !packed2 || !packed3 ||
        !packed4 || !packed5 || !packed6 || !packed7 ||
        !xq || !out || cols == 0 || (cols & 31u) != 0u) {
        return 0;
    }
    const uint32_t blocks = cols >> 5u;
#if defined(__AVX2__)
    const uint8_t* row0 = packed0;
    const uint8_t* row1 = packed1;
    const uint8_t* row2 = packed2;
    const uint8_t* row3 = packed3;
    const uint8_t* row4 = packed4;
    const uint8_t* row5 = packed5;
    const uint8_t* row6 = packed6;
    const uint8_t* row7 = packed7;
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps();
    __m256 acc5 = _mm256_setzero_ps();
    __m256 acc6 = _mm256_setzero_ps();
    __m256 acc7 = _mm256_setzero_ps();
    float min0 = 0.0f;
    float min1 = 0.0f;
    float min2 = 0.0f;
    float min3 = 0.0f;
    float min4 = 0.0f;
    float min5 = 0.0f;
    float min6 = 0.0f;
    float min7 = 0.0f;
    for (uint32_t b = 0; b < blocks; ++b) {
        const moe_q8_1_activation_block_t& xb = xq[b];
        const __m256i qy = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xb.qs));
        moe_q5_1_q8_1_accumulate_row_avx2(&row0, qy, ones, xb.d, xb.s, &acc0, &min0);
        moe_q5_1_q8_1_accumulate_row_avx2(&row1, qy, ones, xb.d, xb.s, &acc1, &min1);
        moe_q5_1_q8_1_accumulate_row_avx2(&row2, qy, ones, xb.d, xb.s, &acc2, &min2);
        moe_q5_1_q8_1_accumulate_row_avx2(&row3, qy, ones, xb.d, xb.s, &acc3, &min3);
        moe_q5_1_q8_1_accumulate_row_avx2(&row4, qy, ones, xb.d, xb.s, &acc4, &min4);
        moe_q5_1_q8_1_accumulate_row_avx2(&row5, qy, ones, xb.d, xb.s, &acc5, &min5);
        moe_q5_1_q8_1_accumulate_row_avx2(&row6, qy, ones, xb.d, xb.s, &acc6, &min6);
        moe_q5_1_q8_1_accumulate_row_avx2(&row7, qy, ones, xb.d, xb.s, &acc7, &min7);
    }
    out[0] = moe_dot_hsum256_ps(acc0) + min0;
    out[1] = moe_dot_hsum256_ps(acc1) + min1;
    out[2] = moe_dot_hsum256_ps(acc2) + min2;
    out[3] = moe_dot_hsum256_ps(acc3) + min3;
    out[4] = moe_dot_hsum256_ps(acc4) + min4;
    out[5] = moe_dot_hsum256_ps(acc5) + min5;
    out[6] = moe_dot_hsum256_ps(acc6) + min6;
    out[7] = moe_dot_hsum256_ps(acc7) + min7;
#else
    const uint8_t* rows[8] = {
        packed0, packed1, packed2, packed3,
        packed4, packed5, packed6, packed7,
    };
    for (uint32_t r = 0; r < 8u; ++r) {
        out[r] = 0.0f;
    }
    for (uint32_t b = 0; b < blocks; ++b) {
        const moe_q8_1_activation_block_t& xb = xq[b];
        for (uint32_t r = 0; r < 8u; ++r) {
            uint16_t d16 = 0;
            uint16_t m16 = 0;
            uint32_t qh = 0;
            std::memcpy(&d16, rows[r], sizeof(d16));
            std::memcpy(&m16, rows[r] + 2u, sizeof(m16));
            std::memcpy(&qh, rows[r] + 4u, sizeof(qh));
            const uint8_t* qs = rows[r] + 8u;
            int32_t sum = 0;
            for (uint32_t i = 0; i < 32u; ++i) {
                const uint8_t q4 = i < 16u ? (qs[i] & 0x0fu) : (qs[i - 16u] >> 4);
                const uint8_t q5 = q4 | (uint8_t)(((qh >> i) & 1u) << 4u);
                sum += (int32_t)q5 * (int32_t)xb.qs[i];
            }
            out[r] += moe_raw_fp16_to_f32(d16) * xb.d * (float)sum +
                moe_raw_fp16_to_f32(m16) * xb.s;
            rows[r] += 24u;
        }
    }
#endif
#if defined(__AVX2__)
    return moe_q5_1_8x8_all_finite(out);
#else
    for (uint32_t r = 0; r < 8u; ++r) {
        if (!std::isfinite(out[r])) {
            return 0;
        }
    }
    return 1;
#endif
}

static uint64_t moe_q5_1_8x8_group_bytes(uint32_t cols) {
    return ((uint64_t)cols + 31ull) / 32ull * 192ull;
}

static int moe_dot_gguf_q5_1_8x8_q8_1_rows_checked(
    const uint8_t* tile,
    const moe_q8_1_activation_block_t* xq,
    uint32_t cols,
    float* out,
    int check_finite
);

#if defined(__AVX2__) && defined(__F16C__)
template <uint32_t RowBase, bool UpdateMin>
static inline void moe_q5_1_8x8_accumulate_four_rows_pair_avx2(
    const uint8_t* tile0,
    const uint8_t* tile1,
    const moe_q8_1_activation_block_t* xq,
    uint32_t blocks,
    __m256* min_acc_vec0,
    __m256* min_acc_vec1,
    float* out0,
    float* out1
) {
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc00 = _mm256_setzero_ps();
    __m256 acc01 = _mm256_setzero_ps();
    __m256 acc02 = _mm256_setzero_ps();
    __m256 acc03 = _mm256_setzero_ps();
    __m256 acc10 = _mm256_setzero_ps();
    __m256 acc11 = _mm256_setzero_ps();
    __m256 acc12 = _mm256_setzero_ps();
    __m256 acc13 = _mm256_setzero_ps();
    for (uint32_t b = 0; b < blocks; ++b, tile0 += 192u, tile1 += 192u) {
        const moe_q8_1_activation_block_t& xb = xq[b];
        const __m256i qy = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xb.qs));
        __m256 dscale0;
        __m256 dscale1;
        if constexpr (UpdateMin) {
            const __m256i dm320 = moe_q5_1_8x8_load_dm32_avx2(tile0);
            const __m256i d320 = _mm256_and_si256(dm320, _mm256_set1_epi32(0xffff));
            const __m256i m320 = _mm256_srli_epi32(dm320, 16);
            const __m128i d160 = _mm_packus_epi32(
                _mm256_castsi256_si128(d320), _mm256_extracti128_si256(d320, 1));
            dscale0 = _mm256_mul_ps(_mm256_cvtph_ps(d160), _mm256_set1_ps(xb.d));
            const __m128i m160 = _mm_packus_epi32(
                _mm256_castsi256_si128(m320), _mm256_extracti128_si256(m320, 1));
            *min_acc_vec0 = moe_madd_ps(
                _mm256_cvtph_ps(m160), _mm256_set1_ps(xb.s), *min_acc_vec0);

            const __m256i dm321 = moe_q5_1_8x8_load_dm32_avx2(tile1);
            const __m256i d321 = _mm256_and_si256(dm321, _mm256_set1_epi32(0xffff));
            const __m256i m321 = _mm256_srli_epi32(dm321, 16);
            const __m128i d161 = _mm_packus_epi32(
                _mm256_castsi256_si128(d321), _mm256_extracti128_si256(d321, 1));
            dscale1 = _mm256_mul_ps(_mm256_cvtph_ps(d161), _mm256_set1_ps(xb.d));
            const __m128i m161 = _mm_packus_epi32(
                _mm256_castsi256_si128(m321), _mm256_extracti128_si256(m321, 1));
            *min_acc_vec1 = moe_madd_ps(
                _mm256_cvtph_ps(m161), _mm256_set1_ps(xb.s), *min_acc_vec1);
        } else {
            dscale0 = moe_q5_1_8x8_load_four_dscales_avx2<RowBase>(tile0, xb.d);
            dscale1 = moe_q5_1_8x8_load_four_dscales_avx2<RowBase>(tile1, xb.d);
        }
        const uint8_t* packed0 = tile0 + RowBase * 24u;
        const uint8_t* packed1 = tile1 + RowBase * 24u;
        acc00 = moe_q5_1_8x8_madd_row_avx2(packed0 + 0u, qy, ones, dscale0, 0u, acc00);
        acc01 = moe_q5_1_8x8_madd_row_avx2(packed0 + 24u, qy, ones, dscale0, 1u, acc01);
        acc02 = moe_q5_1_8x8_madd_row_avx2(packed0 + 48u, qy, ones, dscale0, 2u, acc02);
        acc03 = moe_q5_1_8x8_madd_row_avx2(packed0 + 72u, qy, ones, dscale0, 3u, acc03);
        acc10 = moe_q5_1_8x8_madd_row_avx2(packed1 + 0u, qy, ones, dscale1, 0u, acc10);
        acc11 = moe_q5_1_8x8_madd_row_avx2(packed1 + 24u, qy, ones, dscale1, 1u, acc11);
        acc12 = moe_q5_1_8x8_madd_row_avx2(packed1 + 48u, qy, ones, dscale1, 2u, acc12);
        acc13 = moe_q5_1_8x8_madd_row_avx2(packed1 + 72u, qy, ones, dscale1, 3u, acc13);
    }
    out0[RowBase + 0u] = moe_dot_hsum256_ps(acc00);
    out0[RowBase + 1u] = moe_dot_hsum256_ps(acc01);
    out0[RowBase + 2u] = moe_dot_hsum256_ps(acc02);
    out0[RowBase + 3u] = moe_dot_hsum256_ps(acc03);
    out1[RowBase + 0u] = moe_dot_hsum256_ps(acc10);
    out1[RowBase + 1u] = moe_dot_hsum256_ps(acc11);
    out1[RowBase + 2u] = moe_dot_hsum256_ps(acc12);
    out1[RowBase + 3u] = moe_dot_hsum256_ps(acc13);
}
#endif

static int moe_dot_gguf_q5_1_8x8_q8_1_tile_pair(
    const uint8_t* tile0,
    const uint8_t* tile1,
    const moe_q8_1_activation_block_t* xq,
    uint32_t cols,
    float* out0,
    float* out1
) {
    if (!tile0 || !tile1 || !xq || !out0 || !out1 || cols == 0 || (cols & 31u) != 0u) {
        return 0;
    }
    const uint32_t blocks = cols >> 5u;
#if defined(__AVX2__) && defined(__F16C__)
    __m256 min_acc_vec0 = _mm256_setzero_ps();
    __m256 min_acc_vec1 = _mm256_setzero_ps();
    moe_q5_1_8x8_accumulate_four_rows_pair_avx2<0u, true>(
        tile0, tile1, xq, blocks, &min_acc_vec0, &min_acc_vec1, out0, out1);
    moe_q5_1_8x8_accumulate_four_rows_pair_avx2<4u, false>(
        tile0, tile1, xq, blocks, &min_acc_vec0, &min_acc_vec1, out0, out1);
    float min_acc0[8];
    float min_acc1[8];
    _mm256_storeu_ps(min_acc0, min_acc_vec0);
    _mm256_storeu_ps(min_acc1, min_acc_vec1);
    for (uint32_t r = 0; r < 8u; ++r) {
        out0[r] += min_acc0[r];
        out1[r] += min_acc1[r];
    }
    return moe_q5_1_8x8_all_finite(out0) && moe_q5_1_8x8_all_finite(out1);
#else
    return moe_dot_gguf_q5_1_8x8_q8_1_rows_checked(tile0, xq, cols, out0, 1) &&
           moe_dot_gguf_q5_1_8x8_q8_1_rows_checked(tile1, xq, cols, out1, 1);
#endif
}

static int moe_dot_gguf_q5_1_8x8_q8_1_rows_checked(
    const uint8_t* tile,
    const moe_q8_1_activation_block_t* xq,
    uint32_t cols,
    float* out,
    int check_finite
) {
    if (!tile || !xq || !out || cols == 0 || (cols & 31u) != 0u) {
        return 0;
    }
    const uint32_t blocks = cols >> 5u;
#if defined(__AVX2__)
#if defined(__F16C__)
    // Keep four accumulators resident at a time on AVX2-only CPUs. Each row
    // still visits blocks in the original order, preserving exact arithmetic.
    __m256 min_acc_vec = _mm256_setzero_ps();
    moe_q5_1_8x8_accumulate_four_rows_avx2<0u, true>(
        tile, xq, blocks, &min_acc_vec, out);
    moe_q5_1_8x8_accumulate_four_rows_avx2<4u, false>(
        tile, xq, blocks, &min_acc_vec, out);
    float min_acc[8];
    _mm256_storeu_ps(min_acc, min_acc_vec);
    for (uint32_t r = 0; r < 8u; ++r) {
        out[r] += min_acc[r];
    }
#else
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc[8] = {
        _mm256_setzero_ps(), _mm256_setzero_ps(),
        _mm256_setzero_ps(), _mm256_setzero_ps(),
        _mm256_setzero_ps(), _mm256_setzero_ps(),
        _mm256_setzero_ps(), _mm256_setzero_ps(),
    };
    float min_acc[8]{};
    for (uint32_t b = 0; b < blocks; ++b, tile += 192u) {
        const moe_q8_1_activation_block_t& xb = xq[b];
        const __m256i qy = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xb.qs));
        for (uint32_t r = 0; r < 8u; ++r) {
            const uint8_t* packed = tile + r * 24u;
            uint16_t d16 = 0;
            uint16_t m16 = 0;
            std::memcpy(&d16, packed, sizeof(d16));
            std::memcpy(&m16, packed + 2u, sizeof(m16));
            const __m256i qx = moe_q5_1_unpack_u5x32_avx2(packed);
            const __m256i pair = _mm256_maddubs_epi16(qx, qy);
            const __m256i sum = _mm256_madd_epi16(pair, ones);
            acc[r] = moe_madd_ps(
                _mm256_cvtepi32_ps(sum),
                _mm256_set1_ps(moe_raw_fp16_to_f32(d16) * xb.d),
                acc[r]);
            min_acc[r] += moe_raw_fp16_to_f32(m16) * xb.s;
        }
    }
    for (uint32_t r = 0; r < 8u; ++r) {
        out[r] = moe_dot_hsum256_ps(acc[r]) + min_acc[r];
    }
#endif
#else
    for (uint32_t r = 0; r < 8u; ++r) {
        out[r] = 0.0f;
    }
    for (uint32_t b = 0; b < blocks; ++b, tile += 192u) {
        const moe_q8_1_activation_block_t& xb = xq[b];
        for (uint32_t r = 0; r < 8u; ++r) {
            const uint8_t* packed = tile + r * 24u;
            uint16_t d16 = 0;
            uint16_t m16 = 0;
            uint32_t qh = 0;
            std::memcpy(&d16, packed, sizeof(d16));
            std::memcpy(&m16, packed + 2u, sizeof(m16));
            std::memcpy(&qh, packed + 4u, sizeof(qh));
            const uint8_t* qs = packed + 8u;
            int32_t sum = 0;
            for (uint32_t i = 0; i < 32u; ++i) {
                const uint8_t q4 = i < 16u ? (qs[i] & 0x0fu) : (qs[i - 16u] >> 4);
                const uint8_t q5 = q4 | (uint8_t)(((qh >> i) & 1u) << 4u);
                sum += (int32_t)q5 * (int32_t)xb.qs[i];
            }
            out[r] += moe_raw_fp16_to_f32(d16) * xb.d * (float)sum +
                moe_raw_fp16_to_f32(m16) * xb.s;
        }
    }
#endif
    if (!check_finite) {
        return 1;
    }
#if defined(__AVX2__)
    return moe_q5_1_8x8_all_finite(out);
#else
    for (uint32_t r = 0; r < 8u; ++r) {
        if (!std::isfinite(out[r])) {
            return 0;
        }
    }
    return 1;
#endif
}

static int moe_dot_gguf_q5_1_8x8_q8_1_rows(
    const uint8_t* tile,
    const moe_q8_1_activation_block_t* xq,
    uint32_t cols,
    float* out
) {
    return moe_dot_gguf_q5_1_8x8_q8_1_rows_checked(tile, xq, cols, out, 1);
}

static int moe_dot_gguf_q5_1_8x8_q8_1_rows_unchecked(
    const uint8_t* tile,
    const moe_q8_1_activation_block_t* xq,
    uint32_t cols,
    float* out
) {
    return moe_dot_gguf_q5_1_8x8_q8_1_rows_checked(tile, xq, cols, out, 0);
}
