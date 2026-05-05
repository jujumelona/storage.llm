// IQ quantization kernels
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
    // BUGFIX 807: NULL pointer check
    if (!packed || !x) return 0.0f;
    // BUGFIX 808: Range validation
    if (start > limit) return 0.0f;
    // BUGFIX 809: Scale NaN/Inf check
    if (!std::isfinite(scale)) return 0.0f;

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
    // BUGFIX 810: Result NaN check
    if (!std::isfinite(acc)) return 0.0f;
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
    // BUGFIX 811: NULL pointer check
    if (!packed || !x) return 0.0f;
    // BUGFIX 812: Range validation
    if (start > limit) return 0.0f;
    // BUGFIX 813: Bits validation
    if (bits == 0 || bits > 8) return 0.0f;
    // BUGFIX 814: Scale NaN/Inf check
    if (!std::isfinite(scale)) return 0.0f;

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
    // BUGFIX 815: Result NaN check before scale
    if (!std::isfinite(acc)) return 0.0f;
    acc *= scale;
    // BUGFIX 816: Final result NaN check
    if (!std::isfinite(acc)) return 0.0f;
    return acc;
}

static float moe_dot_linear_quant_row(
    const moe_tensor_record& rec,
    const mmap_context_t* shard,
    const uint8_t* packed,
    const float* x,
    uint32_t bits,
    uint32_t row
) {
    // BUGFIX 817: NULL pointer checks
    if (!shard || !packed || !x) return 0.0f;
    // BUGFIX 818: Cols validation
    if (rec.info.cols == 0 || rec.info.cols > 1000000) return 0.0f;

    const uint32_t ngroups = rec.info.groups ? rec.info.groups : 1u;
    const uint32_t group_size = rec.info.group_size ? rec.info.group_size : rec.info.cols;
    // BUGFIX 819: Groups overflow check
    if (ngroups > 100000) return 0.0f;

    float acc = 0.0f;
    if (bits != 2u && bits != 3u && bits != 4u && bits != 8u) {
        return 0.0f;
    }
    for (uint32_t group = 0; group < ngroups; ++group) {
        const float scale = moe_raw_scale_at(rec, shard, row, group);
        // BUGFIX 820: Scale NaN check
        if (!std::isfinite(scale)) continue;

        const uint32_t start = group * group_size;
        const uint32_t limit = std::min<uint32_t>(start + group_size, rec.info.cols);
        if (bits == 8u) {
            acc += moe_dot_linear_q8_group(packed, x, scale, start, limit);
        } else {
            acc += moe_dot_linear_packed_group(packed, x, scale, bits, start, limit);
        }
        // BUGFIX 821: Accumulator NaN check per group
        if (!std::isfinite(acc)) return 0.0f;
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
    // BUGFIX 822: NULL pointer checks
    if (!shard || !packed || !x) return 0.0f;
    // BUGFIX 823: Cols validation
    if (rec.info.cols == 0 || rec.info.cols > 1000000) return 0.0f;
    // BUGFIX 824: Group size validation
    if (rec.info.group_size == 0 || rec.info.group_size > rec.info.cols) return 0.0f;

    const uint32_t ngroups = rec.info.groups;
    // Fix 28: Shrink thread_local buffer if it grew too large (4x current need)
    static thread_local std::vector<float> tls_scales_buf;
    // BUGFIX 501: ngroups 범위 체크
    if (ngroups > 100000) return 0.0f;
    const int use_scale4 = rec.scale4_entry && engine && engine->scale4;
    if (!use_scale4 && tls_scales_buf.size() < ngroups) {
        tls_scales_buf.resize(ngroups);
    } else if (!use_scale4 && tls_scales_buf.capacity() > ngroups * 4u && ngroups > 0) {
        tls_scales_buf.shrink_to_fit();
        tls_scales_buf.resize(ngroups);
    }
    float* scales_buf = tls_scales_buf.data();
    if (!use_scale4) {
        for (uint32_t g = 0; g < ngroups; ++g) {
            scales_buf[g] = moe_raw_scale_at(rec, shard, row, g);
        }
    }
    float acc = 0.0f;
    for (uint32_t group = 0; group < ngroups; ++group) {
        const float scale = use_scale4
            ? moe_scale4_get_scale(engine->scale4, rec.scale4_entry, row, group)
            : scales_buf[group];
        // BUGFIX 825: Scale NaN check
        if (!std::isfinite(scale)) continue;

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
        // BUGFIX 826: Accumulator NaN check per group
        if (!std::isfinite(acc)) return 0.0f;
    }
    return acc;
}
