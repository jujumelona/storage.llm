static float moe_dot_raw_fp16_row(const uint8_t* packed, const float* x, uint32_t cols) {
#if defined(__F16C__) && defined(__AVX2__)
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
#if defined(__AVX2__)
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
        for (uint32_t i = 0; i < limit; ++i) {
            acc += d * (float)q[i] * x[start + i];
        }
        packed += block_bytes;
    }
    return acc;
}
static float moe_e8m0_to_f32_half(uint8_t v) {
    uint32_t bits = v ? ((uint32_t)v << 23u) : 0x00400000u;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

static float moe_mxfp4_value_from_code(uint8_t code) {
    static constexpr float kMxfp4Table[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    return kMxfp4Table[code & 0x0fu];
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
    for (uint32_t start = 0; start < cols; start += block_cols) {
        const float scale = moe_e8m0_to_f32_half(packed[0]);
        const uint8_t* qs = packed + 1u;
        const uint32_t limit = std::min<uint32_t>(block_cols, cols - start);
        for (uint32_t i = 0; i < limit; ++i) {
            acc += moe_mxfp4_value_from_code(moe_mxfp4_code_at(qs, i)) * scale * x[start + i];
        }
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
            // BUGFIX 218: Fix incorrect index masking for q5_1 block access
            // Problem: i & 15u masks to 0-15, but q5_1 blocks have 32 elements (16 bytes)
            //          This causes reading wrong byte for indices 16-31
            // Solution: Use i >> 1 for byte offset, i & 1 for nibble selection
            // Impact: Fixes data corruption in q5_1 quantized weights
            const uint8_t byte = qs[i >> 1];
            const uint32_t low = (i & 1u) ? (byte >> 4) : (byte & 0x0fu);
            const uint32_t high = (qh >> i) & 1u;
            const float w = d * (float)(low | (high << 4u)) + m;
            acc += w * x[start + i];
        }
        packed += block_bytes;
    }
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
    const uint32_t ngroups = rec.info.groups ? rec.info.groups : 1u;
    const uint32_t group_size = rec.info.group_size ? rec.info.group_size : rec.info.cols;
    float acc = 0.0f;
    if (bits == 8u) {
        for (uint32_t group = 0; group < ngroups; ++group) {
            const float scale = moe_raw_scale_at(rec, shard, row, group);
            const uint32_t start = group * group_size;
            const uint32_t limit = std::min<uint32_t>(start + group_size, rec.info.cols);
            for (uint32_t col = start; col < limit; ++col) {
                int8_t q = 0;
                std::memcpy(&q, packed + col, sizeof(q));
                acc += (float)q * scale * x[col];
            }
        }
        return acc;
    }
    const float midpoint = ((float)((1u << bits) - 1u)) * 0.5f;
    for (uint32_t group = 0; group < ngroups; ++group) {
        const float scale = moe_raw_scale_at(rec, shard, row, group);
        const uint32_t start = group * group_size;
        const uint32_t limit = std::min<uint32_t>(start + group_size, rec.info.cols);
        for (uint32_t col = start; col < limit; ++col) {
            const uint32_t code = moe_packed_lsb_code_at(packed, col, bits);
            acc += ((float)code - midpoint) * scale * x[col];
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
