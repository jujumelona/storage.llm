static float glm5_dot_raw_fp16_row(const uint8_t* packed, const float* x, uint32_t cols) {
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
        acc0 = _mm256_fmadd_ps(f0, x0, acc0);
        acc1 = _mm256_fmadd_ps(f1, x1, acc1);
    }
    for (; col + 7u < cols; col += 8u) {
        __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(fp16 + col));
        __m256 f = _mm256_cvtph_ps(h);
        __m256 xv = _mm256_loadu_ps(x + col);
        acc0 = _mm256_fmadd_ps(f, xv, acc0);
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
        acc += glm5_raw_fp16_to_f32(fp16[col]) * x[col];
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
        a0 += glm5_raw_fp16_to_f32(v0) * x[col + 0u];
        a1 += glm5_raw_fp16_to_f32(v1) * x[col + 1u];
        a2 += glm5_raw_fp16_to_f32(v2) * x[col + 2u];
        a3 += glm5_raw_fp16_to_f32(v3) * x[col + 3u];
    }
    float acc = (a0 + a1) + (a2 + a3);
    for (; col < cols; ++col) {
        uint16_t v;
        std::memcpy(&v, packed + (uint64_t)col * 2ull, sizeof(v));
        acc += glm5_raw_fp16_to_f32(v) * x[col];
    }
    return acc;
#endif
}

static float glm5_dot_raw_fp32_row(const uint8_t* packed, const float* x, uint32_t cols) {
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
        acc0 = _mm256_fmadd_ps(f0, x0, acc0);
        acc1 = _mm256_fmadd_ps(f1, x1, acc1);
    }
    for (; col + 7u < cols; col += 8u) {
        __m256 f = _mm256_loadu_ps(fp32 + col);
        __m256 xv = _mm256_loadu_ps(x + col);
        acc0 = _mm256_fmadd_ps(f, xv, acc0);
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

static float glm5_dot_fp4_row(
    const glm5_tensor_record& rec,
    const mmap_context_t* shard,
    const uint8_t* packed,
    const float* x,
    glm5_pc_engine_t* engine,
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
            scales_buf[g] = glm5_scale4_get_scale(engine->scale4, rec.scale4_entry, row, g);
        }
    } else {
        for (uint32_t g = 0; g < ngroups; ++g) {
            scales_buf[g] = glm5_raw_scale_at(rec, shard, row, g);
        }
    }
    float acc = 0.0f;
    for (uint32_t group = 0; group < ngroups; ++group) {
        const float scale = scales_buf[group];
        const uint32_t start = group * rec.info.group_size;
        const uint32_t limit = std::min<uint32_t>(start + rec.info.group_size, rec.info.cols);
#if defined(__AVX2__)
        // Fix 25: True SIMD nibble extraction — process 16 nibbles (8 bytes) per iteration
        // Each byte contains 2 nibbles: low nibble at even col, high nibble at odd col
        const __m256 vscale = _mm256_set1_ps(scale);
        __m256 vacc = _mm256_setzero_ps();
        uint32_t col = start;
        // Process 8 nibbles (4 bytes) per iteration. Scalar table expansion avoids
        // AVX2 gather latency on the 16-entry FP4 table.
        for (; col + 7u < limit; col += 8u) {
            uint32_t byte_offset = col / 2;
            alignas(32) int32_t idx[8];
            // Extract 8 nibbles from 4 bytes
            for (int k = 0; k < 4; ++k) {
                uint8_t b = packed[byte_offset + k];
                idx[k * 2] = b & 0x0F;           // Low nibble
                idx[k * 2 + 1] = (b >> 4) & 0x0F; // High nibble
            }
            alignas(32) float decoded[8];
            for (int k = 0; k < 8; ++k) {
                decoded[k] = glm5_engine::kNvFp4Table[idx[k] & 0x0F];
            }
            __m256 vals = _mm256_load_ps(decoded);
            vals = _mm256_mul_ps(vals, vscale);
            __m256 xv = _mm256_loadu_ps(x + col);
            vacc = _mm256_fmadd_ps(vals, xv, vacc);
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
            acc += glm5_engine::fp4_value_at(packed, col, scale) * x[col];
        }
#else
        for (uint32_t col = start; col < limit; ++col) {
            acc += glm5_engine::fp4_value_at(packed, col, scale) * x[col];
        }
#endif
    }
    return acc;
}
