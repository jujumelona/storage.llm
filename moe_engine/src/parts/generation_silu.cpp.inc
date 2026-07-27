#if defined(moe_FAST_APPROX_SILU)
static float moe_fast_exp_approx(float x) {
    // Bug 9: Handle NaN input to prevent UB in float?�int conversion
    // NaN comparisons are always false, leading to (uint32_t)NaN which is UB
    if (x != x) return 0.0f;  // NaN check: x != x is true only for NaN
    if (x <= -87.0f) {
        return 0.0f;
    }
    if (x >= 88.0f) {
        return std::numeric_limits<float>::infinity();
    }
    const float scaled = 12102203.0f * x + 1065353216.0f;
    const uint32_t bits = (uint32_t)scaled;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}
#endif

static float moe_generation_silu(float x) {
#if defined(moe_FAST_APPROX_SILU)
    return x / (1.0f + moe_fast_exp_approx(-x));
#else
    if (x <= -87.0f) return 0.0f;
    return x / (1.0f + std::exp(-x));
#endif
}

static float moe_generation_gelu(float x) {
    // PERF: Branchless GELU computation
    return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752440f));
}

static float moe_generation_gated_activation(const moe_pc_engine* engine, float gate, float up) {
    return (moe_engine_prefers_gelu_activation(engine) ?
        moe_generation_gelu(gate) :
        moe_generation_silu(gate)) * up;
}

#if defined(__AVX2__)
static inline void moe_generation_vector_gated_activation_avx2(
    const moe_pc_engine* engine,
    const float* gate,
    const float* up,
    float* out,
    uint32_t count
) {
    uint32_t i = 0;
    _mm_prefetch(reinterpret_cast<const char*>(gate), _MM_HINT_T0);
    _mm_prefetch(reinterpret_cast<const char*>(up), _MM_HINT_T0);
    const int use_gelu = moe_engine_prefers_gelu_activation(engine);
    for (; i + 7u < count; i += 8u) {
        _mm_prefetch(reinterpret_cast<const char*>(gate + i + 16), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(up + i + 16), _MM_HINT_T0);
        __m256 vg = _mm256_loadu_ps(gate + i);
        __m256 vu = _mm256_loadu_ps(up + i);
        __m256 act;
        if (use_gelu) {
            for (int k = 0; k < 8; ++k) {
                reinterpret_cast<float*>(&act)[k] = moe_generation_gelu(reinterpret_cast<const float*>(&vg)[k]);
            }
        } else {
            for (int k = 0; k < 8; ++k) {
                reinterpret_cast<float*>(&act)[k] = moe_generation_silu(reinterpret_cast<const float*>(&vg)[k]);
            }
        }
        __m256 res = _mm256_mul_ps(act, vu);
        _mm256_storeu_ps(out + i, res);
    }
    for (; i < count; ++i) {
        out[i] = moe_generation_gated_activation(engine, gate[i], up[i]);
    }
}
#endif
