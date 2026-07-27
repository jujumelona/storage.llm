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
    return x / (1.0f + std::exp(-x));
#endif
}

static float moe_generation_gelu(float x) {
    // BUGFIX 672: Check for NaN/Inf input ★★
    if (!std::isfinite(x)) {
        return 0.0f;
    }
    const float erf_val = std::erf(x * 0.70710678118654752440f);
    if (!std::isfinite(erf_val)) {
        return x > 0.0f ? x : 0.0f;
    }
    return 0.5f * x * (1.0f + erf_val);
}

static float moe_generation_gated_activation(const moe_pc_engine* engine, float gate, float up) {
    return (moe_engine_prefers_gelu_activation(engine) ?
        moe_generation_gelu(gate) :
        moe_generation_silu(gate)) * up;
}
