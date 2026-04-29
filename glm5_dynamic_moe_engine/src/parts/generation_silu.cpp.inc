#if defined(GLM5_FAST_APPROX_SILU)
static float glm5_fast_exp_approx(float x) {
    // Bug 9: Handle NaN input to prevent UB in float→int conversion
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

static float glm5_generation_silu(float x) {
#if defined(GLM5_FAST_APPROX_SILU)
    return x / (1.0f + glm5_fast_exp_approx(-x));
#else
    return x / (1.0f + std::exp(-x));
#endif
}
