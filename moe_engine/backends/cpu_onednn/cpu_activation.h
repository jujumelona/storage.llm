#pragma once

#include <cmath>
#include <cstdint>

namespace storagellm::cpu_moe {

inline float gelu_erf(float x) {
    return 0.5f * x * (1.0f + std::erf(x * 0.7071067811865476f));
}

inline float gelu_tanh(float x) {
    const float k = 0.7978845608028654f;
    const float inner = k * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + std::tanh(inner));
}

inline float silu(float x) {
    return x > 40.0f ? x : (x < -40.0f ? 0.0f : x / (1.0f + std::exp(-x)));
}

inline float activation(uint32_t mode, float gate, float up) {
    if (!std::isfinite(gate) || !std::isfinite(up)) {
        return 0.0f;
    }
    float activated = 0.0f;
    if (mode == 2u) {
        activated = gelu_tanh(gate);
    } else if (mode == 1u) {
        activated = gelu_erf(gate);
    } else {
        activated = silu(gate);
    }
    const float result = activated * up;
    return std::isfinite(result) ? result : 0.0f;
}

} // namespace storagellm::cpu_moe
