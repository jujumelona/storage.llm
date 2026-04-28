#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define GLM5_HD __host__ __device__
#else
#define GLM5_HD
#endif

namespace glm5_engine {

static constexpr float kNvFp4Table[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

GLM5_HD inline uint8_t fp4_code_at(const uint8_t* packed, std::size_t element_index) {
    const uint8_t byte = packed[element_index >> 1];
    return (element_index & 1) ? static_cast<uint8_t>(byte & 0x0f)
                               : static_cast<uint8_t>(byte >> 4);
}

GLM5_HD inline float fp4_value_at(const uint8_t* packed, std::size_t element_index, float scale) {
    return kNvFp4Table[fp4_code_at(packed, element_index)] * scale;
}

inline void dequant_packed_fp4_scalar(
    const uint8_t* packed,
    std::size_t element_offset,
    std::size_t element_count,
    float scale,
    float* out
) {
    for (std::size_t i = 0; i < element_count; ++i) {
        out[i] = fp4_value_at(packed, element_offset + i, scale);
    }
}

inline float dot_packed_fp4_f32(
    const uint8_t* packed,
    std::size_t element_offset,
    const float* x,
    std::size_t element_count,
    float scale
) {
    float acc = 0.0f;
    for (std::size_t i = 0; i < element_count; ++i) {
        acc += fp4_value_at(packed, element_offset + i, scale) * x[i];
    }
    return acc;
}

}  // namespace glm5_engine

#undef GLM5_HD

