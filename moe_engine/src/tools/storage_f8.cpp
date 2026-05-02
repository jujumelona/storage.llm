#include "../../include/storage_llm_tools.h"

#include <cmath>

float storage_f8_e4m3_to_f32(uint8_t value) {
    const int sign = (value & 0x80u) ? -1 : 1;
    const int exp = (value >> 3) & 0x0f;
    const int mant = value & 0x07;
    // BUGFIX 450: 특수 케이스 처리
    if (exp == 0 && mant == 0) {
        return sign < 0 ? -0.0f : 0.0f;
    }
    // BUGFIX 451: exp == 0x0f 특수 케이스 범위 체크
    if (exp == 0x0f) {
        // BUGFIX 452: mant / 8.0f division by zero는 불가능하지만 명시적 체크
        return sign * std::ldexp(1.0f + mant / 8.0f, 8);
    }
    if (exp == 0) {
        return sign * std::ldexp(mant / 8.0f, -6);
    }
    // BUGFIX 453: 일반 케이스
    return sign * std::ldexp(1.0f + mant / 8.0f, exp - 7);
}

int storage_f8_e4m3_decode(
    const uint8_t* input,
    uint64_t count,
    float* out_values
) {
    if (!input || !out_values) {
        return 0;
    }
    // BUGFIX 438: count 범위 체크
    if (count > UINT64_MAX / sizeof(float)) {
        return 0;
    }
    for (uint64_t i = 0; i < count; ++i) {
        out_values[i] = storage_f8_e4m3_to_f32(input[i]);
    }
    return 1;
}
