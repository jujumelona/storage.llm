#if defined(__F16C__) && !defined(_MSC_VER)
#include <immintrin.h>
static float moe_raw_fp16_to_f32(uint16_t h) {
    return _cvtsh_ss(h);
}
#else
static float moe_raw_fp16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16u;
    uint32_t exp = (h >> 10u) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            exp = 127u - 15u + 1u;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1u;
                --exp;
            }
            out = sign | (exp << 23u) | ((mant & 0x03ffu) << 13u);
        }
    } else if (exp == 31u) {
        out = sign | 0x7f800000u | (mant << 13u);
    } else {
        out = sign | ((exp + 112u) << 23u) | (mant << 13u);
    }
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}
#endif
