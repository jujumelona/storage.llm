#include "qkv_packing.h"
#include <string.h>
#include <climits>

void qkv_pack_indices(const int* indices, uint8_t* packed, int n, int bits) {
    if (!indices || !packed || n <= 0 || bits <= 0 || bits > 8) return;
    if (n > INT_MAX / bits) return;

    const size_t packed_bytes = ((size_t)n * (size_t)bits + 7u) / 8u;
    const uint32_t mask = (bits == 8) ? 0xffu : ((1u << bits) - 1u);
    memset(packed, 0, packed_bytes);

    for (int i = 0; i < n; ++i) {
        const size_t bit_offset = (size_t)i * (size_t)bits;
        const size_t byte_idx = bit_offset / 8u;
        const int bit_shift = (int)(bit_offset % 8u);
        const uint32_t val = (uint32_t)indices[i] & mask;
        const uint32_t shifted = val << bit_shift;
        packed[byte_idx] |= (uint8_t)(shifted & 0xffu);
        if (bit_shift + bits > 8) {
            packed[byte_idx + 1u] |= (uint8_t)((shifted >> 8) & 0xffu);
        }
    }
}

void qkv_unpack_indices(const uint8_t* packed, int* indices, int n, int bits) {
    if (!packed || !indices || n <= 0 || bits <= 0 || bits > 8) return;
    if (n > INT_MAX / bits) return;

    const uint32_t mask = (bits == 8) ? 0xffu : ((1u << bits) - 1u);
    for (int i = 0; i < n; ++i) {
        const size_t bit_offset = (size_t)i * (size_t)bits;
        const size_t byte_idx = bit_offset / 8u;
        const int bit_shift = (int)(bit_offset % 8u);
        uint32_t val = (uint32_t)packed[byte_idx] >> bit_shift;
        if (bit_shift + bits > 8) {
            val |= (uint32_t)packed[byte_idx + 1u] << (8 - bit_shift);
        }
        indices[i] = (int)(val & mask);
    }
}

void qkv_pack_signs(const float* signs, uint8_t* packed, int n) {
    if (!signs || !packed || n <= 0) return;

    memset(packed, 0, ((size_t)n + 7u) / 8u);
    for (int i = 0; i < n; i += 8) {
        uint8_t byte = 0;
        for (int j = 0; j < 8 && (i + j) < n; j++) {
            if (signs[i + j] >= 0.0f) {
                byte |= (uint8_t)(1u << j);
            }
        }
        packed[i / 8] = byte;
    }
}

void qkv_unpack_signs(const uint8_t* packed, float* signs, int n) {
    if (!packed || !signs || n <= 0) return;

    for (int i = 0; i < n; i++) {
        const int byte_idx = i / 8;
        const int bit_offset = i % 8;
        signs[i] = (packed[byte_idx] & (1 << bit_offset)) ? 1.0f : -1.0f;
    }
}

uint16_t qkv_float_to_fp16_bits(float value) {
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int exp = (int)((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        if (shift > 0 && ((mant >> (shift - 1u)) & 1u)) {
            ++half_mant;
        }
        return (uint16_t)(sign | half_mant);
    }
    if (exp >= 31) {
        return (uint16_t)(sign | 0x7c00u | (mant ? 0x0200u : 0u));
    }

    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    if (mant & 0x1000u) {
        ++half;
    }
    return (uint16_t)half;
}

float qkv_fp16_bits_to_float(uint16_t value) {
    const uint32_t sign = ((uint32_t)value & 0x8000u) << 16;
    int exp = (int)(((uint32_t)value >> 10) & 0x1fu);
    uint32_t mant = (uint32_t)value & 0x03ffu;
    uint32_t bits = 0;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03ffu;
            bits = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
    }

    float out = 0.0f;
    memcpy(&out, &bits, sizeof(out));
    return out;
}
