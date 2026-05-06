#pragma once

#include <stdint.h>

// Bit Packing/Unpacking for 1-8-bit quantization indices.

// Pack b-bit indices into bytes
void qkv_pack_indices(const int* indices, uint8_t* packed, int n, int bits);

// Unpack b-bit indices from bytes
void qkv_unpack_indices(const uint8_t* packed, int* indices, int n, int bits);

// Pack 1-bit signs (positive = 1, negative = 0)
void qkv_pack_signs(const float* signs, uint8_t* packed, int n);

// Unpack 1-bit signs (1 = +1.0f, 0 = -1.0f)
void qkv_unpack_signs(const uint8_t* packed, float* signs, int n);

// IEEE fp16 conversion used by raw 16-bit QKV cache entries.
uint16_t qkv_float_to_fp16_bits(float value);
float qkv_fp16_bits_to_float(uint16_t value);
