#include "qkv_packing.h"
#include <string.h>
#include <climits>

void qkv_pack_indices(const int* indices, uint8_t* packed, int n, int bits) {
    // BUGFIX 394: 파라미터 유효성 체크
    if (!indices || !packed || n <= 0 || bits <= 0 || bits > 4) return;

    if (bits == 1) {
        // Bug ②: 1-bit packing — 8 indices per byte
        // BUGFIX 395: n + 7 overflow 방지
        if (n > INT_MAX - 7) return;
        memset(packed, 0, (n + 7) / 8);
        for (int i = 0; i < n; i++) {
            if (indices[i] & 1) packed[i / 8] |= (1 << (i % 8));
        }
    } else if (bits == 2) {
        // 4 indices per byte
        for (int i = 0; i < n; i += 4) {
            uint8_t byte = 0;
            for (int j = 0; j < 4 && (i + j) < n; j++) {
                byte |= (uint8_t)((indices[i + j] & 0x3) << (j * 2));
            }
            packed[i / 4] = byte;
        }
    } else if (bits == 3) {
        // Bug 3 Fix: Use LSB-first packing (consistent with 1/2/4-bit)
        // This prevents future inline optimization bugs where someone assumes
        // LSB-first for all bit widths.
        // BUGFIX 396: n * 3 overflow 방지
        if (n > INT_MAX / 3) return;
        memset(packed, 0, (n * 3 + 7) / 8);
        for (int i = 0; i < n; i++) {
            int bit_offset = i * 3;
            int byte_idx   = bit_offset / 8;
            int bit_shift  = bit_offset % 8;
            int val = indices[i] & 0x7;
            packed[byte_idx] |= (uint8_t)(val << bit_shift);
            if (bit_shift > 5) {  // Spans byte boundary
                packed[byte_idx + 1] |= (uint8_t)(val >> (8 - bit_shift));
            }
        }
    } else if (bits == 4) {
        // 2 indices per byte
        for (int i = 0; i < n; i += 2) {
            uint8_t byte = 0;
            for (int j = 0; j < 2 && (i + j) < n; j++) {
                byte |= (uint8_t)((indices[i + j] & 0xF) << (j * 4));
            }
            packed[i / 2] = byte;
        }
    }
}

void qkv_unpack_indices(const uint8_t* packed, int* indices, int n, int bits) {
    // BUGFIX 397: 파라미터 유효성 체크
    if (!packed || !indices || n <= 0 || bits <= 0 || bits > 4) return;

    if (bits == 1) {
        // Bug ②: 1-bit unpacking
        for (int i = 0; i < n; i++) {
            indices[i] = (packed[i / 8] >> (i % 8)) & 1;
        }
    } else if (bits == 2) {
        for (int i = 0; i < n; i++) {
            int byte_idx = i / 4;
            int bit_offset = (i % 4) * 2;
            indices[i] = (packed[byte_idx] >> bit_offset) & 0x3;
        }
    } else if (bits == 3) {
        // Bug 3 Fix: Use LSB-first unpacking (consistent with pack)
        for (int i = 0; i < n; i++) {
            int bit_offset = i * 3;
            int byte_idx   = bit_offset / 8;
            int bit_shift  = bit_offset % 8;
            int val = (packed[byte_idx] >> bit_shift) & 0x7;
            if (bit_shift > 5) {  // Spans byte boundary
                val |= (packed[byte_idx + 1] << (8 - bit_shift)) & 0x7;
            }
            indices[i] = val;
        }
    } else if (bits == 4) {
        for (int i = 0; i < n; i++) {
            int byte_idx = i / 2;
            int bit_offset = (i % 2) * 4;
            indices[i] = (packed[byte_idx] >> bit_offset) & 0xF;
        }
    }
}

void qkv_pack_signs(const float* signs, uint8_t* packed, int n) {
    // BUGFIX 398: 파라미터 유효성 체크
    if (!signs || !packed || n <= 0) return;

    for (int i = 0; i < n; i += 8) {
        uint8_t byte = 0;
        for (int j = 0; j < 8 && (i + j) < n; j++) {
            if (signs[i + j] >= 0.0f) {
                byte |= (1 << j);
            }
        }
        packed[i / 8] = byte;
    }
}

void qkv_unpack_signs(const uint8_t* packed, float* signs, int n) {
    // BUGFIX 399: 파라미터 유효성 체크
    if (!packed || !signs || n <= 0) return;

    for (int i = 0; i < n; i++) {
        int byte_idx = i / 8;
        int bit_offset = i % 8;
        signs[i] = (packed[byte_idx] & (1 << bit_offset)) ? 1.0f : -1.0f;
    }
}
