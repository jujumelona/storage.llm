#include "qkv_quantize.h"
#include "qkv_helpers.h"
#include "qkv_codebook.h"
#include "qkv_packing.h"
#include "qkv_matrix.h"
#include <cmath>
#include <string.h>
#include <math.h>
#include <climits>

static void qkv_store_raw_vector(uint8_t* output, const float* src, int dim, int bits) {
    if (!output || !src || dim <= 0) return;
    if (bits == 16) {
        for (int i = 0; i < dim; ++i) {
            const uint16_t h = qkv_float_to_fp16_bits(src[i]);
            output[(size_t)i * 2u] = (uint8_t)(h & 0xffu);
            output[(size_t)i * 2u + 1u] = (uint8_t)((h >> 8) & 0xffu);
        }
    } else if (bits == 32) {
        memcpy(output, src, (size_t)dim * sizeof(float));
    }
}

int qkv_quantize_vector_with_state(
    const qkv_state_t* state,
    const qkv_config_t* config,
    const float* input,
    uint8_t* output,
    float* norm_out,
    int dim,
    int bits
) {
    if (!input || !output || !norm_out || dim <= 0 || !qkv_bits_valid(bits)) {
        return 0;
    }

    // BUGFIX 400: dim 범위 체크
    if (dim > 16384) {
        return 0;
    }

    // Step 1: Compute L2 norm
    // BUGFIX 722: Check for NaN/Inf in input before accumulation ★★★
    float l2_norm = 0.0f;
    for (int i = 0; i < dim; ++i) {
        if (!std::isfinite(input[i])) {
            *norm_out = 0.0f;
            memset(output, 0, (size_t)(dim * bits + 7) / 8);
            return 1;
        }
        l2_norm += input[i] * input[i];
    }
    // BUGFIX 723: Check for overflow in l2_norm accumulation ★★
    if (!std::isfinite(l2_norm)) {
        *norm_out = 0.0f;
        memset(output, 0, (size_t)(dim * bits + 7) / 8);
        return 1;
    }
    l2_norm = sqrtf(l2_norm);
    *norm_out = l2_norm;

    // BUGFIX 401: dim * bits overflow 방지
    if (dim > INT_MAX / bits) {
        return 0;
    }
    if (l2_norm < 1e-12f) {
        memset(output, 0, (size_t)(dim * bits + 7) / 8);
        return 1;
    }

    // Step 2: Normalize to unit vector
    float* normalized = state->scratch_residual;
    float* rotated = state->scratch_s_times_r;
    int* indices = state->scratch_indices;
    if (!normalized || !rotated || !indices) {
        return 0;
    }

    // BUGFIX 402: l2_norm이 0일 때 division by zero 방지 (이미 위에서 체크했지만 명시적으로)
    if (l2_norm < 1e-12f) {
        return 0;
    }
    const float inv_norm = 1.0f / l2_norm;
    for (int i = 0; i < dim; ++i) {
        normalized[i] = input[i] * inv_norm;
    }

    // Step 3: Apply random rotation Pi
    const float* src = normalized;
    if (state && state->rotation_matrix && config && config->enable_rotation) {
        if (state->rotation_signs &&
            qkv_apply_hadamard_rotation_forward(normalized, state->rotation_signs, rotated, dim)) {
            src = rotated;
        } else {
        // BUGFIX 403: rotation_matrix 범위 체크
        for (int i = 0; i < dim; i++) {
            float sum = 0.0f;
            for (int j = 0; j < dim; j++) {
                size_t idx = (size_t)i * (size_t)dim + (size_t)j;
                sum += state->rotation_matrix[idx] * normalized[j];
            }
            // BUGFIX 724: Check rotation result for NaN/Inf ★★
            if (!std::isfinite(sum)) {
                sum = 0.0f;
            }
            rotated[i] = sum;
        }
            src = rotated;
        }
    }

    if (qkv_bits_raw(bits)) {
        qkv_store_raw_vector(output, src, dim, bits);
        return 1;
    }

    // Step 4: Lloyd-Max codebook quantization
    if (!qkv_bits_codebook(bits)) return 0;
    const float* centroids_s = qkv_codebook_for_bits(state, bits);
    const float* thresholds_s = qkv_thresholds_for_bits(state, bits);
    // BUGFIX 485: centroids/thresholds null 체크
    if (!centroids_s || !thresholds_s) return 0;
    const int n_levels = 1 << bits;
    // BUGFIX 486: n_levels 범위 체크
    if (n_levels <= 0 || n_levels > 256) return 0;

    for (int i = 0; i < dim; ++i) {
        indices[i] = qkv_find_nearest_centroid(src[i], centroids_s, thresholds_s, n_levels);
    }

    memset(output, 0, (size_t)(dim * bits + 7) / 8);
    qkv_pack_indices(indices, output, dim, bits);
    return 1;
}
