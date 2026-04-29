#include "qkv_quantize.h"
#include "qkv_helpers.h"
#include "qkv_codebook.h"
#include "qkv_packing.h"
#include <string.h>
#include <math.h>

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

    // Step 1: Compute L2 norm
    float l2_norm = 0.0f;
    for (int i = 0; i < dim; ++i) {
        l2_norm += input[i] * input[i];
    }
    l2_norm = sqrtf(l2_norm);
    *norm_out = l2_norm;

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

    const float inv_norm = 1.0f / l2_norm;
    for (int i = 0; i < dim; ++i) {
        normalized[i] = input[i] * inv_norm;
    }

    // Step 3: Apply random rotation Pi
    const float* src = normalized;
    if (state && state->rotation_matrix && config && config->enable_rotation) {
        for (int i = 0; i < dim; i++) {
            float sum = 0.0f;
            for (int j = 0; j < dim; j++) {
                sum += state->rotation_matrix[i * dim + j] * normalized[j];
            }
            rotated[i] = sum;
        }
        src = rotated;
    }

    // Step 4: Lloyd-Max codebook quantization
    const float* centroids_s = qkv_codebook_for_bits(state, bits);
    const float* thresholds_s = qkv_thresholds_for_bits(state, bits);
    const int n_levels = 1 << bits;

    for (int i = 0; i < dim; ++i) {
        indices[i] = qkv_find_nearest_centroid(src[i], centroids_s, thresholds_s, n_levels);
    }

    memset(output, 0, (size_t)(dim * bits + 7) / 8);
    qkv_pack_indices(indices, output, dim, bits);
    return 1;
}
