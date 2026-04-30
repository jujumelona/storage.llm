// QKV Implementation - Faithful to TurboQuant Paper (2504.19874v1)
// 100% Complete Implementation

#include "kv_qkv.h"
#include "qkv_state.h"
#include "qkv_quantize.h"
#include "qkv_dequantize.h"
#include "qkv_attention.h"
#include "qkv_helpers.h"
#include "qkv_codebook.h"
#include "qkv_packing.h"
#include <atomic>
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Global mode flag. StorageLLM offload-native models use packed QKV by default;
// legacy callers can still explicitly disable it for debug comparisons.
std::atomic<bool> g_qkv_mode_enabled{true};

// ============================================================
// Public API - State Management
// ============================================================

int qkv_init(
    qkv_state_t* state,
    const qkv_config_t* config,
    int n_tokens
) {
    return qkv_state_init(state, config, n_tokens);
}

void qkv_free(qkv_state_t* state) {
    qkv_state_free(state);
}

// ============================================================
// Public API - Quantization (Paper Algorithm 1 + Algorithm 2)
// ============================================================

int qkv_quantize(
    qkv_state_t* state,
    const qkv_config_t* config,
    const float* key_data,
    const float* value_data,
    int n_tokens
) {
    if (!state || !config || !key_data || !value_data || n_tokens <= 0) {
        return 0;
    }

    int dim = config->head_dim;
    const bool use_qjl = config->enable_qjl;
    const int k_mse_bits = use_qjl ? state->k_bits - 1 : state->k_bits;
    const int v_mse_bits = use_qjl ? state->v_bits - 1 : state->v_bits;
    if (!qkv_bits_valid(k_mse_bits) || !qkv_bits_valid(v_mse_bits)) {
        return 0;
    }

    for (int t = 0; t < n_tokens; t++) {
        // ===== Quantize K =====
        // Paper Algorithm 1: MSE-optimal quantization
        uint8_t* k_out = state->k_idx + t * ((dim * k_mse_bits + 7) / 8);
        if (!qkv_quantize_vector_with_state(
                state, config,
                key_data + t * dim,
                k_out,
                &state->k_norms[t],
                dim,
                k_mse_bits)) {
            return 0;
        }

        // Paper Algorithm 2: QJL residual for unbiased inner product
        if (use_qjl && state->k_qjl && state->qjl_matrix) {
            // Step 1: Dequantize MSE reconstruction
            float* x_mse = state->scratch_x_tilde;
            if (!x_mse) return 0;

            // Unpack and lookup centroids
            int* indices = state->scratch_indices;
            float* y_tilde = state->scratch_y_tilde;
            if (!indices || !y_tilde) return 0;

            qkv_unpack_indices(k_out, indices, dim, k_mse_bits);
            const float* centroids = qkv_codebook_for_bits(state, k_mse_bits);
            for (int i = 0; i < dim; i++) {
                y_tilde[i] = centroids[indices[i]];
            }

            // Apply inverse rotation
            if (config->enable_rotation && state->rotation_matrix) {
                for (int i = 0; i < dim; i++) {
                    float sum = 0.0f;
                    for (int j = 0; j < dim; j++) {
                        sum += state->rotation_matrix[j * dim + i] * y_tilde[j];
                    }
                    x_mse[i] = sum * state->k_norms[t];
                }
            } else {
                for (int i = 0; i < dim; i++) {
                    x_mse[i] = y_tilde[i] * state->k_norms[t];
                }
            }

            // Step 2: Compute residual r = x - x_mse
            float* residual = state->scratch_residual;
            if (!residual) return 0;

            float r_norm_sq = 0.0f;
            for (int i = 0; i < dim; i++) {
                residual[i] = key_data[t * dim + i] - x_mse[i];
                r_norm_sq += residual[i] * residual[i];
            }
            state->k_residual_norms[t] = sqrtf(r_norm_sq);

            // Step 3: QJL - sign(S * r)
            float* s_times_r = state->scratch_s_times_r;
            if (!s_times_r) return 0;

            for (int i = 0; i < dim; i++) {
                float sum = 0.0f;
                for (int j = 0; j < dim; j++) {
                    sum += state->qjl_matrix[i * dim + j] * residual[j];
                }
                s_times_r[i] = sum;
            }

            // Pack signs
            uint8_t* qjl_out = state->k_qjl + t * ((dim + 7) / 8);
            qkv_pack_signs(s_times_r, qjl_out, dim);
        }

        // ===== Quantize V =====
        uint8_t* v_out = state->v_idx + t * ((dim * v_mse_bits + 7) / 8);
        if (!qkv_quantize_vector_with_state(
                state, config,
                value_data + t * dim,
                v_out,
                &state->v_norms[t],
                dim,
                v_mse_bits)) {
            return 0;
        }

        // QJL residual for V
        if (use_qjl && state->v_qjl && state->qjl_matrix) {
            float* x_mse = state->scratch_x_tilde;
            if (!x_mse) return 0;

            int* indices = state->scratch_indices;
            float* y_tilde = state->scratch_y_tilde;
            if (!indices || !y_tilde) return 0;

            qkv_unpack_indices(v_out, indices, dim, v_mse_bits);
            const float* centroids = qkv_codebook_for_bits(state, v_mse_bits);
            for (int i = 0; i < dim; i++) {
                y_tilde[i] = centroids[indices[i]];
            }

            if (config->enable_rotation && state->rotation_matrix) {
                for (int i = 0; i < dim; i++) {
                    float sum = 0.0f;
                    for (int j = 0; j < dim; j++) {
                        sum += state->rotation_matrix[j * dim + i] * y_tilde[j];
                    }
                    x_mse[i] = sum * state->v_norms[t];
                }
            } else {
                for (int i = 0; i < dim; i++) {
                    x_mse[i] = y_tilde[i] * state->v_norms[t];
                }
            }

            float* residual = state->scratch_residual;
            if (!residual) return 0;

            float r_norm_sq = 0.0f;
            for (int i = 0; i < dim; i++) {
                residual[i] = value_data[t * dim + i] - x_mse[i];
                r_norm_sq += residual[i] * residual[i];
            }
            state->v_residual_norms[t] = sqrtf(r_norm_sq);

            float* s_times_r = state->scratch_s_times_r;
            if (!s_times_r) return 0;

            for (int i = 0; i < dim; i++) {
                float sum = 0.0f;
                for (int j = 0; j < dim; j++) {
                    sum += state->qjl_matrix[i * dim + j] * residual[j];
                }
                s_times_r[i] = sum;
            }

            uint8_t* qjl_out = state->v_qjl + t * ((dim + 7) / 8);
            qkv_pack_signs(s_times_r, qjl_out, dim);
        }
    }

    return 1;
}

// ============================================================
// Public API - Dequantization
// ============================================================

int qkv_dequantize(
    const qkv_state_t* state,
    const qkv_config_t* config,
    float* key_output,
    float* value_output,
    int n_tokens
) {
    if (!state || !config || !key_output || !value_output || n_tokens <= 0) {
        return 0;
    }

    int dim = config->head_dim;
    bool use_qjl = config->enable_qjl && state->k_qjl && state->v_qjl;
    if (use_qjl && (state->k_bits <= 1 || state->v_bits <= 1)) {
        use_qjl = false;
    }

    for (int t = 0; t < n_tokens; t++) {
        // Dequantize K
        if (!qkv_dequant_one(
                state, config,
                state->k_idx, state->k_qjl,
                state->k_residual_norms, state->k_norms,
                t, state->k_bits, use_qjl,
                key_output + t * dim)) {
            return 0;
        }

        // Dequantize V
        if (!qkv_dequant_one(
                state, config,
                state->v_idx, state->v_qjl,
                state->v_residual_norms, state->v_norms,
                t, state->v_bits, use_qjl,
                value_output + t * dim)) {
            return 0;
        }
    }

    return 1;
}

// ============================================================
// Public API - Attention Decode (Paper Main Algorithm)
// ============================================================

int qkv_attention_decode(
    const float* query,
    const qkv_state_t* kv_state,
    const qkv_config_t* kv_config,
    uint32_t context_tokens,
    uint32_t head_dim,
    float* output
) {
    return qkv_attention_decode_impl(
        query, kv_state, kv_config,
        context_tokens, head_dim, output
    );
}
