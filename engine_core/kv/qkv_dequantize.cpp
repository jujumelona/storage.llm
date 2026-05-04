#include "qkv_dequantize.h"
#include "qkv_helpers.h"
#include "qkv_codebook.h"
#include "qkv_packing.h"
#include <math.h>
#include <string.h>
#include <climits>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int qkv_mse_bits_for_total_bits_dequant(int bits, bool use_qjl) {
    const int mse_bits = use_qjl ? bits - 1 : bits;
    return qkv_bits_valid(mse_bits) ? mse_bits : 0;
}

static int qkv_dequant_one_split(
    const qkv_state_t* s,
    const qkv_config_t* cfg,
    int target,
    const uint8_t* qjl,
    const float* residual_norms,
    const float* norms,
    int token_idx,
    bool use_qjl,
    float* output
) {
    if (!s || !cfg || !norms || !output || token_idx < 0) {
        return 0;
    }
    const int d = s->head_dim;
    const int n_out = cfg->outlier_channels;
    const int n_norm = d - n_out;
    const int out_mse_bits = qkv_mse_bits_for_total_bits_dequant(cfg->outlier_bits, use_qjl);
    const int norm_mse_bits = qkv_mse_bits_for_total_bits_dequant(cfg->normal_bits, use_qjl);
    if (d <= 0 || d > 16384 || n_out <= 0 || n_out >= d || n_norm <= 0 ||
        !out_mse_bits || !norm_mse_bits) {
        return 0;
    }
    const int* outlier_channels = qkv_outlier_indices_for_target_const(s, target);
    const uint8_t* split_outlier = qkv_idx_outlier_for_target_const(s, target);
    const uint8_t* split_normal = qkv_idx_normal_for_target_const(s, target);
    const uint8_t* is_outlier = qkv_is_outlier_for_target_const(s, target);
    if (!outlier_channels || !split_outlier || !split_normal || !is_outlier ||
        !s->scratch_indices || !s->scratch_y_tilde || !s->scratch_x_tilde) {
        return 0;
    }
    if (n_out > INT_MAX / out_mse_bits || n_norm > INT_MAX / norm_mse_bits) {
        return 0;
    }
    const int out_stride = (n_out * out_mse_bits + 7) / 8;
    const int norm_stride = (n_norm * norm_mse_bits + 7) / 8;
    const int qstride = (d + 7) / 8;
    if (out_stride <= 0 || norm_stride <= 0 ||
        token_idx > INT_MAX / std::max(out_stride, norm_stride)) {
        return 0;
    }
    const float* out_centroids = qkv_codebook_for_bits(s, out_mse_bits);
    const float* norm_centroids = qkv_codebook_for_bits(s, norm_mse_bits);
    if (!out_centroids || !norm_centroids) {
        return 0;
    }

    int* indices = s->scratch_indices;
    float* y_tilde = s->scratch_y_tilde;
    float* x_tilde = s->scratch_x_tilde;
    const int out_levels = 1 << out_mse_bits;
    const int norm_levels = 1 << norm_mse_bits;

    qkv_unpack_indices(split_outlier + (size_t)token_idx * (size_t)out_stride,
        indices, n_out, out_mse_bits);
    for (int i = 0; i < n_out; ++i) {
        const int channel = outlier_channels[i];
        if (channel < 0 || channel >= d || indices[i] < 0 || indices[i] >= out_levels) {
            return 0;
        }
        y_tilde[channel] = out_centroids[indices[i]];
    }

    qkv_unpack_indices(split_normal + (size_t)token_idx * (size_t)norm_stride,
        indices, n_norm, norm_mse_bits);
    int normal_pos = 0;
    for (int i = 0; i < d; ++i) {
        if (is_outlier[i]) {
            continue;
        }
        if (normal_pos >= n_norm || indices[normal_pos] < 0 || indices[normal_pos] >= norm_levels) {
            return 0;
        }
        y_tilde[i] = norm_centroids[indices[normal_pos++]];
    }
    if (normal_pos != n_norm) {
        return 0;
    }

    if (cfg->enable_rotation && s->rotation_matrix) {
        for (int i = 0; i < d; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < d; ++j) {
                sum += s->rotation_matrix[(size_t)j * (size_t)d + (size_t)i] * y_tilde[j];
            }
            x_tilde[i] = sum;
        }
    } else {
        memcpy(x_tilde, y_tilde, (size_t)d * sizeof(float));
    }

    if (use_qjl && qjl && residual_norms && s->qjl_matrix) {
        const float r_norm = residual_norms[token_idx];
        if (r_norm > 1e-10f) {
            const uint8_t* tqjl = qjl + (size_t)token_idx * (size_t)qstride;
            float* qjl_signs = s->scratch_qjl_signs;
            float* s_t_qjl = s->scratch_s_t_qjl;
            if (!qjl_signs || !s_t_qjl) return 0;
            qkv_unpack_signs(tqjl, qjl_signs, d);
            for (int i = 0; i < d; ++i) {
                float sum = 0.0f;
                for (int j = 0; j < d; ++j) {
                    sum += s->qjl_matrix[(size_t)j * (size_t)d + (size_t)i] * qjl_signs[j];
                }
                s_t_qjl[i] = sum;
            }
            const float qjl_scale = sqrtf((float)M_PI / 2.0f) / (float)d;
            for (int i = 0; i < d; ++i) {
                x_tilde[i] += qjl_scale * r_norm * s_t_qjl[i];
            }
        }
    }

    const float norm = norms[token_idx];
    for (int i = 0; i < d; ++i) {
        output[i] = x_tilde[i] * norm;
    }
    return 1;
}

// Paper Algorithm 2: TurboQuant_prod dequantization
// x_hat = Pi^T * y_hat_mse + sqrt(pi/2) / d * ||r|| * S^T * sign(S*r)
int qkv_dequant_one(
    const qkv_state_t* s,
    const qkv_config_t* cfg,
    const uint8_t* idx,
    const uint8_t* qjl,
    const float* residual_norms,
    const float* norms,
    int token_idx,
    int bits,
    bool use_qjl,
    float* output
) {
    if (!s || !cfg || !idx || !norms || !output || token_idx < 0) {
        return 0;
    }
    if (token_idx >= s->n_tokens) {
        return 0;
    }

    // BUGFIX 370: head_dim 유효성 체크
    const int d = s->head_dim;
    if (d <= 0 || d > 16384) {
        return 0;
    }
    const int mse_bits = use_qjl ? bits - 1 : bits;
    if (!qkv_bits_valid(mse_bits)) {
        return 0;
    }
    const int split_target = qkv_target_from_buffers(s, idx, norms);
    if (split_target && qkv_outlier_split_ready(s, cfg, split_target)) {
        return qkv_dequant_one_split(
            s, cfg, split_target, qjl, residual_norms, norms,
            token_idx, use_qjl, output);
    }
    // BUGFIX 371: stride 계산 overflow 방지
    if (d > INT_MAX / mse_bits) {
        return 0;
    }
    const int stride = (d * mse_bits + 7) / 8;
    const int qstride = (d + 7) / 8;

    // Step 1: Unpack MSE indices
    // BUGFIX 372: token_idx overflow 방지
    if (token_idx > INT_MAX / stride) {
        return 0;
    }
    const uint8_t* tidx = idx + (size_t)token_idx * (size_t)stride;
    int* indices = s->scratch_indices;
    float* y_tilde = s->scratch_y_tilde;
    if (!indices || !y_tilde) return 0;

    qkv_unpack_indices(tidx, indices, d, mse_bits);

    // Step 4: Lookup centroids (in rotated space)
    const float* centroids = qkv_codebook_for_bits(s, mse_bits);
    // BUGFIX 483: centroids null 체크
    if (!centroids) return 0;
    const int max_idx = (1 << mse_bits);
    for (int i = 0; i < d; i++) {
        // BUGFIX 484: indices 범위 체크
        if (indices[i] < 0 || indices[i] >= max_idx) return 0;
        y_tilde[i] = centroids[indices[i]];
    }

    // Step 3: Apply inverse rotation Pi^T
    float* x_tilde = s->scratch_x_tilde;
    if (!x_tilde) return 0;

    if (cfg->enable_rotation && s->rotation_matrix) {
        // x_tilde = Pi^T * y_tilde
        // BUGFIX 373: rotation_matrix 범위 체크
        for (int i = 0; i < d; i++) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                // Pi^T[i,j] = Pi[j,i] (transpose)
                size_t idx = (size_t)j * (size_t)d + (size_t)i;
                sum += s->rotation_matrix[idx] * y_tilde[j];
            }
            x_tilde[i] = sum;
        }
    } else {
        // BUGFIX 449: d * sizeof(float) overflow 방지
        if (d > INT_MAX / (int)sizeof(float)) return 0;
        memcpy(x_tilde, y_tilde, d * sizeof(float));
    }

    // Step 4: Add QJL residual if enabled (Paper Algorithm 2)
    // residual = sqrt(pi/2) / d * ||r|| * S^T * sign(S*r)
    if (use_qjl && qjl && residual_norms && s->qjl_matrix) {
        const float r_norm = residual_norms[token_idx];
        if (r_norm > 1e-10f) {
            // BUGFIX 374: qjl token_idx overflow 방지
            if (token_idx > INT_MAX / qstride) {
                return 0;
            }
            const uint8_t* tqjl = qjl + (size_t)token_idx * (size_t)qstride;
            float* qjl_signs = s->scratch_qjl_signs;
            float* s_t_qjl = s->scratch_s_t_qjl;
            if (!qjl_signs || !s_t_qjl) return 0;

            // Unpack signs
            qkv_unpack_signs(tqjl, qjl_signs, d);

            // Compute S^T * qjl_signs
            // BUGFIX 375: qjl_matrix 범위 체크
            for (int i = 0; i < d; i++) {
                float sum = 0.0f;
                for (int j = 0; j < d; j++) {
                    // S^T[i,j] = S[j,i]
                    size_t idx = (size_t)j * (size_t)d + (size_t)i;
                    sum += s->qjl_matrix[idx] * qjl_signs[j];
                }
                s_t_qjl[i] = sum;
            }

            // Add normalized residual: x_hat = x_tilde + scale * ||r|| * S^T * z
            // BUGFIX 376: d가 0일 때 division by zero 방지
            if (d <= 0) return 0;
            const float qjl_scale = sqrtf((float)M_PI / 2.0f) / (float)d;
            for (int i = 0; i < d; i++) {
                x_tilde[i] += qjl_scale * r_norm * s_t_qjl[i];
            }
        }
    }

    // Step 5: Denormalize by stored norm
    const float norm = norms[token_idx];
    for (int i = 0; i < d; i++) {
        output[i] = x_tilde[i] * norm;
    }

    return 1;
}

// Dot product with MSE split rotated token (for outlier channels)
int qkv_dot_mse_split_rotated_token(
    const qkv_state_t* s,
    const qkv_config_t* cfg,
    int target,
    int token_idx,
    const float* q_rotated,
    float* out_dot
) {
    if (!s || !cfg || !q_rotated || !out_dot) return 0;

    // BUGFIX 377: head_dim 유효성 체크
    const int d = s->head_dim;
    if (d <= 0 || d > 16384) return 0;

    const int n_out = cfg->outlier_channels;
    // BUGFIX 378: outlier_channels 범위 체크
    if (n_out < 0 || n_out > d) return 0;
    const int n_norm = d - n_out;
    const bool use_qjl = cfg->enable_qjl && s->k_qjl && s->v_qjl && s->k_bits > 1 && s->v_bits > 1;
    const int out_bits = qkv_mse_bits_for_total_bits_dequant(cfg->outlier_bits, use_qjl);
    const int norm_bits = qkv_mse_bits_for_total_bits_dequant(cfg->normal_bits, use_qjl);
    if (!out_bits || !norm_bits) return 0;

    const int* outlier_channels = qkv_outlier_indices_for_target_const(s, target);
    const uint8_t* split_outlier = qkv_idx_outlier_for_target_const(s, target);
    const uint8_t* split_normal = qkv_idx_normal_for_target_const(s, target);
    const uint8_t* is_outlier = qkv_is_outlier_for_target_const(s, target);

    if (!outlier_channels || !split_outlier || !split_normal || !is_outlier) {
        return 0;
    }

    const float* out_centroids = qkv_codebook_for_bits(s, out_bits);
    const float* norm_centroids = qkv_codebook_for_bits(s, norm_bits);

    // BUGFIX 379: packed_size overflow 방지
    if (n_out > INT_MAX / out_bits || n_norm > INT_MAX / norm_bits) {
        return 0;
    }
    const int out_packed_size = (n_out * out_bits + 7) / 8;
    const int norm_packed_size = (n_norm * norm_bits + 7) / 8;

    int* indices = s->scratch_indices;
    if (!indices) return 0;

    float dot = 0.0f;

    // Outlier channels
    // BUGFIX 380: token_idx overflow 방지
    if (token_idx < 0 || (out_packed_size > 0 && token_idx > INT_MAX / out_packed_size)) {
        return 0;
    }
    qkv_unpack_indices(split_outlier + (size_t)token_idx * (size_t)out_packed_size, indices, n_out, out_bits);
    for (int i = 0; i < n_out; i++) {
        const int channel = outlier_channels[i];
        if (channel < 0 || channel >= d) return 0;
        dot += q_rotated[channel] * out_centroids[indices[i]];
    }

    // Normal channels
    // BUGFIX 381: token_idx overflow 방지
    if (norm_packed_size > 0 && token_idx > INT_MAX / norm_packed_size) {
        return 0;
    }
    qkv_unpack_indices(split_normal + (size_t)token_idx * (size_t)norm_packed_size, indices, n_norm, norm_bits);
    int normal_pos = 0;
    for (int i = 0; i < d; i++) {
        if (is_outlier[i]) continue;
        // BUGFIX 382: normal_pos 범위 체크
        if (normal_pos >= n_norm) return 0;
        dot += q_rotated[i] * norm_centroids[indices[normal_pos++]];
    }

    *out_dot = dot;
    return 1;
}
