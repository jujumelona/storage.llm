#include "qkv_dequantize.h"
#include "qkv_helpers.h"
#include "qkv_codebook.h"
#include "qkv_packing.h"
#include "qkv_matrix.h"
#include <cmath>
#include <math.h>
#include <string.h>
#include <climits>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


static int qkv_apply_split_rotation_inverse_deq(
    const qkv_config_t* cfg,
    const float* matrix,
    const float* signs,
    const float* input,
    float* output,
    int dim
) {
    if (!input || !output || dim <= 0 || dim > 16384) return 0;
    if (cfg && cfg->enable_rotation && matrix) {
        if (signs && qkv_apply_hadamard_rotation_inverse(input, signs, output, dim)) return 1;
        for (int i = 0; i < dim; ++i) output[i] = 0.0f;
        for (int j = 0; j < dim; ++j) {
            const float y = input[j];
            const float* row = matrix + (size_t)j * (size_t)dim;
            for (int i = 0; i < dim; ++i) output[i] += row[i] * y;
        }
        for (int i = 0; i < dim; ++i) if (!std::isfinite(output[i])) return 0;
        return 1;
    }
    memcpy(output, input, (size_t)dim * sizeof(float));
    return 1;
}

static int qkv_project_qjl_t_deq(const float* matrix, const float* signs, float* out, int dim) {
    if (!matrix || !signs || !out || dim <= 0 || dim > 16384) return 0;
    for (int i = 0; i < dim; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < dim; ++j) sum += matrix[(size_t)j * (size_t)dim + (size_t)i] * signs[j];
        if (!std::isfinite(sum)) return 0;
        out[i] = sum;
    }
    return 1;
}

static const float* qkv_deq_split_norms(const qkv_state_t* s, int target, bool outlier) {
    if (!s) return nullptr;
    if (target == QKV_TARGET_KEY) return outlier ? s->k_norms_outlier : s->k_norms_normal;
    if (target == QKV_TARGET_VALUE) return outlier ? s->v_norms_outlier : s->v_norms_normal;
    return nullptr;
}

static const float* qkv_deq_split_residual_norms(const qkv_state_t* s, int target, bool outlier) {
    if (!s) return nullptr;
    if (target == QKV_TARGET_KEY) return outlier ? s->k_residual_norms_outlier : s->k_residual_norms_normal;
    if (target == QKV_TARGET_VALUE) return outlier ? s->v_residual_norms_outlier : s->v_residual_norms_normal;
    return nullptr;
}

static int qkv_mse_bits_for_total_bits_dequant(int bits, bool use_qjl) {
    if (qkv_bits_raw(bits)) return bits;
    const int mse_bits = use_qjl ? bits - 1 : bits;
    return qkv_bits_codebook(mse_bits) ? mse_bits : 0;
}

static float qkv_load_raw_scalar(const uint8_t* src, int index, int bits) {
    if (!src || index < 0) return 0.0f;
    if (bits == 16) {
        const uint16_t h = (uint16_t)src[(size_t)index * 2u] |
            ((uint16_t)src[(size_t)index * 2u + 1u] << 8);
        return qkv_fp16_bits_to_float(h);
    }
    if (bits == 32) {
        float out = 0.0f;
        memcpy(&out, src + (size_t)index * sizeof(float), sizeof(float));
        return out;
    }
    return 0.0f;
}

static void qkv_load_raw_vector(const uint8_t* src, float* dst, int dim, int bits) {
    if (!src || !dst || dim <= 0) return;
    if (bits == 32) {
        memcpy(dst, src, (size_t)dim * sizeof(float));
        return;
    }
    for (int i = 0; i < dim; ++i) {
        dst[i] = qkv_load_raw_scalar(src, i, bits);
    }
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
    (void)residual_norms;
    (void)norms;
    if (!s || !cfg || !output || token_idx < 0) return 0;
    const int d = s->head_dim;
    const int n_out = cfg->outlier_channels;
    const int n_norm = d - n_out;
    if (d <= 0 || d > 16384 || n_out <= 0 || n_out >= d || n_norm <= 0) return 0;
    const int* outlier_channels = qkv_outlier_indices_for_target_const(s, target);
    const uint8_t* split_outlier = qkv_idx_outlier_for_target_const(s, target);
    const uint8_t* split_normal = qkv_idx_normal_for_target_const(s, target);
    const uint8_t* is_outlier = qkv_is_outlier_for_target_const(s, target);
    if (!outlier_channels || !split_outlier || !split_normal || !is_outlier ||
        !s->scratch_indices || !s->scratch_y_tilde || !s->scratch_x_tilde ||
        !s->scratch_qjl_signs || !s->scratch_s_t_qjl) return 0;

    const int outlier_bits = qkv_outlier_bits_for_target(cfg, target);
    const int normal_bits = qkv_normal_bits_for_target(cfg, target);
    const bool split_use_qjl = use_qjl && qjl && s->qjl_matrix_outlier && s->qjl_matrix_normal &&
        qkv_bits_codebook(outlier_bits) && qkv_bits_codebook(normal_bits) &&
        outlier_bits > 1 && normal_bits > 1;
    const int out_bits = qkv_mse_bits_for_total_bits_dequant(outlier_bits, split_use_qjl);
    const int norm_bits = qkv_mse_bits_for_total_bits_dequant(normal_bits, split_use_qjl);
    if (!qkv_bits_valid(out_bits) || !qkv_bits_valid(norm_bits) ||
        n_out > INT_MAX / out_bits || n_norm > INT_MAX / norm_bits) return 0;
    const int out_stride = (n_out * out_bits + 7) / 8;
    const int norm_stride = (n_norm * norm_bits + 7) / 8;
    const size_t qjl_out_stride = qkv_split_qjl_outlier_bytes(cfg);
    const size_t qjl_norm_stride = qkv_split_qjl_normal_bytes(cfg);
    const size_t qjl_stride = qkv_qjl_token_bytes(s);
    if (split_use_qjl && (qjl_stride < qjl_out_stride + qjl_norm_stride)) return 0;
    const uint8_t* qjl_token = split_use_qjl ? qjl + (size_t)token_idx * qjl_stride : nullptr;

    memset(output, 0, (size_t)d * sizeof(float));

    auto deq_group = [&](bool outlier_group) -> int {
        const int gd = outlier_group ? n_out : n_norm;
        const int bits = outlier_group ? out_bits : norm_bits;
        const uint8_t* src = (outlier_group ? split_outlier : split_normal) +
            (size_t)token_idx * (size_t)((gd * bits + 7) / 8);
        const float* group_norms = qkv_deq_split_norms(s, target, outlier_group);
        const float* group_rnorms = qkv_deq_split_residual_norms(s, target, outlier_group);
        const float* rot = outlier_group ? s->rotation_matrix_outlier : s->rotation_matrix_normal;
        const float* rot_signs = outlier_group ? s->rotation_signs_outlier : s->rotation_signs_normal;
        const float* qjl_mat = outlier_group ? s->qjl_matrix_outlier : s->qjl_matrix_normal;
        const uint8_t* qjl_src = nullptr;
        if (split_use_qjl) qjl_src = outlier_group ? qjl_token : (qjl_token + qjl_out_stride);
        if (!group_norms || (split_use_qjl && (!group_rnorms || !qjl_mat || !qjl_src))) return 0;
        int* indices = s->scratch_indices;
        float* y = s->scratch_y_tilde;
        float* x = s->scratch_x_tilde;
        const bool raw = qkv_bits_raw(bits);
        const float* centroids = raw ? nullptr : qkv_codebook_for_bits_dim(bits, gd, cfg->codebook_distribution);
        if (!raw && !centroids) return 0;
        const int levels = raw ? 0 : (1 << bits);
        if (!raw) qkv_unpack_indices(src, indices, gd, bits);
        for (int i = 0; i < gd; ++i) {
            if (raw) y[i] = qkv_load_raw_scalar(src, i, bits);
            else {
                if (indices[i] < 0 || indices[i] >= levels) return 0;
                y[i] = centroids[indices[i]];
            }
        }
        if (!qkv_apply_split_rotation_inverse_deq(cfg, rot, rot_signs, y, x, gd)) return 0;
        if (split_use_qjl) {
            const float r_norm = group_rnorms[token_idx];
            if (r_norm > 1e-10f) {
                float* signs = s->scratch_qjl_signs;
                float* stz = s->scratch_s_t_qjl;
                qkv_unpack_signs(qjl_src, signs, gd);
                if (!qkv_project_qjl_t_deq(qjl_mat, signs, stz, gd)) return 0;
                const float qjl_scale = sqrtf((float)M_PI / 2.0f) / (float)gd;
                if (!std::isfinite(qjl_scale)) return 0;
                for (int i = 0; i < gd; ++i) x[i] += qjl_scale * r_norm * stz[i];
            }
        }
        const float norm = group_norms[token_idx];
        if (!std::isfinite(norm) || norm < 1e-12f) {
            return 1;
        }
        if (outlier_group) {
            for (int i = 0; i < gd; ++i) {
                const int ch = outlier_channels[i];
                if (ch < 0 || ch >= d) return 0;
                output[ch] = x[i] * norm;
            }
        } else {
            int pos = 0;
            for (int ch = 0; ch < d; ++ch) {
                if (is_outlier[ch]) continue;
                if (pos >= gd) return 0;
                output[ch] = x[pos++] * norm;
            }
            if (pos != gd) return 0;
        }
        return 1;
    };

    if (!deq_group(true)) return 0;
    if (!deq_group(false)) return 0;
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
    if ((uint32_t)token_idx < s->sink_tokens) {
        const int target = qkv_target_from_buffers(s, idx, norms);
        const float* exact = target == QKV_TARGET_KEY ? s->k_sink :
            target == QKV_TARGET_VALUE ? s->v_sink : nullptr;
        if (exact) {
            memcpy(output, exact + (size_t)token_idx * (size_t)d, (size_t)d * sizeof(float));
            return 1;
        }
    }
    const bool base_use_qjl = use_qjl && bits > 1;
    const int mse_bits = base_use_qjl ? bits - 1 : bits;
    if (!qkv_bits_valid(mse_bits)) {
        return 0;
    }
    const int split_target = qkv_target_from_buffers(s, idx, norms);
    if (split_target && qkv_outlier_split_ready(s, cfg, split_target)) {
        const int outlier_bits = qkv_outlier_bits_for_target(cfg, split_target);
        const int normal_bits = qkv_normal_bits_for_target(cfg, split_target);
        const bool split_use_qjl = base_use_qjl &&
            qkv_bits_codebook(outlier_bits) && qkv_bits_codebook(normal_bits) &&
            outlier_bits > 1 && normal_bits > 1;
        return qkv_dequant_one_split(
            s, cfg, split_target, qjl, residual_norms, norms,
            token_idx, split_use_qjl, output);
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

    if (qkv_bits_raw(mse_bits)) {
        qkv_load_raw_vector(tidx, y_tilde, d, mse_bits);
    } else {
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
    }

    // Step 3: Apply inverse rotation Pi^T
    float* x_tilde = s->scratch_x_tilde;
    if (!x_tilde) return 0;

    if (cfg->enable_rotation && s->rotation_matrix) {
        if (s->rotation_signs &&
            qkv_apply_hadamard_rotation_inverse(y_tilde, s->rotation_signs, x_tilde, d)) {
            // Fast inverse rotation path for Hadamard-backed QKV states.
        } else {
        // x_tilde = Pi^T * y_tilde
        // BUGFIX 373: rotation_matrix 범위 체크
        for (int i = 0; i < d; i++) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                // Pi^T[i,j] = Pi[j,i] (transpose)
                size_t idx = (size_t)j * (size_t)d + (size_t)i;
                sum += s->rotation_matrix[idx] * y_tilde[j];
            }
            // BUGFIX 736: Check inverse rotation result for NaN/Inf (main path) ★★
            if (!std::isfinite(sum)) {
                sum = 0.0f;
            }
            x_tilde[i] = sum;
        }
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
                // BUGFIX 726: Check QJL matrix multiplication result for NaN/Inf ★★
                if (!std::isfinite(sum)) {
                    sum = 0.0f;
                }
                s_t_qjl[i] = sum;
            }

            // Add normalized residual: x_hat = x_tilde + scale * ||r|| * S^T * z
            // BUGFIX 376: d가 0일 때 division by zero 방지
            if (d <= 0) return 0;
            const float qjl_scale = sqrtf((float)M_PI / 2.0f) / (float)d;
            // BUGFIX 727: Check QJL scale for NaN/Inf ★★
            if (!std::isfinite(qjl_scale)) return 0;
            for (int i = 0; i < d; i++) {
                float residual_term = qjl_scale * r_norm * s_t_qjl[i];
                // BUGFIX 728: Check residual term for NaN/Inf before adding ★★
                if (std::isfinite(residual_term)) {
                    x_tilde[i] += residual_term;
                }
            }
        }
    }

    // Step 5: Denormalize by stored norm
    const float norm = norms[token_idx];
    // BUGFIX 656: Handle zero norm consistently with quantization ★★
    // Problem: Quantization skips zero vectors (l2_norm < 1e-12f early return)
    //          but dequantization multiplies by 0 → inconsistent behavior
    // Solution: Return zero vector explicitly when norm is too small
    // Impact: Consistent quantization/dequantization behavior → accurate PPL
    // BUGFIX 729: Check norm for NaN/Inf
    if (!std::isfinite(norm) || norm < 1e-12f) {
        memset(output, 0, (size_t)d * sizeof(float));
        return 1;
    }
    for (int i = 0; i < d; i++) {
        float result = x_tilde[i] * norm;
        // BUGFIX 730: Check final denormalized result for NaN/Inf
        if (!std::isfinite(result)) {
            result = 0.0f;
        }
        output[i] = result;
    }

    return 1;
}

// Dot product with MSE split rotated token (for outlier channels)
int qkv_dot_mse_split_rotated_token(
    const qkv_state_t* s,
    const qkv_config_t* cfg,
    int target,
    int token_idx,
    const float* query,
    float* out_dot
) {
    if (!s || !cfg || !query || !out_dot || token_idx < 0) return 0;
    const int d = s->head_dim;
    const int n_out = cfg->outlier_channels;
    const int n_norm = d - n_out;
    if (d <= 0 || d > 16384 || n_out <= 0 || n_out >= d || n_norm <= 0) return 0;
    const int* outlier_channels = qkv_outlier_indices_for_target_const(s, target);
    const uint8_t* is_outlier = qkv_is_outlier_for_target_const(s, target);
    const uint8_t* split_outlier = qkv_idx_outlier_for_target_const(s, target);
    const uint8_t* split_normal = qkv_idx_normal_for_target_const(s, target);
    if (!outlier_channels || !is_outlier || !split_outlier || !split_normal ||
        !s->scratch_residual || !s->scratch_rotated_q || !s->scratch_indices ||
        !s->scratch_qjl_signs || !s->scratch_s_times_r) return 0;

    const int outlier_bits = qkv_outlier_bits_for_target(cfg, target);
    const int normal_bits = qkv_normal_bits_for_target(cfg, target);
    const uint8_t* qjl_base = (target == QKV_TARGET_KEY) ? s->k_qjl : s->v_qjl;
    const bool split_qjl = cfg->enable_qjl && qjl_base && s->qjl_matrix_outlier && s->qjl_matrix_normal &&
        qkv_bits_codebook(outlier_bits) && qkv_bits_codebook(normal_bits) &&
        outlier_bits > 1 && normal_bits > 1;
    const int out_bits = qkv_mse_bits_for_total_bits_dequant(outlier_bits, split_qjl);
    const int norm_bits = qkv_mse_bits_for_total_bits_dequant(normal_bits, split_qjl);
    if (!qkv_bits_valid(out_bits) || !qkv_bits_valid(norm_bits)) return 0;
    const size_t qjl_out_stride = qkv_split_qjl_outlier_bytes(cfg);
    const size_t qjl_stride = qkv_qjl_token_bytes(s);

    auto group_dot = [&](bool outlier_group, float* acc) -> int {
        const int gd = outlier_group ? n_out : n_norm;
        const int bits = outlier_group ? out_bits : norm_bits;
        const uint8_t* src = (outlier_group ? split_outlier : split_normal) +
            (size_t)token_idx * (size_t)((gd * bits + 7) / 8);
        const float* group_norms = qkv_deq_split_norms(s, target, outlier_group);
        const float* group_rnorms = qkv_deq_split_residual_norms(s, target, outlier_group);
        const float* rot = outlier_group ? s->rotation_matrix_outlier : s->rotation_matrix_normal;
        const float* signs = outlier_group ? s->rotation_signs_outlier : s->rotation_signs_normal;
        const float* qjl_mat = outlier_group ? s->qjl_matrix_outlier : s->qjl_matrix_normal;
        if (!group_norms) return 0;
        float* q_subset = s->scratch_residual;
        float* q_rot = s->scratch_rotated_q;
        int* codes = s->scratch_indices;
        if (outlier_group) {
            for (int i = 0; i < gd; ++i) {
                const int ch = outlier_channels[i];
                if (ch < 0 || ch >= d) return 0;
                q_subset[i] = query[ch];
            }
        } else {
            int pos = 0;
            for (int ch = 0; ch < d; ++ch) {
                if (is_outlier[ch]) continue;
                if (pos >= gd) return 0;
                q_subset[pos++] = query[ch];
            }
            if (pos != gd) return 0;
        }
        if (!qkv_apply_split_rotation_inverse_deq(cfg, rot, signs, q_subset, q_rot, gd)) return 0;
        // The helper above applies Pi^T; for an orthogonal matrix this is not
        // the forward query rotation. Use direct forward multiply when rotation
        // is present. The inverse helper handles no-rotation identity.
        if (cfg->enable_rotation && rot) {
            if (signs && qkv_apply_hadamard_rotation_forward(q_subset, signs, q_rot, gd)) {
                // ok
            } else {
                for (int i = 0; i < gd; ++i) {
                    float sum = 0.0f;
                    const float* row = rot + (size_t)i * (size_t)gd;
                    for (int j = 0; j < gd; ++j) sum += row[j] * q_subset[j];
                    if (!std::isfinite(sum)) return 0;
                    q_rot[i] = sum;
                }
            }
        }

        const bool raw = qkv_bits_raw(bits);
        const float* centroids = raw ? nullptr : qkv_codebook_for_bits_dim(bits, gd, cfg->codebook_distribution);
        if (!raw && !centroids) return 0;
        const int levels = raw ? 0 : (1 << bits);
        float dot = 0.0f;
        if (!raw) qkv_unpack_indices(src, codes, gd, bits);
        for (int i = 0; i < gd; ++i) {
            float kv = 0.0f;
            if (raw) kv = qkv_load_raw_scalar(src, i, bits);
            else {
                if (codes[i] < 0 || codes[i] >= levels) return 0;
                kv = centroids[codes[i]];
            }
            dot += q_rot[i] * kv;
        }
        if (split_qjl && qjl_base && group_rnorms && qjl_mat) {
            const float r_norm = group_rnorms[token_idx];
            if (r_norm > 1e-10f) {
                const uint8_t* qjl_token = qjl_base + (size_t)token_idx * qjl_stride;
                const uint8_t* qjl_src = outlier_group ? qjl_token : (qjl_token + qjl_out_stride);
                float* signs_buf = s->scratch_qjl_signs;
                float* sq = s->scratch_s_times_r;
                qkv_unpack_signs(qjl_src, signs_buf, gd);
                if (!qkv_project_qjl_t_deq(qjl_mat, signs_buf, sq, gd)) return 0;
                const float qjl_scale = sqrtf((float)M_PI / 2.0f) / (float)gd;
                for (int i = 0; i < gd; ++i) dot += qjl_scale * r_norm * sq[i] * q_subset[i];
            }
        }
        const float norm = group_norms[token_idx];
        if (!std::isfinite(norm)) return 0;
        *acc += dot * norm;
        return 1;
    };

    float acc = 0.0f;
    if (!group_dot(true, &acc)) return 0;
    if (!group_dot(false, &acc)) return 0;
    if (!std::isfinite(acc)) return 0;
    *out_dot = acc;
    return 1;
}
