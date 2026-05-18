// QKV Implementation - Faithful to TurboQuant Paper (2504.19874v1)
// 100% Complete Implementation
// Guardrail: do not alter QKV to chase PPL unless the change is backed by
// QKV.pdf / TurboQuant semantics. PPL correctness debugging should first stay
// in API, GraphIR, RoPE, norm/residual, MLP/router, final-norm, and LM-head.

#include "kv_qkv.h"
#include "qkv_state.h"
#include "qkv_quantize.h"
#include "qkv_dequantize.h"
#include "qkv_attention.h"
#include "qkv_helpers.h"
#include "qkv_codebook.h"
#include "qkv_packing.h"
#include "qkv_matrix.h"
#include <atomic>
#include <cmath>
#include <math.h>
#include <string.h>
#include <algorithm>
#include <climits>
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Global mode flag. StorageLLM offload-native models use packed QKV by default;
// legacy callers can still explicitly disable it for debug comparisons.
std::atomic<bool> g_qkv_mode_enabled{true};

// BUGFIX Problem 2: Windows QKV working set preparation ★★ IMPORTANT
// Problem: QKV state memory (~0.69 GiB) allocated independently, not included in working set
// → Page faults during attention operations block io_worker threads
// Solution: Forward-declare prepare_windows_staging_working_set and call after QKV allocation
// Impact: Eliminates blocking page faults on Windows during attention
#ifdef _WIN32
extern void prepare_windows_staging_working_set(uint64_t bytes);
#else
static inline void prepare_windows_staging_working_set(uint64_t) {}
#endif

// ============================================================
// BUGFIX Issue 9: Fast Walsh-Hadamard Transform (FWHT)
// ============================================================
// Problem: Rotation matrix multiplication is O(dim²) but for power-of-2 dim,
// the Hadamard structure allows O(dim·log₂dim) computation via FWHT
// Solution: Implement in-place FWHT for power-of-2 dimensions
// Impact: 18x speedup for dim=128 (16,384 ops → 896 ops)
static void fwht_inplace(float* data, int dim) {
    // In-place Fast Walsh-Hadamard Transform
    // Assumes dim is power of 2
    for (int h = 1; h < dim; h *= 2) {
        for (int i = 0; i < dim; i += h * 2) {
            for (int j = i; j < i + h; ++j) {
                const float a = data[j];
                const float b = data[j + h];
                data[j] = a + b;
                data[j + h] = a - b;
            }
        }
    }
}


static const float* qkv_split_rotation_matrix_for_group(const qkv_state_t* s, bool outlier) {
    if (!s) return nullptr;
    return outlier ? s->rotation_matrix_outlier : s->rotation_matrix_normal;
}

static const float* qkv_split_rotation_signs_for_group(const qkv_state_t* s, bool outlier) {
    if (!s) return nullptr;
    return outlier ? s->rotation_signs_outlier : s->rotation_signs_normal;
}

static const float* qkv_split_qjl_matrix_for_group(const qkv_state_t* s, bool outlier) {
    if (!s) return nullptr;
    return outlier ? s->qjl_matrix_outlier : s->qjl_matrix_normal;
}

static float* qkv_split_norms_for_target(qkv_state_t* s, int target, bool outlier) {
    if (!s) return nullptr;
    if (target == QKV_TARGET_KEY) return outlier ? s->k_norms_outlier : s->k_norms_normal;
    if (target == QKV_TARGET_VALUE) return outlier ? s->v_norms_outlier : s->v_norms_normal;
    return nullptr;
}

static const float* qkv_split_norms_for_target_const(const qkv_state_t* s, int target, bool outlier) {
    if (!s) return nullptr;
    if (target == QKV_TARGET_KEY) return outlier ? s->k_norms_outlier : s->k_norms_normal;
    if (target == QKV_TARGET_VALUE) return outlier ? s->v_norms_outlier : s->v_norms_normal;
    return nullptr;
}

static float* qkv_split_residual_norms_for_target(qkv_state_t* s, int target, bool outlier) {
    if (!s) return nullptr;
    if (target == QKV_TARGET_KEY) return outlier ? s->k_residual_norms_outlier : s->k_residual_norms_normal;
    if (target == QKV_TARGET_VALUE) return outlier ? s->v_residual_norms_outlier : s->v_residual_norms_normal;
    return nullptr;
}

static const float* qkv_split_residual_norms_for_target_const(const qkv_state_t* s, int target, bool outlier) {
    if (!s) return nullptr;
    if (target == QKV_TARGET_KEY) return outlier ? s->k_residual_norms_outlier : s->k_residual_norms_normal;
    if (target == QKV_TARGET_VALUE) return outlier ? s->v_residual_norms_outlier : s->v_residual_norms_normal;
    return nullptr;
}

static int qkv_apply_split_rotation_forward(
    const qkv_config_t* cfg,
    const float* matrix,
    const float* signs,
    const float* input,
    float* output,
    int dim
) {
    if (!input || !output || dim <= 0 || dim > 16384) return 0;
    if (cfg && cfg->enable_rotation && matrix) {
        if (signs && qkv_apply_hadamard_rotation_forward(input, signs, output, dim)) return 1;
        for (int i = 0; i < dim; ++i) {
            float sum = 0.0f;
            const float* row = matrix + (size_t)i * (size_t)dim;
            for (int j = 0; j < dim; ++j) sum += row[j] * input[j];
            if (!std::isfinite(sum)) return 0;
            output[i] = sum;
        }
        return 1;
    }
    memcpy(output, input, (size_t)dim * sizeof(float));
    return 1;
}

static int qkv_apply_split_rotation_inverse(
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

static int qkv_project_qjl(const float* matrix, const float* vec, float* out, int dim) {
    if (!matrix || !vec || !out || dim <= 0 || dim > 16384) return 0;
    for (int i = 0; i < dim; ++i) {
        float sum = 0.0f;
        const float* row = matrix + (size_t)i * (size_t)dim;
        for (int j = 0; j < dim; ++j) sum += row[j] * vec[j];
        if (!std::isfinite(sum)) return 0;
        out[i] = sum;
    }
    return 1;
}

static void qkv_store_raw_scalar(uint8_t* dst, int index, float value, int bits) {
    if (!dst || index < 0) return;
    if (bits == 16) {
        const uint16_t h = qkv_float_to_fp16_bits(value);
        dst[(size_t)index * 2u] = (uint8_t)(h & 0xffu);
        dst[(size_t)index * 2u + 1u] = (uint8_t)((h >> 8) & 0xffu);
    } else if (bits == 32) {
        memcpy(dst + (size_t)index * sizeof(float), &value, sizeof(float));
    }
}

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

static int qkv_mse_bits_for_total_bits(int bits, bool use_qjl) {
    if (qkv_bits_raw(bits)) return bits;
    const int mse_bits = use_qjl ? bits - 1 : bits;
    return qkv_bits_codebook(mse_bits) ? mse_bits : 0;
}

static int qkv_quantize_split_vector_with_state(
    qkv_state_t* state,
    const qkv_config_t* config,
    int target,
    const float* input,
    int token_idx,
    float* norm_out,
    uint8_t* qjl_base,
    float* residual_norms
) {
    if (!state || !config || !input || !norm_out || token_idx < 0 ||
        token_idx >= state->n_tokens) {
        return 0;
    }
    const int dim = config->head_dim;
    const int n_out = config->outlier_channels;
    const int n_norm = dim - n_out;
    if (dim <= 0 || dim > 16384 || n_out <= 0 || n_out >= dim || n_norm <= 0) {
        return 0;
    }

    uint8_t* split_outlier = qkv_idx_outlier_for_target(state, target);
    uint8_t* split_normal = qkv_idx_normal_for_target(state, target);
    const int* outlier_channels = qkv_outlier_indices_for_target_const(state, target);
    const uint8_t* is_outlier = qkv_is_outlier_for_target_const(state, target);
    if (!split_outlier || !split_normal || !outlier_channels || !is_outlier ||
        !state->scratch_residual || !state->scratch_s_times_r ||
        !state->scratch_indices || !state->scratch_y_tilde || !state->scratch_x_tilde) {
        return 0;
    }

    const int outlier_bits = qkv_outlier_bits_for_target(config, target);
    const int normal_bits = qkv_normal_bits_for_target(config, target);
    const bool use_qjl = config->enable_qjl && qjl_base && residual_norms &&
        state->qjl_matrix_outlier && state->qjl_matrix_normal &&
        qkv_bits_codebook(outlier_bits) && qkv_bits_codebook(normal_bits) &&
        outlier_bits > 1 && normal_bits > 1;
    const int out_mse_bits = qkv_mse_bits_for_total_bits(outlier_bits, use_qjl);
    const int norm_mse_bits = qkv_mse_bits_for_total_bits(normal_bits, use_qjl);
    if (!qkv_bits_valid(out_mse_bits) || !qkv_bits_valid(norm_mse_bits) ||
        n_out > INT_MAX / out_mse_bits || n_norm > INT_MAX / norm_mse_bits) {
        return 0;
    }

    const int out_stride = (n_out * out_mse_bits + 7) / 8;
    const int norm_stride = (n_norm * norm_mse_bits + 7) / 8;
    const size_t qjl_out_stride = qkv_split_qjl_outlier_bytes(config);
    const size_t qjl_norm_stride = qkv_split_qjl_normal_bytes(config);
    const size_t qjl_stride = qkv_qjl_token_bytes(state);
    if (out_stride <= 0 || norm_stride <= 0 ||
        (use_qjl && (qjl_out_stride == 0 || qjl_norm_stride == 0 ||
                     qjl_stride < qjl_out_stride + qjl_norm_stride))) {
        return 0;
    }

    uint8_t* out_dst = split_outlier + (size_t)token_idx * (size_t)out_stride;
    uint8_t* norm_dst = split_normal + (size_t)token_idx * (size_t)norm_stride;
    uint8_t* qjl_token = use_qjl ? (qjl_base + (size_t)token_idx * qjl_stride) : nullptr;
    float* out_norms = qkv_split_norms_for_target(state, target, true);
    float* norm_norms = qkv_split_norms_for_target(state, target, false);
    float* out_rnorms = qkv_split_residual_norms_for_target(state, target, true);
    float* norm_rnorms = qkv_split_residual_norms_for_target(state, target, false);
    if (!out_norms || !norm_norms || (use_qjl && (!out_rnorms || !norm_rnorms))) return 0;

    float global_norm_sq = 0.0f;
    for (int i = 0; i < dim; ++i) {
        if (!std::isfinite(input[i])) return 0;
        global_norm_sq += input[i] * input[i];
    }
    *norm_out = sqrtf(std::max(0.0f, global_norm_sq));
    if (residual_norms) residual_norms[token_idx] = 0.0f;

    auto quant_group = [&](bool outlier_group) -> int {
        const int gd = outlier_group ? n_out : n_norm;
        const int bits_total = outlier_group ? outlier_bits : normal_bits;
        const int mse_bits = outlier_group ? out_mse_bits : norm_mse_bits;
        uint8_t* dst = outlier_group ? out_dst : norm_dst;
        float* group_norms = outlier_group ? out_norms : norm_norms;
        float* group_rnorms = outlier_group ? out_rnorms : norm_rnorms;
        const float* rot = qkv_split_rotation_matrix_for_group(state, outlier_group);
        const float* signs = qkv_split_rotation_signs_for_group(state, outlier_group);
        const float* qjl_mat = qkv_split_qjl_matrix_for_group(state, outlier_group);
        uint8_t* qjl_dst = nullptr;
        if (use_qjl) {
            qjl_dst = outlier_group ? qjl_token : (qjl_token + qjl_out_stride);
        }
        float* unit = state->scratch_residual;
        float* rotated = state->scratch_s_times_r;
        float* y_tilde = state->scratch_y_tilde;
        float* x_mse = state->scratch_x_tilde;
        int* codes = state->scratch_indices;
        float norm_sq = 0.0f;
        if (outlier_group) {
            for (int i = 0; i < gd; ++i) {
                const int ch = outlier_channels[i];
                if (ch < 0 || ch >= dim) return 0;
                unit[i] = input[ch];
                norm_sq += unit[i] * unit[i];
            }
        } else {
            int pos = 0;
            for (int ch = 0; ch < dim; ++ch) {
                if (is_outlier[ch]) continue;
                if (pos >= gd) return 0;
                unit[pos] = input[ch];
                norm_sq += unit[pos] * unit[pos];
                ++pos;
            }
            if (pos != gd) return 0;
        }
        const float group_norm = sqrtf(std::max(0.0f, norm_sq));
        group_norms[token_idx] = group_norm;
        if (group_rnorms) group_rnorms[token_idx] = 0.0f;
        if (group_norm < 1e-12f) {
            memset(dst, 0, (size_t)((gd * mse_bits + 7) / 8));
            if (qjl_dst) memset(qjl_dst, 0, outlier_group ? qjl_out_stride : qjl_norm_stride);
            return 1;
        }
        const float inv = 1.0f / group_norm;
        for (int i = 0; i < gd; ++i) unit[i] *= inv;

        if (!qkv_apply_split_rotation_forward(config, rot, signs, unit, rotated, gd)) return 0;
        const bool raw = qkv_bits_raw(mse_bits);
        const float* centroids = raw ? nullptr : qkv_codebook_for_bits_dim(mse_bits, gd, config->codebook_distribution);
        const float* thresholds = raw ? nullptr : qkv_thresholds_for_bits_dim(mse_bits, gd, config->codebook_distribution);
        if (!raw && (!centroids || !thresholds)) return 0;
        const int levels = raw ? 0 : (1 << mse_bits);
        for (int i = 0; i < gd; ++i) {
            if (raw) {
                qkv_store_raw_scalar(dst, i, rotated[i], mse_bits);
                y_tilde[i] = rotated[i];
            } else {
                const int code = qkv_find_nearest_centroid(rotated[i], centroids, thresholds, levels);
                if (code < 0 || code >= levels) return 0;
                codes[i] = code;
                y_tilde[i] = centroids[code];
            }
        }
        if (!raw) qkv_pack_indices(codes, dst, gd, mse_bits);

        if (use_qjl) {
            if (!qkv_apply_split_rotation_inverse(config, rot, signs, y_tilde, x_mse, gd)) return 0;
            float r_norm_sq = 0.0f;
            for (int i = 0; i < gd; ++i) {
                unit[i] = unit[i] - x_mse[i];
                r_norm_sq += unit[i] * unit[i];
            }
            group_rnorms[token_idx] = sqrtf(std::max(0.0f, r_norm_sq));
            if (!qkv_project_qjl(qjl_mat, unit, rotated, gd)) return 0;
            qkv_pack_signs(rotated, qjl_dst, gd);
        }
        return 1;
    };

    if (!quant_group(true)) return 0;
    if (!quant_group(false)) return 0;
    return 1;
}


static int qkv_store_qjl_residual_for_token(
    qkv_state_t* state,
    const qkv_config_t* config,
    const float* input,
    const uint8_t* mse_idx,
    float norm,
    uint8_t* qjl_base,
    float* residual_norms,
    int token_idx,
    int total_bits
) {
    if (!state || !config || !input || !mse_idx || !qjl_base || !residual_norms ||
        token_idx < 0 || token_idx >= state->n_tokens) {
        return 0;
    }
    const int dim = config->head_dim;
    const int mse_bits = total_bits - 1;
    if (dim <= 0 || dim > 16384 || !qkv_bits_codebook(mse_bits) ||
        !state->scratch_x_tilde || !state->scratch_indices ||
        !state->scratch_y_tilde || !state->scratch_residual ||
        !state->scratch_s_times_r || !state->qjl_matrix) {
        return 0;
    }
    int* indices = state->scratch_indices;
    float* y_tilde = state->scratch_y_tilde;
    float* x_mse = state->scratch_x_tilde;
    float* residual = state->scratch_residual;
    float* s_times_r = state->scratch_s_times_r;
    qkv_unpack_indices(mse_idx, indices, dim, mse_bits);
    const float* centroids = qkv_codebook_for_bits(state, mse_bits);
    if (!centroids) {
        return 0;
    }
    const int max_idx = 1 << mse_bits;
    for (int i = 0; i < dim; ++i) {
        if (indices[i] < 0 || indices[i] >= max_idx) {
            return 0;
        }
        y_tilde[i] = centroids[indices[i]];
    }
    if (config->enable_rotation && state->rotation_matrix) {
        if (state->rotation_signs &&
            qkv_apply_hadamard_rotation_inverse(y_tilde, state->rotation_signs, x_mse, dim)) {
            // Fast inverse rotation path for Hadamard-backed QKV states.
        } else {
        for (int i = 0; i < dim; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < dim; ++j) {
                sum += state->rotation_matrix[(size_t)j * (size_t)dim + (size_t)i] * y_tilde[j];
            }
            x_mse[i] = sum;
        }
        }
    } else {
        memcpy(x_mse, y_tilde, (size_t)dim * sizeof(float));
    }
    const float inv_norm = norm > 1e-12f ? 1.0f / norm : 0.0f;
    float r_norm_sq = 0.0f;
    for (int i = 0; i < dim; ++i) {
        residual[i] = (inv_norm > 0.0f ? input[i] * inv_norm : 0.0f) - x_mse[i];
        r_norm_sq += residual[i] * residual[i];
    }
    residual_norms[token_idx] = sqrtf(std::max(0.0f, r_norm_sq));
    for (int i = 0; i < dim; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < dim; ++j) {
            sum += state->qjl_matrix[(size_t)i * (size_t)dim + (size_t)j] * residual[j];
        }
        s_times_r[i] = sum;
    }
    const size_t qjl_stride = ((size_t)dim + 7u) / 8u;
    qkv_pack_signs(s_times_r, qjl_base + (size_t)token_idx * qjl_stride, dim);
    return 1;
}

int qkv_quantize_token(
    qkv_state_t* state,
    const qkv_config_t* config,
    const float* key,
    const float* value,
    int token_idx
) {
    if (!state || !config || !key || !value || token_idx < 0 ||
        token_idx >= state->n_tokens) {
        return 0;
    }
    const int dim = config->head_dim;
    if (dim <= 0 || dim > 16384) {
        return 0;
    }
    if ((uint32_t)token_idx < state->sink_tokens && state->k_sink && state->v_sink) {
        memcpy(state->k_sink + (size_t)token_idx * (size_t)dim, key, (size_t)dim * sizeof(float));
        memcpy(state->v_sink + (size_t)token_idx * (size_t)dim, value, (size_t)dim * sizeof(float));
    }
    const bool use_qjl = config->enable_qjl && state->k_qjl && state->v_qjl &&
        qkv_bits_codebook(state->k_bits) && qkv_bits_codebook(state->v_bits) &&
        state->k_bits > 1 && state->v_bits > 1;
    const int k_mse_bits = use_qjl ? state->k_bits - 1 : state->k_bits;
    const int v_mse_bits = use_qjl ? state->v_bits - 1 : state->v_bits;
    if (!qkv_bits_valid(k_mse_bits) || !qkv_bits_valid(v_mse_bits) ||
        !state->k_idx || !state->v_idx || !state->k_norms || !state->v_norms) {
        return 0;
    }

    const size_t k_stride = ((size_t)dim * (size_t)k_mse_bits + 7u) / 8u;
    uint8_t* k_out = state->k_idx + (size_t)token_idx * k_stride;
    const bool k_split = qkv_outlier_split_ready(state, config, QKV_TARGET_KEY);
    if (k_split) {
        if (!qkv_quantize_split_vector_with_state(
                state, config, QKV_TARGET_KEY, key, token_idx,
                &state->k_norms[token_idx], state->k_qjl, state->k_residual_norms)) {
            return 0;
        }
    } else if (!qkv_quantize_vector_with_state(
            state, config, key, k_out, &state->k_norms[token_idx], dim, k_mse_bits)) {
        return 0;
    } else if (use_qjl &&
               !qkv_store_qjl_residual_for_token(
                   state, config, key, k_out, state->k_norms[token_idx],
                   state->k_qjl, state->k_residual_norms, token_idx, state->k_bits)) {
        return 0;
    }

    const size_t v_stride = ((size_t)dim * (size_t)v_mse_bits + 7u) / 8u;
    uint8_t* v_out = state->v_idx + (size_t)token_idx * v_stride;
    const bool v_split = qkv_outlier_split_ready(state, config, QKV_TARGET_VALUE);
    if (v_split) {
        if (!qkv_quantize_split_vector_with_state(
                state, config, QKV_TARGET_VALUE, value, token_idx,
                &state->v_norms[token_idx], state->v_qjl, state->v_residual_norms)) {
            return 0;
        }
    } else if (!qkv_quantize_vector_with_state(
            state, config, value, v_out, &state->v_norms[token_idx], dim, v_mse_bits)) {
        return 0;
    } else if (use_qjl &&
               !qkv_store_qjl_residual_for_token(
                   state, config, value, v_out, state->v_norms[token_idx],
                   state->v_qjl, state->v_residual_norms, token_idx, state->v_bits)) {
        return 0;
    }
    return 1;
}

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

    // BUGFIX 341: head_dim 유효성 체크
    int dim = config->head_dim;
    if (dim <= 0 || dim > 16384) {
        return 0;
    }
    const bool use_qjl = config->enable_qjl && state->k_qjl && state->v_qjl &&
        qkv_bits_codebook(state->k_bits) && qkv_bits_codebook(state->v_bits) &&
        state->k_bits > 1 && state->v_bits > 1;
    const int k_mse_bits = use_qjl ? state->k_bits - 1 : state->k_bits;
    const int v_mse_bits = use_qjl ? state->v_bits - 1 : state->v_bits;
    if (!qkv_bits_valid(k_mse_bits) || !qkv_bits_valid(v_mse_bits)) {
        return 0;
    }

    for (int t = 0; t < n_tokens; t++) {
        if ((uint32_t)t < state->sink_tokens && state->k_sink && state->v_sink) {
            memcpy(state->k_sink + (size_t)t * (size_t)dim, key_data + (size_t)t * (size_t)dim,
                (size_t)dim * sizeof(float));
            memcpy(state->v_sink + (size_t)t * (size_t)dim, value_data + (size_t)t * (size_t)dim,
                (size_t)dim * sizeof(float));
        }
        // ===== Quantize K =====
        // Paper Algorithm 1: MSE-optimal quantization
        // BUGFIX 342: k_idx null 체크 및 overflow 방지
        if (!state->k_idx) return 0;
        size_t k_offset = (size_t)t * (size_t)((dim * k_mse_bits + 7) / 8);
        uint8_t* k_out = state->k_idx + k_offset;
        const bool k_split = qkv_outlier_split_ready(state, config, QKV_TARGET_KEY);
        if (k_split) {
            if (!qkv_quantize_split_vector_with_state(
                    state, config, QKV_TARGET_KEY,
                    key_data + t * dim, t, &state->k_norms[t],
                    state->k_qjl, state->k_residual_norms)) {
                return 0;
            }
        } else if (!qkv_quantize_vector_with_state(
                state, config,
                key_data + t * dim,
                k_out,
                &state->k_norms[t],
                dim,
                k_mse_bits)) {
            return 0;
        }

        // Paper Algorithm 2: QJL residual for unbiased inner product
        if (!k_split && use_qjl && state->k_qjl && state->qjl_matrix) {
            // BUGFIX 343: k_qjl null 체크 및 overflow 방지
            size_t qjl_offset = (size_t)t * (size_t)((dim + 7) / 8);
            // Step 1: Dequantize MSE reconstruction
            float* x_mse = state->scratch_x_tilde;
            if (!x_mse) return 0;

            // Unpack and lookup centroids
            int* indices = state->scratch_indices;
            float* y_tilde = state->scratch_y_tilde;
            if (!indices || !y_tilde) return 0;

            qkv_unpack_indices(k_out, indices, dim, k_mse_bits);
            const float* centroids = qkv_codebook_for_bits(state, k_mse_bits);
            // BUGFIX 467: centroids null 체크
            if (!centroids) return 0;
            const int max_idx = (1 << k_mse_bits);
            for (int i = 0; i < dim; i++) {
                // BUGFIX 468: indices 범위 체크
                if (indices[i] < 0 || indices[i] >= max_idx) return 0;
                y_tilde[i] = centroids[indices[i]];
            }

            // Apply inverse rotation in the normalized domain. Algorithm 2 is
            // defined for x in S^{d-1}; norms[t] is applied only at dequant.
            // BUGFIX Issue 9: Use Fast Hadamard Transform for inverse rotation
            // Inverse: H^T = sqrt(d) * W^T * D^T = sqrt(d) * W * D (both self-transpose)
            // So inverse order is: scale → FWHT → signs (reverse of forward)

            const bool is_power_of_2 = (dim & (dim - 1)) == 0;
            if (config->enable_rotation && state->rotation_matrix &&
                is_power_of_2 && state->rotation_signs) {
                // Fast path: Inverse Hadamard (self-inverse up to scale)
                const float scale = 1.0f / sqrtf((float)dim);
                for (int i = 0; i < dim; ++i) {
                    x_mse[i] = y_tilde[i] * scale;
                }
                fwht_inplace(x_mse, dim);
                // Apply inverse sign vector (sign is self-inverse)
                for (int i = 0; i < dim; ++i) {
                    x_mse[i] *= state->rotation_signs[i];
                }
            } else if (config->enable_rotation && state->rotation_matrix) {
                // BUGFIX 911: Optimize rotation matrix access pattern (location 2/3) ★★ PERFORMANCE
                // BUGFIX 929: Add SIMD optimization for non-power-of-2 dimensions ★★ PERFORMANCE
                // Problem: O(d²) scalar loop is slow for d=128, d=256 (common in KV compression)
                // Solution: Use AVX2 (8 floats) or NEON (4 floats) for inner loop vectorization
                // Impact: 3-4x speedup for rotation matrix multiply on non-power-of-2 dimensions
#if defined(__AVX2__)
                // AVX2 path: Process 8 floats at a time
                for (int i = 0; i < dim; i++) {
                    x_mse[i] = 0.0f;
                }
                for (int j = 0; j < dim; j++) {
                    const float y_val = y_tilde[j];
                    const __m256 y_vec = _mm256_set1_ps(y_val);
                    const float* row = &state->rotation_matrix[j * dim];
                    int i = 0;
                    for (; i + 7 < dim; i += 8) {
                        __m256 row_vec = _mm256_loadu_ps(&row[i]);
                        __m256 acc_vec = _mm256_loadu_ps(&x_mse[i]);
                        acc_vec = _mm256_add_ps(_mm256_mul_ps(row_vec, y_vec), acc_vec);
                        _mm256_storeu_ps(&x_mse[i], acc_vec);
                    }
                    // Scalar tail
                    for (; i < dim; i++) {
                        x_mse[i] += row[i] * y_val;
                    }
                }
#elif defined(__ARM_NEON)
                // NEON path: Process 4 floats at a time
                for (int i = 0; i < dim; i++) {
                    x_mse[i] = 0.0f;
                }
                for (int j = 0; j < dim; j++) {
                    const float y_val = y_tilde[j];
                    const float32x4_t y_vec = vdupq_n_f32(y_val);
                    const float* row = &state->rotation_matrix[j * dim];
                    int i = 0;
                    for (; i + 3 < dim; i += 4) {
                        float32x4_t row_vec = vld1q_f32(&row[i]);
                        float32x4_t acc_vec = vld1q_f32(&x_mse[i]);
                        acc_vec = vmlaq_f32(acc_vec, row_vec, y_vec);
                        vst1q_f32(&x_mse[i], acc_vec);
                    }
                    // Scalar tail
                    for (; i < dim; i++) {
                        x_mse[i] += row[i] * y_val;
                    }
                }
#else
                // Scalar fallback
                for (int i = 0; i < dim; i++) {
                    x_mse[i] = 0.0f;
                }
                for (int j = 0; j < dim; j++) {
                    const float y_val = y_tilde[j];
                    const float* row = &state->rotation_matrix[j * dim];
                    for (int i = 0; i < dim; i++) {
                        x_mse[i] += row[i] * y_val;
                    }
                }
#endif
            } else {
                for (int i = 0; i < dim; i++) {
                    x_mse[i] = y_tilde[i];
                }
            }

            // Step 2: Compute residual r = x_unit - x_mse
            float* residual = state->scratch_residual;
            if (!residual) return 0;

            const float k_norm = state->k_norms[t];
            const float k_inv_norm = (k_norm > 1e-12f) ? (1.0f / k_norm) : 0.0f;
            float r_norm_sq = 0.0f;
            for (int i = 0; i < dim; i++) {
                residual[i] = (k_inv_norm > 0.0f ? key_data[t * dim + i] * k_inv_norm : 0.0f) - x_mse[i];
                r_norm_sq += residual[i] * residual[i];
            }
            // BUGFIX 657: Prevent NaN from negative r_norm_sq due to floating point errors ★★
            // Problem: Floating point accumulation can produce small negative values (e.g., -1e-15)
            // Solution: Clamp to 0 before sqrt to prevent NaN propagation
            // Impact: Prevents NaN in QJL residual norm → stable attention computation
            state->k_residual_norms[t] = sqrtf(std::max(0.0f, r_norm_sq));

            // Step 3: QJL - sign(S * r)
            float* s_times_r = state->scratch_s_times_r;
            if (!s_times_r) return 0;

            // BUGFIX 942: Add SIMD optimization for QJL projection ★★ PERFORMANCE
            // Problem: O(d²) scalar loop for QJL S*r projection
            // Solution: AVX2/NEON vectorization matching rotation matrix pattern
#if defined(__AVX2__)
            for (int i = 0; i < dim; i++) {
                __m256 sum_vec = _mm256_setzero_ps();
                const float* qjl_row = &state->qjl_matrix[i * dim];
                int j = 0;
                for (; j + 7 < dim; j += 8) {
                    __m256 q_vec = _mm256_loadu_ps(&qjl_row[j]);
                    __m256 r_vec = _mm256_loadu_ps(&residual[j]);
                    sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(q_vec, r_vec));
                }
                // Horizontal sum
                __m128 hi = _mm256_extractf128_ps(sum_vec, 1);
                __m128 lo = _mm256_castps256_ps128(sum_vec);
                __m128 sum4 = _mm_add_ps(lo, hi);
                sum4 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
                sum4 = _mm_add_ss(sum4, _mm_shuffle_ps(sum4, sum4, 1));
                float sum = _mm_cvtss_f32(sum4);
                for (; j < dim; j++) {
                    sum += qjl_row[j] * residual[j];
                }
                s_times_r[i] = sum;
            }
#elif defined(__ARM_NEON)
            for (int i = 0; i < dim; i++) {
                float32x4_t sum_vec = vdupq_n_f32(0.0f);
                const float* qjl_row = &state->qjl_matrix[i * dim];
                int j = 0;
                for (; j + 3 < dim; j += 4) {
                    float32x4_t q_vec = vld1q_f32(&qjl_row[j]);
                    float32x4_t r_vec = vld1q_f32(&residual[j]);
                    sum_vec = vmlaq_f32(sum_vec, q_vec, r_vec);
                }
                float sum = vaddvq_f32(sum_vec);
                for (; j < dim; j++) {
                    sum += qjl_row[j] * residual[j];
                }
                s_times_r[i] = sum;
            }
#else
            for (int i = 0; i < dim; i++) {
                float sum = 0.0f;
                for (int j = 0; j < dim; j++) {
                    sum += state->qjl_matrix[i * dim + j] * residual[j];
                }
                s_times_r[i] = sum;
            }
#endif

            // Pack signs
            uint8_t* qjl_out = state->k_qjl + qjl_offset;
            qkv_pack_signs(s_times_r, qjl_out, dim);
        }

        // ===== Quantize V =====
        // BUGFIX 344: v_idx null 체크 및 overflow 방지
        if (!state->v_idx) return 0;
        size_t v_offset = (size_t)t * (size_t)((dim * v_mse_bits + 7) / 8);
        uint8_t* v_out = state->v_idx + v_offset;
        const bool v_split = qkv_outlier_split_ready(state, config, QKV_TARGET_VALUE);
        if (v_split) {
            if (!qkv_quantize_split_vector_with_state(
                    state, config, QKV_TARGET_VALUE,
                    value_data + t * dim, t, &state->v_norms[t],
                    state->v_qjl, state->v_residual_norms)) {
                return 0;
            }
        } else if (!qkv_quantize_vector_with_state(
                state, config,
                value_data + t * dim,
                v_out,
                &state->v_norms[t],
                dim,
                v_mse_bits)) {
            return 0;
        }

        // QJL residual for V
        if (!v_split && use_qjl && state->v_qjl && state->qjl_matrix) {
            float* x_mse = state->scratch_x_tilde;
            if (!x_mse) return 0;

            int* indices = state->scratch_indices;
            float* y_tilde = state->scratch_y_tilde;
            if (!indices || !y_tilde) return 0;

            qkv_unpack_indices(v_out, indices, dim, v_mse_bits);
            const float* centroids = qkv_codebook_for_bits(state, v_mse_bits);
            // BUGFIX 469: centroids null 체크
            if (!centroids) return 0;
            const int max_idx = (1 << v_mse_bits);
            for (int i = 0; i < dim; i++) {
                // BUGFIX 470: indices 범위 체크
                if (indices[i] < 0 || indices[i] >= max_idx) return 0;
                y_tilde[i] = centroids[indices[i]];
            }

            // BUGFIX Issue 9: Use Fast Hadamard Transform for inverse rotation (V path)

            const bool is_power_of_2_v = (dim & (dim - 1)) == 0;
            if (config->enable_rotation && state->rotation_matrix &&
                is_power_of_2_v && state->rotation_signs) {
                // Fast path: Inverse Hadamard
                const float scale = 1.0f / sqrtf((float)dim);
                for (int i = 0; i < dim; ++i) {
                    x_mse[i] = y_tilde[i] * scale;
                }
                fwht_inplace(x_mse, dim);
                for (int i = 0; i < dim; ++i) {
                    x_mse[i] *= state->rotation_signs[i];
                }
            } else if (config->enable_rotation && state->rotation_matrix) {
                // BUGFIX 911: Optimize rotation matrix access pattern (location 3/3) ★★ PERFORMANCE
                // BUGFIX 929: Add SIMD optimization for non-power-of-2 dimensions ★★ PERFORMANCE
#if defined(__AVX2__)
                // AVX2 path: Process 8 floats at a time
                for (int i = 0; i < dim; i++) {
                    x_mse[i] = 0.0f;
                }
                for (int j = 0; j < dim; j++) {
                    const float y_val = y_tilde[j];
                    const __m256 y_vec = _mm256_set1_ps(y_val);
                    const float* row = &state->rotation_matrix[j * dim];
                    int i = 0;
                    for (; i + 7 < dim; i += 8) {
                        __m256 row_vec = _mm256_loadu_ps(&row[i]);
                        __m256 acc_vec = _mm256_loadu_ps(&x_mse[i]);
                        acc_vec = _mm256_add_ps(_mm256_mul_ps(row_vec, y_vec), acc_vec);
                        _mm256_storeu_ps(&x_mse[i], acc_vec);
                    }
                    // Scalar tail
                    for (; i < dim; i++) {
                        x_mse[i] += row[i] * y_val;
                    }
                }
#elif defined(__ARM_NEON)
                // NEON path: Process 4 floats at a time
                for (int i = 0; i < dim; i++) {
                    x_mse[i] = 0.0f;
                }
                for (int j = 0; j < dim; j++) {
                    const float y_val = y_tilde[j];
                    const float32x4_t y_vec = vdupq_n_f32(y_val);
                    const float* row = &state->rotation_matrix[j * dim];
                    int i = 0;
                    for (; i + 3 < dim; i += 4) {
                        float32x4_t row_vec = vld1q_f32(&row[i]);
                        float32x4_t acc_vec = vld1q_f32(&x_mse[i]);
                        acc_vec = vmlaq_f32(acc_vec, row_vec, y_vec);
                        vst1q_f32(&x_mse[i], acc_vec);
                    }
                    // Scalar tail
                    for (; i < dim; i++) {
                        x_mse[i] += row[i] * y_val;
                    }
                }
#else
                // Scalar fallback
                for (int i = 0; i < dim; i++) {
                    x_mse[i] = 0.0f;
                }
                for (int j = 0; j < dim; j++) {
                    const float y_val = y_tilde[j];
                    const float* row = &state->rotation_matrix[j * dim];
                    for (int i = 0; i < dim; i++) {
                        x_mse[i] += row[i] * y_val;
                    }
                }
#endif
            } else {
                for (int i = 0; i < dim; i++) {
                    x_mse[i] = y_tilde[i];
                }
            }

            float* residual = state->scratch_residual;
            if (!residual) return 0;

            const float v_norm = state->v_norms[t];
            const float v_inv_norm = (v_norm > 1e-12f) ? (1.0f / v_norm) : 0.0f;
            float r_norm_sq = 0.0f;
            for (int i = 0; i < dim; i++) {
                residual[i] = (v_inv_norm > 0.0f ? value_data[t * dim + i] * v_inv_norm : 0.0f) - x_mse[i];
                r_norm_sq += residual[i] * residual[i];
            }
            // BUGFIX 657: Prevent NaN from negative r_norm_sq (V path) ★★
            state->v_residual_norms[t] = sqrtf(std::max(0.0f, r_norm_sq));

            float* s_times_r = state->scratch_s_times_r;
            if (!s_times_r) return 0;

            // BUGFIX 942: Add SIMD optimization for QJL projection (V path) ★★ PERFORMANCE
#if defined(__AVX2__)
            for (int i = 0; i < dim; i++) {
                __m256 sum_vec = _mm256_setzero_ps();
                const float* qjl_row = &state->qjl_matrix[i * dim];
                int j = 0;
                for (; j + 7 < dim; j += 8) {
                    __m256 q_vec = _mm256_loadu_ps(&qjl_row[j]);
                    __m256 r_vec = _mm256_loadu_ps(&residual[j]);
                    sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(q_vec, r_vec));
                }
                __m128 hi = _mm256_extractf128_ps(sum_vec, 1);
                __m128 lo = _mm256_castps256_ps128(sum_vec);
                __m128 sum4 = _mm_add_ps(lo, hi);
                sum4 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
                sum4 = _mm_add_ss(sum4, _mm_shuffle_ps(sum4, sum4, 1));
                float sum = _mm_cvtss_f32(sum4);
                for (; j < dim; j++) {
                    sum += qjl_row[j] * residual[j];
                }
                s_times_r[i] = sum;
            }
#elif defined(__ARM_NEON)
            for (int i = 0; i < dim; i++) {
                float32x4_t sum_vec = vdupq_n_f32(0.0f);
                const float* qjl_row = &state->qjl_matrix[i * dim];
                int j = 0;
                for (; j + 3 < dim; j += 4) {
                    float32x4_t q_vec = vld1q_f32(&qjl_row[j]);
                    float32x4_t r_vec = vld1q_f32(&residual[j]);
                    sum_vec = vmlaq_f32(sum_vec, q_vec, r_vec);
                }
                float sum = vaddvq_f32(sum_vec);
                for (; j < dim; j++) {
                    sum += qjl_row[j] * residual[j];
                }
                s_times_r[i] = sum;
            }
#else
            for (int i = 0; i < dim; i++) {
                float sum = 0.0f;
                for (int j = 0; j < dim; j++) {
                    sum += state->qjl_matrix[i * dim + j] * residual[j];
                }
                s_times_r[i] = sum;
            }
#endif

            // BUGFIX 345: v_qjl overflow 방지
            size_t v_qjl_offset = (size_t)t * (size_t)((dim + 7) / 8);
            uint8_t* qjl_out = state->v_qjl + v_qjl_offset;
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

    // BUGFIX 346: head_dim 유효성 체크
    int dim = config->head_dim;
    if (dim <= 0 || dim > 16384) {
        return 0;
    }
    bool use_qjl = config->enable_qjl && state->k_qjl && state->v_qjl &&
        qkv_bits_codebook(state->k_bits) && qkv_bits_codebook(state->v_bits);
    if (use_qjl && (state->k_bits <= 1 || state->v_bits <= 1)) {
        use_qjl = false;
    }

    for (int t = 0; t < n_tokens; t++) {
        // BUGFIX 347: key_output overflow 방지
        size_t key_offset = (size_t)t * (size_t)dim;
        // Dequantize K
        if (!qkv_dequant_one(
                state, config,
                state->k_idx, state->k_qjl,
                state->k_residual_norms, state->k_norms,
                t, state->k_bits, use_qjl,
                key_output + key_offset)) {
            return 0;
        }

        // BUGFIX 348: value_output overflow 방지
        size_t value_offset = (size_t)t * (size_t)dim;
        // Dequantize V
        if (!qkv_dequant_one(
                state, config,
                state->v_idx, state->v_qjl,
                state->v_residual_norms, state->v_norms,
                t, state->v_bits, use_qjl,
                value_output + value_offset)) {
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
