#include "qkv_attention.h"
#include "qkv_helpers.h"
#include "qkv_dequantize.h"
#include "qkv_codebook.h"
#include "qkv_packing.h"
#include "qkv_thread_pool.h"
#include <math.h>
#include <string.h>
#include <vector>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Paper: Attention decode directly on quantized KV cache
// Avoids full dequantize → attention → discard cycle
int qkv_attention_decode_impl(
    const float* query,
    const qkv_state_t* s,
    const qkv_config_t* cfg,
    uint32_t ctx,
    uint32_t hdim,
    float* output
) {
    if (!query || !s || !cfg || !output || !ctx || !hdim) return 0;
    const int d = (int)hdim, n = (int)ctx;
    const bool qjl = cfg->enable_qjl && s->k_bits > 1 && s->v_bits > 1 && s->k_qjl && s->v_qjl;

    // Use scratch buffers
    float* row = s->scratch_residual;
    if (!row || s->head_dim != d) return 0;

    if (n > s->n_tokens) return 0;
    float* att = s->scratch_attention;
    if (!att) return 0;

    // Paper optimization: Pre-rotate query ONCE with Pi (O(d²) × 1)
    // Then dequant uses codebook-only path (skip inverse rotation) — O(d) × N
    // Exploits: <q, Pi^T * y_hat> = <Pi * q, y_hat>
    const float* q_eff = query;
    if (cfg->enable_rotation && s->rotation_matrix) {
        float* rq = s->scratch_rotated_q;
        if (!rq) return 0;
        for (int i = 0; i < d; i++) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += s->rotation_matrix[i * d + j] * query[j];
            }
            rq[i] = sum;
        }
        q_eff = rq;
    }

    const float sc = 1.0f / sqrtf((float)d);

    // Bug 2 Fix: Use pre-computed worker limit instead of OS call
    const int max_workers = s->computed_workers > 0 ? s->computed_workers : 1;
    const int workers = std::max(1, std::min<int>(max_workers, n / 512));

    // Phase 1: K scores — dequant in rotated domain (no inverse rotation needed)
    // Paper Algorithm 2: For unbiased inner product, include QJL residual
    const int k_mse_bits = qjl ? s->k_bits - 1 : s->k_bits;
    if (!qkv_bits_valid(k_mse_bits)) return 0;
    const float* k_centroids = qkv_codebook_for_bits(s, k_mse_bits);
    const int k_stride = (d * k_mse_bits + 7) / 8;
    const int k_qstride = (d + 7) / 8;
    // Bug 1 Fix: Correct QJL scale factor from paper Algorithm 2: sqrt(π/(2d))
    const float qjl_scale = sqrtf((float)M_PI / (2.0f * (float)d));
    const bool k_split = qkv_outlier_split_ready(s, cfg, QKV_TARGET_KEY);
    const bool use_qjl_key_residual = qjl && s->k_qjl && s->qjl_matrix;

    float* s_q_precomputed = s->scratch_s_times_r;
    float* qjl_z = s->scratch_qjl_signs;

    // Precompute S * q_eff for QJL residual
    if (use_qjl_key_residual) {
        if (!s_q_precomputed || !qjl_z) return 0;
        for (int i = 0; i < d; i++) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += s->qjl_matrix[i * d + j] * q_eff[j];
            }
            s_q_precomputed[i] = sum;
        }
    }

    // Parallel K scoring for large context (n >= 1024)
    if (n >= 1024) {
        const int out_bits = cfg->outlier_bits;
        const int norm_bits = cfg->normal_bits;
        const int n_outliers = cfg->outlier_channels;
        const int n_normal = d - n_outliers;
        const int out_packed_size = k_split ? (n_outliers * out_bits + 7) / 8 : 0;
        const int norm_packed_size = k_split ? (n_normal * norm_bits + 7) / 8 : 0;
        const int* outlier_channels = k_split ? qkv_outlier_indices_for_target_const(s, QKV_TARGET_KEY) : nullptr;
        const uint8_t* split_outlier = k_split ? qkv_idx_outlier_for_target_const(s, QKV_TARGET_KEY) : nullptr;
        const uint8_t* split_normal = k_split ? qkv_idx_normal_for_target_const(s, QKV_TARGET_KEY) : nullptr;
        const float* out_centroids = k_split ? qkv_codebook_for_bits(s, out_bits) : nullptr;
        const float* norm_centroids = k_split ? qkv_codebook_for_bits(s, norm_bits) : nullptr;
        const uint8_t* is_outlier = k_split ? qkv_is_outlier_for_target_const(s, QKV_TARGET_KEY) : nullptr;

        const int local_stride = std::max(d, std::max(n_outliers, n_normal));
        if (!s->work_codes_buf || !s->work_qjl_buf ||
            s->work_buf_workers < workers || s->work_buf_stride < local_stride) {
            return 0;
        }

        auto score_range = [&](int begin, int end, int w, int* ok_flag) {
            int* local_codes = s->work_codes_buf + (size_t)w * (size_t)s->work_buf_stride;
            float* local_qjl = s->work_qjl_buf + (size_t)w * (size_t)d;

            for (int t = begin; t < end && *ok_flag; t++) {
                float norm_k = s->k_norms[t];
                float dot = 0.0f;

                if (k_split) {
                    if (!outlier_channels || !split_outlier || !split_normal || !out_centroids || !norm_centroids) {
                        *ok_flag = 0;
                        return;
                    }
                    qkv_unpack_indices(split_outlier + (size_t)t * (size_t)out_packed_size,
                        local_codes, n_outliers, out_bits);
                    for (int i = 0; i < n_outliers; i++) {
                        const int channel = outlier_channels[i];
                        if (channel < 0 || channel >= d) {
                            *ok_flag = 0;
                            return;
                        }
                        dot += q_eff[channel] * out_centroids[local_codes[i]];
                    }
                    qkv_unpack_indices(split_normal + (size_t)t * (size_t)norm_packed_size,
                        local_codes, n_normal, norm_bits);
                    int normal_pos = 0;
                    for (int i = 0; i < d; i++) {
                        if (is_outlier && is_outlier[i]) continue;
                        dot += q_eff[i] * norm_centroids[local_codes[normal_pos++]];
                    }
                } else if (k_mse_bits == 2) {
                    const uint8_t* tidx = s->k_idx + t * k_stride;
                    for (int i = 0; i < d; i++) {
                        int byte_pos = (i * 2) / 8;
                        int bit_pos = (i * 2) % 8;
                        int code = (tidx[byte_pos] >> bit_pos) & 0x3;
                        dot += q_eff[i] * k_centroids[code];
                    }
                } else {
                    const uint8_t* tidx = s->k_idx + t * k_stride;
                    qkv_unpack_indices(tidx, local_codes, d, k_mse_bits);
                    for (int i = 0; i < d; i++) {
                        dot += q_eff[i] * k_centroids[local_codes[i]];
                    }
                }

                // Paper Algorithm 2: Add QJL residual for unbiased inner product
                if (use_qjl_key_residual) {
                    const uint8_t* tqjl = s->k_qjl + t * k_qstride;
                    float r_norm = s->k_residual_norms[t];
                    if (r_norm > 1e-10f) {
                        qkv_unpack_signs(tqjl, local_qjl, d);
                        for (int i = 0; i < d; i++) {
                            dot += qjl_scale * r_norm * s_q_precomputed[i] * local_qjl[i];
                        }
                    }
                }
                att[t] = dot * norm_k * sc;
            }
        };

        std::vector<int> ok_flags((size_t)workers, 1);
        auto k_task = [&](int w) {
            const int begin = (int)(((int64_t)n * w) / workers);
            const int end = (int)(((int64_t)n * (w + 1)) / workers);
            score_range(begin, end, w, &ok_flags[(size_t)w]);
        };

        if (s->thread_pool) {
            static_cast<QkvThreadPool*>(s->thread_pool)->run(workers, k_task);
        } else {
            for (int w = 0; w < workers; ++w) k_task(w);
        }

        for (int v : ok_flags) {
            if (!v) return 0;
        }
    } else {
        // Serial K scoring for small context (n < 1024)
        for (int t = 0; t < n; t++) {
            float norm_k = s->k_norms[t];
            float dot = 0.0f;

            if (k_split) {
                if (!qkv_dot_mse_split_rotated_token(s, cfg, QKV_TARGET_KEY, t, q_eff, &dot)) return 0;
            } else if (k_mse_bits == 2) {
                const uint8_t* tidx = s->k_idx + t * k_stride;
                for (int i = 0; i < d; i++) {
                    int byte_pos = (i * 2) / 8;
                    int bit_pos = (i * 2) % 8;
                    int code = (tidx[byte_pos] >> bit_pos) & 0x3;
                    dot += q_eff[i] * k_centroids[code];
                }
            } else {
                const uint8_t* tidx = s->k_idx + t * k_stride;
                int* indices = s->scratch_indices;
                if (!indices) return 0;
                qkv_unpack_indices(tidx, indices, d, k_mse_bits);
                for (int i = 0; i < d; i++) {
                    dot += q_eff[i] * k_centroids[indices[i]];
                }
            }

            // Add QJL residual
            if (use_qjl_key_residual) {
                const uint8_t* tqjl = s->k_qjl + t * k_qstride;
                float r_norm = s->k_residual_norms[t];
                if (r_norm > 1e-10f) {
                    qkv_unpack_signs(tqjl, qjl_z, d);
                    for (int i = 0; i < d; i++) {
                        dot += qjl_scale * r_norm * s_q_precomputed[i] * qjl_z[i];
                    }
                }
            }
            att[t] = dot * norm_k * sc;
        }
    }

    // Softmax
    float mx = att[0];
    for (int t = 1; t < n; t++) if (att[t] > mx) mx = att[t];
    float se = 0;
    for (int t = 0; t < n; t++) { att[t] = expf(att[t] - mx); se += att[t]; }
    if (se > 0) {
        float iv = 1.0f / se;
        for (int t = 0; t < n; t++) att[t] *= iv;
    }

    // Phase 2: V weighted sum — need full dequant (with inverse rotation)
    if (n >= 1024) {
        std::vector<float> partial((size_t)workers * (size_t)d, 0.0f);
        std::vector<int> ok_flags((size_t)workers, 1);

        std::vector<float> work_row((size_t)workers * (size_t)d);
        std::vector<float> work_residual((size_t)workers * (size_t)d);
        std::vector<float> work_s_times_r((size_t)workers * (size_t)d);
        std::vector<float> work_qjl((size_t)workers * (size_t)d);
        std::vector<float> work_s_t_qjl((size_t)workers * (size_t)d);
        std::vector<float> work_y((size_t)workers * (size_t)d);
        std::vector<float> work_x((size_t)workers * (size_t)d);
        std::vector<int> work_indices((size_t)workers * (size_t)d);

        auto v_task = [&](int w) {
            const int begin = (int)(((int64_t)n * w) / workers);
            const int end = (int)(((int64_t)n * (w + 1)) / workers);
            qkv_state_t local = *s;
            local.scratch_residual = work_residual.data() + w * d;
            local.scratch_s_times_r = work_s_times_r.data() + w * d;
            local.scratch_qjl_signs = work_qjl.data() + w * d;
            local.scratch_s_t_qjl = work_s_t_qjl.data() + w * d;
            local.scratch_y_tilde = work_y.data() + w * d;
            local.scratch_x_tilde = work_x.data() + w * d;
            local.scratch_indices = work_indices.data() + w * d;
            float* dst = partial.data() + (size_t)w * (size_t)d;
            float* local_row = work_row.data() + (size_t)w * (size_t)d;

            for (int t = begin; t < end; ++t) {
                if (!qkv_dequant_one(&local, cfg, local.v_idx, local.v_qjl,
                        local.v_residual_norms, local.v_norms, t, local.v_bits, qjl, local_row)) {
                    ok_flags[(size_t)w] = 0;
                    return;
                }
                const float weight = att[t];
                for (int i = 0; i < d; ++i) {
                    dst[i] += weight * local_row[i];
                }
            }
        };

        if (s->thread_pool) {
            static_cast<QkvThreadPool*>(s->thread_pool)->run(workers, v_task);
        } else {
            for (int w = 0; w < workers; ++w) v_task(w);
        }

        memset(output, 0, d * sizeof(float));
        for (int w = 0; w < workers; ++w) {
            if (!ok_flags[(size_t)w]) return 0;
            const float* src = partial.data() + (size_t)w * (size_t)d;
            for (int i = 0; i < d; ++i) {
                output[i] += src[i];
            }
        }
        return 1;
    }

    // Serial V accumulation for small context
    memset(output, 0, d * sizeof(float));
    for (int t = 0; t < n; t++) {
        if (!qkv_dequant_one(s, cfg, s->v_idx, s->v_qjl,
                s->v_residual_norms, s->v_norms, t, s->v_bits, qjl, row))
            return 0;
        float w = att[t];
        for (int i = 0; i < d; i++) output[i] += w * row[i];
    }
    return 1;
}
