#include "qkv_state.h"
#include "qkv_thread_pool.h"
#include "qkv_codebook.h"
#include "qkv_matrix.h"
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <thread>

int qkv_state_init(
    qkv_state_t* state,
    const qkv_config_t* config,
    int n_tokens
) {
    if (!state || !config || n_tokens <= 0) return 0;

    memset(state, 0, sizeof(*state));

    int dim = config->head_dim;
    int k_bits = config->k_bits;
    int v_bits = config->v_bits;
    if (dim <= 0 || k_bits < 1 || k_bits > 4 || v_bits < 1 || v_bits > 4) {
        return 0;
    }
    if (config->enable_qjl && (k_bits <= 1 || v_bits <= 1)) {
        return 0;
    }

    // Allocate main KV storage
    size_t k_packed_size = ((size_t)n_tokens * (size_t)dim * (size_t)k_bits + 7) / 8;
    size_t v_packed_size = ((size_t)n_tokens * (size_t)dim * (size_t)v_bits + 7) / 8;

    state->k_idx = (uint8_t*)calloc(k_packed_size, 1);
    state->v_idx = (uint8_t*)calloc(v_packed_size, 1);
    state->k_norms = (float*)malloc((size_t)n_tokens * sizeof(float));
    state->v_norms = (float*)malloc((size_t)n_tokens * sizeof(float));

    if (!state->k_idx || !state->v_idx || !state->k_norms || !state->v_norms) {
        qkv_state_free(state);
        return 0;
    }

    // Allocate QJL residual storage
    if (config->enable_qjl) {
        state->k_qjl = (uint8_t*)calloc(((size_t)n_tokens * (size_t)dim + 7) / 8, 1);
        state->k_residual_norms = (float*)malloc((size_t)n_tokens * sizeof(float));
        state->v_qjl = (uint8_t*)calloc(((size_t)n_tokens * (size_t)dim + 7) / 8, 1);
        state->v_residual_norms = (float*)malloc((size_t)n_tokens * sizeof(float));

        if (!state->k_qjl || !state->k_residual_norms ||
            !state->v_qjl || !state->v_residual_norms) {
            qkv_state_free(state);
            return 0;
        }
    }

    // Generate rotation matrix
    if (config->enable_rotation) {
        state->rotation_matrix = (float*)malloc((size_t)dim * (size_t)dim * sizeof(float));
        if (!state->rotation_matrix) {
            qkv_state_free(state);
            return 0;
        }
        qkv_generate_rotation_matrix(state->rotation_matrix, dim, config->rotation_seed);
    }

    // Generate QJL matrix
    if (config->enable_qjl) {
        state->qjl_signs_matrix = NULL;
        state->qjl_matrix = (float*)malloc((size_t)dim * (size_t)dim * sizeof(float));
        if (!state->qjl_matrix) {
            qkv_state_free(state);
            return 0;
        }
        qkv_generate_qjl_matrix(state->qjl_matrix, dim, config->qjl_seed);
    }

    // Allocate codebooks
    state->codebook_1bit = (float*)malloc(2 * sizeof(float));
    state->thresholds_1bit = (float*)malloc(3 * sizeof(float));
    state->codebook_2bit = (float*)malloc(4 * sizeof(float));
    state->thresholds_2bit = (float*)malloc(5 * sizeof(float));
    state->codebook_3bit = (float*)malloc(8 * sizeof(float));
    state->thresholds_3bit = (float*)malloc(9 * sizeof(float));
    state->codebook_4bit = (float*)malloc(16 * sizeof(float));
    state->thresholds_4bit = (float*)malloc(17 * sizeof(float));

    if (!state->codebook_1bit || !state->thresholds_1bit ||
        !state->codebook_2bit || !state->thresholds_2bit ||
        !state->codebook_3bit || !state->thresholds_3bit ||
        !state->codebook_4bit || !state->thresholds_4bit) {
        qkv_state_free(state);
        return 0;
    }

    // Compute codebooks
    qkv_compute_lloyd_max_codebook(state->codebook_1bit, state->thresholds_1bit, 1, dim);
    qkv_compute_lloyd_max_codebook(state->codebook_2bit, state->thresholds_2bit, 2, dim);
    qkv_compute_lloyd_max_codebook(state->codebook_3bit, state->thresholds_3bit, 3, dim);
    qkv_compute_lloyd_max_codebook(state->codebook_4bit, state->thresholds_4bit, 4, dim);

    // Allocate scratch buffers
    state->scratch_qjl_signs = (float*)malloc((size_t)dim * sizeof(float));
    state->scratch_s_t_qjl = (float*)malloc((size_t)dim * sizeof(float));
    state->scratch_residual = (float*)malloc((size_t)dim * sizeof(float));
    state->scratch_s_times_r = (float*)malloc((size_t)dim * sizeof(float));
    state->scratch_indices = (int*)malloc((size_t)dim * sizeof(int));
    state->scratch_y_tilde = (float*)malloc((size_t)dim * sizeof(float));
    // Bug 4 Fix: Only allocate [head_dim], not [n_tokens * head_dim]
    // This buffer is only used for single-token dequant/quantize operations.
    // Parallel V accumulation uses per-worker work_x buffers instead.
    state->scratch_x_tilde = (float*)malloc((size_t)dim * sizeof(float));
    state->scratch_attention = (float*)malloc((size_t)n_tokens * sizeof(float));
    state->scratch_rotated_q = (float*)malloc((size_t)dim * sizeof(float));

    if (!state->scratch_qjl_signs || !state->scratch_s_t_qjl ||
        !state->scratch_residual || !state->scratch_s_times_r ||
        !state->scratch_indices || !state->scratch_y_tilde ||
        !state->scratch_x_tilde || !state->scratch_attention ||
        !state->scratch_rotated_q) {
        qkv_state_free(state);
        return 0;
    }

    // Allocate outlier channel storage
    if (config->outlier_channels > 0 && config->outlier_channels < dim) {
        int n_out = config->outlier_channels;
        int n_norm = dim - n_out;
        int out_bits = config->outlier_bits;
        int norm_bits = config->normal_bits;

        if (out_bits < 1 || out_bits > 4 || norm_bits < 1 || norm_bits > 4) {
            qkv_state_free(state);
            return 0;
        }

        state->outlier_indices = (int*)malloc((size_t)n_out * sizeof(int));
        state->k_outlier_indices = (int*)malloc((size_t)n_out * sizeof(int));
        state->v_outlier_indices = (int*)malloc((size_t)n_out * sizeof(int));
        state->k_is_outlier = (uint8_t*)calloc((size_t)dim, 1);
        state->v_is_outlier = (uint8_t*)calloc((size_t)dim, 1);

        size_t k_out_size = ((size_t)n_tokens * (size_t)n_out * (size_t)out_bits + 7) / 8;
        size_t k_norm_size = ((size_t)n_tokens * (size_t)n_norm * (size_t)norm_bits + 7) / 8;
        size_t v_out_size = ((size_t)n_tokens * (size_t)n_out * (size_t)out_bits + 7) / 8;
        size_t v_norm_size = ((size_t)n_tokens * (size_t)n_norm * (size_t)norm_bits + 7) / 8;

        state->k_idx_outlier = (uint8_t*)calloc(k_out_size, 1);
        state->k_idx_normal = (uint8_t*)calloc(k_norm_size, 1);
        state->v_idx_outlier = (uint8_t*)calloc(v_out_size, 1);
        state->v_idx_normal = (uint8_t*)calloc(v_norm_size, 1);
        state->k_norms_outlier = (float*)malloc((size_t)n_tokens * sizeof(float));
        state->k_norms_normal = (float*)malloc((size_t)n_tokens * sizeof(float));
        state->v_norms_outlier = (float*)malloc((size_t)n_tokens * sizeof(float));
        state->v_norms_normal = (float*)malloc((size_t)n_tokens * sizeof(float));

        if (!state->outlier_indices || !state->k_outlier_indices || !state->v_outlier_indices ||
            !state->k_is_outlier || !state->v_is_outlier ||
            !state->k_idx_outlier || !state->k_idx_normal ||
            !state->v_idx_outlier || !state->v_idx_normal ||
            !state->k_norms_outlier || !state->k_norms_normal ||
            !state->v_norms_outlier || !state->v_norms_normal) {
            qkv_state_free(state);
            return 0;
        }

        const int* src = config->outlier_channel_indices;
        for (int i = 0; i < n_out; i++) {
            int ch = src ? src[i] : i;
            state->outlier_indices[i] = ch;
            state->k_outlier_indices[i] = ch;
            state->v_outlier_indices[i] = ch;
            state->k_is_outlier[ch] = 1;
            state->v_is_outlier[ch] = 1;
        }
    }

    const unsigned hw_raw = std::thread::hardware_concurrency();
    const int io_threads = config->engine_io_thread_count > 0
        ? (int)config->engine_io_thread_count : 0;
    const int available = std::max(1, (int)(hw_raw ? hw_raw : 4) - io_threads - 1);
    const int workers = std::min(available, 16);
    state->thread_pool = new QkvThreadPool(workers);
    state->computed_workers = workers;

    state->n_tokens = n_tokens;
    state->head_dim = dim;
    state->k_bits = k_bits;
    state->v_bits = v_bits;

    return 1;
}

void qkv_state_free(qkv_state_t* state) {
    if (!state) return;

    free(state->k_idx);
    free(state->v_idx);
    free(state->k_norms);
    free(state->v_norms);
    free(state->k_qjl);
    free(state->k_residual_norms);
    free(state->v_qjl);
    free(state->v_residual_norms);
    free(state->rotation_matrix);
    free(state->qjl_matrix);
    free(state->qjl_signs_matrix);
    free(state->codebook_1bit);
    free(state->thresholds_1bit);
    free(state->codebook_2bit);
    free(state->thresholds_2bit);
    free(state->codebook_3bit);
    free(state->thresholds_3bit);
    free(state->codebook_4bit);
    free(state->thresholds_4bit);
    free(state->scratch_qjl_signs);
    free(state->scratch_s_t_qjl);
    free(state->scratch_residual);
    free(state->scratch_s_times_r);
    free(state->scratch_indices);
    free(state->scratch_y_tilde);
    free(state->scratch_x_tilde);
    free(state->scratch_attention);
    free(state->scratch_rotated_q);
    free(state->outlier_indices);
    free(state->k_outlier_indices);
    free(state->v_outlier_indices);
    free(state->k_is_outlier);
    free(state->v_is_outlier);
    free(state->k_idx_outlier);
    free(state->k_idx_normal);
    free(state->v_idx_outlier);
    free(state->v_idx_normal);
    free(state->k_norms_outlier);
    free(state->k_norms_normal);
    free(state->v_norms_outlier);
    free(state->v_norms_normal);

    if (state->thread_pool) {
        delete static_cast<QkvThreadPool*>(state->thread_pool);
        state->thread_pool = nullptr;
    }

    memset(state, 0, sizeof(*state));
}
