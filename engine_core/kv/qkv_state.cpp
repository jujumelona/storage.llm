#include "qkv_state.h"
#include "qkv_thread_pool.h"
#include "qkv_codebook.h"
#include "qkv_helpers.h"
#include "qkv_matrix.h"
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <thread>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <climits>

// BUGFIX Problem 2: Windows QKV working set preparation ★★ IMPORTANT
// Forward declare prepare_windows_staging_working_set from staging_alloc.cpp.inc
#ifdef _WIN32
extern "C" void prepare_windows_staging_working_set(uint64_t bytes);
#else
static inline void prepare_windows_staging_working_set(uint64_t) {}
#endif

static uint64_t qkv_cache_key(int dim, uint64_t tag) {
    uint64_t x = tag ^ ((uint64_t)(uint32_t)dim * 0x9e3779b97f4a7c15ull);
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    return x;
}

static std::shared_ptr<std::vector<float>> qkv_cached_codebook(int dim, int bits, bool thresholds) {
    static std::mutex mutex;
    static std::unordered_map<uint64_t, std::shared_ptr<std::vector<float>>> cache;
    const uint64_t key = qkv_cache_key(dim, (uint64_t)bits | (thresholds ? 0x10000ull : 0ull));
    std::lock_guard<std::mutex> lock(mutex);
    auto found = cache.find(key);
    if (found != cache.end()) return found->second;

    const int levels = 1 << bits;
    auto codebook = std::make_shared<std::vector<float>>((size_t)levels);
    auto thresh = std::make_shared<std::vector<float>>((size_t)levels + 1u);
    qkv_compute_lloyd_max_codebook(codebook->data(), thresh->data(), bits, dim);
    cache[qkv_cache_key(dim, (uint64_t)bits)] = codebook;
    cache[qkv_cache_key(dim, (uint64_t)bits | 0x10000ull)] = thresh;
    return thresholds ? thresh : codebook;
}

static std::shared_ptr<std::vector<float>> qkv_cached_rotation(int dim, uint64_t seed) {
    static std::mutex mutex;
    static std::unordered_map<uint64_t, std::shared_ptr<std::vector<float>>> cache;
    const uint64_t key = qkv_cache_key(dim, seed);
    std::lock_guard<std::mutex> lock(mutex);
    auto found = cache.find(key);
    if (found != cache.end()) return found->second;
    auto matrix = std::make_shared<std::vector<float>>((size_t)dim * (size_t)dim);
    qkv_generate_rotation_matrix(matrix->data(), dim, seed);
    cache[key] = matrix;
    return matrix;
}

static std::shared_ptr<std::vector<float>> qkv_cached_qjl(int dim, uint64_t seed) {
    static std::mutex mutex;
    static std::unordered_map<uint64_t, std::shared_ptr<std::vector<float>>> cache;
    const uint64_t key = qkv_cache_key(dim, seed);
    std::lock_guard<std::mutex> lock(mutex);
    auto found = cache.find(key);
    if (found != cache.end()) return found->second;
    auto matrix = std::make_shared<std::vector<float>>((size_t)dim * (size_t)dim);
    qkv_generate_qjl_matrix(matrix->data(), dim, seed);
    cache[key] = matrix;
    return matrix;
}

static int qkv_state_assign_codebook(qkv_state_t* state, int dim, int bits) {
    if (!state) return 0;
    if (qkv_bits_raw(bits)) return 1;
    if (!qkv_bits_codebook(bits)) return 0;
    auto codebook = qkv_cached_codebook(dim, bits, false);
    auto thresholds = qkv_cached_codebook(dim, bits, true);
    if (!codebook || !thresholds || codebook->empty() || thresholds->empty()) {
        return 0;
    }
    switch (bits) {
    case 1: state->codebook_1bit = codebook->data(); state->thresholds_1bit = thresholds->data(); break;
    case 2: state->codebook_2bit = codebook->data(); state->thresholds_2bit = thresholds->data(); break;
    case 3: state->codebook_3bit = codebook->data(); state->thresholds_3bit = thresholds->data(); break;
    case 4: state->codebook_4bit = codebook->data(); state->thresholds_4bit = thresholds->data(); break;
    case 5: state->codebook_5bit = codebook->data(); state->thresholds_5bit = thresholds->data(); break;
    case 6: state->codebook_6bit = codebook->data(); state->thresholds_6bit = thresholds->data(); break;
    case 7: state->codebook_7bit = codebook->data(); state->thresholds_7bit = thresholds->data(); break;
    case 8: state->codebook_8bit = codebook->data(); state->thresholds_8bit = thresholds->data(); break;
    default: return 0;
    }
    return 1;
}

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
    const int k_outlier_bits = qkv_outlier_bits_for_target(config, QKV_TARGET_KEY);
    const int k_normal_bits = qkv_normal_bits_for_target(config, QKV_TARGET_KEY);
    const int v_outlier_bits = qkv_outlier_bits_for_target(config, QKV_TARGET_VALUE);
    const int v_normal_bits = qkv_normal_bits_for_target(config, QKV_TARGET_VALUE);
    // BUGFIX 404: dim 범위 체크 강화
    if (dim <= 0 || dim > 16384 ||
        !qkv_bits_valid(k_bits) || !qkv_bits_valid(v_bits) ||
        !qkv_bits_valid(k_outlier_bits) || !qkv_bits_valid(k_normal_bits) ||
        !qkv_bits_valid(v_outlier_bits) || !qkv_bits_valid(v_normal_bits)) {
        return 0;
    }
    // BUGFIX 405: n_tokens 범위 체크
    if (n_tokens > INT_MAX / dim) {
        return 0;
    }
    const bool qjl_streams = config->enable_qjl &&
        qkv_bits_codebook(k_bits) && qkv_bits_codebook(v_bits) &&
        k_bits > 1 && v_bits > 1;
    const int k_storage_bits = qjl_streams ? k_bits - 1 : k_bits;
    const int v_storage_bits = qjl_streams ? v_bits - 1 : v_bits;
    if (!qkv_bits_valid(k_storage_bits) || !qkv_bits_valid(v_storage_bits)) {
        return 0;
    }

    // Allocate main KV storage
    // BUGFIX 406: k_packed_size overflow 방지
    if ((size_t)n_tokens > SIZE_MAX / ((size_t)dim * (size_t)k_storage_bits)) {
        return 0;
    }
    size_t k_packed_size = ((size_t)n_tokens * (size_t)dim * (size_t)k_storage_bits + 7) / 8;
    // BUGFIX 407: v_packed_size overflow 방지
    if ((size_t)n_tokens > SIZE_MAX / ((size_t)dim * (size_t)v_storage_bits)) {
        return 0;
    }
    size_t v_packed_size = ((size_t)n_tokens * (size_t)dim * (size_t)v_storage_bits + 7) / 8;

    state->k_idx = (uint8_t*)calloc(k_packed_size, 1);
    state->v_idx = (uint8_t*)calloc(v_packed_size, 1);
    state->k_norms = (float*)malloc((size_t)n_tokens * sizeof(float));
    state->v_norms = (float*)malloc((size_t)n_tokens * sizeof(float));

    if (!state->k_idx || !state->v_idx || !state->k_norms || !state->v_norms) {
        qkv_state_free(state);
        return 0;
    }

    // Allocate QJL residual storage
    if (qjl_streams) {
        // BUGFIX 408: QJL 할당 크기 overflow 방지
        if ((size_t)n_tokens > SIZE_MAX / (size_t)dim) {
            qkv_state_free(state);
            return 0;
        }
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

    if (config->enable_rotation) {
        auto matrix = qkv_cached_rotation(dim, config->rotation_seed);
        // BUGFIX 409: matrix 유효성 체크 강화
        if (!matrix || matrix->empty() || matrix->size() != (size_t)dim * (size_t)dim) {
            qkv_state_free(state);
            return 0;
        }
        state->rotation_matrix = matrix->data();
        if (!state->rotation_matrix) {
            qkv_state_free(state);
            return 0;
        }

        // BUGFIX Issue 9: Extract sign vector for Hadamard structure (power-of-2 dims)
        const bool is_power_of_2 = (dim & (dim - 1)) == 0;
        if (is_power_of_2) {
            state->rotation_signs = (float*)std::malloc((size_t)dim * sizeof(float));
            if (!state->rotation_signs) {
                qkv_state_free(state);
                return 0;
            }
            // Extract column signs from first row (Hadamard structure property)
            for (int col = 0; col < dim; ++col) {
                state->rotation_signs[col] = state->rotation_matrix[col] > 0.0f ? 1.0f : -1.0f;
            }
        }
    }

    if (qjl_streams) {
        state->qjl_signs_matrix = NULL;
        auto matrix = qkv_cached_qjl(dim, config->qjl_seed);
        // BUGFIX 410: matrix 유효성 체크 강화
        if (!matrix || matrix->empty() || matrix->size() != (size_t)dim * (size_t)dim) {
            qkv_state_free(state);
            return 0;
        }
        state->qjl_matrix = matrix->data();
        if (!state->qjl_matrix) {
            qkv_state_free(state);
            return 0;
        }
    }

    if (!qkv_state_assign_codebook(state, dim, k_storage_bits) ||
        !qkv_state_assign_codebook(state, dim, v_storage_bits) ||
        !qkv_state_assign_codebook(state, dim, k_outlier_bits) ||
        !qkv_state_assign_codebook(state, dim, k_normal_bits) ||
        !qkv_state_assign_codebook(state, dim, v_outlier_bits) ||
        !qkv_state_assign_codebook(state, dim, v_normal_bits)) {
        qkv_state_free(state);
        return 0;
    }

    const uint32_t sink_tokens = std::min<uint32_t>(
        config->sink_tokens,
        (uint32_t)std::max(n_tokens, 0));
    if (sink_tokens > 0) {
        if ((size_t)sink_tokens > SIZE_MAX / ((size_t)dim * sizeof(float))) {
            qkv_state_free(state);
            return 0;
        }
        state->k_sink = (float*)malloc((size_t)sink_tokens * (size_t)dim * sizeof(float));
        state->v_sink = (float*)malloc((size_t)sink_tokens * (size_t)dim * sizeof(float));
        if (!state->k_sink || !state->v_sink) {
            qkv_state_free(state);
            return 0;
        }
        state->sink_tokens = sink_tokens;
    }

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
        const bool k_split_qjl = qjl_streams &&
            qkv_bits_codebook(k_outlier_bits) && qkv_bits_codebook(k_normal_bits) &&
            k_outlier_bits > 1 && k_normal_bits > 1;
        const bool v_split_qjl = qjl_streams &&
            qkv_bits_codebook(v_outlier_bits) && qkv_bits_codebook(v_normal_bits) &&
            v_outlier_bits > 1 && v_normal_bits > 1;
        const int k_out_storage_bits = k_split_qjl ? k_outlier_bits - 1 : k_outlier_bits;
        const int k_norm_storage_bits = k_split_qjl ? k_normal_bits - 1 : k_normal_bits;
        const int v_out_storage_bits = v_split_qjl ? v_outlier_bits - 1 : v_outlier_bits;
        const int v_norm_storage_bits = v_split_qjl ? v_normal_bits - 1 : v_normal_bits;

        // BUGFIX 411: bits 범위 체크
        if (!qkv_bits_valid(k_out_storage_bits) || !qkv_bits_valid(k_norm_storage_bits) ||
            !qkv_bits_valid(v_out_storage_bits) || !qkv_bits_valid(v_norm_storage_bits) ||
            !qkv_state_assign_codebook(state, dim, k_out_storage_bits) ||
            !qkv_state_assign_codebook(state, dim, k_norm_storage_bits) ||
            !qkv_state_assign_codebook(state, dim, v_out_storage_bits) ||
            !qkv_state_assign_codebook(state, dim, v_norm_storage_bits)) {
            qkv_state_free(state);
            return 0;
        }

        // BUGFIX 412: n_out, n_norm 범위 체크
        if (n_out <= 0 || n_norm <= 0 || n_out >= dim) {
            qkv_state_free(state);
            return 0;
        }

        state->outlier_indices = (int*)malloc((size_t)n_out * sizeof(int));
        state->k_outlier_indices = (int*)malloc((size_t)n_out * sizeof(int));
        state->v_outlier_indices = (int*)malloc((size_t)n_out * sizeof(int));
        state->k_is_outlier = (uint8_t*)calloc((size_t)dim, 1);
        state->v_is_outlier = (uint8_t*)calloc((size_t)dim, 1);

        // BUGFIX 413: outlier 할당 크기 overflow 방지
        if ((size_t)n_tokens > SIZE_MAX / ((size_t)n_out * (size_t)std::max(k_out_storage_bits, v_out_storage_bits))) {
            qkv_state_free(state);
            return 0;
        }
        if ((size_t)n_tokens > SIZE_MAX / ((size_t)n_norm * (size_t)std::max(k_norm_storage_bits, v_norm_storage_bits))) {
            qkv_state_free(state);
            return 0;
        }
        size_t k_out_size = ((size_t)n_tokens * (size_t)n_out * (size_t)k_out_storage_bits + 7) / 8;
        size_t k_norm_size = ((size_t)n_tokens * (size_t)n_norm * (size_t)k_norm_storage_bits + 7) / 8;
        size_t v_out_size = ((size_t)n_tokens * (size_t)n_out * (size_t)v_out_storage_bits + 7) / 8;
        size_t v_norm_size = ((size_t)n_tokens * (size_t)n_norm * (size_t)v_norm_storage_bits + 7) / 8;

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
            // BUGFIX 414: ch 범위 체크
            if (ch < 0 || ch >= dim) {
                qkv_state_free(state);
                return 0;
            }
            state->outlier_indices[i] = ch;
            state->k_outlier_indices[i] = ch;
            state->v_outlier_indices[i] = ch;
            state->k_is_outlier[ch] = 1;
            state->v_is_outlier[ch] = 1;
        }
    }

    // BUGFIX 44: More conservative QKV thread pool sizing to prevent oversubscription ★★★
    // Old: available = hw - io_threads - 1, workers = min(available, 16)
    // New: available = max(1, hw - io_threads - 2), workers = min(available, 8)
    // Rationale: io_threads includes disk+pinned+gpu workers, but not cpu_row_workers
    //            cpu_row_workers and QKV pool both do CPU matmul at different times
    //            Reserve 2 cores (not 1) for OS + main thread + eviction worker
    //            Cap at 8 (not 16) to leave room for cpu_row_workers
    // Scenario: hw=16, io_threads=6 (3 disk + 2 pinned + 1 gpu)
    //           Old: QKV pool=9, cpu_row=N → 6+9+N threads compete
    //           New: QKV pool=8, cpu_row=N → better balance
    const unsigned hw_raw = std::thread::hardware_concurrency();
    // BUGFIX 415: engine_io_thread_count 음수 체크
    const int io_threads = config->engine_io_thread_count > 0
        ? (int)config->engine_io_thread_count : 0;
    const int available = std::max(1, (int)(hw_raw ? hw_raw : 4) - io_threads - 2);
    const int workers = std::min(available, 8);
    // BUGFIX 416: work_stride overflow 방지
    const int work_stride = std::max(dim, std::max(config->outlier_channels, dim - config->outlier_channels));
    if (work_stride <= 0 || workers <= 0) {
        qkv_state_free(state);
        return 0;
    }
    // BUGFIX 417: work buffer 할당 크기 overflow 방지
    if ((size_t)workers > SIZE_MAX / ((size_t)work_stride * sizeof(int))) {
        qkv_state_free(state);
        return 0;
    }
    if ((size_t)workers > SIZE_MAX / ((size_t)dim * sizeof(float))) {
        qkv_state_free(state);
        return 0;
    }
    state->work_codes_buf = (int*)malloc((size_t)workers * (size_t)work_stride * sizeof(int));
    state->work_qjl_buf = (float*)malloc((size_t)workers * (size_t)dim * sizeof(float));
    if (!state->work_codes_buf || !state->work_qjl_buf) {
        qkv_state_free(state);
        return 0;
    }
    state->work_buf_stride = work_stride;
    state->work_buf_workers = workers;
    // BUGFIX 418: thread_pool 생성 실패 체크
    try {
        state->thread_pool = new QkvThreadPool(workers);
    } catch (...) {
        state->thread_pool = nullptr;
        qkv_state_free(state);
        return 0;
    }
    if (!state->thread_pool) {
        qkv_state_free(state);
        return 0;
    }
    state->computed_workers = workers;

    state->n_tokens = n_tokens;
    state->head_dim = dim;
    state->k_bits = k_bits;
    state->v_bits = v_bits;

    // BUGFIX Problem 2: Windows QKV working set preparation ★★ IMPORTANT
    // Problem: QKV state memory (~0.69 GiB) allocated independently, not included in working set
    // → Page faults during attention operations block io_worker threads
    // Solution: Call prepare_windows_staging_working_set with total QKV allocation size
    // Impact: Eliminates blocking page faults on Windows during attention
    uint64_t total_qkv_bytes = k_packed_size + v_packed_size +
        (size_t)n_tokens * sizeof(float) * 2; // k_norms + v_norms
    total_qkv_bytes += (uint64_t)state->sink_tokens * (uint64_t)dim * (uint64_t)sizeof(float) * 2ull;
    if (qjl_streams) {
        total_qkv_bytes += ((size_t)n_tokens * (size_t)dim + 7) / 8 * 2; // k_qjl + v_qjl
        total_qkv_bytes += (size_t)n_tokens * sizeof(float) * 2; // residual_norms
    }
    if (config->outlier_channels > 0 && config->outlier_channels < dim) {
        int n_out = config->outlier_channels;
        int n_norm = dim - n_out;
        const bool k_split_qjl = qjl_streams &&
            qkv_bits_codebook(k_outlier_bits) && qkv_bits_codebook(k_normal_bits) &&
            k_outlier_bits > 1 && k_normal_bits > 1;
        const bool v_split_qjl = qjl_streams &&
            qkv_bits_codebook(v_outlier_bits) && qkv_bits_codebook(v_normal_bits) &&
            v_outlier_bits > 1 && v_normal_bits > 1;
        const int k_out_storage_bits = k_split_qjl ? k_outlier_bits - 1 : k_outlier_bits;
        const int k_norm_storage_bits = k_split_qjl ? k_normal_bits - 1 : k_normal_bits;
        const int v_out_storage_bits = v_split_qjl ? v_outlier_bits - 1 : v_outlier_bits;
        const int v_norm_storage_bits = v_split_qjl ? v_normal_bits - 1 : v_normal_bits;
        total_qkv_bytes += ((size_t)n_tokens * (size_t)n_out * (size_t)k_out_storage_bits + 7) / 8;
        total_qkv_bytes += ((size_t)n_tokens * (size_t)n_norm * (size_t)k_norm_storage_bits + 7) / 8;
        total_qkv_bytes += ((size_t)n_tokens * (size_t)n_out * (size_t)v_out_storage_bits + 7) / 8;
        total_qkv_bytes += ((size_t)n_tokens * (size_t)n_norm * (size_t)v_norm_storage_bits + 7) / 8;
        total_qkv_bytes += (size_t)n_tokens * sizeof(float) * 4; // outlier/normal norms
    }
    prepare_windows_staging_working_set(total_qkv_bytes);

    return 1;
}

void qkv_state_free(qkv_state_t* state) {
    if (!state) return;

    free(state->k_idx);
    free(state->v_idx);
    free(state->k_norms);
    free(state->v_norms);
    free(state->k_sink);
    free(state->v_sink);
    free(state->k_qjl);
    free(state->k_residual_norms);
    free(state->v_qjl);
    free(state->v_residual_norms);
    if (state->owns_rotation_matrix) free(state->rotation_matrix);
    free(state->rotation_signs);  // BUGFIX Issue 9: Free rotation_signs
    if (state->owns_qjl_matrix) free(state->qjl_matrix);
    free(state->qjl_signs_matrix);
    if (state->owns_codebooks) {
        free(state->codebook_1bit);
        free(state->thresholds_1bit);
        free(state->codebook_2bit);
        free(state->thresholds_2bit);
        free(state->codebook_3bit);
        free(state->thresholds_3bit);
        free(state->codebook_4bit);
        free(state->thresholds_4bit);
        free(state->codebook_5bit);
        free(state->thresholds_5bit);
        free(state->codebook_6bit);
        free(state->thresholds_6bit);
        free(state->codebook_7bit);
        free(state->thresholds_7bit);
        free(state->codebook_8bit);
        free(state->thresholds_8bit);
    }
    free(state->scratch_qjl_signs);
    free(state->scratch_s_t_qjl);
    free(state->scratch_residual);
    free(state->scratch_s_times_r);
    free(state->scratch_indices);
    free(state->scratch_y_tilde);
    free(state->scratch_x_tilde);
    free(state->scratch_attention);
    free(state->scratch_rotated_q);
    free(state->work_codes_buf);
    free(state->work_qjl_buf);
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
