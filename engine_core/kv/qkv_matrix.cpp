#include "qkv_matrix.h"
#include <random>
#include <vector>
#include <math.h>
#include <string.h>

bool qkv_dim_is_power_of_two(int dim) {
    return dim > 0 && (dim & (dim - 1)) == 0;
}

void qkv_fwht_inplace(float* data, int dim) {
    if (!data || !qkv_dim_is_power_of_two(dim)) return;
    for (int h = 1; h < dim; h <<= 1) {
        for (int i = 0; i < dim; i += h << 1) {
            for (int j = i; j < i + h; ++j) {
                const float a = data[j];
                const float b = data[j + h];
                data[j] = a + b;
                data[j + h] = a - b;
            }
        }
    }
}

int qkv_apply_hadamard_rotation_forward(
    const float* input,
    const float* signs,
    float* output,
    int dim
) {
    if (!input || !signs || !output || !qkv_dim_is_power_of_two(dim) || dim > 16384) {
        return 0;
    }
    for (int i = 0; i < dim; ++i) {
        output[i] = input[i] * signs[i];
    }
    qkv_fwht_inplace(output, dim);
    const float scale = 1.0f / sqrtf((float)dim);
    for (int i = 0; i < dim; ++i) {
        output[i] *= scale;
    }
    return 1;
}

int qkv_apply_hadamard_rotation_inverse(
    const float* input,
    const float* signs,
    float* output,
    int dim
) {
    if (!input || !signs || !output || !qkv_dim_is_power_of_two(dim) || dim > 16384) {
        return 0;
    }
    memcpy(output, input, (size_t)dim * sizeof(float));
    qkv_fwht_inplace(output, dim);
    const float scale = 1.0f / sqrtf((float)dim);
    for (int i = 0; i < dim; ++i) {
        output[i] *= scale * signs[i];
    }
    return 1;
}

void qkv_generate_rotation_matrix(
    float* Pi,
    int dim,
    uint64_t seed
) {
    // BUGFIX 386: 파라미터 유효성 체크
    if (!Pi || dim <= 0 || dim > 16384) return;

    std::mt19937_64 rng(seed);

    const bool power_of_two = (dim & (dim - 1)) == 0;
    if (power_of_two) {
        // Fast Hadamard-like construction for power-of-2
        std::vector<float> signs((size_t)dim);
        for (int col = 0; col < dim; ++col) {
            signs[(size_t)col] = (rng() & 1ull) ? 1.0f : -1.0f;
        }
        // BUGFIX 387: dim이 0일 때 division by zero 방지
        if (dim <= 0) return;
        const float scale = 1.0f / sqrtf((float)dim);
        for (int row = 0; row < dim; ++row) {
            for (int col = 0; col < dim; ++col) {
                // BUGFIX 388: 배열 인덱스 overflow 방지
                size_t idx = (size_t)row * (size_t)dim + (size_t)col;
                unsigned v = (unsigned)(row & col);
                v ^= v >> 16;
                v ^= v >> 8;
                v ^= v >> 4;
                v &= 0xFu;
                const int parity = (0x6996u >> v) & 1u;
                Pi[idx] = (parity ? -scale : scale) * signs[(size_t)col];
            }
        }
        return;
    }

    // Non-power-of-2: QR decomposition
    std::normal_distribution<float> dist(0.0f, 1.0f);
    // BUGFIX 389: dim * dim overflow 체크
    if (dim > 46340) return;  // sqrt(INT_MAX) 근사값
    std::vector<float> random_matrix((size_t)dim * (size_t)dim);
    for (int i = 0; i < dim * dim; ++i) {
        random_matrix[(size_t)i] = dist(rng);
    }

    // Modified Gram-Schmidt
    for (int col = 0; col < dim; ++col) {
        for (int row = 0; row < dim; ++row) {
            // BUGFIX 390: 배열 인덱스 overflow 방지
            size_t idx = (size_t)row * (size_t)dim + (size_t)col;
            Pi[idx] = random_matrix[idx];
        }
        for (int prev = 0; prev < col; ++prev) {
            float dot = 0.0f;
            for (int row = 0; row < dim; ++row) {
                size_t col_idx = (size_t)row * (size_t)dim + (size_t)col;
                size_t prev_idx = (size_t)row * (size_t)dim + (size_t)prev;
                dot += Pi[prev_idx] * Pi[col_idx];
            }
            for (int row = 0; row < dim; ++row) {
                size_t col_idx = (size_t)row * (size_t)dim + (size_t)col;
                size_t prev_idx = (size_t)row * (size_t)dim + (size_t)prev;
                Pi[col_idx] -= dot * Pi[prev_idx];
            }
        }
        float norm = 0.0f;
        for (int row = 0; row < dim; ++row) {
            size_t idx = (size_t)row * (size_t)dim + (size_t)col;
            norm += Pi[idx] * Pi[idx];
        }
        norm = sqrtf(norm);
        // BUGFIX 391: norm이 0일 때 division by zero 방지
        if (norm > 1e-8f) {
            for (int row = 0; row < dim; ++row) {
                size_t idx = (size_t)row * (size_t)dim + (size_t)col;
                Pi[idx] /= norm;
            }
        }
    }
}

void qkv_generate_qjl_matrix(
    float* S,
    int dim,
    uint64_t seed
) {
    // BUGFIX 392: 파라미터 유효성 체크
    if (!S || dim <= 0 || dim > 16384) return;

    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // Fill with i.i.d. N(0, 1) entries per paper Lemma 4
    // BUGFIX 393: dim * dim overflow 체크
    if (dim > 46340) return;  // sqrt(INT_MAX) 근사값
    for (int i = 0; i < dim * dim; i++) {
        S[i] = dist(rng);
    }
}
