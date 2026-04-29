#include "qkv_matrix.h"
#include <random>
#include <vector>
#include <math.h>

void qkv_generate_rotation_matrix(
    float* Pi,
    int dim,
    uint64_t seed
) {
    if (!Pi || dim <= 0) return;

    std::mt19937_64 rng(seed);

    const bool power_of_two = (dim & (dim - 1)) == 0;
    if (power_of_two) {
        // Fast Hadamard-like construction for power-of-2
        std::vector<float> signs((size_t)dim);
        for (int col = 0; col < dim; ++col) {
            signs[(size_t)col] = (rng() & 1ull) ? 1.0f : -1.0f;
        }
        const float scale = 1.0f / sqrtf((float)dim);
        for (int row = 0; row < dim; ++row) {
            for (int col = 0; col < dim; ++col) {
                unsigned v = (unsigned)(row & col);
                v ^= v >> 16;
                v ^= v >> 8;
                v ^= v >> 4;
                v &= 0xFu;
                const int parity = (0x6996u >> v) & 1u;
                Pi[row * dim + col] = (parity ? -scale : scale) * signs[(size_t)col];
            }
        }
        return;
    }

    // Non-power-of-2: QR decomposition
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> random_matrix((size_t)dim * (size_t)dim);
    for (int i = 0; i < dim * dim; ++i) {
        random_matrix[(size_t)i] = dist(rng);
    }

    // Modified Gram-Schmidt
    for (int col = 0; col < dim; ++col) {
        for (int row = 0; row < dim; ++row) {
            Pi[row * dim + col] = random_matrix[(size_t)row * (size_t)dim + (size_t)col];
        }
        for (int prev = 0; prev < col; ++prev) {
            float dot = 0.0f;
            for (int row = 0; row < dim; ++row) {
                dot += Pi[row * dim + prev] * Pi[row * dim + col];
            }
            for (int row = 0; row < dim; ++row) {
                Pi[row * dim + col] -= dot * Pi[row * dim + prev];
            }
        }
        float norm = 0.0f;
        for (int row = 0; row < dim; ++row) {
            norm += Pi[row * dim + col] * Pi[row * dim + col];
        }
        norm = sqrtf(norm);
        if (norm > 1e-8f) {
            for (int row = 0; row < dim; ++row) {
                Pi[row * dim + col] /= norm;
            }
        }
    }
}

void qkv_generate_qjl_matrix(
    float* S,
    int dim,
    uint64_t seed
) {
    if (!S || dim <= 0) return;

    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // Fill with i.i.d. N(0, 1) entries per paper Lemma 4
    for (int i = 0; i < dim * dim; i++) {
        S[i] = dist(rng);
    }
}
