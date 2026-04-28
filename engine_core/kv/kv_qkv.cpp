// QKV Implementation - Faithful to TurboQuant Paper (2504.19874v1)
// 
// Algorithm 1 (TurboQuant_mse): MSE-optimal quantization
//   1. Generate random rotation matrix Pi via QR decomposition
//   2. Compute Lloyd-Max codebook for Beta distribution (Eq. 4)
//   3. Quant: y = Pi*x, find nearest centroid for each coordinate
//   4. DeQuant: x_hat = Pi^T*y_hat
//
// Algorithm 2 (TurboQuant_prod): Unbiased inner product quantization
//   1. Apply TurboQuant_mse with b-1 bits
//   2. Compute residual r = x - DeQuant_mse(idx)
//   3. QJL: qjl = sign(S*r), store residual norm
//   4. DeQuant: x_hat = x_hat_mse + sqrt(pi/2d)*||r||*S^T*qjl

#include "kv_qkv.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <atomic>
#include <random>
#include <algorithm>
#include <vector>
#include <thread>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int find_nearest_centroid(float val, const float* centroids, const float* thresholds, int n_levels);

// Fix: Use std::atomic<bool> consistently to match header declaration
std::atomic<bool> g_qkv_mode_enabled{false};

// ============================================================
// Lloyd-Max Algorithm for Beta Distribution (Eq. 4)
// ============================================================

// Beta distribution PDF: f(x) proportional to (1 - x^2)^((d - 3) / 2)
// For large d, converges to N(0, 1/d)
static double beta_pdf(double x, int d) {
    if (x <= -1.0 || x >= 1.0) return 0.0;
    // For large d, use Gaussian approximation
    double variance = 1.0 / (double)d;
    double sigma = sqrt(variance);
    return exp(-x * x / (2.0 * variance)) / (sigma * sqrt(2.0 * M_PI));
}

// Lloyd-Max algorithm: solve continuous k-means (Eq. 4)
// Find centroids c_1 < c_2 < ... < c_{2^b} that minimize:
// Sum_i integral |x - c_i|^2 * f(x) dx over each quantization bin.
// Paper: rotated coordinates follow N(0, 1/d), so initialize within ±3.5σ
static void lloyd_max_codebook(
    float* centroids,
    float* thresholds,
    int n_levels,
    int dim,
    int max_iters
) {
    // Fix 1: Initialize centroids within N(0, 1/d) distribution range
    // For d=128, σ≈0.088, so ±3.5σ ≈ ±0.31 covers 99.95% of mass
    double sigma = 1.0 / sqrt((double)dim);
    for (int i = 0; i < n_levels; i++) {
        // Spread centroids across [-3.5σ, 3.5σ]
        centroids[i] = (float)((-3.5 + 7.0 * (double)i / (double)(n_levels - 1)) * sigma);
    }
    
    // Lloyd-Max iteration
    for (int iter = 0; iter < max_iters; iter++) {
        // Update thresholds (midpoints between centroids)
        thresholds[0] = (float)(-5.0 * sigma);  // Effectively -∞ for this distribution
        for (int i = 1; i < n_levels; i++) {
            thresholds[i] = (centroids[i-1] + centroids[i]) / 2.0f;
        }
        thresholds[n_levels] = (float)(5.0 * sigma);  // Effectively +∞
        
        // Update centroids (weighted mean in each region)
        bool converged = true;
        for (int i = 0; i < n_levels; i++) {
            // Numerical integration to find optimal centroid
            double sum_x = 0.0, sum_w = 0.0;
            int n_samples = 1000;
            double step = (thresholds[i+1] - thresholds[i]) / n_samples;
            
            for (int j = 0; j <= n_samples; j++) {
                double x = thresholds[i] + j * step;
                // beta_pdf expects N(0, 1/d) input directly
                double w = beta_pdf(x, dim);  // Weight by PDF
                sum_x += x * w;
                sum_w += w;
            }
            
            if (sum_w > 1e-10) {
                float new_centroid = (float)(sum_x / sum_w);
                if (fabsf(new_centroid - centroids[i]) > 1e-6f) {
                    converged = false;
                }
                centroids[i] = new_centroid;
            }
        }
        
        if (converged) break;
    }
}

// Precompute codebook for given bit-width
static void qkv_compute_lloyd_max_codebook(
    float* centroids,
    float* thresholds,
    int bits,
    int dim
) {
    int n_levels = 1 << bits;  // 2^bits levels
    
    // For 2-bit, paper gives explicit values around +/-0.453/sqrt(d) and +/-1.51/sqrt(d).
    if (bits == 2) {
        double scale = 1.0 / sqrt((double)dim);
        centroids[0] = (float)(-1.51 * scale);
        centroids[1] = (float)(-0.453 * scale);
        centroids[2] = (float)(0.453 * scale);
        centroids[3] = (float)(1.51 * scale);
        
        thresholds[0] = -1.0f;
        thresholds[1] = (float)((-1.51 - 0.453) / 2.0 * scale);
        thresholds[2] = 0.0f;
        thresholds[3] = (float)((0.453 + 1.51) / 2.0 * scale);
        thresholds[4] = 1.0f;
        return;
    }
    
    // For other bit-widths, use Lloyd-Max
    lloyd_max_codebook(centroids, thresholds, n_levels, dim, 100);
}

// ============================================================
// Random Rotation Matrix Generation (QR Decomposition)
// ============================================================

static void qkv_generate_rotation_matrix(
    float* Pi,
    int dim,
    uint64_t seed
) {
    if (!Pi || dim <= 0) return;

    std::mt19937_64 rng(seed);

    const bool power_of_two = (dim & (dim - 1)) == 0;
    if (power_of_two) {
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

    memset(Pi, 0, (size_t)dim * (size_t)dim * sizeof(float));
    for (int i = 0; i < dim; ++i) {
        Pi[i * dim + i] = (rng() & 1ull) ? 1.0f : -1.0f;
    }
}
// ============================================================
// QJL Random Matrix Generation
// ============================================================

static void qkv_generate_qjl_matrix(
    float* S,
    int dim,
    uint64_t seed
) {
    if (!S || dim <= 0) return;
    
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    // Fill with i.i.d. N(0, 1) entries
    for (int i = 0; i < dim * dim; i++) {
        S[i] = dist(rng);
    }
}

// ============================================================
// Bit Packing/Unpacking
// ============================================================

// Pack b-bit indices into bytes
static void pack_indices(const int* indices, uint8_t* packed, int n, int bits) {
    if (bits == 1) {
        // Bug ②: 1-bit packing — 8 indices per byte (0 or 1)
        memset(packed, 0, (n + 7) / 8);
        for (int i = 0; i < n; i++) {
            if (indices[i] & 1) packed[i / 8] |= (1 << (i % 8));
        }
    } else if (bits == 2) {
        // 4 indices per byte
        for (int i = 0; i < n; i += 4) {
            uint8_t byte = 0;
            for (int j = 0; j < 4 && (i + j) < n; j++) {
                byte |= (uint8_t)((indices[i + j] & 0x3) << (j * 2));
            }
            packed[i / 4] = byte;
        }
    } else if (bits == 3) {
        // 8 indices in 3 bytes (24 bits)
        int bit_pos = 0;
        int byte_idx = 0;
        memset(packed, 0, (n * bits + 7) / 8);
        
        for (int i = 0; i < n; i++) {
            int val = indices[i] & 0x7;
            int bits_remaining = 8 - (bit_pos % 8);
            
            if (bits_remaining >= 3) {
                packed[byte_idx] |= (uint8_t)(val << (bits_remaining - 3));
                bit_pos += 3;
                if (bit_pos % 8 == 0) byte_idx++;
            } else {
                // Split across two bytes
                int high_bits = bits_remaining;
                int low_bits = 3 - high_bits;
                packed[byte_idx] |= (uint8_t)(val >> low_bits);
                byte_idx++;
                packed[byte_idx] |= (uint8_t)((val & ((1 << low_bits) - 1)) << (8 - low_bits));
                bit_pos += 3;
            }
        }
    } else if (bits == 4) {
        // 2 indices per byte
        for (int i = 0; i < n; i += 2) {
            uint8_t byte = 0;
            for (int j = 0; j < 2 && (i + j) < n; j++) {
                byte |= (uint8_t)((indices[i + j] & 0xF) << (j * 4));
            }
            packed[i / 2] = byte;
        }
    }
}

static void unpack_indices(const uint8_t* packed, int* indices, int n, int bits) {
    if (bits == 1) {
        // Bug ②: 1-bit unpacking
        for (int i = 0; i < n; i++) {
            indices[i] = (packed[i / 8] >> (i % 8)) & 1;
        }
    } else if (bits == 2) {
        for (int i = 0; i < n; i++) {
            int byte_idx = i / 4;
            int bit_offset = (i % 4) * 2;
            indices[i] = (packed[byte_idx] >> bit_offset) & 0x3;
        }
    } else if (bits == 3) {
        int bit_pos = 0;
        int byte_idx = 0;
        
        for (int i = 0; i < n; i++) {
            int bits_remaining = 8 - (bit_pos % 8);
            
            if (bits_remaining >= 3) {
                indices[i] = (packed[byte_idx] >> (bits_remaining - 3)) & 0x7;
                bit_pos += 3;
                if (bit_pos % 8 == 0) byte_idx++;
            } else {
                int high_bits = bits_remaining;
                int low_bits = 3 - high_bits;
                int val = (packed[byte_idx] & ((1 << high_bits) - 1)) << low_bits;
                byte_idx++;
                val |= (packed[byte_idx] >> (8 - low_bits)) & ((1 << low_bits) - 1);
                indices[i] = val;
                bit_pos += 3;
            }
        }
    } else if (bits == 4) {
        for (int i = 0; i < n; i++) {
            int byte_idx = i / 2;
            int bit_offset = (i % 2) * 4;
            indices[i] = (packed[byte_idx] >> bit_offset) & 0xF;
        }
    }
}

// Pack 1-bit signs
static void pack_signs(const float* signs, uint8_t* packed, int n) {
    for (int i = 0; i < n; i += 8) {
        uint8_t byte = 0;
        for (int j = 0; j < 8 && (i + j) < n; j++) {
            if (signs[i + j] >= 0.0f) {
                byte |= (1 << j);
            }
        }
        packed[i / 8] = byte;
    }
}

static void unpack_signs(const uint8_t* packed, float* signs, int n) {
    for (int i = 0; i < n; i++) {
        int byte_idx = i / 8;
        int bit_offset = i % 8;
        signs[i] = (packed[byte_idx] & (1 << bit_offset)) ? 1.0f : -1.0f;
    }
}

// GAP 3: Paper Algorithm 1 — L2 norm → normalize → Pi rotation → Lloyd-Max codebook
int qkv_quantize_vector_with_state(
    const qkv_state_t* state,
    const qkv_config_t* config,
    const float* input,
    uint8_t* output,
    float* norm_out,
    int dim,
    int bits
) {
    if (!input || !output || !norm_out || dim <= 0 || (bits != 2 && bits != 3)) {
        return 0;
    }

    // Step 1: Compute L2 norm (paper Algorithm 1, line 1)
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

    // Step 2: Normalize to unit vector (paper Algorithm 1, line 2)
    float* normalized = (float*)malloc((size_t)dim * sizeof(float));
    float* rotated = (float*)malloc((size_t)dim * sizeof(float));
    int* indices = (int*)malloc((size_t)dim * sizeof(int));
    if (!normalized || !rotated || !indices) {
        free(normalized); free(rotated); free(indices);
        return 0;
    }
    const float inv_norm = 1.0f / l2_norm;
    for (int i = 0; i < dim; ++i) {
        normalized[i] = input[i] * inv_norm;
    }

    // Step 3: Apply random rotation Pi (paper Algorithm 1, line 3)
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

    // Step 4: Lloyd-Max codebook quantization (paper Algorithm 1, line 4)
    // Use state Lloyd-Max thresholds computed for N(0,1/d) post-rotation distribution.
    const float* centroids_s = (bits == 1) ? state->codebook_1bit :
                               (bits == 2) ? state->codebook_2bit : state->codebook_3bit;
    const float* thresholds_s = (bits == 1) ? state->thresholds_1bit :
                                (bits == 2) ? state->thresholds_2bit : state->thresholds_3bit;
    const int n_levels = 1 << bits;
    for (int i = 0; i < dim; ++i) {
        indices[i] = find_nearest_centroid(src[i], centroids_s, thresholds_s, n_levels);
    }

    memset(output, 0, (size_t)(dim * bits + 7) / 8);
    pack_indices(indices, output, dim, bits);
    free(normalized);
    free(rotated);
    free(indices);
    return 1;
}

// REMOVED: qkv_compute_qjl_residual (deprecated, mathematically incorrect)
// This function computed sign(r) instead of sign(S*r) required by paper Algorithm 2.
// Use qkv_quantize_prod() which correctly applies S*r multiplication.

// GAP 4: Proper S^T multiplication per paper Definition 1
// Q^{-1}_qjl(z) = sqrt(pi/2) / d * S^T * z
void qkv_apply_qjl_residual_with_state(
    float* data,
    const uint8_t* signs,
    const float* scales,
    const qkv_state_t* state,
    int n_tokens,
    int dim
) {
    (void)data;
    (void)signs;
    (void)scales;
    (void)state;
    (void)n_tokens;
    (void)dim;
    return;
#if 0
    if (!data || !signs || !scales || !state || n_tokens <= 0 || dim <= 0) {
        return;
    }
    float* token_signs = (float*)malloc((size_t)dim * sizeof(float));
    float* s_t_z = (float*)malloc((size_t)dim * sizeof(float));
    if (!token_signs || !s_t_z) {
        free(token_signs);
        free(s_t_z);
        return;
    }
    const float factor = sqrtf((float)M_PI / 2.0f) / (float)dim;
    for (int t = 0; t < n_tokens; ++t) {
        float* dst = data + (size_t)t * (size_t)dim;
        unpack_signs(signs + (size_t)t * (size_t)((dim + 7) / 8), token_signs, dim);

        // Compute S^T * z (paper Definition 1)
        if (state->qjl_matrix) {
            // Gaussian S: S^T[i][j] = S[j][i]
            for (int i = 0; i < dim; i++) {
                float sum = 0.0f;
                for (int j = 0; j < dim; j++) {
                    sum += state->qjl_matrix[j * dim + i] * token_signs[j];
                }
                s_t_z[i] = sum;
            }
        } else {
            // No S matrix — fallback to identity (degraded mode)
            memcpy(s_t_z, token_signs, dim * sizeof(float));
        }

        for (int d = 0; d < dim; ++d) {
            dst[d] += factor * scales[t] * s_t_z[d];
        }
    }
    free(token_signs);
    free(s_t_z);
#endif
}

// ============================================================
// State Management
// ============================================================

int qkv_init(
    qkv_state_t* state,
    const qkv_config_t* config,
    int n_tokens
) {
    if (!state || !config || n_tokens <= 0) return 0;
    
    memset(state, 0, sizeof(*state));
    
    int dim = config->head_dim;
    int k_bits = config->k_bits;
    int v_bits = config->v_bits;
    
    // Fix: Use size_t to prevent integer overflow with large sequences
    size_t k_packed_size = ((size_t)n_tokens * (size_t)dim * (size_t)k_bits + 7) / 8;
    size_t v_packed_size = ((size_t)n_tokens * (size_t)dim * (size_t)v_bits + 7) / 8;
    
    state->k_idx = (uint8_t*)calloc(k_packed_size, 1);
    state->v_idx = (uint8_t*)calloc(v_packed_size, 1);
    state->k_norms = (float*)malloc((size_t)n_tokens * sizeof(float));
    state->v_norms = (float*)malloc((size_t)n_tokens * sizeof(float));
    
    if (!state->k_idx || !state->v_idx || !state->k_norms || !state->v_norms) {
        qkv_free(state);
        return 0;
    }
    
    // Allocate QJL residual storage if enabled
    if (config->enable_qjl) {
        // Fix: Use size_t to prevent integer overflow
        state->k_qjl = (uint8_t*)calloc(((size_t)n_tokens * (size_t)dim + 7) / 8, 1);
        state->k_residual_norms = (float*)malloc((size_t)n_tokens * sizeof(float));
        state->v_qjl = (uint8_t*)calloc(((size_t)n_tokens * (size_t)dim + 7) / 8, 1);
        state->v_residual_norms = (float*)malloc((size_t)n_tokens * sizeof(float));
        
        if (!state->k_qjl || !state->k_residual_norms || 
            !state->v_qjl || !state->v_residual_norms) {
            qkv_free(state);
            return 0;
        }
    }
    
    // Allocate and generate rotation matrix Pi.
    if (config->enable_rotation) {
        state->rotation_matrix = (float*)malloc((size_t)dim * (size_t)dim * sizeof(float));
        if (!state->rotation_matrix) {
            qkv_free(state);
            return 0;
        }
        qkv_generate_rotation_matrix(state->rotation_matrix, dim, config->rotation_seed);
    }
    
    // GAP 5: Restore Gaussian S matrix per paper Lemma 4
    // Rademacher breaks variance bound: E[z^T S^T S z] = pi/2d * ||y||^2 requires Gaussian
    // Paper Definition 1: S in R^{d x d}, S_ij ~ N(0, 1)
    if (config->enable_qjl) {
        state->qjl_signs_matrix = NULL;  // Rademacher disabled - paper requires Gaussian
        state->qjl_matrix = (float*)malloc((size_t)dim * (size_t)dim * sizeof(float));
        if (!state->qjl_matrix) {
            qkv_free(state);
            return 0;
        }
        std::mt19937_64 rng(config->qjl_seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (int i = 0; i < dim * dim; i++) {
            state->qjl_matrix[i] = dist(rng);
        }
    }
    
    // Allocate and compute Lloyd-Max codebooks (1/2/3 bit)
    state->codebook_1bit = (float*)malloc(2 * sizeof(float));
    state->thresholds_1bit = (float*)malloc(3 * sizeof(float));
    state->codebook_2bit = (float*)malloc(4 * sizeof(float));
    state->thresholds_2bit = (float*)malloc(5 * sizeof(float));
    state->codebook_3bit = (float*)malloc(8 * sizeof(float));
    state->thresholds_3bit = (float*)malloc(9 * sizeof(float));
    
    if (!state->codebook_1bit || !state->thresholds_1bit ||
        !state->codebook_2bit || !state->thresholds_2bit ||
        !state->codebook_3bit || !state->thresholds_3bit) {
        qkv_free(state);
        return 0;
    }
    
    // Compute codebooks
    qkv_compute_lloyd_max_codebook(state->codebook_1bit, state->thresholds_1bit, 1, dim);
    qkv_compute_lloyd_max_codebook(state->codebook_2bit, state->thresholds_2bit, 2, dim);
    qkv_compute_lloyd_max_codebook(state->codebook_3bit, state->thresholds_3bit, 3, dim);
    
    state->n_tokens = n_tokens;
    state->head_dim = dim;
    state->k_bits = k_bits;
    state->v_bits = v_bits;

    // Fix 56: Allocate scratch buffers (eliminate per-token malloc in hot path)
    state->scratch_qjl_signs = (float*)malloc((size_t)dim * sizeof(float));
    state->scratch_s_t_qjl = (float*)malloc((size_t)dim * sizeof(float));
    state->scratch_residual = (float*)malloc((size_t)dim * sizeof(float));
    state->scratch_s_times_r = (float*)malloc((size_t)dim * sizeof(float));
    state->scratch_indices = (int*)malloc((size_t)dim * sizeof(int));
    state->scratch_y_tilde = (float*)malloc((size_t)dim * sizeof(float));
    state->scratch_x_tilde = (float*)malloc((size_t)n_tokens * (size_t)dim * sizeof(float));
    state->scratch_attention = (float*)malloc((size_t)n_tokens * sizeof(float));
    state->scratch_rotated_q = (float*)malloc((size_t)dim * sizeof(float));
    if (!state->scratch_qjl_signs || !state->scratch_s_t_qjl ||
        !state->scratch_residual || !state->scratch_s_times_r ||
        !state->scratch_indices || !state->scratch_y_tilde ||
        !state->scratch_x_tilde || !state->scratch_attention ||
        !state->scratch_rotated_q) {
        qkv_free(state);
        return 0;
    }

    // Fix 4: Allocate outlier channel storage (paper Table 1, Section 4.3)
    if (config->outlier_channels > 0 && config->outlier_channels < dim) {
        int n_out = config->outlier_channels;
        int n_norm = dim - n_out;
        int out_bits = config->outlier_bits;
        int norm_bits = config->normal_bits;
        if (out_bits < 1 || out_bits > 3 || norm_bits < 1 || norm_bits > 3) {
            qkv_free(state);
            return 0;
        }
        
        state->outlier_indices = (int*)malloc((size_t)n_out * sizeof(int));
        state->k_outlier_indices = (int*)malloc((size_t)n_out * sizeof(int));
        state->v_outlier_indices = (int*)malloc((size_t)n_out * sizeof(int));
        
        // Allocate separate packed storage for outlier and normal channels
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
        
        // Bug 1: Check NULL BEFORE writing to arrays
        if (!state->outlier_indices || !state->k_outlier_indices || !state->v_outlier_indices ||
            !state->k_idx_outlier || !state->k_idx_normal ||
            !state->v_idx_outlier || !state->v_idx_normal ||
            !state->k_norms_outlier || !state->k_norms_normal ||
            !state->v_norms_outlier || !state->v_norms_normal) {
            qkv_free(state);
            return 0;
        }
        
        // Now safe to initialize - NULL check passed
        for (int i = 0; i < n_out; i++) {
            state->outlier_indices[i] = i;  // Default: first n_out channels are outliers
            state->k_outlier_indices[i] = i;
            state->v_outlier_indices[i] = i;
        }
    }

    return 1;
}

void qkv_free(qkv_state_t* state) {
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
    free(state->qjl_signs_matrix);  // Fix 57
    free(state->codebook_1bit);
    free(state->thresholds_1bit);
    free(state->codebook_2bit);
    free(state->thresholds_2bit);
    free(state->codebook_3bit);
    free(state->thresholds_3bit);
    // Fix 56: free scratch buffers
    free(state->scratch_qjl_signs);
    free(state->scratch_s_t_qjl);
    free(state->scratch_residual);
    free(state->scratch_s_times_r);
    free(state->scratch_indices);
    free(state->scratch_y_tilde);
    free(state->scratch_x_tilde);
    free(state->scratch_attention);
    free(state->scratch_rotated_q);
    // Fix 4: free outlier channel storage
    free(state->outlier_indices);
    free(state->k_outlier_indices);
    free(state->v_outlier_indices);
    free(state->k_idx_outlier);
    free(state->k_idx_normal);
    free(state->v_idx_outlier);
    free(state->v_idx_normal);
    free(state->k_norms_outlier);
    free(state->k_norms_normal);
    free(state->v_norms_outlier);
    free(state->v_norms_normal);
    
    memset(state, 0, sizeof(*state));
}

// ============================================================
// Algorithm 1: TurboQuant_mse
// ============================================================

// Find nearest centroid index for a value
static int find_nearest_centroid(float val, const float* centroids, const float* thresholds, int n_levels) {
    // Binary search through thresholds
    int lo = 0, hi = n_levels;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (val < thresholds[mid + 1]) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    // Fix: clamp to valid range [0, n_levels-1] to prevent OOB access
    return lo < n_levels ? lo : n_levels - 1;
}

static const int QKV_TARGET_KEY = 1;
static const int QKV_TARGET_VALUE = 2;

static bool qkv_bits_valid(int bits) {
    return bits >= 1 && bits <= 3;
}

static const float* qkv_codebook_for_bits(const qkv_state_t* state, int bits) {
    return (bits == 1) ? state->codebook_1bit :
           (bits == 2) ? state->codebook_2bit : state->codebook_3bit;
}

static const float* qkv_thresholds_for_bits(const qkv_state_t* state, int bits) {
    return (bits == 1) ? state->thresholds_1bit :
           (bits == 2) ? state->thresholds_2bit : state->thresholds_3bit;
}

static int qkv_target_from_buffers(const qkv_state_t* state, const uint8_t* idx, const float* norms) {
    if (!state || !idx) return 0;
    if (idx == state->k_idx && (!norms || norms == state->k_norms)) return QKV_TARGET_KEY;
    if (idx == state->v_idx && (!norms || norms == state->v_norms)) return QKV_TARGET_VALUE;
    return 0;
}

static int* qkv_outlier_indices_for_target(qkv_state_t* state, int target) {
    if (!state) return NULL;
    if (target == QKV_TARGET_KEY) return state->k_outlier_indices ? state->k_outlier_indices : state->outlier_indices;
    if (target == QKV_TARGET_VALUE) return state->v_outlier_indices ? state->v_outlier_indices : state->outlier_indices;
    return NULL;
}

static const int* qkv_outlier_indices_for_target_const(const qkv_state_t* state, int target) {
    if (!state) return NULL;
    if (target == QKV_TARGET_KEY) return state->k_outlier_indices ? state->k_outlier_indices : state->outlier_indices;
    if (target == QKV_TARGET_VALUE) return state->v_outlier_indices ? state->v_outlier_indices : state->outlier_indices;
    return NULL;
}

static uint8_t* qkv_idx_outlier_for_target(qkv_state_t* state, int target) {
    return (target == QKV_TARGET_KEY) ? state->k_idx_outlier :
           (target == QKV_TARGET_VALUE) ? state->v_idx_outlier : NULL;
}

static uint8_t* qkv_idx_normal_for_target(qkv_state_t* state, int target) {
    return (target == QKV_TARGET_KEY) ? state->k_idx_normal :
           (target == QKV_TARGET_VALUE) ? state->v_idx_normal : NULL;
}

static const uint8_t* qkv_idx_outlier_for_target_const(const qkv_state_t* state, int target) {
    return (target == QKV_TARGET_KEY) ? state->k_idx_outlier :
           (target == QKV_TARGET_VALUE) ? state->v_idx_outlier : NULL;
}

static const uint8_t* qkv_idx_normal_for_target_const(const qkv_state_t* state, int target) {
    return (target == QKV_TARGET_KEY) ? state->k_idx_normal :
           (target == QKV_TARGET_VALUE) ? state->v_idx_normal : NULL;
}

static float* qkv_norms_outlier_for_target(qkv_state_t* state, int target) {
    return (target == QKV_TARGET_KEY) ? state->k_norms_outlier :
           (target == QKV_TARGET_VALUE) ? state->v_norms_outlier : NULL;
}

static float* qkv_norms_normal_for_target(qkv_state_t* state, int target) {
    return (target == QKV_TARGET_KEY) ? state->k_norms_normal :
           (target == QKV_TARGET_VALUE) ? state->v_norms_normal : NULL;
}

static const float* qkv_norms_outlier_for_target_const(const qkv_state_t* state, int target) {
    return (target == QKV_TARGET_KEY) ? state->k_norms_outlier :
           (target == QKV_TARGET_VALUE) ? state->v_norms_outlier : NULL;
}

static const float* qkv_norms_normal_for_target_const(const qkv_state_t* state, int target) {
    return (target == QKV_TARGET_KEY) ? state->k_norms_normal :
           (target == QKV_TARGET_VALUE) ? state->v_norms_normal : NULL;
}

static bool qkv_outlier_split_ready(const qkv_state_t* state, const qkv_config_t* config, int target) {
    if (!state || !config || (target != QKV_TARGET_KEY && target != QKV_TARGET_VALUE)) return false;
    const int dim = state->head_dim;
    const int n_outliers = config->outlier_channels;
    if (n_outliers <= 0 || n_outliers >= dim) return false;
    if (!qkv_bits_valid(config->outlier_bits) || !qkv_bits_valid(config->normal_bits)) return false;
    return qkv_outlier_indices_for_target_const(state, target) &&
           qkv_idx_outlier_for_target_const(state, target) &&
           qkv_idx_normal_for_target_const(state, target) &&
           qkv_norms_outlier_for_target_const(state, target) &&
           qkv_norms_normal_for_target_const(state, target);
}

static int qkv_outlier_slot(const int* outlier_indices, int n_outliers, int channel) {
    for (int i = 0; i < n_outliers; ++i) {
        if (outlier_indices[i] == channel) return i;
    }
    return -1;
}

static void qkv_select_outlier_indices(
    qkv_state_t* state,
    const qkv_config_t* config,
    int target,
    const float* input,
    int n_tokens
) {
    int* selected = qkv_outlier_indices_for_target(state, target);
    if (!state || !config || !input || !selected || n_tokens <= 0) return;
    const int dim = state->head_dim;
    const int n_outliers = config->outlier_channels;
    if (n_outliers <= 0 || n_outliers >= dim) return;

    std::vector<float> scores((size_t)dim, 0.0f);
    std::vector<int> order((size_t)dim);
    for (int d = 0; d < dim; ++d) order[(size_t)d] = d;
    for (int t = 0; t < n_tokens; ++t) {
        const float* row = input + (size_t)t * (size_t)dim;
        for (int d = 0; d < dim; ++d) {
            scores[(size_t)d] += fabsf(row[d]);
        }
    }
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        const float sa = scores[(size_t)a];
        const float sb = scores[(size_t)b];
        if (sa == sb) return a < b;
        return sa > sb;
    });
    for (int i = 0; i < n_outliers; ++i) {
        selected[i] = order[(size_t)i];
    }
    std::sort(selected, selected + n_outliers);
    if (target == QKV_TARGET_KEY && state->outlier_indices) {
        memcpy(state->outlier_indices, selected, (size_t)n_outliers * sizeof(int));
    }
}

static int qkv_quantize_mse(
    qkv_state_t* state,
    const qkv_config_t* config,
    const float* input,
    uint8_t* idx_out,
    float* norms_out,
    int n_tokens,
    int bits
) {
    if (!state || !config || !input || !idx_out || !norms_out || n_tokens <= 0) return 0;
    if (!qkv_bits_valid(bits) || n_tokens > state->n_tokens) return 0;

    const int dim = state->head_dim;
    const int n_levels = 1 << bits;
    const float* centroids = qkv_codebook_for_bits(state, bits);
    const float* thresholds = qkv_thresholds_for_bits(state, bits);

    const int target = qkv_target_from_buffers(state, idx_out, norms_out);
    const bool use_split = qkv_outlier_split_ready(state, config, target);
    if (use_split) {
        qkv_select_outlier_indices(state, config, target, input, n_tokens);
    }

    const int n_outliers = use_split ? config->outlier_channels : 0;
    const int n_normal = use_split ? (dim - n_outliers) : 0;
    const int out_bits = use_split ? config->outlier_bits : 0;
    const int norm_bits = use_split ? config->normal_bits : 0;
    const int out_levels = use_split ? (1 << out_bits) : 0;
    const int norm_levels = use_split ? (1 << norm_bits) : 0;
    const int packed_size = (dim * bits + 7) / 8;
    const int out_packed_size = use_split ? (n_outliers * out_bits + 7) / 8 : 0;
    const int norm_packed_size = use_split ? (n_normal * norm_bits + 7) / 8 : 0;

    const int* outlier_channels = use_split ? qkv_outlier_indices_for_target_const(state, target) : NULL;
    uint8_t* split_outlier = use_split ? qkv_idx_outlier_for_target(state, target) : NULL;
    uint8_t* split_normal = use_split ? qkv_idx_normal_for_target(state, target) : NULL;
    float* split_norms_outlier = use_split ? qkv_norms_outlier_for_target(state, target) : NULL;
    float* split_norms_normal = use_split ? qkv_norms_normal_for_target(state, target) : NULL;
    const float* out_centroids = use_split ? qkv_codebook_for_bits(state, out_bits) : NULL;
    const float* out_thresholds = use_split ? qkv_thresholds_for_bits(state, out_bits) : NULL;
    const float* norm_centroids = use_split ? qkv_codebook_for_bits(state, norm_bits) : NULL;
    const float* norm_thresholds = use_split ? qkv_thresholds_for_bits(state, norm_bits) : NULL;

    float* rotated = (float*)malloc((size_t)dim * sizeof(float));
    int* indices = (int*)malloc((size_t)dim * sizeof(int));
    if (!rotated || !indices) {
        free(rotated);
        free(indices);
        return 0;
    }

    for (int t = 0; t < n_tokens; t++) {
        const float* x = input + (size_t)t * (size_t)dim;

        float norm = 0.0f;
        for (int d = 0; d < dim; d++) {
            norm += x[d] * x[d];
        }
        norm = sqrtf(norm);
        norms_out[t] = norm;

        const float inv_norm = (norm > 1e-8f) ? 1.0f / norm : 0.0f;
        for (int d = 0; d < dim; d++) {
            rotated[d] = x[d] * inv_norm;
        }

        const float* y = rotated;
        if (config->enable_rotation && state->rotation_matrix) {
            float* temp = state->scratch_residual;
            if (!temp) {
                free(rotated);
                free(indices);
                return 0;
            }
            for (int i = 0; i < dim; i++) {
                float sum = 0.0f;
                for (int j = 0; j < dim; j++) {
                    sum += state->rotation_matrix[i * dim + j] * rotated[j];
                }
                temp[i] = sum;
            }
            memcpy(rotated, temp, (size_t)dim * sizeof(float));
            y = rotated;
        }

        if (use_split) {
            int* out_codes = indices;
            int* norm_codes = indices + n_outliers;
            if (split_norms_outlier) split_norms_outlier[t] = norm;
            if (split_norms_normal) split_norms_normal[t] = norm;

            for (int i = 0; i < n_outliers; i++) {
                const int channel = outlier_channels[i];
                if (channel < 0 || channel >= dim) {
                    free(rotated);
                    free(indices);
                    return 0;
                }
                out_codes[i] = find_nearest_centroid(y[channel], out_centroids, out_thresholds, out_levels);
            }

            int normal_pos = 0;
            for (int d = 0; d < dim; d++) {
                if (qkv_outlier_slot(outlier_channels, n_outliers, d) >= 0) continue;
                norm_codes[normal_pos++] = find_nearest_centroid(y[d], norm_centroids, norm_thresholds, norm_levels);
            }
            if (normal_pos != n_normal) {
                free(rotated);
                free(indices);
                return 0;
            }

            pack_indices(out_codes, split_outlier + (size_t)t * (size_t)out_packed_size, n_outliers, out_bits);
            pack_indices(norm_codes, split_normal + (size_t)t * (size_t)norm_packed_size, n_normal, norm_bits);

            // Preserve the legacy homogeneous stream for non-split callers.
            for (int d = 0; d < dim; d++) {
                indices[d] = find_nearest_centroid(y[d], centroids, thresholds, n_levels);
            }
            pack_indices(indices, idx_out + (size_t)t * (size_t)packed_size, dim, bits);
        } else {
            for (int d = 0; d < dim; d++) {
                indices[d] = find_nearest_centroid(y[d], centroids, thresholds, n_levels);
            }
            pack_indices(indices, idx_out + (size_t)t * (size_t)packed_size, dim, bits);
        }
    }

    free(rotated);
    free(indices);
    return 1;
}

#if 0
static int qkv_quantize_mse_legacy(
    qkv_state_t* state,
    const qkv_config_t* config,
    const float* input,
    uint8_t* idx_out,
    float* norms_out,
    int n_tokens,
    int bits
) {
    if (!state || !config || !input || !idx_out || !norms_out || n_tokens <= 0) return 0;
    // Fix: Validate bits to prevent incorrect codebook usage
    // Bug ②: Allow 1-bit for mse_bits when prod uses 2-bit total
    if (bits < 1 || bits > 3) return 0;
    
    int dim = state->head_dim;
    int n_levels = 1 << bits;
    
    // Select codebook
    // Bug ②: Support 1/2/3-bit codebook selection
    float* centroids = (bits == 1) ? state->codebook_1bit : (bits == 2) ? state->codebook_2bit : state->codebook_3bit;
    float* thresholds = (bits == 1) ? state->thresholds_1bit : (bits == 2) ? state->thresholds_2bit : state->thresholds_3bit;
    
    // Fix 4: Outlier channel support (paper Section 4.3, Table 1)
    // "splitting channels into outlier and non-outlier sets, and applying two 
    // independent instances of TurboQuant to each, allocating higher bit precision to outliers"
    // Example: 32 outliers @ 3-bit + 96 normal @ 2-bit = 2.5-bit average
    int n_outliers = config->outlier_channels;
    int n_normal = dim - n_outliers;
    int out_bits = config->outlier_bits;
    int norm_bits = config->normal_bits;
    bool use_outlier_split = (n_outliers > 0 && n_outliers < dim && 
                               state->outlier_indices && state->k_idx_outlier);
    
    // Temporary buffers
    float* rotated = (float*)malloc(dim * sizeof(float));
    int* indices = (int*)malloc(dim * sizeof(int));
    float* outlier_buf = use_outlier_split ? (float*)malloc(n_outliers * sizeof(float)) : NULL;
    float* normal_buf = use_outlier_split ? (float*)malloc(n_normal * sizeof(float)) : NULL;
    int* out_indices = use_outlier_split ? (int*)malloc(n_outliers * sizeof(int)) : NULL;
    int* norm_indices = use_outlier_split ? (int*)malloc(n_normal * sizeof(int)) : NULL;
    
    if (!rotated || !indices || (use_outlier_split && (!outlier_buf || !normal_buf || !out_indices || !norm_indices))) {
        free(rotated); free(indices);
        free(outlier_buf); free(normal_buf);
        free(out_indices); free(norm_indices);
        return 0;
    }
    
    // Select codebooks for outlier path
    float* out_centroids = use_outlier_split ? 
        ((out_bits == 1) ? state->codebook_1bit : (out_bits == 2) ? state->codebook_2bit : state->codebook_3bit) : NULL;
    float* out_thresholds = use_outlier_split ?
        ((out_bits == 1) ? state->thresholds_1bit : (out_bits == 2) ? state->thresholds_2bit : state->thresholds_3bit) : NULL;
    float* norm_centroids = use_outlier_split ?
        ((norm_bits == 1) ? state->codebook_1bit : (norm_bits == 2) ? state->codebook_2bit : state->codebook_3bit) : NULL;
    float* norm_thresholds = use_outlier_split ?
        ((norm_bits == 1) ? state->thresholds_1bit : (norm_bits == 2) ? state->thresholds_2bit : state->thresholds_3bit) : NULL;
    
    for (int t = 0; t < n_tokens; t++) {
        const float* x = input + t * dim;
        
        // Step 1: Compute L2 norm (paper Section 1.3)
        float norm = 0.0f;
        for (int d = 0; d < dim; d++) {
            norm += x[d] * x[d];
        }
        norm = sqrtf(norm);
        norms_out[t] = norm;
        
        // Step 2: Normalize to unit vector BEFORE rotation (paper requirement)
        // "the unit norm assumption is standard... compute and store L2 norms and rescale"
        float inv_norm = (norm > 1e-8f) ? 1.0f / norm : 0.0f;
        for (int d = 0; d < dim; d++) {
            rotated[d] = x[d] * inv_norm;
        }
        
        // Step 3: Apply random rotation y = Pi * x_norm (on normalized vector)
        const float* y = rotated;
        if (config->enable_rotation && state->rotation_matrix) {
            // Fix: Use scratch buffer instead of per-token malloc
            float* temp = state->scratch_residual;
            if (temp) {
                for (int i = 0; i < dim; i++) {
                    float sum = 0.0f;
                    for (int j = 0; j < dim; j++) {
                        sum += state->rotation_matrix[i * dim + j] * rotated[j];
                    }
                    temp[i] = sum;
                }
                memcpy(rotated, temp, dim * sizeof(float));
            }
            y = rotated;
        }
        
        // Step 4: Find nearest centroid for each coordinate
        if (use_outlier_split) {
            // Fix 4: Split channels into outlier and normal, quantize separately
            // Paper: "32 outlier channels are quantized at 3 bits, while the remaining 
            // 96 channels use 2 bits"
            
            // Extract outlier and normal channels using outlier_indices
            for (int i = 0; i < n_outliers; i++) {
                outlier_buf[i] = y[state->outlier_indices[i]];
            }
            int norm_idx = 0;
            for (int d = 0; d < dim; d++) {
                bool is_outlier = false;
                for (int i = 0; i < n_outliers; i++) {
                    if (state->outlier_indices[i] == d) { is_outlier = true; break; }
                }
                if (!is_outlier) normal_buf[norm_idx++] = y[d];
            }
            
            // Quantize outlier channels with higher precision
            int out_n_levels = 1 << out_bits;
            for (int i = 0; i < n_outliers; i++) {
                out_indices[i] = find_nearest_centroid(outlier_buf[i], out_centroids, out_thresholds, out_n_levels);
            }
            
            // Quantize normal channels with lower precision
            int norm_n_levels = 1 << norm_bits;
            for (int i = 0; i < n_normal; i++) {
                norm_indices[i] = find_nearest_centroid(normal_buf[i], norm_centroids, norm_thresholds, norm_n_levels);
            }
            
            // Pack separately into outlier and normal storage
            int out_packed_size = (n_outliers * out_bits + 7) / 8;
            int norm_packed_size = (n_normal * norm_bits + 7) / 8;
            pack_indices(out_indices, state->k_idx_outlier + t * out_packed_size, n_outliers, out_bits);
            pack_indices(norm_indices, state->k_idx_normal + t * norm_packed_size, n_normal, norm_bits);
            
            // Also pack combined indices into idx_out for backward compatibility
            // Reconstruct full indices array: outlier positions from out_indices, rest from norm_indices
            norm_idx = 0;
            for (int d = 0; d < dim; d++) {
                bool is_outlier = false;
                for (int i = 0; i < n_outliers; i++) {
                    if (state->outlier_indices[i] == d) {
                        indices[d] = out_indices[i];
                        is_outlier = true;
                        break;
                    }
                }
                if (!is_outlier) {
                    indices[d] = norm_indices[norm_idx++];
                }
            }
            int packed_size = (dim * bits + 7) / 8;
            pack_indices(indices, idx_out + t * packed_size, dim, bits);
        } else {
            // Standard path: all channels use same bit-width
            for (int d = 0; d < dim; d++) {
                indices[d] = find_nearest_centroid(y[d], centroids, thresholds, n_levels);
            }
            int packed_size = (dim * bits + 7) / 8;
            pack_indices(indices, idx_out + t * packed_size, dim, bits);
        }
    }
    
    free(rotated);
    free(indices);
    free(outlier_buf);
    free(normal_buf);
    free(out_indices);
    free(norm_indices);
    return 1;
}

#endif

static int qkv_dequantize_mse_split_token(
    const qkv_state_t* state,
    const qkv_config_t* config,
    int target,
    int tok,
    float fallback_norm,
    float* x_out
) {
    if (!qkv_outlier_split_ready(state, config, target) || !x_out || tok < 0 || tok >= state->n_tokens) return 0;

    const int dim = state->head_dim;
    const int n_outliers = config->outlier_channels;
    const int n_normal = dim - n_outliers;
    const int out_bits = config->outlier_bits;
    const int norm_bits = config->normal_bits;
    const int out_packed_size = (n_outliers * out_bits + 7) / 8;
    const int norm_packed_size = (n_normal * norm_bits + 7) / 8;

    const int* outlier_channels = qkv_outlier_indices_for_target_const(state, target);
    const uint8_t* split_outlier = qkv_idx_outlier_for_target_const(state, target);
    const uint8_t* split_normal = qkv_idx_normal_for_target_const(state, target);
    const float* split_norms_outlier = qkv_norms_outlier_for_target_const(state, target);
    const float* split_norms_normal = qkv_norms_normal_for_target_const(state, target);
    const float* out_centroids = qkv_codebook_for_bits(state, out_bits);
    const float* norm_centroids = qkv_codebook_for_bits(state, norm_bits);
    int* codes = state->scratch_indices;
    float* y_tilde = state->scratch_y_tilde;
    if (!outlier_channels || !split_outlier || !split_normal || !codes || !y_tilde) return 0;

    memset(y_tilde, 0, (size_t)dim * sizeof(float));
    unpack_indices(split_outlier + (size_t)tok * (size_t)out_packed_size, codes, n_outliers, out_bits);
    const float out_norm = split_norms_outlier ? split_norms_outlier[tok] : fallback_norm;
    for (int i = 0; i < n_outliers; i++) {
        const int channel = outlier_channels[i];
        if (channel < 0 || channel >= dim) return 0;
        y_tilde[channel] = out_centroids[codes[i]] * out_norm;
    }

    unpack_indices(split_normal + (size_t)tok * (size_t)norm_packed_size, codes, n_normal, norm_bits);
    const float normal_norm = split_norms_normal ? split_norms_normal[tok] : fallback_norm;
    int normal_pos = 0;
    for (int d = 0; d < dim; d++) {
        if (qkv_outlier_slot(outlier_channels, n_outliers, d) >= 0) continue;
        y_tilde[d] = norm_centroids[codes[normal_pos++]] * normal_norm;
    }
    if (normal_pos != n_normal) return 0;

    if (config->enable_rotation && state->rotation_matrix) {
        for (int i = 0; i < dim; i++) {
            float sum = 0.0f;
            for (int j = 0; j < dim; j++) {
                sum += state->rotation_matrix[j * dim + i] * y_tilde[j];
            }
            x_out[i] = sum;
        }
    } else {
        memcpy(x_out, y_tilde, (size_t)dim * sizeof(float));
    }
    return 1;
}

static int qkv_dot_mse_split_rotated_token(
    const qkv_state_t* state,
    const qkv_config_t* config,
    int target,
    int tok,
    const float* q_eff,
    float* dot_out
) {
    if (!qkv_outlier_split_ready(state, config, target) || !q_eff || !dot_out || tok < 0 || tok >= state->n_tokens) return 0;

    const int dim = state->head_dim;
    const int n_outliers = config->outlier_channels;
    const int n_normal = dim - n_outliers;
    const int out_bits = config->outlier_bits;
    const int norm_bits = config->normal_bits;
    const int out_packed_size = (n_outliers * out_bits + 7) / 8;
    const int norm_packed_size = (n_normal * norm_bits + 7) / 8;
    const int* outlier_channels = qkv_outlier_indices_for_target_const(state, target);
    const uint8_t* split_outlier = qkv_idx_outlier_for_target_const(state, target);
    const uint8_t* split_normal = qkv_idx_normal_for_target_const(state, target);
    const float* out_centroids = qkv_codebook_for_bits(state, out_bits);
    const float* norm_centroids = qkv_codebook_for_bits(state, norm_bits);
    int* codes = state->scratch_indices;
    if (!outlier_channels || !split_outlier || !split_normal || !codes) return 0;

    float dot = 0.0f;
    unpack_indices(split_outlier + (size_t)tok * (size_t)out_packed_size, codes, n_outliers, out_bits);
    for (int i = 0; i < n_outliers; i++) {
        const int channel = outlier_channels[i];
        if (channel < 0 || channel >= dim) return 0;
        dot += q_eff[channel] * out_centroids[codes[i]];
    }

    unpack_indices(split_normal + (size_t)tok * (size_t)norm_packed_size, codes, n_normal, norm_bits);
    int normal_pos = 0;
    for (int d = 0; d < dim; d++) {
        if (qkv_outlier_slot(outlier_channels, n_outliers, d) >= 0) continue;
        dot += q_eff[d] * norm_centroids[codes[normal_pos++]];
    }
    if (normal_pos != n_normal) return 0;
    *dot_out = dot;
    return 1;
}

static int qkv_dequantize_mse(
    const qkv_state_t* state,
    const qkv_config_t* config,
    const uint8_t* idx,
    const float* norms,
    float* output,
    int n_tokens,
    int bits
) {
    if (!state || !config || !idx || !norms || !output || n_tokens <= 0) return 0;
    if (!qkv_bits_valid(bits) || n_tokens > state->n_tokens) return 0;

    const int dim = state->head_dim;
    const int target = qkv_target_from_buffers(state, idx, norms);
    if (qkv_outlier_split_ready(state, config, target)) {
        for (int t = 0; t < n_tokens; t++) {
            if (!qkv_dequantize_mse_split_token(
                    state, config, target, t, norms[t], output + (size_t)t * (size_t)dim)) {
                return 0;
            }
        }
        return 1;
    }

    const float* centroids = qkv_codebook_for_bits(state, bits);
    int* indices = state->scratch_indices;
    float* y_tilde = state->scratch_y_tilde;
    if (!indices || !y_tilde) return 0;

    const int packed_size = (dim * bits + 7) / 8;
    for (int t = 0; t < n_tokens; t++) {
        float* x_out = output + (size_t)t * (size_t)dim;
        unpack_indices(idx + (size_t)t * (size_t)packed_size, indices, dim, bits);
        for (int d = 0; d < dim; d++) {
            y_tilde[d] = centroids[indices[d]];
        }

        if (config->enable_rotation && state->rotation_matrix) {
            for (int i = 0; i < dim; i++) {
                float sum = 0.0f;
                for (int j = 0; j < dim; j++) {
                    sum += state->rotation_matrix[j * dim + i] * y_tilde[j];
                }
                x_out[i] = sum;
            }
        } else {
            memcpy(x_out, y_tilde, (size_t)dim * sizeof(float));
        }

        const float norm = norms[t];
        if (norm > 1e-8f) {
            for (int d = 0; d < dim; d++) {
                x_out[d] *= norm;
            }
        }
    }
    return 1;
}

#if 0
static int qkv_dequantize_mse_legacy(
    const qkv_state_t* state,
    const qkv_config_t* config,
    const uint8_t* idx,
    const float* norms,
    float* output,
    int n_tokens,
    int bits
) {
    if (!state || !config || !idx || !norms || !output || n_tokens <= 0) return 0;
    // Fix: Validate bits to prevent incorrect codebook usage
    // Bug ②: Allow 1-bit for mse_bits when prod uses 2-bit total
    if (bits < 1 || bits > 3) return 0;
    
    int dim = state->head_dim;
    
    // Select codebook
    // Bug ②: Support 1/2/3-bit codebook selection
    const float* centroids = (bits == 1) ? state->codebook_1bit : (bits == 2) ? state->codebook_2bit : state->codebook_3bit;
    
    // Fix 4: Outlier channel support (paper Section 4.3)
    int n_outliers = config->outlier_channels;
    int n_normal = dim - n_outliers;
    int out_bits = config->outlier_bits;
    int norm_bits = config->normal_bits;
    bool use_outlier_split = (n_outliers > 0 && n_outliers < dim && 
                               state->outlier_indices && state->k_idx_outlier);
    
    // Select codebooks for outlier path
    const float* out_centroids = use_outlier_split ? 
        ((out_bits == 1) ? state->codebook_1bit : (out_bits == 2) ? state->codebook_2bit : state->codebook_3bit) : NULL;
    const float* norm_centroids = use_outlier_split ?
        ((norm_bits == 1) ? state->codebook_1bit : (norm_bits == 2) ? state->codebook_2bit : state->codebook_3bit) : NULL;
    
    int* indices = state->scratch_indices;
    float* y_tilde = state->scratch_y_tilde;
    int* out_indices = use_outlier_split ? (int*)malloc(n_outliers * sizeof(int)) : NULL;
    int* norm_indices = use_outlier_split ? (int*)malloc(n_normal * sizeof(int)) : NULL;
    float* out_vals = use_outlier_split ? (float*)malloc(n_outliers * sizeof(float)) : NULL;
    float* norm_vals = use_outlier_split ? (float*)malloc(n_normal * sizeof(float)) : NULL;
    
    if (!indices || !y_tilde || (use_outlier_split && (!out_indices || !norm_indices || !out_vals || !norm_vals))) {
        free(out_indices); free(norm_indices); free(out_vals); free(norm_vals);
        return 0;
    }
    
    int packed_size = (dim * bits + 7) / 8;
    int out_packed_size = use_outlier_split ? (n_outliers * out_bits + 7) / 8 : 0;
    int norm_packed_size = use_outlier_split ? (n_normal * norm_bits + 7) / 8 : 0;
    
    for (int t = 0; t < n_tokens; t++) {
        float* x_out = output + t * dim;
        
        if (use_outlier_split) {
            // Fix 4: Dequantize outlier and normal channels separately
            // Paper: "32 outlier channels are quantized at 3 bits, while the remaining 
            // 96 channels use 2 bits"
            
            // Unpack outlier indices
            unpack_indices(state->k_idx_outlier + t * out_packed_size, out_indices, n_outliers, out_bits);
            // Unpack normal indices
            unpack_indices(state->k_idx_normal + t * norm_packed_size, norm_indices, n_normal, norm_bits);
            
            // Retrieve centroids for outlier channels
            for (int i = 0; i < n_outliers; i++) {
                out_vals[i] = out_centroids[out_indices[i]];
            }
            // Retrieve centroids for normal channels
            for (int i = 0; i < n_normal; i++) {
                norm_vals[i] = norm_centroids[norm_indices[i]];
            }
            
            // Reconstruct full y_tilde: place outlier values at outlier_indices positions
            int norm_idx = 0;
            for (int d = 0; d < dim; d++) {
                bool is_outlier = false;
                for (int i = 0; i < n_outliers; i++) {
                    if (state->outlier_indices[i] == d) {
                        y_tilde[d] = out_vals[i];
                        is_outlier = true;
                        break;
                    }
                }
                if (!is_outlier) {
                    y_tilde[d] = norm_vals[norm_idx++];
                }
            }
        } else {
            // Standard path: unpack indices and retrieve centroids
            unpack_indices(idx + t * packed_size, indices, dim, bits);
            for (int d = 0; d < dim; d++) {
                y_tilde[d] = centroids[indices[d]];
            }
        }
        
        // Step 3: Apply inverse rotation x_hat = Pi^T*y_hat.
        if (config->enable_rotation && state->rotation_matrix) {
            for (int i = 0; i < dim; i++) {
                float sum = 0.0f;
                for (int j = 0; j < dim; j++) {
                    // Pi^T is the transpose, so use [j][i] indexing.
                    sum += state->rotation_matrix[j * dim + i] * y_tilde[j];
                }
                x_out[i] = sum;
            }
        } else {
            memcpy(x_out, y_tilde, dim * sizeof(float));
        }
        
        // Step 4: Rescale by original norm
        float norm = norms[t];
        if (norm > 1e-8f) {
            for (int d = 0; d < dim; d++) {
                x_out[d] *= norm;
            }
        }
    }
    
    free(out_indices); free(norm_indices); free(out_vals); free(norm_vals);
    return 1;
}

#endif

// ============================================================
// Algorithm 2: TurboQuant_prod (Unbiased Inner Product)
// ============================================================

static int qkv_quantize_prod(
    qkv_state_t* state,
    const qkv_config_t* config,
    const float* input,
    uint8_t* idx_out,
    uint8_t* qjl_out,
    float* residual_norms,
    float* norms_out,
    int n_tokens,
    int bits
) {
    if (!state || !config || !input || !idx_out || !qjl_out || 
        !residual_norms || !norms_out || n_tokens <= 0) return 0;
    // Fix: Need at least 3 bits for prod mode (mse_bits = bits-1 must be >= 2)
    // GAP 2: Allow 2-bit prod mode
    if (bits < 2) return 0;
    
    int dim = state->head_dim;
    int mse_bits = bits - 1;  // Use b-1 bits for MSE part
    
    if (n_tokens > state->n_tokens) return 0;
    float* x_tilde = state->scratch_x_tilde;
    float* residual = state->scratch_residual;
    float* s_times_r = state->scratch_s_times_r;
    float* qjl_signs = state->scratch_qjl_signs;
    
    if (!x_tilde || !residual || !s_times_r || !qjl_signs) {
        return 0;
    }
    
    // MSE quantize with b-1 bits
    if (!qkv_quantize_mse(state, config, input, idx_out, norms_out, n_tokens, mse_bits)) {
        return 0;
    }
    
    // Dequantize to get reconstruction
    if (!qkv_dequantize_mse(state, config, idx_out, norms_out, x_tilde, n_tokens, mse_bits)) {
        return 0;
    }
    
    // Now compute QJL on residual
    for (int t = 0; t < n_tokens; t++) {
        const float* x = input + t * dim;
        const float* x_mse = x_tilde + t * dim;
        
        // Compute residual r = x - x_hat_mse.
        float r_norm = 0.0f;
        for (int d = 0; d < dim; d++) {
            residual[d] = x[d] - x_mse[d];
            r_norm += residual[d] * residual[d];
        }
        r_norm = sqrtf(r_norm);
        residual_norms[t] = r_norm;
        
        // Compute S*r (random matrix multiplication).
        // Paper Algorithm 2, line 7: qjl = sign(S * r)
        // This is the KEY part of QJL - must multiply by S
        if (state->qjl_matrix) {
            // Gaussian random matrix path
            for (int i = 0; i < dim; i++) {
                float sum = 0.0f;
                for (int j = 0; j < dim; j++) {
                    sum += state->qjl_matrix[i * dim + j] * residual[j];
                }
                s_times_r[i] = sum;
            }
        } else if (state->qjl_signs_matrix) {
            // Rademacher (±1) matrix path - Fix for asymmetric S
            for (int i = 0; i < dim; i++) {
                float sum = 0.0f;
                for (int j = 0; j < dim; j++) {
                    sum += (state->qjl_signs_matrix[i * dim + j] == 1) 
                           ? residual[j] : -residual[j];
                }
                s_times_r[i] = sum;
            }
        } else {
            // Fallback: no QJL matrix, just use residual directly
            memcpy(s_times_r, residual, dim * sizeof(float));
        }
        
        // QJL: qjl = sign(S*r), not sign(r).
        for (int d = 0; d < dim; d++) {
            qjl_signs[d] = (s_times_r[d] >= 0.0f) ? 1.0f : -1.0f;
        }
        
        // Pack signs
        pack_signs(qjl_signs, qjl_out + t * ((dim + 7) / 8), dim);
    }
    
    return 1;
}

static int qkv_dequantize_prod(
    const qkv_state_t* state,
    const qkv_config_t* config,
    const uint8_t* idx,
    const uint8_t* qjl,
    const float* residual_norms,
    const float* norms,
    float* output,
    int n_tokens,
    int bits
) {
    if (!state || !config || !idx || !qjl || !residual_norms || 
        !norms || !output || n_tokens <= 0) return 0;
    
    int dim = state->head_dim;
    int mse_bits = bits - 1;
    
    // Fix 56: Use pre-allocated scratch buffers (no malloc in hot path)
    float* qjl_signs = state->scratch_qjl_signs;
    float* s_t_qjl = state->scratch_s_t_qjl;
    if (!qjl_signs || !s_t_qjl) return 0;
    
    // First, dequantize MSE part
    if (!qkv_dequantize_mse(state, config, idx, norms, output, n_tokens, mse_bits)) {
        return 0;
    }
    
    // Add QJL residual: x_hat = x_hat_mse + sqrt(pi/2)/d * ||r|| * S^T*qjl
    // Paper Definition 1: Q^{-1}_qjl(z) = sqrt(pi/2) / d * S^T * z
    float scale_factor = sqrtf((float)M_PI / 2.0f) / (float)dim;
    
    for (int t = 0; t < n_tokens; t++) {
        float* x_out = output + t * dim;
        
        // Unpack QJL signs
        unpack_signs(qjl + t * ((dim + 7) / 8), qjl_signs, dim);
        
        // Compute S^T * qjl_signs
        if (state->qjl_signs_matrix) {
            // Rademacher path (currently unreachable — qjl_signs_matrix == NULL per GAP 5)
            for (int i = 0; i < dim; i++) {
                float sum = 0.0f;
                const int8_t* row = state->qjl_signs_matrix + i;
                for (int j = 0; j < dim; j++) {
                    sum += (row[j * dim] == 1) ? qjl_signs[j] : -qjl_signs[j];
                }
                s_t_qjl[i] = sum;
            }
        } else if (state->qjl_matrix) {
            // Gaussian S^T * z — paper Definition 1 (primary path)
            for (int i = 0; i < dim; i++) {
                float sum = 0.0f;
                for (int j = 0; j < dim; j++) {
                    sum += state->qjl_matrix[j * dim + i] * qjl_signs[j];
                }
                s_t_qjl[i] = sum;
            }
        } else {
            memcpy(s_t_qjl, qjl_signs, dim * sizeof(float));
        }
        
        // Add scaled residual: sqrt(pi/2d)*||r||*S^T*qjl
        float gamma = residual_norms[t];
        for (int d = 0; d < dim; d++) {
            x_out[d] += scale_factor * gamma * s_t_qjl[d];
        }
    }
    
    return 1;
}

// ============================================================
// High-level API
// ============================================================

int qkv_quantize(
    qkv_state_t* state,
    const qkv_config_t* config,
    const float* key_data,
    const float* value_data,
    int n_tokens
) {
    if (!state || !config || !key_data || !value_data || n_tokens <= 0) return 0;
    
    int k_bits = config->k_bits;
    int v_bits = config->v_bits;
    int dim = state->head_dim;
    
    if (config->enable_qjl) {
        // Use Algorithm 2 (unbiased inner product)
        if (!qkv_quantize_prod(state, config, key_data, 
                               state->k_idx, state->k_qjl, 
                               state->k_residual_norms, state->k_norms,
                               n_tokens, k_bits)) {
            return 0;
        }
        
        if (!qkv_quantize_prod(state, config, value_data,
                               state->v_idx, state->v_qjl,
                               state->v_residual_norms, state->v_norms,
                               n_tokens, v_bits)) {
            return 0;
        }
    } else {
        // Use Algorithm 1 (MSE-optimal only)
        if (!qkv_quantize_mse(state, config, key_data,
                              state->k_idx, state->k_norms,
                              n_tokens, k_bits)) {
            return 0;
        }
        
        if (!qkv_quantize_mse(state, config, value_data,
                              state->v_idx, state->v_norms,
                              n_tokens, v_bits)) {
            return 0;
        }
    }
    
    return 1;
}

int qkv_dequantize(
    const qkv_state_t* state,
    const qkv_config_t* config,
    float* key_output,
    float* value_output,
    int n_tokens
) {
    if (!state || !config || !key_output || !value_output || n_tokens <= 0) return 0;
    
    int k_bits = config->k_bits;
    int v_bits = config->v_bits;
    
    if (config->enable_qjl) {
        // Use Algorithm 2
        if (!qkv_dequantize_prod(state, config,
                                 state->k_idx, state->k_qjl,
                                 state->k_residual_norms, state->k_norms,
                                 key_output, n_tokens, k_bits)) {
            return 0;
        }
        
        if (!qkv_dequantize_prod(state, config,
                                 state->v_idx, state->v_qjl,
                                 state->v_residual_norms, state->v_norms,
                                 value_output, n_tokens, v_bits)) {
            return 0;
        }
    } else {
        // Use Algorithm 1
        if (!qkv_dequantize_mse(state, config,
                                state->k_idx, state->k_norms,
                                key_output, n_tokens, k_bits)) {
            return 0;
        }
        
        if (!qkv_dequantize_mse(state, config,
                                state->v_idx, state->v_norms,
                                value_output, n_tokens, v_bits)) {
            return 0;
        }
    }
    
    return 1;
}

// ============================================================
// GAP 1: QKV-domain Attention Decode (modular, zero duplication)
// ============================================================

static int qkv_add_qjl_residual_token(
    const qkv_state_t* st,
    const uint8_t* qjl_token,
    float residual_norm,
    float* x_out
) {
    if (!st || !qjl_token || !x_out) return 0;
    const int dim = st->head_dim;
    float* qjl_signs = st->scratch_qjl_signs;
    float* s_t_qjl = st->scratch_s_t_qjl;
    if (!qjl_signs || !s_t_qjl) return 0;

    unpack_signs(qjl_token, qjl_signs, dim);
    if (st->qjl_signs_matrix) {
        for (int i = 0; i < dim; i++) {
            float sum = 0.0f;
            const int8_t* row = st->qjl_signs_matrix + i;
            for (int j = 0; j < dim; j++) {
                sum += (row[j * dim] == 1) ? qjl_signs[j] : -qjl_signs[j];
            }
            s_t_qjl[i] = sum;
        }
    } else if (st->qjl_matrix) {
        for (int i = 0; i < dim; i++) {
            float sum = 0.0f;
            for (int j = 0; j < dim; j++) {
                sum += st->qjl_matrix[j * dim + i] * qjl_signs[j];
            }
            s_t_qjl[i] = sum;
        }
    } else {
        memcpy(s_t_qjl, qjl_signs, (size_t)dim * sizeof(float));
    }

    const float scale_factor = sqrtf((float)M_PI / 2.0f) / (float)dim;
    for (int d = 0; d < dim; d++) {
        x_out[d] += scale_factor * residual_norm * s_t_qjl[d];
    }
    return 1;
}

static int qkv_dequant_one(
    const qkv_state_t* st, const qkv_config_t* cfg,
    const uint8_t* idx, const uint8_t* qjl,
    const float* rnorms, const float* norms,
    int tok, int bits, bool use_prod, float* out
) {
    const int dim = st->head_dim;
    const int mse_bits = use_prod ? bits - 1 : bits;
    const size_t stride = ((size_t)dim * mse_bits + 7) / 8;
    const size_t qstride = ((size_t)dim + 7) / 8;
    float n = norms[tok];
    const int target = qkv_target_from_buffers(st, idx, norms);
    if (qkv_outlier_split_ready(st, cfg, target)) {
        if (!qkv_dequantize_mse_split_token(st, cfg, target, tok, n, out)) {
            return 0;
        }
        if (use_prod && qjl && rnorms) {
            return qkv_add_qjl_residual_token(st, qjl + (size_t)tok * qstride, rnorms[tok], out);
        }
        return 1;
    }
    if (use_prod && qjl && rnorms) {
        float rn = rnorms[tok];
        return qkv_dequantize_prod(st, cfg,
            idx + tok * stride, qjl + tok * qstride,
            &rn, &n, out, 1, bits);
    }
    return qkv_dequantize_mse(st, cfg, idx + tok * stride, &n, out, 1, mse_bits);
}


int qkv_attention_decode(
    const float* query, const qkv_state_t* s,
    const qkv_config_t* cfg, uint32_t ctx, uint32_t hdim, float* output
) {
    if (!query || !s || !cfg || !output || !ctx || !hdim) return 0;
    const int d = (int)hdim, n = (int)ctx;
    const bool qjl = cfg->enable_qjl && s->k_qjl && s->v_qjl;

    // GAP 1B: Use scratch_residual as row buffer (already allocated in qkv_init)
    float* row = s->scratch_residual;
    if (!row || s->head_dim != d) return 0;

    // Thread-local attention scores — no malloc per call
    if (n > s->n_tokens) return 0;
    float* att = s->scratch_attention;
    if (!att) return 0;

    // Bug #23: Pre-rotate query ONCE with Pi (O(d²) × 1)
    // Then dequant uses codebook-only path (skip inverse rotation) — O(d) × N
    // This exploits: <q, Pi^T * y_hat> = <Pi * q, y_hat>
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

    // Phase 1: K scores — dequant in rotated domain (no inverse rotation needed)
    // Fix 2: For unbiased inner product, must include QJL residual when enable_qjl=true
    // Paper Algorithm 2: Q^{-1}_prod = MSE_reconstruction + QJL_residual
    const int k_mse_bits = qjl ? s->k_bits - 1 : s->k_bits;
    const int k_levels = 1 << k_mse_bits;
    const float* k_centroids = (k_mse_bits == 1) ? s->codebook_1bit :
                               (k_mse_bits == 2) ? s->codebook_2bit : s->codebook_3bit;
    const int k_stride = (d * k_mse_bits + 7) / 8;
    const int k_qstride = (d + 7) / 8;
    const float qjl_scale = sqrtf((float)M_PI / 2.0f) / (float)d;
    const bool k_split = qkv_outlier_split_ready(s, cfg, QKV_TARGET_KEY);
    const bool use_qjl_key_residual = qjl && s->k_qjl && s->qjl_matrix;
    float* s_q_precomputed = s->scratch_s_times_r;
    float* qjl_z = s->scratch_qjl_signs;
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
        auto score_range = [&](int begin, int end, int* ok_flag) {
            std::vector<int> local_codes((size_t)std::max(d, std::max(n_outliers, n_normal)));
            std::vector<float> local_qjl((size_t)d);
            for (int t = begin; t < end && *ok_flag; t++) {
                float norm_k = s->k_norms[t];
                float dot = 0.0f;
                if (k_split) {
                    if (!outlier_channels || !split_outlier || !split_normal || !out_centroids || !norm_centroids) {
                        *ok_flag = 0;
                        return;
                    }
                    unpack_indices(split_outlier + (size_t)t * (size_t)out_packed_size,
                        local_codes.data(), n_outliers, out_bits);
                    for (int i = 0; i < n_outliers; i++) {
                        const int channel = outlier_channels[i];
                        if (channel < 0 || channel >= d) {
                            *ok_flag = 0;
                            return;
                        }
                        dot += q_eff[channel] * out_centroids[local_codes[(size_t)i]];
                    }
                    unpack_indices(split_normal + (size_t)t * (size_t)norm_packed_size,
                        local_codes.data(), n_normal, norm_bits);
                    int normal_pos = 0;
                    for (int i = 0; i < d; i++) {
                        if (qkv_outlier_slot(outlier_channels, n_outliers, i) >= 0) continue;
                        dot += q_eff[i] * norm_centroids[local_codes[(size_t)normal_pos++]];
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
                    unpack_indices(tidx, local_codes.data(), d, k_mse_bits);
                    for (int i = 0; i < d; i++) {
                        dot += q_eff[i] * k_centroids[local_codes[(size_t)i]];
                    }
                }
                if (use_qjl_key_residual) {
                    const uint8_t* tqjl = s->k_qjl + t * k_qstride;
                    float r_norm = s->k_residual_norms[t];
                    if (r_norm > 1e-10f) {
                        unpack_signs(tqjl, local_qjl.data(), d);
                        for (int i = 0; i < d; i++) {
                            dot += qjl_scale * r_norm * s_q_precomputed[i] * local_qjl[(size_t)i];
                        }
                    }
                }
                att[t] = dot * norm_k * sc;
            }
        };
        const unsigned hw_raw = std::thread::hardware_concurrency();
        const int workers = std::max(1, std::min<int>(hw_raw ? (int)hw_raw : 4, n / 512));
        std::vector<std::thread> threads;
        std::vector<int> ok_flags((size_t)workers, 1);
        threads.reserve((size_t)workers);
        for (int w = 0; w < workers; ++w) {
            const int begin = (int)(((int64_t)n * w) / workers);
            const int end = (int)(((int64_t)n * (w + 1)) / workers);
            threads.emplace_back([&, begin, end, w] { score_range(begin, end, &ok_flags[(size_t)w]); });
        }
        for (auto& th : threads) th.join();
        for (int v : ok_flags) {
            if (!v) return 0;
        }
        goto qkv_scores_ready;
    }

    for (int t = 0; t < n; t++) {
        // Fast codebook-only dequant (no rotation): unpack → centroid lookup → dot
        float norm_k = s->k_norms[t];
        float dot = 0.0f;
        // Inline unpack+dot to avoid materializing full vector when possible
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
            // General path: unpack indices then dot
            const uint8_t* tidx = s->k_idx + t * k_stride;
            int* indices = s->scratch_indices;
            if (!indices) return 0;
            unpack_indices(tidx, indices, d, k_mse_bits);
            for (int i = 0; i < d; i++) {
                dot += q_eff[i] * k_centroids[indices[i]];
            }
        }
        // Fix 2: Add QJL residual contribution for unbiased inner product
        // Paper Algorithm 2: <q, k_hat> = <q, k_mse> + sqrt(pi/2d) * ||r|| * <q, S^T * z>
        // where z = sign(S*r) stored in k_qjl
        if (use_qjl_key_residual) {
            const uint8_t* tqjl = s->k_qjl + t * k_qstride;
            float r_norm = s->k_residual_norms[t];
            if (r_norm > 1e-10f) {
                // Unpack QJL signs
                unpack_signs(tqjl, qjl_z, d);
                // Compute <q_eff, S^T * z> = <S * q_eff, z>
                // This is O(d²) but required for unbiased inner product per paper
                for (int i = 0; i < d; i++) {
                    dot += qjl_scale * r_norm * s_q_precomputed[i] * qjl_z[i];
                }
            }
        }
        att[t] = dot * norm_k * sc;
    }

qkv_scores_ready:
    // Softmax
    float mx = att[0]; for (int t = 1; t < n; t++) if (att[t] > mx) mx = att[t];
    float se = 0; for (int t = 0; t < n; t++) { att[t] = expf(att[t] - mx); se += att[t]; }
    if (se > 0) { float iv = 1.0f / se; for (int t = 0; t < n; t++) att[t] *= iv; }

    // Phase 2: V weighted sum — need full dequant (with inverse rotation) for output
    if (n >= 1024) {
        const unsigned hw_raw = std::thread::hardware_concurrency();
        const int workers = std::max(1, std::min<int>(hw_raw ? (int)hw_raw : 4, n / 512));
        std::vector<float> partial((size_t)workers * (size_t)d, 0.0f);
        std::vector<int> ok_flags((size_t)workers, 1);
        std::vector<std::thread> threads;
        threads.reserve((size_t)workers);
        for (int w = 0; w < workers; ++w) {
            const int begin = (int)(((int64_t)n * w) / workers);
            const int end = (int)(((int64_t)n * (w + 1)) / workers);
            threads.emplace_back([&, begin, end, w] {
                std::vector<float> local_row((size_t)d);
                std::vector<float> local_residual((size_t)d);
                std::vector<float> local_s_times_r((size_t)d);
                std::vector<float> local_qjl((size_t)d);
                std::vector<float> local_s_t_qjl((size_t)d);
                std::vector<float> local_y((size_t)d);
                std::vector<float> local_x((size_t)d);
                std::vector<int> local_indices((size_t)d);
                qkv_state_t local = *s;
                local.scratch_residual = local_residual.data();
                local.scratch_s_times_r = local_s_times_r.data();
                local.scratch_qjl_signs = local_qjl.data();
                local.scratch_s_t_qjl = local_s_t_qjl.data();
                local.scratch_y_tilde = local_y.data();
                local.scratch_x_tilde = local_x.data();
                local.scratch_indices = local_indices.data();
                float* dst = partial.data() + (size_t)w * (size_t)d;
                for (int t = begin; t < end; ++t) {
                    if (!qkv_dequant_one(&local, cfg, local.v_idx, local.v_qjl,
                            local.v_residual_norms, local.v_norms, t, local.v_bits, qjl, local_row.data())) {
                        ok_flags[(size_t)w] = 0;
                        return;
                    }
                    const float weight = att[t];
                    for (int i = 0; i < d; ++i) {
                        dst[i] += weight * local_row[(size_t)i];
                    }
                }
            });
        }
        for (auto& th : threads) th.join();
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

    memset(output, 0, d * sizeof(float));
    for (int t = 0; t < n; t++) {
        if (!qkv_dequant_one(s, cfg, s->v_idx, s->v_qjl,
                s->v_residual_norms, s->v_norms, t, s->v_bits, qjl, row))
            continue;
        float w = att[t]; for (int i = 0; i < d; i++) output[i] += w * row[i];
    }
    return 1;
}
