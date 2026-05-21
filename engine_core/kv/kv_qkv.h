#pragma once

// QKV: Near-Optimal KV Cache Quantization
// Based on: "TurboQuant: Towards Near-Optimal KV Cache Quantization"
// Paper: 2504.19874v1
//
// Key Components:
// 1. Random Rotation (Pi) - Transforms coordinates to follow Beta distribution
// 2. Lloyd-Max Codebook - Optimal scalar quantizer for Beta distribution
// 3. QJL (Quantized Johnson-Lindenstrauss) - Unbiased inner product via residual
//
// Mode: QKV is the normal StorageLLM KV cache contract. The plain float KV
// path is retained only as a debug/fallback path for legacy callers.

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
#include <atomic>
// Fix: std::atomic<bool> must be outside extern "C" block (C++ type)
extern std::atomic<bool> g_qkv_mode_enabled;

static inline void qkv_set_mode(bool enabled) {
    g_qkv_mode_enabled.store(enabled, std::memory_order_release);
}

static inline bool qkv_is_enabled(void) {
    return g_qkv_mode_enabled.load(std::memory_order_acquire);
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================
// QKV Mode Flag (C API)
// ============================================================

#ifndef __cplusplus
// C API: non-atomic bool for C compatibility
extern bool g_qkv_mode_enabled;
void qkv_set_mode(bool enabled);
bool qkv_is_enabled(void);
#endif

// ============================================================
// QKV Configuration
// ============================================================

#ifndef QKV_ROTATION_BACKEND_GAUSSIAN_QR_ORTHOGONAL
#define QKV_ROTATION_BACKEND_GAUSSIAN_QR_ORTHOGONAL 0u
#endif
#ifndef QKV_ROTATION_BACKEND_HADAMARD_SIGN_FAST
#define QKV_ROTATION_BACKEND_HADAMARD_SIGN_FAST 1u
#endif
#ifndef QKV_CODEBOOK_DISTRIBUTION_EXACT_BETA
#define QKV_CODEBOOK_DISTRIBUTION_EXACT_BETA 0u
#endif
#ifndef QKV_CODEBOOK_DISTRIBUTION_GAUSSIAN_APPROX
#define QKV_CODEBOOK_DISTRIBUTION_GAUSSIAN_APPROX 1u
#endif

typedef struct {
    int k_bits;               // Key bits (default: 3)
    int v_bits;               // Value bits (default: 2)
    int head_dim;             // Dimension per head (for rotation matrix)
    bool enable_qjl;          // Enable QJL residual for unbiased inner product
    bool enable_rotation;     // Enable random rotation (recommended: true)
    uint64_t rotation_seed;   // Seed for random rotation matrix
    uint64_t qjl_seed;        // Seed for QJL random matrix
    uint32_t rotation_backend; // 0 = paper Gaussian QR, 1 = Hadamard+sign fast path
    uint32_t codebook_distribution; // 0 = exact Beta, 1 = Gaussian approximation
    uint64_t policy_hash;     // Hash of the QKV_POLICY contract, 0 = derive from config
    // Fix 4: Outlier channel support (paper Table 1, Section 4.3)
    // The paper labels this as a 2.5-bit setup, but 32 @ 3-bit + 96 @ 2-bit
    // over 128 channels computes to 2.25 bits/channel. Keep budgets arithmetic-based.
    int outlier_channels;     // Number of outlier channels (0 = disabled, paper uses 32)
    int outlier_bits;         // Bits for outlier channels (paper uses 3)
    int normal_bits;          // Bits for normal channels (paper uses 2)
    int key_outlier_bits;     // Optional key-specific outlier-channel bits
    int key_normal_bits;      // Optional key-specific normal-channel bits
    int value_outlier_bits;   // Optional value-specific outlier-channel bits
    int value_normal_bits;    // Optional value-specific normal-channel bits
    // Problem 11 Fix: Engine IO thread count to prevent CPU over-subscription
    uint32_t engine_io_thread_count;  // disk+pinned+gpu workers from engine
    const int* outlier_channel_indices; // Bug 4: Allow custom outlier indices instead of hardcoded 0..n
    uint32_t group_size;        // Offload GGUF qkv_cache_schema.group_size
    uint32_t page_size_tokens;  // Offload GGUF qkv_cache_schema.page_size_tokens
    uint32_t sink_tokens;       // Attention sink tokens kept hot by residency policy
    bool plain_kv_persistent_storage; // Must stay false for offload-native GGUF
    float attention_score_scale; // Optional model attention score scale; 0 = 1/sqrt(head_dim)
    float attention_logit_softcap; // Optional model attention-score softcap before softmax
} qkv_config_t;

// Default configuration
static inline qkv_config_t qkv_config_default(int head_dim) {
    qkv_config_t cfg;
    cfg.k_bits = 3;
    cfg.v_bits = 2;
    cfg.head_dim = head_dim;
    cfg.enable_qjl = true;
    cfg.enable_rotation = true;
    cfg.rotation_seed = 42;
    cfg.qjl_seed = 43;
    cfg.rotation_backend = QKV_ROTATION_BACKEND_GAUSSIAN_QR_ORTHOGONAL;
    cfg.codebook_distribution = QKV_CODEBOOK_DISTRIBUTION_EXACT_BETA;
    cfg.policy_hash = 0;
    // Fix 4: Default to no outlier separation (backward compatible)
    cfg.outlier_channels = 0;
    cfg.outlier_bits = 3;
    cfg.normal_bits = 2;
    cfg.key_outlier_bits = cfg.outlier_bits;
    cfg.key_normal_bits = cfg.normal_bits;
    cfg.value_outlier_bits = cfg.outlier_bits;
    cfg.value_normal_bits = cfg.normal_bits;
    // Problem 11 Fix: Default to 0 (no adjustment)
    cfg.engine_io_thread_count = 0;
    cfg.outlier_channel_indices = nullptr;
    cfg.group_size = 64;
    cfg.page_size_tokens = 16;
    cfg.sink_tokens = 4;
    cfg.plain_kv_persistent_storage = false;
    cfg.attention_score_scale = 0.0f;
    cfg.attention_logit_softcap = 0.0f;
    return cfg;
}

static inline float qkv_effective_bits_for_values(int head_dim, int outlier_channels, int outlier_bits, int normal_bits) {
    if (head_dim <= 0 || outlier_channels <= 0) {
        return (float)normal_bits;
    }
    if (outlier_channels > head_dim) {
        outlier_channels = head_dim;
    }
    const int normal_channels = head_dim - outlier_channels;
    return (float)(outlier_channels * outlier_bits + normal_channels * normal_bits) / (float)head_dim;
}

// Helper: compute exact configured average bits across K and V split policies.
static inline float qkv_effective_bits(const qkv_config_t* cfg) {
    if (!cfg) {
        return 3.0f;
    }
    if (cfg->outlier_channels <= 0) {
        return 0.5f * ((float)cfg->k_bits + (float)cfg->v_bits);
    }
    const int key_out = cfg->key_outlier_bits > 0 ? cfg->key_outlier_bits : cfg->outlier_bits;
    const int key_norm = cfg->key_normal_bits > 0 ? cfg->key_normal_bits : cfg->normal_bits;
    const int value_out = cfg->value_outlier_bits > 0 ? cfg->value_outlier_bits : cfg->outlier_bits;
    const int value_norm = cfg->value_normal_bits > 0 ? cfg->value_normal_bits : cfg->normal_bits;
    const float key_bits = qkv_effective_bits_for_values(cfg->head_dim, cfg->outlier_channels, key_out, key_norm);
    const float value_bits = qkv_effective_bits_for_values(cfg->head_dim, cfg->outlier_channels, value_out, value_norm);
    return 0.5f * (key_bits + value_bits);
}

// ============================================================
// QKV State
// ============================================================

typedef struct {
    // Packed quantized index streams.
    uint8_t* k_idx;
    uint8_t* v_idx;
    float* k_norms;
    float* v_norms;
    float* k_sink;
    float* v_sink;
    uint32_t sink_tokens;

    // QJL residual streams. For split TurboQuant, k_qjl/v_qjl store
    // two independent sign streams per token: outlier signs followed by
    // normal signs, each byte-aligned. qjl_token_bytes is the per-token
    // stride for either layout.
    uint8_t* k_qjl;
    uint8_t* v_qjl;
    uint32_t qjl_token_bytes;
    float* k_residual_norms;
    float* v_residual_norms;

    // Shared transforms and codebooks.
    float* rotation_matrix;   // [head_dim, head_dim] random orthogonal
    float* rotation_signs;    // [head_dim] sign vector for Hadamard structure (Issue 9)
    float* qjl_matrix;        // [head_dim, head_dim] Gaussian S, S_ij ~ N(0,1) (paper Lemma 4)
    // Paper Section 4.3: outlier and non-outlier channel groups are two
    // independent TurboQuant instances, not slices through one head_dim
    // transform. These matrices are generated with independent seeds and
    // dimensions n_outlier and n_normal.
    float* rotation_matrix_outlier;
    float* rotation_matrix_normal;
    float* rotation_signs_outlier;
    float* rotation_signs_normal;
    float* qjl_matrix_outlier;
    float* qjl_matrix_normal;
    int8_t* qjl_signs_matrix; // Reserved (NULL — Rademacher disabled per paper)
    float* codebook_1bit;     // Bug ②: 2 levels for prod-mode v_bits=2
    float* thresholds_1bit;
    float* codebook_2bit;
    float* thresholds_2bit;
    float* codebook_3bit;
    float* thresholds_3bit;
    float* codebook_4bit;     // 4-bit support: 16 levels
    float* thresholds_4bit;
    float* codebook_5bit;
    float* thresholds_5bit;
    float* codebook_6bit;
    float* thresholds_6bit;
    float* codebook_7bit;
    float* thresholds_7bit;
    float* codebook_8bit;
    float* thresholds_8bit;
    int owns_rotation_matrix;
    int owns_qjl_matrix;
    int owns_codebooks;

    // Fix 4: Outlier channel indices (paper Section 4.3)
    // Outliers are channels with highest magnitude across calibration data
    int* outlier_indices;     // [outlier_channels] indices of outlier channels
    int* k_outlier_indices;   // [outlier_channels] key-specific outlier channels
    int* v_outlier_indices;   // [outlier_channels] value-specific outlier channels
    uint8_t* k_is_outlier;    // [head_dim] O(1) boolean lookup
    uint8_t* v_is_outlier;    // [head_dim] O(1) boolean lookup
    uint8_t* k_idx_outlier;   // Packed indices for outlier channels
    uint8_t* v_idx_outlier;
    uint8_t* k_idx_normal;    // Packed indices for normal channels
    uint8_t* v_idx_normal;
    float* k_norms_outlier;   // Per-channel-group norms
    float* k_norms_normal;
    float* v_norms_outlier;
    float* v_norms_normal;
    float* k_residual_norms_outlier;
    float* k_residual_norms_normal;
    float* v_residual_norms_outlier;
    float* v_residual_norms_normal;

    // Fix 56: Pre-allocated scratch buffers (eliminate per-token malloc)
    float* scratch_qjl_signs;  // [head_dim]
    float* scratch_s_t_qjl;    // [head_dim]
    float* scratch_residual;   // [head_dim]
    float* scratch_s_times_r;  // [head_dim]
    float* scratch_y_tilde;    // [head_dim] for dequantize
    float* scratch_x_tilde;    // [head_dim] single-token scratch for dequant/quantize
    float* scratch_attention;  // [n_tokens] for attention scores
    float* scratch_rotated_q;  // [head_dim] for rotated query
    int* scratch_indices;      // [head_dim] for dequantize
    int* work_codes_buf;       // [computed_workers, max(head_dim, outlier dims)]
    float* work_qjl_buf;       // [computed_workers, head_dim]
    int work_buf_stride;
    int work_buf_workers;

    // Shape
    int n_tokens;
    int head_dim;
    int k_bits;
    int v_bits;

    void* thread_pool;
    int computed_workers; // Bug 2: Cache thread pool size to avoid OS calls
} qkv_state_t;

// ============================================================
// API Functions
// ============================================================

// Initialize QKV state
int qkv_init(
    qkv_state_t* state,
    const qkv_config_t* config,
    int n_tokens
);

// Free QKV state
void qkv_free(qkv_state_t* state);

// Quantize using QKV (with rotation + Lloyd-Max)
int qkv_quantize(
    qkv_state_t* state,
    const qkv_config_t* config,
    const float* key_data,     // [n_tokens, head_dim]
    const float* value_data,   // [n_tokens, head_dim]
    int n_tokens
);

// Quantize one appended token into an existing QKV state. This follows the
// same split/QJL/raw-bit contract as qkv_quantize(), without requiring the
// caller to rebuild the whole token range.
int qkv_quantize_token(
    qkv_state_t* state,
    const qkv_config_t* config,
    const float* key,          // [head_dim]
    const float* value,        // [head_dim]
    int token_idx
);

// Dequantize using QKV
int qkv_dequantize(
    const qkv_state_t* state,
    const qkv_config_t* config,
    float* key_output,         // [n_tokens, head_dim]
    float* value_output,       // [n_tokens, head_dim]
    int n_tokens
);

// GAP 1: Attention decode operating directly on quantized KV cache
// Avoids full dequantize → attention → discard cycle that wastes memory
// Internal flow: per-token dequantize K row → dot(Q, K_hat) → softmax → dequantize V row → weighted sum
int qkv_attention_decode(
    const float* query,            // [head_dim]  — single query vector
    const qkv_state_t* kv_state,   // quantized KV cache
    const qkv_config_t* kv_config,
    uint32_t context_tokens,       // number of KV tokens
    uint32_t head_dim,
    float* output                  // [head_dim]  — attention output
);

int qkv_attention_decode_exact_current(
    const float* query,
    const qkv_state_t* kv_state,
    const qkv_config_t* kv_config,
    uint32_t context_tokens,
    uint32_t head_dim,
    const float* current_key,
    const float* current_value,
    float* output
);

#ifdef __cplusplus
}
#endif
