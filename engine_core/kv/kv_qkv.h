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

typedef struct {
    int k_bits;               // Key bits (default: 3)
    int v_bits;               // Value bits (default: 2)
    int head_dim;             // Dimension per head (for rotation matrix)
    bool enable_qjl;          // Enable QJL residual for unbiased inner product
    bool enable_rotation;     // Enable random rotation (recommended: true)
    uint64_t rotation_seed;   // Seed for random rotation matrix
    uint64_t qjl_seed;        // Seed for QJL random matrix
    // Fix 4: Outlier channel support (paper Table 1, Section 4.3)
    // Paper uses 32 outlier channels @ 3-bit + 96 normal channels @ 2-bit = 2.5-bit average
    int outlier_channels;     // Number of outlier channels (0 = disabled, paper uses 32)
    int outlier_bits;         // Bits for outlier channels (paper uses 3)
    int normal_bits;          // Bits for normal channels (paper uses 2)
    // Problem 11 Fix: Engine IO thread count to prevent CPU over-subscription
    uint32_t engine_io_thread_count;  // disk+pinned+gpu workers from engine
    const int* outlier_channel_indices; // Bug 4: Allow custom outlier indices instead of hardcoded 0..n
    uint32_t group_size;        // Offload GGUF qkv_cache_schema.group_size
    uint32_t page_size_tokens;  // Offload GGUF qkv_cache_schema.page_size_tokens
    uint32_t sink_tokens;       // Attention sink tokens kept hot by residency policy
    bool plain_kv_persistent_storage; // Must stay false for offload-native GGUF
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
    // Fix 4: Default to no outlier separation (backward compatible)
    cfg.outlier_channels = 0;
    cfg.outlier_bits = 3;
    cfg.normal_bits = 2;
    // Problem 11 Fix: Default to 0 (no adjustment)
    cfg.engine_io_thread_count = 0;
    cfg.outlier_channel_indices = nullptr;
    cfg.group_size = 64;
    cfg.page_size_tokens = 16;
    cfg.sink_tokens = 4;
    cfg.plain_kv_persistent_storage = false;
    return cfg;
}

// Helper: compute exact configured average bits with outlier channels.
static inline float qkv_effective_bits(const qkv_config_t* cfg) {
    if (!cfg || cfg->outlier_channels <= 0) {
        return (float)(cfg ? cfg->k_bits : 3);
    }
    int normal_channels = cfg->head_dim - cfg->outlier_channels;
    return (float)(cfg->outlier_channels * cfg->outlier_bits + normal_channels * cfg->normal_bits) / (float)cfg->head_dim;
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

    // QJL residual streams.
    uint8_t* k_qjl;
    uint8_t* v_qjl;
    float* k_residual_norms;
    float* v_residual_norms;

    // Shared transforms and codebooks.
    float* rotation_matrix;   // [head_dim, head_dim] random orthogonal
    float* qjl_matrix;        // [head_dim, head_dim] Gaussian S, S_ij ~ N(0,1) (paper Lemma 4)
    int8_t* qjl_signs_matrix; // Reserved (NULL — Rademacher disabled per paper)
    float* codebook_1bit;     // Bug ②: 2 levels for prod-mode v_bits=2
    float* thresholds_1bit;
    float* codebook_2bit;
    float* thresholds_2bit;
    float* codebook_3bit;
    float* thresholds_3bit;
    float* codebook_4bit;     // 4-bit support: 16 levels
    float* thresholds_4bit;

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

#ifdef __cplusplus
}
#endif
