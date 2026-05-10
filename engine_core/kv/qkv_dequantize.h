#pragma once

#include "kv_qkv.h"

// Dequantization Operations

// Dequantize a single token (internal helper)
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
);

// Dot product with MSE split rotated token (internal helper)
int qkv_dot_mse_split_rotated_token(
    const qkv_state_t* s,
    const qkv_config_t* cfg,
    int target,
    int token_idx,
    const float* q_rotated,
    float* out_dot
);
