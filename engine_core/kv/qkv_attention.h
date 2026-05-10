#pragma once

#include "kv_qkv.h"

// Attention Decode - Main algorithm

// Attention decode operating directly on quantized KV cache
// Internal flow: per-token dequantize K row → dot(Q, K_hat) → softmax →
//                dequantize V row → weighted sum
int qkv_attention_decode_impl(
    const float* query,
    const qkv_state_t* kv_state,
    const qkv_config_t* kv_config,
    uint32_t context_tokens,
    uint32_t head_dim,
    float* output
);
