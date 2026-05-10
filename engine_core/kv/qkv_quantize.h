#pragma once

#include "kv_qkv.h"

// Quantization Operations

// Quantize a single vector with state (internal helper)
int qkv_quantize_vector_with_state(
    const qkv_state_t* state,
    const qkv_config_t* config,
    const float* input,
    uint8_t* output,
    float* norm_out,
    int dim,
    int bits
);
