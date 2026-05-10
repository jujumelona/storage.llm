#pragma once

#include "kv_qkv.h"

// State Management - Initialization and Cleanup

// Initialize QKV state with all buffers and matrices
int qkv_state_init(
    qkv_state_t* state,
    const qkv_config_t* config,
    int n_tokens
);

// Free all QKV state resources
void qkv_state_free(qkv_state_t* state);
