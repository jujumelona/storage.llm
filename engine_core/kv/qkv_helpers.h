#pragma once

#include "kv_qkv.h"

// Helper Functions - Internal utilities

// Target constants
#define QKV_TARGET_KEY 1
#define QKV_TARGET_VALUE 2

// Check if bits are valid (1-4)
bool qkv_bits_valid(int bits);

// Get codebook for given bit-width
const float* qkv_codebook_for_bits(const qkv_state_t* state, int bits);

// Get thresholds for given bit-width
const float* qkv_thresholds_for_bits(const qkv_state_t* state, int bits);

// Determine target (KEY/VALUE) from buffer pointers
int qkv_target_from_buffers(const qkv_state_t* state, const uint8_t* idx, const float* norms);

// Get outlier indices for target
int* qkv_outlier_indices_for_target(qkv_state_t* state, int target);
const int* qkv_outlier_indices_for_target_const(const qkv_state_t* state, int target);

// Get outlier/normal packed indices for target
uint8_t* qkv_idx_outlier_for_target(qkv_state_t* state, int target);
uint8_t* qkv_idx_normal_for_target(qkv_state_t* state, int target);
const uint8_t* qkv_idx_outlier_for_target_const(const qkv_state_t* state, int target);
const uint8_t* qkv_idx_normal_for_target_const(const qkv_state_t* state, int target);

// Get is_outlier flags for target
const uint8_t* qkv_is_outlier_for_target_const(const qkv_state_t* state, int target);

// Check if outlier split is ready for target
bool qkv_outlier_split_ready(const qkv_state_t* s, const qkv_config_t* cfg, int target);
