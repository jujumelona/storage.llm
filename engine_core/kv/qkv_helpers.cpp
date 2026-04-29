#include "qkv_helpers.h"
#include <stddef.h>

bool qkv_bits_valid(int bits) {
    return bits >= 1 && bits <= 4;
}

const float* qkv_codebook_for_bits(const qkv_state_t* state, int bits) {
    return (bits == 1) ? state->codebook_1bit :
           (bits == 2) ? state->codebook_2bit :
           (bits == 3) ? state->codebook_3bit : state->codebook_4bit;
}

const float* qkv_thresholds_for_bits(const qkv_state_t* state, int bits) {
    return (bits == 1) ? state->thresholds_1bit :
           (bits == 2) ? state->thresholds_2bit :
           (bits == 3) ? state->thresholds_3bit : state->thresholds_4bit;
}

int qkv_target_from_buffers(const qkv_state_t* state, const uint8_t* idx, const float* norms) {
    if (!state || !idx) return 0;
    if (idx == state->k_idx && (!norms || norms == state->k_norms)) return QKV_TARGET_KEY;
    if (idx == state->v_idx && (!norms || norms == state->v_norms)) return QKV_TARGET_VALUE;
    return 0;
}

int* qkv_outlier_indices_for_target(qkv_state_t* state, int target) {
    if (!state) return NULL;
    if (target == QKV_TARGET_KEY) return state->k_outlier_indices ? state->k_outlier_indices : state->outlier_indices;
    if (target == QKV_TARGET_VALUE) return state->v_outlier_indices ? state->v_outlier_indices : state->outlier_indices;
    return NULL;
}

const int* qkv_outlier_indices_for_target_const(const qkv_state_t* state, int target) {
    if (!state) return NULL;
    if (target == QKV_TARGET_KEY) return state->k_outlier_indices ? state->k_outlier_indices : state->outlier_indices;
    if (target == QKV_TARGET_VALUE) return state->v_outlier_indices ? state->v_outlier_indices : state->outlier_indices;
    return NULL;
}

uint8_t* qkv_idx_outlier_for_target(qkv_state_t* state, int target) {
    return (target == QKV_TARGET_KEY) ? state->k_idx_outlier :
           (target == QKV_TARGET_VALUE) ? state->v_idx_outlier : NULL;
}

uint8_t* qkv_idx_normal_for_target(qkv_state_t* state, int target) {
    return (target == QKV_TARGET_KEY) ? state->k_idx_normal :
           (target == QKV_TARGET_VALUE) ? state->v_idx_normal : NULL;
}

const uint8_t* qkv_idx_outlier_for_target_const(const qkv_state_t* state, int target) {
    return (target == QKV_TARGET_KEY) ? state->k_idx_outlier :
           (target == QKV_TARGET_VALUE) ? state->v_idx_outlier : NULL;
}

const uint8_t* qkv_idx_normal_for_target_const(const qkv_state_t* state, int target) {
    return (target == QKV_TARGET_KEY) ? state->k_idx_normal :
           (target == QKV_TARGET_VALUE) ? state->v_idx_normal : NULL;
}

const uint8_t* qkv_is_outlier_for_target_const(const qkv_state_t* state, int target) {
    return (target == QKV_TARGET_KEY) ? state->k_is_outlier :
           (target == QKV_TARGET_VALUE) ? state->v_is_outlier : NULL;
}

bool qkv_outlier_split_ready(const qkv_state_t* s, const qkv_config_t* cfg, int target) {
    if (!s || !cfg || cfg->outlier_channels <= 0) return false;
    const uint8_t* idx_out = qkv_idx_outlier_for_target_const(s, target);
    const uint8_t* idx_norm = qkv_idx_normal_for_target_const(s, target);
    return idx_out && idx_norm;
}
