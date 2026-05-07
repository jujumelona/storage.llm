#include "qkv_helpers.h"
#include <stddef.h>

bool qkv_bits_valid(int bits) {
    return (bits >= 1 && bits <= 8) || bits == 16 || bits == 32;
}

bool qkv_bits_codebook(int bits) {
    return bits >= 1 && bits <= 8;
}

bool qkv_bits_raw(int bits) {
    return bits == 16 || bits == 32;
}

int qkv_outlier_bits_for_target(const qkv_config_t* cfg, int target) {
    if (!cfg) return 0;
    if (target == QKV_TARGET_KEY && cfg->key_outlier_bits > 0) return cfg->key_outlier_bits;
    if (target == QKV_TARGET_VALUE && cfg->value_outlier_bits > 0) return cfg->value_outlier_bits;
    return cfg->outlier_bits;
}

int qkv_normal_bits_for_target(const qkv_config_t* cfg, int target) {
    if (!cfg) return 0;
    if (target == QKV_TARGET_KEY && cfg->key_normal_bits > 0) return cfg->key_normal_bits;
    if (target == QKV_TARGET_VALUE && cfg->value_normal_bits > 0) return cfg->value_normal_bits;
    return cfg->normal_bits;
}

const float* qkv_codebook_for_bits(const qkv_state_t* state, int bits) {
    if (!state) return NULL;
    switch (bits) {
    case 1: return state->codebook_1bit;
    case 2: return state->codebook_2bit;
    case 3: return state->codebook_3bit;
    case 4: return state->codebook_4bit;
    case 5: return state->codebook_5bit;
    case 6: return state->codebook_6bit;
    case 7: return state->codebook_7bit;
    case 8: return state->codebook_8bit;
    default: return NULL;
    }
}

const float* qkv_thresholds_for_bits(const qkv_state_t* state, int bits) {
    if (!state) return NULL;
    switch (bits) {
    case 1: return state->thresholds_1bit;
    case 2: return state->thresholds_2bit;
    case 3: return state->thresholds_3bit;
    case 4: return state->thresholds_4bit;
    case 5: return state->thresholds_5bit;
    case 6: return state->thresholds_6bit;
    case 7: return state->thresholds_7bit;
    case 8: return state->thresholds_8bit;
    default: return NULL;
    }
}

int qkv_target_from_buffers(const qkv_state_t* state, const uint8_t* idx, const float* norms) {
    if (!state) return 0;
    if (!idx) {
        if (norms == state->k_norms) return QKV_TARGET_KEY;
        if (norms == state->v_norms) return QKV_TARGET_VALUE;
        return 0;
    }
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
