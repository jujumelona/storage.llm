#include "qkv_helpers.h"
#include "qkv_codebook.h"
#include <stddef.h>
#include <stdint.h>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>


static uint64_t qkv_codebook_cache_key_local(int dim, int bits, bool thresholds, unsigned distribution) {
    uint64_t x = (uint64_t)(uint32_t)dim * 0x9e3779b97f4a7c15ull;
    x ^= (uint64_t)(uint32_t)bits * 0xbf58476d1ce4e5b9ull;
    x ^= ((uint64_t)distribution << 48);
    if (thresholds) x ^= 0x8000000000000000ull;
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebull;
    x ^= x >> 31;
    return x;
}

static const float* qkv_codebook_or_thresholds_for_dim_local(
    int bits, int dim, unsigned distribution, bool thresholds) {
    if (!qkv_bits_codebook(bits) || dim <= 0 || dim > 16384) return NULL;
    static std::mutex mutex;
    static std::unordered_map<uint64_t, std::shared_ptr<std::vector<float>>> cache;
    const uint64_t key = qkv_codebook_cache_key_local(dim, bits, thresholds, distribution);
    std::lock_guard<std::mutex> lock(mutex);
    auto found = cache.find(key);
    if (found != cache.end()) return found->second->data();

    const int levels = 1 << bits;
    auto cb = std::make_shared<std::vector<float>>((size_t)levels);
    auto th = std::make_shared<std::vector<float>>((size_t)levels + 1u);
    qkv_compute_lloyd_max_codebook_ex(cb->data(), th->data(), bits, dim, distribution);
    const uint64_t cb_key = qkv_codebook_cache_key_local(dim, bits, false, distribution);
    const uint64_t th_key = qkv_codebook_cache_key_local(dim, bits, true, distribution);
    cache[cb_key] = cb;
    cache[th_key] = th;
    return thresholds ? th->data() : cb->data();
}

const float* qkv_codebook_for_bits_dim(int bits, int dim, unsigned distribution) {
    return qkv_codebook_or_thresholds_for_dim_local(bits, dim, distribution, false);
}

const float* qkv_thresholds_for_bits_dim(int bits, int dim, unsigned distribution) {
    return qkv_codebook_or_thresholds_for_dim_local(bits, dim, distribution, true);
}

size_t qkv_split_qjl_outlier_bytes(const qkv_config_t* cfg) {
    if (!cfg || cfg->outlier_channels <= 0) return 0;
    return ((size_t)cfg->outlier_channels + 7u) / 8u;
}

size_t qkv_split_qjl_normal_bytes(const qkv_config_t* cfg) {
    if (!cfg || cfg->head_dim <= 0 || cfg->outlier_channels <= 0 ||
        cfg->outlier_channels >= cfg->head_dim) {
        return 0;
    }
    return ((size_t)(cfg->head_dim - cfg->outlier_channels) + 7u) / 8u;
}

size_t qkv_qjl_token_bytes(const qkv_state_t* state) {
    if (!state || state->head_dim <= 0) return 0;
    if (state->qjl_token_bytes > 0) return (size_t)state->qjl_token_bytes;
    return ((size_t)state->head_dim + 7u) / 8u;
}

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
