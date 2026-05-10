static void add_tier_usage(moe_pc_engine* engine, moe_tier_t tier, uint64_t bytes);
static void subtract_tier_usage(moe_pc_engine* engine, moe_tier_t tier, uint64_t bytes);
static uint64_t tier_used(const moe_pc_engine* engine, moe_tier_t tier);
static uint64_t tier_budget(const moe_pc_engine* engine, moe_tier_t tier);
static uint64_t expert_triplet_bytes(const moe_pc_engine* engine, int layer, int expert);
static int tensor_weight_format(const moe_tensor_record& rec);
static void touch_expert_heat(moe_pc_engine* engine, moe_block_state* state, double boost);

// Error handling (no dependencies)
#include "generation_error.cpp.inc"

// Basic helpers (no dependencies)
#include "generation_silu.cpp.inc"
#include "generation_simd_helpers.cpp.inc"
#include "generation_bf16_convert.cpp.inc"

// Constants (no dependencies)
#include "generation_mla_constants.cpp.inc"

// Raw operations (depends on: silu, simd_helpers)
#include "generation_raw_ops.cpp.inc"

// KV cache (depends on: bf16_convert, mla_constants)
#include "generation_kv_cache.cpp.inc"

// Scratch (depends on: kv_cache, mla_constants)
#include "generation_scratch.cpp.inc"

// QKV-only invariant guards (depends on scratch and qkv runtime state)
#include "generation/qkv_only_runtime.cpp.inc"

// Router (depends on: raw_ops, scratch session telemetry)
#include "generation_router.cpp.inc"

// RoPE (depends on: mla_constants)
#include "generation_rope.cpp.inc"

// Prefetch (no dependencies on other generation modules)
#include "generation_prefetch.cpp.inc"

// Attention (depends on: rope, kv_cache, scratch, simd_helpers, mla_constants)
#include "generation_attention.cpp.inc"

// CPU-only storage execution engine and routing hooks.
// Included before MLP because the MoE MLP calls selected-expert prefetch hooks.
#include "generation/cpu_storage_engine.cpp.inc"

// MLP (depends on: router, raw_ops, silu, scratch, cpu_storage_engine hooks)
#include "generation_mlp.cpp.inc"

// Forward (depends on: attention, mlp, scratch, cpu_storage_engine)
#include "generation_forward.cpp.inc"

// Forward dispatch adapter (depends on generation_forward)
#include "generation/graph_contract_forward.cpp.inc"

// Main API (depends on: forward, kv_cache, scratch, error)
#include "generation_api_main.cpp.inc"

// Eval API (depends on: forward, lm_head logprob, kv_cache, scratch, error)
#include "generation_eval.cpp.inc"
