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

// Router (depends on: raw_ops)
#include "generation_router.cpp.inc"

// KV cache (depends on: bf16_convert, mla_constants)
#include "generation_kv_cache.cpp.inc"

// Scratch (depends on: kv_cache, mla_constants)
#include "generation_scratch.cpp.inc"

// RoPE (depends on: mla_constants)
#include "generation_rope.cpp.inc"

// Prefetch (no dependencies on other generation modules)
#include "generation_prefetch.cpp.inc"

// Attention (depends on: rope, kv_cache, scratch, simd_helpers, mla_constants)
#include "generation_attention.cpp.inc"

// MLP (depends on: router, raw_ops, silu, scratch)
#include "generation_mlp.cpp.inc"

// Forward (depends on: attention, mlp, scratch)
#include "generation_forward.cpp.inc"

// Main API (depends on: forward, kv_cache, scratch, error)
#include "generation_api_main.cpp.inc"

// Eval API (depends on: forward, lm_head logprob, kv_cache, scratch, error)
#include "generation_eval.cpp.inc"
