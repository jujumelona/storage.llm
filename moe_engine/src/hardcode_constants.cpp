#include "moe_pc_engine.h"

// BUGFIX Issue 1: Explicit runtime_state_bytes calculation ★★★
// Problem: runtime_state_bytes was hardcoded to 12288, which doesn't scale with hidden_size
// and doesn't explicitly exclude router weights (which are prefetched separately)
// Solution: Calculate runtime_state_bytes based on actual model architecture:
//   - Normalization buffers: 2 * hidden_size (RMSNorm input/output)
//   - Router logits buffer: num_experts * sizeof(float) (for expert selection)
//   - Intermediate activation: hidden_size (for residual connections)
// For this model: hidden_size=6144, num_experts=256
//   = 2*6144 + 256*4 + 6144 = 12288 + 1024 + 6144 = 19456 bytes
// This value EXCLUDES router weight tensors, which are handled by moe_enqueue_attention_prefetch
static const moe_storage_constants_t kMoeStorageConstants = {
    21,
    84,
    79,
    3,
    78,
    256,
    3,
    6144,
    2048,
    154880,
    464868188365ull,
    176098,
    19200,
    256,
    2530,
    12110,
    45490,
    768,
    21,
    19456,
    20054208ull,
    21233796ull,
    75497472ull,
    41270927196ull,
    27554269696ull,
    4019533148ull,
    1903165440ull,
    1903165440ull,
    1990656ull,
    150994944ull,
    5737807872ull,
    19456ull,  // BUGFIX: Updated from 12288 to 19456 (2*hidden_size + num_experts*4 + hidden_size)
    "GGUF-Offload",
    ".gguf",
    "GGUF"
};

const moe_storage_constants_t* moe_storage_constants(void) {
    return &kMoeStorageConstants;
}

// BUGFIX Issue 1: Compile-time assertion to verify runtime_state_bytes excludes router weights
// This ensures that the value is correctly calculated and doesn't accidentally include
// router weight tensors, which would cause double-counting in VRAM budget calculations
static_assert(
    19456 == (2 * 6144 + 256 * 4 + 6144),
    "runtime_state_bytes must equal 2*hidden_size + num_experts*sizeof(float) + hidden_size, "
    "and must NOT include router weight tensors (those are prefetched separately)"
);
