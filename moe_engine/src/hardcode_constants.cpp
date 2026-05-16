#include "moe_pc_engine.h"

// Model-agnostic default constants.
// All shape values are zero so that the dynamic Graph IR contract
// (engine->dynamic_shape populated by the JUJU parser) takes priority
// via moe_engine_model_shape_or_static() / moe_engine_storage_constants_or_static().
// No model names, no model-specific numbers.

static const moe_storage_constants_t kMoeStorageConstants = {
    0,      // physical_part_count
    0,      // source_shard_count
    0,      // num_hidden_layers
    0,      // first_moe_layer
    0,      // last_moe_layer
    0,      // experts_per_moe_layer
    3,      // expert_projection_count (gate/up/down is universal for MoE)
    0,      // hidden_size
    0,      // expert_intermediate_size
    0,      // vocab_size
    0ull,   // file_bytes
    0,      // block_count
    0,      // expert_bundle_count
    0,      // raw_expert_bundle_count
    0,      // raw_tensor_count
    0,      // scale4_count
    0,      // raw_scale_count
    0,      // raw_expert_scale_count
    0,      // logical_part_count
    0,      // total_expert_count  (runtime: layers * experts_per_layer)
    0ull,   // normal_bundle_min_bytes
    0ull,   // normal_bundle_max_bytes
    0ull,   // layer78_raw_bundle_bytes
    0ull,   // raw_tensor_total_bytes
    0ull,   // raw_tensor_attention_bytes
    0ull,   // raw_tensor_dense_or_router_mlp_bytes
    0ull,   // raw_tensor_embedding_bytes
    0ull,   // raw_tensor_lm_head_bytes
    0ull,   // raw_tensor_normalization_bytes
    0ull,   // raw_tensor_other_bytes
    0ull,   // raw_tensor_shared_expert_bytes
    0ull,   // runtime_state_bytes  (computed dynamically from hidden_size)
    "GGUF-Offload",
    ".gguf",
    "GGUF"
};

const moe_storage_constants_t* moe_storage_constants(void) {
    return &kMoeStorageConstants;
}
