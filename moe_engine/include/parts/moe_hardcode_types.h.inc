typedef struct {
    uint32_t physical_part_count;
    uint32_t source_shard_count;
    uint32_t num_hidden_layers;
    uint32_t first_moe_layer;
    uint32_t last_moe_layer;
    uint32_t experts_per_moe_layer;
    uint32_t expert_projection_count;
    uint32_t hidden_size;
    uint32_t expert_intermediate_size;
    uint32_t vocab_size;
    uint64_t file_bytes;
    uint32_t block_count;
    uint32_t expert_bundle_count;
    uint32_t raw_expert_bundle_count;
    uint32_t raw_tensor_count;
    uint32_t scale4_count;
    uint32_t raw_scale_count;
    uint32_t raw_expert_scale_count;
    uint32_t logical_part_count;
    uint32_t total_expert_count;
    uint64_t normal_bundle_min_bytes;
    uint64_t normal_bundle_max_bytes;
    uint64_t layer78_raw_bundle_bytes;
    uint64_t raw_tensor_total_bytes;
    uint64_t raw_tensor_attention_bytes;
    uint64_t raw_tensor_dense_or_router_mlp_bytes;
    uint64_t raw_tensor_embedding_bytes;
    uint64_t raw_tensor_lm_head_bytes;
    uint64_t raw_tensor_normalization_bytes;
    uint64_t raw_tensor_other_bytes;
    uint64_t raw_tensor_shared_expert_bytes;
    uint64_t runtime_state_bytes;
    const char* storage_repo;
    const char* extension;
    const char* magic;
} moe_storage_constants_t;

typedef struct {
    uint32_t part;
    const char* path;
    uint64_t bytes;
    uint32_t block_count;
    uint32_t expert_bundle_count;
    uint32_t raw_expert_bundle_count;
    uint32_t raw_tensor_count;
    uint64_t raw_tensor_bytes;
    uint64_t raw_tensor_offset;
    uint64_t raw_tensor_span_bytes;
    uint32_t primary_first;
    uint32_t primary_count;
    uint32_t read_first;
    uint32_t read_count;
} moe_storage_part_spec_t;

typedef struct {
    int32_t layer;
    uint32_t part;
    uint64_t offset;
    uint64_t bytes;
    uint32_t role;
} moe_storage_raw_span_spec_t;

typedef struct {
    uint32_t layer;
    int has_moe_experts;
    uint32_t expert_count;
    uint32_t part0;
    uint32_t part1;
    uint32_t source_shard0;
    uint32_t source_shard1;
    uint64_t bundle_total_bytes;
    uint64_t bundle_min_bytes;
    uint64_t bundle_max_bytes;
    uint32_t scale4_count;
    uint32_t raw_scale_count;
    uint32_t raw_expert_scale_count;
    uint32_t raw_tensor_count;
} moe_storage_layer_spec_t;

typedef struct {
    uint32_t part_count;
    uint32_t shard_first;
    uint32_t shard_last;
    uint32_t shard_covered_count;
    uint32_t missing_shard_count;
    uint32_t duplicate_primary_shard_count;
    uint32_t first_missing_shard;
    uint32_t first_duplicate_primary_shard;
    uint64_t total_bytes;
    uint32_t block_count;
    uint32_t expert_bundle_count;
    uint32_t raw_expert_bundle_count;
    uint32_t raw_tensor_count;
    uint64_t raw_tensor_bytes;
    uint32_t scale4_count;
    uint32_t raw_scale_count;
    uint32_t raw_expert_scale_count;
    int matches_constants;
} moe_storage_model_state_t;

typedef struct {
    uint32_t expected_part_count;
    uint32_t present_part_count;
    uint32_t missing_part_count;
    uint32_t size_mismatch_count;
    uint32_t first_missing_part;
    uint32_t first_size_mismatch_part;
    uint64_t expected_total_bytes;
    uint64_t present_total_bytes;
    uint64_t first_expected_bytes;
    uint64_t first_actual_bytes;
    char first_missing_path[512];
    char first_size_mismatch_path[512];
    int valid;
} moe_model_root_check_t;

typedef struct {
    moe_projection_t proj;
    uint32_t rows;
    uint32_t cols;
    uint32_t scale_groups;
    uint32_t group_size;
} moe_projection_shape_spec_t;
