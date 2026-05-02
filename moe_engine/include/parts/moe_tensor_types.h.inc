typedef enum {
    moe_WEIGHT_ENCODING_UNKNOWN = 0,
    moe_WEIGHT_ENCODING_RAW_FP16 = 1,
    moe_WEIGHT_ENCODING_RAW_FP32 = 2,
    moe_WEIGHT_ENCODING_NVFP4 = 3,
    moe_WEIGHT_ENCODING_MXFP4 = 4,
    moe_WEIGHT_ENCODING_LINEAR_Q2 = 5,
    moe_WEIGHT_ENCODING_LINEAR_Q3 = 6,
    moe_WEIGHT_ENCODING_LINEAR_Q4 = 7,
    moe_WEIGHT_ENCODING_LINEAR_Q8 = 8,
    moe_WEIGHT_ENCODING_GGUF_IQ2 = 9,
    moe_WEIGHT_ENCODING_GGUF_IQ3 = 10,
    moe_WEIGHT_ENCODING_GGUF_IQ4 = 11,
    moe_WEIGHT_ENCODING_GGUF_Q5_1 = 12,
    moe_WEIGHT_ENCODING_GGUF_Q8_0 = 13,
    moe_WEIGHT_ENCODING_GGUF_Q5_K = 14,
    moe_WEIGHT_ENCODING_GGUF_Q2_K = 15,
    moe_WEIGHT_ENCODING_GGUF_Q3_K = 16,
    moe_WEIGHT_ENCODING_GGUF_Q4_K = 17,
    moe_WEIGHT_ENCODING_GGUF_Q6_K = 18,
    moe_WEIGHT_ENCODING_GGUF_IQ2_XXS = 19,
    moe_WEIGHT_ENCODING_GGUF_IQ3_XXS = 20
} moe_weight_encoding_t;

typedef struct {
    uint64_t weight_byte_offset;
    uint64_t weight_byte_length;
    uint64_t scale_byte_offset;
    uint64_t scale_byte_length;
    uint64_t scale2_byte_offset;
    uint64_t scale2_byte_length;
    uint32_t part;
    uint32_t shard;
    uint32_t layer;
    uint32_t expert;
    moe_projection_t proj;
    uint32_t rows;
    uint32_t cols;
    uint32_t groups;
    uint32_t group_size;
    int has_scale4;
    uint32_t weight_encoding;
    uint32_t weight_bits;
    uint32_t weight_block_size;
} moe_tensor_info_t;
typedef struct {
    uint32_t token_index;
    uint32_t layer;
    uint32_t expert;
    float router_score;
} moe_token_expert_assignment_t;

typedef struct {
    uint32_t layer;
    uint32_t expert;
    uint32_t first_sorted_index;
    uint32_t token_count;
} moe_expert_batch_t;

typedef struct {
    uint32_t layer;
    uint32_t expert;
    uint64_t bytes;
    moe_tier_t target_tier;
    moe_prefetch_reason_t reason;
    float priority;
} moe_expert_prefetch_item_t;


