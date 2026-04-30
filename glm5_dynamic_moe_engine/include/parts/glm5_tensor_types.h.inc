typedef enum {
    GLM5_WEIGHT_ENCODING_UNKNOWN = 0,
    GLM5_WEIGHT_ENCODING_RAW_FP16 = 1,
    GLM5_WEIGHT_ENCODING_RAW_FP32 = 2,
    GLM5_WEIGHT_ENCODING_NVFP4 = 3,
    GLM5_WEIGHT_ENCODING_MXFP4 = 4,
    GLM5_WEIGHT_ENCODING_LINEAR_Q2 = 5,
    GLM5_WEIGHT_ENCODING_LINEAR_Q3 = 6,
    GLM5_WEIGHT_ENCODING_LINEAR_Q4 = 7,
    GLM5_WEIGHT_ENCODING_LINEAR_Q8 = 8,
    GLM5_WEIGHT_ENCODING_GGUF_IQ2 = 9,
    GLM5_WEIGHT_ENCODING_GGUF_IQ3 = 10,
    GLM5_WEIGHT_ENCODING_GGUF_IQ4 = 11
} glm5_weight_encoding_t;

typedef struct {
    uint32_t part;
    uint32_t shard;
    uint32_t layer;
    uint32_t expert;
    glm5_projection_t proj;
    uint32_t rows;
    uint32_t cols;
    uint32_t groups;
    uint32_t group_size;
    uint64_t weight_byte_offset;
    uint64_t weight_byte_length;
    uint64_t scale_byte_offset;
    uint64_t scale_byte_length;
    uint64_t scale2_byte_offset;
    uint64_t scale2_byte_length;
    int has_scale4;
    uint32_t weight_encoding;
    uint32_t weight_bits;
    uint32_t weight_block_size;
} glm5_tensor_info_t;

typedef struct {
    uint32_t token_index;
    uint32_t layer;
    uint32_t expert;
    float router_score;
} glm5_token_expert_assignment_t;

typedef struct {
    uint32_t layer;
    uint32_t expert;
    uint32_t first_sorted_index;
    uint32_t token_count;
} glm5_expert_batch_t;

typedef struct {
    uint32_t layer;
    uint32_t expert;
    uint64_t bytes;
    glm5_tier_t target_tier;
    glm5_prefetch_reason_t reason;
    float priority;
} glm5_expert_prefetch_item_t;


