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


