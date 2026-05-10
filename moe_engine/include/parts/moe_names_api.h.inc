int moe_pc_group_tokens_by_expert(
    const moe_token_expert_assignment_t* assignments,
    uint32_t assignment_count,
    moe_token_expert_assignment_t* out_sorted_assignments,
    moe_expert_batch_t* out_batches,
    uint32_t max_batches,
    uint32_t* out_batch_count
);

const char* moe_kv_mode_name(moe_kv_mode_t mode);
const char* moe_tier_name(moe_tier_t tier);
const char* moe_projection_name(moe_projection_t proj);
const char* moe_io_path_name(moe_io_path_t path);
const char* moe_backend_name(moe_backend_t backend);
const char* moe_platform_name(moe_platform_t platform);
