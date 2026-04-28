int glm5_pc_group_tokens_by_expert(
    const glm5_token_expert_assignment_t* assignments,
    uint32_t assignment_count,
    glm5_token_expert_assignment_t* out_sorted_assignments,
    glm5_expert_batch_t* out_batches,
    uint32_t max_batches,
    uint32_t* out_batch_count
);

const char* glm5_kv_mode_name(glm5_kv_mode_t mode);
const char* glm5_tier_name(glm5_tier_t tier);
const char* glm5_projection_name(glm5_projection_t proj);
const char* glm5_io_path_name(glm5_io_path_t path);
const char* glm5_backend_name(glm5_backend_t backend);
const char* glm5_platform_name(glm5_platform_t platform);
