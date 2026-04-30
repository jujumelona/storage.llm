const glm5_storage_constants_t* glm5_storage_constants(void);

uint32_t glm5_storage_part_count(void);

const glm5_storage_part_spec_t* glm5_storage_part_at(uint32_t index);

const glm5_storage_part_spec_t* glm5_storage_part_by_id(uint32_t part);

uint32_t glm5_storage_raw_span_count(void);

const glm5_storage_raw_span_spec_t* glm5_storage_raw_span_at(uint32_t index);

uint32_t glm5_storage_raw_span_find_first(int32_t layer);

uint32_t glm5_storage_raw_span_find_end(int32_t layer);

// BUGFIX 30: O(1) attention span index lookup ★★★
// Returns span index for attention tensor kinds (Q_A_PROJ, Q_B_PROJ, KV_A_PROJ_WITH_MQA, KV_B_PROJ, O_PROJ)
// kind_index: 0=Q_A_PROJ, 1=Q_B_PROJ, 2=KV_A_PROJ_WITH_MQA, 3=KV_B_PROJ, 4=O_PROJ
// Returns 0xFFFFFFFF if layer/kind_index out of range or span not found
uint32_t glm5_storage_attention_span_index(int32_t layer, uint32_t kind_index);

uint32_t glm5_storage_layer_count(void);

const glm5_storage_layer_spec_t* glm5_storage_layer_at(uint32_t layer);

const glm5_storage_model_state_t* glm5_storage_model_state_summary(void);

int glm5_storage_model_state_valid(void);

int glm5_storage_validate_model_root(
    const char* model_root,
    glm5_model_root_check_t* out_check
);

const glm5_projection_shape_spec_t* glm5_storage_projection_shape(glm5_projection_t proj);
