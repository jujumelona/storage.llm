typedef struct {
    int storage_state_valid;
    int tensor_table_loaded;
    int expert_triplet_available;
    int tokenizer_ready;
    int embedding_ready;
    int lm_head_ready;
    int attention_projector_ready;
    int attention_decode_ready;
    int kv_cache_ready;
    int attention_ready;
    int sampler_ready;
    int decode_loop_ready;
    int chat_graph_ready;
    uint64_t tensor_count;
} glm5_forward_status_t;

typedef enum {
    GLM5_RAW_TENSOR_INPUT_LAYER_NORM = 0,
    GLM5_RAW_TENSOR_POST_ATTENTION_LAYER_NORM = 1,
    GLM5_RAW_TENSOR_Q_A_LAYER_NORM = 2,
    GLM5_RAW_TENSOR_KV_A_LAYER_NORM = 3,
    GLM5_RAW_TENSOR_Q_A_PROJ = 4,
    GLM5_RAW_TENSOR_Q_B_PROJ = 5,
    GLM5_RAW_TENSOR_KV_A_PROJ_WITH_MQA = 6,
    GLM5_RAW_TENSOR_KV_B_PROJ = 7,
    GLM5_RAW_TENSOR_O_PROJ = 8,
    GLM5_RAW_TENSOR_INDEXER_K_NORM_BIAS = 9,
    GLM5_RAW_TENSOR_INDEXER_K_NORM_WEIGHT = 10,
    GLM5_RAW_TENSOR_INDEXER_WEIGHTS_PROJ = 11,
    GLM5_RAW_TENSOR_INDEXER_WK = 12,
    GLM5_RAW_TENSOR_INDEXER_WQ_B = 13
} glm5_raw_tensor_kind_t;

typedef struct {
    int32_t layer;
    uint32_t part;
    uint64_t offset;
    uint64_t bytes;
    uint32_t rows;
    uint32_t cols;
    int bf16;
} glm5_raw_tensor_info_t;

typedef struct {
    uint32_t max_new_tokens;
    uint32_t eos_token_id;
    int stop_on_eos;
} glm5_generation_config_t;

typedef struct {
    uint32_t prompt_tokens;
    uint32_t completion_tokens;
    uint32_t finish_reason;
    float last_logit;
    char error[160];
} glm5_generation_stats_t;

typedef struct {
    uint32_t input_tokens;
    uint32_t evaluated_tokens;
    double nll;
    double mean_nll;
    double perplexity;
    char error[160];
} glm5_eval_stats_t;

int glm5_pc_engine_get_forward_status(
    const glm5_pc_engine_t* engine,
    glm5_forward_status_t* out_status
);

int glm5_pc_engine_run_expert_triplet_f32(
    glm5_pc_engine_t* engine,
    int layer,
    int expert,
    const float* hidden,
    uint32_t hidden_count,
    float* scratch_gate,
    float* scratch_up,
    uint32_t intermediate_count,
    float* out_hidden,
    uint32_t out_count
);

int glm5_pc_engine_embed_token_f32(
    glm5_pc_engine_t* engine,
    uint32_t token_id,
    float* out_hidden,
    uint32_t out_count
);

int glm5_pc_engine_embed_tokens_f32(
    glm5_pc_engine_t* engine,
    const int32_t* token_ids,
    uint32_t token_count,
    float* out_hidden,
    uint32_t out_count
);

int glm5_pc_engine_lm_head_top1_f32(
    glm5_pc_engine_t* engine,
    const float* hidden,
    uint32_t hidden_count,
    uint32_t* out_token_id,
    float* out_logit
);

int glm5_pc_engine_lm_head_token_logprob_f32(
    glm5_pc_engine_t* engine,
    const float* hidden,
    uint32_t hidden_count,
    uint32_t target_token_id,
    float* out_target_logit,
    float* out_logsumexp,
    float* out_logprob
);

int glm5_pc_engine_generate_token_ids(
    glm5_pc_engine_t* engine,
    const int32_t* input_token_ids,
    uint32_t input_token_count,
    const glm5_generation_config_t* config,
    uint32_t* out_token_ids,
    uint32_t out_token_capacity,
    glm5_generation_stats_t* out_stats
);

int glm5_pc_engine_eval_token_ids(
    glm5_pc_engine_t* engine,
    const int32_t* input_token_ids,
    uint32_t input_token_count,
    glm5_eval_stats_t* out_stats
);

int glm5_pc_engine_find_raw_tensor(
    glm5_pc_engine_t* engine,
    int32_t layer,
    glm5_raw_tensor_kind_t kind,
    glm5_raw_tensor_info_t* out_info
);

int glm5_pc_engine_raw_linear_f32(
    glm5_pc_engine_t* engine,
    int32_t layer,
    glm5_raw_tensor_kind_t kind,
    const float* input,
    uint32_t input_count,
    float* output,
    uint32_t output_count
);

int glm5_pc_engine_raw_rmsnorm_f32(
    glm5_pc_engine_t* engine,
    int32_t layer,
    glm5_raw_tensor_kind_t kind,
    const float* input,
    uint32_t input_count,
    float* output,
    uint32_t output_count,
    float eps
);

int glm5_pc_engine_attention_decode_f32(
    glm5_pc_engine_t* engine,
    const float* query,
    const float* key_cache,
    const float* value_cache,
    uint32_t context_tokens,
    uint32_t head_count,
    uint32_t head_dim,
    float* output
);
