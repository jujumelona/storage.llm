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
    uint32_t forward_adapter;
    int forward_adapter_executable;
    int dynamic_shape_ready;
    uint32_t dynamic_num_hidden_layers;
    uint32_t dynamic_hidden_size;
    uint32_t dynamic_vocab_size;
    uint32_t dynamic_max_seq_len;
    int graph_ir_ready;
    int graph_ir_required;
    uint32_t graph_ir_version;
    uint64_t graph_ir_layer_count;
    uint64_t graph_ir_op_count;
    int runtime_execution_manifest_ready;
    int runtime_access_plan_ready;
    uint64_t runtime_access_hotset_count;
    uint64_t runtime_access_file_group_count;
    uint64_t runtime_access_executor_tensor_count;
    uint64_t runtime_access_layer_prefetch_plan_count;
    int kv_layout_contract_ready;
    uint64_t kv_layout_max_entry_dim;
    uint64_t kv_layout_page_size_tokens;
    int cpu_only_runtime;
    int cpu_storage_engine_ready;
    int cpu_storage_execution_plan_ready;
    int cpu_quant_registry_ready;
    int graph_ir_used_for_forward;
    int gpu_path_disabled_for_cpu_only;
    uint32_t cpu_execution_plan_layers;
    uint64_t cpu_execution_plan_tensors;
    uint64_t cpu_execution_plan_expert_tensors;
    uint64_t cpu_execution_plan_bytes;
    uint64_t cpu_prefetch_scheduled;
    uint64_t cpu_selected_expert_prefetch_scheduled;
    uint64_t cpu_layer_begin_count;
int cpu_hot_tensor_views_ready;
uint64_t cpu_hot_tensor_view_count;
uint64_t cpu_hot_tensor_simple_dot_count;
uint64_t cpu_hot_tensor_bytes;
uint64_t cpu_hot_tensor_reused;
int cpu_layer_hot_plan_ready;
uint64_t cpu_layer_hot_tensor_count;
uint64_t cpu_layer_hot_prefetch_count;
uint64_t cpu_layer_hot_prefetch_bytes;
uint64_t cpu_layer_prefetch_duplicate_skips;
int storage_io_plan_ready;
uint32_t storage_profile;
uint32_t storage_prefetch_workers;
uint32_t storage_prefetch_queue_depth;
uint64_t storage_io_span_count;
uint64_t storage_io_merged_span_count;
uint64_t storage_io_prefetch_count;
uint64_t storage_io_prefetch_bytes;
uint64_t storage_io_fadvise_count;
uint64_t storage_io_readahead_count;
uint64_t storage_io_madvise_count;
uint64_t storage_io_page_touch_count;
uint64_t storage_io_duplicate_skips;
int storage_iouring_available;
int storage_iouring_enabled;
uint64_t storage_iouring_submit_count;
uint64_t storage_iouring_bytes;
uint64_t storage_iouring_fallback_count;
int storage_async_prefetch_enabled;
uint64_t storage_async_prefetch_enqueued;
uint64_t storage_async_prefetch_completed;
uint64_t storage_async_prefetch_dropped;
uint64_t storage_async_prefetch_inline;
uint64_t storage_async_prefetch_pending;
int storage_gpu_staging_plan_ready;
uint64_t storage_gpu_staging_hook_count;
uint64_t storage_gpu_staging_bytes;
uint64_t storage_cpu_pagecache_hook_count;
uint64_t storage_cpu_pagecache_bytes;
int storage_expert_index_ready;
uint64_t storage_expert_index_entry_count;
uint64_t storage_expert_index_view_count;
uint64_t storage_selected_expert_index_hits;
uint64_t storage_selected_expert_linear_fallbacks;
uint64_t storage_selected_expert_prefetch_bytes;
uint64_t storage_selected_expert_duplicate_skips;
int storage_format_plan_available;
} moe_forward_status_t;

typedef enum {
    moe_RAW_TENSOR_INPUT_LAYER_NORM = 0,
    moe_RAW_TENSOR_POST_ATTENTION_LAYER_NORM = 1,
    moe_RAW_TENSOR_Q_A_LAYER_NORM = 2,
    moe_RAW_TENSOR_KV_A_LAYER_NORM = 3,
    moe_RAW_TENSOR_Q_A_PROJ = 4,
    moe_RAW_TENSOR_Q_B_PROJ = 5,
    moe_RAW_TENSOR_KV_A_PROJ_WITH_MQA = 6,
    moe_RAW_TENSOR_KV_B_PROJ = 7,
    moe_RAW_TENSOR_O_PROJ = 8,
    moe_RAW_TENSOR_INDEXER_K_NORM_BIAS = 9,
    moe_RAW_TENSOR_INDEXER_K_NORM_WEIGHT = 10,
    moe_RAW_TENSOR_INDEXER_WEIGHTS_PROJ = 11,
    moe_RAW_TENSOR_INDEXER_WK = 12,
    moe_RAW_TENSOR_INDEXER_WQ_B = 13,
    moe_RAW_TENSOR_V_A_LAYER_NORM = 14
} moe_raw_tensor_kind_t;

typedef struct {
    int32_t layer;
    uint32_t part;
    uint64_t offset;
    uint64_t bytes;
    uint32_t rows;
    uint32_t cols;
    int bf16;
} moe_raw_tensor_info_t;

typedef struct {
    uint32_t max_new_tokens;
    uint32_t eos_token_id;
    int stop_on_eos;
} moe_generation_config_t;

typedef struct {
    uint32_t prompt_tokens;
    uint32_t completion_tokens;
    uint32_t finish_reason;
    float last_logit;
    char error[160];
} moe_generation_stats_t;

typedef struct {
    uint32_t input_tokens;
    uint32_t evaluated_tokens;
    double nll;
    double mean_nll;
    double perplexity;
    char error[160];
} moe_eval_stats_t;

int moe_pc_engine_get_forward_status(
    const moe_pc_engine_t* engine,
    moe_forward_status_t* out_status
);

int moe_pc_engine_run_expert_triplet_f32(
    moe_pc_engine_t* engine,
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

int moe_pc_engine_embed_token_f32(
    moe_pc_engine_t* engine,
    uint32_t token_id,
    float* out_hidden,
    uint32_t out_count
);

int moe_pc_engine_embed_tokens_f32(
    moe_pc_engine_t* engine,
    const int32_t* token_ids,
    uint32_t token_count,
    float* out_hidden,
    uint32_t out_count
);

int moe_pc_engine_lm_head_top1_f32(
    moe_pc_engine_t* engine,
    const float* hidden,
    uint32_t hidden_count,
    uint32_t* out_token_id,
    float* out_logit
);

int moe_pc_engine_lm_head_token_logprob_f32(
    moe_pc_engine_t* engine,
    const float* hidden,
    uint32_t hidden_count,
    uint32_t target_token_id,
    float* out_target_logit,
    float* out_logsumexp,
    float* out_logprob
);

int moe_pc_engine_generate_token_ids(
    moe_pc_engine_t* engine,
    const int32_t* input_token_ids,
    uint32_t input_token_count,
    const moe_generation_config_t* config,
    uint32_t* out_token_ids,
    uint32_t out_token_capacity,
    moe_generation_stats_t* out_stats
);

int moe_pc_engine_eval_token_ids(
    moe_pc_engine_t* engine,
    const int32_t* input_token_ids,
    uint32_t input_token_count,
    moe_eval_stats_t* out_stats
);

int moe_pc_engine_eval_token_ids_from(
    moe_pc_engine_t* engine,
    const int32_t* input_token_ids,
    uint32_t input_token_count,
    uint32_t first_target_index,
    moe_eval_stats_t* out_stats
);

// Continuous Batching API (always built; paged-KV is optional allocator, not a compile gate)
int moe_pc_engine_submit_request(
    moe_pc_engine_t* engine,
    uint64_t request_id,
    const int32_t* prompt_tokens,
    uint32_t token_count
);

int moe_pc_engine_submit_request_ex(
    moe_pc_engine_t* engine,
    uint64_t request_id,
    const int32_t* prompt_tokens,
    uint32_t token_count,
    uint32_t max_new_tokens,
    uint32_t eos_token_id,
    uint32_t stop_on_eos
);

int moe_pc_engine_step_batch(moe_pc_engine_t* engine);

int moe_pc_engine_fetch_request_result(
    moe_pc_engine_t* engine,
    uint64_t request_id,
    int32_t* out_token_ids,
    uint32_t out_token_capacity,
    uint32_t* out_token_count,
    uint32_t* out_prompt_token_count,
    int* out_failed,
    int remove_after_fetch
);

int moe_pc_engine_cancel_request(
    moe_pc_engine_t* engine,
    uint64_t request_id
);

int moe_pc_engine_wait_request_result(
    moe_pc_engine_t* engine,
    uint64_t request_id,
    uint32_t timeout_ms
);

int moe_pc_engine_find_raw_tensor(
    moe_pc_engine_t* engine,
    int32_t layer,
    moe_raw_tensor_kind_t kind,
    moe_raw_tensor_info_t* out_info
);

int moe_pc_engine_raw_linear_f32(
    moe_pc_engine_t* engine,
    int32_t layer,
    moe_raw_tensor_kind_t kind,
    const float* input,
    uint32_t input_count,
    float* output,
    uint32_t output_count
);

int moe_pc_engine_raw_rmsnorm_f32(
    moe_pc_engine_t* engine,
    int32_t layer,
    moe_raw_tensor_kind_t kind,
    const float* input,
    uint32_t input_count,
    float* output,
    uint32_t output_count,
    float eps
);

int moe_pc_engine_attention_decode_f32(
    moe_pc_engine_t* engine,
    const float* query,
    const float* key_cache,
    const float* value_cache,
    uint32_t context_tokens,
    uint32_t head_count,
    uint32_t head_dim,
    float* output
);
