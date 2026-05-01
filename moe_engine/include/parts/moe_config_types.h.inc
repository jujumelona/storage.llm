typedef struct {
    uint64_t vram_budget_bytes;
    uint64_t ram_budget_bytes;
    uint64_t db_budget_bytes;
    moe_kv_mode_t kv_mode;
    int prefer_gpu;
    int allow_db_streaming;
    moe_backend_t preferred_backend;
    moe_platform_t platform;
    uint64_t common_vram_reserve_bytes;
    const char* model_lib_cache_dir;
    const char* model_lib_path;
    const char* model_lib_compile_command;
    int enable_model_lib_jit;
} moe_pc_engine_config_t;

typedef enum {
    moe_ATTN_TYPE_UNKNOWN = 0,
    moe_ATTN_TYPE_MHA = 1,
    moe_ATTN_TYPE_GQA = 2,
    moe_ATTN_TYPE_MQA = 3,
    moe_ATTN_TYPE_MLA = 4
} moe_attn_type_t;

typedef struct {
    moe_attn_type_t type;
    uint32_t num_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    uint32_t v_head_dim;
    uint32_t kv_lora_rank;
    uint32_t qk_nope_head_dim;
    uint32_t qk_rope_head_dim;
    uint32_t q_lora_rank;
    uint32_t max_seq_len;
    uint32_t rope_table_len;
    float rope_theta;
    float rms_norm_eps;
} moe_attn_config_t;

typedef struct {
    uint32_t io_worker_threads;
    uint32_t prefetch_window_layers;
    uint32_t max_prefetch_queue;
    uint32_t topology_max_candidates;
    uint32_t topology_max_edges_per_source;
    uint32_t disk_worker_threads;
    uint32_t pinned_worker_threads;
    uint32_t gpu_worker_threads;
    uint64_t pinned_staging_bytes;
    uint64_t direct_upload_min_bytes;
    int enable_compute_io_overlap;
    int enable_multistage_pipeline;
    int enable_lookahead_prefetch;
    int enable_topology_prefetch;
    int enable_mmap_prefetch;
    int enable_pinned_host;
    int enable_directstorage;
    int enable_gpudirect_storage;
    int enable_bam_nvme;
    int enable_common_raw_vram_pin;
    float int4_fallback_threshold;
    uint8_t vram_eviction_threshold_pct;
    uint8_t ram_eviction_threshold_pct;
    uint8_t attention_prefetch_threshold_pct;
    uint8_t metal_memory_budget_pct;  // BUGFIX 199: Metal VRAM budget percentage (default 60, safe range 60-85)
} moe_io_config_t;

typedef struct {
    moe_execution_policy_t policy;
    float cpu_load_pct;
    float gpu_load_pct;
    uint32_t active_tokens;
    uint32_t active_os_tasks;
} moe_runtime_orchestrator_state_t;

typedef struct {
    moe_execution_policy_t policy;
    moe_tier_t selected_tier;
    moe_tier_t speculative_tier;
    uint32_t disk_workers;
    uint32_t pinned_workers;
    uint32_t gpu_workers;
    uint32_t prefetch_window_layers;
    uint32_t planned_prefetch_count;
    uint64_t selected_expert_bytes;
    int allow_gpu_dense;
    int route_sparse_to_cpu;
    int throttle_speculative;
    int drop_speculative;
} moe_runtime_orchestrator_plan_t;
