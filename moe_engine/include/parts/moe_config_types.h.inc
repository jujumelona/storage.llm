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
    float vram_kv_ratio;  // BUGFIX 문제F: KV/expert VRAM split ratio (0.0=auto, 0.1-0.3 recommended) ★★
    int32_t device_id;  // BUGFIX Problem 5: GPU device index for multi-GPU systems (-1=auto, default 0) ★★
    uint32_t max_context_len;  // BUGFIX Problem 7: Override KV cache context length (0=use default 8192) ★★★
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
    uint32_t directstorage_staging_queue_capacity;
    uint32_t directstorage_gpu_queue_capacity;
    int enable_compute_io_overlap;
    int enable_multistage_pipeline;
    int enable_lookahead_prefetch;
    int enable_topology_prefetch;
    int enable_mmap_prefetch;
    int enable_bw_benchmark;
    int enable_pinned_host;
    int enable_directstorage;
    int enable_gpudirect_storage;
    int enable_bam_nvme;
    int enable_common_raw_vram_pin;
    int allow_required_sync_fallback;
    int require_bundle_native_experts;
    float int4_fallback_threshold;
    uint8_t vram_eviction_threshold_pct;
    uint8_t ram_eviction_threshold_pct;
    uint8_t ram_eviction_target_fill_pct;  // RAM eviction target percentage (0=default 75%)
    uint8_t attention_prefetch_threshold_pct;
    uint8_t madv_cold_threshold_pct;
    uint8_t metal_memory_budget_pct;  // Metal unified-memory budget percentage (0=auto, safe range 50-90)
    uint8_t vulkan_memory_headroom_pct;  // Vulkan free-memory headroom percentage (0=default 10)
    uint8_t level_zero_memory_headroom_pct;  // Level Zero free-memory headroom percentage (0=default 10)
    int prefer_overlapped_io;  // BUGFIX 897: Windows non-DirectStorage async I/O (0=mmap+VirtualLock, 1=OVERLAPPED ReadFile) ★★★
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
    int allow_speculative_prefetch;  // Renamed from route_sparse_to_cpu for clarity
    int throttle_speculative;
    int drop_speculative;
} moe_runtime_orchestrator_plan_t;
