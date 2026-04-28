typedef struct {
    uint64_t vram_budget_bytes;
    uint64_t ram_budget_bytes;
    uint64_t db_budget_bytes;
    glm5_kv_mode_t kv_mode;
    int prefer_gpu;
    int allow_db_streaming;
    glm5_backend_t preferred_backend;
    glm5_platform_t platform;
    uint64_t common_vram_reserve_bytes;
    const char* model_lib_cache_dir;
    const char* model_lib_path;
    const char* model_lib_compile_command;
    int enable_model_lib_jit;
} glm5_pc_engine_config_t;

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
} glm5_io_config_t;

typedef struct {
    glm5_execution_policy_t policy;
    float cpu_load_pct;
    float gpu_load_pct;
    uint32_t active_tokens;
    uint32_t active_os_tasks;
} glm5_runtime_orchestrator_state_t;

typedef struct {
    glm5_execution_policy_t policy;
    glm5_tier_t selected_tier;
    glm5_tier_t speculative_tier;
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
} glm5_runtime_orchestrator_plan_t;
