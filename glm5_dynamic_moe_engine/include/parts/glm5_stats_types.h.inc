enum {
    GLM5_GEN_PHASE_IDLE = 0u,
    GLM5_GEN_PHASE_INIT = 1u,
    GLM5_GEN_PHASE_PREFILL = 2u,
    GLM5_GEN_PHASE_ATTENTION = 3u,
    GLM5_GEN_PHASE_MLP = 4u,
    GLM5_GEN_PHASE_LM_HEAD = 5u,
    GLM5_GEN_PHASE_DONE = 6u
};

typedef struct {
    uint64_t vram_used_bytes;
    uint64_t ram_used_bytes;
    uint64_t db_used_bytes;
    uint64_t common_vram_reserved_bytes;
    uint64_t vram_expert_used_bytes;
    uint64_t common_raw_prefetched_bytes;
    uint64_t common_raw_prefetched_spans;
    uint64_t device_allocated_bytes;
    uint64_t device_allocation_count;
    uint64_t device_fixed_bytes;
    uint64_t device_activation_bytes;
    uint64_t device_expert_bytes;
    uint64_t device_stream_bytes;
    uint64_t device_total_bytes;
    uint64_t device_free_bytes;
    int model_lib_loaded;
    int model_lib_generated;
    int model_lib_compile_attempted;
    int model_lib_compile_succeeded;
    uint32_t model_lib_abi_version;
    char model_lib_path[260];
    uint64_t hot_expert_count;
    uint64_t vram_expert_count;
    uint64_t ram_expert_count;
    uint64_t db_expert_count;
    uint64_t common_raw_tensor_count;
    uint64_t generation_started;
    uint64_t generation_completed;
    uint64_t generation_failed;
    uint32_t generation_active;
    uint32_t generation_token;
    uint32_t generation_layer;
    uint32_t generation_phase;
    glm5_kv_mode_t kv_mode;
} glm5_pc_engine_stats_t;

typedef struct {
    uint64_t queued_requests;
    uint64_t completed_requests;
    uint64_t failed_requests;
    uint64_t dropped_requests;
    uint64_t bytes_prefetched;
    uint64_t bytes_disk_to_ram;
    uint64_t bytes_ram_to_pinned;
    uint64_t bytes_pinned_to_gpu;
    uint64_t bytes_direct_to_gpu;
    uint64_t bytes_touch_fallback;
    uint64_t bytes_sync_h2d_fallback;
    uint64_t device_alloc_failures;
    uint64_t device_copy_failures;
    uint64_t disk_stage_done;
    uint64_t pinned_stage_done;
    uint64_t gpu_stage_done;
    uint64_t direct_stage_done;
    uint64_t staging_wraps;
    uint32_t staging_slot_deficit;
    uint64_t staging_slot_waits;
    uint64_t staging_slot_contention;
    uint64_t recommended_staging_bytes;
    uint64_t topology_predictions;
    uint64_t topology_updates;
    uint32_t active_workers;
    uint32_t disk_queue_depth;
    uint32_t pinned_queue_depth;
    uint32_t gpu_queue_depth;
    uint64_t bytes_disk_hint_issued;
    glm5_io_path_t active_path;
    int pinned_host_active;
    int directstorage_active;
    int gpudirect_storage_active;
    int bam_nvme_active;
} glm5_io_stats_t;
