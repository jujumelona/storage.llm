struct glm5_pc_engine {
    glm5_pc_engine_config_t config{};
    glm5_io_config_t io_config{};
    glm5_io_atomic_stats io_atomic{};
    glm5_backend_caps_t backend_caps{};
    std::vector<glm5_block_state> expert_states;
    std::vector<glm5_tensor_record> tensors;
    std::vector<std::string> tensor_paths;
    std::vector<size_t> tensor_slots;
    std::vector<uint64_t> expert_bytes;
    uint64_t max_tensor_stream_bytes = 0;
    std::unordered_map<std::string, mmap_context_t*> shard_mmaps;
    std::unordered_map<uint64_t, mmap_context_t*> shard_mmaps_by_id;
    std::unordered_set<mmap_context_t*> locked_mmaps;
    std::atomic<mmap_context_t*> source_mmaps[85]{};
    std::string model_root;
    glm5_scale4_file_t* scale4 = nullptr;
    glm5_cuda_driver_api cuda{};
    void* model_lib_handle = nullptr;
    std::string model_lib_cache_dir;
    std::string model_lib_path;
    std::string model_lib_source_path;
    std::string model_lib_compile_command;
    int model_lib_loaded = 0;
    int model_lib_generated = 0;
    int model_lib_compile_attempted = 0;
    int model_lib_compile_succeeded = 0;
    uint32_t model_lib_abi_version = 0;

    mutable std::mutex map_mutex;       // protects expert_states
    mutable std::shared_mutex shard_mutex; // protects shard_mmaps; cached reads use shared_lock
    std::unordered_map<std::string, std::shared_ptr<std::condition_variable_any>> shard_mmap_cvs;
    mutable std::mutex device_mutex;
    mutable std::shared_mutex topology_mutex;
    mutable std::mutex topology_queue_mutex;
    std::vector<std::pair<uint64_t, uint64_t>> topology_update_queue;
    mutable std::mutex done_mutex;
    std::condition_variable io_done_cv;
    glm5_stage_queue disk_stage;
    glm5_stage_queue pinned_stage;
    glm5_stage_queue gpu_stage;
    std::atomic<uint32_t> total_queue_depth{0};
    std::vector<std::thread> io_workers;
    std::vector<std::thread> pinned_workers;
    std::vector<std::thread> gpu_workers;
    std::atomic<bool> io_stop{false};
    std::atomic<uint32_t> disk_worker_target{0};
    std::atomic<uint32_t> pinned_worker_target{0};
    std::atomic<uint32_t> gpu_worker_target{0};
    std::atomic<uint32_t> runtime_prefetch_window_layers{0};
    std::atomic<uint32_t> runtime_disk_worker_threads{0};
    std::atomic<uint32_t> runtime_pinned_worker_threads{0};
    std::atomic<uint32_t> runtime_gpu_worker_threads{0};
    uint32_t worker_resize_pending_disk{0};
    uint32_t worker_resize_pending_pinned{0};
    uint32_t worker_resize_pending_gpu{0};
    int worker_resize_pending_direction{0};
    uint32_t worker_resize_streak{0};
    uint8_t* staging = nullptr;
    uint64_t staging_size = 0;
    bool staging_pinned = false;
    std::vector<glm5_staging_slot> staging_slots;
    uint64_t staging_slot_size = 0;
    std::mutex staging_slot_mutex;
    std::condition_variable staging_slot_cv;
    std::string topology_path;
    std::unordered_map<uint64_t, std::unordered_map<uint64_t, uint64_t>> topology_edges;
    std::mutex prefetch_plan_mutex;
    std::unordered_set<uint64_t> prefetch_plan_seen_set;
    alignas(64) std::atomic<uint64_t> vram_used_bytes{0};
    alignas(64) std::atomic<uint64_t> ram_used_bytes{0};
    alignas(64) std::atomic<uint64_t> db_used_bytes{0};
    std::atomic<uint32_t> vram_expert_count{0};
    std::atomic<uint32_t> ram_expert_count{0};
    std::atomic<uint32_t> db_expert_count{0};
    std::atomic<bool> common_vram_upload_complete{false};
    std::atomic<uint64_t> common_vram_reserved_bytes{0};
    std::atomic<uint64_t> common_raw_prefetched_bytes{0};
    std::atomic<uint64_t> common_raw_prefetched_spans{0};
    std::atomic<uint64_t> device_allocated_bytes{0};
    std::atomic<uint64_t> device_allocation_count{0};
    std::atomic<uint64_t> device_fixed_bytes{0};
    std::atomic<uint64_t> device_activation_bytes{0};
    std::atomic<uint64_t> device_expert_bytes{0};
    std::atomic<uint64_t> device_stream_bytes{0};
    std::atomic<uint64_t> device_total_bytes{0};
    std::atomic<uint64_t> device_free_bytes{0};
    // Problem 3 Fix: Track external VRAM usage (other processes, CUDA JIT, NCCL, etc.)
    std::atomic<uint64_t> device_external_usage_estimate{0};
    std::atomic<int64_t> last_mem_query_ms{0};  // Fix 41: per-engine cuMemGetInfo throttle
    std::atomic<int> device_allocator_ready{0};
    std::unordered_map<uint64_t, glm5_device_allocation> device_allocations;
    std::unordered_set<uint64_t> activation_keys;
    std::priority_queue<
        glm5_device_evict_candidate,
        std::vector<glm5_device_evict_candidate>,
        glm5_device_evict_colder_first> activation_evict_heap;
    std::atomic<int> last_activation_layer{-1};
    std::atomic<uint64_t> expert_touch_epoch{0};
    std::priority_queue<glm5_evict_candidate, std::vector<glm5_evict_candidate>, glm5_evict_colder_first> evict_heaps[3];
    std::mutex heap_mutexes[3];  // Per-tier evict heap locks: VRAM, RAM, DB.
    std::atomic<uint64_t> evict_heap_versions[3]{};
    std::mutex heap_mutex_abi_compat_unused;  // DEAD LOCK: do not use. Use heap_mutexes[0/1/2] via evict_heap_mutex(); ABI layout only.
    std::atomic<bool> vram_eviction_in_progress{false};
    std::condition_variable vram_eviction_cv;
    alignas(64) std::atomic<uint32_t> staging_free_slot_count{0};
    std::atomic<bool> io_proactive_eviction_in_progress{false};
    std::mutex io_start_mutex;
    std::mutex cpu_row_mutex;
    std::condition_variable cpu_row_cv;
    std::condition_variable cpu_row_done_cv;
    std::vector<std::thread> cpu_row_workers;
    int (*cpu_row_fn)(void*, uint32_t) = nullptr;
    void* cpu_row_ctx = nullptr;
    std::atomic<uint32_t> cpu_next_row{0};
    std::atomic<uint32_t> cpu_active_workers{0};
    std::atomic<uint32_t> cpu_row_target_workers{0};
    std::atomic<int> cpu_rows_ok{1};
    uint32_t cpu_row_count = 0;
    uint32_t cpu_block_rows = 1;
    uint64_t cpu_row_generation = 0;
    uint64_t cpu_row_done_generation = 0;
    bool cpu_row_stop = false;
    std::atomic<int> cpu_row_active{0};  // Re-entry counter for parallel_rows
    std::mutex forward_scratch_mutex;
    std::vector<float> lm_head_scores;
    std::atomic<int> model_root_valid{0};  // Fix 59: cache model root validation result
    mutable std::atomic<int> tokenizer_valid{-1};
    mutable std::atomic<int> attention_projector_valid{-1};  // Fix 72: -1=unchecked, 0=no, 1=yes
    std::vector<std::thread> deferred_joins;  // Fix 44: engine-owned deferred worker joins
    std::atomic<uint32_t> staging_pinned_slot_count{0};
    mutable std::mutex raw_file_size_cache_mutex;
    mutable std::unordered_map<int32_t, uint64_t> raw_file_size_cache;
    mutable std::shared_mutex raw_tensor_cache_mutex;
    std::unordered_map<uint64_t, glm5_raw_tensor_cache_entry> raw_tensor_cache;

    // GAP 1A: Persistent per-head quantized KV cache
    // Key = head_index, value = qkv_state_t with quantized K/V for that head
    std::mutex qkv_mutex;  // protects qkv_cfg and head-state vector growth
    std::vector<qkv_state_t> qkv_head_state_slots;
    std::vector<uint8_t> qkv_head_state_valid;
    std::vector<std::unique_ptr<std::mutex>> qkv_head_state_locks;
    qkv_config_t qkv_cfg{};
    std::atomic<bool> qkv_cfg_initialized{false};
    uint32_t qkv_last_ctx = 0;  // Track context size for reallocation
};

static uint32_t runtime_disk_workers(const glm5_pc_engine* engine) {
    if (!engine) {
        return 1u;
    }
    uint32_t workers = engine->runtime_disk_worker_threads.load(std::memory_order_relaxed);
    if (workers == 0) {
        workers = engine->io_config.disk_worker_threads ?
            engine->io_config.disk_worker_threads : engine->io_config.io_worker_threads;
    }
    return workers ? workers : 1u;
}

static uint32_t runtime_pinned_workers(const glm5_pc_engine* engine) {
    if (!engine) {
        return 1u;
    }
    uint32_t workers = engine->runtime_pinned_worker_threads.load(std::memory_order_relaxed);
    if (workers == 0) {
        workers = engine->io_config.pinned_worker_threads;
    }
    return workers ? workers : 1u;
}

static uint32_t runtime_gpu_workers(const glm5_pc_engine* engine) {
    if (!engine) {
        return 1u;
    }
    uint32_t workers = engine->runtime_gpu_worker_threads.load(std::memory_order_relaxed);
    if (workers == 0) {
        workers = engine->io_config.gpu_worker_threads;
    }
    return workers ? workers : 1u;
}

static void runtime_store_io_workers(glm5_pc_engine* engine, uint32_t disk, uint32_t pinned, uint32_t gpu) {
    if (!engine) {
        return;
    }
    engine->runtime_disk_worker_threads.store(disk, std::memory_order_relaxed);
    engine->runtime_pinned_worker_threads.store(pinned, std::memory_order_relaxed);
    engine->runtime_gpu_worker_threads.store(gpu, std::memory_order_relaxed);
}
