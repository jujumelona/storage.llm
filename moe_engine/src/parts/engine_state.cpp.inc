#if defined(__linux__) && defined(MOE_HAVE_LIBURING) && MOE_HAVE_LIBURING && defined(MOE_ENABLE_LEGACY_URING_STATE)
#include <liburing.h>
struct io_uring_state_t {
    struct io_uring ring;
    std::vector<struct iovec> registered_iovecs;
    bool initialized = false;
};
#endif

static constexpr uint32_t moe_SOURCE_MMAP_FAST_SLOTS = 1024u;
static constexpr size_t moe_GATE_SNAPSHOT_MAX_LAYER = 10000u;
static constexpr size_t moe_GATE_SNAPSHOT_EPOCH_SLOTS = (moe_GATE_SNAPSHOT_MAX_LAYER + 1u) * 2u;

static std::atomic<uint64_t> g_moe_engine_generation{0};

struct moe_pc_engine {
    uint64_t engine_generation = 0;
    moe_pc_engine_config_t config{};
    moe_io_config_t io_config{};
    uint32_t disk_stage_depth_cap{2048};
    uint32_t pinned_stage_depth_cap{4096};
    uint32_t gpu_stage_depth_cap{6144};
    moe_io_atomic_stats io_atomic{};
    moe_backend_caps_t backend_caps{};
    moe_io_ring_adapter ring_adapter{};
    std::vector<moe_block_state> expert_states;
    std::vector<moe_tensor_record> tensors;
    std::vector<std::string> tensor_paths;
    std::vector<size_t> tensor_slots;
    std::vector<moe_gguf_common_tensor_record> gguf_common_tensors;
    std::unordered_map<std::string, size_t> gguf_common_tensor_slots;
    std::vector<uint64_t> expert_bytes;
    std::vector<uint16_t> expert_prefetch_priorities;
    std::vector<uint8_t> expert_residency_hints;
    uint64_t avg_expert_bytes = 0;
    uint64_t max_tensor_stream_bytes = 0;
    std::unordered_map<std::string, mmap_context_t*> shard_mmaps;
    std::unordered_map<uint64_t, mmap_context_t*> shard_mmaps_by_id;
    std::unordered_set<mmap_context_t*> locked_mmaps;
    std::atomic<mmap_context_t*> source_mmaps[moe_SOURCE_MMAP_FAST_SLOTS]{};
    std::string model_root;
    moe_scale4_file_t* scale4 = nullptr;
    moe_cuda_driver_api cuda{};
    moe_hip_driver_api hip{};
    moe_level_zero_driver_api ze{};
    moe_vulkan_driver_api vk{};
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
    std::unordered_map<uint64_t, moe_device_allocation> device_allocations;
    mutable std::shared_mutex topology_mutex;
    mutable std::mutex topology_queue_mutex;
    // BUGFIX 166: Use std::deque instead of std::vector to prevent iterator invalidation
    // std::vector::emplace_back() can trigger reallocation → all iterators invalidated
    // std::deque never invalidates iterators on push_back (except end())
    std::deque<std::pair<uint64_t, uint64_t>> topology_update_queue;
    mutable std::mutex done_mutex;
    std::condition_variable io_done_cv;
    moe_stage_queue disk_stage;
    moe_stage_queue pinned_stage;
    moe_stage_queue gpu_stage;

    // BUGFIX 32: Separate hot atomic variables into different cachelines ?�★??
    // Old: total_queue_depth, io_stop, worker_targets in consecutive memory ??false sharing
    // New: Group by access pattern, align each group to 64-byte cacheline
    // Group 1: Read-mostly by all workers (io_stop + total_queue_depth)
    alignas(64) std::atomic<bool> io_stop{false};
    std::atomic<uint32_t> total_queue_depth{0};

    std::vector<std::thread> io_workers;
    std::vector<std::thread> pinned_workers;
    std::vector<std::thread> gpu_workers;
    std::thread eviction_worker;

    // BUGFIX 32: Group 2: Written by orchestrator, read by workers (worker targets)
    alignas(64) std::atomic<uint32_t> disk_worker_target{0};
    std::atomic<uint32_t> pinned_worker_target{0};
    std::atomic<uint32_t> gpu_worker_target{0};
    std::atomic<uint32_t> runtime_prefetch_window_layers{0};

    // BUGFIX 32: Group 3: Runtime worker counts (read by orchestrator, written by lifecycle)
    alignas(64) std::atomic<uint32_t> runtime_disk_worker_threads{0};
    std::atomic<uint32_t> runtime_pinned_worker_threads{0};
    std::atomic<uint32_t> runtime_gpu_worker_threads{0};

    uint32_t worker_resize_pending_disk{0};
    uint32_t worker_resize_pending_pinned{0};
    uint32_t worker_resize_pending_gpu{0};
    int worker_resize_pending_direction{0};
    uint32_t worker_resize_streak{0};
    uint64_t last_dropped_requests_snapshot{0};

    // BUG 3 FIX: Prefetch window hysteresis (same pattern as worker resize) ?�★
    uint32_t prefetch_window_pending{0};
    uint32_t prefetch_window_streak{0};
    uint8_t* staging = nullptr;
    uint64_t staging_size = 0;
    bool staging_pinned = false;
    bool staging_numa_allocated = false;
    std::vector<moe_staging_slot> staging_slots;
    uint64_t staging_slot_size = 0;
#ifdef ENABLE_LOCKFREE_STAGING
    std::atomic<uint64_t> staging_slot_free_mask{0};
#endif
    std::mutex staging_slot_mutex;
    std::condition_variable staging_slot_cv;
    std::atomic<uint64_t> staging_generation{0};
    std::condition_variable staging_pinned_cv;
    std::condition_variable staging_gpu_cv;
    std::condition_variable staging_direct_cv;
    std::string topology_path;
    std::unordered_map<uint64_t, std::unordered_map<uint64_t, uint64_t>> topology_edges;
    std::mutex prefetch_plan_mutex;
    // PIPELINE FIX 9: Replace unordered_set with generation counter ?�★
    // Old: prefetch_plan_seen_set cleared every orchestrate (O(bucket_count))
    // New: generation counter + map (O(1) clear, bounded growth ~20k entries)
    uint64_t prefetch_plan_generation{0};
    std::unordered_map<uint64_t, uint64_t> prefetch_plan_seen_gen;  // key ??generation

    // Co-activation Speculative Prefetch statistics
    // [layer][expert_n_minus_2][expert_n] -> frequency
    // BUGFIX 문제9: Add mutex to protect co_activation data structures ★★★ CRITICAL
    std::shared_mutex co_activation_mutex;
    std::vector<std::unordered_map<int, std::unordered_map<int, float>>> co_activation_stats;
    std::vector<std::unordered_map<int, std::unordered_map<int, float>>> co_activation_stats_snapshot;
    std::vector<std::vector<int>> last_selected_experts;
    // BUGFIX 701: Sliding window token counters to bound co_activation_stats memory ★★
    // co_activation_token_count: total update calls since last stats reset
    // last_merge_build_token_count: snapshot at last moe_build_expert_merge_map
    // Stats are cleared after merge map rebuild so only post-rebuild data accumulates.
    uint64_t co_activation_token_count{0};
    uint64_t last_merge_build_token_count{0};
    moe_paged_kv_manager_t paged_kv_manager;
    moe_continuous_batch_scheduler_t batch_scheduler; // History of selected experts per layer
    std::mutex generation_api_mutex; // BUGFIX: Mutex to separate generation paths and protect KV manager
    moe_expert_merge_map expert_merge_map; // BUGFIX: Move merge map from global static to per-engine instance
    std::mutex gate_snapshot_mutex;
    std::vector<std::vector<float>> gate_input_snapshots;
    std::atomic<uint64_t> gate_input_snapshot_epochs[moe_GATE_SNAPSHOT_EPOCH_SLOTS]{};
    std::mutex gate_input_snapshot_mutex;
    std::atomic<uint32_t> adaptive_prefetch_depth{2};
    std::atomic<uint32_t> last_router_entropy_milli{0};
    mutable std::atomic<int> cached_gelu_activation_mode{-1};
    mutable std::atomic<int> cached_router_use_sigmoid{-1};
    mutable std::atomic<int> cached_router_norm_topk_prob{-1};
    mutable std::atomic<int> cached_router_top_k_hint{-1};
    mutable std::atomic<int> cached_router_group_count{-1};
    mutable std::atomic<int> cached_router_limited_group_count{-1};
    mutable std::atomic<uint32_t> cached_router_route_scale_bits{0};
    mutable std::atomic<uint32_t> cached_embedding_scale_hidden{0};
    mutable std::atomic<uint32_t> cached_embedding_scale_bits{0};
    alignas(64) std::atomic<uint64_t> vram_used_bytes{0};
    alignas(64) std::atomic<uint64_t> ram_used_bytes{0};
    alignas(64) std::atomic<uint64_t> db_used_bytes{0};

    // BUGFIX 문제4: Add KV cache VRAM budget tracking ★★★
    // Problem: VRAM budget shared between experts and KV, no separation
    // Solution: Split VRAM budget into expert and KV portions
    // Default: 80% experts, 20% KV (configurable)
    alignas(64) std::atomic<uint64_t> vram_kv_used_bytes{0};
    std::atomic<uint64_t> vram_kv_budget_bytes{0};  // Set during init

    // BUGFIX 문제8: Merge map rebuild trigger ★★★
    std::atomic<bool> merge_map_rebuild_pending{false};

    std::atomic<uint32_t> vram_expert_count{0};
    std::atomic<uint32_t> ram_expert_count{0};
    std::atomic<uint32_t> db_expert_count{0};
    std::atomic<bool> common_vram_upload_complete{false};
    std::atomic<uint64_t> common_vram_reserved_bytes{0};
    std::atomic<uint64_t> common_raw_prefetched_bytes{0};
    std::atomic<uint64_t> common_raw_prefetched_spans{0};
    // BUGFIX 33: Separate counter for queued vs actually transferred bytes ?�★??
    // Old: common_raw_prefetched_bytes mixed enqueue_only and actual transfer
    // New: common_raw_queued_bytes tracks enqueue_only, prefetched_bytes tracks actual transfer
    // Prevents stats confusion where "prefetched bytes" includes queued-but-not-transferred
    std::atomic<uint64_t> common_raw_queued_bytes{0};
    std::atomic<uint64_t> common_raw_queued_spans{0};
    std::atomic<uint64_t> device_allocated_bytes{0};
    std::atomic<uint64_t> device_allocation_count{0};
    std::atomic<uint64_t> device_fixed_bytes{0};
    std::atomic<uint64_t> device_activation_bytes{0};
    std::atomic<uint64_t> device_activation_count{0};
    std::atomic<uint64_t> device_expert_bytes{0};
    std::atomic<uint64_t> device_stream_bytes{0};
    std::atomic<uint64_t> device_kv_cache_bytes{0};  // BUGFIX 문제4: KV cache pool counter ★★★
    std::atomic<uint64_t> device_total_bytes{0};
    std::atomic<uint64_t> device_free_bytes{0};
    std::atomic<uint64_t> pending_ram_reservation{0};
    std::atomic<uint64_t> pending_vram_reservation{0}; // Fix 15: VRAM double admission prevention
    std::atomic<int64_t> last_mem_query_ms{0};  // Fix 41: per-engine cuMemGetInfo throttle
    std::atomic<int> device_allocator_ready{0};
    std::unordered_set<uint64_t> activation_keys;
    std::priority_queue<
        moe_device_evict_candidate,
        std::vector<moe_device_evict_candidate>,
        moe_device_evict_colder_first> activation_evict_heap;
    std::atomic<int> last_activation_layer{-1};
    std::atomic<uint64_t> expert_touch_epoch{0};
    std::priority_queue<moe_evict_candidate, std::vector<moe_evict_candidate>, moe_evict_colder_first> evict_heaps[3];
    std::mutex heap_mutexes[3];  // Per-tier evict heap locks: VRAM, RAM, DB.
    std::atomic<uint64_t> evict_heap_versions[3]{};
    std::atomic<bool> evict_heap_compact_needed{false};
    [[deprecated("DO NOT USE: causes deadlock. Use heap_mutexes[0/1/2] via evict_heap_mutex()")]]
    std::mutex heap_mutex_abi_compat_unused;  // DEAD LOCK: do not use. Use heap_mutexes[0/1/2] via evict_heap_mutex(); ABI layout only.
    std::atomic<bool> vram_eviction_in_progress{false};
    std::mutex vram_eviction_mutex;
    std::condition_variable vram_eviction_cv;
    std::atomic<bool> vram_eviction_needed{false};
    std::mutex ram_eviction_mutex;
    std::condition_variable ram_eviction_cv;
    std::atomic<bool> ram_eviction_needed{false};
    alignas(64) std::atomic<uint32_t> staging_free_slot_count{0};
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
    std::atomic<uint64_t> generation_started{0};
    std::atomic<uint64_t> generation_completed{0};
    std::atomic<uint64_t> generation_failed{0};
    std::atomic<uint32_t> generation_active{0};
    std::atomic<uint32_t> generation_token{0};
    std::atomic<uint32_t> generation_layer{0xffffffffu};
    std::atomic<uint32_t> generation_phase{moe_GEN_PHASE_IDLE};
    std::mutex forward_scratch_mutex;
    std::vector<float> lm_head_scores;
    std::atomic<int> model_root_valid{0};  // Fix 59: cache model root validation result
    std::atomic<int> offload_gguf_valid{0};
    std::atomic<uint32_t> offload_gguf_file_count{0};
    std::atomic<uint64_t> offload_gguf_tensor_count{0};
    std::atomic<uint64_t> offload_gguf_executable_tensor_count{0};
    std::atomic<int> qkv_forced_by_format{0};
    std::atomic<bool> offload_qkv_cfg_available{false};
    qkv_config_t offload_qkv_cfg{};
    std::atomic<uint32_t> offload_weight_quant_bits{0};
    std::atomic<uint32_t> offload_weight_encoding{0};
    std::atomic<uint32_t> offload_weight_block_size{0};
    std::atomic<uint32_t> offload_optimization_catalog_count{0};
    std::atomic<int> juju_strategy_valid{0};
    moe_juju_execution_path_t juju_execution_path{moe_JUJU_EXEC_STORAGE_STREAMING};
    moe_juju_pipeline_config_t juju_pipeline_config{};
    std::string offload_graph_ir_json;
    std::atomic<int> offload_graph_ir_ready{0};
    std::atomic<int> offload_graph_ir_required{0};
    std::atomic<uint32_t> offload_graph_ir_version{0};
    std::atomic<uint64_t> offload_graph_ir_op_count{0};
    std::atomic<uint64_t> offload_graph_ir_layer_count{0};
    std::atomic<uint32_t> juju_idx_schema_version{0};
    std::atomic<uint32_t> juju_split_group_count{0};
    std::atomic<uint32_t> juju_split_missing_count{0};
    std::atomic<uint64_t> juju_runtime_hint_tensor_count{0};
    std::atomic<uint64_t> juju_priority_tensor_count{0};
    std::atomic<uint64_t> juju_fastmem_tensor_count{0};
    std::atomic<uint64_t> juju_slowmem_tensor_count{0};
    std::string offload_gguf_first_file;
    std::string offload_gguf_metadata_json;
    std::string offload_weight_quant_family;
    std::string offload_weight_kernel_family;
    moe_model_shape_t dynamic_shape{};
    std::string dynamic_architecture;
    std::string dynamic_model_id;
    std::atomic<int> dynamic_shape_valid{0};
    moe_attn_config_t attn_config{};
    std::atomic<int> attn_config_valid{0};
    std::atomic<uint32_t> forward_adapter{moe_FORWARD_ADAPTER_NONE};
    std::atomic<int> forward_adapter_executable{0};
    mutable std::atomic<int> tokenizer_valid{-1};
    mutable std::atomic<int> attention_projector_valid{-1};  // Fix 72: -1=unchecked, 0=no, 1=yes
    std::vector<std::thread> deferred_joins;  // Fix 44: engine-owned deferred worker joins
    std::atomic<uint32_t> staging_pinned_slot_count{0};
    std::atomic<uint32_t> staging_gpu_slot_count{0};
    std::atomic<uint32_t> staging_direct_slot_count{0};
    mutable std::mutex raw_file_size_cache_mutex;
    mutable std::unordered_map<int32_t, uint64_t> raw_file_size_cache;
    mutable std::shared_mutex raw_tensor_cache_mutex;
    std::unordered_map<uint64_t, moe_raw_tensor_cache_entry> raw_tensor_cache;
    mutable std::shared_mutex common_tensor_suffix_cache_mutex;
    mutable std::unordered_map<std::string, int> common_tensor_suffix_cache;
    mutable std::shared_mutex text_mention_cache_mutex;
    mutable std::unordered_map<std::string, int> graph_ir_mention_cache;
    mutable std::unordered_map<std::string, int> metadata_mention_cache;
    std::vector<uint8_t> model_config_full_attention_layers;
    std::atomic<int> model_config_layer_types_valid{0};
    std::atomic<int> model_config_attention_valid{0};
    uint32_t model_config_num_attention_heads = 0;
    uint32_t model_config_num_key_value_heads = 0;
    uint32_t model_config_num_global_key_value_heads = 0;
    uint32_t model_config_head_dim = 0;
    uint32_t model_config_global_head_dim = 0;
    uint32_t model_config_sliding_window = 0;
    float model_config_sliding_rope_theta = 0.0f;
    float model_config_full_rope_theta = 0.0f;
    float model_config_partial_rotary_factor = 0.0f;

    // GAP 1A: Persistent per-head quantized KV cache
    // Key = head_index, value = qkv_state_t with quantized K/V for that head
    std::mutex qkv_mutex;  // protects qkv_cfg and head-state vector growth
    std::vector<qkv_state_t> qkv_head_state_slots;
    std::vector<uint8_t> qkv_head_state_valid;
    std::vector<std::unique_ptr<std::mutex>> qkv_head_state_locks;
    qkv_config_t qkv_cfg{};
    std::atomic<bool> qkv_cfg_initialized{false};
    uint32_t qkv_last_ctx = 0;  // Track context size for reallocation
    std::vector<CUstream> stream_pool;
    std::mutex stream_pool_mutex;
    std::atomic<uint32_t> stream_pool_available{0};
};

static moe_model_shape_t moe_engine_model_shape_or_static(const moe_pc_engine* engine) {
    moe_model_shape_t shape{};
    const moe_storage_constants_t* storage = moe_storage_constants();
    if (storage) {
        shape.num_hidden_layers = storage->num_hidden_layers;
        shape.first_moe_layer = storage->first_moe_layer;
        shape.last_moe_layer = storage->last_moe_layer;
        shape.experts_per_moe_layer = storage->experts_per_moe_layer;
        shape.hidden_size = storage->hidden_size;
        shape.expert_intermediate_size = storage->expert_intermediate_size;
        shape.vocab_size = storage->vocab_size;
        shape.projection_count = storage->expert_projection_count;
    }
    if (engine) {
        const moe_model_shape_t& dynamic = engine->dynamic_shape;
        if (dynamic.num_hidden_layers) shape.num_hidden_layers = dynamic.num_hidden_layers;
        if (dynamic.first_moe_layer || dynamic.last_moe_layer) shape.first_moe_layer = dynamic.first_moe_layer;
        if (dynamic.last_moe_layer) shape.last_moe_layer = dynamic.last_moe_layer;
        if (dynamic.experts_per_moe_layer) shape.experts_per_moe_layer = dynamic.experts_per_moe_layer;
        if (dynamic.hidden_size) shape.hidden_size = dynamic.hidden_size;
        if (dynamic.expert_intermediate_size) shape.expert_intermediate_size = dynamic.expert_intermediate_size;
        if (dynamic.vocab_size) shape.vocab_size = dynamic.vocab_size;
        if (dynamic.projection_count) shape.projection_count = dynamic.projection_count;
        if (shape.experts_per_moe_layer && shape.num_hidden_layers &&
            shape.first_moe_layer == 0 && shape.last_moe_layer == 0) {
            shape.last_moe_layer = shape.num_hidden_layers - 1u;
        }
    }
    return shape;
}

static moe_storage_constants_t moe_engine_storage_constants_or_static(const moe_pc_engine* engine) {
    moe_storage_constants_t out{};
    const moe_storage_constants_t* storage = moe_storage_constants();
    if (storage) {
        out = *storage;
    }
    const moe_model_shape_t shape = moe_engine_model_shape_or_static(engine);
    if (shape.num_hidden_layers) out.num_hidden_layers = shape.num_hidden_layers;
    if (shape.first_moe_layer || shape.last_moe_layer) out.first_moe_layer = shape.first_moe_layer;
    if (shape.last_moe_layer) out.last_moe_layer = shape.last_moe_layer;
    if (shape.experts_per_moe_layer) out.experts_per_moe_layer = shape.experts_per_moe_layer;
    if (shape.hidden_size) out.hidden_size = shape.hidden_size;
    if (shape.expert_intermediate_size) out.expert_intermediate_size = shape.expert_intermediate_size;
    if (shape.vocab_size) out.vocab_size = shape.vocab_size;
    if (shape.projection_count) out.expert_projection_count = shape.projection_count;
    if (out.num_hidden_layers && out.experts_per_moe_layer &&
        out.last_moe_layer >= out.first_moe_layer) {
        out.total_expert_count =
            (out.last_moe_layer - out.first_moe_layer + 1u) * out.experts_per_moe_layer;
    }
    return out;
}

static uint32_t runtime_disk_workers(const moe_pc_engine* engine) {
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

static uint32_t runtime_pinned_workers(const moe_pc_engine* engine) {
    if (!engine) {
        return 1u;
    }
    if (engine->backend_caps.supports_zero_copy_host ||
        !engine->io_config.enable_pinned_host ||
        !engine->io_config.enable_multistage_pipeline) {
        return 0u;
    }
    uint32_t workers = engine->runtime_pinned_worker_threads.load(std::memory_order_relaxed);
    if (workers == 0) {
        workers = engine->io_config.pinned_worker_threads;
    }
    return workers ? workers : 1u;
}

static uint32_t runtime_gpu_workers(const moe_pc_engine* engine) {
    if (!engine) {
        return 1u;
    }
    if (engine->backend_caps.supports_zero_copy_host ||
        !engine->config.prefer_gpu ||
        (engine->backend_caps.backend != moe_BACKEND_CUDA &&
         engine->backend_caps.backend != moe_BACKEND_HIP)) {
        return 0u;
    }
    uint32_t workers = engine->runtime_gpu_worker_threads.load(std::memory_order_relaxed);
    if (workers == 0) {
        workers = engine->io_config.gpu_worker_threads;
    }
    return workers ? workers : 1u;
}

static void runtime_store_io_workers(moe_pc_engine* engine, uint32_t disk, uint32_t pinned, uint32_t gpu) {
    if (!engine) {
        return;
    }
    engine->runtime_disk_worker_threads.store(disk, std::memory_order_relaxed);
    engine->runtime_pinned_worker_threads.store(pinned, std::memory_order_relaxed);
    engine->runtime_gpu_worker_threads.store(gpu, std::memory_order_relaxed);
}
