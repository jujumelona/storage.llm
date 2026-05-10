#if defined(__linux__) && defined(MOE_HAVE_LIBURING) && MOE_HAVE_LIBURING
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
    moe_directstorage_state directstorage{};
    std::atomic<bool> runtime_gpudirect_ready{false};
    std::atomic<bool> runtime_directstorage_staging_ready{false};
    std::atomic<bool> runtime_directstorage_gpu_ready{false};
    std::atomic<bool> true_zero_copy_validated{false};
    std::atomic<bool> juju_bundle_4k_aligned{false};
    std::atomic<bool> io_reconfiguring{false};
#if defined(__linux__) && defined(MOE_HAVE_LIBURING) && MOE_HAVE_LIBURING
    io_uring_state_t uring_state{};
#endif
    std::vector<moe_block_state> expert_states;
    std::vector<moe_tensor_record> tensors;
    std::vector<std::string> tensor_paths;
    std::vector<size_t> tensor_slots;
    std::vector<moe_gguf_common_tensor_record> gguf_common_tensors;
    std::unordered_map<std::string, size_t> gguf_common_tensor_slots;
    std::vector<uint64_t> expert_bytes;
    std::vector<uint16_t> expert_prefetch_priorities;
    std::vector<uint8_t> expert_residency_hints;
    std::vector<uint8_t> expert_segment_policies;
    std::vector<uint8_t> expert_partial_exec_flags;
    std::vector<uint16_t> expert_partial_accuracy_milli;
    std::vector<std::vector<int>> juju_buddy_expert_ids;
    uint64_t avg_expert_bytes = 0;
    std::atomic<uint64_t> max_tensor_stream_bytes{0};
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
    void* model_lib_expert_batch_f32 = nullptr;
    void* model_lib_expert_batch_contiguous_f32 = nullptr;
    void* model_lib_expert_batch_device_f32 = nullptr;
    void* model_lib_grouped_expert_triplets_f32 = nullptr;
    void* model_lib_grouped_expert_triplets_device_f32 = nullptr;
    void* model_lib_grouped_expert_triplets_indexed_f32 = nullptr;
    void* model_lib_grouped_expert_triplets_indexed_device_f32 = nullptr;
    void* model_lib_grouped_expert_triplets_indexed_device_f32_v2 = nullptr;
    void* model_lib_prefill_activation_acquire_f32 = nullptr;
    void* model_lib_prefill_activation_release_f32 = nullptr;
    void* model_lib_gpu_expert_scratch_wait_ns = nullptr;
    void* model_lib_gpu_expert_scratch_backpressure_count = nullptr;
    void* model_lib_fast_backend_mask = nullptr;
    const void* model_lib_l0_mmap_checksum_spv = nullptr;
    uint32_t model_lib_l0_mmap_checksum_spv_size = 0;
    const char* model_lib_l0_mmap_checksum_kernel_name = nullptr;
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
    // std::vector::emplace_back() can trigger reallocation ??all iterators invalidated
    // std::deque never invalidates iterators on push_back (except end())
    std::deque<std::pair<uint64_t, uint64_t>> topology_update_queue;
    mutable std::mutex done_mutex;
    std::condition_variable io_done_cv;
    moe_stage_queue disk_stage;
    moe_stage_queue pinned_stage;
    moe_stage_queue gpu_stage;

    // BUGFIX 32: Separate hot atomic variables into different cachelines ?占썩쁾??
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
    // BUGFIX 958: Use atomic for thread-safe access ?끸쁾
    // Problem: Plain uint64_t written from orchestration loop which can be called
    // from multiple threads ??data race (C++ UB). Value is always same (monotonic
    // snapshot of dropped_requests), but undefined behavior nonetheless.
    // Solution: std::atomic<uint64_t> with relaxed ordering (no perf impact).
    std::atomic<uint64_t> last_dropped_requests_snapshot{0};

    // BUG 3 FIX: Prefetch window hysteresis (same pattern as worker resize) ?占썩쁾
    // BUGFIX 1000: Make window hysteresis fields atomic ?끸쁾 DEFENSIVE
    // Problem: Non-atomic uint32_t accessed from fill_orchestrator_plan which can
    // be called from multiple threads ??data race per C++ standard (UB).
    // Solution: std::atomic<uint32_t> with relaxed ordering. Zero perf cost on x86.
    std::atomic<uint32_t> prefetch_window_pending{0};
    std::atomic<uint32_t> prefetch_window_streak{0};
    uint8_t* staging = nullptr;
    uint64_t staging_size = 0;
    bool staging_pinned = false;
    bool staging_numa_allocated = false;
    bool staging_device_host_allocated = false;
    std::vector<moe_staging_slot> staging_slots;
    uint64_t staging_slot_size = 0;
#ifdef ENABLE_LOCKFREE_STAGING
    std::atomic<uint64_t> staging_slot_free_mask{0};
#endif
    std::mutex staging_slot_mutex;
    std::condition_variable staging_slot_cv;
    std::atomic<uint64_t> staging_generation{0};
    std::atomic<bool> staging_reset_in_progress{false};
    // BUGFIX 949: Orphaned events from staging timeout ?끸쁾??
    // When reset_staging_slots times out, in-use slots' last_copy_event handles
    // are moved here instead of being destroyed (GPU workers may still reference them).
    // release_staging_slot drains this list to prevent unbounded growth.
    std::mutex orphaned_events_mutex;
    std::vector<CUevent> orphaned_events;
    std::condition_variable staging_pinned_cv;
    std::condition_variable staging_gpu_cv;
    std::condition_variable staging_direct_cv;
    std::atomic<bool> staging_reinit_needed{false};
    // BUGFIX 986: Atomic mirror for prefer_overlapped_io ?끸쁾 IMPORTANT
    // Problem: io_config.prefer_overlapped_io is plain int (public C ABI).
    // Multiple io_worker threads write it concurrently ??UB (data race on non-atomic int).
    // Solution: Internal atomic bool is the source of truth. Reads use this;
    // writes update both this AND io_config.prefer_overlapped_io for external callers.
    std::atomic<bool> prefer_overlapped_io_atomic{false};
    std::atomic<uint32_t> prefer_overlapped_io_generation{0};
    std::atomic<bool> overlapped_io_transition_done{false};
    std::atomic<bool> vlock_recovery_reset_pending{false};
    // BUGFIX Problem 3: Engine-wide VirtualLock recovery counters ?끸쁾??CRITICAL
    // Problem: thread_local counters ??each worker tracks independently
    // ??Recovery transition races with in-flight OVERLAPPED operations
    // Solution: Engine-wide atomic counters shared by all IO workers
    // Impact: Consistent recovery state, proper synchronization
    std::atomic<uint32_t> vlock_recovery_counter{0};
    std::atomic<uint32_t> vlock_fail_counter{0};

    std::string topology_path;
    std::unordered_map<uint64_t, std::unordered_map<uint64_t, uint64_t>> topology_edges;
    std::mutex prefetch_plan_mutex;
    // PIPELINE FIX 9: Replace unordered_set with generation counter ?占썩쁾
    // Old: prefetch_plan_seen_set cleared every orchestrate (O(bucket_count))
    // New: generation counter + map (O(1) clear, bounded growth ~20k entries)
    uint64_t prefetch_plan_generation{0};
    std::unordered_map<uint64_t, uint64_t> prefetch_plan_seen_gen;  // key ??generation

    // Co-activation Speculative Prefetch statistics
    // [layer][expert_n_minus_2][expert_n] -> frequency
    // BUGFIX 臾몄젣9: Add mutex to protect co_activation data structures ?끸쁾??CRITICAL
    std::shared_mutex co_activation_mutex;
    std::vector<std::unordered_map<int, std::unordered_map<int, float>>> co_activation_stats;
    std::vector<std::unordered_map<int, std::unordered_map<int, float>>> co_activation_stats_snapshot;
    std::vector<std::vector<int>> last_selected_experts;
    // BUGFIX 701: Sliding window token counters to bound co_activation_stats memory ?끸쁾
    // co_activation_token_count: total update calls since last stats reset
    // last_merge_build_token_count: snapshot at last moe_build_expert_merge_map
    // Stats are cleared after merge map rebuild so only post-rebuild data accumulates.
    uint64_t co_activation_token_count{0};
    uint64_t last_merge_build_token_count{0};
    moe_paged_kv_manager_t paged_kv_manager;
    moe_continuous_batch_scheduler_t batch_scheduler; // History of selected experts per layer
    // Serializes legacy single-request generation/eval only. Real continuous-batch
    // server path does not take this lock; it uses batch_scheduler state plus the
    // narrower shared_forward_io_mutex below.
    std::mutex generation_api_mutex;
    // Protects engine-global IO/forward resources still shared by batch requests:
    // staging slots, residency maps, prefetch queues, and backend stream/event pools.
    // This is intentionally narrower than generation_api_mutex and can later be
    // split into residency/prefetch/staging/backend locks without changing the
    // public generation API.
    std::mutex shared_forward_io_mutex;
    moe_expert_merge_map expert_merge_map; // BUGFIX: Move merge map from global static to per-engine instance
    std::mutex gate_snapshot_mutex;
    std::vector<std::vector<float>> gate_input_snapshots;
    std::atomic<uint64_t> gate_input_snapshot_epochs[moe_GATE_SNAPSHOT_EPOCH_SLOTS]{};
    std::mutex gate_input_snapshot_mutex;
    std::atomic<uint32_t> adaptive_prefetch_depth{2};
    std::atomic<uint32_t> last_router_entropy_milli{0};
    std::atomic<uint64_t> expert_cache_request_count{0};
    std::atomic<uint64_t> expert_cache_hit_count{0};
    std::atomic<uint64_t> expert_cache_vram_hit_count{0};
    std::atomic<uint64_t> expert_cache_ram_hit_count{0};
    std::atomic<uint64_t> expert_cache_miss_count{0};
    std::atomic<uint64_t> expert_cache_miss_bytes{0};
    std::atomic<uint64_t> expert_miss_latency_us_total{0};
    std::atomic<uint64_t> speculative_prefetch_planned_count{0};
    std::atomic<uint64_t> speculative_prefetch_planned_bytes{0};
    std::atomic<uint32_t> expert_hit_rate_milli{0};
    std::atomic<uint32_t> prefetch_waste_ratio_milli{0};
    std::atomic<uint32_t> gpu_copy_idle_milli{0};
    std::atomic<uint32_t> adaptive_prefetch_window_last{0};
    std::atomic<uint64_t> autotune_last_tick_ms{0};
    std::atomic<uint64_t> autotune_prev_tokens{0};
    std::atomic<uint64_t> autotune_prev_hits{0};
    std::atomic<uint64_t> autotune_prev_misses{0};
    std::atomic<uint64_t> autotune_prev_prefetch_planned{0};
    std::atomic<uint64_t> autotune_prev_prefetch_wasted{0};
    std::atomic<uint64_t> autotune_prev_sync_h2d{0};
    std::atomic<uint64_t> autotune_prev_directstorage_bytes{0};
    std::atomic<uint64_t> autotune_prev_directstorage_queue_wait{0};
    std::atomic<uint64_t> autotune_prev_h2d_bytes{0};
    std::atomic<uint64_t> autotune_prev_staging_waits{0};
    mutable std::atomic<int> cached_gelu_activation_mode{-1};
    mutable std::atomic<int> cached_router_use_sigmoid{-1};
    mutable std::atomic<int> cached_router_norm_topk_prob{-1};
    mutable std::atomic<int> cached_router_top_k_hint{-1};
    mutable std::atomic<int> cached_router_group_count{-1};
    mutable std::atomic<int> cached_router_limited_group_count{-1};
    mutable std::atomic<uint32_t> cached_router_route_scale_bits{0};
    mutable std::atomic<uint32_t> cached_embedding_scale_hidden{0};
    mutable std::atomic<uint32_t> cached_embedding_scale_bits{0};
    mutable std::atomic<int> cached_rmsnorm_unit_offset{-1};
    alignas(64) std::atomic<uint64_t> vram_used_bytes{0};
    alignas(64) std::atomic<uint64_t> vram_resident_bytes{0};
    alignas(64) std::atomic<uint64_t> vram_inflight_bytes{0};
    alignas(64) std::atomic<uint64_t> vram_kv_bytes{0};
    alignas(64) std::atomic<uint64_t> vram_common_bytes{0};
    alignas(64) std::atomic<uint64_t> ram_used_bytes{0};
    alignas(64) std::atomic<uint64_t> db_used_bytes{0};



    // BUGFIX 臾몄젣4: Add KV cache VRAM budget tracking ?끸쁾??
    // Problem: VRAM budget shared between experts and KV, no separation
    // Solution: Split VRAM budget into expert and KV portions
    // Default: 80% experts, 20% KV (configurable)
    std::atomic<uint64_t> vram_kv_budget_bytes{0};  // Set during init

    // BUGFIX Issue 4: Add per-engine warning timers ?끸쁾??
    // Problem: Static timers shared across all engines suppress warnings
    // Solution: Per-engine atomic timers for independent warning rate-limiting
    std::atomic<int64_t> last_budget_warn_ms{0};
    std::atomic<uint64_t> last_expert_budget_warn_ms{0};

    // BUGFIX 臾몄젣8: Merge map rebuild trigger ?끸쁾??
    std::atomic<bool> merge_map_rebuild_pending{false};

    std::atomic<uint32_t> vram_expert_count{0};
    std::atomic<uint32_t> ram_expert_count{0};
    std::atomic<uint32_t> db_expert_count{0};
    std::atomic<bool> common_vram_upload_complete{false};
    std::atomic<uint64_t> common_vram_reserved_bytes{0};
    std::atomic<uint64_t> common_raw_prefetched_bytes{0};
    std::atomic<uint64_t> common_raw_prefetched_spans{0};
    // BUGFIX 33: Separate counter for queued vs actually transferred bytes ?占썩쁾??
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
    std::atomic<uint64_t> device_kv_cache_bytes{0};
    std::atomic<bool> kv_actually_in_vram{false};
    std::mutex cufile_cache_mutex;
    std::unordered_map<moe_cufile_handle_key, void*, moe_cufile_handle_key_hash> cufile_file_handle_cache;
    std::unordered_map<uint64_t, uint64_t> cufile_buffer_registration_cache;
    std::mutex directstorage_pending_mutex;
    std::vector<moe_directstorage_device_pending> directstorage_pending_device_copies;
    std::mutex directstorage_staging_pending_mutex;
    std::vector<moe_directstorage_staging_pending> directstorage_pending_staging_copies;
    std::atomic<uint32_t> directstorage_pending_count{0};
    std::atomic<bool> directstorage_recreate_requested{false};
    std::atomic<bool> directstorage_submission_paused{false};
    std::atomic<bool> directstorage_completion_event_supported{false};
    std::atomic<uint64_t> directstorage_last_recreate_ms{0};
    std::atomic<uint32_t> directstorage_queue_pressure_windows{0};
    std::thread directstorage_completion_worker;
    std::mutex directstorage_gpu_submit_mutex;
    std::mutex directstorage_staging_submit_mutex;
    std::mutex gds_stream_cache_mutex;
    std::unordered_set<CUstream> gds_registered_streams;
    std::mutex gds_blocking_mutex;
    std::condition_variable gds_blocking_cv;
    std::deque<moe_gds_blocking_copy> gds_blocking_queue;
    std::vector<std::thread> gds_blocking_workers;
    std::atomic<uint32_t> gds_blocking_worker_target{0};
    std::atomic<uint32_t> gds_blocking_pending{0};
    std::atomic<uint64_t> gds_async_attempted{0};
    std::atomic<uint64_t> gds_async_submitted{0};
    std::atomic<uint64_t> gds_async_completed{0};
    std::atomic<uint64_t> gds_blocking_fallback_count{0};
    moe_io_path_perf gds_perf{};
    moe_io_path_perf ds_gpu_perf{};
    moe_io_path_perf ds_staging_perf{};
    moe_io_path_perf pinned_perf{};
    std::atomic<uint64_t> device_total_bytes{0};
    std::atomic<uint64_t> device_free_bytes{0};
    std::atomic<uint64_t> pending_ram_reservation{0};
    std::atomic<int64_t> last_mem_query_ms{0};  // Fix 41: per-engine cuMemGetInfo throttle
    std::atomic<int> device_allocator_ready{0};

    std::atomic<bool> lifecycle_terminated{false};
    std::atomic<bool> juju_co_activation_bank_initialized{false};
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
    std::atomic<bool> evict_heap_urgent_compact{false};
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
    std::atomic<uint32_t> staging_free_pinned_slots{0};
    std::atomic<uint32_t> staging_free_gpu_slots{0};
    std::atomic<uint32_t> staging_free_direct_slots{0};
    std::atomic<uint64_t> staging_allocated_bytes{0};
    std::mutex staging_resize_mutex;
    std::mutex staging_slot_released_mutex;
    std::condition_variable staging_slot_released_cv;
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
    // QKV-only invariant diagnostics. These counters must remain zero on a
    // clean QKV-only PPL/generation path except allocation_skips/evals.
    std::atomic<uint64_t> qkv_plain_kv_forbidden_touches{0};
    std::atomic<uint64_t> qkv_plain_kv_allocation_skips{0};
    std::atomic<uint64_t> qkv_plain_kv_readiness_touches{0};
    std::atomic<uint64_t> qkv_plain_kv_capacity_touches{0};
    std::atomic<uint64_t> qkv_plain_kv_store_touches{0};
    std::atomic<uint64_t> qkv_plain_kv_decode_cache_touches{0};
    std::atomic<uint64_t> qkv_ppl_qkv_only_evals{0};
    std::atomic<uint64_t> qkv_env_override_blocked{0};
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
    std::atomic<uint32_t> juju_runtime_execution_manifest_ready{0};
    std::atomic<uint32_t> juju_runtime_access_plan_ready{0};
    std::atomic<uint64_t> juju_runtime_access_hotset_count{0};
    std::atomic<uint64_t> juju_runtime_access_file_group_count{0};
    std::atomic<uint64_t> juju_runtime_executor_tensor_count{0};
    std::atomic<uint64_t> juju_runtime_layer_prefetch_plan_count{0};
    std::atomic<uint32_t> juju_kv_layout_contract_ready{0};
    std::atomic<uint64_t> juju_kv_layout_max_entry_dim{0};
    std::atomic<uint64_t> juju_kv_layout_page_size_tokens{0};
    std::atomic<uint32_t> juju_idx_schema_version{0};
    std::atomic<uint32_t> juju_split_group_count{0};
    std::atomic<uint32_t> juju_split_missing_count{0};
    std::atomic<uint64_t> juju_runtime_hint_tensor_count{0};
    std::atomic<uint64_t> juju_priority_tensor_count{0};
    std::atomic<uint64_t> juju_fastmem_tensor_count{0};
    std::atomic<uint64_t> juju_slowmem_tensor_count{0};
    std::atomic<uint32_t> juju_predictor_metadata_ready{0};
    std::atomic<uint32_t> juju_predictor_layer_count{0};
    std::atomic<uint64_t> juju_predictor_expert_tensor_count{0};
    std::atomic<uint32_t> juju_buddy_map_ready{0};
    std::atomic<uint32_t> juju_buddy_unit_count{0};
    std::atomic<uint64_t> juju_buddy_tensor_count{0};
    std::atomic<uint64_t> juju_buddy_explicit_edge_count{0};
    std::atomic<uint32_t> juju_tier_hint_ready{0};
    std::atomic<uint64_t> juju_tier_hot_expert_tensor_count{0};
    std::atomic<uint64_t> juju_tier_warm_expert_tensor_count{0};
    std::atomic<uint64_t> juju_tier_cold_expert_tensor_count{0};
    std::atomic<uint64_t> juju_partial_expert_tensor_count{0};
    std::atomic<uint32_t> juju_moe_layer_bitmask_ready{0};
    std::atomic<uint64_t> juju_moe_layer_count{0};
    std::vector<uint64_t> juju_moe_layer_bitmask_words;
    std::atomic<uint32_t> juju_io_section_hints_ready{0};
    std::atomic<uint32_t> juju_runtime_contract_io_policy_ready{0};
    std::atomic<uint32_t> juju_runtime_contract_mmap_allowed{1};
    std::atomic<uint32_t> juju_shared_mmap_friendly{1};
    std::atomic<uint32_t> juju_hot_mmap_friendly{1};
    std::atomic<uint32_t> juju_warm_mmap_friendly{1};
    std::atomic<uint32_t> juju_cold_mmap_friendly{1};
    std::atomic<uint32_t> juju_shared_sequential_block_size{0};
    std::atomic<uint32_t> juju_hot_sequential_block_size{0};
    std::atomic<uint32_t> juju_warm_sequential_block_size{0};
    std::atomic<uint32_t> juju_cold_sequential_block_size{0};
    std::atomic<uint32_t> juju_max_prefetch_distance{0};
    std::atomic<uint64_t> juju_shared_section_bytes{0};
    std::atomic<uint64_t> juju_hot_section_bytes{0};
    std::atomic<uint64_t> juju_warm_section_bytes{0};
    std::atomic<uint64_t> juju_cold_section_bytes{0};
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
    // CPU-only forward routing/state.  These are intentionally separate from
    // graph_ir readiness so CPU backends cannot silently fall back to the slow
    // graph interpreter when storage execution is available.
    std::atomic<int> cpu_only_runtime{0};
    std::atomic<int> cpu_storage_engine_ready{0};
    std::atomic<int> cpu_storage_execution_plan_ready{0};
    std::atomic<int> cpu_quant_registry_ready{0};
    std::atomic<int> graph_ir_used_for_forward{0};
    std::atomic<int> gpu_path_disabled_for_cpu_only{0};
    std::atomic<uint32_t> cpu_execution_plan_layers{0};
    std::atomic<uint64_t> cpu_execution_plan_tensors{0};
    std::atomic<uint64_t> cpu_execution_plan_expert_tensors{0};
    std::atomic<uint64_t> cpu_execution_plan_bytes{0};
    std::atomic<uint64_t> cpu_prefetch_scheduled{0};
    std::atomic<uint64_t> cpu_selected_expert_prefetch_scheduled{0};
    std::atomic<uint64_t> cpu_layer_begin_count{0};
// CPUStorageEngine hot tensor view cache.  Indexed 1:1 with engine->tensors.
// Filled once after JUJU/IDX parsing.  The MLP hot path copies these views
// into scratch instead of calling get_or_map_tensor_shard/mmap_get_size and
// re-validating row bounds on every token/layer/expert.
std::vector<moe_cpu_hot_tensor_view_t> cpu_hot_tensor_views;
std::atomic<int> cpu_hot_tensor_views_ready{0};
std::atomic<uint64_t> cpu_hot_tensor_view_count{0};
std::atomic<uint64_t> cpu_hot_tensor_simple_dot_count{0};
std::atomic<uint64_t> cpu_hot_tensor_bytes{0};
std::atomic<uint64_t> cpu_hot_tensor_reused{0};
// CPUStorageEngine layer-local hotset plan.  Built once from JUJU/IDX
// execution_layer + row/byte metadata.  The token loop then schedules layer
// warmups from these flat offsets without scanning tensors or parsing names.
std::vector<uint32_t> cpu_layer_hot_offsets;
std::vector<uint32_t> cpu_layer_hot_tensor_indices;
std::vector<uint32_t> cpu_layer_prefetch_epoch;
mutable std::mutex cpu_layer_prefetch_mutex;
std::atomic<uint32_t> cpu_prefetch_epoch{0};
std::atomic<int> cpu_layer_hot_plan_ready{0};
std::atomic<uint64_t> cpu_layer_hot_tensor_count{0};
std::atomic<uint64_t> cpu_layer_hot_prefetch_count{0};
std::atomic<uint64_t> cpu_layer_hot_prefetch_bytes{0};
std::atomic<uint64_t> cpu_layer_prefetch_duplicate_skips{0};

// Common storage I/O span plan.  This is shared by CPU-only and GPU/offload
// because SSD/NVMe optimization should happen below the compute backend.
std::vector<moe_storage_io_span_t> storage_io_spans;
std::vector<uint32_t> storage_layer_span_offsets;
std::vector<uint32_t> storage_layer_prefetch_epoch;
mutable std::mutex storage_io_prefetch_mutex;
std::atomic<int> storage_io_plan_ready{0};
std::atomic<uint32_t> storage_profile{moe_STORAGE_PROFILE_AUTO};
std::atomic<uint32_t> storage_prefetch_workers{0};
std::atomic<uint32_t> storage_prefetch_queue_depth{0};
std::atomic<uint64_t> storage_io_span_count{0};
std::atomic<uint64_t> storage_io_merged_span_count{0};
std::atomic<uint64_t> storage_io_prefetch_count{0};
std::atomic<uint64_t> storage_io_prefetch_bytes{0};
std::atomic<uint64_t> storage_io_fadvise_count{0};
std::atomic<uint64_t> storage_io_readahead_count{0};
std::atomic<uint64_t> storage_io_madvise_count{0};
std::atomic<uint64_t> storage_io_page_touch_count{0};
std::atomic<uint64_t> storage_io_duplicate_skips{0};
// Storage-required common backend counters.  The span plan is backend-neutral;
// these counters show which consumer path actually used it.
std::atomic<int> storage_iouring_available{0};
std::atomic<int> storage_iouring_enabled{0};
std::atomic<uint64_t> storage_iouring_submit_count{0};
std::atomic<uint64_t> storage_iouring_bytes{0};
std::atomic<uint64_t> storage_iouring_fallback_count{0};
// Dedicated storage prefetch workers.  These keep blocking io_uring/IORing,
// readahead and optional page-touch work off the token compute thread.
std::deque<moe_storage_prefetch_request_t> storage_prefetch_queue;
std::vector<std::thread> storage_prefetch_worker_threads;
mutable std::mutex storage_prefetch_queue_mutex;
std::condition_variable storage_prefetch_queue_cv;
std::atomic<int> storage_prefetch_stop{0};
std::atomic<int> storage_async_prefetch_enabled{0};
std::atomic<uint64_t> storage_async_prefetch_enqueued{0};
std::atomic<uint64_t> storage_async_prefetch_completed{0};
std::atomic<uint64_t> storage_async_prefetch_dropped{0};
std::atomic<uint64_t> storage_async_prefetch_inline{0};
std::atomic<uint64_t> storage_async_prefetch_pending{0};
std::atomic<int> storage_gpu_staging_plan_ready{0};
std::atomic<uint64_t> storage_gpu_staging_hook_count{0};
std::atomic<uint64_t> storage_gpu_staging_bytes{0};
std::atomic<uint64_t> storage_cpu_pagecache_hook_count{0};
std::atomic<uint64_t> storage_cpu_pagecache_bytes{0};
// Router-selected expert SSD/RAM prefetch index.  storage_expert_view_offsets
// is sized layers * (experts_per_layer + 1).  For (layer, expert), the views
// are storage_expert_view_indices[offsets[layer*(E+1)+expert] .. offsets[...+1]).
std::vector<uint32_t> storage_expert_view_offsets;
std::vector<uint32_t> storage_expert_view_indices;
std::atomic<int> storage_expert_index_ready{0};
std::atomic<uint64_t> storage_expert_index_entry_count{0};
std::atomic<uint64_t> storage_expert_index_view_count{0};
std::atomic<uint64_t> storage_selected_expert_index_hits{0};
std::atomic<uint64_t> storage_selected_expert_linear_fallbacks{0};
std::atomic<uint64_t> storage_selected_expert_prefetch_bytes{0};
// Per-token duplicate suppression for router-selected expert prefetch.
// A token can ask for the same (layer, expert) through multiple hooks; do not
// repeat fadvise/readahead/madvise for the same storage bucket in one epoch.
std::vector<uint32_t> storage_expert_prefetch_epoch;
mutable std::mutex storage_expert_prefetch_mutex;
std::atomic<uint64_t> storage_selected_expert_duplicate_skips{0};
std::atomic<int> storage_format_plan_available{0};

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
    mutable std::shared_mutex rmsnorm_unit_offset_cache_mutex;
    mutable std::unordered_map<uint64_t, int> rmsnorm_unit_offset_cache;
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
    float model_config_query_pre_attn_scalar = 0.0f;
    float model_config_attention_scale = 0.0f;
    std::atomic<int> juju_attention_contract_ready{0};
    std::atomic<int> juju_exact_ppl_contract_ready{0};
    std::atomic<int> juju_required_features_contract_ready{0};
    float model_config_sliding_rope_theta = 0.0f;
    float model_config_full_rope_theta = 0.0f;
    float model_config_partial_rotary_factor = 0.0f;

    // GAP 1A: Persistent per-layer/head quantized KV cache.
    // Key = layer * kv_heads + head. Head-only reuse corrupts KV across layers.
    std::mutex qkv_mutex;  // protects qkv_cfg and head-state vector growth
    std::vector<qkv_state_t> qkv_head_state_slots;
    std::vector<uint8_t> qkv_head_state_valid;
    std::vector<std::unique_ptr<std::mutex>> qkv_head_state_locks;
    std::vector<uint32_t> qkv_head_state_layers;
    std::vector<uint32_t> qkv_head_state_heads;
    std::vector<uint64_t> qkv_head_state_fingerprints;
    std::vector<uint64_t> qkv_head_state_epochs;
    qkv_config_t qkv_cfg{};
    std::atomic<bool> qkv_cfg_initialized{false};
    uint32_t qkv_last_ctx = 0;  // Track context size for reallocation
    uint64_t qkv_context_epoch = 1;
    std::atomic<uint32_t> eval_plain_kv_active{0};
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
        uint64_t moe_layer_count = (uint64_t)out.last_moe_layer - (uint64_t)out.first_moe_layer + 1ull;
        if (engine &&
            engine->juju_moe_layer_bitmask_ready.load(std::memory_order_acquire) &&
            engine->juju_moe_layer_count.load(std::memory_order_relaxed) > 0) {
            moe_layer_count = engine->juju_moe_layer_count.load(std::memory_order_relaxed);
        }
        out.total_expert_count = moe_layer_count * out.experts_per_moe_layer;
    }
    return out;
}

static int moe_layer_is_moe_runtime(
    const moe_pc_engine* engine,
    const moe_storage_constants_t* c,
    int32_t layer
) {
    if (layer < 0) {
        return 0;
    }
    if (engine &&
        engine->juju_moe_layer_bitmask_ready.load(std::memory_order_acquire) &&
        !engine->juju_moe_layer_bitmask_words.empty()) {
        const size_t word_index = (size_t)layer / 64u;
        const uint32_t bit_index = (uint32_t)layer & 63u;
        if (word_index < engine->juju_moe_layer_bitmask_words.size()) {
            return ((engine->juju_moe_layer_bitmask_words[word_index] >> bit_index) & 1ull) ? 1 : 0;
        }
        return 0;
    }
    return c && layer >= (int32_t)c->first_moe_layer && layer <= (int32_t)c->last_moe_layer;
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
    if (engine->true_zero_copy_validated.load(std::memory_order_acquire) ||
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
    if (engine->true_zero_copy_validated.load(std::memory_order_acquire) ||
        !engine->config.prefer_gpu) {
        return 0u;
    }
    const bool upload_backend =
        engine->backend_caps.backend == moe_BACKEND_CUDA ||
        engine->backend_caps.backend == moe_BACKEND_HIP ||
        (engine->backend_caps.backend == moe_BACKEND_DIRECTML &&
         engine->runtime_directstorage_gpu_ready.load(std::memory_order_acquire));
    if (!upload_backend) {
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
