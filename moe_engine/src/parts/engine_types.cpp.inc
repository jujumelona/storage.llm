struct moe_pc_engine;
static int validate_true_zero_copy_mmap(moe_pc_engine* engine, void* mapped_file_ptr, size_t bytes);

struct moe_block_state {
    uint64_t bytes = 0;
    moe_tier_t tier = moe_TIER_DB;
    double hot_score = 0.0;
    uint64_t last_touch_epoch = 0;
    uint64_t predicted_next_use_epoch = UINT64_MAX;
    uint16_t access_count = 0;  // ARC: frequency counter for T1/T2 classification
    // BUGFIX 196: Add avg_gap_ema for better next-use prediction ★★
    // Problem: predicted_next_use_epoch calculated from hot_score inverse
    // Doesn't distinguish periodic vs irregular access patterns
    // Solution: Track actual call interval EMA (exponential moving average)
    // Impact: More accurate eviction decisions, better cache hit rate
    uint64_t avg_gap_ema = 0;  // Average gap between accesses (epoch units)
    bool active = false;
    bool pending_evict = false;
};

enum moe_juju_residency_hint_internal : uint8_t {
    moe_JUJU_RESIDENCY_AUTO = 0,
    moe_JUJU_RESIDENCY_FAST_MEM = 1,
    moe_JUJU_RESIDENCY_FAST_MEM_STREAMABLE = 2,
    moe_JUJU_RESIDENCY_SLOW_MEM = 3
};

enum moe_juju_graph_role_internal : uint8_t {
    moe_JUJU_ROLE_UNKNOWN = 0,
    moe_JUJU_ROLE_SHARED_CORE = 1,
    moe_JUJU_ROLE_ATTENTION = 2,
    moe_JUJU_ROLE_ROUTER = 3,
    moe_JUJU_ROLE_EXPERT = 4,
    moe_JUJU_ROLE_DENSE_FFN = 5,
    moe_JUJU_ROLE_NORM = 6
};

enum moe_juju_prefetch_class_internal : uint8_t {
    moe_JUJU_PREFETCH_AUTO = 0,
    moe_JUJU_PREFETCH_STARTUP_HOT = 1,
    moe_JUJU_PREFETCH_LAYER_HOT = 2,
    moe_JUJU_PREFETCH_ROUTER_HOT = 3,
    moe_JUJU_PREFETCH_LAYER_WARM = 4,
    moe_JUJU_PREFETCH_EXPERT_STREAM = 5,
    moe_JUJU_PREFETCH_STREAM = 6
};

struct moe_tensor_record {
    moe_tensor_info_t info{};
    uint16_t resolved_path_index = 0xFFFF;
    const moe_scale4_entry_t* scale4_entry = nullptr;
    uint64_t weight_row_bytes = 0;
    uint8_t weight_format = 0;
    uint8_t weight_format_inferred = 0;
    uint8_t weight_bits = 0;
    uint8_t weight_block_size = 0;
    uint8_t scale_format = 0;
    uint8_t cpu_executable = 0;
    uint8_t gpu_executable = 0;
    uint16_t runtime_priority = 50;
    uint16_t prefetch_priority = 50;
    uint16_t execution_order = 500;
    uint16_t hotset_rank = 1000;
    uint8_t residency_hint = moe_JUJU_RESIDENCY_AUTO;
    uint8_t graph_role = moe_JUJU_ROLE_UNKNOWN;
    uint8_t prefetch_class = moe_JUJU_PREFETCH_AUTO;
    uint32_t juju_section_type = 0;
    uint8_t bundle_native = 0;
    uint8_t bundle_member_role = 0;
    uint32_t bundle_id = UINT32_MAX;
    uint64_t bundle_offset = 0;
    uint64_t bundle_size = 0;
    uint64_t bundle_member_offset = 0;
    uint64_t bundle_member_size = 0;
    uint64_t bundle_alignment = 0;
    uint8_t expert_gpu_layout_kind = 0;
    uint64_t expert_gpu_layout_offset = 0;
    uint64_t expert_gpu_layout_size = 0;
    uint64_t expert_gpu_layout_row_bytes = 0;
};

enum moe_raw_scalar_encoding_t : uint8_t {
    moe_RAW_SCALAR_BF16 = 0,
    moe_RAW_SCALAR_F16 = 1,
    moe_RAW_SCALAR_F32 = 2,
    moe_RAW_SCALAR_UNKNOWN = 255
};


// CPUStorageEngine persistent per-tensor row view.  Built once from JUJU/IDX
// tensor metadata so the token hot loop never re-parses names, JSON, row
// byte contracts or mmap bounds.  This is deliberately independent from the
// transient moe_tensor_row_fast_view_t used inside MLP scratch.
struct moe_cpu_hot_tensor_view_t {
    const moe_tensor_record* rec = nullptr;
    mmap_context_t* shard = nullptr;
    const uint8_t* base = nullptr;
    uint64_t shard_size = 0;
    uint64_t base_offset = 0;
    uint64_t row_bytes = 0;
    uint32_t rows = 0;
    uint32_t cols = 0;
    int weight_format = 0;
    uint8_t ready = 0;
    uint8_t simple_dot = 0;
};


// Common SSD/NVMe storage span used by CPUStorageEngine and GPU/offload
// prefetch.  Built from JUJU/IDX offsets once, then reused by token loops so
// the hot path does not scan tensors or issue hundreds of tiny random hints.
struct moe_storage_io_span_t {
    uint32_t view_index = UINT32_MAX;   // index into cpu_hot_tensor_views
    uint32_t layer = UINT32_MAX;
    uint64_t offset = 0;                // file offset relative to mmap shard
    uint64_t bytes = 0;
    uint8_t merged = 0;
};

// Backend-neutral storage prefetch request.  The compute path enqueues these
// and returns immediately; dedicated storage workers then execute the best
// profile-specific warmup path (io_uring/IORing when useful, otherwise
// fadvise/readahead/madvise/page-cache).
struct moe_storage_prefetch_request_t {
    mmap_context_t* shard = nullptr;
    const uint8_t* mapped_base = nullptr;
    uint64_t offset = 0;
    uint64_t bytes = 0;
    uint32_t layer = UINT32_MAX;
    uint32_t profile = 0;
    uint8_t gpu_consumer = 0;
};

// Per-(layer, expert) indirection into cpu_hot_tensor_views.  This removes the
// last selected-expert storage bottleneck: router top-k prefetch no longer scans
// every tensor/view for every selected expert.
struct moe_storage_expert_view_index_t {
    uint32_t layer = UINT32_MAX;
    uint32_t expert = UINT32_MAX;
    uint32_t begin = 0;
    uint32_t end = 0;
};

enum moe_storage_profile_kind : uint32_t {
    moe_STORAGE_PROFILE_AUTO = 0,
    moe_STORAGE_PROFILE_NVME = 1,
    moe_STORAGE_PROFILE_SATA_SSD = 2,
    moe_STORAGE_PROFILE_HDD = 3,
    moe_STORAGE_PROFILE_NETWORK = 4,
    moe_STORAGE_PROFILE_RAMDISK = 5
};


struct moe_gguf_common_tensor_record {
    std::string name;
    std::string op_role;
    std::string execution_op;
    std::string graph_role_text;
    std::string bundle_member_role;
    std::string codec_id;
    std::string codec_registry_version;
    std::string storage_layout;
    uint16_t resolved_path_index = 0xFFFF;
    uint32_t dims = 0;
    uint64_t shape[4]{};
    uint32_t source_dims = 0;
    uint64_t source_shape[4]{};
    int32_t execution_layer = INT32_MIN;
    uint32_t math_rows = 0;
    uint32_t math_cols = 0;
    uint8_t math_transposed = 0;
    uint32_t type = 0;
    uint8_t weight_encoding = moe_WEIGHT_ENCODING_UNKNOWN;
    uint64_t file_offset = 0;
    uint64_t bytes = 0;
    uint64_t weight_row_bytes = 0;
    uint16_t runtime_priority = 50;
    uint16_t prefetch_priority = 50;
    uint16_t execution_order = 500;
    uint16_t hotset_rank = 1000;
    uint8_t residency_hint = moe_JUJU_RESIDENCY_AUTO;
    uint8_t graph_role = moe_JUJU_ROLE_UNKNOWN;
    uint8_t prefetch_class = moe_JUJU_PREFETCH_AUTO;
};

enum moe_forward_adapter_kind : uint32_t {
    moe_FORWARD_ADAPTER_NONE = 0,
    moe_FORWARD_ADAPTER_RAW_STATIC = 1,
    moe_FORWARD_ADAPTER_GGUF_GENERIC = 2,
    moe_FORWARD_ADAPTER_GGUF_MOE = 3,
    moe_FORWARD_ADAPTER_GRAPH_IR = 4,
    // CPU-only execution must not be described as graph_ir.  The CPU storage
    // engine uses a prebuilt JUJU/IDX execution plan, row views, CPU quant
    // kernels and CPU prefetch hooks.
    moe_FORWARD_ADAPTER_CPU_STORAGE = 10,
    // Device/offload path marker for health/routing when a GPU-capable backend is active.
    moe_FORWARD_ADAPTER_GPU_OFFLOAD = 20
};

static const char* moe_forward_adapter_name(uint32_t adapter) {
    switch (adapter) {
        case moe_FORWARD_ADAPTER_RAW_STATIC: return "raw_static";
        case moe_FORWARD_ADAPTER_GGUF_GENERIC: return "gguf_generic";
        case moe_FORWARD_ADAPTER_GGUF_MOE: return "gguf_moe";
        case moe_FORWARD_ADAPTER_GRAPH_IR: return "graph_ir";
        case moe_FORWARD_ADAPTER_CPU_STORAGE: return "cpu_storage";
        case moe_FORWARD_ADAPTER_GPU_OFFLOAD: return "gpu_offload";
        default: return "none";
    }
}

struct moe_raw_tensor_cache_entry {
    const moe_storage_raw_span_spec_t* span = nullptr;
    mmap_context_t* mapped = nullptr;
    const uint8_t* base = nullptr;
    uint64_t rel_offset = 0;
    uint64_t bytes = 0;
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint8_t scalar_encoding = moe_RAW_SCALAR_BF16;
    uint8_t weight_format = moe_WEIGHT_ENCODING_RAW_FP16;
    uint64_t weight_row_bytes = 0;
    uint8_t math_transposed = 0;
    uint32_t physical_rows = 0;
    uint32_t physical_cols = 0;
};

struct moe_prefetch_task {
    size_t tensor_index = 0;
    moe_tier_t target_tier = moe_TIER_RAM;
    moe_transfer_stage_t stage = moe_TRANSFER_STAGE_DISK_TO_RAM;
    uint32_t staging_slot = UINT32_MAX;
    uint64_t staging_bytes = 0;
    uint64_t bundle_offsets[3]{};
    uint64_t bundle_bytes[3]{};
    int32_t bundle_layer = -1;
    int32_t bundle_expert = -1;
    int32_t raw_span_index = -1;
    uint8_t raw_pool = 0;
    int staging_valid = 0;
    int bundle_triplet = 0;
    int urgent = 0;
    int allow_sync_fallback = 0;
    moe_prefetch_reason_t prefetch_reason = moe_PREFETCH_REASON_LOOKAHEAD;
    uint8_t required_now = 0;
    uint8_t speculative = 1;
    uint8_t urgent_required = 0;
    uint8_t requeue_count = 0;
    // Bytes accounted to the currently popped stage item. This is intentionally
    // reset on enqueue and set by pop_stage_task so queue-pressure accounting
    // tracks both queued bytes and in-flight transfer bytes.
    uint64_t stage_accounted_bytes = 0;
    vram_reservation_token vram_token{};
};

struct moe_stage_queue {
    std::mutex mtx;
    std::condition_variable cv;
    std::condition_variable pop_cv;
    std::deque<moe_prefetch_task> queue;
    std::deque<moe_prefetch_task> urgent_queue;  // Fix 12: separate deque for O(1) urgent insertion
    std::atomic<uint32_t> depth{0};
    std::atomic<uint32_t> inflight{0};
    // Byte-level queue pressure. Count-only caps miss the real movement bottleneck
    // when a few large tensors/expert bundles occupy staging/VRAM copy bandwidth.
    std::atomic<uint64_t> queued_bytes{0};
    // Bytes already popped by workers but not yet completed. Without this,
    // admission sees an empty queue while DMA/disk/memcpy is still saturated.
    std::atomic<uint64_t> inflight_bytes{0};
};

struct moe_runtime_perf_window {
    uint64_t tokens = 0;
    uint64_t expert_hits = 0;
    uint64_t expert_misses = 0;
    uint64_t prefetch_used = 0;
    uint64_t prefetch_wasted = 0;
    uint64_t sync_h2d_fallbacks = 0;
    uint64_t disk_read_ns = 0;
    uint64_t h2d_copy_ns = 0;
    uint64_t gpu_idle_ns = 0;
    uint64_t staging_wait_ns = 0;
    uint64_t directstorage_queue_wait_ns = 0;
};

struct moe_io_path_perf {
    std::atomic<uint64_t> submitted{0};
    std::atomic<uint64_t> completed{0};
    std::atomic<uint64_t> bytes{0};
    std::atomic<uint64_t> wait_ns{0};
    std::atomic<uint64_t> fail_count{0};
};

static void moe_io_path_perf_submit(moe_io_path_perf& p, uint64_t bytes) {
    p.submitted.fetch_add(1, std::memory_order_relaxed);
    p.bytes.fetch_add(bytes, std::memory_order_relaxed);
}

static void moe_io_path_perf_complete(moe_io_path_perf& p, uint64_t wait_ns) {
    p.completed.fetch_add(1, std::memory_order_relaxed);
    p.wait_ns.fetch_add(wait_ns, std::memory_order_relaxed);
}

static void moe_io_path_perf_fail(moe_io_path_perf& p, uint64_t wait_ns) {
    p.fail_count.fetch_add(1, std::memory_order_relaxed);
    p.wait_ns.fetch_add(wait_ns, std::memory_order_relaxed);
}

struct moe_directstorage_device_pending {
    moe_dstorage_pending_copy io{};
    uint64_t keys[4]{};
    uint64_t bytes[4]{};
    vram_reservation_token tokens[4]{};
    bool mark_resident[4]{};
    uint8_t count = 0;
};

struct moe_directstorage_staging_pending {
    moe_dstorage_pending_copy io{};
    uint32_t staging_slot = UINT32_MAX;
    moe_prefetch_task next_task{};
    uint64_t expected_bytes = 0;
};

struct moe_staging_slot {
    std::atomic<uint8_t*> buf{nullptr};
    void* metal_buffer = nullptr; // BUGFIX 83: Issue 4 - Metal Zero-Copy path support
    uint64_t size = 0;
    bool pinned = false;
    bool numa_allocated = false;
    bool device_host_allocated = false;
    std::atomic<bool> in_use{false};
    // BUGFIX 912: Add generation field for timeout detection ★★ IMPORTANT
    uint32_t generation = 0;
    uint32_t io_generation = 0;
    // BUGFIX: Per-slot DMA fence for immediate release after cuMemcpyAsync
    // Acquirer must cuStreamWaitEvent on this before writing to buf
    CUevent last_copy_event = nullptr;
    CUstream last_copy_stream = nullptr;
    std::atomic<bool> last_copy_event_recorded{false};

    moe_staging_slot() = default;
    moe_staging_slot(const moe_staging_slot&) = delete;
    moe_staging_slot& operator=(const moe_staging_slot&) = delete;
    moe_staging_slot(moe_staging_slot&& other) noexcept {
        buf.store(other.buf.load(std::memory_order_relaxed), std::memory_order_relaxed);
        metal_buffer = other.metal_buffer;
        size = other.size;
        pinned = other.pinned;
        numa_allocated = other.numa_allocated;
        device_host_allocated = other.device_host_allocated;
        in_use.store(other.in_use.load());
        generation = other.generation;
        io_generation = other.io_generation;
        last_copy_event = other.last_copy_event;
        last_copy_stream = other.last_copy_stream;
        last_copy_event_recorded.store(
            other.last_copy_event_recorded.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
        other.buf.store(nullptr, std::memory_order_relaxed);
        other.metal_buffer = nullptr;
        other.size = 0;
        other.pinned = false;
        other.numa_allocated = false;
        other.device_host_allocated = false;
        other.in_use.store(false);
        other.generation = 0;
        other.io_generation = 0;
        other.last_copy_event = nullptr;
        other.last_copy_stream = nullptr;
        other.last_copy_event_recorded.store(false, std::memory_order_relaxed);
    }
    moe_staging_slot& operator=(moe_staging_slot&& other) noexcept {
        buf.store(other.buf.load(std::memory_order_relaxed), std::memory_order_relaxed);
        metal_buffer = other.metal_buffer;
        size = other.size;
        pinned = other.pinned;
        numa_allocated = other.numa_allocated;
        device_host_allocated = other.device_host_allocated;
        in_use.store(other.in_use.load());
        generation = other.generation;
        io_generation = other.io_generation;
        last_copy_event = other.last_copy_event;
        last_copy_stream = other.last_copy_stream;
        last_copy_event_recorded.store(
            other.last_copy_event_recorded.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
        other.buf.store(nullptr, std::memory_order_relaxed);
        other.metal_buffer = nullptr;
        other.size = 0;
        other.pinned = false;
        other.numa_allocated = false;
        other.device_host_allocated = false;
        other.in_use.store(false);
        other.generation = 0;
        other.io_generation = 0;
        other.last_copy_event = nullptr;
        other.last_copy_stream = nullptr;
        other.last_copy_event_recorded.store(false, std::memory_order_relaxed);
        return *this;
    }
};

struct moe_evict_candidate {
    size_t slot = SIZE_MAX;
    moe_tier_t tier = moe_TIER_DB;
    uint64_t bytes = 0;
    double hot_score = 0.0;
    uint64_t last_touch_epoch = 0;
    uint64_t predicted_next_use_epoch = UINT64_MAX;
};

struct moe_evict_colder_first {
    bool operator()(const moe_evict_candidate& a, const moe_evict_candidate& b) const {
        if (a.predicted_next_use_epoch != b.predicted_next_use_epoch) {
            return a.predicted_next_use_epoch < b.predicted_next_use_epoch;
        }
        if (a.hot_score != b.hot_score) {
            return a.hot_score > b.hot_score;
        }
        return a.last_touch_epoch > b.last_touch_epoch;
    }
};

struct moe_device_evict_candidate {
    uint64_t key = 0;
    uint64_t bytes = 0;
    uint64_t last_touch_epoch = 0;
};

struct moe_device_evict_colder_first {
    bool operator()(const moe_device_evict_candidate& a, const moe_device_evict_candidate& b) const {
        return a.last_touch_epoch > b.last_touch_epoch;
    }
};

struct moe_expert_merge_map {
    std::shared_mutex mtx;
    std::vector<std::vector<int>> merge_groups;   // [layer][expert] → group_id
    std::vector<std::unordered_map<int, std::vector<int>>> group_members; // [layer][group] → experts
    bool valid = false;
};
