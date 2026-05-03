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
    uint8_t residency_hint = moe_JUJU_RESIDENCY_AUTO;
    uint8_t graph_role = moe_JUJU_ROLE_UNKNOWN;
    uint8_t prefetch_class = moe_JUJU_PREFETCH_AUTO;
    uint8_t juju_section_type = 0;
};

enum moe_raw_scalar_encoding_t : uint8_t {
    moe_RAW_SCALAR_BF16 = 0,
    moe_RAW_SCALAR_F16 = 1,
    moe_RAW_SCALAR_F32 = 2,
    moe_RAW_SCALAR_UNKNOWN = 255
};

struct moe_gguf_common_tensor_record {
    std::string name;
    uint16_t resolved_path_index = 0xFFFF;
    uint32_t dims = 0;
    uint64_t shape[4]{};
    uint32_t type = 0;
    uint8_t weight_encoding = moe_WEIGHT_ENCODING_UNKNOWN;
    uint64_t file_offset = 0;
    uint64_t bytes = 0;
    uint16_t runtime_priority = 50;
    uint16_t prefetch_priority = 50;
    uint8_t residency_hint = moe_JUJU_RESIDENCY_AUTO;
    uint8_t graph_role = moe_JUJU_ROLE_UNKNOWN;
    uint8_t prefetch_class = moe_JUJU_PREFETCH_AUTO;
};

enum moe_forward_adapter_kind : uint32_t {
    moe_FORWARD_ADAPTER_NONE = 0,
    moe_FORWARD_ADAPTER_RAW_STATIC = 1,
    moe_FORWARD_ADAPTER_GGUF_GENERIC = 2,
    moe_FORWARD_ADAPTER_GGUF_MOE = 3,
    moe_FORWARD_ADAPTER_GRAPH_IR = 4
};

static const char* moe_forward_adapter_name(uint32_t adapter) {
    switch (adapter) {
        case moe_FORWARD_ADAPTER_RAW_STATIC: return "raw_static";
        case moe_FORWARD_ADAPTER_GGUF_GENERIC: return "gguf_generic";
        case moe_FORWARD_ADAPTER_GGUF_MOE: return "gguf_moe";
        case moe_FORWARD_ADAPTER_GRAPH_IR: return "graph_ir";
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
    uint8_t requeue_count = 0;
};

struct moe_stage_queue {
    std::mutex mtx;
    std::condition_variable cv;
    std::condition_variable pop_cv;
    std::deque<moe_prefetch_task> queue;
    std::deque<moe_prefetch_task> urgent_queue;  // Fix 12: separate deque for O(1) urgent insertion
    std::atomic<uint32_t> depth{0};
    std::atomic<uint32_t> inflight{0};
};

struct moe_staging_slot {
    uint8_t* buf = nullptr;
    void* metal_buffer = nullptr; // BUGFIX 83: Issue 4 - Metal Zero-Copy path support
    uint64_t size = 0;
    bool pinned = false;
    bool numa_allocated = false;
    std::atomic<bool> in_use{false};
    // BUGFIX: Per-slot DMA fence for immediate release after cuMemcpyAsync
    // Acquirer must cuStreamWaitEvent on this before writing to buf
    CUevent last_copy_event = nullptr;
    CUstream last_copy_stream = nullptr;

    moe_staging_slot() = default;
    moe_staging_slot(const moe_staging_slot&) = delete;
    moe_staging_slot& operator=(const moe_staging_slot&) = delete;
    moe_staging_slot(moe_staging_slot&& other) noexcept {
        buf = other.buf;
        metal_buffer = other.metal_buffer;
        size = other.size;
        pinned = other.pinned;
        numa_allocated = other.numa_allocated;
        in_use.store(other.in_use.load());
        last_copy_event = other.last_copy_event;
        last_copy_stream = other.last_copy_stream;
        other.buf = nullptr;
        other.metal_buffer = nullptr;
        other.size = 0;
        other.pinned = false;
        other.numa_allocated = false;
        other.in_use.store(false);
        other.last_copy_event = nullptr;
        other.last_copy_stream = nullptr;
    }
    moe_staging_slot& operator=(moe_staging_slot&& other) noexcept {
        buf = other.buf;
        metal_buffer = other.metal_buffer;
        size = other.size;
        pinned = other.pinned;
        numa_allocated = other.numa_allocated;
        in_use.store(other.in_use.load());
        last_copy_event = other.last_copy_event;
        last_copy_stream = other.last_copy_stream;
        other.buf = nullptr;
        other.metal_buffer = nullptr;
        other.size = 0;
        other.pinned = false;
        other.numa_allocated = false;
        other.in_use.store(false);
        other.last_copy_event = nullptr;
        other.last_copy_stream = nullptr;
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
