struct glm5_block_state {
    uint64_t bytes = 0;
    glm5_tier_t tier = GLM5_TIER_DB;
    double hot_score = 0.0;
    uint64_t last_touch_epoch = 0;
    bool active = false;
    bool pending_evict = false;
};

struct glm5_tensor_record {
    glm5_tensor_info_t info{};
    uint16_t resolved_path_index = 0xFFFF;
    const glm5_scale4_entry_t* scale4_entry = nullptr;
    uint64_t weight_row_bytes = 0;
    uint8_t weight_format = 0;
    uint8_t weight_bits = 0;
    uint8_t weight_block_size = 0;
    uint8_t scale_format = 0;
    uint8_t cpu_executable = 0;
    uint8_t gpu_executable = 0;
};

struct glm5_raw_tensor_cache_entry {
    const glm5_storage_raw_span_spec_t* span = nullptr;
    mmap_context_t* mapped = nullptr;
    const uint8_t* base = nullptr;
    uint64_t rel_offset = 0;
    uint64_t bytes = 0;
    uint32_t rows = 0;
    uint32_t cols = 0;
};

struct glm5_prefetch_task {
    size_t tensor_index = 0;
    glm5_tier_t target_tier = GLM5_TIER_RAM;
    glm5_transfer_stage_t stage = GLM5_TRANSFER_STAGE_DISK_TO_RAM;
    uint32_t staging_slot = UINT32_MAX;
    uint64_t staging_bytes = 0;
    int32_t raw_span_index = -1;
    uint8_t raw_pool = 0;
    int staging_valid = 0;
    int urgent = 0;
    uint8_t requeue_count = 0;
};

struct glm5_stage_queue {
    std::mutex mtx;
    std::condition_variable cv;
    std::condition_variable pop_cv;
    std::deque<glm5_prefetch_task> queue;
    std::deque<glm5_prefetch_task> urgent_queue;  // Fix 12: separate deque for O(1) urgent insertion
    std::atomic<uint32_t> depth{0};
    std::atomic<uint32_t> inflight{0};
};

struct glm5_staging_slot {
    uint8_t* buf = nullptr;
    uint64_t size = 0;
    bool pinned = false;
    std::atomic<bool> in_use{false};

    glm5_staging_slot() = default;
    glm5_staging_slot(const glm5_staging_slot&) = delete;
    glm5_staging_slot& operator=(const glm5_staging_slot&) = delete;
    glm5_staging_slot(glm5_staging_slot&& other) noexcept {
        buf = other.buf;
        size = other.size;
        in_use.store(other.in_use.load());
        other.buf = nullptr;
        other.size = 0;
        other.in_use.store(false);
    }
    glm5_staging_slot& operator=(glm5_staging_slot&& other) noexcept {
        buf = other.buf;
        size = other.size;
        in_use.store(other.in_use.load());
        other.buf = nullptr;
        other.size = 0;
        other.in_use.store(false);
        return *this;
    }
};

struct glm5_evict_candidate {
    size_t slot = SIZE_MAX;
    glm5_tier_t tier = GLM5_TIER_DB;
    uint64_t bytes = 0;
    double hot_score = 0.0;
    uint64_t last_touch_epoch = 0;
};

struct glm5_evict_colder_first {
    bool operator()(const glm5_evict_candidate& a, const glm5_evict_candidate& b) const {
        if (a.hot_score != b.hot_score) {
            return a.hot_score > b.hot_score;
        }
        return a.last_touch_epoch > b.last_touch_epoch;
    }
};

struct glm5_device_evict_candidate {
    uint64_t key = 0;
    uint64_t bytes = 0;
    uint64_t last_touch_epoch = 0;
};

struct glm5_device_evict_colder_first {
    bool operator()(const glm5_device_evict_candidate& a, const glm5_device_evict_candidate& b) const {
        return a.last_touch_epoch > b.last_touch_epoch;
    }
};
