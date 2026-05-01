// BUGFIX 72: Issue 10 - Optimize KV cache layout to [pos][layer][dim] ?�★
// Problem: Old layout [layer][pos][dim] requires 79 separate memcpy during realloc
// Poor cache locality during attention decode (layers not contiguous)
// Solution: Change to [pos][layer][dim] layout
// - Single resize() for reallocation (layers become contiguous)
// - Better cache efficiency during attention decode
// - Single memcpy instead of 79 separate copies
// Impact: Index calculation changes in store_entry, load_entry
struct moe_kv_cache {
    uint16_t* data = nullptr;  // bf16 storage: 2 bytes/entry
    uint32_t num_layers = 0;
    uint32_t max_seq_len = 0;
    uint32_t seq_len = 0;
    uint64_t tracked_ram_bytes = 0;

    ~moe_kv_cache() { if (data) std::free(data); }
    moe_kv_cache() = default;
    moe_kv_cache(const moe_kv_cache&) = delete;
    moe_kv_cache& operator=(const moe_kv_cache&) = delete;
    moe_kv_cache(moe_kv_cache&& other) noexcept : data(other.data), num_layers(other.num_layers), max_seq_len(other.max_seq_len), seq_len(other.seq_len), tracked_ram_bytes(other.tracked_ram_bytes) { other.data = nullptr; }
    moe_kv_cache& operator=(moe_kv_cache&& other) noexcept {
        if (this != &other) {
            if (data) std::free(data);
            data = other.data; num_layers = other.num_layers; max_seq_len = other.max_seq_len; seq_len = other.seq_len; tracked_ram_bytes = other.tracked_ram_bytes;
            other.data = nullptr;
        }
        return *this;
    }

    void init(uint32_t layers, bool pre_allocate = false) {
        num_layers = layers;
        max_seq_len = pre_allocate ? moe_MLA_MAX_SEQ_LEN : moe_MLA_KV_INITIAL_CHUNK;
        seq_len = 0;
        if (data) std::free(data);
        data = (uint16_t*)std::calloc((size_t)max_seq_len * layers * moe_MLA_KV_ENTRY_DIM, sizeof(uint16_t));
    }

    // BUGFIX 71: Issue 9 - Remove KV cache zero-init waste ★
    // BUGFIX 72: Issue 10 - Single memcpy realloc with [pos][layer][dim] layout ★
    // Problem: Zero-initializing entire buffer wastes 2.8GB per realloc
    // Old layout [layer][pos][dim] required 79 separate memcpy
    // Solution: Use [pos][layer][dim] layout + resize() without value-init
    // - Single memcpy for all old data (layers are now contiguous per position)
    // - No zero-init waste
    // - Better cache locality during attention decode
    //
    // TODO: Issue 10 - Further optimize large memcpy (2.8GB) ★
    // Current: Single memcpy of 2.8GB blocks generation for ~20-50ms
    // Options:
    // 1. Incremental realloc: Grow by smaller chunks (e.g., 512MB)
    // 2. Ring buffer: Avoid realloc entirely, wrap around at max_seq_len
    // 3. Async memcpy: Background thread copies old data while generation continues
    // Trade-off: Complexity vs latency reduction
    // Status: Documented, not implemented (requires architectural change)
    void ensure_capacity(uint32_t needed_pos) {
        if (needed_pos < max_seq_len) return;
#ifdef _WIN32
        // Windows: grow by smaller increments to bound copy cost
        uint32_t new_cap = max_seq_len + 256;
        while (new_cap <= needed_pos && new_cap < moe_MLA_MAX_SEQ_LEN) {
            new_cap += 256;
        }
#else
        // Linux: doubling is fine, mremap avoids copy
        uint32_t new_cap = max_seq_len > 0 ? max_seq_len : 1u;
        while (new_cap <= needed_pos && new_cap < moe_MLA_MAX_SEQ_LEN) {
            new_cap *= 2u;
        }
#endif
        if (new_cap > moe_MLA_MAX_SEQ_LEN) new_cap = moe_MLA_MAX_SEQ_LEN;
        if (new_cap == max_seq_len) return;
        const size_t new_total = (size_t)new_cap * num_layers * moe_MLA_KV_ENTRY_DIM;
        // BUGFIX 89: Issue 10 - Use realloc to avoid 2.8GB memcpy during KV cache resize
        // OS extends VM mappings via mremap (Linux) without physical copy when expanding.
        uint16_t* new_data = (uint16_t*)std::realloc(data, new_total * sizeof(uint16_t));
        if (new_data) {
            data = new_data;
            max_seq_len = new_cap;
        }
    }

    // BUGFIX 72: Issue 10 - Update index calculation for [pos][layer][dim] layout ?�★
    void store_entry(uint32_t layer, uint32_t pos, const float* src, uint32_t count) {
        if (layer >= num_layers || pos >= max_seq_len || count > moe_MLA_KV_ENTRY_DIM) return;
        // New layout: data[pos * num_layers * dim + layer * dim + d]
        uint16_t* dst = data + ((size_t)pos * num_layers + layer) * moe_MLA_KV_ENTRY_DIM;
        uint32_t i = 0;
#if defined(__AVX2__)
        for (; i + 7u < count; i += 8u) {
            __m256 v = _mm256_loadu_ps(src + i);
            __m256i vi = _mm256_castps_si256(v);
            __m256i shift16 = _mm256_srli_epi32(vi, 16);
            __m256i and1 = _mm256_and_si256(shift16, _mm256_set1_epi32(1));
            __m256i rounding = _mm256_add_epi32(_mm256_set1_epi32(0x7FFF), and1);
            __m256i rounded = _mm256_add_epi32(vi, rounding);
            __m256i shifted = _mm256_srli_epi32(rounded, 16);

            __m256i shuf_mask = _mm256_set_epi8(
                -1,-1,-1,-1, -1,-1,-1,-1,
                13,12, 9,8, 5,4, 1,0,
                -1,-1,-1,-1, -1,-1,-1,-1,
                13,12, 9,8, 5,4, 1,0
            );
            __m256i shuf = _mm256_shuffle_epi8(shifted, shuf_mask);
            __m128i lo = _mm256_castsi256_si128(shuf);
            __m128i hi = _mm256_extracti128_si256(shuf, 1);
            _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + i), lo);
            _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + i + 4), hi);
        }
#endif
        for (; i < count; ++i) {
            dst[i] = moe_f32_to_bf16(src[i]);
        }
    }

    // BUGFIX 72: Issue 10 - Update index calculation for [pos][layer][dim] layout ?�★
    // Load bf16 entry to float32 buffer
    void load_entry(uint32_t layer, uint32_t pos, float* dst, uint32_t count) const {
        if (layer >= num_layers || pos >= max_seq_len || count > moe_MLA_KV_ENTRY_DIM) return;
        // New layout: data[pos * num_layers * dim + layer * dim + d]
        const uint16_t* src = data + ((size_t)pos * num_layers + layer) * moe_MLA_KV_ENTRY_DIM;
        uint32_t i = 0;
#if defined(__AVX2__)
        for (; i + 7u < count; i += 8u) {
            __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
            __m256i expanded = _mm256_cvtepu16_epi32(packed);
            __m256 values = _mm256_castsi256_ps(_mm256_slli_epi32(expanded, 16));
            _mm256_storeu_ps(dst + i, values);
        }
#endif
        for (; i < count; ++i) {
            dst[i] = moe_bf16_to_f32(src[i]);
        }
    }
};

static uint64_t moe_kv_cache_bytes(uint32_t layers, uint32_t seq_len) {
    if (layers == 0 || seq_len == 0) {
        return 0;
    }
    constexpr uint64_t entry_bytes = (uint64_t)moe_MLA_KV_ENTRY_DIM * sizeof(uint16_t);
    // BUGFIX 72: [pos][layer][dim] layout - order doesn't affect total size
    if ((uint64_t)seq_len > UINT64_MAX / (uint64_t)layers ||
        (uint64_t)seq_len * (uint64_t)layers > UINT64_MAX / entry_bytes) {
        return UINT64_MAX;
    }
    return (uint64_t)seq_len * (uint64_t)layers * entry_bytes;
}

static uint64_t moe_kv_cache_allocated_bytes(const moe_kv_cache& cache) {
    if ((!cache.data)) {
        return 0;
    }
    return moe_kv_cache_bytes(cache.num_layers, cache.max_seq_len);
}

static void moe_kv_cache_track_ram(moe_pc_engine* engine, moe_kv_cache* cache) {
    if (!engine || !cache) {
        return;
    }
    const uint64_t current = moe_kv_cache_allocated_bytes(*cache);
    if (current > cache->tracked_ram_bytes) {
        add_tier_usage(engine, moe_TIER_RAM, current - cache->tracked_ram_bytes);
    } else if (cache->tracked_ram_bytes > current) {
        subtract_tier_usage(engine, moe_TIER_RAM, cache->tracked_ram_bytes - current);
    }
    cache->tracked_ram_bytes = current;
}

static void moe_kv_cache_release_ram(moe_pc_engine* engine, moe_kv_cache* cache) {
    if (!engine || !cache || cache->tracked_ram_bytes == 0) {
        return;
    }
    subtract_tier_usage(engine, moe_TIER_RAM, cache->tracked_ram_bytes);
    cache->tracked_ram_bytes = 0;
}

struct moe_kv_cache_ram_scope {
    moe_pc_engine* engine = nullptr;
    moe_kv_cache* cache = nullptr;
    moe_kv_cache_ram_scope(moe_pc_engine* e, moe_kv_cache* c) : engine(e), cache(c) {}
    ~moe_kv_cache_ram_scope() {
        moe_kv_cache_release_ram(engine, cache);
    }
};

static int moe_generation_should_preallocate_kv_cache(const moe_pc_engine* engine, uint32_t layers) {
    const uint64_t full_kv_bytes = moe_kv_cache_bytes(layers, moe_MLA_MAX_SEQ_LEN);
    if (!engine || full_kv_bytes == 0 || full_kv_bytes == UINT64_MAX) {
        return 0;
    }
    const uint64_t ram_budget = engine->config.ram_budget_bytes;
    if (ram_budget != 0) {
        const uint64_t ram_used = tier_used(engine, moe_TIER_RAM);
        if (ram_used >= ram_budget || full_kv_bytes > ram_budget - ram_used) {
            return 0;
        }
    }
    const uint64_t available = storagellm::available_ram_bytes();
    if (available != 0) {
        const uint64_t guard_bytes = std::max<uint64_t>(128ull * 1024ull * 1024ull, full_kv_bytes / 8ull);
        if (available <= guard_bytes || full_kv_bytes > available - guard_bytes) {
            return 0;
        }
    }
    return 1;
}

static int moe_generation_should_use_layer_first_prefill(
    const moe_pc_engine* engine,
    uint32_t prefill_count,
    uint32_t hidden_size
) {
    if (prefill_count <= 1u || hidden_size == 0) {
        return 0;
    }
    if ((uint64_t)prefill_count > UINT64_MAX / (uint64_t)hidden_size ||
        (uint64_t)prefill_count * (uint64_t)hidden_size > UINT64_MAX / sizeof(float)) {
        return 0;
    }
    const uint64_t batch_bytes = (uint64_t)prefill_count * (uint64_t)hidden_size * sizeof(float);
    const uint64_t small_batch = 64ull * 1024ull * 1024ull;
    if (batch_bytes <= small_batch) {
        return 1;
    }
    if (engine && engine->config.ram_budget_bytes != 0 && batch_bytes > engine->config.ram_budget_bytes / 8ull) {
        return 0;
    }
    const uint64_t available = storagellm::available_ram_bytes();
    if (available != 0 && batch_bytes > available / 8ull) {
        return 0;
    }
    return 1;
}

