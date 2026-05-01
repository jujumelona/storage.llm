// BUGFIX 72: Issue 10 - Optimize KV cache layout to [pos][layer][dim] ?�★
// Problem: Old layout [layer][pos][dim] requires 79 separate memcpy during realloc
// Poor cache locality during attention decode (layers not contiguous)
// Solution: Change to [pos][layer][dim] layout
// - Single resize() for reallocation (layers become contiguous)
// - Better cache efficiency during attention decode
// - Single memcpy instead of 79 separate copies
// Impact: Index calculation changes in store_entry, load_entry
struct moe_kv_cache {  // KV Asymmetric Compression (KVQuant / KIVI)
    // Compresses MLA latent vectors from BF16 to INT8.
    // Each token/layer entry has 576 INT8 values + 1 FP32 scale + 1 FP32 zero.
    // Reduces memory from 1152 bytes to 584 bytes (~1.97x capacity).
    uint8_t* data = nullptr;
    float* scales = nullptr;
    float* zeros = nullptr;
    uint32_t num_layers = 0;
    uint32_t max_seq_len = 0;
    uint32_t seq_len = 0;
    uint64_t tracked_ram_bytes = 0;

    ~moe_kv_cache() {
        if (data) std::free(data);
        if (scales) std::free(scales);
        if (zeros) std::free(zeros);
    }
    moe_kv_cache() = default;
    moe_kv_cache(const moe_kv_cache&) = delete;
    moe_kv_cache& operator=(const moe_kv_cache&) = delete;
    moe_kv_cache(moe_kv_cache&& other) noexcept :
        data(other.data), scales(other.scales), zeros(other.zeros),
        num_layers(other.num_layers), max_seq_len(other.max_seq_len),
        seq_len(other.seq_len), tracked_ram_bytes(other.tracked_ram_bytes) {
        other.data = nullptr; other.scales = nullptr; other.zeros = nullptr;
    }
    moe_kv_cache& operator=(moe_kv_cache&& other) noexcept {
        if (this != &other) {
            if (data) std::free(data);
            if (scales) std::free(scales);
            if (zeros) std::free(zeros);
            data = other.data; scales = other.scales; zeros = other.zeros;
            num_layers = other.num_layers; max_seq_len = other.max_seq_len;
            seq_len = other.seq_len; tracked_ram_bytes = other.tracked_ram_bytes;
            other.data = nullptr; other.scales = nullptr; other.zeros = nullptr;
        }
        return *this;
    }

    void init(uint32_t layers, bool pre_allocate = false) {
        num_layers = layers;
        max_seq_len = pre_allocate ? moe_MLA_MAX_SEQ_LEN : moe_MLA_KV_INITIAL_CHUNK;
        seq_len = 0;
        if (data) std::free(data);
        if (scales) std::free(scales);
        if (zeros) std::free(zeros);
        const size_t total_entries = (size_t)max_seq_len * layers;
        data = (uint8_t*)std::calloc(total_entries * moe_MLA_KV_ENTRY_DIM, sizeof(uint8_t));
        scales = (float*)std::calloc(total_entries, sizeof(float));
        zeros = (float*)std::calloc(total_entries, sizeof(float));
    }

    void ensure_capacity(uint32_t needed_pos) {
        if (needed_pos < max_seq_len) return;
        uint32_t new_cap = needed_pos + 1;
        new_cap = (new_cap + 4095u) & ~4095u;
        if (new_cap > moe_MLA_MAX_SEQ_LEN) new_cap = moe_MLA_MAX_SEQ_LEN;
        if (new_cap == max_seq_len) return;

        const size_t old_entries = (size_t)max_seq_len * num_layers;
        const size_t new_entries = (size_t)new_cap * num_layers;

        uint8_t* new_data = (uint8_t*)std::realloc(data, new_entries * moe_MLA_KV_ENTRY_DIM);
        float* new_scales = (float*)std::realloc(scales, new_entries * sizeof(float));
        float* new_zeros = (float*)std::realloc(zeros, new_entries * sizeof(float));

        if (new_data && new_scales && new_zeros) {
            std::memset(new_data + old_entries * moe_MLA_KV_ENTRY_DIM, 0, (new_entries - old_entries) * moe_MLA_KV_ENTRY_DIM);
            std::memset(new_scales + old_entries, 0, (new_entries - old_entries) * sizeof(float));
            std::memset(new_zeros + old_entries, 0, (new_entries - old_entries) * sizeof(float));
            data = new_data;
            scales = new_scales;
            zeros = new_zeros;
            max_seq_len = new_cap;
        }
    }

    // Asymmetric INT8 Quantization (Store)
    void store_entry(uint32_t layer, uint32_t pos, const float* src, uint32_t count) {
        if (layer >= num_layers || pos >= max_seq_len || count > moe_MLA_KV_ENTRY_DIM) return;

        const size_t entry_idx = (size_t)pos * num_layers + layer;
        uint8_t* dst = data + entry_idx * moe_MLA_KV_ENTRY_DIM;

        // Find min and max
        float min_val = src[0];
        float max_val = src[0];
        for (uint32_t i = 1; i < count; ++i) {
            if (src[i] < min_val) min_val = src[i];
            if (src[i] > max_val) max_val = src[i];
        }

        // Calculate scale and zero point (Asymmetric [0, 255])
        float scale = (max_val - min_val) / 255.0f;
        if (scale < 1e-8f) scale = 1e-8f; // Avoid division by zero
        float zero = min_val;

        scales[entry_idx] = scale;
        zeros[entry_idx] = zero;

        const float inv_scale = 1.0f / scale;
        uint32_t i = 0;
#if defined(__AVX2__)
        __m256 v_inv_scale = _mm256_set1_ps(inv_scale);
        __m256 v_zero = _mm256_set1_ps(zero);
        for (; i + 7u < count; i += 8u) {
            __m256 v = _mm256_loadu_ps(src + i);
            v = _mm256_mul_ps(_mm256_sub_ps(v, v_zero), v_inv_scale);
            __m256i vi = _mm256_cvtps_epi32(v);
            // Clamp to [0, 255]
            vi = _mm256_max_epi32(vi, _mm256_setzero_si256());
            vi = _mm256_min_epi32(vi, _mm256_set1_epi32(255));

            // Pack 32-bit to 8-bit
            __m256i shuf = _mm256_shuffle_epi8(vi, _mm256_set_epi8(
                -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 12, 8, 4, 0,
                -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 12, 8, 4, 0
            ));
            __m128i lo = _mm256_castsi256_si128(shuf);
            __m128i hi = _mm256_extracti128_si256(shuf, 1);
            __m128i packed = _mm_unpacklo_epi32(lo, hi);
            _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + i), packed);
        }
#endif
        for (; i < count; ++i) {
            float val = (src[i] - zero) * inv_scale;
            val = std::max(0.0f, std::min(255.0f, std::round(val)));
            dst[i] = static_cast<uint8_t>(val);
        }
    }

    // Asymmetric INT8 Dequantization (Load)
    void load_entry(uint32_t layer, uint32_t pos, float* dst, uint32_t count) const {
        if (layer >= num_layers || pos >= max_seq_len || count > moe_MLA_KV_ENTRY_DIM) return;

        const size_t entry_idx = (size_t)pos * num_layers + layer;
        const uint8_t* src = data + entry_idx * moe_MLA_KV_ENTRY_DIM;
        const float scale = scales[entry_idx];
        const float zero = zeros[entry_idx];

        uint32_t i = 0;
#if defined(__AVX2__)
        __m256 v_scale = _mm256_set1_ps(scale);
        __m256 v_zero = _mm256_set1_ps(zero);
        for (; i + 7u < count; i += 8u) {
            __m128i packed = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + i));
            __m256i expanded = _mm256_cvtepu8_epi32(packed);
            __m256 v = _mm256_cvtepi32_ps(expanded);
            v = _mm256_add_ps(_mm256_mul_ps(v, v_scale), v_zero);
            _mm256_storeu_ps(dst + i, v);
        }
#endif
        for (; i < count; ++i) {
            dst[i] = static_cast<float>(src[i]) * scale + zero;
        }
    }
};

static uint64_t moe_kv_cache_bytes(uint32_t layers, uint32_t seq_len) {
    if (layers == 0 || seq_len == 0) {
        return 0;
    }
    // KV Asymmetric INT8: data (uint8_t) + scale (float) + zero (float)
    constexpr uint64_t entry_bytes = (uint64_t)moe_MLA_KV_ENTRY_DIM * sizeof(uint8_t) + 2 * sizeof(float);
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

