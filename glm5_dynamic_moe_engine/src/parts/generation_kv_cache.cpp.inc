struct glm5_kv_cache {
    std::vector<uint16_t> data;  // bf16 storage: 2 bytes/entry (was 4)
    uint32_t num_layers = 0;
    uint32_t max_seq_len = 0;
    uint32_t seq_len = 0;
    uint64_t tracked_ram_bytes = 0;

    void init(uint32_t layers, bool pre_allocate = false) {
        num_layers = layers;
        max_seq_len = pre_allocate ? GLM5_MLA_MAX_SEQ_LEN : GLM5_MLA_KV_INITIAL_CHUNK;
        seq_len = 0;
        data.assign((size_t)layers * max_seq_len * GLM5_MLA_KV_ENTRY_DIM, 0u);
    }

    void ensure_capacity(uint32_t needed_pos) {
        if (needed_pos < max_seq_len) return;
        uint32_t new_cap = max_seq_len > 0 ? max_seq_len : 1u;
        while (new_cap <= needed_pos && new_cap < GLM5_MLA_MAX_SEQ_LEN) {
            new_cap *= 2u;
        }
        if (new_cap > GLM5_MLA_MAX_SEQ_LEN) new_cap = GLM5_MLA_MAX_SEQ_LEN;
        if (new_cap == max_seq_len) return;
        std::vector<uint16_t> new_data((size_t)num_layers * new_cap * GLM5_MLA_KV_ENTRY_DIM, 0u);
        for (uint32_t l = 0; l < num_layers; ++l) {
            const size_t old_off = (size_t)l * max_seq_len * GLM5_MLA_KV_ENTRY_DIM;
            const size_t new_off = (size_t)l * new_cap * GLM5_MLA_KV_ENTRY_DIM;
            const size_t copy_n = (size_t)max_seq_len * GLM5_MLA_KV_ENTRY_DIM;
            std::memcpy(new_data.data() + new_off, data.data() + old_off, copy_n * sizeof(uint16_t));
        }
        data = std::move(new_data);
        max_seq_len = new_cap;
    }

    void store_entry(uint32_t layer, uint32_t pos, const float* src, uint32_t count) {
        if (layer >= num_layers || pos >= max_seq_len || count > GLM5_MLA_KV_ENTRY_DIM) return;
        uint16_t* dst = data.data() + ((size_t)layer * max_seq_len + pos) * GLM5_MLA_KV_ENTRY_DIM;
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
            dst[i] = glm5_f32_to_bf16(src[i]);
        }
    }

    // Load bf16 entry to float32 buffer
    void load_entry(uint32_t layer, uint32_t pos, float* dst, uint32_t count) const {
        if (layer >= num_layers || pos >= max_seq_len || count > GLM5_MLA_KV_ENTRY_DIM) return;
        const uint16_t* src = data.data() + ((size_t)layer * max_seq_len + pos) * GLM5_MLA_KV_ENTRY_DIM;
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
            dst[i] = glm5_bf16_to_f32(src[i]);
        }
    }
};

static uint64_t glm5_kv_cache_bytes(uint32_t layers, uint32_t seq_len) {
    if (layers == 0 || seq_len == 0) {
        return 0;
    }
    constexpr uint64_t entry_bytes = (uint64_t)GLM5_MLA_KV_ENTRY_DIM * sizeof(uint16_t);
    if ((uint64_t)layers > UINT64_MAX / (uint64_t)seq_len ||
        (uint64_t)layers * (uint64_t)seq_len > UINT64_MAX / entry_bytes) {
        return UINT64_MAX;
    }
    return (uint64_t)layers * (uint64_t)seq_len * entry_bytes;
}

static uint64_t glm5_kv_cache_allocated_bytes(const glm5_kv_cache& cache) {
    if (cache.data.empty()) {
        return 0;
    }
    return glm5_kv_cache_bytes(cache.num_layers, cache.max_seq_len);
}

static void glm5_kv_cache_track_ram(glm5_pc_engine* engine, glm5_kv_cache* cache) {
    if (!engine || !cache) {
        return;
    }
    const uint64_t current = glm5_kv_cache_allocated_bytes(*cache);
    if (current > cache->tracked_ram_bytes) {
        add_tier_usage(engine, GLM5_TIER_RAM, current - cache->tracked_ram_bytes);
    } else if (cache->tracked_ram_bytes > current) {
        subtract_tier_usage(engine, GLM5_TIER_RAM, cache->tracked_ram_bytes - current);
    }
    cache->tracked_ram_bytes = current;
}

static void glm5_kv_cache_release_ram(glm5_pc_engine* engine, glm5_kv_cache* cache) {
    if (!engine || !cache || cache->tracked_ram_bytes == 0) {
        return;
    }
    subtract_tier_usage(engine, GLM5_TIER_RAM, cache->tracked_ram_bytes);
    cache->tracked_ram_bytes = 0;
}

struct glm5_kv_cache_ram_scope {
    glm5_pc_engine* engine = nullptr;
    glm5_kv_cache* cache = nullptr;
    glm5_kv_cache_ram_scope(glm5_pc_engine* e, glm5_kv_cache* c) : engine(e), cache(c) {}
    ~glm5_kv_cache_ram_scope() {
        glm5_kv_cache_release_ram(engine, cache);
    }
};

static int glm5_generation_should_preallocate_kv_cache(const glm5_pc_engine* engine, uint32_t layers) {
    const uint64_t full_kv_bytes = glm5_kv_cache_bytes(layers, GLM5_MLA_MAX_SEQ_LEN);
    if (!engine || full_kv_bytes == 0 || full_kv_bytes == UINT64_MAX) {
        return 0;
    }
    const uint64_t ram_budget = engine->config.ram_budget_bytes;
    if (ram_budget != 0) {
        const uint64_t ram_used = tier_used(engine, GLM5_TIER_RAM);
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

static int glm5_generation_should_use_layer_first_prefill(
    const glm5_pc_engine* engine,
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
