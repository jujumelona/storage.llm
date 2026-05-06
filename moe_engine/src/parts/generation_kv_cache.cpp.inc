// BUGFIX 72: Issue 10 - Optimize KV cache layout to [pos][layer][dim] ?끸쁾
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

    moe_pc_engine_t* engine = nullptr;
    uint32_t num_layers = 0;
    uint32_t max_seq_len = 0;
    uint32_t seq_len = 0;
    uint32_t entry_dim = moe_MLA_KV_ENTRY_DIM;
    uint32_t seq_limit = moe_MLA_MAX_SEQ_LEN;
    uint64_t tracked_ram_bytes = 0;

    ~moe_kv_cache() {
        // BUGFIX 879: Release RAM tier usage before freeing memory ★★★ CRITICAL
        // Problem: Destructor frees data/scales/zeros but doesn't call subtract_tier_usage.
        // Engine's tier usage counter remains elevated, causing premature RAM admission rejection.
        // This occurs during batch request cancellation or exception paths.
        // Solution: Call subtract_tier_usage before freeing memory.
        // Impact: Prevents RAM usage counter leak, enables correct tensor loading decisions.
        if (engine && tracked_ram_bytes > 0) {
            subtract_tier_usage(engine, moe_TIER_RAM, tracked_ram_bytes);
            tracked_ram_bytes = 0;
        }
        if (data) std::free(data);
        if (scales) std::free(scales);
        if (zeros) std::free(zeros);
    }
    moe_kv_cache() = default;
    moe_kv_cache(const moe_kv_cache&) = delete;
    moe_kv_cache& operator=(const moe_kv_cache&) = delete;
    moe_kv_cache(moe_kv_cache&& other) noexcept :
        data(other.data), scales(other.scales), zeros(other.zeros),
        engine(other.engine),
        num_layers(other.num_layers), max_seq_len(other.max_seq_len),
        seq_len(other.seq_len), entry_dim(other.entry_dim), seq_limit(other.seq_limit),
        tracked_ram_bytes(other.tracked_ram_bytes) {
        other.data = nullptr; other.scales = nullptr; other.zeros = nullptr;
        other.engine = nullptr;
        other.tracked_ram_bytes = 0;
    }
    moe_kv_cache& operator=(moe_kv_cache&& other) noexcept {
        if (this != &other) {
            if (engine && tracked_ram_bytes > 0) {
                subtract_tier_usage(engine, moe_TIER_RAM, tracked_ram_bytes);
                tracked_ram_bytes = 0;
            }
            if (data) std::free(data);
            if (scales) std::free(scales);
            if (zeros) std::free(zeros);

            data = other.data; scales = other.scales; zeros = other.zeros;
            engine = other.engine;
            num_layers = other.num_layers; max_seq_len = other.max_seq_len;
            seq_len = other.seq_len; entry_dim = other.entry_dim; seq_limit = other.seq_limit;
            tracked_ram_bytes = other.tracked_ram_bytes;

            other.data = nullptr; other.scales = nullptr; other.zeros = nullptr;
            other.engine = nullptr;
            other.tracked_ram_bytes = 0;
        }
        return *this;
    }

    void init(uint32_t layers, uint32_t entry_dim_in, uint32_t seq_limit_in, bool pre_allocate = false) {
        num_layers = layers;
        entry_dim = entry_dim_in ? entry_dim_in : moe_MLA_KV_ENTRY_DIM;
        seq_limit = seq_limit_in ? seq_limit_in : moe_MLA_MAX_SEQ_LEN;
        max_seq_len = pre_allocate ? seq_limit : std::min<uint32_t>(moe_MLA_KV_INITIAL_CHUNK, seq_limit);
        if (max_seq_len == 0) {
            max_seq_len = 1;
        }
        seq_len = 0;
        auto kv_bytes_for = [&](uint32_t cap) -> uint64_t {
            if (layers == 0 || cap == 0 || entry_dim == 0) {
                return 0;
            }
            if ((uint64_t)cap > UINT64_MAX / (uint64_t)layers) {
                return UINT64_MAX;
            }
            const uint64_t entries = (uint64_t)cap * (uint64_t)layers;
            const uint64_t entry_bytes = (uint64_t)entry_dim + 2ull * sizeof(float);
            return entries > UINT64_MAX / entry_bytes ? UINT64_MAX : entries * entry_bytes;
        };
        if (pre_allocate) {
            const uint64_t want = kv_bytes_for(max_seq_len);
            const uint64_t avail = storagellm::available_ram_bytes();
            const uint64_t guard = std::max<uint64_t>(256ull * 1024ull * 1024ull, want / 8ull);
            if (want == UINT64_MAX || (avail != 0 && (avail <= guard || want > avail - guard))) {
                max_seq_len = std::min<uint32_t>(moe_MLA_KV_INITIAL_CHUNK, seq_limit);
                if (max_seq_len == 0) max_seq_len = 1;
                std::fprintf(stderr,
                    "[storageLLM WARNING] KV preallocation disabled under RAM pressure requested=%llu available=%llu chunk=%u\n",
                    (unsigned long long)want,
                    (unsigned long long)avail,
                    max_seq_len);
            }
        }
        if ((uint64_t)max_seq_len > SIZE_MAX / (uint64_t)std::max<uint32_t>(layers, 1u) ||
            (uint64_t)max_seq_len * (uint64_t)std::max<uint32_t>(layers, 1u) > SIZE_MAX / (uint64_t)std::max<uint32_t>(entry_dim, 1u)) {
            if (engine && tracked_ram_bytes > 0) {
                subtract_tier_usage(engine, moe_TIER_RAM, tracked_ram_bytes);
                tracked_ram_bytes = 0;
            }
            if (data) std::free(data);
            if (scales) std::free(scales);
            if (zeros) std::free(zeros);
            data = nullptr; scales = nullptr; zeros = nullptr;
            return;
        }
        const size_t total_entries = (size_t)max_seq_len * (size_t)layers;
        uint8_t* new_data = (uint8_t*)std::calloc(total_entries * entry_dim, sizeof(uint8_t));
        float* new_scales = (float*)std::calloc(total_entries, sizeof(float));
        float* new_zeros = (float*)std::calloc(total_entries, sizeof(float));
        if (!new_data || !new_scales || !new_zeros) {
            if (new_data) std::free(new_data);
            if (new_scales) std::free(new_scales);
            if (new_zeros) std::free(new_zeros);
            if (pre_allocate && max_seq_len > std::min<uint32_t>(moe_MLA_KV_INITIAL_CHUNK, seq_limit)) {
                max_seq_len = std::min<uint32_t>(moe_MLA_KV_INITIAL_CHUNK, seq_limit);
                if (max_seq_len == 0) max_seq_len = 1;
                const size_t retry_entries = (size_t)max_seq_len * (size_t)layers;
                new_data = (uint8_t*)std::calloc(retry_entries * entry_dim, sizeof(uint8_t));
                new_scales = (float*)std::calloc(retry_entries, sizeof(float));
                new_zeros = (float*)std::calloc(retry_entries, sizeof(float));
            }
        }
        if (engine && tracked_ram_bytes > 0) {
            subtract_tier_usage(engine, moe_TIER_RAM, tracked_ram_bytes);
            tracked_ram_bytes = 0;
        }
        if (data) std::free(data);
        if (scales) std::free(scales);
        if (zeros) std::free(zeros);
        data = new_data;
        scales = new_scales;
        zeros = new_zeros;
        if (!data || !scales || !zeros) {
            if (data) std::free(data);
            if (scales) std::free(scales);
            if (zeros) std::free(zeros);
            data = nullptr;
            scales = nullptr;
            zeros = nullptr;
        } else {
            // BUGFIX 954: Update tracked_ram_bytes after successful init() ★★★ CRITICAL
            // Problem: subtract_tier_usage(old) called above, but tracked_ram_bytes
            // left at 0 after new allocation → destructor sees 0 → no subtract → RAM leak.
            // Hundreds of MB of KV cache invisible to tier counter →
            // prefetch_admitted_tier over-admits experts → RAM OOM.
            // Solution: Compute new allocation size and register with tier system.
            // Impact: RAM tier counter now accurate, eviction triggers correctly.
            // BUGFIX 999: Set tracked_ram_bytes BEFORE add_tier_usage ★★★ CRITICAL
            // Problem: If tracked_ram_bytes=0 when add_tier_usage runs, a concurrent
            // moe_kv_cache_track_ram() call sees current > tracked(0) and calls
            // add_tier_usage again → double-counting. Setting tracked_ram_bytes first
            // makes the concurrent call see current == tracked → no-op.
            const uint64_t new_total_entries = (uint64_t)max_seq_len * (uint64_t)num_layers;
            const uint64_t new_bytes = new_total_entries * ((uint64_t)entry_dim + 2ull * sizeof(float));
            if (engine && new_bytes > 0) {
                tracked_ram_bytes = new_bytes;  // Set FIRST to prevent concurrent double-add
                add_tier_usage(engine, moe_TIER_RAM, new_bytes);
            }
        }
    }

    void init(uint32_t layers, bool pre_allocate = false) {
        init(layers, moe_MLA_KV_ENTRY_DIM, moe_MLA_MAX_SEQ_LEN, pre_allocate);
    }

    int ensure_capacity(uint32_t needed_pos) {
        if (needed_pos < max_seq_len) return 1;
        uint32_t new_cap = needed_pos + 1;
        new_cap = (new_cap + 4095u) & ~4095u;
        if (new_cap > seq_limit) new_cap = seq_limit;
        if (new_cap == max_seq_len) return needed_pos < max_seq_len ? 1 : 0;

        const size_t old_entries = (size_t)max_seq_len * num_layers;
        const size_t new_entries = (size_t)new_cap * num_layers;

        if (new_entries > SIZE_MAX / entry_dim ||
            new_entries > SIZE_MAX / sizeof(float)) {
            return 0;
        }

        const size_t new_data_bytes = new_entries * entry_dim;
        const size_t new_scales_bytes = new_entries * sizeof(float);

        // BUGFIX 1006: Pre-check available RAM before triple malloc ★★★ CRITICAL
        // Problem: malloc+memcpy+free pattern requires old + new buffers simultaneously.
        //   Peak memory = old_data + old_scales*2 + new_data + new_scales*2.
        //   Without pre-check, first malloc succeeds → OS RAM pressure spikes →
        //   second/third malloc fails → partial free + return 0, but damage done.
        // Solution: Compare required peak against available RAM before any malloc.
        //   available_ram_bytes() is a snapshot (may be stale), but catches the
        //   common case of insufficient headroom.
        // Impact: Prevents cascading OOM in other components (staging, prefetch).
        if (engine) {
            const size_t old_data_bytes = old_entries * entry_dim;
            const size_t old_scales_bytes = old_entries * sizeof(float);
            const size_t peak_bytes = new_data_bytes + new_scales_bytes * 2;
            // old buffers still alive during malloc, so total transient = peak + old
            const size_t transient_total = peak_bytes + old_data_bytes + old_scales_bytes * 2;
            const uint64_t avail = storagellm::available_ram_bytes();
            if (avail > 0 && transient_total > avail) {
                return 0;  // Pre-reject: not enough RAM for safe resize
            }
        }

        // BUGFIX 923: Use malloc+memcpy instead of realloc to avoid pointer leak ★★★ CRITICAL
        // Problem: realloc(data, ...) success invalidates original 'data' pointer.
        // If subsequent realloc(scales, ...) fails, we try to free nd but 'data' is dangling.
        // Solution: Use malloc+memcpy+free pattern for atomic all-or-nothing allocation.
        // Trade-off: Peak memory usage temporarily doubles (old + new), but safer.
        // In tight RAM situations, this may trigger OOM earlier than realloc would.
        void* nd = std::malloc(new_data_bytes);
        void* ns = std::malloc(new_scales_bytes);
        void* nz = std::malloc(new_scales_bytes);

        if (!nd || !ns || !nz) {
            // Allocation failure: free any successful allocations and return failure
            // Original pointers (data, scales, zeros) remain valid
            std::free(nd);
            std::free(ns);
            std::free(nz);
            return 0;
        }

        // Copy existing data
        const size_t copy_entries = old_entries < new_entries ? old_entries : new_entries;
        std::memcpy(nd, data, copy_entries * entry_dim);
        std::memcpy(ns, scales, copy_entries * sizeof(float));
        std::memcpy(nz, zeros, copy_entries * sizeof(float));

        // Free old allocations
        std::free(data);
        std::free(scales);
        std::free(zeros);

        // Update pointers atomically
        data = (uint8_t*)nd;
        scales = (float*)ns;
        zeros = (float*)nz;

        // Initialize new entries to zero
        if (new_entries > old_entries) {
            std::memset(data + old_entries * entry_dim, 0, (new_entries - old_entries) * entry_dim);
            std::memset(scales + old_entries, 0, (new_entries - old_entries) * sizeof(float));
            std::memset(zeros + old_entries, 0, (new_entries - old_entries) * sizeof(float));
        }

        max_seq_len = new_cap;

        // BUGFIX 921: Remove tracked_ram_bytes update ★★★ CRITICAL
        // Problem: BUGFIX 917 set tracked_ram_bytes = new_bytes here,
        // but moe_kv_cache_track_ram() calculates current = moe_kv_cache_allocated_bytes()
        // which uses same formula → current == tracked_ram_bytes → no delta → no RAM tracking!
        // Solution: Don't touch tracked_ram_bytes here, let moe_kv_cache_track_ram() detect
        // current (new size) > tracked (old size) and call add_tier_usage() correctly
        // Impact: KV cache RAM usage properly tracked after realloc
        // Note: Callers already call moe_kv_cache_track_ram() after ensure_capacity()

        return 1;
    }

    // Asymmetric INT8 Quantization (Store)
    void store_entry(uint32_t layer, uint32_t pos, const float* src, uint32_t count) {
        // BUGFIX 297: Add null pointer check for src ??
        // Problem: src can be null if caller passes invalid pointer
        // Dereferencing null src causes crash
        // Solution: Check src before use
        if (!src || layer >= num_layers || pos >= max_seq_len || count > entry_dim) return;

        const size_t entry_idx = (size_t)pos * num_layers + layer;
        uint8_t* dst = data + entry_idx * entry_dim;

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
        // BUGFIX 298: Add null pointer check for dst ??
        // Problem: dst can be null if caller passes invalid pointer
        // Writing to null dst causes crash
        // Solution: Check dst before use
        if (!dst || layer >= num_layers || pos >= max_seq_len || count > entry_dim) return;

        const size_t entry_idx = (size_t)pos * num_layers + layer;
        const uint8_t* src = data + entry_idx * entry_dim;
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

static uint64_t moe_kv_cache_bytes(uint32_t layers, uint32_t seq_len, uint32_t entry_dim = moe_MLA_KV_ENTRY_DIM) {
    if (layers == 0 || seq_len == 0) {
        return 0;
    }
    // KV Asymmetric INT8: data (uint8_t) + scale (float) + zero (float)
    const uint64_t entry_bytes = (uint64_t)(entry_dim ? entry_dim : moe_MLA_KV_ENTRY_DIM) * sizeof(uint8_t) + 2 * sizeof(float);
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
    return moe_kv_cache_bytes(cache.num_layers, cache.max_seq_len, cache.entry_dim);
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

static int moe_generation_should_preallocate_kv_cache(
    const moe_pc_engine* engine,
    uint32_t layers,
    uint32_t entry_dim = moe_MLA_KV_ENTRY_DIM,
    uint32_t max_seq_len = moe_MLA_MAX_SEQ_LEN
) {
    const uint64_t full_kv_bytes = moe_kv_cache_bytes(layers, max_seq_len, entry_dim);
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
    if (engine &&
        (moe_engine_graph_ir_mentions(engine, "post_ffw_norm") ||
         moe_engine_graph_ir_mentions(engine, "layer_output_scale") ||
         moe_engine_has_common_tensor_suffix(engine, 0, "post_ffw_norm.weight") ||
         moe_engine_has_common_tensor_suffix(engine, 0, "layer_output_scale.weight"))) {
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
