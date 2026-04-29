static void glm5_generation_set_error(glm5_generation_stats_t* stats, const char* message) {
    if (!stats || !message) {
        return;
    }
    std::snprintf(stats->error, sizeof(stats->error), "%s", message);
}

#if defined(GLM5_FAST_APPROX_SILU)
static float glm5_fast_exp_approx(float x) {
    // Bug 9: Handle NaN input to prevent UB in float→int conversion
    // NaN comparisons are always false, leading to (uint32_t)NaN which is UB
    if (x != x) return 0.0f;  // NaN check: x != x is true only for NaN
    if (x <= -87.0f) {
        return 0.0f;
    }
    if (x >= 88.0f) {
        return std::numeric_limits<float>::infinity();
    }
    const float scaled = 12102203.0f * x + 1065353216.0f;
    const uint32_t bits = (uint32_t)scaled;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}
#endif

static float glm5_generation_silu(float x) {
#if defined(GLM5_FAST_APPROX_SILU)
    return x / (1.0f + glm5_fast_exp_approx(-x));
#else
    return x / (1.0f + std::exp(-x));
#endif
}

static int glm5_raw_linear_span_f32(
    glm5_pc_engine* engine,
    const glm5_storage_raw_span_spec_t* span,
    uint64_t rel_offset,
    uint32_t rows,
    uint32_t cols,
    const float* input,
    uint32_t input_count,
    float* output,
    uint32_t output_count
) {
    if (!engine || !span || !input || !output || cols == 0 ||
        input_count != cols || output_count != rows) {
        return 0;
    }
    mmap_context_t* mapped = map_raw_span_part(engine, span);
    if (!raw_span_bounds_ok(span, mapped)) {
        return 0;
    }
    const uint64_t bytes = (uint64_t)rows * (uint64_t)cols * 2ull;
    const uint8_t* base = raw_span_ptr(span, mapped, rel_offset, bytes);
    if (!base) {
        return 0;
    }
    const uint64_t row_bytes = (uint64_t)cols * 2ull;
    return glm5_parallel_rows(engine, rows, 0, [&](uint32_t row) {
        output[row] = glm5_dot_bf16_row(base + row_bytes * (uint64_t)row, input, input_count);
        return 1;
    });
}

static int glm5_raw_mlp_span_f32(
    glm5_pc_engine* engine,
    const glm5_storage_raw_span_spec_t* span,
    const float* hidden,
    uint32_t hidden_size,
    uint32_t intermediate_size,
    float* scratch_gate,
    float* scratch_up,
    float* out_hidden
) {
    if (!engine || !span || !hidden || !scratch_gate || !scratch_up || !out_hidden ||
        hidden_size == 0 || intermediate_size == 0) {
        return 0;
    }
    const uint64_t matrix_bytes = (uint64_t)intermediate_size * (uint64_t)hidden_size * 2ull;
    const uint64_t expected_bytes = matrix_bytes * 2ull + (uint64_t)hidden_size * (uint64_t)intermediate_size * 2ull;
    if (span->bytes < expected_bytes) {
        return 0;
    }
    mmap_context_t* mapped = map_raw_span_part(engine, span);
    if (!raw_span_bounds_ok(span, mapped)) {
        return 0;
    }
    const uint8_t* gate_base = raw_span_ptr(span, mapped, 0, matrix_bytes);
    const uint8_t* up_base = raw_span_ptr(span, mapped, matrix_bytes, matrix_bytes);
    if (!gate_base || !up_base) {
        return 0;
    }
    const uint64_t row_bytes = (uint64_t)hidden_size * 2ull;
    if (!glm5_parallel_rows(engine, intermediate_size, 0, [&](uint32_t row) {
            const float gate = glm5_dot_bf16_row(gate_base + row_bytes * (uint64_t)row, hidden, hidden_size);
            const float up = glm5_dot_bf16_row(up_base + row_bytes * (uint64_t)row, hidden, hidden_size);
            scratch_up[row] = up;
            scratch_gate[row] = glm5_generation_silu(gate) * up;
            return 1;
        })) {
        return 0;
    }
    return glm5_raw_linear_span_f32(
        engine,
        span,
        matrix_bytes * 2ull,
        hidden_size,
        intermediate_size,
        scratch_gate,
        intermediate_size,
        out_hidden,
        hidden_size
    );
}

static int glm5_router_topk_f32(
    glm5_pc_engine* engine,
    int32_t layer,
    const float* hidden,
    uint32_t hidden_size,
    uint32_t experts,
    uint32_t top_k,
    std::vector<int>* out_experts,
    std::vector<float>* out_weights,
    std::vector<float>* scratch_logits,
    std::vector<uint32_t>* scratch_order
) {
    if (!engine || !hidden || !out_experts || !out_weights || !scratch_logits || !scratch_order ||
        hidden_size == 0 || experts == 0 || top_k == 0) {
        return 0;
    }
    const uint64_t router_bytes = (uint64_t)experts * (uint64_t)hidden_size * 2ull;
    const glm5_storage_raw_span_spec_t* span =
        find_raw_role_span_bytes(layer, GLM5_RAW_ROLE_ROUTER, router_bytes, 0);
    if (!span) {
        return 0;
    }
    scratch_logits->assign(experts, 0.0f);
    if (!glm5_raw_linear_span_f32(
            engine,
            span,
            0,
            experts,
            hidden_size,
            hidden,
            hidden_size,
            scratch_logits->data(),
            experts)) {
        return 0;
    }

    auto sigmoid = [](float x) {
        if (x >= 0.0f) {
            const float z = std::exp(-x);
            return 1.0f / (1.0f + z);
        }
        const float z = std::exp(x);
        return z / (1.0f + z);
    };
    for (float& v : *scratch_logits) {
        v = sigmoid(v);
    }

    top_k = std::min(top_k, experts);
    scratch_order->resize(experts);
    uint32_t* order = scratch_order->data();
    for (uint32_t i = 0; i < experts; ++i) {
        order[i] = i;
    }
    // Fix 9: nth_element O(N) avg + sort O(K log K) beats partial_sort O(N log K)
    auto cmp = [&](uint32_t a, uint32_t b) {
        return (*scratch_logits)[a] > (*scratch_logits)[b];
    };
    if (top_k < experts) {
        std::nth_element(order, order + top_k, order + experts, cmp);
    }
    std::sort(order, order + top_k, cmp);

    float denom = 0.0f;
    out_experts->clear();
    out_weights->clear();
    for (uint32_t i = 0; i < top_k; ++i) {
        const float w = (*scratch_logits)[order[i]];
        out_experts->push_back((int)order[i]);
        out_weights->push_back(w);
        denom += w;
    }
    // Bug 10: Clamp denom to prevent silent failure on underflow
    // When all sigmoid outputs are near zero, denom can underflow to 0.0f
    if (denom <= 0.0f) {
        denom = 1e-8f;  // Clamp to small positive value, maintain weight ratios
    }
    constexpr float routed_scaling_factor = 2.5f;
    for (float& w : *out_weights) {
        w = (w / denom) * routed_scaling_factor;
    }
    return 1;
}

// MLA constants for GLM-5.1
static constexpr uint32_t GLM5_MLA_KV_COMPRESSED_DIM = 512u;
static constexpr uint32_t GLM5_MLA_K_ROPE_DIM = 64u;
static constexpr uint32_t GLM5_MLA_KV_A_PROJ_DIM = GLM5_MLA_KV_COMPRESSED_DIM + GLM5_MLA_K_ROPE_DIM; // 576
static constexpr uint32_t GLM5_MLA_KV_ENTRY_DIM = GLM5_MLA_KV_A_PROJ_DIM;
static constexpr uint32_t GLM5_MLA_MAX_SEQ_LEN = 8192u;
static constexpr uint32_t GLM5_MLA_ROPE_TABLE_LEN = 16384u;
// Bonus: Enforce consistency between ROPE table and KV cache max length
static_assert(GLM5_MLA_ROPE_TABLE_LEN >= GLM5_MLA_MAX_SEQ_LEN,
              "ROPE table must cover at least MAX_SEQ_LEN positions");
static constexpr uint32_t GLM5_MLA_KV_INITIAL_CHUNK = 1024u;
static constexpr uint32_t GLM5_MLA_NUM_HEADS = 64u;
static constexpr uint32_t GLM5_MLA_HEAD_DIM = 256u;
static constexpr uint32_t GLM5_MLA_NOPE_DIM = 192u;
static constexpr uint32_t GLM5_MLA_ROPE_DIM = 64u;
static constexpr uint32_t GLM5_MLA_V_HEAD_DIM = 256u;
static constexpr uint32_t GLM5_MLA_Q_B_DIM = GLM5_MLA_NUM_HEADS * GLM5_MLA_HEAD_DIM; // 16384
static constexpr uint32_t GLM5_MLA_K_NOPE_TOTAL_DIM = GLM5_MLA_NUM_HEADS * GLM5_MLA_NOPE_DIM; // 12288
static constexpr uint32_t GLM5_MLA_V_TOTAL_DIM = GLM5_MLA_NUM_HEADS * GLM5_MLA_V_HEAD_DIM; // 16384
static_assert(GLM5_MLA_ROPE_DIM == GLM5_MLA_K_ROPE_DIM, "Q/K RoPE dims must match");
static_assert(GLM5_MLA_Q_B_DIM == 16384u, "GLM5 q_b projection width mismatch");
static_assert(GLM5_MLA_K_NOPE_TOTAL_DIM == 12288u, "GLM5 K_nope width mismatch");
static_assert(GLM5_MLA_V_TOTAL_DIM == 16384u, "GLM5 V width mismatch");
static constexpr float GLM5_ROPE_THETA = 500000.0f;
static constexpr uint32_t GLM5_MOE_TOP_EXPERTS = 8u;

#if defined(__AVX2__)
static inline float glm5_hsum256_ps(__m256 v) {
#if defined(__AVX2__)
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
#else
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, v);
    return lanes[0] + lanes[1] + lanes[2] + lanes[3] +
           lanes[4] + lanes[5] + lanes[6] + lanes[7];
#endif
}
#endif

static inline float glm5_dot_f32(const float* a, const float* b, uint32_t count) {
    float sum = 0.0f;
    uint32_t i = 0;
#if defined(__AVX2__)
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    for (; i + 15u < count; i += 16u) {
        const __m256 av0 = _mm256_loadu_ps(a + i);
        const __m256 bv0 = _mm256_loadu_ps(b + i);
        const __m256 av1 = _mm256_loadu_ps(a + i + 8u);
        const __m256 bv1 = _mm256_loadu_ps(b + i + 8u);
        acc0 = glm5_madd_ps(av0, bv0, acc0);
        acc1 = glm5_madd_ps(av1, bv1, acc1);
    }
    sum = glm5_hsum256_ps(_mm256_add_ps(acc0, acc1));
#endif
    for (; i < count; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

static inline void glm5_scale_f32(float* values, float scale, uint32_t count) {
    uint32_t i = 0;
#if defined(__AVX2__)
    const __m256 vscale = _mm256_set1_ps(scale);
    for (; i + 7u < count; i += 8u) {
        __m256 v = _mm256_loadu_ps(values + i);
        v = _mm256_mul_ps(v, vscale);
        _mm256_storeu_ps(values + i, v);
    }
#endif
    for (; i < count; ++i) {
        values[i] *= scale;
    }
}

static inline void glm5_axpy_f32(float* dst, const float* src, float alpha, uint32_t count) {
    uint32_t i = 0;
#if defined(__AVX2__)
    const __m256 valpha = _mm256_set1_ps(alpha);
    for (; i + 7u < count; i += 8u) {
        const __m256 s = _mm256_loadu_ps(src + i);
        __m256 d = _mm256_loadu_ps(dst + i);
        d = glm5_madd_ps(valpha, s, d);
        _mm256_storeu_ps(dst + i, d);
    }
#endif
    for (; i < count; ++i) {
        dst[i] += alpha * src[i];
    }
}

static inline void glm5_axpy_bf16_row_f32(float* dst, const uint8_t* row, float alpha, uint32_t count) {
    uint32_t i = 0;
#if defined(__AVX2__)
    const __m256 valpha = _mm256_set1_ps(alpha);
    for (; i + 7u < count; i += 8u) {
        const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + (uint64_t)i * 2ull));
        const __m256i expanded = _mm256_cvtepu16_epi32(packed);
        const __m256 values = _mm256_castsi256_ps(_mm256_slli_epi32(expanded, 16));
        __m256 accum = _mm256_loadu_ps(dst + i);
        accum = glm5_madd_ps(valpha, values, accum);
        _mm256_storeu_ps(dst + i, accum);
    }
#endif
    for (; i < count; ++i) {
        uint16_t bf16;
        std::memcpy(&bf16, row + (uint64_t)i * 2ull, sizeof(bf16));
        dst[i] += alpha * glm5_raw_bf16_to_f32(bf16);
    }
}

// bf16 conversion helpers for KV cache
static inline uint16_t glm5_f32_to_bf16(float v) {
    uint32_t u;
    std::memcpy(&u, &v, sizeof(u));
    // Round-to-nearest-even: add 0x7FFF + bit[16] for tie-breaking
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
static inline float glm5_bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float v;
    std::memcpy(&v, &u, sizeof(v));
    return v;
}

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

struct glm5_expert_work_t {
    float* gate = nullptr;
    float* up = nullptr;
    float* out = nullptr;
    float weight = 0.0f;
    int expert_id = 0;
    int ok = 0;
};

struct glm5_decode_scratch {
    std::vector<float> hidden;
    std::vector<float> residual;
    std::vector<float> norm;
    std::vector<float> attn;
    std::vector<float> attn_kv_a;
    std::vector<float> attn_kv_norm;
    std::vector<float> attn_kv_b;      // kept for API compat; unused with absorption
    std::vector<float> attn_value;
    std::vector<float> mlp;
    std::vector<float> shared;
    std::vector<float> expert;
    std::vector<float> gate;
    std::vector<float> up;
    std::vector<float> router_logits;
    std::vector<uint32_t> router_order;
    std::vector<int> selected_experts;
    std::vector<float> selected_weights;
    std::vector<glm5_expert_work_t> expert_work;
    std::vector<float> expert_gate_pool;
    std::vector<float> expert_up_pool;
    std::vector<float> expert_out_pool;
    glm5_kv_cache kv_cache;
    uint32_t position = 0;
    // Absorption scratch (moved from thread_local)
    std::vector<float> q_a;
    std::vector<float> q_a_norm;
    std::vector<float> q_b;
    std::vector<float> q_absorbed;    // [num_heads * 512]
    std::vector<float> c_weighted;    // [num_heads * 512]
    std::vector<float> head_max;     // [num_heads]
    std::vector<float> head_denom;   // [num_heads]
    std::vector<float> rope_k_cos;
    std::vector<float> rope_k_sin;
    std::vector<float> rope_q_cos;
    std::vector<float> rope_q_sin;
    // Legacy single-layer decode buffer; retained for compatibility with older paths.
    std::vector<float> kv_decode_buf;
    std::vector<std::vector<float>> kv_decode_layers;
    std::vector<uint32_t> kv_decoded_up_to;
    uint64_t kv_decode_tracked_ram_bytes = 0;
};

static uint64_t glm5_kv_decode_cache_allocated_bytes(const glm5_decode_scratch* s) {
    if (!s) {
        return 0;
    }
    uint64_t bytes = 0;
    for (const auto& layer : s->kv_decode_layers) {
        const uint64_t cap = (uint64_t)layer.size();
        if (cap > (UINT64_MAX - bytes) / sizeof(float)) {
            return UINT64_MAX;
        }
        bytes += cap * sizeof(float);
    }
    return bytes;
}

static void glm5_kv_decode_cache_track_ram(glm5_pc_engine* engine, glm5_decode_scratch* s) {
    if (!engine || !s) {
        return;
    }
    const uint64_t current = glm5_kv_decode_cache_allocated_bytes(s);
    if (current > s->kv_decode_tracked_ram_bytes) {
        add_tier_usage(engine, GLM5_TIER_RAM, current - s->kv_decode_tracked_ram_bytes);
    } else if (s->kv_decode_tracked_ram_bytes > current) {
        subtract_tier_usage(engine, GLM5_TIER_RAM, s->kv_decode_tracked_ram_bytes - current);
    }
    s->kv_decode_tracked_ram_bytes = current;
}

static void glm5_kv_decode_cache_release_ram(glm5_pc_engine* engine, glm5_decode_scratch* s) {
    if (!engine || !s || s->kv_decode_tracked_ram_bytes == 0) {
        return;
    }
    subtract_tier_usage(engine, GLM5_TIER_RAM, s->kv_decode_tracked_ram_bytes);
    s->kv_decode_tracked_ram_bytes = 0;
}

struct glm5_kv_decode_cache_ram_scope {
    glm5_pc_engine* engine = nullptr;
    glm5_decode_scratch* scratch = nullptr;
    glm5_kv_decode_cache_ram_scope(glm5_pc_engine* e, glm5_decode_scratch* s) : engine(e), scratch(s) {}
    ~glm5_kv_decode_cache_ram_scope() {
        glm5_kv_decode_cache_release_ram(engine, scratch);
    }
};

static int glm5_kv_decode_layer_cache_prepare(
    glm5_pc_engine* engine,
    glm5_decode_scratch* s,
    uint32_t layer,
    uint32_t decoded_count,
    std::vector<float>** out_buf,
    uint32_t* out_start
) {
    if (!engine || !s || !out_buf || !out_start || layer >= s->kv_cache.num_layers) {
        return 0;
    }
    if (s->kv_decode_layers.size() != s->kv_cache.num_layers) {
        s->kv_decode_layers.clear();
        s->kv_decode_layers.resize(s->kv_cache.num_layers);
        s->kv_decoded_up_to.assign(s->kv_cache.num_layers, 0u);
        glm5_kv_decode_cache_track_ram(engine, s);
    } else if (s->kv_decoded_up_to.size() != s->kv_cache.num_layers) {
        s->kv_decoded_up_to.assign(s->kv_cache.num_layers, 0u);
    }
    const size_t needed = (size_t)decoded_count * GLM5_MLA_KV_ENTRY_DIM;
    std::vector<float>& buf = s->kv_decode_layers[layer];
    if (buf.size() < needed) {
        buf.resize(needed);
        glm5_kv_decode_cache_track_ram(engine, s);
    }
    *out_start = std::min(s->kv_decoded_up_to[layer], decoded_count);
    *out_buf = &buf;
    return 1;
}

static void glm5_rope_table_resize(
    std::vector<float>* cos_table,
    std::vector<float>* sin_table,
    uint32_t dim,
    float base
) {
    if (!cos_table || !sin_table || dim < 2) {
        return;
    }
    const uint32_t half_dim = dim / 2u;
    const size_t needed = (size_t)GLM5_MLA_ROPE_TABLE_LEN * (size_t)half_dim;
    if (cos_table->size() == needed && sin_table->size() == needed) {
        return;
    }
    cos_table->resize(needed);
    sin_table->resize(needed);
    std::vector<float> freqs(half_dim);
    for (uint32_t j = 0; j < half_dim; ++j) {
        const uint32_t i = j * 2u;
        freqs[j] = 1.0f / std::pow(base, (float)i / (float)dim);
    }
    for (uint32_t pos = 0; pos < GLM5_MLA_ROPE_TABLE_LEN; ++pos) {
        float* cos_row = cos_table->data() + (size_t)pos * (size_t)half_dim;
        float* sin_row = sin_table->data() + (size_t)pos * (size_t)half_dim;
        for (uint32_t j = 0; j < half_dim; ++j) {
            const float theta = (float)pos * freqs[j];
            cos_row[j] = std::cos(theta);
            sin_row[j] = std::sin(theta);
        }
    }
}

static void glm5_decode_scratch_resize(
    glm5_decode_scratch* s,
    const glm5_storage_constants_t* c,
    int pre_allocate_kv
) {
    const uint32_t hidden = c->hidden_size;
    const uint32_t expert_intermediate = c->expert_intermediate_size;
    s->hidden.resize(hidden);
    s->residual.resize(hidden);
    s->norm.resize(hidden);
    s->attn.resize(hidden);
    s->attn_kv_a.resize(GLM5_MLA_KV_A_PROJ_DIM);
    s->attn_kv_norm.resize(GLM5_MLA_KV_COMPRESSED_DIM);
    // attn_kv_b no longer allocated; KV absorption reads W_kv_b directly.
    s->attn_value.resize(GLM5_MLA_V_TOTAL_DIM);
    s->mlp.resize(hidden);
    s->shared.resize(hidden);
    s->expert.resize(hidden);
    s->gate.resize(std::max<uint32_t>(12288u, expert_intermediate));
    // Absorption buffers
    s->q_a.resize(2048);
    s->q_a_norm.resize(2048);
    s->q_b.resize(GLM5_MLA_Q_B_DIM);
    s->q_absorbed.resize((size_t)GLM5_MLA_NUM_HEADS * GLM5_MLA_KV_COMPRESSED_DIM);
    s->c_weighted.resize((size_t)GLM5_MLA_NUM_HEADS * GLM5_MLA_KV_COMPRESSED_DIM);
    s->head_max.resize(GLM5_MLA_NUM_HEADS);
    s->head_denom.resize(GLM5_MLA_NUM_HEADS);
    s->up.resize(std::max<uint32_t>(12288u, expert_intermediate));
    s->router_logits.resize(c->experts_per_moe_layer);
    s->router_order.resize(c->experts_per_moe_layer);
    s->selected_experts.reserve(GLM5_MOE_TOP_EXPERTS);
    s->selected_weights.reserve(GLM5_MOE_TOP_EXPERTS);
    s->expert_work.resize(GLM5_MOE_TOP_EXPERTS);
    s->expert_gate_pool.resize((size_t)GLM5_MOE_TOP_EXPERTS * (size_t)expert_intermediate);
    s->expert_up_pool.resize((size_t)GLM5_MOE_TOP_EXPERTS * (size_t)expert_intermediate);
    s->expert_out_pool.resize((size_t)GLM5_MOE_TOP_EXPERTS * (size_t)hidden);
    glm5_rope_table_resize(&s->rope_k_cos, &s->rope_k_sin, GLM5_MLA_K_ROPE_DIM, GLM5_ROPE_THETA);
    glm5_rope_table_resize(&s->rope_q_cos, &s->rope_q_sin, GLM5_MLA_ROPE_DIM, GLM5_ROPE_THETA);
    if (s->kv_cache.num_layers != c->num_hidden_layers || s->kv_cache.data.empty() ||
        (pre_allocate_kv && s->kv_cache.max_seq_len < GLM5_MLA_MAX_SEQ_LEN)) {
        s->kv_cache.init(c->num_hidden_layers, pre_allocate_kv != 0);
        s->kv_decode_layers.clear();
        s->kv_decoded_up_to.clear();
        s->position = 0;
    }
}

// Bug #1 Fix: Full MLA attention with Q projection, RoPE, scaled dot-product.
// GLM4/GLM5 Multi-head Latent Attention (MLA):
//   KV path: hidden -> kv_a_proj(576) -> [kv_a_norm(512), mqa_rope(64)]
//            kv_a_norm -> kv_b_proj(28672) -> [K_nope(12288), V(16384)]
//   Q path:  hidden -> q_a_proj(2048) -> q_a_norm(2048) -> q_b_proj(12288+4096=16384)
//            -> [Q_nope(12288), Q_rope(4096)] split by heads
//   Score = softmax(Q_nope*K_nope + Q_rope*K_rope) * V -> o_proj -> attn output

// Keep this tied to the converted model metadata; tensor shapes alone do not determine theta.
static void glm5_rope_inplace(
    float* data,
    uint32_t dim,
    uint32_t position,
    const std::vector<float>& cos_table,
    const std::vector<float>& sin_table,
    float base = GLM5_ROPE_THETA
) {
    const uint32_t half_dim = dim / 2u;
    const size_t row_offset = (size_t)position * (size_t)half_dim;
    if (position < GLM5_MLA_ROPE_TABLE_LEN &&
        cos_table.size() >= row_offset + half_dim &&
        sin_table.size() >= row_offset + half_dim) {
        const float* cos_row = cos_table.data() + row_offset;
        const float* sin_row = sin_table.data() + row_offset;
        for (uint32_t j = 0; j < half_dim; ++j) {
            const uint32_t i = j * 2u;
            const float cos_t = cos_row[j];
            const float sin_t = sin_row[j];
            const float x0 = data[i];
            const float x1 = data[i + 1u];
            data[i] = x0 * cos_t - x1 * sin_t;
            data[i + 1u] = x0 * sin_t + x1 * cos_t;
        }
        return;
    }
    for (uint32_t i = 0; i + 1u < dim; i += 2u) {
        const float freq = 1.0f / std::pow(base, (float)i / (float)dim);
        const float theta = (float)position * freq;
        const float cos_t = std::cos(theta);
        const float sin_t = std::sin(theta);
        const float x0 = data[i];
        const float x1 = data[i + 1];
        data[i]     = x0 * cos_t - x1 * sin_t;
        data[i + 1] = x0 * sin_t + x1 * cos_t;
    }
}

static void glm5_enqueue_attention_prefetch(glm5_pc_engine* engine, int32_t layer) {
    if (!engine || layer < 0) {
        return;
    }
    const glm5_raw_tensor_kind_t kinds[] = {
        GLM5_RAW_TENSOR_Q_A_PROJ,
        GLM5_RAW_TENSOR_Q_B_PROJ,
        GLM5_RAW_TENSOR_KV_A_PROJ_WITH_MQA,
        GLM5_RAW_TENSOR_KV_B_PROJ,
        GLM5_RAW_TENSOR_O_PROJ
    };
    uint32_t queued_spans[sizeof(kinds) / sizeof(kinds[0]) + 1u] = {};
    uint32_t queued_count = 0;

    const glm5_storage_constants_t* c = glm5_storage_constants();
    if (c && layer >= c->first_moe_layer && layer <= c->last_moe_layer) {
        const glm5_storage_raw_span_spec_t* router_span = find_raw_role_span(layer, GLM5_RAW_ROLE_ROUTER);
        if (router_span) {
            const uint32_t begin = glm5_storage_raw_span_find_first(layer);
            const uint32_t end = glm5_storage_raw_span_find_end(layer);
            for (uint32_t si = begin; si < end; ++si) {
                if (glm5_storage_raw_span_at(si) == router_span) {
                    queued_spans[queued_count++] = si;
                    enqueue_raw_span_prefetch(engine, si, GLM5_DEVICE_POOL_ACTIVATION);
                    break;
                }
            }
        }
    }

    for (glm5_raw_tensor_kind_t kind : kinds) {
        glm5_raw_tensor_desc desc{};
        const glm5_storage_raw_span_spec_t* span = nullptr;
        if (!find_raw_tensor_desc_and_span(layer, kind, &desc, &span) || !span) {
            continue;
        }
        const uint32_t begin = glm5_storage_raw_span_find_first(layer);
        const uint32_t end = glm5_storage_raw_span_find_end(layer);
        for (uint32_t si = begin; si < end; ++si) {
            if (glm5_storage_raw_span_at(si) != span) {
                continue;
            }
            bool seen = false;
            for (uint32_t qi = 0; qi < queued_count; ++qi) {
                if (queued_spans[qi] == si) {
                    seen = true;
                    break;
                }
            }
            if (!seen && queued_count < sizeof(queued_spans) / sizeof(queued_spans[0])) {
                queued_spans[queued_count++] = si;
                enqueue_raw_span_prefetch(engine, si, GLM5_DEVICE_POOL_ACTIVATION);
            }
            break;
        }
    }
}

static uint64_t glm5_attention_prefetch_bytes(int32_t layer) {
    if (layer < 0) {
        return 0;
    }
    const glm5_raw_tensor_kind_t kinds[] = {
        GLM5_RAW_TENSOR_Q_A_PROJ,
        GLM5_RAW_TENSOR_Q_B_PROJ,
        GLM5_RAW_TENSOR_KV_A_PROJ_WITH_MQA,
        GLM5_RAW_TENSOR_KV_B_PROJ,
        GLM5_RAW_TENSOR_O_PROJ
    };
    const glm5_storage_raw_span_spec_t* spans[sizeof(kinds) / sizeof(kinds[0])] = {};
    uint32_t span_count = 0;
    uint64_t bytes = 0;
    for (glm5_raw_tensor_kind_t kind : kinds) {
        glm5_raw_tensor_desc desc{};
        const glm5_storage_raw_span_spec_t* span = nullptr;
        if (!find_raw_tensor_desc_and_span(layer, kind, &desc, &span) || !span) {
            continue;
        }
        bool seen = false;
        for (uint32_t i = 0; i < span_count; ++i) {
            if (spans[i] == span) {
                seen = true;
                break;
            }
        }
        if (!seen && span_count < sizeof(spans) / sizeof(spans[0])) {
            spans[span_count++] = span;
            bytes += span->bytes;
        }
    }
    return bytes;
}

static int glm5_can_enqueue_attention_lookahead(glm5_pc_engine* engine, int32_t layer) {
    if (!engine || layer < 0 || engine->config.vram_budget_bytes == 0) {
        return 1;
    }
    const uint64_t bytes = glm5_attention_prefetch_bytes(layer);
    if (bytes == 0) {
        return 1;
    }
    const uint64_t used = std::max(
        tier_used(engine, GLM5_TIER_VRAM),
        engine->device_allocated_bytes.load(std::memory_order_relaxed));
    const uint64_t budget = engine->config.vram_budget_bytes;
    const uint64_t threshold = budget - budget / 6u;
    return used < threshold && bytes <= threshold - used;
}

static int glm5_layer_attention_f32(
    glm5_pc_engine* engine,
    int32_t layer,
    glm5_decode_scratch* s,
    float* hidden,
    uint32_t hidden_size,
    uint32_t position
) {
    constexpr uint32_t cdim = GLM5_MLA_KV_COMPRESSED_DIM; // 512
    constexpr uint32_t num_heads = GLM5_MLA_NUM_HEADS;     // 64
    constexpr uint32_t head_dim = GLM5_MLA_HEAD_DIM;       // 256
    constexpr uint32_t nope_dim = GLM5_MLA_NOPE_DIM;       // 192
    constexpr uint32_t rope_dim = GLM5_MLA_ROPE_DIM;       // 64
    constexpr uint32_t v_head_dim = GLM5_MLA_V_HEAD_DIM;   // 256
    constexpr uint64_t kv_b_row_bytes = cdim * 2ull;       // bf16

    // --- Pre-attention RMSNorm ---
    if (!glm5_pc_engine_raw_rmsnorm_f32(
            engine, layer, GLM5_RAW_TENSOR_INPUT_LAYER_NORM,
            hidden, hidden_size, s->norm.data(), hidden_size, 1.0e-6f)) {
        return 0;
    }

    // --- KV path: kv_a_proj -> kv_a_norm (NO kv_b_proj; absorbed) ---
    // kv_a is small (576 outputs) and runs inline. No std::async: creating
    // OS threads for 80 layers times N tokens is too expensive per inference.
    if (!glm5_pc_engine_raw_linear_f32(
            engine, layer, GLM5_RAW_TENSOR_KV_A_PROJ_WITH_MQA,
            s->norm.data(), hidden_size, s->attn_kv_a.data(), GLM5_MLA_KV_A_PROJ_DIM)) {
        return 0;
    }
    if (!glm5_pc_engine_raw_rmsnorm_f32(
            engine, layer, GLM5_RAW_TENSOR_KV_A_LAYER_NORM,
            s->attn_kv_a.data(), cdim, s->attn_kv_norm.data(), cdim, 1.0e-6f)) {
        return 0;
    }
    float* k_rope = s->attn_kv_a.data() + cdim;

    // --- Q path: q_a_proj -> q_a_norm -> q_b_proj ---
    if (!glm5_pc_engine_raw_linear_f32(
            engine, layer, GLM5_RAW_TENSOR_Q_A_PROJ,
            s->norm.data(), hidden_size, s->q_a.data(), 2048u) ||
        !glm5_pc_engine_raw_rmsnorm_f32(
            engine, layer, GLM5_RAW_TENSOR_Q_A_LAYER_NORM,
            s->q_a.data(), 2048u, s->q_a_norm.data(), 2048u, 1.0e-6f) ||
        !glm5_pc_engine_raw_linear_f32(
            engine, layer, GLM5_RAW_TENSOR_Q_B_PROJ,
            s->q_a_norm.data(), 2048u, s->q_b.data(), GLM5_MLA_Q_B_DIM)) {
        return 0;
    }

    // Apply K RoPE before storing the current KV entry. Q RoPE is fused below
    // with Q-side absorption and historical KV decode in one barrier.
    glm5_rope_inplace(k_rope, GLM5_MLA_K_ROPE_DIM, position, s->rope_k_cos, s->rope_k_sin);

    // Cache compressed KV + RoPE'd k_rope as bf16
    if (position >= s->kv_cache.max_seq_len) {
        uint32_t new_cap = s->kv_cache.max_seq_len > 0 ? s->kv_cache.max_seq_len : 1u;
        while (new_cap <= position && new_cap < GLM5_MLA_MAX_SEQ_LEN) {
            new_cap *= 2u;
        }
        if (new_cap > GLM5_MLA_MAX_SEQ_LEN) {
            new_cap = GLM5_MLA_MAX_SEQ_LEN;
        }
        const uint64_t old_bytes = glm5_kv_cache_bytes(s->kv_cache.num_layers, s->kv_cache.max_seq_len);
        const uint64_t new_bytes = glm5_kv_cache_bytes(s->kv_cache.num_layers, new_cap);
        // Bug 8: Account for realloc peak (old + new simultaneously in memory during memcpy)
        // ensure_capacity allocates new_data, copies from data, then moves
        // Peak memory = old_bytes + new_bytes during the copy phase
        const uint64_t realloc_peak = old_bytes + new_bytes;
        const uint64_t ram_budget = tier_budget(engine, GLM5_TIER_RAM);
        if (ram_budget > 0) {
            const uint64_t ram_used = tier_used(engine, GLM5_TIER_RAM);
            // Reserve peak memory to prevent underestimating RAM pressure
            if (ram_used + realloc_peak > ram_budget) {
                std::vector<std::pair<int,int>> to_free;
                demote_cold_until(engine, GLM5_TIER_RAM, realloc_peak, to_free);
                for (auto& p : to_free) {
                    device_free_expert(engine, p.first, p.second);
                }
            }
        }
    }
    s->kv_cache.ensure_capacity(position);
    glm5_kv_cache_track_ram(engine, &s->kv_cache);
    if (position < s->kv_cache.max_seq_len) {
        // Build float32 entry, then store as bf16
        float kv_entry[GLM5_MLA_KV_ENTRY_DIM];
        std::memcpy(kv_entry, s->attn_kv_norm.data(), cdim * sizeof(float));
        std::memcpy(kv_entry + cdim, k_rope, GLM5_MLA_K_ROPE_DIM * sizeof(float));
        s->kv_cache.store_entry((uint32_t)layer, position, kv_entry, GLM5_MLA_KV_ENTRY_DIM);
    }
    s->kv_cache.seq_len = std::min(position + 1u, s->kv_cache.max_seq_len);
    const uint32_t seq_len = std::min(position + 1u, s->kv_cache.max_seq_len);

    // --- Cached W_kv_b for KV absorption ---
    glm5_raw_tensor_cache_entry kv_b_raw{};
    if (!get_cached_raw_tensor(engine, layer, GLM5_RAW_TENSOR_KV_B_PROJ, &kv_b_raw) ||
        kv_b_raw.cols != cdim || kv_b_raw.rows < (GLM5_MLA_K_NOPE_TOTAL_DIM + GLM5_MLA_V_TOTAL_DIM)) {
        return 0;
    }
    const uint8_t* kv_b_base = kv_b_raw.base;
    if (!kv_b_base) return 0;

    // Q RoPE, Q-side absorption, and historical KV decode are independent.
    // Run them in one flat dispatch to avoid three separate row-worker barriers.
    std::fill(s->q_absorbed.begin(), s->q_absorbed.end(), 0.0f);
    const uint32_t decoded_kv_count = position < seq_len ? position : seq_len;
    std::vector<float>* layer_decode = nullptr;
    uint32_t decode_start = 0;
    if (decoded_kv_count > 0 &&
        !glm5_kv_decode_layer_cache_prepare(
            engine, s, (uint32_t)layer, decoded_kv_count, &layer_decode, &decode_start)) {
        return 0;
    }
    const uint32_t decode_delta = decoded_kv_count > decode_start ? decoded_kv_count - decode_start : 0u;
    const uint32_t fused_rows = num_heads + decode_delta;
    if (fused_rows > 0) {
        if (!glm5_parallel_rows(engine, fused_rows, 0, [&](uint32_t row) {
            if (row < num_heads) {
                const uint32_t h = row;
                glm5_rope_inplace(
                    s->q_b.data() + h * head_dim + nope_dim,
                    rope_dim,
                    position,
                    s->rope_q_cos,
                    s->rope_q_sin
                );
                float* qa_h = s->q_absorbed.data() + h * cdim;
                for (uint32_t d = 0; d < nope_dim; ++d) {
                    const float qv = s->q_b[h * head_dim + d];
                    if (std::abs(qv) < 1e-12f) continue;
                    const uint8_t* w_row = kv_b_base + kv_b_row_bytes * (h * nope_dim + d);
                    glm5_axpy_bf16_row_f32(qa_h, w_row, qv, cdim);
                }
            } else {
                const uint32_t p = decode_start + (row - num_heads);
                float* decode = layer_decode->data() + (size_t)p * GLM5_MLA_KV_ENTRY_DIM;
                s->kv_cache.load_entry((uint32_t)layer, p, decode, GLM5_MLA_KV_ENTRY_DIM);
            }
            return 1;
        })) {
            return 0;
        }
        if (decoded_kv_count > 0) {
            s->kv_decoded_up_to[(uint32_t)layer] = std::max(
                s->kv_decoded_up_to[(uint32_t)layer],
                decoded_kv_count);
        }
    }

    // --- Online softmax: head-outer (parallel), position-inner ---
    // Each head has independent running max/denom/c_weighted; decoded KV is read-only shared.
    const float scale = 1.0f / std::sqrt((float)head_dim);
    std::fill(s->c_weighted.begin(), s->c_weighted.end(), 0.0f);
    s->head_max.assign(num_heads, -1e30f);
    s->head_denom.assign(num_heads, 0.0f);

    glm5_parallel_rows(engine, num_heads, 0, [&](uint32_t h) {
        const float* qa_h = s->q_absorbed.data() + h * cdim;
        const float* q_rope_h = s->q_b.data() + h * head_dim + nope_dim;
        float* cw_h = s->c_weighted.data() + h * cdim;
        float h_max = -1e30f;
        float h_denom = 0.0f;

        uint32_t loop_end = position < s->kv_cache.max_seq_len ? seq_len : seq_len + 1;
        for (uint32_t p = 0; p < loop_end; ++p) {
            const float* c_kv;
            const float* p_k_rope;
            if (p == position || (position >= s->kv_cache.max_seq_len && p == seq_len)) {
                c_kv = s->attn_kv_norm.data();
                p_k_rope = k_rope;
            } else {
                const float* decode = layer_decode->data() + (size_t)p * GLM5_MLA_KV_ENTRY_DIM;
                c_kv = decode;
                p_k_rope = decode + cdim;
            }

            float score = glm5_dot_f32(qa_h, c_kv, cdim);
            score += glm5_dot_f32(q_rope_h, p_k_rope, std::min(rope_dim, GLM5_MLA_K_ROPE_DIM));
            score *= scale;

            if (score > h_max) {
                const float rescale = std::exp(h_max - score);
                h_denom *= rescale;
                glm5_scale_f32(cw_h, rescale, cdim);
                h_max = score;
            }
            const float w = std::exp(score - h_max);
            h_denom += w;
            glm5_axpy_f32(cw_h, c_kv, w, cdim);
        }

        // Normalize
        if (h_denom > 0.0f) {
            const float inv = 1.0f / h_denom;
            glm5_scale_f32(cw_h, inv, cdim);
        }
        s->head_max[h] = h_max;
        s->head_denom[h] = h_denom;
        return 1;
    });

    // --- V-side absorption: head-parallel expansion ---
    const uint8_t* kv_b_v_base = kv_b_base + kv_b_row_bytes * GLM5_MLA_K_NOPE_TOTAL_DIM;
    glm5_parallel_rows(engine, num_heads, 0, [&](uint32_t h) {
        float* out_h = s->attn_value.data() + h * v_head_dim;
        const float* cw_h = s->c_weighted.data() + h * cdim;
        for (uint32_t d = 0; d < v_head_dim; ++d) {
            const uint8_t* w_row = kv_b_v_base + kv_b_row_bytes * (h * v_head_dim + d);
            out_h[d] = glm5_dot_bf16_row(w_row, cw_h, cdim);
        }
        return 1;
    });

    // --- O projection: attention value width -> hidden_size ---
    if (!glm5_pc_engine_raw_linear_f32(
            engine, layer, GLM5_RAW_TENSOR_O_PROJ,
            s->attn_value.data(), GLM5_MLA_V_TOTAL_DIM, s->attn.data(), hidden_size)) {
        return 0;
    }
    for (uint32_t i = 0; i < hidden_size; ++i) {
        hidden[i] += s->attn[i];
    }
    return 1;
}

static int glm5_layer_dense_mlp_f32(
    glm5_pc_engine* engine,
    int32_t layer,
    glm5_decode_scratch* s,
    float* hidden,
    uint32_t hidden_size
) {
    const uint32_t dense_intermediate = 12288u;
    const uint64_t bytes =
        2ull * (uint64_t)dense_intermediate * (uint64_t)hidden_size * 2ull +
        (uint64_t)hidden_size * (uint64_t)dense_intermediate * 2ull;
    // Bug #5 Fix: Dense MLP should use DENSE_MLP role, not ROUTER.
    // Fall back to ROUTER for backward compatibility with old manifests.
    const glm5_storage_raw_span_spec_t* span = find_raw_role_span_bytes(layer, GLM5_RAW_ROLE_DENSE_MLP, bytes, 0);
    if (!span) {
        span = find_raw_role_span_bytes(layer, GLM5_RAW_ROLE_ROUTER, bytes, 0);
    }
    if (!span) {
        return 0;
    }
    if (!glm5_raw_mlp_span_f32(
            engine, span, s->norm.data(), hidden_size, dense_intermediate,
            s->gate.data(), s->up.data(), s->mlp.data())) {
        return 0;
    }
    for (uint32_t i = 0; i < hidden_size; ++i) {
        hidden[i] += s->mlp[i];
    }
    return 1;
}

static int glm5_layer_moe_mlp_f32(
    glm5_pc_engine* engine,
    int32_t layer,
    glm5_decode_scratch* s,
    float* hidden,
    const glm5_storage_constants_t* c
) {
    const uint32_t hidden_size = c->hidden_size;
    const uint32_t expert_intermediate = c->expert_intermediate_size;
    std::fill(s->mlp.begin(), s->mlp.end(), 0.0f);

    const uint64_t shared_bytes =
        2ull * (uint64_t)expert_intermediate * (uint64_t)hidden_size * 2ull +
        (uint64_t)hidden_size * (uint64_t)expert_intermediate * 2ull;
    const glm5_storage_raw_span_spec_t* shared = find_raw_role_span_bytes(layer, GLM5_RAW_ROLE_SHARED, shared_bytes, 0);
    if (shared &&
        glm5_raw_mlp_span_f32(
            engine, shared, s->norm.data(), hidden_size, expert_intermediate,
            s->gate.data(), s->up.data(), s->shared.data())) {
        for (uint32_t i = 0; i < hidden_size; ++i) {
            s->mlp[i] += s->shared[i];
        }
    }

    if (!glm5_router_topk_f32(
            engine,
            layer,
            s->norm.data(),
            hidden_size,
            c->experts_per_moe_layer,
            GLM5_MOE_TOP_EXPERTS,
            &s->selected_experts,
            &s->selected_weights,
            &s->router_logits,
            &s->router_order)) {
        return 0;
    }

    // Fix 1: Expert-level parallelism.
    // Phase 1: Batch-request all experts upfront so prefetch can overlap.
    const glm5_tier_t target_tier = prefetch_plan_selected_tier(engine);
    const size_t selected_count = std::min(s->selected_experts.size(), (size_t)GLM5_MOE_TOP_EXPERTS);
    uint8_t stream_inflight[GLM5_MOE_TOP_EXPERTS] = {};
    {
        std::lock_guard<std::mutex> lock(engine->map_mutex);
        for (size_t e = 0; e < selected_count; ++e) {
            const int expert_id = s->selected_experts[e];
            const size_t eslot = expert_dense_slot(layer, expert_id);
            if (eslot < engine->expert_states.size()) {
                const glm5_block_state& state = engine->expert_states[eslot];
                stream_inflight[e] = state.active && state.tier == GLM5_TIER_STREAM ? 1u : 0u;
            }
        }
    }
    for (size_t e = 0; e < selected_count; ++e) {
        const int expert_id = s->selected_experts[e];
        if (stream_inflight[e]) {
            continue;
        }
        glm5_pc_engine_request_expert(engine, layer, expert_id,
            expert_triplet_bytes(engine, layer, expert_id), target_tier);
        enqueue_tensor_prefetch(engine, layer, expert_id, GLM5_PROJ_GATE, target_tier, 1);
        enqueue_tensor_prefetch(engine, layer, expert_id, GLM5_PROJ_UP, target_tier, 1);
        enqueue_tensor_prefetch(engine, layer, expert_id, GLM5_PROJ_DOWN, target_tier, 1);
    }
    if (engine->io_config.enable_compute_io_overlap) {
        engine->disk_stage.cv.notify_all();
    }

    // Phase 2: Run experts in parallel with per-expert scratch buffers.
    // Each expert gets its own gate/up/out buffers to prevent data races.
    const size_t num_selected = s->selected_experts.size();
    if (num_selected > s->expert_work.size()) {
        return 0;
    }
    glm5_expert_work_t* work = s->expert_work.data();
    for (size_t e = 0; e < num_selected; ++e) {
        work[e].gate = s->expert_gate_pool.data() + e * (size_t)expert_intermediate;
        work[e].up = s->expert_up_pool.data() + e * (size_t)expert_intermediate;
        work[e].out = s->expert_out_pool.data() + e * (size_t)hidden_size;
        std::fill(work[e].out, work[e].out + hidden_size, 0.0f);
        work[e].weight = s->selected_weights[e];
        work[e].expert_id = s->selected_experts[e];
        work[e].ok = 0;
    }

    // Fix 2: Flat row-parallel expert dispatch.
    // Instead of 8 sequential run_expert_triplet_f32 calls (each taking the full
    // pool via cpu_row_active), we pre-map all shards, then dispatch ALL experts'
    // rows in one parallel_rows call. This eliminates cpu_row_active serialization.
    const glm5_projection_shape_spec_t* gate_shape = glm5_storage_projection_shape(GLM5_PROJ_GATE);
    const glm5_projection_shape_spec_t* up_shape = glm5_storage_projection_shape(GLM5_PROJ_UP);
    const glm5_projection_shape_spec_t* down_shape = glm5_storage_projection_shape(GLM5_PROJ_DOWN);
    if (!gate_shape || !up_shape || !down_shape) return 0;

    // Pre-map all expert weight shards (these are cached, so fast for hot experts)
    struct expert_shards_t {
        const glm5_tensor_record* gate_rec = nullptr;
        const glm5_tensor_record* up_rec = nullptr;
        const glm5_tensor_record* down_rec = nullptr;
        mmap_context_t* gate_shard = nullptr;
        mmap_context_t* up_shard = nullptr;
        mmap_context_t* down_shard = nullptr;
    };
    expert_shards_t shards[GLM5_MOE_TOP_EXPERTS];
    std::atomic<int> expert_ok[GLM5_MOE_TOP_EXPERTS];
    for (uint32_t e = 0; e < GLM5_MOE_TOP_EXPERTS; ++e) {
        expert_ok[e].store(0, std::memory_order_relaxed);
    }
    for (size_t e = 0; e < num_selected; ++e) {
        shards[e].gate_rec = find_tensor_record(engine, layer, work[e].expert_id, GLM5_PROJ_GATE);
        shards[e].up_rec = find_tensor_record(engine, layer, work[e].expert_id, GLM5_PROJ_UP);
        shards[e].down_rec = find_tensor_record(engine, layer, work[e].expert_id, GLM5_PROJ_DOWN);
        if (!shards[e].gate_rec || !shards[e].up_rec || !shards[e].down_rec) {
            continue;
        }
        if (shards[e].gate_rec->info.rows != gate_shape->rows || shards[e].gate_rec->info.cols != gate_shape->cols ||
            shards[e].up_rec->info.rows != up_shape->rows || shards[e].up_rec->info.cols != up_shape->cols ||
            shards[e].down_rec->info.rows != down_shape->rows || shards[e].down_rec->info.cols != down_shape->cols) {
            continue;
        }
        shards[e].gate_shard = get_or_map_tensor_shard(engine, *shards[e].gate_rec);
        shards[e].up_shard = get_or_map_tensor_shard(engine, *shards[e].up_rec);
        shards[e].down_shard = get_or_map_tensor_shard(engine, *shards[e].down_rec);
        if (!shards[e].gate_shard || !shards[e].up_shard || !shards[e].down_shard) {
            continue;
        }
        expert_ok[e].store(1, std::memory_order_release);
    }

    // Phase 2a: Flat dispatch: gate+up+SiLU for all experts simultaneously.
    // Total rows = num_selected * intermediate_size, row i maps to expert e and row r.
    const uint32_t total_gate_rows = (uint32_t)num_selected * expert_intermediate;
    const int gate_up_ok = glm5_parallel_rows(engine, total_gate_rows, 0, [&](uint32_t flat_row) {
        const uint32_t e = flat_row / expert_intermediate;
        const uint32_t r = flat_row % expert_intermediate;
        if (!expert_ok[e].load(std::memory_order_acquire)) return 1;
        if (!glm5_dot_tensor_record_row(engine, shards[e].gate_rec, shards[e].gate_shard,
                r, s->norm.data(), hidden_size, &work[e].gate[r]) ||
            !glm5_dot_tensor_record_row(engine, shards[e].up_rec, shards[e].up_shard,
                r, s->norm.data(), hidden_size, &work[e].up[r])) {
            expert_ok[e].store(0, std::memory_order_release);
            return 1;
        }
        work[e].gate[r] = glm5_generation_silu(work[e].gate[r]) * work[e].up[r];
        return 1;
    });
    if (!gate_up_ok) return 0;

    // Phase 2b: Flat dispatch: down_proj for all experts simultaneously.
    const uint32_t total_down_rows = (uint32_t)num_selected * hidden_size;
    const int down_ok = glm5_parallel_rows(engine, total_down_rows, 0, [&](uint32_t flat_row) {
        const uint32_t e = flat_row / hidden_size;
        const uint32_t r = flat_row % hidden_size;
        if (!expert_ok[e].load(std::memory_order_acquire)) return 1;
        if (!glm5_dot_tensor_record_row(engine, shards[e].down_rec, shards[e].down_shard,
                r, work[e].gate, expert_intermediate, &work[e].out[r])) {
            expert_ok[e].store(0, std::memory_order_release);
        }
        return 1;
    });
    if (!down_ok) return 0;
    for (size_t e = 0; e < num_selected; ++e) {
        work[e].ok = expert_ok[e].load(std::memory_order_acquire);
    }

    // Phase 3: Accumulate weighted expert outputs into mlp.
    for (size_t e = 0; e < num_selected; ++e) {
        if (!work[e].ok) continue;
        glm5_axpy_f32(s->mlp.data(), work[e].out, work[e].weight, hidden_size);
    }

    for (uint32_t i = 0; i < hidden_size; ++i) {
        hidden[i] += s->mlp[i];
    }
    return 1;
}

static int glm5_forward_one_token_f32(
    glm5_pc_engine* engine,
    uint32_t token_id,
    glm5_decode_scratch* s,
    glm5_generation_stats_t* stats,
    uint32_t position,
    int pre_allocate_kv
) {
    const glm5_storage_constants_t* c = glm5_storage_constants();
    if (!engine || !c || !s || c->hidden_size == 0 || token_id >= c->vocab_size) {
        glm5_generation_set_error(stats, "invalid token forward arguments");
        return 0;
    }
    glm5_decode_scratch_resize(s, c, pre_allocate_kv);
    glm5_kv_cache_track_ram(engine, &s->kv_cache);
    float* hidden = s->hidden.data();
    if (!glm5_pc_engine_embed_token_f32(engine, token_id, hidden, c->hidden_size)) {
        glm5_generation_set_error(stats, "failed to read token embedding");
        return 0;
    }

    for (uint32_t layer = 0; layer < c->num_hidden_layers; ++layer) {
        // Fix #5: Lookahead attention prefetch: enqueue layer+2 attention spans
        // so DMA has time to complete before that layer's attention begins.
        // Current-layer synchronous mmap hints were too late.
        if (layer + 2u < c->num_hidden_layers &&
            glm5_can_enqueue_attention_lookahead(engine, (int32_t)(layer + 2u))) {
            glm5_enqueue_attention_prefetch(engine, (int32_t)(layer + 2u));
        }
        if (!glm5_layer_attention_f32(engine, (int32_t)layer, s, s->hidden.data(), c->hidden_size, position)) {
            glm5_generation_set_error(stats, "attention block failed");
            return 0;
        }
        if (!glm5_pc_engine_raw_rmsnorm_f32(
                engine, (int32_t)layer, GLM5_RAW_TENSOR_POST_ATTENTION_LAYER_NORM,
                hidden, c->hidden_size, s->norm.data(), c->hidden_size, 1.0e-6f)) {
            glm5_generation_set_error(stats, "post-attention rmsnorm failed");
            return 0;
        }
        const int is_moe = layer >= c->first_moe_layer && layer <= c->last_moe_layer;

        // Next-layer attention VRAM prefetch: enqueue layer+1 attention spans
        // so DMA overlaps with current layer's MLP compute.
        if (layer + 1u < c->num_hidden_layers) {
            glm5_enqueue_attention_prefetch(engine, (int32_t)(layer + 1u));
        }

        const int ok = is_moe ?
            glm5_layer_moe_mlp_f32(engine, (int32_t)layer, s, s->hidden.data(), c) :
            glm5_layer_dense_mlp_f32(engine, (int32_t)layer, s, s->hidden.data(), c->hidden_size);
        if (!ok) {
            glm5_generation_set_error(stats, "mlp/moe block failed");
            return 0;
        }
    }

    if (!glm5_pc_engine_raw_rmsnorm_f32(
            engine, (int32_t)c->num_hidden_layers, GLM5_RAW_TENSOR_INPUT_LAYER_NORM,
            hidden, c->hidden_size, s->norm.data(), c->hidden_size, 1.0e-6f)) {
        glm5_generation_set_error(stats, "final rmsnorm failed");
        return 0;
    }
    s->hidden.swap(s->norm);
    return 1;
}

int glm5_pc_engine_generate_token_ids(
    glm5_pc_engine_t* engine,
    const int32_t* input_token_ids,
    uint32_t input_token_count,
    const glm5_generation_config_t* config,
    uint32_t* out_token_ids,
    uint32_t out_token_capacity,
    glm5_generation_stats_t* out_stats
) {
    if (out_stats) {
        std::memset(out_stats, 0, sizeof(*out_stats));
    }
    const glm5_storage_constants_t* c = glm5_storage_constants();
    if (!engine || !c || !input_token_ids || input_token_count == 0 || !out_token_ids || out_token_capacity == 0) {
        glm5_generation_set_error(out_stats, "invalid generation arguments");
        return 0;
    }
    glm5_forward_status_t forward{};
    if (!glm5_pc_engine_get_forward_status(engine, &forward) ||
        !forward.storage_state_valid ||
        !forward.tensor_table_loaded ||
        !forward.expert_triplet_available ||
        !forward.embedding_ready ||
        !forward.lm_head_ready ||
        !forward.attention_ready ||
        !forward.sampler_ready) {
        glm5_generation_set_error(out_stats, "full generation path is not ready");
        return 0;
    }

    uint32_t max_new_tokens = config && config->max_new_tokens ? config->max_new_tokens : 128u;
    if (max_new_tokens > out_token_capacity) {
        max_new_tokens = out_token_capacity;
    }
    if (max_new_tokens == 0) {
        glm5_generation_set_error(out_stats, "max_new_tokens resolved to zero");
        return 0;
    }
    if (input_token_count > GLM5_MLA_MAX_SEQ_LEN) {
        glm5_generation_set_error(out_stats, "input exceeds GLM5 MLA context length");
        return 0;
    }
    const uint32_t max_new_by_context = GLM5_MLA_MAX_SEQ_LEN - input_token_count + 1u;
    if (max_new_tokens > max_new_by_context) {
        max_new_tokens = max_new_by_context;
    }
    if (max_new_tokens == 0) {
        glm5_generation_set_error(out_stats, "context window has no room for generation");
        return 0;
    }

    int32_t current = input_token_ids[input_token_count - 1u];
    if (current < 0 || (uint32_t)current >= c->vocab_size) {
        glm5_generation_set_error(out_stats, "last input token is outside vocabulary");
        return 0;
    }

    glm5_decode_scratch scratch;
    glm5_kv_cache_ram_scope kv_scope{engine, &scratch.kv_cache};
    glm5_kv_decode_cache_ram_scope decode_cache_scope{engine, &scratch};
    uint32_t produced = 0;
    int failed = 0;
    float last_logit = 0.0f;
    const int pre_allocate_kv =
        glm5_generation_should_preallocate_kv_cache(engine, c->num_hidden_layers);

    // Fix 1+2: Initialize KV cache and position counter
    glm5_decode_scratch_resize(&scratch, c, pre_allocate_kv);
    glm5_kv_cache_track_ram(engine, &scratch.kv_cache);
    scratch.position = 0;

    // Layer-first prefill: invert token-by-layer into layer-by-token.
    // Weight IO becomes N*80 to 80 layer passes when page cache is hot.
    // Attention remains O(N^2), but weights stay hotter across prompt tokens.
    // Memory is N*hidden_size*4 bytes, about 14.7 MiB for N=512 and hidden=7168.
    const uint32_t prefill_count = input_token_count - 1u;
    if (prefill_count > 0) {
        // Validate all prompt tokens first
        for (uint32_t i = 0; i < prefill_count; ++i) {
            if (input_token_ids[i] < 0 || (uint32_t)input_token_ids[i] >= c->vocab_size) {
                glm5_generation_set_error(out_stats, "prompt token is outside vocabulary");
                return 0;
            }
        }

        if (!glm5_generation_should_use_layer_first_prefill(engine, prefill_count, c->hidden_size)) {
            for (uint32_t i = 0; i < prefill_count; ++i) {
                if (!glm5_forward_one_token_f32(
                        engine, (uint32_t)input_token_ids[i], &scratch, out_stats,
                        scratch.position, pre_allocate_kv)) {
                    return 0;
                }
                ++scratch.position;
            }
        } else {
            // Allocate batch hidden states: one per prefill token
            const size_t hs = c->hidden_size;
            std::vector<float> batch_hidden(prefill_count * hs);

            // Phase 1: Embed all tokens in parallel
            for (uint32_t i = 0; i < prefill_count; ++i) {
                if (!glm5_pc_engine_embed_token_f32(
                        engine, (uint32_t)input_token_ids[i],
                        batch_hidden.data() + i * hs, (uint32_t)hs)) {
                    glm5_generation_set_error(out_stats, "prefill embedding failed");
                    return 0;
                }
            }

            // Ensure scratch is sized
            glm5_decode_scratch_resize(&scratch, c, pre_allocate_kv);
            glm5_kv_cache_track_ram(engine, &scratch.kv_cache);

            // Bug 8: Pre-allocate KV cache capacity to avoid reallocation inside tight loop
            // Without pre-allocation, ensure_capacity() inside glm5_layer_attention_f32 triggers
            // ~176MB reallocation at t=1024, ~353MB at t=2048, blocking GPU prefetch pipeline
            if (!pre_allocate_kv && prefill_count > scratch.kv_cache.max_seq_len) {
                scratch.kv_cache.ensure_capacity(prefill_count - 1);
                glm5_kv_cache_track_ram(engine, &scratch.kv_cache);
            }

            // Phase 2: layer-first, process all prompt tokens per layer.
            for (uint32_t layer = 0; layer < c->num_hidden_layers; ++layer) {
                // Lookahead prefetch: load layer+2 attention spans into VRAM/page cache
                if (layer + 2u < c->num_hidden_layers &&
                    glm5_can_enqueue_attention_lookahead(engine, (int32_t)(layer + 2u))) {
                    glm5_enqueue_attention_prefetch(engine, (int32_t)(layer + 2u));
                }

                // Inner loop: all tokens through this layer's attention + MLP
                if (layer + 1u < c->num_hidden_layers) {
                    glm5_enqueue_attention_prefetch(engine, (int32_t)(layer + 1u));
                }

                for (uint32_t t = 0; t < prefill_count; ++t) {
                    float* current_hidden = batch_hidden.data() + t * hs;

                    if (!glm5_layer_attention_f32(engine, (int32_t)layer, &scratch, current_hidden, (uint32_t)hs, t)) {
                        glm5_generation_set_error(out_stats, "prefill attention failed");
                        return 0;
                    }

                    if (!glm5_pc_engine_raw_rmsnorm_f32(
                            engine, (int32_t)layer, GLM5_RAW_TENSOR_POST_ATTENTION_LAYER_NORM,
                            current_hidden, (uint32_t)hs, scratch.norm.data(), (uint32_t)hs, 1.0e-6f)) {
                        glm5_generation_set_error(out_stats, "prefill post-attention norm failed");
                        return 0;
                    }

                    const int is_moe = layer >= c->first_moe_layer && layer <= c->last_moe_layer;
                    const int mlp_ok = is_moe ?
                        glm5_layer_moe_mlp_f32(engine, (int32_t)layer, &scratch, current_hidden, c) :
                        glm5_layer_dense_mlp_f32(engine, (int32_t)layer, &scratch, current_hidden, (uint32_t)hs);
                    if (!mlp_ok) {
                        glm5_generation_set_error(out_stats, "prefill mlp failed");
                        return 0;
                    }
                }
            }

            float* last_hidden = batch_hidden.data() + (prefill_count - 1u) * hs;
            if (!glm5_pc_engine_raw_rmsnorm_f32(
                    engine, (int32_t)c->num_hidden_layers, GLM5_RAW_TENSOR_INPUT_LAYER_NORM,
                    last_hidden, (uint32_t)hs, scratch.norm.data(), (uint32_t)hs, 1.0e-6f)) {
                glm5_generation_set_error(out_stats, "prefill final norm failed");
                return 0;
            }
            std::memcpy(scratch.hidden.data(), scratch.norm.data(), hs * sizeof(float));
            scratch.position = prefill_count;
        }
    }
    // Decode: generate new tokens
    for (; produced < max_new_tokens; ++produced) {
        if (!glm5_forward_one_token_f32(
                engine, (uint32_t)current, &scratch, out_stats, scratch.position, pre_allocate_kv)) {
            failed = 1;
            break;
        }
        uint32_t next_token = 0;
        float next_logit = 0.0f;
        if (!glm5_pc_engine_lm_head_top1_f32(engine, scratch.hidden.data(), c->hidden_size, &next_token, &next_logit)) {
            glm5_generation_set_error(out_stats, "failed to sample from lm_head");
            failed = 1;
            break;
        }
        out_token_ids[produced] = next_token;
        last_logit = next_logit;
        current = (int32_t)next_token;
        ++scratch.position;
        if (config && config->stop_on_eos && next_token == config->eos_token_id) {
            ++produced;
            break;
        }
    }

    if (out_stats) {
        out_stats->prompt_tokens = input_token_count;
        out_stats->completion_tokens = produced;
        out_stats->finish_reason = produced >= max_new_tokens ? 1u : 0u;
        out_stats->last_logit = last_logit;
    }
    return !failed && produced > 0 ? 1 : 0;
}
