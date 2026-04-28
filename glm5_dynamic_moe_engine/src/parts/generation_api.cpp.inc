static void glm5_generation_set_error(glm5_generation_stats_t* stats, const char* message) {
    if (!stats || !message) {
        return;
    }
    std::snprintf(stats->error, sizeof(stats->error), "%s", message);
}

static float glm5_fast_exp_approx(float x) {
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

static float glm5_generation_silu(float x) {
#ifdef GLM5_EXACT_SILU
    return x / (1.0f + std::exp(-x));
#else
    return x / (1.0f + glm5_fast_exp_approx(-x));
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
    std::nth_element(order, order + top_k, order + experts, cmp);
    std::sort(order, order + top_k, cmp);

    float max_logit = -std::numeric_limits<float>::infinity();
    for (uint32_t i = 0; i < top_k; ++i) {
        max_logit = std::max(max_logit, (*scratch_logits)[order[i]]);
    }
    float denom = 0.0f;
    out_experts->clear();
    out_weights->clear();
    for (uint32_t i = 0; i < top_k; ++i) {
        const float w = std::exp((*scratch_logits)[order[i]] - max_logit);
        out_experts->push_back((int)order[i]);
        out_weights->push_back(w);
        denom += w;
    }
    if (denom <= 0.0f) {
        return 0;
    }
    for (float& w : *out_weights) {
        w /= denom;
    }
    return 1;
}

// MLA constants for GLM-5.1
static constexpr uint32_t GLM5_MLA_KV_COMPRESSED_DIM = 512u;
static constexpr uint32_t GLM5_MLA_K_ROPE_DIM = 64u;
static constexpr uint32_t GLM5_MLA_KV_ENTRY_DIM = GLM5_MLA_KV_COMPRESSED_DIM + GLM5_MLA_K_ROPE_DIM; // 576
static constexpr uint32_t GLM5_MLA_MAX_SEQ_LEN = 8192u;
static constexpr uint32_t GLM5_MLA_KV_INITIAL_CHUNK = 256u;
static constexpr uint32_t GLM5_MLA_NUM_HEADS = 128u;
static constexpr uint32_t GLM5_MLA_HEAD_DIM = 128u;
static constexpr uint32_t GLM5_MLA_NOPE_DIM = 96u;  // K_nope per head = 12288/128
static constexpr uint32_t GLM5_MLA_ROPE_DIM = 32u;  // Q rope per head = 128-96
static constexpr uint32_t GLM5_MLA_V_HEAD_DIM = 128u;
static constexpr float GLM5_ROPE_THETA = 500000.0f;
static constexpr uint32_t GLM5_MOE_TOP_EXPERTS = 8u;

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

    void init(uint32_t layers, bool pre_allocate = false) {
        num_layers = layers;
        max_seq_len = pre_allocate ? GLM5_MLA_MAX_SEQ_LEN : GLM5_MLA_KV_INITIAL_CHUNK;
        seq_len = 0;
        data.assign((size_t)layers * max_seq_len * GLM5_MLA_KV_ENTRY_DIM, 0u);
    }

    void ensure_capacity(uint32_t needed_pos) {
        if (needed_pos < max_seq_len) return;
        uint32_t new_cap = max_seq_len;
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

    // Store float32 entry as bf16
    void store_entry(uint32_t layer, uint32_t pos, const float* src, uint32_t count) {
        uint16_t* dst = data.data() + ((size_t)layer * max_seq_len + pos) * GLM5_MLA_KV_ENTRY_DIM;
        for (uint32_t i = 0; i < count; ++i) {
            dst[i] = glm5_f32_to_bf16(src[i]);
        }
    }

    // Load bf16 entry to float32 buffer
    void load_entry(uint32_t layer, uint32_t pos, float* dst, uint32_t count) const {
        const uint16_t* src = data.data() + ((size_t)layer * max_seq_len + pos) * GLM5_MLA_KV_ENTRY_DIM;
        for (uint32_t i = 0; i < count; ++i) {
            dst[i] = glm5_bf16_to_f32(src[i]);
        }
    }
};

static uint64_t glm5_kv_cache_bytes(uint32_t layers, uint32_t seq_len) {
    if (layers == 0 || seq_len == 0) {
        return 0;
    }
    constexpr uint64_t entry_bytes = (uint64_t)GLM5_MLA_KV_ENTRY_DIM * sizeof(float);
    if ((uint64_t)layers > UINT64_MAX / (uint64_t)seq_len ||
        (uint64_t)layers * (uint64_t)seq_len > UINT64_MAX / entry_bytes) {
        return UINT64_MAX;
    }
    return (uint64_t)layers * (uint64_t)seq_len * entry_bytes;
}

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
        const uint64_t guard_bytes = std::max<uint64_t>(512ull * 1024ull * 1024ull, full_kv_bytes / 8ull);
        if (available <= guard_bytes || full_kv_bytes > available - guard_bytes) {
            return 0;
        }
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
    // Per-head scratch for bf16 KV decode (576 floats × num_heads)
    std::vector<float> kv_decode_buf;
};

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
    const size_t needed = (size_t)GLM5_MLA_MAX_SEQ_LEN * (size_t)half_dim;
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
    for (uint32_t pos = 0; pos < GLM5_MLA_MAX_SEQ_LEN; ++pos) {
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
    s->attn_kv_a.resize(576);
    s->attn_kv_norm.resize(512);
    // attn_kv_b no longer allocated — KV absorption reads W_kv_b directly
    s->attn_value.resize(16384);
    s->mlp.resize(hidden);
    s->shared.resize(hidden);
    s->expert.resize(hidden);
    s->gate.resize(std::max<uint32_t>(12288u, expert_intermediate));
    // Absorption buffers
    s->q_a.resize(2048);
    s->q_a_norm.resize(2048);
    s->q_b.resize(16384);
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
    // Per-head scratch for bf16→f32 KV decode in softmax loop
    s->kv_decode_buf.resize((size_t)GLM5_MLA_NUM_HEADS * GLM5_MLA_KV_ENTRY_DIM);
    // Fast path: allocate full decode KV once to avoid mid-generation realloc/copy.
    if (s->kv_cache.num_layers != c->num_hidden_layers ||
        s->kv_cache.max_seq_len < GLM5_MLA_MAX_SEQ_LEN) {
        s->kv_cache.init(c->num_hidden_layers, pre_allocate_kv != 0);
        s->position = 0;
    }
}

// Bug #1 Fix: Full MLA attention with Q projection, RoPE, scaled dot-product.
// GLM4/GLM5 Multi-head Latent Attention (MLA):
//   KV path: hidden -> kv_a_proj(576) -> [kv_a_norm(512), mqa_rope(64)]
//            kv_a_norm -> kv_b_proj(28672) -> [K_nope(12288), V(16384)]
//   Q path:  hidden -> q_a_proj(2048) -> q_a_norm(2048) -> q_b_proj(16384+4096=20480)
//            -> [Q_nope(12288), Q_rope(4096)] split by heads
//   Score = softmax(Q_nope*K_nope + Q_rope*K_rope) * V -> o_proj -> attn output

// Fix #4: Default base changed from 10000 to GLM5_ROPE_THETA (500000)
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
    if (position < GLM5_MLA_MAX_SEQ_LEN &&
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
    uint32_t queued_spans[sizeof(kinds) / sizeof(kinds[0])] = {};
    uint32_t queued_count = 0;
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

static int glm5_layer_attention_f32(
    glm5_pc_engine* engine,
    int32_t layer,
    glm5_decode_scratch* s,
    uint32_t hidden_size,
    uint32_t position
) {
    constexpr uint32_t cdim = GLM5_MLA_KV_COMPRESSED_DIM; // 512
    constexpr uint32_t num_heads = GLM5_MLA_NUM_HEADS;     // 128
    constexpr uint32_t head_dim = GLM5_MLA_HEAD_DIM;       // 128
    constexpr uint32_t nope_dim = GLM5_MLA_NOPE_DIM;       // 96
    constexpr uint32_t rope_dim = GLM5_MLA_ROPE_DIM;       // 32
    constexpr uint32_t v_head_dim = GLM5_MLA_V_HEAD_DIM;   // 128
    constexpr uint64_t kv_b_row_bytes = cdim * 2ull;       // bf16

    // --- Pre-attention RMSNorm ---
    if (!glm5_pc_engine_raw_rmsnorm_f32(
            engine, layer, GLM5_RAW_TENSOR_INPUT_LAYER_NORM,
            s->hidden.data(), hidden_size, s->norm.data(), hidden_size, 1.0e-6f)) {
        return 0;
    }

    // --- KV path: kv_a_proj -> kv_a_norm (NO kv_b_proj — absorbed) ---
    // kv_a is small (576 outputs) — runs inline. No std::async: creating
    // OS threads 80 layers × N tokens = 100K+ create/destroy per inference.
    if (!glm5_pc_engine_raw_linear_f32(
            engine, layer, GLM5_RAW_TENSOR_KV_A_PROJ_WITH_MQA,
            s->norm.data(), hidden_size, s->attn_kv_a.data(), 576u)) {
        return 0;
    }
    if (!glm5_pc_engine_raw_rmsnorm_f32(
            engine, layer, GLM5_RAW_TENSOR_KV_A_LAYER_NORM,
            s->attn_kv_a.data(), 512u, s->attn_kv_norm.data(), 512u, 1.0e-6f)) {
        return 0;
    }
    float* k_rope = s->attn_kv_a.data() + 512u;

    // --- Q path: q_a_proj -> q_a_norm -> q_b_proj ---
    if (!glm5_pc_engine_raw_linear_f32(
            engine, layer, GLM5_RAW_TENSOR_Q_A_PROJ,
            s->norm.data(), hidden_size, s->q_a.data(), 2048u) ||
        !glm5_pc_engine_raw_rmsnorm_f32(
            engine, layer, GLM5_RAW_TENSOR_Q_A_LAYER_NORM,
            s->q_a.data(), 2048u, s->q_a_norm.data(), 2048u, 1.0e-6f) ||
        !glm5_pc_engine_raw_linear_f32(
            engine, layer, GLM5_RAW_TENSOR_Q_B_PROJ,
            s->q_a_norm.data(), 2048u, s->q_b.data(), 16384u)) {
        return 0;
    }

    // Apply RoPE — K rope is single vector, Q rope parallelized over 128 heads
    glm5_rope_inplace(k_rope, GLM5_MLA_K_ROPE_DIM, position, s->rope_k_cos, s->rope_k_sin);
    glm5_parallel_rows(engine, num_heads, 0, [&](uint32_t h) {
        glm5_rope_inplace(
            s->q_b.data() + h * head_dim + nope_dim,
            rope_dim,
            position,
            s->rope_q_cos,
            s->rope_q_sin
        );
        return 1;
    });

    // Cache compressed KV + RoPE'd k_rope as bf16
    s->kv_cache.ensure_capacity(position);
    if (position < s->kv_cache.max_seq_len) {
        // Build float32 entry, then store as bf16
        float kv_entry[GLM5_MLA_KV_ENTRY_DIM];
        std::memcpy(kv_entry, s->attn_kv_norm.data(), cdim * sizeof(float));
        std::memcpy(kv_entry + cdim, k_rope, GLM5_MLA_K_ROPE_DIM * sizeof(float));
        s->kv_cache.store_entry((uint32_t)layer, position, kv_entry, GLM5_MLA_KV_ENTRY_DIM);
    }
    const uint32_t seq_len = std::min(position + 1u, s->kv_cache.max_seq_len);

    // --- Map W_kv_b for KV absorption ---
    glm5_raw_tensor_desc kv_b_desc{};
    const glm5_storage_raw_span_spec_t* kv_b_span = nullptr;
    if (!find_raw_tensor_desc_and_span(layer, GLM5_RAW_TENSOR_KV_B_PROJ, &kv_b_desc, &kv_b_span)) {
        return 0;
    }
    mmap_context_t* kv_b_mapped = map_raw_span_part(engine, kv_b_span);
    if (!raw_span_bounds_ok(kv_b_span, kv_b_mapped)) return 0;
    const uint8_t* kv_b_base = raw_span_ptr(
        kv_b_span, kv_b_mapped, kv_b_desc.rel_offset, kv_b_desc.bytes);
    if (!kv_b_base) return 0;

    // --- Q-side absorption: q_absorbed_h[512] = q_nope_h @ W_kv_b_nope_h^T ---
    // Parallelized over heads — each head reads its own 96 W_kv_b rows,
    // writes to its own qa_h slice. Zero contention.
    std::fill(s->q_absorbed.begin(), s->q_absorbed.end(), 0.0f);
    glm5_parallel_rows(engine, num_heads, 0, [&](uint32_t h) {
        float* qa_h = s->q_absorbed.data() + h * cdim;
        for (uint32_t d = 0; d < nope_dim; ++d) {
            const float qv = s->q_b[h * head_dim + d];
            if (std::abs(qv) < 1e-12f) continue;
            const uint8_t* w_row = kv_b_base + kv_b_row_bytes * (h * nope_dim + d);
            for (uint32_t j = 0; j < cdim; ++j) {
                uint16_t bf16;
                std::memcpy(&bf16, w_row + j * 2ull, sizeof(bf16));
                qa_h[j] += qv * glm5_raw_bf16_to_f32(bf16);
            }
        }
        return 1;
    });

    // --- Online softmax: head-outer (parallel), position-inner ---
    // Each head has independent running max/denom/c_weighted — no race.
    // c_kv data (seq_len * 512 floats) fits in L3 and is read-only shared.
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

        for (uint32_t p = 0; p < seq_len; ++p) {
            const float* c_kv;
            const float* p_k_rope;
            if (p == position) {
                c_kv = s->attn_kv_norm.data();
                p_k_rope = k_rope;
            } else {
                // Decode bf16 KV entry to per-head float32 scratch
                float* decode = s->kv_decode_buf.data() + h * GLM5_MLA_KV_ENTRY_DIM;
                s->kv_cache.load_entry((uint32_t)layer, p, decode, GLM5_MLA_KV_ENTRY_DIM);
                c_kv = decode;
                p_k_rope = decode + cdim;
            }

            float score = 0.0f;
            for (uint32_t j = 0; j < cdim; ++j) {
                score += qa_h[j] * c_kv[j];
            }
            for (uint32_t d = 0; d < rope_dim && d < GLM5_MLA_K_ROPE_DIM; ++d) {
                score += q_rope_h[d] * p_k_rope[d];
            }
            score *= scale;

            if (score > h_max) {
                const float rescale = std::exp(h_max - score);
                h_denom *= rescale;
                for (uint32_t j = 0; j < cdim; ++j) {
                    cw_h[j] *= rescale;
                }
                h_max = score;
            }
            const float w = std::exp(score - h_max);
            h_denom += w;
            for (uint32_t j = 0; j < cdim; ++j) {
                cw_h[j] += w * c_kv[j];
            }
        }

        // Normalize
        if (h_denom > 0.0f) {
            const float inv = 1.0f / h_denom;
            for (uint32_t j = 0; j < cdim; ++j) {
                cw_h[j] *= inv;
            }
        }
        s->head_max[h] = h_max;
        s->head_denom[h] = h_denom;
        return 1;
    });

    // --- V-side absorption: head-parallel expansion ---
    const uint8_t* kv_b_v_base = kv_b_base + kv_b_row_bytes * 12288ull;
    glm5_parallel_rows(engine, num_heads, 0, [&](uint32_t h) {
        float* out_h = s->attn_value.data() + h * v_head_dim;
        const float* cw_h = s->c_weighted.data() + h * cdim;
        for (uint32_t d = 0; d < v_head_dim; ++d) {
            const uint8_t* w_row = kv_b_v_base + kv_b_row_bytes * (h * v_head_dim + d);
            out_h[d] = glm5_dot_bf16_row(w_row, cw_h, cdim);
        }
        return 1;
    });

    // --- O projection: 16384 -> hidden_size ---
    if (!glm5_pc_engine_raw_linear_f32(
            engine, layer, GLM5_RAW_TENSOR_O_PROJ,
            s->attn_value.data(), 16384u, s->attn.data(), hidden_size)) {
        return 0;
    }
    for (uint32_t i = 0; i < hidden_size; ++i) {
        s->hidden[i] += s->attn[i];
    }
    return 1;
}

static int glm5_layer_dense_mlp_f32(
    glm5_pc_engine* engine,
    int32_t layer,
    glm5_decode_scratch* s,
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
        s->hidden[i] += s->mlp[i];
    }
    return 1;
}

static int glm5_layer_moe_mlp_f32(
    glm5_pc_engine* engine,
    int32_t layer,
    glm5_decode_scratch* s,
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
    for (size_t e = 0; e < s->selected_experts.size(); ++e) {
        const int expert_id = s->selected_experts[e];
        glm5_pc_engine_request_expert(engine, layer, expert_id,
            expert_triplet_bytes(engine, layer, expert_id), target_tier);
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

    // Sequential dispatch: each expert gets the full thread pool for its matmul.
    // Expert-parallel via glm5_parallel_rows was removed because cpu_row_active
    // contention forces each expert's inner matmul to single-thread.
    for (size_t e = 0; e < num_selected; ++e) {
        work[e].ok = glm5_pc_engine_run_expert_triplet_f32(
            engine, layer, work[e].expert_id,
            s->norm.data(), hidden_size,
            work[e].gate, work[e].up,
            expert_intermediate,
            work[e].out, hidden_size) ? 1 : 0;
    }

    // Phase 3: Accumulate weighted expert outputs into mlp.
    for (size_t e = 0; e < num_selected; ++e) {
        if (!work[e].ok) continue;
        const float w = work[e].weight;
        const float* out_e = work[e].out;
        float* mlp = s->mlp.data();
        uint32_t i = 0;
#if defined(__AVX2__)
        const __m256 vw = _mm256_set1_ps(w);
        for (; i + 7u < hidden_size; i += 8u) {
            __m256 vm = _mm256_loadu_ps(mlp + i);
            __m256 vo = _mm256_loadu_ps(out_e + i);
            vm = _mm256_fmadd_ps(vw, vo, vm);
            _mm256_storeu_ps(mlp + i, vm);
        }
#endif
        for (; i < hidden_size; ++i) {
            mlp[i] += w * out_e[i];
        }
    }

    for (uint32_t i = 0; i < hidden_size; ++i) {
        s->hidden[i] += s->mlp[i];
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
    if (!glm5_pc_engine_embed_token_f32(engine, token_id, s->hidden.data(), c->hidden_size)) {
        glm5_generation_set_error(stats, "failed to read token embedding");
        return 0;
    }

    for (uint32_t layer = 0; layer < c->num_hidden_layers; ++layer) {
        // Fix #5: Lookahead attention prefetch — enqueue layer+2 attention spans
        // so DMA has time to complete before that layer's attention begins.
        // Current-layer synchronous mmap hints were too late.
        if (layer + 2u < c->num_hidden_layers) {
            glm5_enqueue_attention_prefetch(engine, (int32_t)(layer + 2u));
        }
        if (!glm5_layer_attention_f32(engine, (int32_t)layer, s, c->hidden_size, position)) {
            glm5_generation_set_error(stats, "attention block failed");
            return 0;
        }
        if (!glm5_pc_engine_raw_rmsnorm_f32(
                engine, (int32_t)layer, GLM5_RAW_TENSOR_POST_ATTENTION_LAYER_NORM,
                s->hidden.data(), c->hidden_size, s->norm.data(), c->hidden_size, 1.0e-6f)) {
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
            glm5_layer_moe_mlp_f32(engine, (int32_t)layer, s, c) :
            glm5_layer_dense_mlp_f32(engine, (int32_t)layer, s, c->hidden_size);
        if (!ok) {
            glm5_generation_set_error(stats, "mlp/moe block failed");
            return 0;
        }
    }

    if (!glm5_pc_engine_raw_rmsnorm_f32(
            engine, (int32_t)c->num_hidden_layers, GLM5_RAW_TENSOR_INPUT_LAYER_NORM,
            s->hidden.data(), c->hidden_size, s->norm.data(), c->hidden_size, 1.0e-6f)) {
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

    int32_t current = input_token_ids[input_token_count - 1u];
    if (current < 0 || (uint32_t)current >= c->vocab_size) {
        glm5_generation_set_error(out_stats, "last input token is outside vocabulary");
        return 0;
    }

    glm5_decode_scratch scratch;
    uint32_t produced = 0;
    int failed = 0;
    float last_logit = 0.0f;
    const int pre_allocate_kv =
        glm5_generation_should_preallocate_kv_cache(engine, c->num_hidden_layers);

    // Fix 1+2: Initialize KV cache and position counter
    glm5_decode_scratch_resize(&scratch, c, pre_allocate_kv);
    scratch.position = 0;

    // Prefill: process prompt tokens, building KV cache.
    // WARNING Fix #2: Token-by-token prefill is O(N²) because each forward pass
    // re-runs attention over all cached positions. For seq_len=512 this costs
    // ~50x more than batched prefill. To fix: refactor glm5_layer_attention_f32
    // to accept (T, hidden) batched input, move the layer loop outside the token
    // loop, and apply causal mask on the QK^T matrix in one pass per layer.
    // This is a major architectural change tracked separately.
    for (uint32_t i = 0; i + 1u < input_token_count; ++i) {
        const int32_t prompt_token = input_token_ids[i];
        if (prompt_token < 0 || (uint32_t)prompt_token >= c->vocab_size) {
            glm5_generation_set_error(out_stats, "prompt token is outside vocabulary");
            return 0;
        }
        if (!glm5_forward_one_token_f32(
                engine, (uint32_t)prompt_token, &scratch, out_stats, scratch.position, pre_allocate_kv)) {
            return 0;
        }
        ++scratch.position;
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
