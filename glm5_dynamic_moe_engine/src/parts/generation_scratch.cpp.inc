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
    // BUGFIX 53: Issue 24 - Removed attn_kv_b (dead allocation) ★
    // Problem: attn_kv_b allocated hidden_size floats per scratch instance but never used.
    // Comment said "kept for API compat; unused with absorption" but this is internal struct.
    // Solution: Remove field entirely. No ABI impact (internal struct, not exposed in C API).
    // Saves hidden_size * 4 bytes per decode scratch (6144 * 4 = 24KB per instance).
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
    // BUGFIX 53: attn_kv_b removed (dead allocation)
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
