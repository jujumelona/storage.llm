struct moe_expert_work_t {
    float* gate = nullptr;
    float* up = nullptr;
    float* out = nullptr;
    float weight = 0.0f;
    int expert_id = 0;
    int ok = 0;
};

struct moe_decode_scratch {
    std::vector<float> hidden;
    std::vector<float> residual;
    std::vector<float> norm;
    std::vector<float> attn;
    std::vector<float> attn_kv_a;
    std::vector<float> attn_kv_norm;
    // BUGFIX 53: Issue 24 - Removed attn_kv_b (dead allocation) ??
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
    std::vector<moe_expert_work_t> expert_work;
    std::vector<float> expert_gate_pool;
    std::vector<float> expert_up_pool;
    std::vector<float> expert_out_pool;
    moe_kv_cache kv_cache;
    uint32_t position = 0;
    // Absorption scratch (moved from thread_local)
    std::vector<float> q_a;
    std::vector<float> q_a_norm;
    std::vector<float> q_b;
    std::vector<float> q_absorbed;    // [num_heads * kv_lora_rank]
    std::vector<float> c_weighted;    // [num_heads * kv_lora_rank]
    std::vector<float> head_max;     // [num_heads]
    std::vector<float> head_denom;   // [num_heads]
    std::vector<float> rope_k_cos;
    std::vector<float> rope_k_sin;
    std::vector<float> rope_q_cos;
    std::vector<float> rope_q_sin;
    std::vector<float> kv_entry;
    // Fix 38: kv_decode_buf removed (legacy dead allocation, same pattern as BUGFIX 53)
    std::vector<float> kv_decode_flat;
    std::vector<uint32_t> kv_decoded_up_to;
    uint64_t kv_decode_tracked_ram_bytes = 0;
};

static uint64_t moe_kv_decode_cache_allocated_bytes(const moe_decode_scratch* s) {
    if (!s) return 0;
    return (uint64_t)s->kv_decode_flat.size() * sizeof(float);
}

static void moe_kv_decode_cache_track_ram(moe_pc_engine* engine, moe_decode_scratch* s) {
    if (!engine || !s) {
        return;
    }
    const uint64_t current = moe_kv_decode_cache_allocated_bytes(s);
    if (current > s->kv_decode_tracked_ram_bytes) {
        add_tier_usage(engine, moe_TIER_RAM, current - s->kv_decode_tracked_ram_bytes);
    } else if (s->kv_decode_tracked_ram_bytes > current) {
        subtract_tier_usage(engine, moe_TIER_RAM, s->kv_decode_tracked_ram_bytes - current);
    }
    s->kv_decode_tracked_ram_bytes = current;
}

static void moe_kv_decode_cache_release_ram(moe_pc_engine* engine, moe_decode_scratch* s) {
    if (!engine || !s || s->kv_decode_tracked_ram_bytes == 0) {
        return;
    }
    subtract_tier_usage(engine, moe_TIER_RAM, s->kv_decode_tracked_ram_bytes);
    s->kv_decode_tracked_ram_bytes = 0;
}

struct moe_kv_decode_cache_ram_scope {
    moe_pc_engine* engine = nullptr;
    moe_decode_scratch* scratch = nullptr;
    moe_kv_decode_cache_ram_scope(moe_pc_engine* e, moe_decode_scratch* s) : engine(e), scratch(s) {}
    ~moe_kv_decode_cache_ram_scope() {
        moe_kv_decode_cache_release_ram(engine, scratch);
    }
};

static int moe_kv_decode_layer_cache_prepare(
    moe_pc_engine* engine,
    moe_decode_scratch* s,
    uint32_t layer,
    uint32_t decoded_count,
    float** out_buf,
    uint32_t* out_start
) {
    if (!engine || !s || !out_buf || !out_start || layer >= s->kv_cache.num_layers) {
        return 0;
    }
    const uint32_t entry_dim = s->kv_cache.entry_dim ? s->kv_cache.entry_dim : moe_MLA_KV_ENTRY_DIM;
    const size_t total_needed = (size_t)s->kv_cache.num_layers * s->kv_cache.max_seq_len * entry_dim;

    // BUGFIX 537: Check total_needed before resize to prevent overflow ★
    // Problem: num_layers * max_seq_len * entry_dim can overflow causing huge allocation
    // Solution: Check total_needed < 10000000000 (10GB reasonable max)
    if (total_needed > 10000000000ULL) {
        return 0;
    }

    // Fix 33: Reserve full capacity upfront to prevent mid-session realloc+memcpy spikes
    // With 79 layers × 8192 seq_len × 576 entry_dim, this can be ~1.5GB.
    // A single reserve() avoids std::vector's 2x growth pattern that peaks at 3GB.
    if (s->kv_decode_flat.capacity() < total_needed) {
        s->kv_decode_flat.reserve(total_needed);
    }
    if (s->kv_decode_flat.size() < total_needed) {
        s->kv_decode_flat.resize(total_needed);
        s->kv_decoded_up_to.assign(s->kv_cache.num_layers, 0u);
        moe_kv_decode_cache_track_ram(engine, s);
    }
    if (s->kv_decoded_up_to.size() != s->kv_cache.num_layers) {
        s->kv_decoded_up_to.assign(s->kv_cache.num_layers, 0u);
    }
    *out_start = std::min(s->kv_decoded_up_to[layer], decoded_count);
    *out_buf = s->kv_decode_flat.data() + (layer * s->kv_cache.max_seq_len * entry_dim);
    return 1;
}

static void moe_rope_table_resize(
    std::vector<float>* cos_table,
    std::vector<float>* sin_table,
    uint32_t dim,
    float base,
    uint32_t table_len = moe_MLA_ROPE_TABLE_LEN
) {
    // BUGFIX 223: Validate dim > 0 to prevent division by zero in frequency calculation ★★
    // Problem: dim=0 causes division by zero in freqs[j] = 1.0f / std::pow(base, (float)i / (float)dim)
    // Solution: Return early if dim == 0 (already checked dim < 2, but explicit check for clarity)
    // Impact: Prevents crash in RoPE table initialization with invalid config
    if (!cos_table || !sin_table || dim == 0 || dim < 2) {
        return;
    }
    static const uint32_t rope_precompute_cap = []() -> uint32_t {
        const char* env = std::getenv("STORAGELLM_ROPE_PRECOMPUTE");
        if (env && env[0]) {
            char* end = nullptr;
            const unsigned long parsed = std::strtoul(env, &end, 10);
            if (end != env && parsed > 0ul) {
                return parsed > 1048576ul ? 1048576u : (uint32_t)parsed;
            }
        }
        return 4096u;
    }();
    if (table_len > rope_precompute_cap) {
        table_len = rope_precompute_cap;
    }
    const uint32_t half_dim = dim / 2u;
    const size_t needed = (size_t)table_len * (size_t)half_dim;

    // BUGFIX 536: Check needed size before resize to prevent overflow ★
    // Problem: table_len * half_dim can overflow size_t causing huge allocation
    // Solution: Check needed < 100000000 (reasonable max for RoPE table)
    if (needed > 100000000) {
        return;
    }

    if (cos_table->size() == needed && sin_table->size() == needed) {
        return;
    }
    cos_table->resize(needed);
    sin_table->resize(needed);
    std::vector<float> freqs(half_dim);
    // Fix 14: Use geometric multiplier
    const float multiplier = std::pow(base, -2.0f / (float)dim);
    float freq = 1.0f;
    for (uint32_t j = 0; j < half_dim; ++j) {
        freqs[j] = freq;
        freq *= multiplier;
    }
    for (uint32_t pos = 0; pos < table_len; ++pos) {
        float* cos_row = cos_table->data() + (size_t)pos * (size_t)half_dim;
        float* sin_row = sin_table->data() + (size_t)pos * (size_t)half_dim;
        for (uint32_t j = 0; j < half_dim; ++j) {
            const float theta = (float)pos * freqs[j];
            cos_row[j] = std::cos(theta);
            sin_row[j] = std::sin(theta);
        }
    }
}

static void moe_decode_scratch_resize(
    moe_decode_scratch* s,
    const moe_storage_constants_t* c,
    const moe_attn_config_t* attn_cfg,
    int pre_allocate_kv,
    moe_pc_engine* engine = nullptr
) {
    // BUGFIX 538: Validate input parameters before use ★
    // Problem: c can have invalid values causing overflow in resize operations
    // Solution: Check ranges before use
    if (!s || !c || c->hidden_size > 100000 || c->expert_intermediate_size > 100000 ||
        c->experts_per_moe_layer > 10000 || c->num_hidden_layers > 1000) {
        return;
    }

    const uint32_t hidden = c->hidden_size;
    const uint32_t expert_intermediate = c->expert_intermediate_size;
    const moe_attn_config_t cfg = attn_cfg ? *attn_cfg : moe_legacy_mla_attn_config();
    const uint32_t q_lora_rank = cfg.q_lora_rank ? cfg.q_lora_rank : 2048u;
    const uint32_t kv_a_dim = moe_attn_kv_a_proj_dim(cfg);
    const uint32_t kv_entry_dim = moe_attn_kv_entry_dim(cfg);
    const uint32_t q_b_dim = moe_attn_q_b_dim(cfg);
    const uint32_t v_total_dim = moe_attn_v_total_dim(cfg);

    // BUGFIX 539: Check computed dimensions before resize ★
    // Problem: Computed dimensions can overflow causing huge allocations
    // Solution: Validate all computed dimensions
    if (kv_a_dim > 100000 || kv_entry_dim > 100000 || q_b_dim > 100000 ||
        v_total_dim > 100000 || q_lora_rank > 100000 || cfg.kv_lora_rank > 100000 ||
        cfg.num_heads > 1000) {
        return;
    }

    // Fix 34: Pre-allocate all float vectors in a single sequential burst.
    // When all vectors are first allocated in rapid succession, the OS heap allocator
    // typically places them contiguously in virtual memory. This improves L2/L3
    // prefetcher hit rate during the forward pass (hidden → residual → norm → attn
    // are accessed sequentially).
    // Total: ~25 vectors × hidden_size(7168) × 4 bytes ≈ 700KB contiguous.
    const bool first_alloc = s->hidden.empty();
    s->hidden.resize(hidden);
    s->residual.resize(hidden);
    s->norm.resize(hidden);
    s->attn.resize(hidden);
    s->attn_kv_a.resize(kv_a_dim);
    s->attn_kv_norm.resize(cfg.kv_lora_rank);
    // BUGFIX 53: attn_kv_b removed (dead allocation)
    s->attn_value.resize(v_total_dim);
    s->mlp.resize(hidden);
    s->shared.resize(hidden);
    s->expert.resize(hidden);
    s->gate.resize(std::max<uint32_t>(12288u, expert_intermediate));
    // Absorption buffers
    s->q_a.resize(q_lora_rank);
    s->q_a_norm.resize(q_lora_rank);
    s->q_b.resize(q_b_dim);
    s->q_absorbed.resize((size_t)cfg.num_heads * cfg.kv_lora_rank);
    s->c_weighted.resize((size_t)cfg.num_heads * cfg.kv_lora_rank);
    s->head_max.resize(cfg.num_heads);
    s->head_denom.resize(cfg.num_heads);
    s->kv_entry.resize(kv_entry_dim);
    s->up.resize(std::max<uint32_t>(12288u, expert_intermediate));
    s->router_logits.resize(c->experts_per_moe_layer);
    s->router_order.resize(c->experts_per_moe_layer);
    s->selected_experts.reserve(moe_MOE_TOP_EXPERTS);
    s->selected_weights.reserve(moe_MOE_TOP_EXPERTS);
    s->expert_work.resize(moe_MOE_TOP_EXPERTS);
    s->expert_gate_pool.resize((size_t)moe_MOE_TOP_EXPERTS * (size_t)expert_intermediate);
    s->expert_up_pool.resize((size_t)moe_MOE_TOP_EXPERTS * (size_t)expert_intermediate);
    s->expert_out_pool.resize((size_t)moe_MOE_TOP_EXPERTS * (size_t)hidden);
    // On first allocation, shrink all vectors to exact capacity to prevent
    // 2x over-allocation from vector growth and to pack them tightly.
    if (first_alloc) {
        s->hidden.shrink_to_fit();
        s->residual.shrink_to_fit();
        s->norm.shrink_to_fit();
        s->attn.shrink_to_fit();
        s->mlp.shrink_to_fit();
        s->shared.shrink_to_fit();
        s->expert.shrink_to_fit();
        s->gate.shrink_to_fit();
        s->up.shrink_to_fit();
        s->expert_gate_pool.shrink_to_fit();
        s->expert_up_pool.shrink_to_fit();
        s->expert_out_pool.shrink_to_fit();
    }
    moe_rope_table_resize(&s->rope_k_cos, &s->rope_k_sin, cfg.qk_rope_head_dim, cfg.rope_theta, cfg.rope_table_len);
    moe_rope_table_resize(&s->rope_q_cos, &s->rope_q_sin, cfg.qk_rope_head_dim, cfg.rope_theta, cfg.rope_table_len);
    if (s->kv_cache.num_layers != c->num_hidden_layers ||
        s->kv_cache.entry_dim != kv_entry_dim ||
        s->kv_cache.seq_limit != cfg.max_seq_len ||
        s->kv_cache.data == nullptr ||
        (pre_allocate_kv && s->kv_cache.max_seq_len < cfg.max_seq_len)) {
        s->kv_cache.init(c->num_hidden_layers, kv_entry_dim, cfg.max_seq_len, pre_allocate_kv != 0);
        // FIX 9: Immediately sync RAM accounting after kv_cache reinit.
        // Without this, tracked_ram_bytes holds the old value until the next
        // ensure_capacity() call—which never fires when pre_allocate=true.
        if (engine) {
            moe_kv_cache_track_ram(engine, &s->kv_cache);
        }
        s->kv_decoded_up_to.clear();
        s->position = 0;
    }
}

