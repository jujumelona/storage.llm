struct moe_expert_work_t {
    float* gate = nullptr;
    float* up = nullptr;
    float* out = nullptr;
    float weight = 0.0f;
    int expert_id = 0;
    int ok = 0;
};

// CPUStorageEngine hot-loop plan for the selected experts of one layer.
// This lives in decode scratch so decode does not allocate a shards vector
// or per-expert atomic array on every layer.  The views are rebuilt only for
// the selected experts, then reused by gate/up/down loops as pointer+row_bytes.
struct moe_cpu_expert_plan_t {
    const moe_tensor_record* gate_rec = nullptr;
    const moe_tensor_record* up_rec = nullptr;
    const moe_tensor_record* down_rec = nullptr;
    mmap_context_t* gate_shard = nullptr;
    mmap_context_t* up_shard = nullptr;
    mmap_context_t* down_shard = nullptr;
    moe_tensor_row_fast_view_t gate_view{};
    moe_tensor_row_fast_view_t up_view{};
    moe_tensor_row_fast_view_t down_view{};
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
    // CPU-only decode hot buffers.  Kept in scratch to avoid per-layer vector
    // allocation while routing through CPUStorageEngine.
    std::vector<uint8_t> cpu_expert_ready;
    std::vector<float> cpu_down_scales;
    std::vector<moe_cpu_expert_plan_t> cpu_expert_plan;
    std::vector<uint32_t> cpu_active_expert_indices;
    std::vector<float> ple_inputs;
    std::vector<float> ple_projected;
    // Legacy/plain KV cache. QKV-only models must not allocate or use this; it
    // remains only for legacy/debug plain-KV models and is guarded by the
    // qkv_only_runtime fields below.
    moe_kv_cache kv_cache;
    uint8_t qkv_only_runtime = 0;
    uint32_t qkv_seq_len = 0;
    uint32_t qkv_capacity_tokens = 0;
    uint32_t qkv_seq_limit = 0;
    uint32_t position = 0;
    // Per-session router EMA state (BUGFIX Problem F, stage 2: wiring) ★★★ PERFORMANCE
    // Problem: Engine-level smoothed_router_entropy_milli shared across all sessions
    // → Different domains (code gen vs chat) mix EMA state
    // → Hysteresis effectiveness reduced, cache hit rate drops
    // Solution: Each decode scratch maintains independent entropy and prefetch depth
    // → Router writes here instead of engine->last_router_entropy_milli
    // → MLP reads depth from here instead of engine->adaptive_prefetch_depth
    // → Engine-level atomics kept as secondary sync for diagnostics/profiling only
    uint32_t session_router_entropy_milli = 0;
    uint32_t session_adaptive_prefetch_depth = 2;
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
    uint32_t kv_decode_stride_tokens = 0;
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

static void moe_engine_signal_memory_pressure_recovery(
    moe_pc_engine* engine,
    const char* reason,
    uint64_t requested_bytes = 0
) {
    if (!engine) {
        return;
    }
    std::fprintf(stderr,
        "[storageLLM WARNING] memory pressure recovery reason=%s requested=%llu ram_used=%llu ram_budget=%llu vram_alloc=%llu\n",
        reason ? reason : "unknown",
        (unsigned long long)requested_bytes,
        (unsigned long long)tier_used(engine, moe_TIER_RAM),
        (unsigned long long)engine->config.ram_budget_bytes,
        (unsigned long long)engine->device_allocated_bytes.load(std::memory_order_relaxed));
    engine->ram_eviction_needed.store(true, std::memory_order_relaxed);
    engine->vram_eviction_needed.store(true, std::memory_order_relaxed);
    engine->vram_eviction_cv.notify_one();
    device_refresh_mem_info(engine, 1);
    device_free_pool(engine, moe_DEVICE_POOL_STREAM);
    device_free_pool(engine, moe_DEVICE_POOL_ACTIVATION);
}

static void moe_decode_scratch_drop_transient_buffers(moe_pc_engine* engine, moe_decode_scratch* s) {
    if (!s) {
        return;
    }
    // BUGFIX 925: Release kv_decode RAM tracking before dropping buffers ★★ IMPORTANT
    // Problem: kv_decode_flat is freed but kv_decode_tracked_ram_bytes not cleared
    // → RAM usage counter stays high until next moe_kv_decode_cache_track_ram() call
    // → RAM budget miscalculated during token latency window
    // Solution: Call subtract_tier_usage() before swap to immediately update RAM counter
    // Impact: Accurate RAM accounting during transient buffer drops
    if (engine && s->kv_decode_tracked_ram_bytes > 0) {
        subtract_tier_usage(engine, moe_TIER_RAM, s->kv_decode_tracked_ram_bytes);
        s->kv_decode_tracked_ram_bytes = 0;
    }

    std::vector<float>().swap(s->hidden);
    std::vector<float>().swap(s->residual);
    std::vector<float>().swap(s->norm);
    std::vector<float>().swap(s->attn);
    std::vector<float>().swap(s->attn_kv_a);
    std::vector<float>().swap(s->attn_kv_norm);
    std::vector<float>().swap(s->attn_value);
    std::vector<float>().swap(s->mlp);
    std::vector<float>().swap(s->shared);
    std::vector<float>().swap(s->expert);
    std::vector<float>().swap(s->gate);
    std::vector<float>().swap(s->up);
    std::vector<float>().swap(s->router_logits);
    std::vector<uint32_t>().swap(s->router_order);
    std::vector<int>().swap(s->selected_experts);
    std::vector<float>().swap(s->selected_weights);
    std::vector<moe_expert_work_t>().swap(s->expert_work);
    std::vector<float>().swap(s->expert_gate_pool);
    std::vector<float>().swap(s->expert_up_pool);
    std::vector<float>().swap(s->expert_out_pool);
    std::vector<uint8_t>().swap(s->cpu_expert_ready);
    std::vector<float>().swap(s->cpu_down_scales);
    std::vector<moe_cpu_expert_plan_t>().swap(s->cpu_expert_plan);
    std::vector<uint32_t>().swap(s->cpu_active_expert_indices);
    std::vector<float>().swap(s->ple_inputs);
    std::vector<float>().swap(s->ple_projected);
    std::vector<float>().swap(s->q_a);
    std::vector<float>().swap(s->q_a_norm);
    std::vector<float>().swap(s->q_b);
    std::vector<float>().swap(s->q_absorbed);
    std::vector<float>().swap(s->c_weighted);
    std::vector<float>().swap(s->head_max);
    std::vector<float>().swap(s->head_denom);
    std::vector<float>().swap(s->kv_entry);
    std::vector<float>().swap(s->rope_k_cos);
    std::vector<float>().swap(s->rope_k_sin);
    std::vector<float>().swap(s->rope_q_cos);
    std::vector<float>().swap(s->rope_q_sin);
    std::vector<float>().swap(s->kv_decode_flat);
    std::vector<uint32_t>().swap(s->kv_decoded_up_to);
    s->kv_decode_stride_tokens = 0;
}

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
    uint32_t needed_stride = decoded_count > 0 ? decoded_count : 1u;
    if (needed_stride > s->kv_cache.max_seq_len) {
        needed_stride = s->kv_cache.max_seq_len;
    }
    if (needed_stride == 0) {
        needed_stride = 1u;
    }
    const uint32_t stride_chunk = 64u;
    uint32_t target_stride = (needed_stride + stride_chunk - 1u) & ~(stride_chunk - 1u);
    if (target_stride < needed_stride) {
        target_stride = needed_stride;
    }
    if (target_stride > s->kv_cache.max_seq_len) {
        target_stride = s->kv_cache.max_seq_len;
    }
    if (target_stride == 0) {
        target_stride = needed_stride;
    }
    const size_t total_needed = (size_t)s->kv_cache.num_layers * target_stride * entry_dim;

    // BUGFIX 537: Check total_needed before resize to prevent overflow ★
    // Problem: num_layers * max_seq_len * entry_dim can overflow causing huge allocation
    // Solution: Check total_needed < 10000000000 (10GB reasonable max)
    if (total_needed > 10000000000ULL) {
        return 0;
    }

    // Decode only needs historical tokens that have actually been reached.
    // Grow the per-layer decoded KV stride in chunks instead of allocating the
    // full max_seq_len window for every short prompt.
    try {
        if (s->kv_decode_flat.size() < total_needed ||
            s->kv_decode_stride_tokens < target_stride ||
            s->kv_decode_stride_tokens == 0) {
            const uint32_t old_stride = s->kv_decode_stride_tokens;
            std::vector<float> next;
            next.resize(total_needed);
            if (!s->kv_decode_flat.empty() && old_stride > 0 &&
                s->kv_decoded_up_to.size() == s->kv_cache.num_layers) {
                for (uint32_t l = 0; l < s->kv_cache.num_layers; ++l) {
                    uint32_t copy_tokens = s->kv_decoded_up_to[l];
                    copy_tokens = std::min(copy_tokens, old_stride);
                    copy_tokens = std::min(copy_tokens, target_stride);
                    if (copy_tokens == 0) {
                        continue;
                    }
                    const size_t old_off = (size_t)l * old_stride * entry_dim;
                    const size_t new_off = (size_t)l * target_stride * entry_dim;
                    std::memcpy(
                        next.data() + new_off,
                        s->kv_decode_flat.data() + old_off,
                        (size_t)copy_tokens * entry_dim * sizeof(float));
                }
            }
            s->kv_decode_flat.swap(next);
            s->kv_decode_stride_tokens = target_stride;
            if (s->kv_decoded_up_to.size() != s->kv_cache.num_layers) {
                s->kv_decoded_up_to.assign(s->kv_cache.num_layers, 0u);
            }
            moe_kv_decode_cache_track_ram(engine, s);
        }
        if (s->kv_decoded_up_to.size() != s->kv_cache.num_layers) {
            s->kv_decoded_up_to.assign(s->kv_cache.num_layers, 0u);
        }
    } catch (const std::bad_alloc&) {
        moe_kv_decode_cache_release_ram(engine, s);
        std::vector<float>().swap(s->kv_decode_flat);
        std::vector<uint32_t>().swap(s->kv_decoded_up_to);
        s->kv_decode_stride_tokens = 0;
        moe_engine_signal_memory_pressure_recovery(
            engine, "kv_decode_cache_bad_alloc", (uint64_t)total_needed * sizeof(float));
        return 0;
    }
    *out_start = std::min(s->kv_decoded_up_to[layer], decoded_count);
    const uint32_t stride = s->kv_decode_stride_tokens ? s->kv_decode_stride_tokens : target_stride;
    *out_buf = s->kv_decode_flat.data() + ((size_t)layer * stride * entry_dim);
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
    struct moe_rope_table_cache_entry {
        uint32_t dim = 0;
        uint32_t table_len = 0;
        float base = 0.0f;
        std::vector<float> cos;
        std::vector<float> sin;
    };
    static std::mutex rope_cache_mutex;
    static std::vector<moe_rope_table_cache_entry> rope_cache;
    {
        std::lock_guard<std::mutex> lock(rope_cache_mutex);
        for (const moe_rope_table_cache_entry& entry : rope_cache) {
            if (entry.dim == dim && entry.table_len == table_len && entry.base == base &&
                entry.cos.size() == needed && entry.sin.size() == needed) {
                *cos_table = entry.cos;
                *sin_table = entry.sin;
                return;
            }
        }
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
    {
        std::lock_guard<std::mutex> lock(rope_cache_mutex);
        if (rope_cache.size() > 16u) {
            rope_cache.erase(rope_cache.begin(), rope_cache.begin() + 4);
        }
        moe_rope_table_cache_entry entry{};
        entry.dim = dim;
        entry.table_len = table_len;
        entry.base = base;
        entry.cos = *cos_table;
        entry.sin = *sin_table;
        rope_cache.push_back(std::move(entry));
    }
}

static int moe_decode_scratch_resize(
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
        return 0;
    }

    const uint32_t hidden = c->hidden_size;
    const uint32_t expert_intermediate = c->expert_intermediate_size;
    const moe_attn_config_t cfg = attn_cfg ? *attn_cfg : (engine ? moe_generation_attn_config(engine) : moe_attn_config_t{});
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
        return 0;
    }

    // Fix 34: Pre-allocate all float vectors in a single sequential burst.
    // When all vectors are first allocated in rapid succession, the OS heap allocator
    // typically places them contiguously in virtual memory. This improves L2/L3
    // prefetcher hit rate during the forward pass (hidden → residual → norm → attn
    // are accessed sequentially).
    // Total: ~25 vectors × hidden_size(7168) × 4 bytes ≈ 700KB contiguous.
    try {
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
        s->cpu_expert_plan.resize(moe_MOE_TOP_EXPERTS);
        s->cpu_expert_ready.resize(moe_MOE_TOP_EXPERTS);
        s->cpu_down_scales.resize(moe_MOE_TOP_EXPERTS);
        s->cpu_active_expert_indices.reserve(moe_MOE_TOP_EXPERTS);
        s->expert_gate_pool.resize((size_t)moe_MOE_TOP_EXPERTS * (size_t)expert_intermediate);
        s->expert_up_pool.resize((size_t)moe_MOE_TOP_EXPERTS * (size_t)expert_intermediate);
        // CPUStorageEngine fuses down+accumulation and normally never needs
        // expert_out_pool.  Avoid allocating selected_experts*hidden floats at
        // scratch setup; fallback code allocates it lazily only when required.
        const char* cpu_storage_env = std::getenv("STORAGELLM_CPU_STORAGE_ENGINE");
        const char* cpu_only_env = std::getenv("STORAGELLM_CPU_ONLY");
        const char* cpu_max_env = std::getenv("STORAGELLM_CPU_STORAGE_MAX");
        const int cpu_storage_fast =
            (cpu_storage_env && cpu_storage_env[0] && cpu_storage_env[0] != '0') ||
            (cpu_only_env && cpu_only_env[0] && cpu_only_env[0] != '0') ||
            (cpu_max_env && cpu_max_env[0] && cpu_max_env[0] != '0');
        if (!cpu_storage_fast) {
            s->expert_out_pool.resize((size_t)moe_MOE_TOP_EXPERTS * (size_t)hidden);
        }
        (void)first_alloc;
        moe_rope_table_resize(&s->rope_k_cos, &s->rope_k_sin, cfg.qk_rope_head_dim, cfg.rope_theta, cfg.rope_table_len);
        moe_rope_table_resize(&s->rope_q_cos, &s->rope_q_sin, cfg.qk_rope_head_dim, cfg.rope_theta, cfg.rope_table_len);
    } catch (const std::bad_alloc&) {
        moe_kv_decode_cache_release_ram(engine, s);
        moe_decode_scratch_drop_transient_buffers(engine, s);
        moe_engine_signal_memory_pressure_recovery(engine, "decode_scratch_bad_alloc");
        return 0;
    }
    const int qkv_force_plain_ab = moe_qkv_force_plain_ab_enabled();
    const int qkv_only_required = engine && !qkv_force_plain_ab &&
        (engine->qkv_forced_by_format.load(std::memory_order_acquire) ||
         engine->config.kv_mode == moe_KV_MODE_QKV);
    if (qkv_only_required) {
        s->qkv_only_runtime = 1;
        s->qkv_seq_limit = cfg.max_seq_len ? cfg.max_seq_len : moe_MLA_MAX_SEQ_LEN;
        if (s->qkv_seq_limit == 0) {
            s->qkv_seq_limit = 1;
        }
        uint32_t cap = pre_allocate_kv ? s->qkv_seq_limit :
            std::min<uint32_t>(moe_MLA_KV_INITIAL_CHUNK, s->qkv_seq_limit);
        if (cap == 0) {
            cap = 1;
        }
        if (s->qkv_capacity_tokens == 0 ||
            s->qkv_capacity_tokens > s->qkv_seq_limit ||
            (pre_allocate_kv && s->qkv_capacity_tokens < s->qkv_seq_limit)) {
            s->qkv_capacity_tokens = cap;
        }
        if (s->kv_cache.data || s->kv_cache.scales || s->kv_cache.zeros ||
            s->kv_cache.tracked_ram_bytes != 0) {
            if (engine && s->kv_cache.tracked_ram_bytes > 0) {
                subtract_tier_usage(engine, moe_TIER_RAM, s->kv_cache.tracked_ram_bytes);
                s->kv_cache.tracked_ram_bytes = 0;
            }
            std::free(s->kv_cache.data);
            std::free(s->kv_cache.scales);
            std::free(s->kv_cache.zeros);
            s->kv_cache.data = nullptr;
            s->kv_cache.scales = nullptr;
            s->kv_cache.zeros = nullptr;
        }
        s->kv_cache.num_layers = 0;
        s->kv_cache.max_seq_len = 0;
        s->kv_cache.seq_len = 0;
        s->kv_cache.entry_dim = kv_entry_dim;
        s->kv_cache.seq_limit = s->qkv_seq_limit;
        s->kv_decoded_up_to.clear();
        s->kv_decode_stride_tokens = 0;
        if (engine) {
            engine->qkv_plain_kv_allocation_skips.fetch_add(1, std::memory_order_relaxed);
        }
        return s->hidden.size() >= hidden && s->norm.size() >= hidden &&
            s->qkv_only_runtime && s->qkv_capacity_tokens > 0;
    }
    s->qkv_only_runtime = 0;
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
        s->kv_decode_stride_tokens = 0;
        s->position = 0;
    }
    return s->hidden.size() >= hidden && s->norm.size() >= hidden && s->kv_cache.data != nullptr;
}
