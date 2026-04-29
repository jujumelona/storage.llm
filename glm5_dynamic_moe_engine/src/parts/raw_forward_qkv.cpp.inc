// Raw forward - QKV attention integration

// GAP 1A: Quantize plain float K/V into persistent per-head qkv_state_t
// Called once per head when context changes; state persists across decode calls
static void glm5_qkv_ensure_head_storage_locked(glm5_pc_engine* engine, size_t head_count) {
    if (!engine || head_count == 0) {
        return;
    }
    if (engine->qkv_head_state_slots.size() < head_count) {
        engine->qkv_head_state_slots.resize(head_count);
    }
    if (engine->qkv_head_state_valid.size() < head_count) {
        engine->qkv_head_state_valid.resize(head_count, 0);
    }
    if (engine->qkv_head_state_locks.size() < head_count) {
        engine->qkv_head_state_locks.resize(head_count);
    }
    for (size_t i = 0; i < head_count; ++i) {
        if (!engine->qkv_head_state_locks[i]) {
            engine->qkv_head_state_locks[i] = std::make_unique<std::mutex>();
        }
    }
}

static int glm5_qkv_prepare_head_states(
    glm5_pc_engine* engine,
    uint32_t head_count,
    uint32_t head_dim
) {
    if (!engine || head_count == 0 || head_dim == 0) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(engine->qkv_mutex);
    if (!engine->qkv_cfg_initialized.load(std::memory_order_acquire)) {
        engine->qkv_cfg = qkv_config_default((int)head_dim);
        engine->qkv_cfg_initialized.store(true, std::memory_order_release);
    }
    glm5_qkv_ensure_head_storage_locked(engine, head_count);
    return 1;
}

static qkv_state_t* glm5_qkv_ensure_head_state(
    glm5_pc_engine* engine,
    uint32_t head,
    const float* head_keys,
    const float* head_values,
    uint32_t context_tokens,
    uint32_t head_dim
) {
    if (!engine || !head_keys || !head_values || !context_tokens || !head_dim) return nullptr;
    std::mutex* head_lock = nullptr;

    {
        std::lock_guard<std::mutex> lock(engine->qkv_mutex);
        if (!engine->qkv_cfg_initialized.load(std::memory_order_acquire)) {
            engine->qkv_cfg = qkv_config_default((int)head_dim);
            engine->qkv_cfg_initialized.store(true, std::memory_order_release);
        }
        glm5_qkv_ensure_head_storage_locked(engine, (size_t)head + 1u);
        head_lock = engine->qkv_head_state_locks[head].get();
    }
    if (!head_lock) {
        return nullptr;
    }
    std::lock_guard<std::mutex> head_guard(*head_lock);

    qkv_state_t* slot = &engine->qkv_head_state_slots[head];
    if (engine->qkv_head_state_valid[head] && slot->n_tokens == (int)context_tokens) {
        if (!qkv_quantize(slot, &engine->qkv_cfg, head_keys, head_values, (int)context_tokens)) {
            return nullptr;
        }
        return slot;
    }

    if (engine->qkv_head_state_valid[head]) {
        qkv_free(slot);
        *slot = qkv_state_t{};
        engine->qkv_head_state_valid[head] = 0;
    }

    // Allocate new state
    qkv_state_t st{};
    if (!qkv_init(&st, &engine->qkv_cfg, (int)context_tokens)) {
        return nullptr;
    }
    if (!qkv_quantize(&st, &engine->qkv_cfg, head_keys, head_values, (int)context_tokens)) {
        qkv_free(&st);
        return nullptr;
    }
    *slot = st;
    engine->qkv_head_state_valid[head] = 1;
    return slot;
}

int glm5_pc_engine_attention_decode_f32(
    glm5_pc_engine_t* engine,
    const float* query,
    const float* key_cache,
    const float* value_cache,
    uint32_t context_tokens,
    uint32_t head_count,
    uint32_t head_dim,
    float* output
) {
    if (!engine || !query || !key_cache || !value_cache || !output ||
        context_tokens == 0 || head_count == 0 || head_dim == 0) {
        return 0;
    }
    const float inv_sqrt = 1.0f / std::sqrt((float)head_dim);
    if (context_tokens > (uint32_t)std::numeric_limits<int>::max() ||
        head_dim > (uint32_t)std::numeric_limits<int>::max()) {
        return 0;
    }
    if (engine->config.kv_mode == GLM5_KV_MODE_QKV && qkv_is_enabled() &&
        !glm5_qkv_prepare_head_states(engine, head_count, head_dim)) {
        return 0;
    }
    return glm5_parallel_rows(engine, head_count, 0, [&](uint32_t head) {
        const float* q = query + (uint64_t)head * (uint64_t)head_dim;
        float* out = output + (uint64_t)head * (uint64_t)head_dim;
        const float* head_keys = key_cache + (uint64_t)head * (uint64_t)context_tokens * (uint64_t)head_dim;
        const float* head_values = value_cache + (uint64_t)head * (uint64_t)context_tokens * (uint64_t)head_dim;
        // GAP 1A: QKV mode; quantize into persistent state, attention in quantized domain.
        if (engine->config.kv_mode == GLM5_KV_MODE_QKV && qkv_is_enabled()) {
            qkv_state_t* st = glm5_qkv_ensure_head_state(
                engine, head, head_keys, head_values, context_tokens, head_dim);
            if (!st) return 0;
            return qkv_attention_decode(q, st, &engine->qkv_cfg, context_tokens, head_dim, out);
        }
        float max_score = -std::numeric_limits<float>::infinity();

        static thread_local std::vector<float> tls_scores;
        if (tls_scores.size() < context_tokens) {
            tls_scores.resize(context_tokens);
        }
        float* scores = tls_scores.data();

        for (uint32_t t = 0; t < context_tokens; ++t) {
            // [head][t][d] layout
            const float* k = head_keys + (uint64_t)t * (uint64_t)head_dim;
            float score = 0.0f;
            for (uint32_t d = 0; d < head_dim; ++d) {
                score += q[d] * k[d];
            }
            score *= inv_sqrt;
            scores[t] = score;
            if (score > max_score) {
                max_score = score;
            }
        }
        for (uint32_t d = 0; d < head_dim; ++d) {
            out[d] = 0.0f;
        }
        float denom = 0.0f;
        for (uint32_t t = 0; t < context_tokens; ++t) {
            // [head][t][d] layout
            const float* v = head_values + (uint64_t)t * (uint64_t)head_dim;
            const float weight = std::exp(scores[t] - max_score);
            denom += weight;
            for (uint32_t d = 0; d < head_dim; ++d) {
                out[d] += weight * v[d];
            }
        }
        if (denom <= 0.0f) {
            return 0;
        }
        const float scale = 1.0f / denom;
        for (uint32_t d = 0; d < head_dim; ++d) {
            out[d] *= scale;
        }
        return 1;
    });
}
