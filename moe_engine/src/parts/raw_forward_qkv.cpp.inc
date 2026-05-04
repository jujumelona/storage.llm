// Raw forward - QKV attention integration

// GAP 1A: Quantize plain float K/V into persistent per-head qkv_state_t
// Called once per head when context changes; state persists across decode calls
static void moe_qkv_ensure_head_storage_locked(moe_pc_engine* engine, size_t head_count) {
    if (!engine || head_count == 0) {
        return;
    }
    // BUGFIX 618: Fix return type mismatch - function returns void, not int ★
    // Problem: Function declared as void but has "return 0;" statement
    // Solution: Remove return value from range check
    if (head_count > 10000) {
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

static void moe_qkv_free_head_states_locked(moe_pc_engine* engine) {
    if (!engine) {
        return;
    }
    for (size_t i = 0; i < engine->qkv_head_state_slots.size(); ++i) {
        if (i < engine->qkv_head_state_valid.size() && engine->qkv_head_state_valid[i]) {
            qkv_free(&engine->qkv_head_state_slots[i]);
        }
    }
    engine->qkv_head_state_slots.clear();
    engine->qkv_head_state_valid.clear();
    engine->qkv_head_state_locks.clear();
    engine->qkv_last_ctx = 0;
}

static qkv_config_t moe_qkv_contract_config_for_head_dim(
    const moe_pc_engine* engine,
    uint32_t head_dim
) {
    qkv_config_t cfg = qkv_config_default((int)head_dim);
    if (engine && engine->offload_qkv_cfg_available.load(std::memory_order_acquire)) {
        cfg = engine->offload_qkv_cfg;
        cfg.head_dim = (int)head_dim;
        if (cfg.outlier_channels < 0) {
            cfg.outlier_channels = 0;
        }
        if (cfg.outlier_channels > (int)head_dim) {
            cfg.outlier_channels = (int)head_dim;
        }
        if (cfg.group_size == 0) {
            cfg.group_size = 64;
        }
        if (cfg.page_size_tokens == 0) {
            cfg.page_size_tokens = 16;
        }
        if (cfg.sink_tokens == 0) {
            cfg.sink_tokens = 4;
        }
    }
    cfg.engine_io_thread_count =
        runtime_disk_workers(engine) + runtime_pinned_workers(engine) + runtime_gpu_workers(engine);
    return cfg;
}

static bool moe_qkv_config_matches_contract(
    const qkv_config_t& current,
    const qkv_config_t& desired
) {
    return current.k_bits == desired.k_bits &&
        current.v_bits == desired.v_bits &&
        current.head_dim == desired.head_dim &&
        current.enable_qjl == desired.enable_qjl &&
        current.enable_rotation == desired.enable_rotation &&
        current.rotation_seed == desired.rotation_seed &&
        current.qjl_seed == desired.qjl_seed &&
        current.outlier_channels == desired.outlier_channels &&
        current.outlier_bits == desired.outlier_bits &&
        current.normal_bits == desired.normal_bits &&
        current.group_size == desired.group_size &&
        current.page_size_tokens == desired.page_size_tokens &&
        current.sink_tokens == desired.sink_tokens &&
        current.plain_kv_persistent_storage == desired.plain_kv_persistent_storage;
}

static void moe_qkv_apply_contract_config_locked(
    moe_pc_engine* engine,
    uint32_t head_dim
) {
    if (!engine) {
        return;
    }
    const qkv_config_t desired = moe_qkv_contract_config_for_head_dim(engine, head_dim);
    if (!engine->qkv_cfg_initialized.load(std::memory_order_acquire) ||
        !moe_qkv_config_matches_contract(engine->qkv_cfg, desired)) {
        moe_qkv_free_head_states_locked(engine);
        engine->qkv_cfg = desired;
        engine->qkv_cfg_initialized.store(true, std::memory_order_release);
        moe_profile_log(
            "qkv_config",
            "head_dim=%u k_bits=%d v_bits=%d group=%u page=%u sink=%u "
            "rotation=%d rotation_seed=%llu qjl=%d qjl_seed=%llu "
            "outlier_channels=%d outlier_bits=%d normal_bits=%d effective_bits=%.3f plain_kv=%d",
            head_dim,
            desired.k_bits,
            desired.v_bits,
            desired.group_size,
            desired.page_size_tokens,
            desired.sink_tokens,
            desired.enable_rotation ? 1 : 0,
            (unsigned long long)desired.rotation_seed,
            desired.enable_qjl ? 1 : 0,
            (unsigned long long)desired.qjl_seed,
            desired.outlier_channels,
            desired.outlier_bits,
            desired.normal_bits,
            (double)qkv_effective_bits(&desired),
            desired.plain_kv_persistent_storage ? 1 : 0);
    }
}

static int moe_qkv_prepare_head_states(
    moe_pc_engine* engine,
    uint32_t head_count,
    uint32_t head_dim
) {
    if (!engine || head_count == 0 || head_dim == 0) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(engine->qkv_mutex);
    moe_qkv_apply_contract_config_locked(engine, head_dim);
    moe_qkv_ensure_head_storage_locked(engine, head_count);
    return 1;
}

static qkv_state_t* moe_qkv_ensure_head_state(
    moe_pc_engine* engine,
    uint32_t head,
    const float* head_keys,
    const float* head_values,
    uint32_t context_tokens,
    uint32_t head_dim
) {
    if (!engine || !head_keys || !head_values || !context_tokens || !head_dim) return nullptr;
    std::mutex* head_lock = nullptr;

    if (head < engine->qkv_head_state_locks.size()) {
        head_lock = engine->qkv_head_state_locks[head].get();
    }
    if (!head_lock) {
        std::lock_guard<std::mutex> lock(engine->qkv_mutex);
        moe_qkv_apply_contract_config_locked(engine, head_dim);
        moe_qkv_ensure_head_storage_locked(engine, (size_t)head + 1u);
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

int moe_pc_engine_attention_decode_f32(
    moe_pc_engine_t* engine,
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
    // BUGFIX 640: head_dim 체크를 sqrt 전에 수행 ★
    // Problem: head_dim == 0 체크가 sqrt 호출 후에 있음
    // 위 if문에서 head_dim == 0 체크하므로 안전하지만, 명시적으로 확인
    // Solution: 이미 체크되어 있으므로 안전함 (주석 추가)
    const float inv_sqrt = 1.0f / std::sqrt((float)head_dim);  // Safe: head_dim > 0 guaranteed by check above
    if (context_tokens > (uint32_t)std::numeric_limits<int>::max() ||
        head_dim > (uint32_t)std::numeric_limits<int>::max()) {
        return 0;
    }
    // BUGFIX 630: Check for overflow in pointer arithmetic ★
    // Problem: head * context_tokens * head_dim can overflow uint64_t
    // Solution: Check multiplication overflow before pointer arithmetic
    if (head_count > 0 && context_tokens > 0 && head_dim > 0) {
        if (context_tokens > UINT64_MAX / head_dim) {
            return 0;  // context_tokens * head_dim would overflow
        }
        uint64_t tokens_times_dim = (uint64_t)context_tokens * (uint64_t)head_dim;
        if (head_count > UINT64_MAX / tokens_times_dim) {
            return 0;  // head_count * context_tokens * head_dim would overflow
        }
    }
    if (engine->config.kv_mode == moe_KV_MODE_QKV && qkv_is_enabled() &&
        !moe_qkv_prepare_head_states(engine, head_count, head_dim)) {
        return 0;
    }
    return moe_parallel_rows(engine, head_count, 0, [&](uint32_t head) {
        const float* q = query + (uint64_t)head * (uint64_t)head_dim;
        float* out = output + (uint64_t)head * (uint64_t)head_dim;
        const float* head_keys = key_cache + (uint64_t)head * (uint64_t)context_tokens * (uint64_t)head_dim;
        const float* head_values = value_cache + (uint64_t)head * (uint64_t)context_tokens * (uint64_t)head_dim;
        // GAP 1A: QKV mode; quantize into persistent state, attention in quantized domain.
        if (engine->config.kv_mode == moe_KV_MODE_QKV && qkv_is_enabled()) {
            qkv_state_t* st = moe_qkv_ensure_head_state(
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

// FATE Predictor: Save hidden state snapshot just before gate
static void save_gate_input_snapshot(moe_pc_engine* engine, int layer, const float* hidden, uint32_t dim) {
    if (!engine || layer < 0 || !hidden || dim == 0) return;
    const size_t layer_u = (size_t)layer;
    // BUGFIX 503: layer_u 범위 체크
    if (layer_u > moe_GATE_SNAPSHOT_MAX_LAYER) return;
    const size_t req_size = (layer_u + 1u) * 2u;
    if (req_size > moe_GATE_SNAPSHOT_EPOCH_SLOTS) return;
    {
        std::lock_guard<std::mutex> lock(engine->gate_input_snapshot_mutex);
        if (engine->gate_input_snapshots.size() < req_size) {
            engine->gate_input_snapshots.resize(req_size);
        }

        const size_t slot0 = layer_u * 2u;
        const size_t slot1 = slot0 + 1u;
        std::atomic<uint64_t>& epoch0_ref = engine->gate_input_snapshot_epochs[slot0];
        std::atomic<uint64_t>& epoch1_ref = engine->gate_input_snapshot_epochs[slot1];
        const uint64_t e0 = epoch0_ref.load(std::memory_order_acquire);
        const uint64_t e1 = epoch1_ref.load(std::memory_order_acquire);
        const int write_idx = (e0 >= e1) ? 1 : 0;
        const size_t write_slot = layer_u * 2u + (size_t)write_idx;

        engine->gate_input_snapshots[write_slot].assign(hidden, hidden + dim);
        const uint64_t new_epoch = std::max(e0, e1) + 1u;
        if (write_idx == 0) {
            epoch0_ref.store(new_epoch, std::memory_order_release);
        } else {
            epoch1_ref.store(new_epoch, std::memory_order_release);
        }
    }
}

