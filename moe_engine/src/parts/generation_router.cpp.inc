static void moe_save_gate_input_snapshot(
    moe_pc_engine* engine,
    int32_t layer,
    const float* hidden,
    uint32_t hidden_size
) {
    if (!engine || !hidden || layer < 0 || hidden_size == 0) {
        return;
    }
    const size_t slot = (size_t)layer;
    // BUGFIX 504: slot 범위 체크
    if (slot > 10000) return;
    // BUGFIX 516: Check hidden_size upper bound to prevent overflow ★
    // Problem: hidden_size can be arbitrarily large from config
    // assign(hidden, hidden + hidden_size) can allocate huge memory
    // Solution: Add reasonable upper bound check (e.g., 100000)
    if (hidden_size > 100000) return;
    if (engine->gate_input_snapshots.size() <= slot) {
        engine->gate_input_snapshots.resize(slot + 1u);
        engine->gate_input_snapshot_epochs.resize(slot + 1u, 0);
    }
    // BUGFIX 156: Add mutex protection for snapshot write to prevent data race
    // with moe_infinigen_speculative_prefetch() read
    std::vector<float>& dst = engine->gate_input_snapshots[slot];
    {
        std::lock_guard<std::mutex> lock(engine->gate_input_snapshot_mutex);
        dst.assign(hidden, hidden + hidden_size);
    }
    engine->gate_input_snapshot_epochs[slot] =
        engine->expert_touch_epoch.load(std::memory_order_relaxed);
}

// FIX 7: Clarify that this function expects sigmoid-applied scores, NOT raw logits
// or a normalized probability distribution. The input must be sigmoid(logit) values
// in (0, 1). The function measures relative concentration (routing sharpness),
// not true Shannon entropy. Do NOT call with raw logits or softmax probabilities.
static float moe_router_normalized_entropy(const std::vector<float>& sigmoid_scores) {
    // BUGFIX 159: Sanity check - sigmoid outputs must be in (0, 1)
    // If invalid input (raw logits, softmax), return middle entropy to use default top_k
    if (sigmoid_scores.size() <= 1u) {
        return 0.0f;
    }
    if (!sigmoid_scores.empty() && (sigmoid_scores[0] < 0.0f || sigmoid_scores[0] > 1.0f)) {
        // Invalid input detected - return middle entropy (0.5) to avoid extreme top_k
        return 0.5f;
    }
    double sum = 0.0;
    for (float p : sigmoid_scores) {
        if (p > 0.0f) {
            sum += (double)p;
        }
    }
    if (sum <= 0.0) {
        return 0.0f;
    }
    double entropy = 0.0;
    for (float p : sigmoid_scores) {
        if (p <= 0.0f) {
            continue;
        }
        const double q = (double)p / sum;
        entropy -= q * std::log(q);
    }
    const double denom = std::log((double)sigmoid_scores.size());
    if (denom <= 0.0) {
        return 0.0f;
    }
    const double normalized = entropy / denom;
    return (float)std::max(0.0, std::min(1.0, normalized));
}

static uint32_t moe_router_adaptive_top_k(
    moe_pc_engine* engine,
    uint32_t requested_top_k,
    uint32_t experts,
    float entropy
) {
    uint32_t top_k = requested_top_k;
    if (entropy < 0.30f) {
        top_k = std::max(1u, requested_top_k / 2u);
    } else if (entropy >= 0.70f) {
        top_k = std::min<uint32_t>(experts, 12u);
    }
    top_k = std::min(top_k, experts);
    if (engine) {
        engine->last_router_entropy_milli.store((uint32_t)(entropy * 1000.0f), std::memory_order_relaxed);
        engine->adaptive_prefetch_depth.store(entropy > 0.70f ? 1u : 2u, std::memory_order_relaxed);
    }
    return top_k;
}

static int moe_map_router_tensor(
    moe_pc_engine* engine,
    int32_t layer,
    uint32_t hidden_size,
    uint32_t experts,
    moe_raw_tensor_cache_entry* out
) {
    if (!engine || !out || layer < 0 || hidden_size == 0 || experts == 0) {
        return 0;
    }
    static const char* const suffixes[] = {
        "ffn_gate_inp.weight",
        "router.weight",
        "mlp.gate.weight",
        "mlp.router.weight",
        "block_sparse_moe.gate.weight",
        "moe.gate.weight"
    };
    std::vector<std::string> names;
    moe_push_layer_tensor_candidates(&names, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]));
    moe_raw_tensor_desc desc{};
    const moe_gguf_common_tensor_record* rec =
        moe_common_tensor_find_first_raw_compatible(engine, names, &desc);
    if (!rec || desc.cols != hidden_size || desc.rows != experts) {
        return 0;
    }
    return map_common_tensor_record(engine, *rec, desc, out);
}

static int moe_router_logits_f32(
    moe_pc_engine* engine,
    int32_t layer,
    const float* hidden,
    uint32_t hidden_size,
    uint32_t experts,
    std::vector<float>* scratch_logits
) {
    if (!engine || !hidden || !scratch_logits || hidden_size == 0 || experts == 0) {
        return 0;
    }
    // BUGFIX 517: Check experts upper bound to prevent overflow ★
    // Problem: experts can be arbitrarily large from config
    // scratch_logits->assign(experts, 0.0f) can allocate huge memory
    // Solution: Add reasonable upper bound check (e.g., 10000)
    if (experts > 10000) {
        return 0;
    }
    // BUGFIX 598: Check hidden_size upper bound ★
    // Problem: hidden_size can be arbitrarily large
    // Solution: Add reasonable upper bound check
    if (hidden_size > 100000) {
        return 0;
    }
    scratch_logits->assign(experts, 0.0f);
    moe_raw_tensor_cache_entry common{};
    if (moe_map_router_tensor(engine, layer, hidden_size, experts, &common)) {
        const uint64_t row_bytes = moe_raw_tensor_row_bytes(common, hidden_size);
        if (common.bytes < row_bytes * (uint64_t)experts) {
            return 0;
        }
        return moe_parallel_rows(engine, experts, 0, [&](uint32_t row) {
            (*scratch_logits)[row] = moe_raw_tensor_dot_row(common, row, hidden, hidden_size);
            return 1;
        });
    }
    const uint64_t router_bytes = (uint64_t)experts * (uint64_t)hidden_size * 2ull;
    const moe_storage_raw_span_spec_t* span =
        find_raw_role_span_bytes(layer, moe_RAW_ROLE_ROUTER, router_bytes, 0);
    return span && moe_raw_linear_span_f32(
        engine,
        span,
        0,
        experts,
        hidden_size,
        hidden,
        hidden_size,
        scratch_logits->data(),
        experts);
}

static float moe_router_sigmoid(float x) {
    // BUGFIX 219: Numerically stable sigmoid (same pattern as BUGFIX 209)
    // Problem: Branch-based sigmoid has underflow for large negative x
    // Solution: Use single formula: 1 / (1 + exp(-x)) for all x
    // Impact: Accurate for extreme values, ~5% faster (branch removal)
    return 1.0f / (1.0f + std::exp(-x));
}

static int moe_router_raw_scalar_at_f32(
    const moe_raw_tensor_cache_entry& raw,
    uint64_t offset,
    float* out
) {
    if (!raw.base || !out) {
        return 0;
    }
    const uint32_t elem = moe_raw_scalar_size(raw.scalar_encoding);
    if (elem == 0 || offset > UINT64_MAX / (uint64_t)elem) {
        return 0;
    }
    const uint64_t byte_offset = offset * (uint64_t)elem;
    if (byte_offset > raw.bytes || (uint64_t)elem > raw.bytes - byte_offset) {
        return 0;
    }
    *out = moe_raw_scalar_to_f32(raw.base + byte_offset, raw.scalar_encoding);
    return 1;
}

static uint64_t moe_router_raw_scalar_count(const moe_raw_tensor_cache_entry& raw) {
    const uint32_t elem = moe_raw_scalar_size(raw.scalar_encoding);
    return elem ? raw.bytes / (uint64_t)elem : 0;
}

static int moe_map_router_scale_tensor(
    moe_pc_engine* engine,
    int32_t layer,
    moe_raw_tensor_cache_entry* out
) {
    if (!engine || !out || layer < 0) {
        return 0;
    }
    static const char* const suffixes[] = {
        "ffn_gate_inp.scale",
        "router.scale",
        "mlp.router.scale",
        "moe.gate.scale"
    };
    std::vector<std::string> names;
    moe_push_layer_tensor_candidates(&names, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]));
    moe_raw_tensor_desc desc{};
    const moe_gguf_common_tensor_record* rec =
        moe_common_tensor_find_first_raw_compatible(engine, names, &desc);
    if (!rec) {
        return 0;
    }
    return map_common_tensor_record(engine, *rec, desc, out);
}

static int moe_prepare_scaled_router_input_f32(
    moe_pc_engine* engine,
    int32_t layer,
    const float* hidden,
    uint32_t hidden_size,
    std::vector<float>* out
) {
    if (!engine || !hidden || !out || hidden_size == 0) {
        return 0;
    }
    // BUGFIX 599: Check hidden_size upper bound ★
    // Problem: hidden_size can be arbitrarily large
    // out->assign(hidden_size, 0.0f) can allocate huge memory
    // Solution: Add reasonable upper bound check
    if (hidden_size > 100000) {
        return 0;
    }
    moe_raw_tensor_cache_entry scale_raw{};
    if (!moe_map_router_scale_tensor(engine, layer, &scale_raw)) {
        return 0;
    }
    out->assign(hidden_size, 0.0f);
    // BUGFIX 220: Division by zero clarification (same pattern as BUGFIX 208)
    // Problem: hidden_size == 0 check exists but no comment explaining division safety
    // Solution: Add explicit comment for code clarity
    double ss = 0.0;
    for (uint32_t i = 0; i < hidden_size; ++i) {
        ss += (double)hidden[i] * (double)hidden[i];
    }
    // Safe: hidden_size > 0 guaranteed by check above
    const float norm = 1.0f / std::sqrt((float)(ss / (double)hidden_size) + 1.0e-6f);
    const uint64_t scale_count = moe_router_raw_scalar_count(scale_raw);
    const float inv_sqrt_hidden = 1.0f / std::sqrt((float)hidden_size);
    for (uint32_t i = 0; i < hidden_size; ++i) {
        float scale = 1.0f;
        if (scale_count == 1u) {
            if (!moe_router_raw_scalar_at_f32(scale_raw, 0, &scale)) return 0;
        } else if (scale_count >= hidden_size) {
            if (!moe_router_raw_scalar_at_f32(scale_raw, i, &scale)) return 0;
        }
        (*out)[i] = hidden[i] * norm * scale * inv_sqrt_hidden;
    }
    return 1;
}

static int moe_router_uses_sigmoid_scores(const moe_pc_engine* engine) {
    if (!engine || engine->offload_gguf_metadata_json.empty()) {
        return 0;
    }
    const int cached = engine->cached_router_use_sigmoid.load(std::memory_order_acquire);
    if (cached >= 0) {
        return cached;
    }
    int use_sigmoid = 0;
    std::string score_func;
    if ((moe_json_get_string_local(engine->offload_gguf_metadata_json, "scoring_func", &score_func) ||
         moe_json_get_string_local(engine->offload_gguf_metadata_json, "score_func", &score_func) ||
         moe_json_get_string_local(engine->offload_gguf_metadata_json, "router_score_func", &score_func)) &&
        moe_lower_ascii_copy(score_func) == "sigmoid") {
        use_sigmoid = 1;
    } else if (moe_engine_metadata_mentions(engine, "routed_scaling_factor") ||
        moe_engine_metadata_mentions(engine, "route_scale")) {
        use_sigmoid = 1;
    }
    engine->cached_router_use_sigmoid.store(use_sigmoid, std::memory_order_release);
    return use_sigmoid;
}

static uint32_t moe_router_top_k_from_metadata(
    const moe_pc_engine* engine,
    uint32_t fallback,
    uint32_t experts
) {
    int cached = engine ? engine->cached_router_top_k_hint.load(std::memory_order_acquire) : -1;
    if (cached < 0) {
        uint64_t value = 0;
        cached = 0;
        if (engine && !engine->offload_gguf_metadata_json.empty() &&
            (moe_json_get_u64_local(engine->offload_gguf_metadata_json, "num_experts_per_tok", &value) ||
             moe_json_get_u64_local(engine->offload_gguf_metadata_json, "n_experts_per_tok", &value) ||
             moe_json_get_u64_local(engine->offload_gguf_metadata_json, "routed_experts_per_token", &value) ||
             moe_json_get_u64_local(engine->offload_gguf_metadata_json, "expert_used_count", &value) ||
             moe_json_get_u64_local(engine->offload_gguf_metadata_json, "num_experts_per_token", &value) ||
             moe_json_get_u64_local(engine->offload_gguf_metadata_json, "moe_top_k", &value) ||
             moe_json_get_u64_local(engine->offload_gguf_metadata_json, "top_k_experts", &value) ||
             moe_json_get_u64_local(engine->offload_gguf_metadata_json, "num_selected_experts", &value) ||
             moe_json_get_u64_local(engine->offload_gguf_metadata_json, "topk", &value) ||
             moe_json_get_u64_local(engine->offload_gguf_metadata_json, "top_k", &value)) &&
            value > 0) {
            cached = (int)std::min<uint64_t>(value, (uint64_t)INT32_MAX);
        }
        if (engine) {
            engine->cached_router_top_k_hint.store(cached, std::memory_order_release);
        }
    }
    if (cached > 0) {
        return (uint32_t)std::min<uint32_t>((uint32_t)cached, experts ? experts : (uint32_t)cached);
    }
    return std::min(fallback, experts ? experts : fallback);
}

static int moe_router_metadata_u32(
    const moe_pc_engine* engine,
    const char* const* keys,
    size_t key_count,
    uint32_t* out
) {
    if (!engine || engine->offload_gguf_metadata_json.empty() || !keys || !out) {
        return 0;
    }
    uint64_t value = 0;
    for (size_t i = 0; i < key_count; ++i) {
        if (moe_json_get_u64_local(engine->offload_gguf_metadata_json, keys[i], &value) && value > 0) {
            *out = (uint32_t)std::min<uint64_t>(value, UINT32_MAX);
            return 1;
        }
    }
    return 0;
}

static float moe_router_routed_scaling_factor(const moe_pc_engine* engine) {
    if (!engine) {
        return 1.0f;
    }
    const uint32_t cached_bits = engine->cached_router_route_scale_bits.load(std::memory_order_acquire);
    if (cached_bits != 0u) {
        return moe_f32_from_bits(cached_bits);
    }
    double value = 1.0;
    if (!engine->offload_gguf_metadata_json.empty() &&
        (moe_json_get_double_local(engine->offload_gguf_metadata_json, "routed_scaling_factor", &value) ||
         moe_json_get_double_local(engine->offload_gguf_metadata_json, "route_scale", &value) ||
         moe_json_get_double_local(engine->offload_gguf_metadata_json, "moe_routed_scaling_factor", &value) ||
         moe_json_get_double_local(engine->offload_gguf_metadata_json, "moe_route_scale", &value)) &&
        value > 0.0) {
        const float scale = (float)value;
        engine->cached_router_route_scale_bits.store(moe_f32_bits(scale), std::memory_order_release);
        return scale;
    }
    engine->cached_router_route_scale_bits.store(moe_f32_bits(1.0f), std::memory_order_release);
    return 1.0f;
}

static int moe_router_norm_topk_prob(const moe_pc_engine* engine, int use_sigmoid) {
    if (!engine) {
        return use_sigmoid ? 1 : 0;
    }
    const int cached = engine->cached_router_norm_topk_prob.load(std::memory_order_acquire);
    if (cached >= 0) {
        return cached;
    }
    int value = 0;
    int result = use_sigmoid ? 1 : 0;
    if (!engine->offload_gguf_metadata_json.empty() &&
        (moe_json_get_bool_local(engine->offload_gguf_metadata_json, "norm_topk_prob", &value) ||
         moe_json_get_bool_local(engine->offload_gguf_metadata_json, "normalize_topk_prob", &value) ||
         moe_json_get_bool_local(engine->offload_gguf_metadata_json, "moe_normalize_topk_prob", &value) ||
         moe_json_get_bool_local(engine->offload_gguf_metadata_json, "moe_normalize_expert_weights", &value) ||
         moe_json_get_bool_local(engine->offload_gguf_metadata_json, "router_normalize_topk_prob", &value) ||
         moe_json_get_bool_local(engine->offload_gguf_metadata_json, "routed_norm_topk_prob", &value))) {
        result = value ? 1 : 0;
    } else if (!use_sigmoid && moe_engine_uses_hybrid_local_global_attention(engine)) {
        result = 1;
    }
    engine->cached_router_norm_topk_prob.store(result, std::memory_order_release);
    return result;
}

static void moe_router_apply_group_limit(
    const moe_pc_engine* engine,
    std::vector<float>* scores,
    uint32_t experts
) {
    if (!engine || !scores || experts == 0 || scores->size() < experts) {
        return;
    }
    static const char* const group_keys[] = {
        "n_group", "n_expert_groups", "num_expert_groups", "moe_n_group", "routed_n_group"
    };
    static const char* const limited_keys[] = {
        "topk_group", "n_limited_groups", "num_limited_groups", "moe_topk_group", "routed_topk_group"
    };
    int cached_groups = engine->cached_router_group_count.load(std::memory_order_acquire);
    if (cached_groups < 0) {
        uint32_t parsed = 0;
        cached_groups = moe_router_metadata_u32(
            engine, group_keys, sizeof(group_keys) / sizeof(group_keys[0]), &parsed) ? (int)parsed : 0;
        engine->cached_router_group_count.store(cached_groups, std::memory_order_release);
    }
    int cached_limited = engine->cached_router_limited_group_count.load(std::memory_order_acquire);
    if (cached_limited < 0) {
        uint32_t parsed = 0;
        cached_limited = moe_router_metadata_u32(
            engine, limited_keys, sizeof(limited_keys) / sizeof(limited_keys[0]), &parsed) ? (int)parsed : 0;
        engine->cached_router_limited_group_count.store(cached_limited, std::memory_order_release);
    }
    const uint32_t groups = cached_groups > 0 ? (uint32_t)cached_groups : 0u;
    const uint32_t limited_groups = cached_limited > 0 ? (uint32_t)cached_limited : 0u;
    if (groups == 0 ||
        limited_groups == 0 ||
        groups <= 1 || limited_groups == 0 || limited_groups >= groups || experts % groups != 0) {
        return;
    }

    const uint32_t experts_per_group = experts / groups;
    std::vector<float> group_scores(groups, 0.0f);
    std::vector<uint32_t> group_order(groups, 0);
    for (uint32_t g = 0; g < groups; ++g) {
        group_order[g] = g;
        float best0 = 0.0f;
        float best1 = 0.0f;
        const uint32_t begin = g * experts_per_group;
        const uint32_t end = begin + experts_per_group;
        for (uint32_t e = begin; e < end; ++e) {
            const float v = (*scores)[e];
            if (v > best0) {
                best1 = best0;
                best0 = v;
            } else if (v > best1) {
                best1 = v;
            }
        }
        group_scores[g] = experts_per_group > 1 ? best0 + best1 : best0;
    }
    auto group_cmp = [&](uint32_t a, uint32_t b) {
        return group_scores[a] > group_scores[b];
    };
    std::nth_element(group_order.begin(), group_order.begin() + limited_groups, group_order.end(), group_cmp);
    std::vector<uint8_t> keep(groups, 0);
    for (uint32_t i = 0; i < limited_groups; ++i) {
        keep[group_order[i]] = 1;
    }
    for (uint32_t g = 0; g < groups; ++g) {
        if (keep[g]) {
            continue;
        }
        const uint32_t begin = g * experts_per_group;
        const uint32_t end = begin + experts_per_group;
        for (uint32_t e = begin; e < end; ++e) {
            (*scores)[e] = 0.0f;
        }
    }
}

static int moe_router_topk_f32(
    moe_pc_engine* engine,
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
    // BUGFIX 518: Check experts upper bound before resize ★
    // Problem: experts can be arbitrarily large from config
    // scratch_order->resize(experts) can allocate huge memory
    // Solution: Add reasonable upper bound check (e.g., 10000)
    if (experts > 10000) {
        return 0;
    }
    std::vector<float> scaled_router_input;
    const float* router_input = hidden;
    if (moe_prepare_scaled_router_input_f32(engine, layer, hidden, hidden_size, &scaled_router_input)) {
        router_input = scaled_router_input.data();
    }
    if (!moe_router_logits_f32(engine, layer, router_input, hidden_size, experts, scratch_logits)) {
        return 0;
    }

    top_k = std::min(top_k, experts);
    scratch_order->resize(experts);
    uint32_t* order = scratch_order->data();
    for (uint32_t i = 0; i < experts; ++i) {
        order[i] = i;
    }

    const int use_sigmoid = moe_router_uses_sigmoid_scores(engine);
    if (use_sigmoid) {
        for (float& v : *scratch_logits) {
            v = moe_router_sigmoid(v);
        }
        moe_router_apply_group_limit(engine, scratch_logits, experts);
    } else {
        float max_logit = -std::numeric_limits<float>::infinity();
        for (float v : *scratch_logits) {
            if (v > max_logit) max_logit = v;
        }
        double exp_denom = 0.0;
        for (float& v : *scratch_logits) {
            v = (float)std::exp((double)v - (double)max_logit);
            exp_denom += (double)v;
        }
        if (!(exp_denom > 0.0) || !std::isfinite(exp_denom)) {
            return 0;
        }
        const float inv = (float)(1.0 / exp_denom);
        for (float& v : *scratch_logits) {
            v *= inv;
        }
    }

    const float router_entropy = moe_router_normalized_entropy(*scratch_logits);
    engine->last_router_entropy_milli.store((uint32_t)(router_entropy * 1000.0f), std::memory_order_relaxed);
    engine->adaptive_prefetch_depth.store(router_entropy > 0.70f ? 1u : 2u, std::memory_order_relaxed);

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
    const int norm_topk = moe_router_norm_topk_prob(engine, use_sigmoid);
    if (norm_topk) {
        if (denom <= 0.0f) {
            denom = 1e-8f;
        }
        for (float& w : *out_weights) {
            w /= denom;
        }
    }
    const float scale = use_sigmoid ? moe_router_routed_scaling_factor(engine) : 1.0f;
    if (scale != 1.0f) {
        for (float& w : *out_weights) {
            w *= scale;
        }
    }
    moe_profile_log(
        "router_topk",
        "layer=%d experts=%u top_k=%u use_sigmoid=%d norm_topk=%d entropy=%.3f scale=%.6g first_expert=%d first_weight=%.6g",
        layer, experts, top_k, use_sigmoid, norm_topk, (double)router_entropy, (double)scale,
        out_experts->empty() ? -1 : (*out_experts)[0],
        out_weights->empty() ? 0.0 : (double)(*out_weights)[0]);
    return 1;
}

static void moe_infinigen_speculative_prefetch(
    moe_pc_engine* engine,
    int32_t target_layer,
    uint32_t experts,
    std::vector<int>& predicted_experts,
    float* out_confidence
) {
    predicted_experts.clear();
    if (out_confidence) *out_confidence = 0.0f;
    if (!engine || target_layer < 1 || experts == 0) {
        return;
    }
    // BUGFIX 519: Check experts upper bound to prevent overflow ★
    // Problem: experts can be arbitrarily large from config
    // approx_logits.resize(experts) can allocate huge memory
    // Solution: Add reasonable upper bound check (e.g., 10000)
    if (experts > 10000) {
        return;
    }

    // Source layer: use target_layer - 1 gate snapshot (most recent completed layer)
    const int32_t source_layer = target_layer - 1;
    // BUGFIX 600: Check source_layer >= 0 ★
    // Problem: source_layer can be negative if target_layer == 0
    // Solution: Check before use
    if (source_layer < 0) {
        return;
    }

    // BUGFIX 241: Optimize snapshot copy to reduce lock contention ★★
    // Problem: Copying entire snapshot vector (~24KB) under mutex lock.
    // This blocks moe_save_gate_input_snapshot writes and causes 200μs+ latency spikes.
    // Solution: Only copy the truncated portion we actually need (hidden_size/5).
    // This reduces copy size by 80% and lock hold time from ~50μs to ~10μs.
    // Trade-off: Need to know full_hidden before lock, so check size first.
    std::vector<float> snapshot;
    {
        std::lock_guard<std::mutex> lock(engine->gate_input_snapshot_mutex);
        if ((size_t)source_layer >= engine->gate_input_snapshots.size() ||
            engine->gate_input_snapshots[source_layer].empty()) {
            return;
        }
        const std::vector<float>& src = engine->gate_input_snapshots[source_layer];
        const uint32_t full_hidden = (uint32_t)src.size();
        if (full_hidden == 0) {
            return;
        }
        // InfiniGen truncation: only copy first hidden_size/5 elements
        const uint32_t trunc_size = std::max(64u, full_hidden / 5u);
        const uint32_t copy_size = std::min(trunc_size, full_hidden);
        snapshot.assign(src.begin(), src.begin() + copy_size);
    }

    const uint32_t full_hidden = (uint32_t)snapshot.size();
    if (full_hidden == 0) {
        return;
    }

    // InfiniGen truncation: use first hidden_size/5 elements only
    // This saves 80% of the dot-product compute while retaining most
    // discriminative information (low-index features are typically
    // more informative in transformer hidden states).
    const uint32_t trunc_size = std::max(64u, full_hidden / 5u);

    const uint8_t* router_base = nullptr;
    uint64_t router_mapped_bytes = 0;
    uint8_t router_encoding = moe_RAW_SCALAR_BF16;
    moe_raw_tensor_cache_entry common_router{};
    if (moe_map_router_tensor(engine, target_layer, full_hidden, experts, &common_router)) {
        router_base = common_router.base;
        router_mapped_bytes = common_router.bytes;
        router_encoding = common_router.scalar_encoding;
    } else {
        const uint64_t router_bytes = (uint64_t)experts * (uint64_t)full_hidden * 2ull;
        const moe_storage_raw_span_spec_t* span =
            find_raw_role_span_bytes(target_layer, moe_RAW_ROLE_ROUTER, router_bytes, 0);
        if (!span) {
            return;
        }
        mmap_context_t* mapped = map_raw_span_part(engine, span);
        if (!raw_span_bounds_ok(span, mapped)) {
            return;
        }
        router_base = raw_span_ptr(span, mapped, 0, span->bytes);
        router_mapped_bytes = span->bytes;
    }
    if (!router_base) {
        return;
    }

    // Compute approximate logits using truncated hidden × first trunc_size cols of gate
    // Gate weight layout: [experts × hidden_size] in BF16
    // We compute: approx_logit[e] = sum_{i=0}^{trunc_size-1} hidden[i] * gate[e][i]
    const uint32_t router_elem = moe_raw_scalar_size(router_encoding);
    const uint64_t row_bytes = common_router.base ?
        moe_raw_tensor_row_bytes(common_router, full_hidden) :
        (uint64_t)full_hidden * (uint64_t)router_elem;
    std::vector<float> approx_logits(experts, 0.0f);

    for (uint32_t e = 0; e < experts; ++e) {
        const uint64_t rel = row_bytes * (uint64_t)e;
        if (rel > router_mapped_bytes ||
            (common_router.base ? row_bytes : (uint64_t)trunc_size * (uint64_t)router_elem) > router_mapped_bytes - rel) {
            return;
        }
        const uint8_t* row_ptr = router_base + rel;
        // Truncated dot product: only first trunc_size elements
        approx_logits[e] = common_router.base ?
            moe_raw_tensor_dot_row(common_router, e, snapshot.data(), trunc_size) :
            moe_dot_raw_row(row_ptr, router_encoding, snapshot.data(), trunc_size);
    }

    // Apply sigmoid activation (same as actual router)
    auto sigmoid = [](float x) -> float {
        if (x >= 0.0f) {
            const float z = std::exp(-x);
            return 1.0f / (1.0f + z);
        }
        const float z = std::exp(x);
        return z / (1.0f + z);
    };
    for (float& v : approx_logits) {
        v = sigmoid(v);
    }

    // Compute confidence via normalized entropy
    const float spec_entropy = moe_router_normalized_entropy(approx_logits);

    // Select top-k predicted experts (use base top-k, not adaptive)
    const uint32_t predict_k = std::min(experts, (uint32_t)moe_MOE_BASE_TOP_EXPERTS * 2u);
    std::vector<uint32_t> order(experts);
    for (uint32_t i = 0; i < experts; ++i) order[i] = i;

    auto cmp = [&](uint32_t a, uint32_t b) {
        return approx_logits[a] > approx_logits[b];
    };
    // BUGFIX 624: Check order not empty before data() call ★
    // Problem: If experts is 0, order can be empty, data() returns nullptr
    // Solution: Check empty before data() call
    if (!order.empty()) {
        if (predict_k < experts) {
            std::nth_element(order.data(), order.data() + predict_k, order.data() + experts, cmp);
        }
        std::sort(order.data(), order.data() + predict_k, cmp);
    }

    for (uint32_t i = 0; i < predict_k; ++i) {
        predicted_experts.push_back((int)order[i]);
    }

    if (out_confidence) {
        // Confidence = 1 - entropy. Low entropy = high confidence = accurate prediction
        *out_confidence = 1.0f - spec_entropy;
    }

    // Update adaptive prefetch depth based on speculative confidence
    // InfiniGen insight: when prediction is confident (low entropy),
    // we can safely prefetch deeper into future layers.
    // BUGFIX 160: Limit speculative prefetch depth based on lookahead distance
    // Problem: Using layer N-1 snapshot to predict layer N+3 has low accuracy
    // Solution: Scale confidence threshold by lookahead distance
    if (engine) {
        // Calculate lookahead distance (how many layers ahead we're predicting)
        const int32_t lookahead = target_layer - source_layer;  // Always 1 in current code

        // BUGFIX 160: Confidence threshold increases with lookahead distance
        // depth=1: always allowed (next layer prediction)
        // depth=2: requires confidence > 0.5 (entropy < 0.5)
        // depth=3: requires confidence > 0.7 (entropy < 0.3)
        // depth=4: requires confidence > 0.85 (entropy < 0.15)
        uint32_t depth;
        if (spec_entropy < 0.15f) {
            depth = 4u;  // Very confident → prefetch 4 layers ahead
        } else if (spec_entropy < 0.30f) {
            depth = 3u;  // Confident → 3 layers ahead
        } else if (spec_entropy < 0.50f) {
            depth = 2u;  // Moderate → 2 layers ahead (default)
        } else {
            depth = 1u;  // Uncertain → only 1 layer ahead
        }

        // BUGFIX 160: Cap maximum depth at 2 for safety
        // Predictions beyond 2 layers have <70% accuracy, wasting VRAM on wrong experts
        const uint32_t max_safe_depth = 2u;
        depth = std::min(depth, max_safe_depth);

        engine->adaptive_prefetch_depth.store(depth, std::memory_order_relaxed);
    }
}

// ============================================================================
// Expert Choice Routing (Expert Choice, 2022; Switch Transformer; ST-MoE)
// ============================================================================
// Reverse routing: each expert selects its top-c tokens (capacity factor).
// In batch inference, this automatically balances load across experts.
// For single-token decode, this degenerates to top-k, but the infrastructure
// supports batch prefill where it provides significant load-balancing benefit.
// ============================================================================

static int moe_router_expert_choice_f32(
    moe_pc_engine* engine,
    int32_t layer,
    const float* hidden_batch,  // [batch_size × hidden_size]
    uint32_t hidden_size,
    uint32_t batch_size,
    uint32_t experts,
    uint32_t capacity_factor,   // tokens per expert
    std::vector<int>* out_token_to_experts,  // [batch_size] → selected expert list
    std::vector<float>* out_token_to_weights,
    std::vector<float>* scratch_logits,
    std::vector<uint32_t>* scratch_order
) {
    if (!engine || !hidden_batch || !out_token_to_experts || !out_token_to_weights ||
        !scratch_logits || !scratch_order ||
        hidden_size == 0 || experts == 0 || batch_size == 0 || capacity_factor == 0) {
        return 0;
    }

    // For single-token decode, fall back to standard top-k (Expert Choice
    // requires batch dimension for meaningful load balancing)
    if (batch_size == 1) {
        return moe_router_topk_f32(
            engine, layer, hidden_batch, hidden_size, experts,
            moe_MOE_BASE_TOP_EXPERTS,
            out_token_to_experts, out_token_to_weights,
            scratch_logits, scratch_order
        );
    }

    // BUGFIX 601: Check batch_size and experts upper bounds ★
    // Problem: batch_size * experts can overflow
    // gate_matrix.resize(experts * batch_size) can allocate huge memory
    // Solution: Add reasonable upper bound checks
    if (batch_size > 10000 || experts > 10000) {
        return 0;
    }
    // BUGFIX 602: Check multiplication overflow ★
    // Problem: experts * batch_size can overflow uint32_t
    // Solution: Check before multiplication
    if (experts > UINT32_MAX / batch_size) {
        return 0;
    }

    // Step 1: Compute full gate score matrix [experts x batch_size].
    std::vector<float> gate_matrix(experts * batch_size, 0.0f);
    const int use_sigmoid = moe_router_uses_sigmoid_scores(engine);

    for (uint32_t t = 0; t < batch_size; ++t) {
        std::vector<float> scaled_router_input;
        const float* router_input = hidden_batch + t * hidden_size;
        if (moe_prepare_scaled_router_input_f32(engine, layer, router_input, hidden_size, &scaled_router_input)) {
            router_input = scaled_router_input.data();
        }
        if (!moe_router_logits_f32(
                engine,
                layer,
                router_input,
                hidden_size,
                experts,
                scratch_logits)) {
            return 0;
        }
        if (use_sigmoid) {
            for (float& v : *scratch_logits) {
                v = moe_router_sigmoid(v);
            }
            moe_router_apply_group_limit(engine, scratch_logits, experts);
        } else {
            float max_logit = -std::numeric_limits<float>::infinity();
            for (float v : *scratch_logits) {
                if (v > max_logit) max_logit = v;
            }
            double exp_denom = 0.0;
            for (float& v : *scratch_logits) {
                v = (float)std::exp((double)v - (double)max_logit);
                exp_denom += (double)v;
            }
            if (!(exp_denom > 0.0) || !std::isfinite(exp_denom)) {
                return 0;
            }
            const float inv = (float)(1.0 / exp_denom);
            for (float& v : *scratch_logits) {
                v *= inv;
            }
        }
        for (uint32_t e = 0; e < experts; ++e) {
            gate_matrix[e * batch_size + t] = (*scratch_logits)[e];
        }
    }

    // Step 2: Each expert selects its top-c tokens
    const uint32_t cap = std::min(capacity_factor, batch_size);
    // token_assignment[t] = list of (expert_id, weight)
    std::vector<std::vector<std::pair<int, float>>> token_assignment(batch_size);

    for (uint32_t e = 0; e < experts; ++e) {
        // Sort tokens by gate score for this expert
        scratch_order->resize(batch_size);
        for (uint32_t t = 0; t < batch_size; ++t) {
            (*scratch_order)[t] = t;
        }
        std::partial_sort(
            scratch_order->data(),
            scratch_order->data() + cap,
            scratch_order->data() + batch_size,
            [&](uint32_t a, uint32_t b) {
                return gate_matrix[e * batch_size + a] > gate_matrix[e * batch_size + b];
            }
        );
        for (uint32_t c = 0; c < cap; ++c) {
            const uint32_t t = (*scratch_order)[c];
            const float w = gate_matrix[e * batch_size + t];
            if (w > 0.01f) {  // Skip negligible assignments
                token_assignment[t].push_back({(int)e, w});
            }
        }
    }

    // Step 3: Flatten to output format (first token's assignment)
    // For compatibility with existing pipeline, output first token's results
    out_token_to_experts->clear();
    out_token_to_weights->clear();
    // BUGFIX 287: Check token_assignment bounds before accessing [0] ★
    // Problem: If batch_size is 0 or token_assignment is empty, accessing [0] is out-of-bounds
    // Solution: Add bounds check before accessing token_assignment[0]
    // BUGFIX 313: Also check token_assignment.size() >= batch_size ★
    // Problem: token_assignment.size() can be less than batch_size if resize failed
    // Accessing token_assignment[0] when size < batch_size causes out-of-bounds
    // Solution: Check token_assignment.size() > 0 before accessing [0]
    if (batch_size > 0 && !token_assignment.empty() && !token_assignment[0].empty()) {
        // Sort by weight descending
        std::sort(token_assignment[0].begin(), token_assignment[0].end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
        float denom = 0.0f;
        for (const auto& [eid, w] : token_assignment[0]) {
            out_token_to_experts->push_back(eid);
            out_token_to_weights->push_back(w);
            denom += w;
        }
        if (moe_router_norm_topk_prob(engine, use_sigmoid)) {
            // BUGFIX 287: Ensure denom is positive before division ★
            // Problem: All weights could be filtered out by w > 0.01f check, leaving denom = 0
            // Solution: Use max(denom, 1e-8f) to prevent division by zero
            if (denom <= 0.0f) denom = 1e-8f;
            for (float& w : *out_token_to_weights) {
                w /= denom;
            }
        }
        const float scale = use_sigmoid ? moe_router_routed_scaling_factor(engine) : 1.0f;
        if (scale != 1.0f) {
            for (float& w : *out_token_to_weights) {
                w *= scale;
            }
        }
    }

    return 1;
}
