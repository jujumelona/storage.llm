// Router top-K selection

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
    // BUGFIX 668: Remove duplicate limited_groups == 0 check ★
    if (groups <= 1 || limited_groups == 0 || limited_groups >= groups || experts % groups != 0) {
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
    std::vector<uint32_t>* scratch_order,
    moe_decode_scratch* session = nullptr
) {
    if (!engine || !hidden || !out_experts || !out_weights || !scratch_logits || !scratch_order ||
        hidden_size == 0 || experts == 0 || top_k == 0) {
        return 0;
    }
    // BUGFIX 518: Check experts upper bound before resize ??
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
        // BUGFIX 665: Check for NaN/Inf before softmax computation ★★★
        // BUGFIX 667: Prevent NaN propagation in router logits ★★
        float max_logit = -std::numeric_limits<float>::infinity();
        bool has_invalid = false;
        for (float v : *scratch_logits) {
            if (!std::isfinite(v)) {
                has_invalid = true;
                break;
            }
            if (v > max_logit) max_logit = v;
        }
        if (has_invalid || !std::isfinite(max_logit)) {
            return 0;
        } else {
            double exp_denom = 0.0;
            std::vector<float> exp_values(scratch_logits->size());
            for (size_t i = 0; i < scratch_logits->size(); ++i) {
                exp_values[i] = (float)std::exp((double)(*scratch_logits)[i] - (double)max_logit);
                exp_denom += (double)exp_values[i];
            }
            if (!(exp_denom > 0.0) || !std::isfinite(exp_denom)) {
                return 0;
            } else {
                const float inv = (float)(1.0 / exp_denom);
                for (size_t i = 0; i < scratch_logits->size(); ++i) {
                    (*scratch_logits)[i] = exp_values[i] * inv;
                }
            }
        }
    }

    float router_entropy = 0.5f;
    if (use_sigmoid) {
        router_entropy = moe_router_normalized_entropy(*scratch_logits);
        const uint32_t entropy_milli = (uint32_t)(router_entropy * 1000.0f);
        // BUGFIX Problem F stage 2: Per-session entropy and prefetch depth ★★★
        // Primary: write to session-level fields for proper isolation
        // Secondary: engine-level atomics kept for profiling/diagnostics only
        if (session) {
            session->session_router_entropy_milli = entropy_milli;
        }
        engine->last_router_entropy_milli.store(entropy_milli, std::memory_order_relaxed);
        // Fix 22: Hysteresis for adaptive_prefetch_depth (now per-session)
        const uint32_t current_depth = session ?
            session->session_adaptive_prefetch_depth :
            engine->adaptive_prefetch_depth.load(std::memory_order_relaxed);
        uint32_t new_depth = router_entropy > 0.70f ? 1u : 2u;
        if (new_depth == 1u && current_depth == 2u) {
            if (router_entropy > 0.70f * 1.05f) new_depth = 1u; else new_depth = current_depth;
        } else if (new_depth == 2u && current_depth == 1u) {
            if (router_entropy < 0.70f * 0.95f) new_depth = 2u; else new_depth = current_depth;
        }
        if (session) {
            session->session_adaptive_prefetch_depth = new_depth;
        }
        engine->adaptive_prefetch_depth.store(new_depth, std::memory_order_relaxed);
    } else {
        // Raw-logit/softmax routers do not feed the sigmoid entropy heuristic.
        if (session) {
            session->session_router_entropy_milli = 500u;
        }
        engine->last_router_entropy_milli.store(500u, std::memory_order_relaxed);
    }

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
        // BUGFIX 666: Prevent weight explosion from near-zero denom ★★★
        // Problem: denom <= 0 → set to 1e-8f → weights explode by 1e8x → PPL corruption
        // Solution: Use uniform distribution fallback when denom too small
        if (denom <= 1e-6f) {
            return 0;
        } else {
            for (float& w : *out_weights) {
                w /= denom;
            }
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
    float* out_confidence,
    moe_decode_scratch* session = nullptr
) {
    predicted_experts.clear();
    if (out_confidence) *out_confidence = 0.0f;
    if (!engine || target_layer < 1 || experts == 0) {
        return;
    }
    // BUGFIX 519: Check experts upper bound to prevent overflow ??
    // Problem: experts can be arbitrarily large from config
    // approx_logits.resize(experts) can allocate huge memory
    // Solution: Add reasonable upper bound check (e.g., 10000)
    if (experts > 10000) {
        return;
    }

    // Source layer: use target_layer - 1 gate snapshot (most recent completed layer)
    const int32_t source_layer = target_layer - 1;
    // BUGFIX 600: Check source_layer >= 0 ??
    // Problem: source_layer can be negative if target_layer == 0
    // Solution: Check before use
    if (source_layer < 0) {
        return;
    }

    // BUGFIX 241: Optimize snapshot copy to reduce lock contention ?�★
    // Problem: Copying entire snapshot vector (~24KB) under mutex lock.
