// MLP normalization operations
}

static int moe_layer_shared_expert_gate_f32(
    moe_pc_engine* engine,
    int32_t layer,
    const float* input,
    uint32_t hidden_size,
    float* out_gate
) {
    if (!out_gate) {
        return 0;
    }
    *out_gate = 1.0f;
    if (!engine || !input || hidden_size == 0) {
        return 0;
    }
    static const char* const suffixes[] = {
        "shared_expert_gate.weight",
        "shared_expert.gate.weight",
        "ffn_shared_gate.weight",
        "shared_gate.weight"
    };
    moe_raw_tensor_cache_entry raw{};
    if (!moe_map_layer_common_tensor(engine, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]), &raw)) {
        return 1;
    }
    if (raw.cols != hidden_size || raw.rows == 0) {
        return 0;
    }
    const float logit = moe_raw_tensor_dot_row(raw, 0, input, hidden_size);
    // BUGFIX 209: Numerically stable sigmoid computation
    // Problem: logit < 0 branch uses exp(logit) / (1 + exp(logit)) which underflows for large negative logit
    // Solution: Use log-sum-exp trick: sigmoid(x) = 1 / (1 + exp(-x)) for all x
    // For x < 0: 1 / (1 + exp(-x)) = exp(x) / (exp(x) + 1) but direct formula is more stable
    *out_gate = 1.0f / (1.0f + std::exp(-logit));
    return 1;
}

static int moe_layer_apply_expert_down_scale_f32(
    moe_pc_engine* engine,
    int32_t layer,
    int expert_id,
    uint32_t experts,
    float* values,
    uint32_t hidden_size
) {
    if (!engine || !values || expert_id < 0 || hidden_size == 0) {
        return 0;
    }
    // BUGFIX 508: Check hidden_size upper bound to prevent overflow ??
    // Problem: hidden_size can be arbitrarily large from config
    // Loop for (i = 0; i < hidden_size; ++i) can run billions of iterations
    // Solution: Add reasonable upper bound check (e.g., 100000)
    if (hidden_size > 100000) {
        return 0;
    }
    // BUGFIX 509: Check experts upper bound to prevent overflow ??
    // Problem: experts can be arbitrarily large from config
    // expert_id < experts check can pass with huge experts value
    // Solution: Add reasonable upper bound check (e.g., 10000)
    if (experts > 10000) {
        return 0;
    }
    static const char* const suffixes[] = {
        "ffn_down_exps.scale",
        "ffn_gate_inp.per_expert_scale",
        "router.per_expert_scale",
        "mlp.router.per_expert_scale",
        "moe.gate.per_expert_scale"
    };
    moe_raw_tensor_cache_entry raw{};
    if (!moe_map_layer_common_tensor(engine, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]), &raw)) {
        return 1;
    }
    const uint64_t count = moe_raw_tensor_scalar_count(raw);
    if (count == 1u) {
        float scale = 1.0f;
        if (!moe_raw_tensor_scalar_at_f32(raw, 0, &scale)) return 0;
        moe_scale_f32(values, scale, hidden_size);
        return 1;
    }
    if (experts && count == experts && (uint32_t)expert_id < experts) {
        float scale = 1.0f;
        if (!moe_raw_tensor_scalar_at_f32(raw, (uint64_t)expert_id, &scale)) return 0;
        moe_scale_f32(values, scale, hidden_size);
        return 1;
    }
    const uint64_t base = (uint64_t)expert_id * (uint64_t)hidden_size;
    if (base <= count && (uint64_t)hidden_size <= count - base) {
        for (uint32_t i = 0; i < hidden_size; ++i) {
            float scale = 1.0f;
            if (!moe_raw_tensor_scalar_at_f32(raw, base + i, &scale)) return 0;
            values[i] *= scale;
        }
        return 1;
    }
    return 1;
}

static int moe_add_mlp_residual_f32(
    moe_pc_engine* engine,
    int32_t layer,
    moe_decode_scratch* s,
    float* hidden,
    uint32_t hidden_size
) {
    if (!engine || !s || !hidden) {
        return 0;
    }
    const moe_attn_config_t cfg = moe_generation_attn_config(engine);
    if (moe_layer_post_ffw_norm_f32(
            engine, layer, s->mlp.data(), hidden_size, s->shared.data(), cfg.rms_norm_eps)) {
        for (uint32_t i = 0; i < hidden_size; ++i) {
            hidden[i] += s->shared[i];
        }
        return 1;
    }
    for (uint32_t i = 0; i < hidden_size; ++i) {
        hidden[i] += s->mlp[i];
    }
    return 1;
}

static int moe_layer_common_dense_mlp_f32(
    moe_pc_engine* engine,
    int32_t layer,
    moe_decode_scratch* s,
    const float* input,
    uint32_t hidden_size,
    float* out_hidden
) {
    static const char* const gate_suffixes[] = {
        "ffn_gate.weight",
        "mlp.gate_proj.weight",
        "gate_proj.weight",
        "feed_forward.w1.weight",
        "ffn.w1.weight",
        "w1.weight",
        "shared_expert.gate_proj.weight",
        "shared_experts.gate_proj.weight",
        "shared_expert.w1.weight",
        "shared_experts.w1.weight",
        "ffn_shared_expert_gate.weight",
        "ffn_shared_exps_gate.weight",
        "ffn_shared_gate_proj.weight"
    };
    static const char* const up_suffixes[] = {
        "ffn_up.weight",
        "mlp.up_proj.weight",
        "up_proj.weight",
        "feed_forward.w3.weight",
        "ffn.w3.weight",
        "w3.weight",
        "shared_expert.up_proj.weight",
        "shared_experts.up_proj.weight",
        "shared_expert.w3.weight",
        "shared_experts.w3.weight",
        "ffn_shared_expert_up.weight",
        "ffn_shared_exps_up.weight",
        "ffn_shared_up_proj.weight"
    };
    static const char* const down_suffixes[] = {
        "ffn_down.weight",
        "mlp.down_proj.weight",
        "down_proj.weight",
        "feed_forward.w2.weight",
        "ffn.w2.weight",
        "w2.weight",
        "shared_expert.down_proj.weight",
        "shared_experts.down_proj.weight",
        "shared_expert.w2.weight",
        "shared_experts.w2.weight",
        "ffn_shared_expert_down.weight",
        "ffn_shared_exps_down.weight",
        "ffn_shared_down_proj.weight"
    };
    const uint64_t total_start_us = moe_profile_now_us();
    const uint64_t map_start_us = moe_profile_now_us();
    moe_profile_log("mlp_dense_begin", "layer=%d hidden=%u", layer, hidden_size);
    moe_raw_tensor_cache_entry gate{};
    moe_raw_tensor_cache_entry up{};
    moe_raw_tensor_cache_entry down{};
    if (!moe_map_layer_common_tensor(engine, layer, gate_suffixes, sizeof(gate_suffixes) / sizeof(gate_suffixes[0]), &gate) ||
        !moe_map_layer_common_tensor(engine, layer, up_suffixes, sizeof(up_suffixes) / sizeof(up_suffixes[0]), &up) ||
        !moe_map_layer_common_tensor(engine, layer, down_suffixes, sizeof(down_suffixes) / sizeof(down_suffixes[0]), &down)) {
        return 0;
    }
    const uint32_t intermediate = gate.rows;
    // BUGFIX 510: Check intermediate upper bound to prevent overflow ??
    // Problem: intermediate can be arbitrarily large from tensor shape
    // Used in scratch buffer resize and parallel_rows loop
    // Solution: Add reasonable upper bound check (e.g., 100000)
    if (intermediate == 0 || intermediate > 100000) {
        return 0;
    }
    if (up.rows != intermediate ||
        gate.cols != hidden_size || up.cols != hidden_size ||
        down.rows != hidden_size || down.cols != intermediate) {
        return 0;
    }
    // BUGFIX 213: Add diagnostic logging for scratch buffer size mismatch
    // Problem: Silent failure when scratch buffers are undersized, hard to debug
    // Solution: Log error with layer and size info before returning
    if (s->gate.size() < intermediate || s->up.size() < intermediate || s->shared.size() < hidden_size) {
        fprintf(stderr, "[storageLLM ERROR] Dense MLP scratch buffer size mismatch: "
            "layer %d, need gate=%u up=%u shared=%u, have gate=%zu up=%zu shared=%zu\n",
            layer, intermediate, intermediate, hidden_size,
            s->gate.size(), s->up.size(), s->shared.size());
        return 0;
    }
    const uint64_t gate_row_bytes = gate.base ? moe_raw_tensor_row_bytes(gate, gate.cols) : 0;
    const uint64_t up_row_bytes = up.base ? moe_raw_tensor_row_bytes(up, up.cols) : 0;
    const uint64_t down_row_bytes = down.base ? moe_raw_tensor_row_bytes(down, down.cols) : 0;
    const uint8_t gate_effective_format = moe_effective_weight_format_for_row_bytes(gate, gate.cols);
    const uint8_t up_effective_format = moe_effective_weight_format_for_row_bytes(up, up.cols);
    const uint8_t down_effective_format = moe_effective_weight_format_for_row_bytes(down, down.cols);
    moe_profile_log(
        "mlp_dense_map",
        "layer=%d intermediate=%u gate=%ux%u up=%ux%u down=%ux%u "
        "gate_fmt=%u gate_eff_fmt=%u gate_scalar=%u gate_row_bytes=%llu gate_weight_row_bytes=%llu gate_bytes=%llu "
        "up_fmt=%u up_eff_fmt=%u up_scalar=%u up_row_bytes=%llu up_weight_row_bytes=%llu up_bytes=%llu "
        "down_fmt=%u down_eff_fmt=%u down_scalar=%u down_row_bytes=%llu down_weight_row_bytes=%llu down_bytes=%llu ms=%.3f",
        layer, intermediate, gate.rows, gate.cols, up.rows, up.cols, down.rows, down.cols,
        (unsigned)gate.weight_format, (unsigned)gate_effective_format, (unsigned)gate.scalar_encoding,
        (unsigned long long)gate_row_bytes, (unsigned long long)gate.weight_row_bytes, (unsigned long long)gate.bytes,
        (unsigned)up.weight_format, (unsigned)up_effective_format, (unsigned)up.scalar_encoding,
        (unsigned long long)up_row_bytes, (unsigned long long)up.weight_row_bytes, (unsigned long long)up.bytes,
        (unsigned)down.weight_format, (unsigned)down_effective_format, (unsigned)down.scalar_encoding,
        (unsigned long long)down_row_bytes, (unsigned long long)down.weight_row_bytes, (unsigned long long)down.bytes,
        moe_profile_elapsed_ms(map_start_us));
    const uint64_t gate_up_start_us = moe_profile_now_us();
    const int activation_mode = moe_engine_gelu_activation_mode(engine);
    {
        moe_profile_parallel_label_scope rows_label("dense_mlp_gate_up");
        if (!moe_parallel_rows(engine, intermediate, 0, [&](uint32_t row) {
            const float g = moe_raw_tensor_dot_row(gate, row, input, hidden_size);
            const float u = moe_raw_tensor_dot_row(up, row, input, hidden_size);
            s->gate[row] = moe_generation_gated_activation_cached(activation_mode, g, u);
            return 1;
        })) {
            return 0;
        }
    }
    const double gate_up_ms = moe_profile_elapsed_ms(gate_up_start_us);
    const uint64_t down_start_us = moe_profile_now_us();
    if (!moe_common_linear_f32(engine, down, s->gate.data(), intermediate, out_hidden, hidden_size)) {
        return 0;
    }
    moe_profile_log(
        "mlp_dense_end",
        "layer=%d intermediate=%u hidden=%u gate_up_ms=%.3f down_ms=%.3f total_ms=%.3f",
        layer, intermediate, hidden_size, gate_up_ms,
        moe_profile_elapsed_ms(down_start_us), moe_profile_elapsed_ms(total_start_us));
    return 1;
}

static int moe_layer_dense_mlp_f32(
    moe_pc_engine* engine,
    int32_t layer,
    moe_decode_scratch* s,
    float* hidden,
    uint32_t hidden_size
) {
    const uint64_t total_start_us = moe_profile_now_us();
    if (moe_layer_common_dense_mlp_f32(engine, layer, s, s->norm.data(), hidden_size, s->mlp.data())) {
        const uint64_t residual_start_us = moe_profile_now_us();
        const int ok = moe_add_mlp_residual_f32(engine, layer, s, hidden, hidden_size);
        moe_profile_log(
            "mlp_dense_residual",
            "layer=%d ok=%d residual_ms=%.3f total_ms=%.3f",
            layer, ok, moe_profile_elapsed_ms(residual_start_us), moe_profile_elapsed_ms(total_start_us));
        return ok;
    }
    const uint32_t dense_intermediate = 12288u;
    const uint64_t bytes =
        2ull * (uint64_t)dense_intermediate * (uint64_t)hidden_size * 2ull +
        (uint64_t)hidden_size * (uint64_t)dense_intermediate * 2ull;
    // Bug #5 Fix: Dense MLP should use DENSE_MLP role, not ROUTER.
    // Fall back to ROUTER for backward compatibility with old manifests.
    const moe_storage_raw_span_spec_t* span = find_raw_role_span_bytes(layer, moe_RAW_ROLE_DENSE_MLP, bytes, 0);
    if (!span) {
        span = find_raw_role_span_bytes(layer, moe_RAW_ROLE_ROUTER, bytes, 0);
    }
    if (!span) {
        return 0;
    }
    moe_profile_log("mlp_dense_fallback_begin", "layer=%d hidden=%u intermediate=%u", layer, hidden_size, dense_intermediate);
    const uint64_t raw_span_start_us = moe_profile_now_us();
    if (!moe_raw_mlp_span_f32(
            engine, span, s->norm.data(), hidden_size, dense_intermediate,
            s->gate.data(), s->up.data(), s->mlp.data())) {
        return 0;
    }
    const uint64_t residual_start_us = moe_profile_now_us();
    const int ok = moe_add_mlp_residual_f32(engine, layer, s, hidden, hidden_size);
    moe_profile_log(
        "mlp_dense_fallback_end",
        "layer=%d ok=%d raw_span_ms=%.3f residual_ms=%.3f total_ms=%.3f",
        layer, ok, moe_profile_elapsed_ms(raw_span_start_us),
        moe_profile_elapsed_ms(residual_start_us), moe_profile_elapsed_ms(total_start_us));
    return ok;
}

static int moe_layer_moe_mlp_f32(
    moe_pc_engine* engine,
    int32_t layer,
    moe_decode_scratch* s,
    float* hidden,
    const moe_storage_constants_t* c
) {
    const uint64_t total_start_us = moe_profile_now_us();
    const uint32_t hidden_size = c->hidden_size;
    const uint32_t configured_expert_intermediate = c->expert_intermediate_size;
    const moe_attn_config_t attn_cfg = moe_generation_attn_config(engine);
    moe_profile_log(
        "mlp_moe_begin",
        "layer=%d hidden=%u configured_intermediate=%u experts=%u first_moe=%u last_moe=%u",
        layer, hidden_size, configured_expert_intermediate,
        c->experts_per_moe_layer, c->first_moe_layer, c->last_moe_layer);
    moe_profile_log(
        "mlp_moe_scratch",
        "layer=%d hidden=%u mlp=%zu shared=%zu expert=%zu norm=%zu residual=%zu",
        layer,
        hidden_size,
        s ? s->mlp.size() : 0u,
        s ? s->shared.size() : 0u,
        s ? s->expert.size() : 0u,
        s ? s->norm.size() : 0u,
        s ? s->residual.size() : 0u);
    if (!s || s->mlp.size() < hidden_size || s->shared.size() < hidden_size ||
        s->expert.size() < hidden_size || s->norm.size() < hidden_size ||
        s->residual.size() < hidden_size) {
        std::fprintf(stderr,
            "[storagellm mlp] scratch too small layer=%d hidden=%u mlp=%zu shared=%zu expert=%zu norm=%zu residual=%zu\n",
            layer,
            hidden_size,
            s ? s->mlp.size() : 0u,
            s ? s->shared.size() : 0u,
            s ? s->expert.size() : 0u,
            s ? s->norm.size() : 0u,
            s ? s->residual.size() : 0u);
        return 0;
    }
    std::fill(s->mlp.begin(), s->mlp.begin() + hidden_size, 0.0f);
    moe_profile_log("mlp_moe_zero", "layer=%d hidden=%u", layer, hidden_size);

    const uint64_t norm_start_us = moe_profile_now_us();
    const float* shared_input = s->norm.data();
    const float* expert_input = s->norm.data();
    moe_profile_log(
        "mlp_moe_norm_begin",
        "layer=%d hidden=%u hidden_ptr=%p expert_ptr=%p",
        layer, hidden_size, (void*)hidden, (void*)s->expert.data());
    const int moe_norm_ok = moe_layer_moe_norm_f32(
        engine, layer, hidden, hidden_size, s->expert.data(), attn_cfg.rms_norm_eps);
    if (moe_norm_ok) {
        expert_input = s->expert.data();
    }
    moe_profile_log("mlp_moe_norm", "layer=%d ok=%d ms=%.3f", layer, moe_norm_ok, moe_profile_elapsed_ms(norm_start_us));

    const uint64_t shared_bytes =
        2ull * (uint64_t)configured_expert_intermediate * (uint64_t)hidden_size * 2ull +
        (uint64_t)hidden_size * (uint64_t)configured_expert_intermediate * 2ull;
    const moe_storage_raw_span_spec_t* shared = find_raw_role_span_bytes(layer, moe_RAW_ROLE_SHARED, shared_bytes, 0);
    int shared_ok = 0;
    const uint64_t shared_start_us = moe_profile_now_us();
    if (moe_layer_common_dense_mlp_f32(engine, layer, s, shared_input, hidden_size, s->shared.data())) {
        shared_ok = 1;
    } else if (shared &&
        moe_raw_mlp_span_f32(
            engine, shared, shared_input, hidden_size, configured_expert_intermediate,
            s->gate.data(), s->up.data(), s->shared.data())) {
        shared_ok = 1;
    }
    if (shared_ok) {
        float shared_gate = 1.0f;
        if (!moe_layer_shared_expert_gate_f32(engine, layer, shared_input, hidden_size, &shared_gate)) {
            return 0;
        }
        const float* shared_out = s->shared.data();
        if (moe_layer_post_ffw_norm1_f32(
                engine, layer, s->shared.data(), hidden_size, s->residual.data(), attn_cfg.rms_norm_eps)) {
            shared_out = s->residual.data();
        }
        for (uint32_t i = 0; i < hidden_size; ++i) {
            s->mlp[i] += shared_out[i] * shared_gate;
        }
    }
    moe_profile_log(
        "mlp_moe_shared",
        "layer=%d ok=%d has_raw_span=%d ms=%.3f",
        layer, shared_ok, shared ? 1 : 0, moe_profile_elapsed_ms(shared_start_us));

    const int router_has_internal_norm_scale =
        moe_engine_has_common_tensor_suffix(engine, layer, "ffn_gate_inp.scale") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "router.scale") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "mlp.router.scale") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "moe.gate.scale");
    const float* router_input = router_has_internal_norm_scale ? hidden : expert_input;
    moe_save_gate_input_snapshot(engine, layer, router_input, hidden_size);
    const uint64_t router_start_us = moe_profile_now_us();
    if (!moe_router_topk_f32(
            engine,
            layer,
            router_input,
            hidden_size,
            c->experts_per_moe_layer,
            moe_router_top_k_from_metadata(engine, moe_MOE_BASE_TOP_EXPERTS, c->experts_per_moe_layer),
            &s->selected_experts,
            &s->selected_weights,
            &s->router_logits,
            &s->router_order)) {
        std::fprintf(stderr,
            "[storagellm mlp] router top-k failed layer=%d hidden=%u experts=%u topk=%u\n",
            layer,
            hidden_size,
            c->experts_per_moe_layer,
            moe_router_top_k_from_metadata(engine, moe_MOE_BASE_TOP_EXPERTS, c->experts_per_moe_layer));
        return 0;
    }
    moe_juju_update_co_activation(engine, layer, s->selected_experts);

    // BUGFIX 문제10: Update heat for actually used experts ?�★
    // Problem: touch_expert_heat only called during tier transitions, not during actual use
    // Solution: Touch heat for each selected expert after router selection
    // This ensures frequently-used experts maintain high heat scores
    // Fix 23: Removed map_mutex from hot path. expert_states vector is stable after init,
    // so reading tier/active without lock is safe. touch_expert_heat only writes hot_score
    // which is tolerant of stale reads (proactive eviction re-sorts every 25ms).
    for (int expert_id : s->selected_experts) {
        const size_t slot = expert_dense_slot(engine, layer, expert_id);
        if (slot == tensor_slot_invalid() || slot >= engine->expert_states.size()) {
            continue;
        }
        moe_block_state& state = engine->expert_states[slot];
        if (!state.active) continue;
        const moe_tier_t tier = state.tier;
        const double boost = tier == moe_TIER_VRAM ? 4.0 :
                            tier == moe_TIER_RAM ? 1.0 : 0.25;
        touch_expert_heat(engine, &state, boost);
    }

    moe_profile_log(
        "mlp_moe_router",
        "layer=%d selected=%zu weights=%zu logits=%zu internal_norm_scale=%d ms=%.3f",
        layer, s->selected_experts.size(), s->selected_weights.size(),
        s->router_logits.size(), router_has_internal_norm_scale,
        moe_profile_elapsed_ms(router_start_us));
    const uint32_t phase = engine->generation_phase.load(std::memory_order_relaxed);
    const uint64_t prefetch_start_us = moe_profile_now_us();
    if (phase != moe_GEN_PHASE_PREFILL) {
        const uint32_t depth = engine->adaptive_prefetch_depth.load(std::memory_order_relaxed);

        // InfiniGen speculative prefetch: use truncated hidden dot-product
        // with next layer's gate weights for ~85% accurate expert prediction.
        // This fires 200μs before the actual router runs, giving I/O a head start.
        // Fix 24: Stack-allocated arrays instead of heap vectors
        int infinigen_predicted_buf[256];
        int infinigen_predicted_count = 0;
        float spec_confidence = 0.0f;
        if (depth > 0u && layer + 1 < (int32_t)c->num_hidden_layers) {
            std::vector<int> infinigen_predicted_vec;
            moe_infinigen_speculative_prefetch(
                engine,
                layer + 1,
                c->experts_per_moe_layer,
                infinigen_predicted_vec,
                &spec_confidence
            );
            infinigen_predicted_count = (int)std::min(infinigen_predicted_vec.size(), (size_t)256);
            for (int ii = 0; ii < infinigen_predicted_count; ++ii)
                infinigen_predicted_buf[ii] = infinigen_predicted_vec[ii];
        }

        // Fallback: co-activation statistics if InfiniGen yields nothing
