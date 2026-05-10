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


static int moe_layer_get_expert_down_uniform_scale_f32(
    moe_pc_engine* engine,
    int32_t layer,
    int expert_id,
    uint32_t experts,
    float* out_scale
) {
    if (!out_scale) {
        return 0;
    }
    *out_scale = 1.0f;
    if (!engine || expert_id < 0 || experts == 0 || (uint32_t)expert_id >= experts) {
        return 0;
    }
    // This helper is deliberately stricter than moe_layer_apply_expert_down_scale_f32:
    // it only succeeds when the per-expert down scale is a single scalar.  That lets
    // the CPU-only MLP path fuse down_proj and expert accumulation without first
    // materializing a full [top_k, hidden] output buffer.  Per-channel scale tensors
    // still fall back to the fully general path for correctness.
    static const char* const suffixes[] = {
        "ffn_down_exps.scale",
        "ffn_gate_inp.per_expert_scale",
        "router.per_expert_scale",
        "mlp.router.per_expert_scale",
        "moe.gate.per_expert_scale"
    };
    moe_raw_tensor_cache_entry raw{};
    if (!moe_map_layer_common_tensor(engine, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]), &raw)) {
        *out_scale = 1.0f;
        return 1;
    }
    const uint64_t count = moe_raw_tensor_scalar_count(raw);
    float scale = 1.0f;
    if (count == 1u) {
        if (!moe_raw_tensor_scalar_at_f32(raw, 0, &scale) || !std::isfinite(scale)) return 0;
        *out_scale = scale;
        return 1;
    }
    if (count == experts && (uint32_t)expert_id < experts) {
        if (!moe_raw_tensor_scalar_at_f32(raw, (uint64_t)expert_id, &scale) || !std::isfinite(scale)) return 0;
        *out_scale = scale;
        return 1;
    }
    // Vector/per-channel down scales require materializing each expert output before
    // scaling, so the fused accumulator path must not be used.
    return 0;
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
    // BUGFIX 842: Check expert_id bounds ★★
    if ((uint32_t)expert_id >= experts) {
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
        // BUGFIX 843: Check scale for NaN/Inf ★★★
        if (!std::isfinite(scale)) return 0;
        moe_scale_f32(values, scale, hidden_size);
        return 1;
    }
    if (experts && count == experts && (uint32_t)expert_id < experts) {
        float scale = 1.0f;
        if (!moe_raw_tensor_scalar_at_f32(raw, (uint64_t)expert_id, &scale)) return 0;
        // BUGFIX 844: Check scale for NaN/Inf ★★★
        if (!std::isfinite(scale)) return 0;
        moe_scale_f32(values, scale, hidden_size);
        return 1;
    }
    const uint64_t base = (uint64_t)expert_id * (uint64_t)hidden_size;
    // BUGFIX 845: Check base overflow ★★
    if (base > UINT64_MAX - hidden_size) return 0;

    if (base <= count && (uint64_t)hidden_size <= count - base) {
        for (uint32_t i = 0; i < hidden_size; ++i) {
            float scale = 1.0f;
            if (!moe_raw_tensor_scalar_at_f32(raw, base + i, &scale)) return 0;
            // BUGFIX 846: Check scale for NaN/Inf ★★★
            if (!std::isfinite(scale)) return 0;
            values[i] *= scale;
        }
        return 1;
    }
    std::fprintf(stderr,
        "[storagellm mlp] expert down scale tensor has invalid shape layer=%d expert=%d count=%llu experts=%u hidden=%u\n",
        layer, expert_id, (unsigned long long)count, experts, hidden_size);
    return 0;
}

static int moe_add_mlp_residual_f32(
    moe_pc_engine* engine,
    int32_t layer,
    moe_decode_scratch* s,
    float* hidden,
    uint32_t hidden_size
) {
    // BUGFIX 716: MLP residual addition NaN handling ★★
    // Problem: No validation of residual values before addition
    // Impact: NaN in MLP output propagates to hidden state
    // Solution: Check for NaN in residual values
    if (!engine || !s || !hidden) {
        return 0;
    }
    const moe_attn_config_t cfg = moe_generation_attn_config(engine);
    const int post_ffw_norm_required = moe_layer_has_post_ffw_norm_weight(engine, layer);
    if (moe_layer_post_ffw_norm_f32(
            engine, layer, s->mlp.data(), hidden_size, s->shared.data(), cfg.rms_norm_eps)) {
        for (uint32_t i = 0; i < hidden_size; ++i) {
            if (!std::isfinite(s->shared[i])) {
                std::fprintf(stderr,
                    "[storagellm mlp] non-finite post-ffw norm output layer=%d index=%u\n",
                    layer, i);
                return 0;
            }
            hidden[i] += s->shared[i];
        }
        return 1;
    }
    if (post_ffw_norm_required) {
        std::fprintf(stderr,
            "[storagellm mlp] post-ffw norm tensor is required but could not be mapped layer=%d\n",
            layer);
        return 0;
    }
    for (uint32_t i = 0; i < hidden_size; ++i) {
        if (!std::isfinite(s->mlp[i])) {
            std::fprintf(stderr,
                "[storagellm mlp] non-finite mlp output layer=%d index=%u\n",
                layer, i);
            return 0;
        }
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
        "layer=%d intermediate=%u hidden=%u activation_mode=%d gate_up_ms=%.3f down_ms=%.3f total_ms=%.3f",
        layer, intermediate, hidden_size, activation_mode, gate_up_ms,
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
    if (!span && (!engine || !engine->offload_graph_ir_ready.load(std::memory_order_relaxed))) {
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
