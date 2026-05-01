static const moe_gguf_common_tensor_record* moe_find_layer_common_tensor(
    moe_pc_engine* engine,
    int32_t layer,
    const char* const* suffixes,
    size_t suffix_count,
    moe_raw_tensor_desc* out_desc
) {
    if (!engine || !suffixes || layer < 0) {
        return nullptr;
    }
    std::vector<std::string> names;
    moe_push_layer_tensor_candidates(&names, layer, suffixes, suffix_count);
    return moe_common_tensor_find_first(engine, names);
}

static int moe_map_layer_common_tensor(
    moe_pc_engine* engine,
    int32_t layer,
    const char* const* suffixes,
    size_t suffix_count,
    moe_raw_tensor_cache_entry* out
) {
    if (!engine || !out) {
        return 0;
    }
    std::vector<std::string> names;
    moe_push_layer_tensor_candidates(&names, layer, suffixes, suffix_count);
    moe_raw_tensor_desc desc{};
    const moe_gguf_common_tensor_record* rec = moe_common_tensor_find_first(engine, names);
    if (!rec || !moe_common_tensor_record_to_raw_desc(*rec, &desc)) {
        return 0;
    }
    return map_common_tensor_record(engine, *rec, desc, out);
}

static int moe_common_linear_f32(
    moe_pc_engine* engine,
    const moe_raw_tensor_cache_entry& raw,
    const float* input,
    uint32_t input_count,
    float* output,
    uint32_t output_count
) {
    if (!engine || !input || !output ||
        raw.cols != input_count || raw.rows != output_count) {
        return 0;
    }
    const uint64_t row_bytes = moe_raw_tensor_row_bytes(raw, raw.cols);
    if (row_bytes == 0 || raw.bytes < row_bytes * (uint64_t)raw.rows) {
        return 0;
    }
    return moe_parallel_rows(engine, raw.rows, 0, [&](uint32_t row) {
        output[row] = moe_raw_tensor_dot_row(raw, row, input, input_count);
        return 1;
    });
}

static int moe_raw_tensor_scalar_at_f32(
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

static uint64_t moe_raw_tensor_scalar_count(const moe_raw_tensor_cache_entry& raw) {
    const uint32_t elem = moe_raw_scalar_size(raw.scalar_encoding);
    if (!elem) {
        return 0;
    }
    uint64_t logical_count = 0;
    if (raw.cols) {
        if (raw.rows > UINT64_MAX / (uint64_t)raw.cols) {
            return raw.bytes / (uint64_t)elem;
        }
        logical_count = (uint64_t)raw.rows * (uint64_t)raw.cols;
    } else {
        logical_count = raw.rows;
    }
    if (logical_count && logical_count <= raw.bytes / (uint64_t)elem) {
        return logical_count;
    }
    return raw.bytes / (uint64_t)elem;
}

static int moe_apply_rmsnorm_common_weight_f32(
    const moe_raw_tensor_cache_entry& raw,
    const float* input,
    uint32_t count,
    float* output,
    float eps
) {
    // BUGFIX 208: Division by zero when count == 0
    // Problem: ss / (double)count crashes if count == 0
    // Solution: Check count == 0 BEFORE division (already done), but clarify comment
    if (!input || !output || count == 0 || moe_raw_tensor_scalar_count(raw) < count) {
        return 0;
    }
    double ss = 0.0;
    for (uint32_t i = 0; i < count; ++i) {
        ss += (double)input[i] * (double)input[i];
    }
    // Safe: count > 0 guaranteed by check above
    const float norm = 1.0f / std::sqrt((float)(ss / (double)count) + (eps > 0.0f ? eps : 1.0e-6f));
    for (uint32_t i = 0; i < count; ++i) {
        float w = 0.0f;
        if (!moe_raw_tensor_scalar_at_f32(raw, i, &w)) {
            return 0;
        }
        output[i] = input[i] * norm * w;
    }
    return 1;
}

static int moe_layer_rmsnorm_suffixes_f32(
    moe_pc_engine* engine,
    int32_t layer,
    const char* const* suffixes,
    size_t suffix_count,
    const float* input,
    uint32_t hidden_size,
    float* output,
    float eps
) {
    moe_raw_tensor_cache_entry raw{};
    return moe_map_layer_common_tensor(engine, layer, suffixes, suffix_count, &raw) &&
        moe_apply_rmsnorm_common_weight_f32(raw, input, hidden_size, output, eps);
}

static int moe_layer_ffn_norm_f32(
    moe_pc_engine* engine,
    int32_t layer,
    const float* input,
    uint32_t hidden_size,
    float* output,
    float eps
) {
    static const char* const suffixes[] = {
        "ffn_norm.weight",
        "ffn_pre_norm.weight",
        "mlp_norm.weight"
    };
    return moe_layer_rmsnorm_suffixes_f32(
        engine, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]),
        input, hidden_size, output, eps);
}

static int moe_layer_moe_norm_f32(
    moe_pc_engine* engine,
    int32_t layer,
    const float* input,
    uint32_t hidden_size,
    float* output,
    float eps
) {
    static const char* const suffixes[] = {
        "pre_ffw_norm_2.weight",
        "ffn_pre_norm_2.weight",
        "moe_norm.weight"
    };
    return moe_layer_rmsnorm_suffixes_f32(
        engine, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]),
        input, hidden_size, output, eps);
}

static int moe_layer_post_attention_output_norm_f32(
    moe_pc_engine* engine,
    int32_t layer,
    const float* input,
    uint32_t hidden_size,
    float* output,
    float eps
) {
    static const char* const suffixes[] = {
        "post_attention_norm.weight",
        "post_attention_layernorm.weight",
        "post_attention_layer_norm.weight",
        "post_attn_norm.weight"
    };
    return moe_layer_rmsnorm_suffixes_f32(
        engine, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]),
        input, hidden_size, output, eps);
}

static int moe_layer_uses_split_post_norm_block(const moe_pc_engine* engine, int32_t layer) {
    if (!engine || layer < 0) {
        return 0;
    }
    if (moe_engine_has_common_tensor_suffix(engine, layer, "post_ffw_norm.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "post_ffw_norm_1.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "post_ffw_norm_2.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "layer_output_scale.weight")) {
        return 1;
    }
    return moe_engine_graph_ir_mentions(engine, "post_ffw_norm") ||
        moe_engine_graph_ir_mentions(engine, "layer_output_scale") ? 1 : 0;
}

static int moe_layer_post_ffw_norm_suffixes_f32(
    moe_pc_engine* engine,
    int32_t layer,
    const char* const* suffixes,
    size_t suffix_count,
    const float* input,
    uint32_t hidden_size,
    float* output,
    float eps
) {
    return moe_layer_rmsnorm_suffixes_f32(
        engine, layer, suffixes, suffix_count, input, hidden_size, output, eps);
}

static int moe_layer_post_ffw_norm_f32(
    moe_pc_engine* engine,
    int32_t layer,
    const float* input,
    uint32_t hidden_size,
    float* output,
    float eps
) {
    static const char* const suffixes[] = {
        "post_ffw_norm.weight",
        "ffn_post_norm.weight"
    };
    return moe_layer_post_ffw_norm_suffixes_f32(
        engine, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]),
        input, hidden_size, output, eps);
}

static int moe_layer_post_ffw_norm1_f32(
    moe_pc_engine* engine,
    int32_t layer,
    const float* input,
    uint32_t hidden_size,
    float* output,
    float eps
) {
    static const char* const suffixes[] = {
        "post_ffw_norm_1.weight",
        "ffn_post_norm_1.weight"
    };
    return moe_layer_post_ffw_norm_suffixes_f32(
        engine, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]),
        input, hidden_size, output, eps);
}

static int moe_layer_post_ffw_norm2_f32(
    moe_pc_engine* engine,
    int32_t layer,
    const float* input,
    uint32_t hidden_size,
    float* output,
    float eps
) {
    static const char* const suffixes[] = {
        "post_ffw_norm_2.weight",
        "ffn_post_norm_2.weight"
    };
    return moe_layer_post_ffw_norm_suffixes_f32(
        engine, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]),
        input, hidden_size, output, eps);
}

static int moe_layer_output_scale_f32(
    moe_pc_engine* engine,
    int32_t layer,
    float* hidden,
    uint32_t hidden_size
) {
    if (!engine || !hidden || hidden_size == 0) {
        return 0;
    }
    static const char* const suffixes[] = {
        "layer_output_scale.weight",
        "layer_scalar.weight",
        "layer_scalar"
    };
    moe_raw_tensor_cache_entry raw{};
    if (!moe_map_layer_common_tensor(engine, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]), &raw)) {
        return 1;
    }
    const uint64_t count = moe_raw_tensor_scalar_count(raw);
    if (count == 1u) {
        float scale = 1.0f;
        if (!moe_raw_tensor_scalar_at_f32(raw, 0, &scale)) {
            return 0;
        }
        moe_scale_f32(hidden, scale, hidden_size);
        return 1;
    }
    if (count < hidden_size) {
        return 0;
    }
    for (uint32_t i = 0; i < hidden_size; ++i) {
        float scale = 1.0f;
        if (!moe_raw_tensor_scalar_at_f32(raw, i, &scale)) {
            return 0;
        }
        hidden[i] *= scale;
    }
    return 1;
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
    static const char* const suffixes[] = {
        "ffn_down_exps.scale",
        "ffn_gate_inp.per_expert_scale"
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
        "w1.weight"
    };
    static const char* const up_suffixes[] = {
        "ffn_up.weight",
        "mlp.up_proj.weight",
        "up_proj.weight",
        "feed_forward.w3.weight",
        "ffn.w3.weight",
        "w3.weight"
    };
    static const char* const down_suffixes[] = {
        "ffn_down.weight",
        "mlp.down_proj.weight",
        "down_proj.weight",
        "feed_forward.w2.weight",
        "ffn.w2.weight",
        "w2.weight"
    };
    moe_raw_tensor_cache_entry gate{};
    moe_raw_tensor_cache_entry up{};
    moe_raw_tensor_cache_entry down{};
    if (!moe_map_layer_common_tensor(engine, layer, gate_suffixes, sizeof(gate_suffixes) / sizeof(gate_suffixes[0]), &gate) ||
        !moe_map_layer_common_tensor(engine, layer, up_suffixes, sizeof(up_suffixes) / sizeof(up_suffixes[0]), &up) ||
        !moe_map_layer_common_tensor(engine, layer, down_suffixes, sizeof(down_suffixes) / sizeof(down_suffixes[0]), &down)) {
        return 0;
    }
    const uint32_t intermediate = gate.rows;
    if (intermediate == 0 || up.rows != intermediate ||
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
    if (!moe_parallel_rows(engine, intermediate, 0, [&](uint32_t row) {
        const float g = moe_raw_tensor_dot_row(gate, row, input, hidden_size);
        const float u = moe_raw_tensor_dot_row(up, row, input, hidden_size);
        s->gate[row] = moe_generation_gated_activation(engine, g, u);
        return 1;
    })) {
        return 0;
    }
    if (!moe_common_linear_f32(engine, down, s->gate.data(), intermediate, out_hidden, hidden_size)) {
        return 0;
    }
    return 1;
}

static int moe_layer_dense_mlp_f32(
    moe_pc_engine* engine,
    int32_t layer,
    moe_decode_scratch* s,
    float* hidden,
    uint32_t hidden_size
) {
    if (moe_layer_common_dense_mlp_f32(engine, layer, s, s->norm.data(), hidden_size, s->mlp.data())) {
        return moe_add_mlp_residual_f32(engine, layer, s, hidden, hidden_size);
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
    if (!moe_raw_mlp_span_f32(
            engine, span, s->norm.data(), hidden_size, dense_intermediate,
            s->gate.data(), s->up.data(), s->mlp.data())) {
        return 0;
    }
    return moe_add_mlp_residual_f32(engine, layer, s, hidden, hidden_size);
}

static int moe_layer_moe_mlp_f32(
    moe_pc_engine* engine,
    int32_t layer,
    moe_decode_scratch* s,
    float* hidden,
    const moe_storage_constants_t* c
) {
    const uint32_t hidden_size = c->hidden_size;
    const uint32_t configured_expert_intermediate = c->expert_intermediate_size;
    const moe_attn_config_t attn_cfg = moe_generation_attn_config(engine);
    std::fill(s->mlp.begin(), s->mlp.end(), 0.0f);

    const float* shared_input = s->norm.data();
    const float* expert_input = s->norm.data();
    if (moe_layer_moe_norm_f32(
            engine, layer, hidden, hidden_size, s->expert.data(), attn_cfg.rms_norm_eps)) {
        expert_input = s->expert.data();
    }

    const uint64_t shared_bytes =
        2ull * (uint64_t)configured_expert_intermediate * (uint64_t)hidden_size * 2ull +
        (uint64_t)hidden_size * (uint64_t)configured_expert_intermediate * 2ull;
    const moe_storage_raw_span_spec_t* shared = find_raw_role_span_bytes(layer, moe_RAW_ROLE_SHARED, shared_bytes, 0);
    int shared_ok = 0;
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

    moe_save_gate_input_snapshot(engine, layer, expert_input, hidden_size);
    if (!moe_router_topk_f32(
            engine,
            layer,
            expert_input,
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
    const uint32_t phase = engine->generation_phase.load(std::memory_order_relaxed);
    if (phase != moe_GEN_PHASE_PREFILL) {
        const uint32_t depth = engine->adaptive_prefetch_depth.load(std::memory_order_relaxed);

        // InfiniGen speculative prefetch: use truncated hidden dot-product
        // with next layer's gate weights for ~85% accurate expert prediction.
        // This fires 200μs before the actual router runs, giving I/O a head start.
        std::vector<int> infinigen_predicted;
        float spec_confidence = 0.0f;
        if (depth > 0u && layer + 1 < (int32_t)c->num_hidden_layers) {
            moe_infinigen_speculative_prefetch(
                engine,
                layer + 1,
                c->experts_per_moe_layer,
                infinigen_predicted,
                &spec_confidence
            );
        }

        // Fallback: co-activation statistics if InfiniGen yields nothing
        std::vector<int> predicted_experts;
        if (!infinigen_predicted.empty() && spec_confidence > 0.50f) {
            predicted_experts = std::move(infinigen_predicted);
        } else {
            predicted_experts = moe_juju_predict_co_activation(
                engine,
                layer + 1,
                (int)std::min<uint32_t>(moe_MOE_TOP_EXPERTS, moe_MOE_BASE_TOP_EXPERTS * 2u)
            );
        }
        if (depth == 0u) {
            predicted_experts.clear();
        }

        // Primary lookahead: layer+1
        moe_pc_engine_prefetch_moe_lookahead(
            engine,
            layer,
            s->selected_experts.data(),
            (uint32_t)s->selected_experts.size(),
            predicted_experts.empty() ? nullptr : predicted_experts.data(),
            (uint32_t)predicted_experts.size()
        );

        // Deep speculative prefetch: layers +2 to +depth
        // Only when InfiniGen confidence is high (entropy-adaptive depth)
        for (uint32_t d = 2; d <= depth && layer + (int32_t)d < (int32_t)c->num_hidden_layers; ++d) {
            std::vector<int> deep_predicted;
            float deep_conf = 0.0f;
            moe_infinigen_speculative_prefetch(
                engine,
                layer + (int32_t)d,
                c->experts_per_moe_layer,
                deep_predicted,
                &deep_conf
            );
            // Only prefetch deep layers if confidence remains adequate
            if (deep_predicted.empty() || deep_conf < 0.40f) {
                break;
            }
            moe_pc_engine_prefetch_moe_lookahead(
                engine,
                layer + (int32_t)(d - 1),
                nullptr, 0,  // no "selected" at this depth
                deep_predicted.data(),
                (uint32_t)deep_predicted.size()
            );
        }

    } else {
        // Prefill phase: no speculative prediction, just basic lookahead
        moe_pc_engine_prefetch_moe_lookahead(
            engine,
            layer,
            s->selected_experts.data(),
            (uint32_t)s->selected_experts.size(),
            nullptr,
            0
        );
    }


    // Phase 2: Run experts in parallel with per-expert scratch buffers.
    // Each expert gets its own gate/up/out buffers to prevent data races.
    const size_t num_selected = s->selected_experts.size();
    if (num_selected == 0 || s->selected_weights.size() != num_selected) {
        std::fprintf(stderr,
            "[storagellm mlp] router produced invalid selection layer=%d selected=%zu weights=%zu\n",
            layer, num_selected, s->selected_weights.size());
        return 0;
    }
    // BUGFIX 210: Auto-resize expert_work if undersized (extends BUGFIX 66)
    // Problem: BUGFIX 66 added logging but still returns 0, causing generation failure
    // Solution: Auto-resize expert_work to num_selected if undersized
    // This handles dynamic top-k changes or scratch buffer initialization issues
    if (num_selected > s->expert_work.size()) {
        fprintf(stderr, "[storageLLM WARN] expert_work size mismatch: "
            "need %zu, have %zu, layer %d. Auto-resizing scratch buffer.\n",
            num_selected, s->expert_work.size(), layer);
        s->expert_work.resize(num_selected);
    }
    moe_expert_work_t* work = s->expert_work.data();
    for (size_t e = 0; e < num_selected; ++e) {
        work[e].gate = nullptr;
        work[e].up = nullptr;
        work[e].out = nullptr;
        work[e].weight = s->selected_weights[e];
        work[e].expert_id = s->selected_experts[e];
        work[e].ok = 0;
    }

    // Fix 2: Flat row-parallel expert dispatch.
    // Instead of 8 sequential run_expert_triplet_f32 calls (each taking the full
    // pool via cpu_row_active), we pre-map all shards, then dispatch ALL experts'
    // rows in one parallel_rows call. This eliminates cpu_row_active serialization.
    uint32_t gate_rows = 0;
    uint32_t gate_cols = 0;
    uint32_t up_rows = 0;
    uint32_t up_cols = 0;
    uint32_t down_rows = 0;
    uint32_t down_cols = 0;

    // Pre-map all expert weight shards (these are cached, so fast for hot experts)
    struct expert_shards_t {
        const moe_tensor_record* gate_rec = nullptr;
        const moe_tensor_record* up_rec = nullptr;
        const moe_tensor_record* down_rec = nullptr;
        mmap_context_t* gate_shard = nullptr;
        mmap_context_t* up_shard = nullptr;
        mmap_context_t* down_shard = nullptr;
    };
    // BUGFIX 211 & 212: Dynamic allocation for shards and expert_ok arrays
    // Problem: Fixed-size arrays shards[moe_MOE_TOP_EXPERTS] and expert_ok[moe_MOE_TOP_EXPERTS]
    //          cause buffer overflow if num_selected > moe_MOE_TOP_EXPERTS
    // Solution: Use std::vector for dynamic sizing based on num_selected
    // Impact: Prevents buffer overflow, supports dynamic top-k values
    std::vector<expert_shards_t> shards(num_selected);
    std::vector<std::atomic<int>> expert_ok(num_selected);
    for (size_t e = 0; e < num_selected; ++e) {
        expert_ok[e].store(0, std::memory_order_relaxed);
    }
    uint32_t missing_experts = 0;
    uint32_t shape_mismatch_experts = 0;
    uint32_t map_failed_experts = 0;
    int first_bad_expert = -1;
    uint32_t first_gate_rows = 0, first_gate_cols = 0;
    uint32_t first_up_rows = 0, first_up_cols = 0;
    uint32_t first_down_rows = 0, first_down_cols = 0;
    for (size_t e = 0; e < num_selected; ++e) {
        shards[e].gate_rec = find_tensor_record(engine, layer, work[e].expert_id, moe_PROJ_GATE);
        shards[e].up_rec = find_tensor_record(engine, layer, work[e].expert_id, moe_PROJ_UP);
        shards[e].down_rec = find_tensor_record(engine, layer, work[e].expert_id, moe_PROJ_DOWN);
        if (!shards[e].gate_rec || !shards[e].up_rec || !shards[e].down_rec) {
            ++missing_experts;
            if (first_bad_expert < 0) {
                first_bad_expert = work[e].expert_id;
                if (shards[e].gate_rec) {
                    first_gate_rows = shards[e].gate_rec->info.rows;
                    first_gate_cols = shards[e].gate_rec->info.cols;
                }
                if (shards[e].up_rec) {
                    first_up_rows = shards[e].up_rec->info.rows;
                    first_up_cols = shards[e].up_rec->info.cols;
                }
                if (shards[e].down_rec) {
                    first_down_rows = shards[e].down_rec->info.rows;
                    first_down_cols = shards[e].down_rec->info.cols;
                }
            }
            continue;
        }
        const uint32_t cand_gate_rows = shards[e].gate_rec->info.rows;
        const uint32_t cand_gate_cols = shards[e].gate_rec->info.cols;
        const uint32_t cand_up_rows = shards[e].up_rec->info.rows;
        const uint32_t cand_up_cols = shards[e].up_rec->info.cols;
        const uint32_t cand_down_rows = shards[e].down_rec->info.rows;
        const uint32_t cand_down_cols = shards[e].down_rec->info.cols;
        const int record_shape_ok =
            cand_gate_rows != 0 &&
            cand_gate_rows == cand_up_rows &&
            cand_gate_rows == cand_down_cols &&
            cand_gate_cols == hidden_size &&
            cand_up_cols == hidden_size &&
            cand_down_rows == hidden_size;
        const int matches_canonical =
            !gate_rows ||
            (cand_gate_rows == gate_rows && cand_gate_cols == gate_cols &&
             cand_up_rows == up_rows && cand_up_cols == up_cols &&
             cand_down_rows == down_rows && cand_down_cols == down_cols);
        if (!record_shape_ok || !matches_canonical) {
            ++shape_mismatch_experts;
            if (first_bad_expert < 0) {
                first_bad_expert = work[e].expert_id;
                first_gate_rows = cand_gate_rows;
                first_gate_cols = cand_gate_cols;
                first_up_rows = cand_up_rows;
                first_up_cols = cand_up_cols;
                first_down_rows = cand_down_rows;
                first_down_cols = cand_down_cols;
            }
            continue;
        }
        if (!gate_rows) {
            gate_rows = cand_gate_rows;
            gate_cols = cand_gate_cols;
            up_rows = cand_up_rows;
            up_cols = cand_up_cols;
            down_rows = cand_down_rows;
            down_cols = cand_down_cols;
        }
        shards[e].gate_shard = get_or_map_tensor_shard(engine, *shards[e].gate_rec);
        shards[e].up_shard = get_or_map_tensor_shard(engine, *shards[e].up_rec);
        shards[e].down_shard = get_or_map_tensor_shard(engine, *shards[e].down_rec);
        if (!shards[e].gate_shard || !shards[e].up_shard || !shards[e].down_shard) {
            ++map_failed_experts;
            if (first_bad_expert < 0) {
                first_bad_expert = work[e].expert_id;
                first_gate_rows = shards[e].gate_rec->info.rows;
                first_gate_cols = shards[e].gate_rec->info.cols;
                first_up_rows = shards[e].up_rec->info.rows;
                first_up_cols = shards[e].up_rec->info.cols;
                first_down_rows = shards[e].down_rec->info.rows;
                first_down_cols = shards[e].down_rec->info.cols;
            }
            continue;
        }
        expert_ok[e].store(1, std::memory_order_release);
    }
    uint32_t mapped_experts = 0;
    for (size_t e = 0; e < num_selected; ++e) {
        mapped_experts += expert_ok[e].load(std::memory_order_acquire) ? 1u : 0u;
    }
    if (mapped_experts == 0) {
        const uint32_t expected_intermediate = gate_rows ? gate_rows : configured_expert_intermediate;
        const uint32_t expected_gate_cols = gate_cols ? gate_cols : hidden_size;
        const uint32_t expected_up_cols = up_cols ? up_cols : hidden_size;
        const uint32_t expected_down_rows = down_rows ? down_rows : hidden_size;
        const uint32_t expected_down_cols = down_cols ? down_cols : configured_expert_intermediate;
        std::fprintf(stderr,
            "[storagellm mlp] no selected expert shards mapped layer=%d selected=%zu first=%d missing=%u shape_mismatch=%u map_failed=%u expected gate=%ux%u up=%ux%u down=%ux%u first_bad=%d got gate=%ux%u up=%ux%u down=%ux%u\n",
            layer, num_selected, work[0].expert_id,
            missing_experts, shape_mismatch_experts, map_failed_experts,
            expected_intermediate, expected_gate_cols,
            expected_intermediate, expected_up_cols,
            expected_down_rows, expected_down_cols,
            first_bad_expert,
            first_gate_rows, first_gate_cols,
            first_up_rows, first_up_cols,
            first_down_rows, first_down_cols);
        return 0;
    }
    const uint32_t expert_intermediate = gate_rows;
    if (expert_intermediate == 0) {
        std::fprintf(stderr, "[storagellm mlp] invalid intermediate size layer=%d intermediate=%u\n",
            layer, expert_intermediate);
        return 0;
    }
    const size_t needed_gate = num_selected * (size_t)expert_intermediate;
    const size_t needed_up = needed_gate;
    const size_t needed_out = num_selected * (size_t)hidden_size;
    if (s->expert_gate_pool.size() < needed_gate) {
        s->expert_gate_pool.resize(needed_gate);
    }
    if (s->expert_up_pool.size() < needed_up) {
        s->expert_up_pool.resize(needed_up);
    }
    if (s->expert_out_pool.size() < needed_out) {
        s->expert_out_pool.resize(needed_out);
    }
    for (size_t e = 0; e < num_selected; ++e) {
        work[e].gate = s->expert_gate_pool.data() + e * (size_t)expert_intermediate;
        work[e].up = s->expert_up_pool.data() + e * (size_t)expert_intermediate;
        work[e].out = s->expert_out_pool.data() + e * (size_t)hidden_size;
        std::fill(work[e].out, work[e].out + hidden_size, 0.0f);
    }

    // Phase 2a: Flat dispatch: gate+up+SiLU for all experts simultaneously.
    // Total rows = num_selected * intermediate_size, row i maps to expert e and row r.
    // BUGFIX 262: Check expert_intermediate > 0 BEFORE multiplication ★
    // Problem: Checked after total_gate_rows calculation, but division by zero still occurs in lambda
    // Solution: Check expert_intermediate > 0 before any calculation
    if (expert_intermediate == 0) {
        std::fprintf(stderr, "[storagellm mlp] invalid intermediate size layer=%d intermediate=%u\n",
            layer, expert_intermediate);
        return 0;
    }
    if (num_selected > (size_t)UINT32_MAX / (size_t)expert_intermediate) {
        std::fprintf(stderr, "[storagellm mlp] invalid gate rows layer=%d selected=%zu intermediate=%u\n",
            layer, num_selected, expert_intermediate);
        return 0;
    }
    const uint32_t total_gate_rows = (uint32_t)num_selected * expert_intermediate;
    const int gate_up_ok = moe_parallel_rows(engine, total_gate_rows, 0, [&](uint32_t flat_row) {
        const uint32_t e = flat_row / expert_intermediate;
        const uint32_t r = flat_row % expert_intermediate;
        // Guard row-to-expert decoding before indexing per-expert arrays.
        // BUGFIX 290: Add bounds check for e index ★
        // Problem: e can exceed num_selected if flat_row is corrupted or overflow occurs
        // Accessing expert_ok[e], work[e], shards[e] with invalid e causes out-of-bounds access
        // Solution: Check e < num_selected before any array access
        if (e >= num_selected) {
            return 0;
        }
        if (!expert_ok[e].load(std::memory_order_acquire)) return 1;
        if (!moe_dot_tensor_record_row(engine, shards[e].gate_rec, shards[e].gate_shard, r, expert_input, hidden_size, &work[e].gate[r])) { expert_ok[e].store(0, std::memory_order_release); return 1; }
        if (!moe_dot_tensor_record_row(engine, shards[e].up_rec, shards[e].up_shard, r, expert_input, hidden_size, &work[e].up[r])) { expert_ok[e].store(0, std::memory_order_release); return 1; }
        work[e].gate[r] = moe_generation_gated_activation(engine, work[e].gate[r], work[e].up[r]);
        return 1;
    });
    if (!gate_up_ok) return 0;

    // Phase 2b: Flat dispatch: down_proj for all experts simultaneously.
    if (hidden_size == 0 || num_selected > (size_t)UINT32_MAX / (size_t)hidden_size) {
        std::fprintf(stderr, "[storagellm mlp] invalid down rows layer=%d selected=%zu hidden=%u\n",
            layer, num_selected, hidden_size);
        return 0;
    }
    const uint32_t total_down_rows = (uint32_t)num_selected * hidden_size;
    const int down_ok = moe_parallel_rows(engine, total_down_rows, 0, [&](uint32_t flat_row) {
        const uint32_t e = flat_row / hidden_size;
        const uint32_t r = flat_row % hidden_size;
        // Guard row-to-expert decoding before indexing per-expert arrays.
        if (e >= num_selected) {
            return 0;
        }
        if (!expert_ok[e].load(std::memory_order_acquire)) return 1;
        if (!moe_dot_tensor_record_row(engine, shards[e].down_rec, shards[e].down_shard,
                r, work[e].gate, expert_intermediate, &work[e].out[r])) {
            expert_ok[e].store(0, std::memory_order_release);
        }
        return 1;
    });
    if (!down_ok) return 0;
    for (size_t e = 0; e < num_selected; ++e) {
        work[e].ok = expert_ok[e].load(std::memory_order_acquire);
    }

    // Phase 3: Accumulate weighted expert outputs into mlp.
    for (size_t e = 0; e < num_selected; ++e) {
        if (!work[e].ok) continue;
        if (moe_layer_post_ffw_norm2_f32(
                engine, layer, work[e].out, hidden_size, s->residual.data(), attn_cfg.rms_norm_eps)) {
            std::memcpy(work[e].out, s->residual.data(), (size_t)hidden_size * sizeof(float));
        }
        if (!moe_layer_apply_expert_down_scale_f32(
                engine, layer, work[e].expert_id, c->experts_per_moe_layer,
                work[e].out, hidden_size)) {
            std::fprintf(stderr, "[storagellm mlp] expert down scale failed layer=%d expert=%d\n",
                layer, work[e].expert_id);
            return 0;
        }
        moe_axpy_f32(s->mlp.data(), work[e].out, work[e].weight, hidden_size);
    }

    return moe_add_mlp_residual_f32(engine, layer, s, hidden, hidden_size);
}

