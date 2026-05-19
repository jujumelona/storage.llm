// MLP normalization operations

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
        "ffn_gate_inp_shexp.weight",
        "shared_expert_gate.weight",
        "shared_expert.gate.weight",
        "shared_expert.gate_inp.weight",
        "shared_expert_gate_proj.weight",
        "ffn_shared_gate.weight",
        "ffn_shared_expert_gate_inp.weight",
        "shared_gate.weight"
    };
    moe_raw_tensor_cache_entry raw{};
    if (!moe_map_layer_common_tensor(engine, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]), &raw)) {
        return 1;
    }
    float logit = 0.0f;
    if (raw.cols == hidden_size && raw.rows > 0) {
        logit = moe_raw_tensor_dot_row(raw, 0, input, hidden_size);
    } else {
        const uint64_t scalar_count = moe_raw_tensor_scalar_count(raw);
        if (scalar_count != hidden_size) {
            return 0;
        }
        double acc = 0.0;
        for (uint32_t i = 0; i < hidden_size; ++i) {
            float w = 0.0f;
            if (!moe_raw_tensor_scalar_at_f32(raw, i, &w) || !std::isfinite(w)) {
                return 0;
            }
            acc += (double)w * (double)input[i];
        }
        logit = (float)acc;
    }
    if (!std::isfinite(logit)) {
        return 0;
    }
    // BUGFIX 209: Numerically stable sigmoid computation
    // Problem: logit < 0 branch uses exp(logit) / (1 + exp(logit)) which underflows for large negative logit
    // Solution: Use log-sum-exp trick: sigmoid(x) = 1 / (1 + exp(-x)) for all x
    // For x < 0: 1 / (1 + exp(-x)) = exp(x) / (exp(x) + 1) but direct formula is more stable
    if (logit >= 0.0f) {
        *out_gate = 1.0f / (1.0f + std::exp(-logit));
    } else {
        const float e = std::exp(logit);
        *out_gate = e / (1.0f + e);
    }
    return std::isfinite(*out_gate) ? 1 : 0;
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
        "ffn_gate_inp.per_expert_scale",
        "router.per_expert_scale",
        "mlp.router.per_expert_scale",
        "moe.gate.per_expert_scale"
    };
    static const char* const direct_contract_suffixes[] = {
        "per_expert_scale",
        "moe.per_expert_scale",
        "mlp.moe.per_expert_scale",
        "experts.per_expert_scale",
        "ffn_gate_inp.per_expert_scale",
        "router.per_expert_scale",
        "mlp.router.per_expert_scale",
        "moe.gate.per_expert_scale"
    };
    const int direct_scale_contract = moe_engine_contract_uses_direct_per_expert_scale(engine, layer);
    // Do not bind ffn_down_exps.scale here. In the current JUJU contract it is
    // listed as a routed expert tensor, but applying it as an expert-output
    // multiplier regresses chat PPL. Runtime expert scales require explicit
    // per_expert_scale-style names.
    const char* const* scale_suffixes = direct_scale_contract ? direct_contract_suffixes : suffixes;
    const size_t scale_suffix_count = direct_scale_contract ?
        (sizeof(direct_contract_suffixes) / sizeof(direct_contract_suffixes[0])) :
        (sizeof(suffixes) / sizeof(suffixes[0]));
    moe_raw_tensor_cache_entry raw{};
    if (!moe_map_layer_common_tensor(engine, layer, scale_suffixes, scale_suffix_count, &raw)) {
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
        "ffn_gate_inp.per_expert_scale",
        "router.per_expert_scale",
        "mlp.router.per_expert_scale",
        "moe.gate.per_expert_scale"
    };
    static const char* const direct_contract_suffixes[] = {
        "per_expert_scale",
        "moe.per_expert_scale",
        "mlp.moe.per_expert_scale",
        "experts.per_expert_scale",
        "ffn_gate_inp.per_expert_scale",
        "router.per_expert_scale",
        "mlp.router.per_expert_scale",
        "moe.gate.per_expert_scale"
    };
    const int direct_scale_contract = moe_engine_contract_uses_direct_per_expert_scale(engine, layer);
    // Do not bind ffn_down_exps.scale here. In the current JUJU contract it is
    // listed as a routed expert tensor, but applying it as an expert-output
    // multiplier regresses chat PPL. Runtime expert scales require explicit
    // per_expert_scale-style names.
    const char* const* scale_suffixes = direct_scale_contract ? direct_contract_suffixes : suffixes;
    const size_t scale_suffix_count = direct_scale_contract ?
        (sizeof(direct_contract_suffixes) / sizeof(direct_contract_suffixes[0])) :
        (sizeof(suffixes) / sizeof(suffixes[0]));
    moe_raw_tensor_cache_entry raw{};
    if (!moe_map_layer_common_tensor(engine, layer, scale_suffixes, scale_suffix_count, &raw)) {
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
    // Keep this helper branch-local: it forms the FFN/MoE branch, applies
    // optional post-FFW normalization to that branch when present, then adds it
    // to the residual stream. Layer-level output/cvec transforms are handled by
    // the layer plan after residual addition.
    if (!engine || !s || !hidden) {
        return 0;
    }
    const moe_attn_config_t cfg = moe_generation_attn_config(engine);
    int combined_post_norm_contract = 0;
    const char* combined_post_norm_source = nullptr;
    const int has_combined_post_norm_contract = moe_graph_ir_layer_contract_bool_key_f32(
        engine,
        layer,
        "mlp",
        "post_ffw_norm_applies_to_combined_ffn_before_residual",
        &combined_post_norm_contract,
        &combined_post_norm_source);
    const int post_ffw_norm_required =
        moe_layer_has_post_ffw_norm_weight(engine, layer) ||
        (has_combined_post_norm_contract && combined_post_norm_contract);
    float* branch = s->mlp.data();
    const char* branch_source = "mlp";
    if (has_combined_post_norm_contract) {
        moe_profile_log(
            "mlp_post_ffw_norm_contract",
            "layer=%d combined_before_residual=%d source=%s",
            layer,
            combined_post_norm_contract ? 1 : 0,
            combined_post_norm_source ? combined_post_norm_source : "layer_execution_contract_table");
    }
    if (moe_layer_post_ffw_norm_f32(
            engine, layer, s->mlp.data(), hidden_size, s->shared.data(), cfg.rms_norm_eps)) {
        branch = s->shared.data();
        branch_source = "post_ffw_norm";
    } else if (post_ffw_norm_required) {
        std::fprintf(stderr,
            "[storagellm mlp] post-ffw norm tensor is required but could not be mapped layer=%d\n",
            layer);
        return 0;
    }
    for (uint32_t i = 0; i < hidden_size; ++i) {
        if (!std::isfinite(branch[i])) {
            std::fprintf(stderr,
                "[storagellm mlp] non-finite mlp branch layer=%d index=%u source=%s\n",
                layer, i, branch_source);
            return 0;
        }
        hidden[i] += branch[i];
    }
    return 1;
}

static int moe_layer_dense_mlp_from_suffixes_f32(
    moe_pc_engine* engine,
    int32_t layer,
    moe_decode_scratch* s,
    const float* input,
    uint32_t hidden_size,
    float* out_hidden,
    const char* const* gate_suffixes,
    size_t gate_suffix_count,
    const char* const* up_suffixes,
    size_t up_suffix_count,
    const char* const* down_suffixes,
    size_t down_suffix_count,
    const char* profile_tag,
    const char* const* gate_roles = nullptr,
    size_t gate_role_count = 0,
    const char* const* up_roles = nullptr,
    size_t up_role_count = 0,
    const char* const* down_roles = nullptr,
    size_t down_role_count = 0,
    int allow_graph_suffix_fallback = 0
) {
    const uint64_t total_start_us = moe_profile_now_us();
    const uint64_t map_start_us = moe_profile_now_us();
    moe_profile_log(profile_tag ? profile_tag : "mlp_dense_begin", "layer=%d hidden=%u", layer, hidden_size);
    moe_raw_tensor_cache_entry gate{};
    moe_raw_tensor_cache_entry up{};
    moe_raw_tensor_cache_entry down{};
    if (!gate_suffixes || !up_suffixes || !down_suffixes) {
        return 0;
    }
    int gate_declared = 0;
    int up_declared = 0;
    int down_declared = 0;
    const int graph_required = moe_graph_ir_math_required(engine);
    int gate_mapped = moe_map_layer_tensor_by_roles_or_suffixes(
        engine, layer, gate_roles, gate_role_count, gate_suffixes, gate_suffix_count,
        &gate, &gate_declared);
    int up_mapped = moe_map_layer_tensor_by_roles_or_suffixes(
        engine, layer, up_roles, up_role_count, up_suffixes, up_suffix_count,
        &up, &up_declared);
    int down_mapped = moe_map_layer_tensor_by_roles_or_suffixes(
        engine, layer, down_roles, down_role_count, down_suffixes, down_suffix_count,
        &down, &down_declared);
    if (graph_required && allow_graph_suffix_fallback) {
        // Shared-expert tensors are safe to bind by explicit shared suffixes even
        // when GraphIR op_role metadata is incomplete.  This keeps generic FFN
        // fallback forbidden while matching explicit shared-expert suffix
        // families such as ffn_*_shexp.
        if (!gate_mapped) {
            gate_mapped = moe_map_layer_common_tensor(
                engine, layer, gate_suffixes, gate_suffix_count, &gate);
        }
        if (!up_mapped) {
            up_mapped = moe_map_layer_common_tensor(
                engine, layer, up_suffixes, up_suffix_count, &up);
        }
        if (!down_mapped) {
            down_mapped = moe_map_layer_common_tensor(
                engine, layer, down_suffixes, down_suffix_count, &down);
        }
    }
    if (!gate_mapped || !up_mapped || !down_mapped) {
        if (graph_required && (gate_declared || up_declared || down_declared)) {
            std::fprintf(stderr,
                "[storagellm mlp] graph-declared dense tensor bind failed layer=%d tag=%s gate=%d/%d up=%d/%d down=%d/%d\n",
                layer, profile_tag ? profile_tag : "mlp_dense_begin",
                gate_mapped, gate_declared, up_mapped, up_declared, down_mapped, down_declared);
        }
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



static int moe_router_contract_scale_name_is_weight_sidecar_f32(const std::string& name) {
    const std::string lower = moe_lower_ascii_copy(name);
    static const char* const markers[] = {
        "ffn_gate_inp.scale",
        "ffn_gate_inp.scales",
        "ffn_gate_inp.weight.scale",
        "ffn_gate_inp.weight.scales",
        ".weight.scale",
        ".weight.scales",
        "weight_sidecar",
        "quant_sidecar",
        "router_weight_scale_sidecar",
        "mxfp"
    };
    for (const char* marker : markers) {
        if (lower.find(marker) != std::string::npos) {
            return 1;
        }
    }
    return 0;
}

static int moe_router_contract_name_is_activation_scale_f32(const std::string& name) {
    if (name.empty() || moe_router_contract_scale_name_is_weight_sidecar_f32(name)) {
        return 0;
    }
    const std::string lower = moe_lower_ascii_copy(name);
    static const char* const accepted_exact[] = {
        "router_input_scale",
        "router_input_scales",
        "router.input_scale",
        "router.input_scales",
        "mlp.router_input_scale",
        "mlp.router_input_scales",
        "moe.gate_input_scale",
        "moe.gate_input_scales",
        "router_norm.weight",
        "router.input_layernorm.weight",
        "router.input_norm.weight",
        "moe_router_norm.weight",
        "moe.gate_norm.weight",
        "mlp.router_norm.weight"
    };
    for (const char* accepted : accepted_exact) {
        if (lower == accepted) {
            return 1;
        }
    }
    static const char* const accepted_suffixes[] = {
        ".router.input_scale",
        ".router.input_scales",
        ".mlp.router_input_scale",
        ".mlp.router_input_scales",
        ".moe.gate_input_scale",
        ".moe.gate_input_scales",
        ".router_norm.weight",
        ".router.input_layernorm.weight",
        ".router.input_norm.weight",
        ".moe_router_norm.weight",
        ".moe.gate_norm.weight",
        ".mlp.router_norm.weight"
    };
    for (const char* suffix : accepted_suffixes) {
        if (lower.size() >= std::strlen(suffix) &&
            lower.compare(lower.size() - std::strlen(suffix), std::strlen(suffix), suffix) == 0) {
            return 1;
        }
    }
    // Generic router.scale/mlp.router.scale/moe.gate.scale remain ambiguous:
    // they may describe router projection scale or quantization metadata, so
    // they do not satisfy the raw-hidden/internal-scale rule by name.
    return 0;
}

static int moe_router_contract_name_list_has_activation_scale_f32(
    const std::vector<std::string>& names,
    int trust_contract_router_scale_names = 0
) {
    for (const std::string& name : names) {
        if (moe_router_contract_name_is_activation_scale_f32(name)) {
            return 1;
        }
        if (trust_contract_router_scale_names && !moe_router_contract_scale_name_is_weight_sidecar_f32(name)) {
            const std::string lower = moe_lower_ascii_copy(name);
            static const char* const contract_suffixes[] = {
                "router.scale",
                "mlp.router.scale",
                "moe.gate.scale"
            };
            for (const char* suffix : contract_suffixes) {
                const size_t n = std::strlen(suffix);
                if (lower == suffix ||
                    (lower.size() > n && lower.compare(lower.size() - n, n, suffix) == 0)) {
                    return 1;
                }
            }
        }
    }
    return 0;
}

static int moe_router_contract_object_has_activation_scale_f32(
    const std::string& doc,
    size_t obj_begin,
    size_t obj_end
) {
    static const char* const scale_keys[] = {
        "router_scale", "router_scales", "scale", "scales",
        "input_scale", "input_scales", "router_input_scale", "router_input_scales",
        "router_input_scale_tensors", "activation_scale", "activation_scales"
    };
    std::vector<std::string> names;
    for (const char* key : scale_keys) {
        names.clear();
        const int trust_generator_router_scale =
            key && (std::strcmp(key, "router_scale") == 0 ||
                    std::strcmp(key, "router_scales") == 0 ||
                    std::strcmp(key, "router_input_scale") == 0 ||
                    std::strcmp(key, "router_input_scales") == 0);
        if (moe_json_get_string_array_slice(doc, obj_begin, obj_end, key, &names) &&
            moe_router_contract_name_list_has_activation_scale_f32(names, trust_generator_router_scale)) {
            return 1;
        }
        std::string single_name;
        if (moe_json_get_string_slice(doc, obj_begin, obj_end, key, &single_name) &&
            moe_router_contract_name_list_has_activation_scale_f32(
                std::vector<std::string>{single_name}, trust_generator_router_scale)) {
            return 1;
        }
    }
    return 0;
}

static int moe_router_input_text_is_raw_hidden_f32(const std::string& text) {
    const std::string v = moe_lower_ascii_copy(text);
    if (v.empty()) {
        return 0;
    }
    if (v.find("raw_hidden") != std::string::npos ||
        v.find("raw_residual") != std::string::npos ||
        v == "hidden" ||
        v == "hidden_states" ||
        v == "residual" ||
        v == "layer_input" ||
        v == "block_input" ||
        v.find("residual_input") != std::string::npos) {
        return 1;
    }
    return 0;
}

static int moe_router_contract_object_has_raw_hidden_input_f32(
    const std::string& doc,
    size_t obj_begin,
    size_t obj_end
) {
    static const char* const string_keys[] = {
        "router_input_source", "input_source", "source", "input",
        "producer", "input_producer", "router_input", "from"
    };
    std::string value;
    for (const char* key : string_keys) {
        value.clear();
        if (moe_json_get_string_slice(doc, obj_begin, obj_end, key, &value) &&
            moe_router_input_text_is_raw_hidden_f32(value)) {
            return 1;
        }
    }
    static const char* const array_keys[] = {
        "inputs", "input_names", "source_inputs", "producers",
        "input_producers", "router_inputs"
    };
    std::vector<std::string> names;
    for (const char* key : array_keys) {
        names.clear();
        if (!moe_json_get_string_array_slice(doc, obj_begin, obj_end, key, &names)) {
            continue;
        }
        for (const std::string& name : names) {
            if (moe_router_input_text_is_raw_hidden_f32(name)) {
                return 1;
            }
        }
    }
    return 0;
}

static int moe_router_rule_uses_hidden_only_with_activation_scale_f32(
    const std::string& rule
) {
    const std::string lowered_rule = moe_lower_ascii_copy(rule);
    return
        lowered_rule.find("use_hidden_only_when_explicit_router_input_scale_present") != std::string::npos ||
        lowered_rule.find("hidden_only_when_explicit_router_input_scale_present") != std::string::npos ||
        lowered_rule.find("use_hidden_when_router_has_internal_scale") != std::string::npos ||
        lowered_rule.find("hidden_when_router_has_internal_scale") != std::string::npos ||
        lowered_rule.find("use_hidden_when_router_has_input_scale") != std::string::npos ||
        lowered_rule.find("hidden_when_router_has_input_scale") != std::string::npos;
}

static int moe_layer_router_contract_has_hidden_input_f32(
    const std::string& doc,
    size_t obj_begin,
    size_t obj_end
) {
    int raw_contract = 0;
    if (moe_json_get_bool_slice(doc, obj_begin, obj_end, "raw_residual_contract", &raw_contract) &&
        raw_contract) {
        return 1;
    }
    std::string rule;
    if (moe_json_get_string_slice(doc, obj_begin, obj_end, "rule", &rule)) {
        // This rule is conditional: hidden/raw residual input is valid only
        // when the router op has a real activation/input scale.  The op may
        // still list both hidden and expert_ffn_input as possible inputs, so
        // decide the conditional rule before scanning generic input names.
        if (moe_router_rule_uses_hidden_only_with_activation_scale_f32(rule)) {
            return moe_router_contract_object_has_activation_scale_f32(doc, obj_begin, obj_end) ? 1 : 0;
        }
    }
    if (moe_router_contract_object_has_raw_hidden_input_f32(doc, obj_begin, obj_end)) {
        return 1;
    }
    return 0;
}

static int moe_layer_execution_contract_router_uses_hidden_f32(
    const moe_pc_engine* engine,
    int32_t layer
) {
    if (!engine || layer < 0) {
        return 0;
    }
    for (uint32_t doc_i = 0; doc_i < 2u; ++doc_i) {
        const std::string* docp = moe_contract_doc_by_index_f32(engine, doc_i);
        if (!docp || docp->empty()) {
            continue;
        }
        const std::string& doc = *docp;
        std::vector<std::pair<size_t, size_t>> rows;
        if (!moe_contract_layer_rows_cached_f32(engine, doc_i, layer, &rows)) {
            continue;
        }
        for (const auto& row_pair : rows) {
            const size_t obj_begin = row_pair.first;
            const size_t obj_end = row_pair.second;
            if (obj_begin >= obj_end || obj_end > doc.size()) {
                continue;
            }
            if (moe_router_row_contract_implies_sidecar_rms_root_transform_f32(doc, obj_begin, obj_end)) {
                return 1;
            }
            size_t router_begin = obj_begin;
            size_t router_end = obj_end;
            const std::string row_slice = doc.substr(obj_begin, obj_end - obj_begin);
            if (moe_json_find_object_local(row_slice, "router", &router_begin, &router_end)) {
                router_begin += obj_begin;
                router_end += obj_begin;
            }
            if (moe_layer_router_contract_has_hidden_input_f32(doc, router_begin, router_end)) {
                return 1;
            }
            auto router_scope_uses_hidden = [&](size_t scope_begin, size_t scope_end) -> int {
                int uses_hidden = 0;
                if (moe_json_get_bool_slice(
                        doc, scope_begin, scope_end,
                        "router_uses_hidden_when_internal_scale_present", &uses_hidden) &&
                    uses_hidden) {
                    // The boolean is conditional and becomes true only with a real
                    // router input activation-scale/norm contract, not a weight
                    // quantization sidecar.
                    return moe_router_contract_object_has_activation_scale_f32(
                        doc, scope_begin, scope_end) ? 1 : 0;
                }
                return 0;
            };
            if (router_scope_uses_hidden(router_begin, router_end)) {
                return 1;
            }
            size_t semantic_begin = 0;
            size_t semantic_end = 0;
            if (moe_json_find_object_local(row_slice, "semantic_contract", &semantic_begin, &semantic_end) &&
                router_scope_uses_hidden(obj_begin + semantic_begin, obj_begin + semantic_end)) {
                return 1;
            }
        }
    }
    return 0;
}

static int moe_layer_ops_router_uses_hidden_f32(
    const moe_pc_engine* engine,
    int32_t layer
) {
    if (!engine || layer < 0) {
        return 0;
    }
    for (uint32_t doc_i = 0; doc_i < 2u; ++doc_i) {
        const std::string* docp = moe_contract_doc_by_index_f32(engine, doc_i);
        if (!docp || docp->empty()) {
            continue;
        }
        const std::string& doc = *docp;
        size_t ops_begin = 0;
        size_t ops_end = 0;
        if (!moe_router_layer_ops_object_slice_cached_f32(engine, doc_i, layer, &ops_begin, &ops_end)) {
            continue;
        }
        size_t op_cursor = ops_begin + 1u;
        while (op_cursor < ops_end) {
            size_t op_begin = 0;
            size_t op_end = 0;
            if (!moe_json_next_object_in_array(doc, &op_cursor, ops_end, &op_begin, &op_end)) {
                break;
            }
            std::string name;
            const int has_name = moe_json_get_string_slice(doc, op_begin, op_end, "name", &name);
            const std::string lowered_name = has_name ? moe_lower_ascii_copy(name) : std::string();
            if (lowered_name != "router_input" && lowered_name != "moe_router" && lowered_name != "router") {
                continue;
            }
            if (moe_layer_router_contract_has_hidden_input_f32(doc, op_begin, op_end)) {
                return 1;
            }
        }
    }
    return 0;
}

static int moe_layer_router_uses_raw_residual_input(
    const moe_pc_engine* engine,
    int32_t layer
) {
    if (!engine || layer < 0 || !moe_graph_ir_math_required(engine)) {
        return 0;
    }
    if (moe_layer_execution_contract_router_uses_hidden_f32(engine, layer)) {
        return 1;
    }
    return moe_layer_ops_router_uses_hidden_f32(engine, layer);
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
    static const char* const gate_roles[] = {
        "ffn_gate", "mlp_gate", "gate_projection", "ffn_gate_projection"
    };
    static const char* const up_roles[] = {
        "ffn_up", "mlp_up", "up_projection", "ffn_up_projection"
    };
    static const char* const down_roles[] = {
        "ffn_down", "mlp_down", "down_projection", "ffn_down_projection"
    };
    return moe_layer_dense_mlp_from_suffixes_f32(
        engine, layer, s, input, hidden_size, out_hidden,
        gate_suffixes, sizeof(gate_suffixes) / sizeof(gate_suffixes[0]),
        up_suffixes, sizeof(up_suffixes) / sizeof(up_suffixes[0]),
        down_suffixes, sizeof(down_suffixes) / sizeof(down_suffixes[0]),
        "mlp_dense_begin",
        gate_roles, sizeof(gate_roles) / sizeof(gate_roles[0]),
        up_roles, sizeof(up_roles) / sizeof(up_roles[0]),
        down_roles, sizeof(down_roles) / sizeof(down_roles[0]),
        1);
}


struct moe_shared_expert_graph_candidate_f32 {
    moe_raw_tensor_cache_entry raw{};
    std::string name;
    std::string role;
    uint64_t file_offset = 0;
    uint16_t execution_order = 0;
};

static int moe_shared_expert_text_is_shared_f32(
    const std::string& text
) {
    const std::string lower = moe_lower_ascii_copy(text);
    return lower.find("shared_expert") != std::string::npos ||
        lower.find("shared_experts") != std::string::npos ||
        lower.find("shared_exp") != std::string::npos ||
        lower.find("shared_exps") != std::string::npos ||
        lower.find("expert_shared") != std::string::npos ||
        lower.find("experts_shared") != std::string::npos ||
        lower.find("exps_shared") != std::string::npos ||
        lower.find("shexp") != std::string::npos ||
        lower.find("sh_expert") != std::string::npos ||
        lower.find("ffn_shared") != std::string::npos;
}

static int moe_shared_expert_role_is_shared_f32(
    const std::string& role
) {
    return role == "shared_expert_mlp" ||
        role == "shared_experts" ||
        role == "shared_expert" ||
        role == "shared_ffn" ||
        role == "shared_core";
}

static int moe_shared_expert_name_has_part_f32(
    const std::string& name,
    const char* part
) {
    if (!part || !part[0]) {
        return 0;
    }
    const std::string lower = moe_lower_ascii_copy(name);
    if (!moe_shared_expert_text_is_shared_f32(lower)) {
        return 0;
    }
    if (std::strcmp(part, "gate") == 0) {
        return lower.find("gate_proj") != std::string::npos ||
            lower.find("gate_inp") != std::string::npos ||
            lower.find("gate.weight") != std::string::npos ||
            lower.find("_gate_") != std::string::npos ||
            lower.find(".gate_") != std::string::npos ||
            lower.find("_gate.") != std::string::npos ||
            lower.find(".gate.") != std::string::npos ||
            lower.find("w1.weight") != std::string::npos ||
            lower.find("_w1_") != std::string::npos ||
            lower.find(".w1.") != std::string::npos;
    }
    if (std::strcmp(part, "up") == 0) {
        return lower.find("up_proj") != std::string::npos ||
            lower.find("up_inp") != std::string::npos ||
            lower.find("up.weight") != std::string::npos ||
            lower.find("_up_") != std::string::npos ||
            lower.find(".up_") != std::string::npos ||
            lower.find("_up.") != std::string::npos ||
            lower.find(".up.") != std::string::npos ||
            lower.find("w3.weight") != std::string::npos ||
            lower.find("_w3_") != std::string::npos ||
            lower.find(".w3.") != std::string::npos;
    }
    if (std::strcmp(part, "down") == 0) {
        return lower.find("down_proj") != std::string::npos ||
            lower.find("down_inp") != std::string::npos ||
            lower.find("down.weight") != std::string::npos ||
            lower.find("_down_") != std::string::npos ||
            lower.find(".down_") != std::string::npos ||
            lower.find("_down.") != std::string::npos ||
            lower.find(".down.") != std::string::npos ||
            lower.find("w2.weight") != std::string::npos ||
            lower.find("_w2_") != std::string::npos ||
            lower.find(".w2.") != std::string::npos;
    }
    return 0;
}

static void moe_shared_expert_sort_candidates_f32(
    std::vector<moe_shared_expert_graph_candidate_f32>* values
) {
    if (!values) {
        return;
    }
    std::stable_sort(values->begin(), values->end(), [](
        const moe_shared_expert_graph_candidate_f32& a,
        const moe_shared_expert_graph_candidate_f32& b) {
        if (a.execution_order != b.execution_order) {
            return a.execution_order < b.execution_order;
        }
        if (a.file_offset != b.file_offset) {
            return a.file_offset < b.file_offset;
        }
        return a.name < b.name;
    });
}

static int moe_shared_expert_append_candidate_f32(
    moe_pc_engine* engine,
    const moe_gguf_common_tensor_record& rec,
    const moe_raw_tensor_desc& desc,
    std::vector<moe_shared_expert_graph_candidate_f32>* out
) {
    if (!engine || !out) {
        return 0;
    }
    moe_raw_tensor_cache_entry raw{};
    if (!map_common_tensor_record(engine, rec, desc, &raw)) {
        return 0;
    }
    moe_shared_expert_graph_candidate_f32 cand{};
    cand.raw = raw;
    cand.name = rec.name;
    cand.role = !rec.op_role.empty() ? rec.op_role :
        (!rec.execution_op.empty() ? rec.execution_op :
         (!rec.bundle_member_role.empty() ? rec.bundle_member_role : rec.graph_role_text));
    cand.file_offset = rec.file_offset;
    cand.execution_order = rec.execution_order;
    out->push_back(cand);
    return 1;
}

static int moe_map_shared_expert_graph_role_f32(
    moe_pc_engine* engine,
    int32_t layer,
    uint32_t hidden_size,
    moe_raw_tensor_cache_entry* gate_out,
    moe_raw_tensor_cache_entry* up_out,
    moe_raw_tensor_cache_entry* down_out
) {
    if (!engine || layer < 0 || hidden_size == 0 || !gate_out || !up_out || !down_out) {
        return 0;
    }
    std::vector<moe_shared_expert_graph_candidate_f32> gate_named;
    std::vector<moe_shared_expert_graph_candidate_f32> up_named;
    std::vector<moe_shared_expert_graph_candidate_f32> down_named;
    std::vector<moe_shared_expert_graph_candidate_f32> gate_up_shape;
    std::vector<moe_shared_expert_graph_candidate_f32> down_shape;
    uint32_t scanned = 0;
    uint32_t role_shared = 0;
    uint32_t name_shared = 0;
    for (const moe_gguf_common_tensor_record& rec : engine->gguf_common_tensors) {
        if (!moe_common_tensor_layer_matches(rec, layer)) {
            continue;
        }
        const int shared_by_role =
            moe_shared_expert_role_is_shared_f32(rec.op_role) ||
            moe_shared_expert_role_is_shared_f32(rec.execution_op) ||
            moe_shared_expert_role_is_shared_f32(rec.bundle_member_role) ||
            moe_shared_expert_text_is_shared_f32(rec.graph_role_text) ||
            rec.graph_role == moe_JUJU_ROLE_SHARED_CORE;
        const int shared_by_name = moe_shared_expert_text_is_shared_f32(rec.name);
        if (!shared_by_role && !shared_by_name) {
            continue;
        }
        scanned++;
        role_shared += shared_by_role ? 1u : 0u;
        name_shared += shared_by_name ? 1u : 0u;
        moe_raw_tensor_desc desc{};
        if (!moe_common_tensor_record_to_raw_desc(rec, &desc) ||
            !moe_common_tensor_record_raw_compatible(rec, desc)) {
            continue;
        }
        const uint8_t fmt = moe_common_tensor_effective_weight_format(rec, desc);
        if (!moe_common_tensor_raw_weight_format_supported(fmt) || desc.rows == 0) {
            continue;
        }
        const int gateup_shape = desc.cols == hidden_size && desc.rows != hidden_size;
        const int down_shape_ok = desc.rows == hidden_size && desc.cols != 0 && desc.cols != hidden_size;
        if (!gateup_shape && !down_shape_ok) {
            continue;
        }
        const int is_gate = gateup_shape && moe_shared_expert_name_has_part_f32(rec.name, "gate");
        const int is_up = gateup_shape && moe_shared_expert_name_has_part_f32(rec.name, "up");
        const int is_down = down_shape_ok && moe_shared_expert_name_has_part_f32(rec.name, "down");
        if (is_gate) {
            moe_shared_expert_append_candidate_f32(engine, rec, desc, &gate_named);
        } else if (is_up) {
            moe_shared_expert_append_candidate_f32(engine, rec, desc, &up_named);
        } else if (is_down) {
            moe_shared_expert_append_candidate_f32(engine, rec, desc, &down_named);
        } else if (gateup_shape) {
            moe_shared_expert_append_candidate_f32(engine, rec, desc, &gate_up_shape);
        } else if (down_shape_ok) {
            moe_shared_expert_append_candidate_f32(engine, rec, desc, &down_shape);
        }
    }
    moe_shared_expert_sort_candidates_f32(&gate_named);
    moe_shared_expert_sort_candidates_f32(&up_named);
    moe_shared_expert_sort_candidates_f32(&down_named);
    moe_shared_expert_sort_candidates_f32(&gate_up_shape);
    moe_shared_expert_sort_candidates_f32(&down_shape);

    int have_gate = !gate_named.empty();
    int have_up = !up_named.empty();
    int have_down = !down_named.empty();
    moe_raw_tensor_cache_entry gate = have_gate ? gate_named[0].raw : moe_raw_tensor_cache_entry{};
    moe_raw_tensor_cache_entry up = have_up ? up_named[0].raw : moe_raw_tensor_cache_entry{};
    moe_raw_tensor_cache_entry down = have_down ? down_named[0].raw : moe_raw_tensor_cache_entry{};
    std::string gate_name = have_gate ? gate_named[0].name : std::string();
    std::string up_name = have_up ? up_named[0].name : std::string();
    std::string down_name = have_down ? down_named[0].name : std::string();

    if (!have_gate || !have_up) {
        std::vector<moe_shared_expert_graph_candidate_f32> remaining;
        remaining.reserve(gate_up_shape.size() + gate_named.size() + up_named.size());
        for (const auto& cand : gate_up_shape) {
            if ((!have_gate || cand.raw.rows == gate.rows || gate.rows == 0) &&
                (!have_up || cand.raw.rows == up.rows || up.rows == 0)) {
                remaining.push_back(cand);
            }
        }
        if (!have_gate && !have_up && remaining.size() >= 2) {
            gate = remaining[0].raw;
            up = remaining[1].raw;
            gate_name = remaining[0].name;
            up_name = remaining[1].name;
            have_gate = 1;
            have_up = 1;
        } else if (!have_gate && have_up) {
            for (const auto& cand : remaining) {
                if (cand.raw.rows == up.rows && cand.name != up_name) {
                    gate = cand.raw;
                    gate_name = cand.name;
                    have_gate = 1;
                    break;
                }
            }
        } else if (have_gate && !have_up) {
            for (const auto& cand : remaining) {
                if (cand.raw.rows == gate.rows && cand.name != gate_name) {
                    up = cand.raw;
                    up_name = cand.name;
                    have_up = 1;
                    break;
                }
            }
        }
    }
    if (!have_down && !down_shape.empty()) {
        down = down_shape[0].raw;
        down_name = down_shape[0].name;
        have_down = 1;
    }
    if (!have_gate || !have_up || !have_down) {
        moe_profile_log(
            "mlp_shared_graph_role_candidates",
            "layer=%d ok=0 scanned=%u role_shared=%u name_shared=%u gate_named=%zu up_named=%zu down_named=%zu gateup_shape=%zu down_shape=%zu",
            layer, scanned, role_shared, name_shared,
            gate_named.size(), up_named.size(), down_named.size(), gate_up_shape.size(), down_shape.size());
        return 0;
    }
    const uint32_t intermediate = gate.rows;
    if (intermediate == 0 || intermediate > 100000 ||
        up.rows != intermediate || gate.cols != hidden_size || up.cols != hidden_size ||
        down.rows != hidden_size || down.cols != intermediate) {
        moe_profile_log(
            "mlp_shared_graph_role_shape_mismatch",
            "layer=%d gate=%ux%u up=%ux%u down=%ux%u hidden=%u gate_name=%s up_name=%s down_name=%s",
            layer, gate.rows, gate.cols, up.rows, up.cols, down.rows, down.cols, hidden_size,
            gate_name.c_str(), up_name.c_str(), down_name.c_str());
        return 0;
    }
    *gate_out = gate;
    *up_out = up;
    *down_out = down;
    moe_profile_log(
        "mlp_shared_graph_role_bind",
        "layer=%d gate=%s up=%s down=%s intermediate=%u hidden=%u",
        layer, gate_name.c_str(), up_name.c_str(), down_name.c_str(), intermediate, hidden_size);
    return 1;
}

static int moe_layer_shared_expert_dense_mlp_f32(
    moe_pc_engine* engine,
    int32_t layer,
    moe_decode_scratch* s,
    const float* input,
    uint32_t hidden_size,
    float* out_hidden
) {
    static const char* const gate_suffixes[] = {
        "shared_expert.gate_proj.weight",
        "shared_experts.gate_proj.weight",
        "shared_expert.gate.weight",
        "shared_experts.gate.weight",
        "shared_expert_gate.weight",
        "shared_expert_gate_proj.weight",
        "shared_expert.gate_inp.weight",
        "ffn_gate_shexp.weight",
        "ffn_gate_inp_shexp.weight",
        "ffn_shared_gate.weight",
        "ffn_shared_gate_proj.weight",
        "ffn_shared_gate_inp.weight",
        "ffn_shared_expert_gate.weight",
        "ffn_shared_expert_gate_inp.weight",
        "ffn_shared_shexp_gate.weight",
        "shared_gate.weight",
        "shared_gate_proj.weight",
        "shared_expert.w1.weight",
        "shared_experts.w1.weight",
        "ffn_shared_exps_gate.weight"
    };
    static const char* const up_suffixes[] = {
        "shared_expert.up_proj.weight",
        "shared_experts.up_proj.weight",
        "shared_expert.up.weight",
        "shared_experts.up.weight",
        "shared_expert_up.weight",
        "shared_expert_up_proj.weight",
        "shared_expert.up_inp.weight",
        "ffn_up_shexp.weight",
        "ffn_up_inp_shexp.weight",
        "ffn_shared_up.weight",
        "ffn_shared_up_proj.weight",
        "ffn_shared_up_inp.weight",
        "ffn_shared_expert_up.weight",
        "ffn_shared_expert_up_inp.weight",
        "ffn_shared_shexp_up.weight",
        "shared_up.weight",
        "shared_up_proj.weight",
        "shared_expert.w3.weight",
        "shared_experts.w3.weight",
        "ffn_shared_exps_up.weight"
    };
    static const char* const down_suffixes[] = {
        "shared_expert.down_proj.weight",
        "shared_experts.down_proj.weight",
        "shared_expert.down.weight",
        "shared_experts.down.weight",
        "shared_expert_down.weight",
        "shared_expert_down_proj.weight",
        "shared_expert.down_inp.weight",
        "ffn_down_shexp.weight",
        "ffn_down_inp_shexp.weight",
        "ffn_shared_down.weight",
        "ffn_shared_down_proj.weight",
        "ffn_shared_down_inp.weight",
        "ffn_shared_expert_down.weight",
        "ffn_shared_expert_down_inp.weight",
        "ffn_shared_shexp_down.weight",
        "shared_down.weight",
        "shared_down_proj.weight",
        "shared_expert.w2.weight",
        "shared_experts.w2.weight",
        "ffn_shared_exps_down.weight"
    };
    static const char* const gate_roles[] = {
        "shared_expert_gate",
        "shared_expert_gate_projection",
        "shared_expert_gate_proj",
        "shared_expert_gate_inp",
        "shared_expert_mlp_gate",
        "shared_experts_gate",
        "shared_ffn_gate",
        "ffn_shared_gate"
    };
    static const char* const up_roles[] = {
        "shared_expert_up",
        "shared_expert_up_projection",
        "shared_expert_up_proj",
        "shared_expert_up_inp",
        "shared_expert_mlp_up",
        "shared_experts_up",
        "shared_ffn_up",
        "ffn_shared_up"
    };
    static const char* const down_roles[] = {
        "shared_expert_down",
        "shared_expert_down_projection",
        "shared_expert_down_proj",
        "shared_expert_down_inp",
        "shared_expert_mlp_down",
        "shared_experts_down",
        "shared_ffn_down",
        "ffn_shared_down"
    };
    if (moe_layer_dense_mlp_from_suffixes_f32(
            engine, layer, s, input, hidden_size, out_hidden,
            gate_suffixes, sizeof(gate_suffixes) / sizeof(gate_suffixes[0]),
            up_suffixes, sizeof(up_suffixes) / sizeof(up_suffixes[0]),
            down_suffixes, sizeof(down_suffixes) / sizeof(down_suffixes[0]),
            "mlp_shared_dense_begin",
            gate_roles, sizeof(gate_roles) / sizeof(gate_roles[0]),
            up_roles, sizeof(up_roles) / sizeof(up_roles[0]),
            down_roles, sizeof(down_roles) / sizeof(down_roles[0]),
            1)) {
        return 1;
    }
    if (!moe_graph_ir_math_required(engine)) {
        return 0;
    }
    moe_raw_tensor_cache_entry gate{};
    moe_raw_tensor_cache_entry up{};
    moe_raw_tensor_cache_entry down{};
    if (!moe_map_shared_expert_graph_role_f32(engine, layer, hidden_size, &gate, &up, &down)) {
        return 0;
    }
    const uint32_t intermediate = gate.rows;
    if (!s || s->gate.size() < intermediate || s->up.size() < intermediate || s->shared.size() < hidden_size) {
        return 0;
    }
    const uint64_t total_start_us = moe_profile_now_us();
    const int activation_mode = moe_engine_gelu_activation_mode(engine);
    {
        moe_profile_parallel_label_scope rows_label("shared_dense_mlp_gate_up");
        if (!moe_parallel_rows(engine, intermediate, 0, [&](uint32_t row) {
            const float g = moe_raw_tensor_dot_row(gate, row, input, hidden_size);
            const float u = moe_raw_tensor_dot_row(up, row, input, hidden_size);
            s->gate[row] = moe_generation_gated_activation_cached(activation_mode, g, u);
            return 1;
        })) {
            return 0;
        }
    }
    if (!moe_common_linear_f32(engine, down, s->gate.data(), intermediate, out_hidden, hidden_size)) {
        return 0;
    }
    moe_profile_log(
        "mlp_shared_dense_graph_role_end",
        "layer=%d intermediate=%u hidden=%u activation_mode=%d total_ms=%.3f",
        layer, intermediate, hidden_size, activation_mode, moe_profile_elapsed_ms(total_start_us));
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
    // Dense MLP fallback path — only reached when graph IR common tensor
    // binding fails AND offload_graph_ir is unavailable.  Derive intermediate
    // size dynamically from engine metadata rather than hardcoding.
    uint32_t dense_intermediate = 0;
    if (engine) {
        double ir_value = 0.0;
        if (!engine->offload_gguf_metadata_json.empty() &&
            (moe_json_get_double_local(engine->offload_gguf_metadata_json, "intermediate_size", &ir_value) ||
             moe_json_get_double_local(engine->offload_gguf_metadata_json, "ffn_dim", &ir_value) ||
             moe_json_get_double_local(engine->offload_gguf_metadata_json, "feed_forward_length", &ir_value)) &&
            ir_value > 0.0 && ir_value < 1000000.0) {
            dense_intermediate = (uint32_t)ir_value;
        } else if (!engine->offload_graph_ir_json.empty() &&
            (moe_json_get_double_local(engine->offload_graph_ir_json, "intermediate_size", &ir_value) ||
             moe_json_get_double_local(engine->offload_graph_ir_json, "ffn_dim", &ir_value) ||
             moe_json_get_double_local(engine->offload_graph_ir_json, "feed_forward_length", &ir_value)) &&
            ir_value > 0.0 && ir_value < 1000000.0) {
            dense_intermediate = (uint32_t)ir_value;
        }
    }
    if (dense_intermediate == 0) {
        // Universal default: 4 * hidden_size is the standard MLP expansion ratio
        dense_intermediate = hidden_size * 4u;
    }
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
