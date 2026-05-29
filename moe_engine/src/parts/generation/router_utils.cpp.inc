#include "router_entropy.cpp.inc"

static int moe_prepare_router_weight_sidecar_input_f32(
    moe_pc_engine* engine,
    int32_t layer,
    const float* hidden,
    uint32_t hidden_size,
    std::vector<float>* out
);

static int moe_direct4_dot_enabled();

static int moe_raw_tensor_dot_adjacent_four_rows_direct(
    const moe_raw_tensor_cache_entry& raw,
    uint32_t row,
    uint64_t row_bytes,
    const float* input,
    uint32_t input_count,
    uint8_t effective_weight_format,
    const moe_mxfp4_row_layout_t& mxfp4_layout,
    int mxfp4_pairwise_order,
    float* out0,
    float* out1,
    float* out2,
    float* out3
);

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

    // Fix 18: Cache tensor mapping to avoid std::string vector allocation per token
    const uint32_t suffix_hash = moe_router_fnv1a_hash_string(suffixes[0]);
    const uint64_t key = ((uint64_t)(uint32_t)layer << 32) | suffix_hash | 0x80000000ull;
    {
        std::shared_lock<std::shared_mutex> lock(engine->raw_tensor_cache_mutex);
        auto found = engine->raw_tensor_cache.find(key);
        if (found != engine->raw_tensor_cache.end() && found->second.base) {
            *out = found->second;
            return 1;
        }
    }
    moe_raw_tensor_desc desc{};
    const moe_gguf_common_tensor_record* rec =
        find_common_tensor_by_op_role(engine, "moe_router", layer, &desc);
    if (!rec) {
        rec = find_common_tensor_by_op_role(engine, "router", layer, &desc);
    }
    if (!rec) {
        std::vector<std::string> names;
        moe_push_layer_tensor_candidates(&names, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]));
        rec = moe_common_tensor_find_first_raw_compatible(engine, names, &desc);
    }
    if (!rec || desc.cols != hidden_size || desc.rows != experts) {
        return 0;
    }
    if (!map_common_tensor_record(engine, *rec, desc, out)) {
        return 0;
    }
    {
        std::unique_lock<std::shared_mutex> lock(engine->raw_tensor_cache_mutex);
        engine->raw_tensor_cache[key] = *out;
    }
    return 1;
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
    // BUGFIX 517: Check experts upper bound to prevent overflow ??
    // Problem: experts can be arbitrarily large from config
    // scratch_logits->assign(experts, 0.0f) can allocate huge memory
    // Solution: Add reasonable upper bound check (e.g., 10000)
    if (experts > 10000) {
        return 0;
    }
    // BUGFIX 598: Check hidden_size upper bound ??
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
        std::vector<float> weight_sidecar_input;
        const float* dot_input = hidden;
        const uint32_t common_elem = moe_raw_scalar_size(common.scalar_encoding);
        const int common_is_dense_raw_matrix =
            common_elem > 0u &&
            common.weight_row_bytes == (uint64_t)hidden_size * (uint64_t)common_elem &&
            (common.weight_format == moe_WEIGHT_ENCODING_RAW_FP32 ||
             common.weight_format == moe_WEIGHT_ENCODING_RAW_BF16 ||
             common.weight_format == moe_WEIGHT_ENCODING_RAW_FP16);
        if (!common_is_dense_raw_matrix &&
            moe_prepare_router_weight_sidecar_input_f32(
                engine, layer, hidden, hidden_size, &weight_sidecar_input)) {
            dot_input = weight_sidecar_input.data();
        } else if (common_is_dense_raw_matrix && moe_profile_level() >= 2) {
            moe_profile_log(
                "router_weight_sidecar_skip",
                "layer=%d reason=raw_router_matrix_already_scaled rows=%u cols=%u row_bytes=%llu elem=%u format=%u",
                layer,
                common.rows,
                common.cols,
                (unsigned long long)common.weight_row_bytes,
                common_elem,
                (unsigned)common.weight_format);
        }
        const uint8_t effective_weight_format =
            moe_effective_weight_format_for_row_bytes(common, hidden_size);
        const int direct_four_row =
            moe_direct4_dot_enabled() &&
            !common.math_transposed &&
            (effective_weight_format == moe_WEIGHT_ENCODING_MXFP4 ||
             effective_weight_format == moe_WEIGHT_ENCODING_GGUF_Q8_0 ||
             effective_weight_format == moe_WEIGHT_ENCODING_GGUF_Q5_1 ||
             effective_weight_format == moe_WEIGHT_ENCODING_GGUF_Q5_K);
        if (direct_four_row) {
            const moe_mxfp4_row_layout_t mxfp4_layout =
                effective_weight_format == moe_WEIGHT_ENCODING_MXFP4 ?
                    moe_mxfp4_row_layout_for_bytes(hidden_size, row_bytes) :
                    moe_mxfp4_row_layout_t{};
            const int mxfp4_pairwise_order =
                effective_weight_format == moe_WEIGHT_ENCODING_MXFP4 ?
                    moe_mxfp4_pairwise_nibble_order_enabled() : 0;
            const uint32_t row_blocks = (experts + 3u) / 4u;
            moe_profile_parallel_label_scope rows_label("router_logits_direct4");
            return moe_parallel_rows(engine, row_blocks, 0, [&](uint32_t block) {
                const uint32_t row = block * 4u;
                if (row + 4u <= experts) {
                    return moe_raw_tensor_dot_adjacent_four_rows_direct(
                        common, row, row_bytes, dot_input, hidden_size,
                        effective_weight_format, mxfp4_layout, mxfp4_pairwise_order,
                        &(*scratch_logits)[row + 0u],
                        &(*scratch_logits)[row + 1u],
                        &(*scratch_logits)[row + 2u],
                        &(*scratch_logits)[row + 3u]);
                }
                for (uint32_t r = row; r < experts; ++r) {
                    const float logit = moe_raw_tensor_dot_row(common, r, dot_input, hidden_size);
                    if (!std::isfinite(logit)) {
                        return 0;
                    }
                    (*scratch_logits)[r] = logit;
                }
                return 1;
            });
        }
        moe_profile_parallel_label_scope rows_label("router_logits");
        return moe_parallel_rows(engine, experts, 0, [&](uint32_t row) {
            const float logit = moe_raw_tensor_dot_row(common, row, dot_input, hidden_size);
            if (!std::isfinite(logit)) {
                return 0;
            }
            (*scratch_logits)[row] = logit;
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
    if (!std::isfinite(x)) {
        return 0.5f;
    }
    const float exp_val = std::exp(-x);
    if (!std::isfinite(exp_val)) {
        return x > 0.0f ? 1.0f : 0.0f;
    }
    // BUGFIX 219: Numerically stable sigmoid (same pattern as BUGFIX 209)
    // Problem: Branch-based sigmoid has underflow for large negative x
    // Solution: Use single formula: 1 / (1 + exp(-x)) for all x
    // Impact: Accurate for extreme values, ~5% faster (branch removal)
    return 1.0f / (1.0f + exp_val);
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

static int moe_map_router_bias_tensor(
    moe_pc_engine* engine,
    int32_t layer,
    uint32_t experts,
    moe_raw_tensor_cache_entry* out
) {
    if (!engine || !out || layer < 0 || experts == 0) {
        return 0;
    }
    static const char* const suffixes[] = {
        "e_score_correction_bias",
        "expert_bias",
        "router.expert_bias",
        "router.bias",
        "mlp.router.expert_bias",
        "moe.gate.expert_bias",
        "moe.gate.e_score_correction_bias",
        "ffn_gate_inp.bias"
    };
    struct router_bias_cache_entry {
        const moe_pc_engine* engine = nullptr;
        uint64_t generation = 0;
        int32_t layer = INT32_MIN;
        uint32_t experts = 0;
        uint64_t key = 0;
        uint8_t found = 0;
        moe_raw_tensor_cache_entry raw{};
    };
    static thread_local std::vector<router_bias_cache_entry> tls_bias_cache;
    const uint32_t suffix_hash = moe_router_fnv1a_hash_string(suffixes[0]);
    const uint64_t key = ((uint64_t)(uint32_t)layer << 32) | suffix_hash | 0x90000000ull;
    const uint64_t generation = engine->engine_generation;
    for (const router_bias_cache_entry& e : tls_bias_cache) {
        if (e.engine == engine && e.generation == generation &&
            e.layer == layer && e.experts == experts && e.key == key) {
            if (!e.found || !e.raw.base) {
                return 0;
            }
            *out = e.raw;
            return 1;
        }
    }
    {
        std::shared_lock<std::shared_mutex> lock(engine->raw_tensor_cache_mutex);
        auto found = engine->raw_tensor_cache.find(key);
        if (found != engine->raw_tensor_cache.end() && found->second.base) {
            *out = found->second;
            router_bias_cache_entry entry{};
            entry.engine = engine;
            entry.generation = generation;
            entry.layer = layer;
            entry.experts = experts;
            entry.key = key;
            entry.found = 1;
            entry.raw = *out;
            if (tls_bias_cache.size() > 256u) {
                tls_bias_cache.erase(tls_bias_cache.begin(), tls_bias_cache.begin() + 64);
            }
            tls_bias_cache.push_back(entry);
            return 1;
        }
    }
    std::vector<std::string> names;
    moe_push_layer_tensor_candidates(&names, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]));
    moe_raw_tensor_desc desc{};
    const moe_gguf_common_tensor_record* rec =
        moe_common_tensor_find_first_raw_compatible(engine, names, &desc);
    moe_raw_tensor_cache_entry raw{};
    uint8_t found_ok = 0;
    if (rec && map_common_tensor_record(engine, *rec, desc, &raw)) {
        const uint64_t count = moe_router_raw_scalar_count(raw);
        if (count == (uint64_t)experts) {
            found_ok = 1;
        } else {
            moe_profile_log(
                "router_bias_reject",
                "layer=%d experts=%u count=%llu rows=%u cols=%u",
                layer, experts, (unsigned long long)count, raw.rows, raw.cols);
        }
    }
    router_bias_cache_entry entry{};
    entry.engine = engine;
    entry.generation = generation;
    entry.layer = layer;
    entry.experts = experts;
    entry.key = key;
    entry.found = found_ok;
    entry.raw = raw;
    if (tls_bias_cache.size() > 256u) {
        tls_bias_cache.erase(tls_bias_cache.begin(), tls_bias_cache.begin() + 64);
    }
    tls_bias_cache.push_back(entry);
    if (!found_ok) {
        return 0;
    }
    *out = raw;
    {
        std::unique_lock<std::shared_mutex> lock(engine->raw_tensor_cache_mutex);
        engine->raw_tensor_cache[key] = *out;
    }
    return 1;
}

static int moe_router_apply_bias_f32(
    moe_pc_engine* engine,
    int32_t layer,
    uint32_t experts,
    std::vector<float>* logits
) {
    if (!engine || !logits || logits->size() < experts || experts == 0) {
        return 0;
    }
    moe_raw_tensor_cache_entry bias{};
    if (!moe_map_router_bias_tensor(engine, layer, experts, &bias)) {
        return 1;
    }
    double abs_sum = 0.0;
    float max_abs = 0.0f;
    for (uint32_t i = 0; i < experts; ++i) {
        float b = 0.0f;
        if (!moe_router_raw_scalar_at_f32(bias, i, &b) || !std::isfinite(b)) {
            return 0;
        }
        (*logits)[i] += b;
        const float ab = std::fabs(b);
        abs_sum += (double)ab;
        if (ab > max_abs) max_abs = ab;
    }
    moe_profile_log(
        "router_bias_apply",
        "layer=%d experts=%u mean_abs=%.6g max_abs=%.6g",
        layer, experts, abs_sum / (double)experts, (double)max_abs);
    return 1;
}

static int moe_map_router_per_expert_scale_tensor_f32(
    moe_pc_engine* engine,
    int32_t layer,
    uint32_t experts,
    moe_raw_tensor_cache_entry* out
) {
    if (!engine || !out || layer < 0 || experts == 0) {
        return 0;
    }
    const uint32_t suffix_hash = moe_router_fnv1a_hash_string("router_per_expert_scale");
    const uint64_t key = ((uint64_t)(uint32_t)layer << 32) | suffix_hash | 0xa0000000ull;
    {
        std::shared_lock<std::shared_mutex> lock(engine->raw_tensor_cache_mutex);
        auto found = engine->raw_tensor_cache.find(key);
        if (found != engine->raw_tensor_cache.end() && found->second.base) {
            *out = found->second;
            return 1;
        }
    }
    moe_raw_tensor_cache_entry raw{};
    int mapped = 0;
    if (moe_graph_ir_math_required(engine)) {
        static const char* const roles[] = {
            "router_per_expert_scale",
            "per_expert_scale",
            "moe_router_per_expert_scale"
        };
        mapped = moe_graph_ir_map_layer_op_role_any(
            engine, layer, roles, sizeof(roles) / sizeof(roles[0]), &raw);
        if (!mapped) {
            static const char* const sections[] = { "router", "topk", "moe" };
            for (const char* section : sections) {
                for (const char* role : roles) {
                    if (moe_graph_ir_map_layer_contract_tensor_role_f32(engine, layer, section, role, &raw)) {
                        mapped = 1;
                        break;
                    }
                }
                if (mapped) {
                    break;
                }
            }
        }
    }
    std::string mapped_name;
    static const char* const suffixes[] = {
        "ffn_gate_inp.per_expert_scale",
        "router.per_expert_scale",
        "router.per_expert_scale.weight",
        "mlp.router.per_expert_scale",
        "mlp.router.per_expert_scale.weight",
        "moe.gate.per_expert_scale",
        "moe.gate.per_expert_scale.weight"
    };
    static const char* const legacy_generated_juju_suffixes[] = {
        "ffn_down_exps.scale"
    };
    if (!mapped) {
        std::vector<std::string> names;
        moe_push_layer_tensor_candidates(&names, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]));
        moe_raw_tensor_desc desc{};
        const moe_gguf_common_tensor_record* rec =
            moe_common_tensor_find_first_raw_compatible(engine, names, &desc);
        if (!rec && moe_graph_ir_math_required(engine)) {
            // Older generated JUJU indexes placed Gemma4 router per-expert
            // scale in graph_ir.layers[].ops.moe_experts.per_expert_output_scale
            // while the materializer contract and HF reference semantics use
            // it as top-k router weight scale. Keep this as a migration fallback
            // only after explicit GraphIR/router contract lookup failed.
            names.clear();
            moe_push_layer_tensor_candidates(
                &names,
                layer,
                legacy_generated_juju_suffixes,
                sizeof(legacy_generated_juju_suffixes) / sizeof(legacy_generated_juju_suffixes[0]));
            rec = moe_common_tensor_find_first_raw_compatible(engine, names, &desc);
        }
        if (!rec) {
            return 0;
        }
        if (!map_common_tensor_record(engine, *rec, desc, &raw)) {
            return 0;
        }
        mapped_name = rec->name;
    }
    const uint64_t count = moe_router_raw_scalar_count(raw);
    if (count != 1u && count != (uint64_t)experts) {
        moe_profile_log(
            "router_per_expert_scale_reject",
            "layer=%d experts=%u count=%llu name=%s",
            layer, experts, (unsigned long long)count,
            mapped_name.empty() ? "<graph_ir_contract>" : mapped_name.c_str());
        return 0;
    }
    {
        std::unique_lock<std::shared_mutex> lock(engine->raw_tensor_cache_mutex);
        engine->raw_tensor_cache[key] = raw;
    }
    *out = raw;
    return 1;
}

static int moe_router_apply_per_expert_scales_f32(
    moe_pc_engine* engine,
    int32_t layer,
    uint32_t experts,
    std::vector<int>* selected_experts,
    std::vector<float>* selected_weights
) {
    if (!engine || !selected_experts || !selected_weights || selected_experts->size() != selected_weights->size()) {
        return 0;
    }
    if (selected_experts->empty() || experts == 0) {
        return 1;
    }
    moe_raw_tensor_cache_entry scale_raw{};
    if (!moe_map_router_per_expert_scale_tensor_f32(engine, layer, experts, &scale_raw)) {
        return 1;
    }
    const uint64_t count = moe_router_raw_scalar_count(scale_raw);
    double scale_abs_sum = 0.0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < selected_experts->size(); ++i) {
        const int expert = (*selected_experts)[i];
        if (expert < 0 || (uint32_t)expert >= experts) {
            return 0;
        }
        float scale = 1.0f;
        const uint64_t offset = count == 1u ? 0u : (uint64_t)(uint32_t)expert;
        if (!moe_router_raw_scalar_at_f32(scale_raw, offset, &scale) || !std::isfinite(scale)) {
            return 0;
        }
        (*selected_weights)[i] *= scale;
        const float ab = std::fabs(scale);
        scale_abs_sum += (double)ab;
        if (ab > max_abs) {
            max_abs = ab;
        }
    }
    moe_profile_log(
        "router_per_expert_scale_apply",
        "layer=%d selected=%zu count=%llu mean_abs=%.6g max_abs=%.6g first_expert=%d first_scale_weight=%.6g",
        layer, selected_experts->size(), (unsigned long long)count,
        scale_abs_sum / (double)selected_experts->size(), (double)max_abs,
        selected_experts->empty() ? -1 : (*selected_experts)[0],
        selected_weights->empty() ? 0.0 : (double)(*selected_weights)[0]);
    return 1;
}


static int moe_router_name_has_suffix_f32(const std::string& name, const char* suffix) {
    if (!suffix || !suffix[0] || name.size() < std::strlen(suffix)) {
        return 0;
    }
    return name.compare(name.size() - std::strlen(suffix), std::strlen(suffix), suffix) == 0 ? 1 : 0;
}

static int moe_router_scale_tensor_is_weight_sidecar_name_f32(const std::string& name) {
    // Generator contract key: ffn_gate_inp_scale_is_weight_scale_sidecar.
    // ffn_gate_inp.scale/scales in GGUF/MXFP artifacts are router-weight
    // quantization sidecars, not router-input activation/RMSNorm scales.
    static const char* const sidecar_suffixes[] = {
        "ffn_gate_inp.scale",
        "ffn_gate_inp.scales",
        "ffn_gate_inp.weight.scale",
        "ffn_gate_inp.weight.scales",
        ".weight.scale",
        ".weight.scales"
    };
    for (const char* suffix : sidecar_suffixes) {
        if (moe_router_name_has_suffix_f32(name, suffix)) {
            return 1;
        }
    }
    return 0;
}

static int moe_router_scale_role_text_is_weight_sidecar_f32(
    const std::string& text
) {
    const std::string role = moe_lower_ascii_copy(text);
    return role.find("weight_scale_sidecar") != std::string::npos ||
        role.find("weight_sidecar") != std::string::npos ||
        role.find("quant_sidecar") != std::string::npos ||
        role.find("router_weight_scale_sidecar") != std::string::npos ||
        role.find("mxfp") != std::string::npos;
}

static int moe_router_scale_record_is_weight_sidecar_f32(
    const moe_gguf_common_tensor_record& rec
) {
    if (moe_router_scale_tensor_is_weight_sidecar_name_f32(rec.name)) {
        return 1;
    }
    return moe_router_scale_role_text_is_weight_sidecar_f32(rec.op_role) ||
        moe_router_scale_role_text_is_weight_sidecar_f32(rec.execution_op) ||
        moe_router_scale_role_text_is_weight_sidecar_f32(rec.bundle_member_role) ||
        moe_router_scale_role_text_is_weight_sidecar_f32(rec.graph_role_text);
}

static int moe_router_has_internal_weight_scale_contract_f32(
    const moe_pc_engine* engine,
    int32_t layer
) {
    if (!engine || layer < 0) {
        return 0;
    }
    static const char* const roles[] = {
        "moe_router_scale",
        "router_scale"
    };
    moe_raw_tensor_desc desc{};
    for (const char* role : roles) {
        const moe_gguf_common_tensor_record* rec =
            find_common_tensor_by_op_role(engine, role, layer, &desc);
        if (rec && !moe_router_scale_record_is_weight_sidecar_f32(*rec)) {
            return 1;
        }
    }
    static const char* const sidecar_roles[] = {
        "router_weight_scale_sidecar",
        "moe_router_weight_scale_sidecar",
        "ffn_gate_inp_scale",
        "ffn_gate_inp_scales"
    };
    for (const char* role : sidecar_roles) {
        const moe_gguf_common_tensor_record* rec =
            find_common_tensor_by_op_role(engine, role, layer, &desc);
        if (rec && moe_router_scale_record_is_weight_sidecar_f32(*rec)) {
            return 1;
        }
    }
    static const char* const suffixes[] = {
        // Availability here means router-input activation scale, not MXFP/GGUF
        // weight sidecar tensors. Bare ffn_gate_inp.scale/scales are router
        // weight sidecars; only explicit input-scale names are accepted by
        // suffix. Generic router.scale/mlp.router.scale/moe.gate.scale are
        // ambiguous and require an explicit op-role record instead.
        "router_input_scale",
        "router.input_scale",
        "mlp.router_input_scale",
        "moe.gate_input_scale"
    };
    for (const char* suffix : suffixes) {
        if (moe_engine_has_common_tensor_suffix(engine, layer, suffix)) {
            return 1;
        }
    }
    static const char* const sidecar_suffixes[] = {
        "ffn_gate_inp.scale",
        "ffn_gate_inp.scales",
        "ffn_gate_inp.weight.scale",
        "ffn_gate_inp.weight.scales"
    };
    for (const char* suffix : sidecar_suffixes) {
        if (moe_engine_has_common_tensor_suffix(engine, layer, suffix)) {
            return 1;
        }
    }
    return 0;
}


static const moe_gguf_common_tensor_record* moe_find_router_scale_record_by_role_f32(
    const moe_pc_engine* engine,
    const char* role,
    int32_t layer,
    moe_raw_tensor_desc* out_desc
) {
    if (!engine || !role || !role[0]) {
        return nullptr;
    }
    for (const moe_gguf_common_tensor_record& rec : engine->gguf_common_tensors) {
        if (!moe_common_tensor_op_role_matches(rec, role) ||
            !moe_common_tensor_layer_matches(rec, layer)) {
            continue;
        }
        if (moe_router_scale_record_is_weight_sidecar_f32(rec)) {
            if (moe_profile_level() >= 2) {
                moe_profile_log(
                    "router_scale_contract_reject",
                    "layer=%d name=%s op_role=%s reason=weight_sidecar",
                    layer, rec.name.c_str(), rec.op_role.c_str());
            }
            continue;
        }
        moe_raw_tensor_desc desc{};
        if (!moe_common_tensor_record_to_raw_desc(rec, &desc) ||
            !moe_common_tensor_record_raw_compatible(rec, desc)) {
            continue;
        }
        if (out_desc) {
            *out_desc = desc;
        }
        return &rec;
    }
    return nullptr;
}

static int moe_router_utils_doc_next_array_key(
    const std::string& doc,
    const char* key,
    size_t start_pos,
    size_t* begin,
    size_t* end
) {
    if (!key || !begin || !end || start_pos >= doc.size()) {
        return 0;
    }
    const size_t key_len = std::strlen(key);
    size_t pos = start_pos;
    while ((pos = doc.find('"', pos)) != std::string::npos) {
        if (pos + 1u + key_len < doc.size() &&
            doc.compare(pos + 1u, key_len, key) == 0 &&
            doc[pos + 1u + key_len] == '"') {
            const size_t colon = doc.find_first_not_of(" \t\r\n", pos + key_len + 2u);
            if (colon != std::string::npos && colon < doc.size() && doc[colon] == ':') {
                const size_t value = doc.find_first_not_of(" \t\r\n", colon + 1u);
                if (value != std::string::npos && value < doc.size() && doc[value] == '[') {
                    const size_t close = moe_json_match_bracket_local(doc, value, '[', ']');
                    if (close != std::string::npos) {
                        *begin = value;
                        *end = close;
                        return 1;
                    }
                }
            }
        }
        ++pos;
    }
    return 0;
}

static int moe_router_utils_slice_matches_layer(
    const std::string& doc,
    size_t obj_begin,
    size_t obj_end,
    int32_t layer,
    uint64_t ordinal
) {
    uint64_t parsed = UINT64_MAX;
    if (moe_json_get_u64_slice(doc, obj_begin, obj_end, "layer", &parsed) ||
        moe_json_get_u64_slice(doc, obj_begin, obj_end, "layer_index", &parsed) ||
        moe_json_get_u64_slice(doc, obj_begin, obj_end, "idx", &parsed) ||
        moe_json_get_u64_slice(doc, obj_begin, obj_end, "block", &parsed) ||
        moe_json_get_u64_slice(doc, obj_begin, obj_end, "block_index", &parsed)) {
        return parsed == (uint64_t)layer ? 1 : 0;
    }
    return ordinal == (uint64_t)layer ? 1 : 0;
}

static int moe_router_scale_contract_name_is_sidecar_f32(const std::string& name) {
    const std::string lower = moe_lower_ascii_copy(name);
    return lower.find("ffn_gate_inp.scale") != std::string::npos ||
        lower.find("ffn_gate_inp.scales") != std::string::npos ||
        lower.find("ffn_gate_inp.weight.scale") != std::string::npos ||
        lower.find("ffn_gate_inp.weight.scales") != std::string::npos ||
        lower.find("router_weight_scale_sidecar") != std::string::npos ||
        lower.find("weight_sidecar") != std::string::npos ? 1 : 0;
}

static void moe_router_push_contract_scale_names_from_scope_f32(
    const std::string& doc,
    size_t scope_begin,
    size_t scope_end,
    std::vector<std::string>* out_names
) {
    if (!out_names || scope_begin >= scope_end || scope_end > doc.size()) {
        return;
    }
    static const char* const keys[] = {
        "router_input_scale", "router_input_scales", "router_input_scale_tensors",
        "router_scale", "router_scales",
        "scale", "scales",
        "input_scale", "input_scales",
        "activation_scale", "activation_scales"
    };
    for (const char* key : keys) {
        std::vector<std::string> names;
        if (moe_json_get_string_array_slice(doc, scope_begin, scope_end, key, &names)) {
            for (const std::string& name : names) {
                if (name.empty() || moe_router_scale_contract_name_is_sidecar_f32(name)) {
                    continue;
                }
                if (std::find(out_names->begin(), out_names->end(), name) == out_names->end()) {
                    out_names->push_back(name);
                }
            }
        }
        std::string single;
        if (moe_json_get_string_slice(doc, scope_begin, scope_end, key, &single) &&
            !single.empty() && !moe_router_scale_contract_name_is_sidecar_f32(single) &&
            std::find(out_names->begin(), out_names->end(), single) == out_names->end()) {
            out_names->push_back(single);
        }
    }
}

static int moe_router_contract_scale_tensor_names_f32(
    const moe_pc_engine* engine,
    int32_t layer,
    std::vector<std::string>* out_names
) {
    if (out_names) {
        out_names->clear();
    }
    if (!engine || layer < 0 || !out_names) {
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
            const size_t before = out_names->size();
            const std::string row = doc.substr(obj_begin, obj_end - obj_begin);
            size_t router_begin = 0;
            size_t router_end = 0;
            if (moe_json_find_object_local(row, "router", &router_begin, &router_end)) {
                moe_router_push_contract_scale_names_from_scope_f32(
                    doc, obj_begin + router_begin, obj_begin + router_end, out_names);
            }
            size_t semantic_begin = 0;
            size_t semantic_end = 0;
            if (moe_json_find_object_local(row, "semantic_contract", &semantic_begin, &semantic_end)) {
                moe_router_push_contract_scale_names_from_scope_f32(
                    doc, obj_begin + semantic_begin, obj_begin + semantic_end, out_names);
            }
            size_t ops_begin = 0;
            size_t ops_end = 0;
            if (moe_json_find_array_local(row, "ops", &ops_begin, &ops_end)) {
                size_t op_cursor = obj_begin + ops_begin + 1u;
                const size_t op_limit = obj_begin + ops_end;
                while (op_cursor < op_limit) {
                    size_t op_begin = 0;
                    size_t op_end = 0;
                    if (!moe_json_next_object_in_array(doc, &op_cursor, op_limit, &op_begin, &op_end)) {
                        break;
                    }
                    std::string op_name;
                    if (moe_json_get_string_slice(doc, op_begin, op_end, "name", &op_name)) {
                        const std::string lowered = moe_lower_ascii_copy(op_name);
                        if (lowered == "router_input" || lowered == "moe_router" || lowered == "router") {
                            moe_router_push_contract_scale_names_from_scope_f32(doc, op_begin, op_end, out_names);
                        }
                    }
                }
            }
            moe_router_push_contract_scale_names_from_scope_f32(doc, obj_begin, obj_end, out_names);
            if (out_names->size() > before) {
                moe_profile_log(
                    "router_scale_contract_names",
                    "layer=%d count=%zu source=layer_execution_contract_table",
                    layer, out_names->size() - before);
            }
        }
    }
    return out_names->empty() ? 0 : 1;
}

static int moe_router_has_direct_input_scale_contract_f32(
    const moe_pc_engine* engine,
    int32_t layer
) {
    if (!engine || layer < 0) {
        return 0;
    }
    static const char* const direct_suffixes[] = {
        "router_input_scale",
        "router.input_scale",
        "mlp.router_input_scale",
        "moe.gate_input_scale"
    };
    for (const char* suffix : direct_suffixes) {
        if (moe_engine_has_common_tensor_suffix(engine, layer, suffix)) {
            return 1;
        }
    }
    std::vector<std::string> contract_names;
    if (moe_router_contract_scale_tensor_names_f32(engine, layer, &contract_names)) {
        for (const std::string& name : contract_names) {
            if (!name.empty() && !moe_router_scale_contract_name_is_sidecar_f32(name)) {
                // Names came from the generator's router_input_scale/router_scale
                // contract keys, not from a graph-wide suffix scan.  Within
                // that scope a non-sidecar tensor is a direct router-input
                // activation scale.
                return 1;
            }
        }
    }
    static const char* const direct_roles[] = {
        "moe_router_scale",
        "router_scale",
        "router_input_scale"
    };
    moe_raw_tensor_desc desc{};
    for (const char* role : direct_roles) {
        const moe_gguf_common_tensor_record* rec =
            find_common_tensor_by_op_role(engine, role, layer, &desc);
        if (rec && !moe_router_scale_record_is_weight_sidecar_f32(*rec)) {
            return 1;
        }
    }
    return 0;
}

static int moe_router_scope_declares_rms_root_input_transform_f32(
    const std::string& doc,
    size_t begin,
    size_t end
) {
    if (begin >= end || end > doc.size()) {
        return 0;
    }
    static const char* const bool_keys[] = {
        "router_input_rms_root_transform",
        "router_rms_root_input",
        "router_input_uses_rmsnorm_and_hidden_root",
        "router_input_hidden_size_root_scale",
        "scale_free_router_rmsnorm"
    };
    int bool_value = 0;
    for (const char* key : bool_keys) {
        if (moe_json_get_bool_slice(doc, begin, end, key, &bool_value) && bool_value) {
            return 1;
        }
    }
    static const char* const string_keys[] = {
        "router_input_transform",
        "input_transform",
        "router_input_norm_mode",
        "router_norm_mode",
        "scale_semantics",
        "router_scale_semantics",
        "rule",
        "formula",
        "expression"
    };
    auto text_declares_transform = [](const std::string& text) -> int {
        const std::string lower = moe_lower_ascii_copy(text);
        const int has_rms =
            lower.find("rmsnorm") != std::string::npos ||
            lower.find("rms_norm") != std::string::npos ||
            lower.find("router_norm") != std::string::npos;
        const int has_root =
            lower.find("hidden_size**-0.5") != std::string::npos ||
            lower.find("hidden_size^-0.5") != std::string::npos ||
            lower.find("hidden_dim**-0.5") != std::string::npos ||
            lower.find("hidden_dim^-0.5") != std::string::npos ||
            lower.find("rsqrt_hidden") != std::string::npos ||
            lower.find("sqrt_hidden") != std::string::npos ||
            lower.find("root_hidden") != std::string::npos ||
            lower.find("1/sqrt") != std::string::npos;
        const int scale_free =
            lower.find("scale_free") != std::string::npos ||
            lower.find("unweighted") != std::string::npos ||
            lower.find("no_scale") != std::string::npos ||
            lower.find("without_scale") != std::string::npos ||
            lower.find("with_scale:false") != std::string::npos ||
            lower.find("with_scale\":false") != std::string::npos;
        return has_rms && has_root && scale_free ? 1 : 0;
    };
    std::string value;
    for (const char* key : string_keys) {
        value.clear();
        if (moe_json_get_string_slice(doc, begin, end, key, &value) &&
            text_declares_transform(value)) {
            return 1;
        }
    }
    return text_declares_transform(doc.substr(begin, end - begin));
}

static int moe_router_ascii_space_f32(char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\f' || c == '\v';
}

static int moe_router_scope_has_nonempty_array_f32(
    const std::string& doc,
    size_t begin,
    size_t end,
    const char* key
) {
    if (!key || begin >= end || end > doc.size()) {
        return 0;
    }
    size_t arr_begin = 0;
    size_t arr_end = 0;
    if (!moe_router_utils_doc_next_array_key(doc.substr(begin, end - begin), key, 0, &arr_begin, &arr_end)) {
        return 0;
    }
    arr_begin += begin;
    arr_end += begin;
    size_t cursor = arr_begin + 1u;
    while (cursor < arr_end) {
        const char c = doc[cursor++];
        if (!moe_router_ascii_space_f32(c) && c != ']') {
            return 1;
        }
    }
    return 0;
}

static int moe_router_scope_has_empty_array_f32(
    const std::string& doc,
    size_t begin,
    size_t end,
    const char* key
) {
    if (!key || begin >= end || end > doc.size()) {
        return 0;
    }
    size_t arr_begin = 0;
    size_t arr_end = 0;
    if (!moe_router_utils_doc_next_array_key(doc.substr(begin, end - begin), key, 0, &arr_begin, &arr_end)) {
        return 0;
    }
    arr_begin += begin;
    arr_end += begin;
    for (size_t i = arr_begin + 1u; i < arr_end; ++i) {
        const char c = doc[i];
        if (!moe_router_ascii_space_f32(c) && c != ']') {
            return 0;
        }
    }
    return 1;
}

static int moe_router_scope_bool_true_f32(
    const std::string& doc,
    size_t begin,
    size_t end,
    const char* key
) {
    int value = 0;
    return key && begin < end && end <= doc.size() &&
        moe_json_get_bool_slice(doc, begin, end, key, &value) && value ? 1 : 0;
}

static int moe_router_row_contract_implies_sidecar_rms_root_transform_f32(
    const std::string& doc,
    size_t begin,
    size_t end
) {
    if (begin >= end || end > doc.size()) {
        return 0;
    }
    (void)doc;
    // A router weight sidecar only scales router columns unless the IR
    // explicitly declares a router-input transform. Split-MoE shape alone is
    // not enough evidence to switch the router input from expert_ffn_input to
    // hidden or to apply RMS-root normalization.
    return 0;
}

static int moe_router_weight_sidecar_requires_rms_root_transform_uncached_f32(
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
            const std::string row = doc.substr(obj_begin, obj_end - obj_begin);
            size_t router_begin = 0;
            size_t router_end = 0;
            if (moe_json_find_object_local(row, "router", &router_begin, &router_end) &&
                moe_router_scope_declares_rms_root_input_transform_f32(
                    doc, obj_begin + router_begin, obj_begin + router_end)) {
                return 1;
            }
            size_t semantic_begin = 0;
            size_t semantic_end = 0;
            if (moe_json_find_object_local(row, "semantic_contract", &semantic_begin, &semantic_end) &&
                moe_router_scope_declares_rms_root_input_transform_f32(
                    doc, obj_begin + semantic_begin, obj_begin + semantic_end)) {
                return 1;
            }
            size_t ops_begin = 0;
            size_t ops_end = 0;
            if (moe_json_find_array_local(row, "ops", &ops_begin, &ops_end)) {
                size_t op_cursor = obj_begin + ops_begin + 1u;
                const size_t op_limit = obj_begin + ops_end;
                while (op_cursor < op_limit) {
                    size_t op_begin = 0;
                    size_t op_end = 0;
                    if (!moe_json_next_object_in_array(doc, &op_cursor, op_limit, &op_begin, &op_end)) {
                        break;
                    }
                    std::string op_name;
                    if (moe_json_get_string_slice(doc, op_begin, op_end, "name", &op_name)) {
                        const std::string lowered = moe_lower_ascii_copy(op_name);
                        if ((lowered == "router_input" || lowered == "moe_router" || lowered == "router") &&
                            moe_router_scope_declares_rms_root_input_transform_f32(doc, op_begin, op_end)) {
                            return 1;
                        }
                    }
                }
            }
            if (moe_router_row_contract_implies_sidecar_rms_root_transform_f32(doc, obj_begin, obj_end)) {
                return 1;
            }
        }
    }
    return 0;
}

static int moe_router_weight_sidecar_requires_rms_root_transform_f32(
    const moe_pc_engine* engine,
    int32_t layer
) {
    if (!engine || layer < 0) {
        return 0;
    }
    struct router_sidecar_transform_cache_entry {
        const moe_pc_engine* engine = nullptr;
        uint64_t generation = 0;
        int32_t layer = INT32_MIN;
        uint8_t value = 0;
    };
    static thread_local std::vector<router_sidecar_transform_cache_entry> tls_cache;
    const uint64_t generation = engine->engine_generation;
    for (const router_sidecar_transform_cache_entry& entry : tls_cache) {
        if (entry.engine == engine && entry.generation == generation && entry.layer == layer) {
            return entry.value ? 1 : 0;
        }
    }
    const int value =
        moe_router_weight_sidecar_requires_rms_root_transform_uncached_f32(engine, layer);
    if (tls_cache.size() > 256u) {
        tls_cache.erase(tls_cache.begin(), tls_cache.begin() + 64);
    }
    router_sidecar_transform_cache_entry entry{};
    entry.engine = engine;
    entry.generation = generation;
    entry.layer = layer;
    entry.value = value ? 1u : 0u;
    tls_cache.push_back(entry);
    return value;
}

#include "router_scale_inputs.cpp.inc"
