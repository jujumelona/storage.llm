static int moe_map_router_scale_tensor(
    moe_pc_engine* engine,
    int32_t layer,
    moe_raw_tensor_cache_entry* out
) {
    if (!engine || !out || layer < 0) {
        return 0;
    }
    // Router scale tensors are absent for most current JUJU contracts.  Without
    // a negative cache, every MoE layer probes role bindings, contract names,
    // and suffix candidates multiple times before routing, which shows up as a
    // large mlp_moe_router stall even though the router dot itself is sub-ms.
    struct router_scale_miss_cache_entry {
        const moe_pc_engine* engine = nullptr;
        uint64_t generation = 0;
        int32_t layer = INT32_MIN;
        uint64_t key = 0;
    };
    static thread_local std::vector<router_scale_miss_cache_entry> miss_cache;
    static const char* const role_names[] = {
        "moe_router_scale",
        "router_scale",
        "router_input_scale",
        "moe_router_norm",
        "router_norm",
        "router_input_norm"
    };
    static const char* const suffixes[] = {
        // Prefer direct router-input activation scales over router RMSNorm gamma
        // fallbacks.  Bare ffn_gate_inp.scale/scales are router-weight sidecars
        // and must not be mapped as activation scales.
        "router_input_scale",
        "router.input_scale",
        "mlp.router_input_scale",
        "moe.gate_input_scale",
        // Router input RMSNorm gamma fallback.  These imply a router-input
        // transform but are not bare direct activation scales.
        "router_norm.weight",
        "router.input_layernorm.weight",
        "router.input_norm.weight",
        "moe_router_norm.weight",
        "moe.gate_norm.weight",
        "mlp.router_norm.weight"
    };

    // Fix 18: Cache tensor mapping to avoid std::string vector allocation per token.
    // Use a role-specific hash so GraphIR op-role bindings and suffix fallback do
    // not alias the same cache entry.
    const uint32_t cache_hash =
        moe_router_fnv1a_hash_string("moe_router_scale|router_scale|router_norm");
    const uint64_t key = ((uint64_t)(uint32_t)layer << 32) | cache_hash | 0x80000000ull;
    const uint64_t generation = engine->engine_generation;
    for (const router_scale_miss_cache_entry& miss : miss_cache) {
        if (miss.engine == engine &&
            miss.generation == generation &&
            miss.layer == layer &&
            miss.key == key) {
            return 0;
        }
    }
    {
        std::shared_lock<std::shared_mutex> lock(engine->raw_tensor_cache_mutex);
        auto found = engine->raw_tensor_cache.find(key);
        if (found != engine->raw_tensor_cache.end() && found->second.base) {
            *out = found->second;
            return 1;
        }
    }

    moe_raw_tensor_desc desc{};
    const moe_gguf_common_tensor_record* rec = nullptr;
    for (const char* role : role_names) {
        rec = moe_find_router_scale_record_by_role_f32(engine, role, layer, &desc);
        if (rec) {
            break;
        }
    }
    if (!rec) {
        std::vector<std::string> contract_names;
        if (moe_router_contract_scale_tensor_names_f32(engine, layer, &contract_names)) {
            for (const std::string& name : contract_names) {
                const moe_gguf_common_tensor_record* by_contract =
                    moe_common_tensor_find(engine, name.c_str());
                if (!by_contract || moe_router_scale_record_is_weight_sidecar_f32(*by_contract) ||
                    !moe_common_tensor_record_to_raw_desc(*by_contract, &desc) ||
                    !moe_common_tensor_record_raw_compatible(*by_contract, desc)) {
                    continue;
                }
                rec = by_contract;
                moe_profile_log(
                    "router_scale_contract_bind",
                    "layer=%d name=%s source=layer_execution_contract_table",
                    layer, name.c_str());
                break;
            }
        }
    }
    if (!rec) {
        std::vector<std::string> names;
        moe_push_layer_tensor_candidates(&names, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]));
        rec = moe_common_tensor_find_first_raw_compatible(engine, names, &desc);
    }
    if (!rec) {
        if (miss_cache.size() > 256u) {
            miss_cache.erase(miss_cache.begin(), miss_cache.begin() + 64);
        }
        router_scale_miss_cache_entry miss{};
        miss.engine = engine;
        miss.generation = generation;
        miss.layer = layer;
        miss.key = key;
        miss_cache.push_back(miss);
        return 0;
    }
    if (moe_router_scale_record_is_weight_sidecar_f32(*rec)) {
        if (moe_profile_level() >= 2) {
            moe_profile_log(
                "router_scale_contract_reject",
                "layer=%d name=%s op_role=%s reason=weight_sidecar",
                layer, rec->name.c_str(), rec->op_role.c_str());
        }
        if (miss_cache.size() > 256u) {
            miss_cache.erase(miss_cache.begin(), miss_cache.begin() + 64);
        }
        router_scale_miss_cache_entry miss{};
        miss.engine = engine;
        miss.generation = generation;
        miss.layer = layer;
        miss.key = key;
        miss_cache.push_back(miss);
        return 0;
    }
    if (!map_common_tensor_record(engine, *rec, desc, out)) {
        if (miss_cache.size() > 256u) {
            miss_cache.erase(miss_cache.begin(), miss_cache.begin() + 64);
        }
        router_scale_miss_cache_entry miss{};
        miss.engine = engine;
        miss.generation = generation;
        miss.layer = layer;
        miss.key = key;
        miss_cache.push_back(miss);
        return 0;
    }
    {
        std::unique_lock<std::shared_mutex> lock(engine->raw_tensor_cache_mutex);
        engine->raw_tensor_cache[key] = *out;
    }
    return 1;
}

static int moe_map_router_weight_scale_sidecar_tensor(
    moe_pc_engine* engine,
    int32_t layer,
    uint32_t hidden_size,
    moe_raw_tensor_cache_entry* out
) {
    if (!engine || !out || layer < 0 || hidden_size == 0 || hidden_size > 100000) {
        return 0;
    }
    struct router_weight_sidecar_cache_entry {
        const moe_pc_engine* engine = nullptr;
        uint64_t generation = 0;
        int32_t layer = INT32_MIN;
        uint32_t hidden_size = 0;
        uint64_t key = 0;
        uint8_t found = 0;
        moe_raw_tensor_cache_entry raw{};
    };
    static thread_local std::vector<router_weight_sidecar_cache_entry> tls_cache;
    static const char* const role_names[] = {
        "router_weight_scale_sidecar",
        "moe_router_weight_scale_sidecar",
        "ffn_gate_inp_scale",
        "ffn_gate_inp_scales"
    };
    static const char* const suffixes[] = {
        "ffn_gate_inp.scale",
        "ffn_gate_inp.scales",
        "ffn_gate_inp.weight.scale",
        "ffn_gate_inp.weight.scales"
    };
    const uint32_t cache_hash = moe_router_fnv1a_hash_string("router_weight_scale_sidecar");
    const uint64_t key = ((uint64_t)(uint32_t)layer << 32) | cache_hash | 0xA0000000ull;
    const uint64_t generation = engine->engine_generation;
    for (const router_weight_sidecar_cache_entry& entry : tls_cache) {
        if (entry.engine == engine && entry.generation == generation &&
            entry.layer == layer && entry.hidden_size == hidden_size && entry.key == key) {
            if (!entry.found || !entry.raw.base) {
                return 0;
            }
            *out = entry.raw;
            return 1;
        }
    }
    {
        std::shared_lock<std::shared_mutex> lock(engine->raw_tensor_cache_mutex);
        auto found = engine->raw_tensor_cache.find(key);
        if (found != engine->raw_tensor_cache.end() && found->second.base) {
            *out = found->second;
            router_weight_sidecar_cache_entry entry{};
            entry.engine = engine;
            entry.generation = generation;
            entry.layer = layer;
            entry.hidden_size = hidden_size;
            entry.key = key;
            entry.found = 1;
            entry.raw = *out;
            if (tls_cache.size() > 256u) {
                tls_cache.erase(tls_cache.begin(), tls_cache.begin() + 64);
            }
            tls_cache.push_back(entry);
            return 1;
        }
    }
    moe_raw_tensor_desc desc{};
    const moe_gguf_common_tensor_record* rec = nullptr;
    for (const char* role : role_names) {
        rec = find_common_tensor_by_op_role(engine, role, layer, &desc);
        if (rec) {
            break;
        }
    }
    if (!rec) {
        std::vector<std::string> names;
        moe_push_layer_tensor_candidates(&names, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]));
        rec = moe_common_tensor_find_first_raw_compatible(engine, names, &desc);
    }

    moe_raw_tensor_cache_entry raw{};
    uint8_t found_ok = 0;
    if (rec && moe_router_scale_record_is_weight_sidecar_f32(*rec) &&
        map_common_tensor_record(engine, *rec, desc, &raw)) {
        const uint64_t count = moe_router_raw_scalar_count(raw);
        if (count == 1u || count == (uint64_t)hidden_size) {
            found_ok = 1;
        } else {
            moe_profile_log(
                "router_weight_sidecar_reject",
                "layer=%d name=%s count=%llu hidden=%u rows=%u cols=%u",
                layer,
                rec->name.c_str(),
                (unsigned long long)count,
                hidden_size,
                raw.rows,
                raw.cols);
        }
    }

    router_weight_sidecar_cache_entry entry{};
    entry.engine = engine;
    entry.generation = generation;
    entry.layer = layer;
    entry.hidden_size = hidden_size;
    entry.key = key;
    entry.found = found_ok;
    entry.raw = raw;
    if (tls_cache.size() > 256u) {
        tls_cache.erase(tls_cache.begin(), tls_cache.begin() + 64);
    }
    tls_cache.push_back(entry);
    if (!found_ok) {
        return 0;
    }
    {
        std::unique_lock<std::shared_mutex> lock(engine->raw_tensor_cache_mutex);
        engine->raw_tensor_cache[key] = raw;
    }
    *out = raw;
    return 1;
}

static int moe_prepare_router_weight_sidecar_input_f32(
    moe_pc_engine* engine,
    int32_t layer,
    const float* hidden,
    uint32_t hidden_size,
    std::vector<float>* out
) {
    if (!engine || !hidden || !out || hidden_size == 0 || hidden_size > 100000) {
        return 0;
    }
    moe_raw_tensor_cache_entry sidecar{};
    if (!moe_map_router_weight_scale_sidecar_tensor(engine, layer, hidden_size, &sidecar)) {
        return 0;
    }
    const uint64_t count = moe_router_raw_scalar_count(sidecar);
    if (count != 1u && count != (uint64_t)hidden_size) {
        return 0;
    }
    out->assign(hidden_size, 0.0f);
    double ss = 0.0;
    for (uint32_t i = 0; i < hidden_size; ++i) {
        const double v = (double)hidden[i];
        if (!std::isfinite(v)) {
            return 0;
        }
        ss += v * v;
    }
    if (!std::isfinite(ss)) {
        return 0;
    }
    // Gemma4 router sidecar scale is applied after a scale-free RMSNorm and
    // hidden_size^-0.5. Applying the sidecar directly to raw residuals makes
    // router logits saturate and collapses top-k probabilities to 1.0.
    const float norm =
        (1.0f / std::sqrt((float)(ss / (double)hidden_size) + 1.0e-6f)) *
        (1.0f / std::sqrt((float)hidden_size));
    if (!std::isfinite(norm) || !(norm > 0.0f)) {
        return 0;
    }
    double abs_sum = 0.0;
    double max_abs = 0.0;
    for (uint32_t i = 0; i < hidden_size; ++i) {
        float scale = 1.0f;
        if (!moe_router_raw_scalar_at_f32(sidecar, count == 1u ? 0u : (uint64_t)i, &scale) ||
            !std::isfinite(scale)) {
            return 0;
        }
        const float v = hidden[i] * norm * scale;
        if (!std::isfinite(v)) {
            return 0;
        }
        (*out)[i] = v;
        const double a = std::fabs((double)scale);
        abs_sum += a;
        max_abs = std::max(max_abs, a);
    }
    if (moe_profile_level() >= 2) {
        moe_profile_log(
            "router_weight_sidecar_apply",
            "layer=%d hidden=%u count=%llu mean_abs=%.6g max_abs=%.6g input_norm=%.6g source=weight_sidecar_rms_root",
            layer,
            hidden_size,
            (unsigned long long)count,
            abs_sum / (double)hidden_size,
            max_abs,
            (double)norm);
    }
    return 1;
}


static int moe_router_has_contract_input_scale_f32(
    moe_pc_engine* engine,
    int32_t layer,
    uint32_t hidden_size
) {
    if (!engine || layer < 0 || hidden_size == 0 || hidden_size > 100000) {
        return 0;
    }
    moe_raw_tensor_cache_entry scale_raw{};
    if (!moe_map_router_scale_tensor(engine, layer, &scale_raw)) {
        return 0;
    }
    const uint64_t scale_count = moe_router_raw_scalar_count(scale_raw);
    if (scale_count != 1u && scale_count < (uint64_t)hidden_size) {
        return 0;
    }
    return 1;
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
    // BUGFIX 599: Check hidden_size upper bound ??
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
        const double val = (double)hidden[i];
        if (!std::isfinite(val)) {
            return 0;
        }
        ss += val * val;
    }
    if (!std::isfinite(ss)) {
        return 0;
    }
    // Safe: hidden_size > 0 guaranteed by check above.  Contract-marked router
    // input scale tensors apply RMSNorm, then hidden_size^-0.5, then the
    // learned router scale.
    float norm = 1.0f / std::sqrt((float)(ss / (double)hidden_size) + 1.0e-6f);
    const int direct_input_scale = moe_router_has_direct_input_scale_contract_f32(engine, layer);
    if (direct_input_scale) {
        const float root = 1.0f / std::sqrt((float)hidden_size);
        if (!std::isfinite(root) || !(root > 0.0f)) {
            return 0;
        }
        norm *= root;
    }
    const uint64_t scale_count = moe_router_raw_scalar_count(scale_raw);
    if (scale_count != 1u && scale_count < (uint64_t)hidden_size) {
        std::fprintf(stderr,
            "[storagellm router] router scale tensor has invalid shape layer=%d count=%llu hidden=%u\n",
            layer, (unsigned long long)scale_count, hidden_size);
        return 0;
    }
    const int unit_offset = direct_input_scale
        ? 0
        : moe_engine_rmsnorm_unit_offset(engine, scale_raw);

    for (uint32_t i = 0; i < hidden_size; ++i) {
        float scale = 1.0f;
        if (scale_count == 1u) {
            if (!moe_router_raw_scalar_at_f32(scale_raw, 0, &scale)) return 0;
        } else {
            if (!moe_router_raw_scalar_at_f32(scale_raw, i, &scale)) return 0;
        }
        if (!std::isfinite(scale)) {
            return 0;
        }
        const float effective_scale = unit_offset ? (1.0f + scale) : scale;
        if (!std::isfinite(effective_scale)) {
            return 0;
        }
        // Graph contract: this tensor is a router input RMSNorm/scale weight.
        // The RMSNorm denominator is mean-square based; do not add an extra
        // hidden-size divisor here.
        const float result = hidden[i] * norm * effective_scale;
        if (!std::isfinite(result)) {
            return 0;
        }
        (*out)[i] = result;
    }
    if (moe_profile_level() >= 2) {
        moe_profile_log(
            "router_scale_apply",
            "layer=%d hidden=%u count=%llu unit_offset=%d input_norm=%.6g source=contract",
            layer,
            hidden_size,
            (unsigned long long)scale_count,
            unit_offset,
            (double)norm);
    }
    return 1;
}
