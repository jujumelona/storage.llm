// Shared GraphIR/MLP helper contract.
// Included before generation_mlp.cpp.inc so mlp_common_tensors,
// mlp_normalization, mlp_forward, and generation_forward all see the same
// declarations/definitions in this single-TU build.

static int moe_env_flag_enabled_cached(const char* name) {
    if (!name || !name[0]) {
        return 0;
    }
    const char* v = std::getenv(name);
    if (!v || !v[0]) {
        return 0;
    }
    return (std::strcmp(v, "0") != 0 &&
            std::strcmp(v, "false") != 0 &&
            std::strcmp(v, "FALSE") != 0 &&
            std::strcmp(v, "off") != 0 &&
            std::strcmp(v, "OFF") != 0) ? 1 : 0;
}

static int moe_graph_ir_math_required(const moe_pc_engine* engine) {
    return engine &&
        engine->offload_graph_ir_required.load(std::memory_order_relaxed) &&
        engine->offload_graph_ir_ready.load(std::memory_order_relaxed);
}

static int moe_json_find_graph_layers_array_f32(
    const std::string& doc,
    size_t* out_begin,
    size_t* out_end
) {
    if (!out_begin || !out_end) {
        return 0;
    }
    static const char* const keys[] = {
        "layers", "blocks", "decoder_layers", "transformer_layers", "layer_plans"
    };
    for (const char* key : keys) {
        if (moe_json_find_array_local(doc, key, out_begin, out_end)) {
            return 1;
        }
    }
    return 0;
}

static int moe_json_slice_matches_layer_index_f32(
    const std::string& doc,
    size_t begin,
    size_t end,
    int32_t layer,
    uint64_t implicit_index
) {
    if (layer < 0 || begin >= end || end > doc.size()) {
        return 0;
    }
    uint64_t parsed = UINT64_MAX;
    int saw_explicit_layer_key = 0;
    static const char* const u64_keys[] = {
        "layer", "layer_index", "layer_id", "block", "block_index", "idx", "index"
    };
    for (const char* key : u64_keys) {
        if (moe_json_get_u64_slice(doc, begin, end, key, &parsed)) {
            saw_explicit_layer_key = 1;
            if (parsed == (uint64_t)layer) {
                return 1;
            }
        }
    }
    std::string text;
    const std::string want = std::to_string((long long)layer);
    static const char* const string_keys[] = {
        "layer", "layer_index", "layer_id", "block", "block_index"
    };
    for (const char* key : string_keys) {
        if (moe_json_get_string_slice(doc, begin, end, key, &text)) {
            saw_explicit_layer_key = 1;
            if (text == want) {
                return 1;
            }
        }
    }
    if (!saw_explicit_layer_key && implicit_index != UINT64_MAX &&
        implicit_index == (uint64_t)layer) {
        return 1;
    }
    return 0;
}

static int moe_map_layer_tensor_by_roles_or_suffixes(
    moe_pc_engine* engine,
    int32_t layer,
    const char* const* roles,
    size_t role_count,
    const char* const* suffixes,
    size_t suffix_count,
    moe_raw_tensor_cache_entry* out,
    int* out_declared
) {
    if (out_declared) {
        *out_declared = 0;
    }
    if (!engine || !out || layer < 0) {
        return 0;
    }
    if (roles) {
        for (size_t i = 0; i < role_count; ++i) {
            if (!roles[i] || !roles[i][0]) {
                continue;
            }
            moe_raw_tensor_desc desc{};
            const moe_gguf_common_tensor_record* rec =
                find_common_tensor_by_op_role(engine, roles[i], layer, &desc);
            if (!rec) {
                continue;
            }
            if (out_declared) {
                *out_declared = 1;
            }
            return map_common_tensor_record(engine, *rec, desc, out) ? 1 : 0;
        }
    }

    // In GraphIR-required mode, semantic math must come from op_role metadata.
    // Do not silently bind a generic suffix fallback unless the caller explicitly
    // does so outside this helper for a narrow, audited case such as shared expert
    // explicit suffix fallback.
    if (moe_graph_ir_math_required(engine)) {
        return 0;
    }
    if (!suffixes || suffix_count == 0) {
        return 0;
    }

    std::vector<std::string> names;
    moe_push_layer_tensor_candidates(&names, layer, suffixes, suffix_count);
    moe_raw_tensor_desc desc{};
    const moe_gguf_common_tensor_record* rec =
        moe_common_tensor_find_first_raw_compatible(engine, names, &desc);
    if (!rec) {
        return 0;
    }
    return map_common_tensor_record(engine, *rec, desc, out) ? 1 : 0;
}
