// Router top-K selection

static int moe_router_json_string_equals_any(
    const std::string& text,
    const char* const* keys,
    size_t key_count,
    const char* expected
) {
    if (text.empty() || !keys || !expected || !expected[0]) {
        return 0;
    }
    std::string value;
    for (size_t i = 0; i < key_count; ++i) {
        if (keys[i] && moe_json_get_string_local(text, keys[i], &value) &&
            moe_lower_ascii_copy(value) == expected) {
            return 1;
        }
    }
    return 0;
}

static int moe_router_json_mentions_any(
    const std::string& text,
    const char* const* needles,
    size_t needle_count
) {
    if (text.empty() || !needles) {
        return 0;
    }
    const std::string lower = moe_lower_ascii_copy(text);
    for (size_t i = 0; i < needle_count; ++i) {
        if (needles[i] && needles[i][0] && lower.find(needles[i]) != std::string::npos) {
            return 1;
        }
    }
    return 0;
}

static int moe_router_json_get_double_any(
    const std::string& text,
    const char* const* keys,
    size_t key_count,
    double* out
) {
    if (text.empty() || !keys || !out) {
        return 0;
    }
    for (size_t i = 0; i < key_count; ++i) {
        if (keys[i] && moe_json_get_double_local(text, keys[i], out)) {
            return 1;
        }
    }
    return 0;
}

static int moe_router_json_get_bool_any(
    const std::string& text,
    const char* const* keys,
    size_t key_count,
    int* out
) {
    if (text.empty() || !keys || !out) {
        return 0;
    }
    for (size_t i = 0; i < key_count; ++i) {
        if (keys[i] && moe_json_get_bool_local(text, keys[i], out)) {
            return 1;
        }
    }
    return 0;
}

static int moe_router_json_find_graph_layers_array_f32(
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

static int moe_router_json_slice_matches_layer_index_f32(
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

static int moe_router_score_value_is_sigmoid_f32(const std::string& value) {
    const std::string v = moe_lower_ascii_copy(value);
    return (v == "sigmoid" || v == "sigmoid_score" || v == "sigmoid_scores" ||
            v == "router_sigmoid" || v == "sigmoid_topk" ||
            v.find("sigmoid") != std::string::npos) ? 1 : 0;
}

static int moe_router_score_value_is_softmax_f32(const std::string& value) {
    const std::string v = moe_lower_ascii_copy(value);
    return (v == "softmax" || v == "softmax_score" || v == "softmax_scores" ||
            v == "router_softmax" || v == "softmax_topk" ||
            v.find("softmax") != std::string::npos) ? 1 : 0;
}

static int moe_router_json_string_score_any_slice_f32(
    const std::string& text,
    size_t begin,
    size_t end,
    const char* const* keys,
    size_t key_count,
    int* out_use_sigmoid
) {
    if (out_use_sigmoid) {
        *out_use_sigmoid = 0;
    }
    if (text.empty() || begin >= end || end > text.size() || !keys || !out_use_sigmoid) {
        return 0;
    }
    std::string value;
    for (size_t i = 0; i < key_count; ++i) {
        if (!keys[i]) {
            continue;
        }
        if (!moe_json_get_string_slice(text, begin, end, keys[i], &value)) {
            continue;
        }
        if (moe_router_score_value_is_sigmoid_f32(value)) {
            *out_use_sigmoid = 1;
            return 1;
        }
        if (moe_router_score_value_is_softmax_f32(value)) {
            *out_use_sigmoid = 0;
            return 1;
        }
    }
    return 0;
}

static int moe_router_json_bool_any_slice_f32(
    const std::string& text,
    size_t begin,
    size_t end,
    const char* const* keys,
    size_t key_count,
    int* out
) {
    if (!out) {
        return 0;
    }
    *out = 0;
    if (text.empty() || begin >= end || end > text.size() || !keys) {
        return 0;
    }
    for (size_t i = 0; i < key_count; ++i) {
        if (keys[i] && moe_json_get_bool_slice(text, begin, end, keys[i], out)) {
            return 1;
        }
    }
    return 0;
}

static int moe_router_json_double_any_slice_f32(
    const std::string& text,
    size_t begin,
    size_t end,
    const char* const* keys,
    size_t key_count,
    double* out
) {
    if (!out) {
        return 0;
    }
    if (text.empty() || begin >= end || end > text.size() || !keys) {
        return 0;
    }
    for (size_t i = 0; i < key_count; ++i) {
        if (keys[i] && moe_json_get_double_slice(text, begin, end, keys[i], out)) {
            return 1;
        }
    }
    return 0;
}

static int moe_router_layer_contract_object_slice_f32(
    const std::string& doc,
    int32_t layer,
    size_t* out_begin,
    size_t* out_end
) {
    if (out_begin) *out_begin = 0;
    if (out_end) *out_end = 0;
    if (doc.empty() || layer < 0 || !out_begin || !out_end) {
        return 0;
    }
    size_t arr_begin = 0;
    size_t arr_end = 0;
    if (!moe_json_find_array_local(doc, "layer_execution_contract_table", &arr_begin, &arr_end)) {
        return 0;
    }
    size_t cursor = arr_begin + 1u;
    uint64_t row_ordinal = 0;
    while (cursor < arr_end) {
        size_t obj_begin = 0;
        size_t obj_end = 0;
        if (!moe_json_next_object_in_array(doc, &cursor, arr_end, &obj_begin, &obj_end)) {
            break;
        }
        const uint64_t this_row_ordinal = row_ordinal++;
        if (moe_router_json_slice_matches_layer_index_f32(
                doc, obj_begin, obj_end, layer, this_row_ordinal)) {
            *out_begin = obj_begin;
            *out_end = obj_end;
            return 1;
        }
    }
    return 0;
}

static int moe_router_layer_contract_object_slice_cached_f32(
    const moe_pc_engine* engine,
    int32_t layer,
    size_t* out_begin,
    size_t* out_end
) {
    if (out_begin) *out_begin = 0;
    if (out_end) *out_end = 0;
    if (!engine || layer < 0 || engine->offload_graph_ir_json.empty() || !out_begin || !out_end) {
        return 0;
    }
    const size_t miss = std::numeric_limits<size_t>::max();
    {
        std::shared_lock<std::shared_mutex> lock(engine->router_layer_contract_slice_cache_mutex);
        auto it = engine->router_layer_contract_slice_cache.find(layer);
        if (it != engine->router_layer_contract_slice_cache.end()) {
            if (it->second.first == miss && it->second.second == miss) {
                return 0;
            }
            *out_begin = it->second.first;
            *out_end = it->second.second;
            return *out_begin < *out_end && *out_end <= engine->offload_graph_ir_json.size() ? 1 : 0;
        }
    }
    size_t begin = 0;
    size_t end = 0;
    const int ok = moe_router_layer_contract_object_slice_f32(
        engine->offload_graph_ir_json, layer, &begin, &end);
    {
        std::unique_lock<std::shared_mutex> lock(engine->router_layer_contract_slice_cache_mutex);
        engine->router_layer_contract_slice_cache[layer] = ok ?
            std::make_pair(begin, end) : std::make_pair(miss, miss);
    }
    if (!ok) {
        return 0;
    }
    *out_begin = begin;
    *out_end = end;
    return 1;
}


static int moe_router_layer_ops_object_slice_f32(
    const std::string& doc,
    int32_t layer,
    size_t* out_begin,
    size_t* out_end
) {
    if (out_begin) *out_begin = 0;
    if (out_end) *out_end = 0;
    if (doc.empty() || layer < 0 || !out_begin || !out_end) {
        return 0;
    }
    size_t layers_begin = 0;
    size_t layers_end = 0;
    if (!moe_router_json_find_graph_layers_array_f32(doc, &layers_begin, &layers_end)) {
        return 0;
    }
    size_t cursor = layers_begin + 1u;
    uint64_t layer_ordinal = 0;
    while (cursor < layers_end) {
        size_t layer_obj_begin = 0;
        size_t layer_obj_end = 0;
        if (!moe_json_next_object_in_array(doc, &cursor, layers_end, &layer_obj_begin, &layer_obj_end)) {
            break;
        }
        const uint64_t this_layer_ordinal = layer_ordinal++;
        if (!moe_router_json_slice_matches_layer_index_f32(
                doc, layer_obj_begin, layer_obj_end, layer, this_layer_ordinal)) {
            continue;
        }
        size_t ops_begin_local = 0;
        size_t ops_end_local = 0;
        const std::string layer_slice = doc.substr(layer_obj_begin, layer_obj_end - layer_obj_begin);
        if (!moe_json_find_array_local(layer_slice, "ops", &ops_begin_local, &ops_end_local)) {
            return 0;
        }
        *out_begin = layer_obj_begin + ops_begin_local;
        *out_end = layer_obj_begin + ops_end_local;
        return 1;
    }
    return 0;
}

static int moe_router_op_slice_is_router_f32(
    const std::string& doc,
    size_t begin,
    size_t end
) {
    static const char* const router_tokens[] = {
        "moe_router", "expert_select", "router_topk", "router", "router_input"
    };
    static const char* const fields[] = {
        "op", "name", "execution_op", "op_role", "role", "graph_role",
        "graph_role_text", "bundle_member_role", "type"
    };
    std::string value;
    std::string merged;
    for (const char* field : fields) {
        if (moe_json_get_string_slice(doc, begin, end, field, &value)) {
            merged += " ";
            merged += moe_lower_ascii_copy(value);
        }
    }
    for (const char* token : router_tokens) {
        if (token && merged.find(token) != std::string::npos) {
            return 1;
        }
    }
    return 0;
}

static int moe_router_layer_op_score_contract_f32(
    const std::string& doc,
    int32_t layer,
    const char* const* score_keys,
    size_t score_key_count,
    int* out_use_sigmoid,
    const char** out_source
) {
    if (out_use_sigmoid) *out_use_sigmoid = 0;
    if (out_source) *out_source = "none";
    size_t ops_begin = 0;
    size_t ops_end = 0;
    if (!moe_router_layer_ops_object_slice_f32(doc, layer, &ops_begin, &ops_end)) {
        return 0;
    }
    size_t cursor = ops_begin + 1u;
    while (cursor < ops_end) {
        size_t op_begin = 0;
        size_t op_end = 0;
        if (!moe_json_next_object_in_array(doc, &cursor, ops_end, &op_begin, &op_end)) {
            break;
        }
        if (!moe_router_op_slice_is_router_f32(doc, op_begin, op_end)) {
            continue;
        }
        int use_sigmoid = 0;
        if (moe_router_json_string_score_any_slice_f32(
                doc, op_begin, op_end, score_keys, score_key_count, &use_sigmoid)) {
            if (out_use_sigmoid) *out_use_sigmoid = use_sigmoid;
            if (out_source) *out_source = "graph_ir_layer_op";
            return 1;
        }
    }
    return 0;
}

static int moe_router_layer_score_contract_f32(
    const moe_pc_engine* engine,
    int32_t layer,
    int* out_use_sigmoid,
    const char** out_source
) {
    if (out_use_sigmoid) *out_use_sigmoid = 0;
    if (out_source) *out_source = "default";
    if (!engine) {
        return 0;
    }
    static const char* const score_keys[] = {
        "scoring_func",
        "score_func",
        "router_score_func",
        "router_scoring_func",
        "router_score_function",
        "router_scoring_function",
        "router_score_activation",
        "router_activation",
        "routing_score_func",
        "routing_score_function",
        "routing_score_activation",
        "moe_router_score_func",
        "moe_router_scoring_func",
        "moe_router_score_function",
        "moe_router_score_activation",
        "moe_score_func",
        "moe_scoring_func",
        "expert_score_func",
        "expert_scoring_func",
        "score_activation",
        "score_type",
        "score_fn",
        "router_score",
        "router_score_type",
        "router_logits_activation",
        "router_logits_score_func",
        "router_probs_activation",
        "expert_score_type"
    };
    int has_layer_contract_row = 0;
    if (layer >= 0 && !engine->offload_graph_ir_json.empty()) {
        size_t obj_begin = 0;
        size_t obj_end = 0;
        if (moe_router_layer_contract_object_slice_cached_f32(engine, layer, &obj_begin, &obj_end)) {
            has_layer_contract_row = 1;
            int use_sigmoid = 0;
            if (moe_router_json_string_score_any_slice_f32(
                    engine->offload_graph_ir_json, obj_begin, obj_end,
                    score_keys, sizeof(score_keys) / sizeof(score_keys[0]), &use_sigmoid)) {
                if (out_use_sigmoid) *out_use_sigmoid = use_sigmoid;
                if (out_source) *out_source = "layer_execution_contract_table";
                return 1;
            }
        }
        if (!has_layer_contract_row) {
            int use_sigmoid = 0;
            const char* source = "none";
            if (moe_router_layer_op_score_contract_f32(
                    engine->offload_graph_ir_json, layer,
                    score_keys, sizeof(score_keys) / sizeof(score_keys[0]),
                    &use_sigmoid, &source)) {
                if (out_use_sigmoid) *out_use_sigmoid = use_sigmoid;
                if (out_source) *out_source = source;
                return 1;
            }
        }
    }
    if (moe_router_json_string_equals_any(
            engine->offload_gguf_metadata_json, score_keys,
            sizeof(score_keys) / sizeof(score_keys[0]), "sigmoid")) {
        if (out_use_sigmoid) *out_use_sigmoid = 1;
        if (out_source) *out_source = "metadata";
        return 1;
    }
    if (moe_router_json_string_equals_any(
            engine->offload_gguf_metadata_json, score_keys,
            sizeof(score_keys) / sizeof(score_keys[0]), "softmax")) {
        if (out_use_sigmoid) *out_use_sigmoid = 0;
        if (out_source) *out_source = "metadata";
        return 1;
    }
    if (has_layer_contract_row) {
        return 0;
    }
    const std::string* graph_ir_doc_for_score = &engine->offload_graph_ir_json;
    if (moe_router_json_string_equals_any(
            *graph_ir_doc_for_score, score_keys,
            sizeof(score_keys) / sizeof(score_keys[0]), "sigmoid")) {
        if (out_use_sigmoid) *out_use_sigmoid = 1;
        if (out_source) *out_source = "graph_ir";
        return 1;
    }
    if (moe_router_json_string_equals_any(
            *graph_ir_doc_for_score, score_keys,
            sizeof(score_keys) / sizeof(score_keys[0]), "softmax")) {
        if (out_use_sigmoid) *out_use_sigmoid = 0;
        if (out_source) *out_source = "graph_ir";
        return 1;
    }
    static const char* const sigmoid_needles[] = {
        "\"scoring_func\":\"sigmoid\"",
        "\"scoring_func\": \"sigmoid\"",
        "\"router_score_func\":\"sigmoid\"",
        "\"router_score_func\": \"sigmoid\"",
        "\"router_scoring_func\":\"sigmoid\"",
        "\"router_scoring_func\": \"sigmoid\"",
        "\"router_score_function\":\"sigmoid\"",
        "\"router_score_function\": \"sigmoid\"",
        "\"router_score_activation\":\"sigmoid\"",
        "\"router_score_activation\": \"sigmoid\"",
        "\"routing_score_func\":\"sigmoid\"",
        "\"routing_score_func\": \"sigmoid\"",
        "\"routing_score_function\":\"sigmoid\"",
        "\"routing_score_function\": \"sigmoid\"",
        "\"score_activation\":\"sigmoid\"",
        "\"score_activation\": \"sigmoid\"",
        "\"score_fn\":\"sigmoid\"",
        "\"score_fn\": \"sigmoid\""
    };
    if (moe_router_json_mentions_any(
            engine->offload_gguf_metadata_json, sigmoid_needles,
            sizeof(sigmoid_needles) / sizeof(sigmoid_needles[0]))) {
        if (out_use_sigmoid) *out_use_sigmoid = 1;
        if (out_source) *out_source = "metadata";
        return 1;
    }
    if (moe_router_json_mentions_any(
            *graph_ir_doc_for_score, sigmoid_needles,
            sizeof(sigmoid_needles) / sizeof(sigmoid_needles[0]))) {
        if (out_use_sigmoid) *out_use_sigmoid = 1;
        if (out_source) *out_source = "graph_ir";
        return 1;
    }
    return 0;
}

static int moe_router_uses_sigmoid_scores_for_layer(const moe_pc_engine* engine, int32_t layer) {
    if (!engine) {
        return 0;
    }
    int use_sigmoid = 0;
    const char* source = "default";
    if (moe_router_layer_score_contract_f32(engine, layer, &use_sigmoid, &source)) {
        moe_profile_log(
            "router_score_contract",
            "layer=%d use_sigmoid=%d source=%s",
            layer, use_sigmoid, source ? source : "unknown");
        return use_sigmoid ? 1 : 0;
    }
    if (layer < 0) {
        const int cached = engine->cached_router_use_sigmoid.load(std::memory_order_acquire);
        if (cached >= 0) {
            return cached;
        }
        engine->cached_router_use_sigmoid.store(0, std::memory_order_release);
    }
    moe_profile_log("router_score_contract", "layer=%d use_sigmoid=0 source=default", layer);
    return 0;
}

static int moe_router_uses_sigmoid_scores(const moe_pc_engine* engine) {
    return moe_router_uses_sigmoid_scores_for_layer(engine, -1);
}

static int moe_router_layer_u32_contract_f32(
    const moe_pc_engine* engine,
    int32_t layer,
    const char* const* keys,
    size_t key_count,
    uint32_t* out,
    const char** out_source
) {
    if (out_source) *out_source = "none";
    if (!engine || !keys || !out) {
        return 0;
    }
    uint64_t value = 0;
    int has_layer_contract_row = 0;
    if (layer >= 0 && !engine->offload_graph_ir_json.empty()) {
        size_t obj_begin = 0;
        size_t obj_end = 0;
        if (moe_router_layer_contract_object_slice_cached_f32(engine, layer, &obj_begin, &obj_end)) {
            has_layer_contract_row = 1;
            for (size_t i = 0; i < key_count; ++i) {
                if (keys[i] && moe_json_get_u64_slice(engine->offload_graph_ir_json, obj_begin, obj_end, keys[i], &value) && value > 0) {
                    *out = (uint32_t)std::min<uint64_t>(value, UINT32_MAX);
                    if (out_source) *out_source = "layer_execution_contract_table";
                    return 1;
                }
            }
        }
        size_t ops_begin = 0;
        size_t ops_end = 0;
        if (!has_layer_contract_row && moe_router_layer_ops_object_slice_f32(engine->offload_graph_ir_json, layer, &ops_begin, &ops_end)) {
            size_t cursor = ops_begin + 1u;
            while (cursor < ops_end) {
                size_t op_begin = 0;
                size_t op_end = 0;
                if (!moe_json_next_object_in_array(engine->offload_graph_ir_json, &cursor, ops_end, &op_begin, &op_end)) {
                    break;
                }
                if (!moe_router_op_slice_is_router_f32(engine->offload_graph_ir_json, op_begin, op_end)) {
                    continue;
                }
                for (size_t i = 0; i < key_count; ++i) {
                    if (keys[i] && moe_json_get_u64_slice(engine->offload_graph_ir_json, op_begin, op_end, keys[i], &value) && value > 0) {
                        *out = (uint32_t)std::min<uint64_t>(value, UINT32_MAX);
                        if (out_source) *out_source = "graph_ir_layer_op";
                        return 1;
                    }
                }
            }
        }
    }
    for (size_t i = 0; i < key_count; ++i) {
        if (keys[i] && moe_json_get_u64_local(engine->offload_gguf_metadata_json, keys[i], &value) && value > 0) {
            *out = (uint32_t)std::min<uint64_t>(value, UINT32_MAX);
            if (out_source) *out_source = "metadata";
            return 1;
        }
    }
    if (has_layer_contract_row) {
        return 0;
    }
    for (size_t i = 0; i < key_count; ++i) {
        if (keys[i] && moe_json_get_u64_local(engine->offload_graph_ir_json, keys[i], &value) && value > 0) {
            *out = (uint32_t)std::min<uint64_t>(value, UINT32_MAX);
            if (out_source) *out_source = "graph_ir";
            return 1;
        }
    }
    return 0;
}

static uint32_t moe_router_top_k_from_metadata_for_layer(
    const moe_pc_engine* engine,
    int32_t layer,
    uint32_t fallback,
    uint32_t experts
) {
    static const char* const topk_keys[] = {
        "num_experts_per_tok",
        "n_experts_per_tok",
        "routed_experts_per_token",
        "expert_used_count",
        "num_experts_per_token",
        "moe_top_k",
        "top_k_experts",
        "num_selected_experts",
        "topk",
        "top_k"
    };
    uint32_t parsed = 0;
    const char* source = "default";
    if (moe_router_layer_u32_contract_f32(
            engine, layer, topk_keys, sizeof(topk_keys) / sizeof(topk_keys[0]),
            &parsed, &source) && parsed > 0) {
        const uint32_t limited = std::min<uint32_t>(parsed, experts ? experts : parsed);
        moe_profile_log(
            "router_topk_contract",
            "layer=%d top_k=%u source=%s experts=%u",
            layer, limited, source ? source : "unknown", experts);
        return limited;
    }
    int cached = engine ? engine->cached_router_top_k_hint.load(std::memory_order_acquire) : -1;
    if (cached < 0) {
        cached = 0;
        if (engine) {
            engine->cached_router_top_k_hint.store(cached, std::memory_order_release);
        }
    }
    if (cached > 0) {
        return (uint32_t)std::min<uint32_t>((uint32_t)cached, experts ? experts : (uint32_t)cached);
    }
    return std::min(fallback, experts ? experts : fallback);
}

static uint32_t moe_router_top_k_from_metadata(
    const moe_pc_engine* engine,
    uint32_t fallback,
    uint32_t experts
) {
    return moe_router_top_k_from_metadata_for_layer(engine, -1, fallback, experts);
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

static int moe_router_layer_double_contract_f32(
    const moe_pc_engine* engine,
    int32_t layer,
    const char* const* keys,
    size_t key_count,
    double* out,
    const char** out_source
) {
    if (out_source) *out_source = "none";
    if (!engine || !keys || !out) {
        return 0;
    }
    int has_layer_contract_row = 0;
    if (layer >= 0 && !engine->offload_graph_ir_json.empty()) {
        size_t obj_begin = 0;
        size_t obj_end = 0;
        if (moe_router_layer_contract_object_slice_cached_f32(engine, layer, &obj_begin, &obj_end)) {
            has_layer_contract_row = 1;
            if (moe_router_json_double_any_slice_f32(
                    engine->offload_graph_ir_json, obj_begin, obj_end, keys, key_count, out)) {
                if (out_source) *out_source = "layer_execution_contract_table";
                return 1;
            }
        }
        size_t ops_begin = 0;
        size_t ops_end = 0;
        if (!has_layer_contract_row && moe_router_layer_ops_object_slice_f32(engine->offload_graph_ir_json, layer, &ops_begin, &ops_end)) {
            size_t cursor = ops_begin + 1u;
            while (cursor < ops_end) {
                size_t op_begin = 0;
                size_t op_end = 0;
                if (!moe_json_next_object_in_array(engine->offload_graph_ir_json, &cursor, ops_end, &op_begin, &op_end)) {
                    break;
                }
                if (!moe_router_op_slice_is_router_f32(engine->offload_graph_ir_json, op_begin, op_end)) {
                    continue;
                }
                if (moe_router_json_double_any_slice_f32(
                        engine->offload_graph_ir_json, op_begin, op_end, keys, key_count, out)) {
                    if (out_source) *out_source = "graph_ir_layer_op";
                    return 1;
                }
            }
        }
    }
    if (moe_router_json_get_double_any(engine->offload_gguf_metadata_json, keys, key_count, out)) {
        if (out_source) *out_source = "metadata";
        return 1;
    }
    if (!has_layer_contract_row && moe_router_json_get_double_any(engine->offload_graph_ir_json, keys, key_count, out)) {
        if (out_source) *out_source = "graph_ir";
        return 1;
    }
    return 0;
}

static int moe_router_layer_bool_contract_f32(
    const moe_pc_engine* engine,
    int32_t layer,
    const char* const* keys,
    size_t key_count,
    int* out,
    const char** out_source
) {
    if (out_source) *out_source = "none";
    if (!engine || !keys || !out) {
        return 0;
    }
    int has_layer_contract_row = 0;
    if (layer >= 0 && !engine->offload_graph_ir_json.empty()) {
        size_t obj_begin = 0;
        size_t obj_end = 0;
        if (moe_router_layer_contract_object_slice_cached_f32(engine, layer, &obj_begin, &obj_end)) {
            has_layer_contract_row = 1;
            if (moe_router_json_bool_any_slice_f32(
                    engine->offload_graph_ir_json, obj_begin, obj_end, keys, key_count, out)) {
                if (out_source) *out_source = "layer_execution_contract_table";
                return 1;
            }
        }
        size_t ops_begin = 0;
        size_t ops_end = 0;
        if (!has_layer_contract_row && moe_router_layer_ops_object_slice_f32(engine->offload_graph_ir_json, layer, &ops_begin, &ops_end)) {
            size_t cursor = ops_begin + 1u;
            while (cursor < ops_end) {
                size_t op_begin = 0;
                size_t op_end = 0;
                if (!moe_json_next_object_in_array(engine->offload_graph_ir_json, &cursor, ops_end, &op_begin, &op_end)) {
                    break;
                }
                if (!moe_router_op_slice_is_router_f32(engine->offload_graph_ir_json, op_begin, op_end)) {
                    continue;
                }
                if (moe_router_json_bool_any_slice_f32(
                        engine->offload_graph_ir_json, op_begin, op_end, keys, key_count, out)) {
                    if (out_source) *out_source = "graph_ir_layer_op";
                    return 1;
                }
            }
        }
    }
    if (moe_router_json_get_bool_any(engine->offload_gguf_metadata_json, keys, key_count, out)) {
        if (out_source) *out_source = "metadata";
        return 1;
    }
    if (!has_layer_contract_row && moe_router_json_get_bool_any(engine->offload_graph_ir_json, keys, key_count, out)) {
        if (out_source) *out_source = "graph_ir";
        return 1;
    }
    return 0;
}

static float moe_router_routed_scaling_factor_for_layer(const moe_pc_engine* engine, int32_t layer) {
    if (!engine) {
        return 1.0f;
    }
    double value = 1.0;
    // Contract: this value is the routed-expert output multiplier for
    // sigmoid-score MoE routers.  It is not an activation/tensor scale, and it
    // must not be applied to softmax routers whose selected top-k weights are
    // already probability-normalized.  Do not accept generic names such as
    // "router_scale", "expert_scale", "route_scale" or "routed_scale": old
    // JUJU/GraphIR uses those names for tensor-role counts/bindings and
    // per-weight sidecars.
    static const char* const scale_keys[] = {
        "routed_scaling_factor",
        "moe_routed_scaling_factor",
        "router_routed_scaling_factor",
        "expert_routed_scaling_factor"
    };
    const char* source = "default";
    if (moe_router_layer_double_contract_f32(
            engine, layer, scale_keys, sizeof(scale_keys) / sizeof(scale_keys[0]),
            &value, &source) &&
        value > 0.0 && std::isfinite(value)) {
        moe_profile_log(
            "router_routed_scale_contract",
            "layer=%d scale=%.9g source=%s",
            layer, value, source ? source : "unknown");
        return (float)value;
    }
    if (layer < 0) {
        const uint32_t cached_bits = engine->cached_router_route_scale_bits.load(std::memory_order_acquire);
        if (cached_bits != 0u) {
            return moe_f32_from_bits(cached_bits);
        }
        engine->cached_router_route_scale_bits.store(moe_f32_bits(1.0f), std::memory_order_release);
    }
    return 1.0f;
}

static float moe_router_routed_scaling_factor(const moe_pc_engine* engine) {
    return moe_router_routed_scaling_factor_for_layer(engine, -1);
}

static float moe_router_effective_routed_scaling_factor_for_layer(
    const moe_pc_engine* engine,
    int32_t layer,
    int use_sigmoid
) {
    if (!use_sigmoid) {
        // Softmax routers produce a normalized probability simplex.  Multiplying
        // the normalized top-k weights by routed_scaling_factor makes
        // norm_topk=1 produce weights greater than 1.0 and diverges from the
        // adaptive router path, which only applies routed scaling to sigmoid
        // routers.
        return 1.0f;
    }
    return moe_router_routed_scaling_factor_for_layer(engine, layer);
}

static float moe_router_effective_routed_scaling_factor(const moe_pc_engine* engine, int use_sigmoid) {
    return moe_router_effective_routed_scaling_factor_for_layer(engine, -1, use_sigmoid);
}

static int moe_router_norm_topk_prob_for_layer(
    const moe_pc_engine* engine,
    int32_t layer,
    int use_sigmoid
) {
    if (!engine) {
        return use_sigmoid ? 1 : 0;
    }
    int value = 0;
    int result = use_sigmoid ? 1 : 0;
    static const char* const norm_keys[] = {
        "norm_topk_prob",
        "normalize_topk_prob",
        "moe_normalize_topk_prob",
        "moe_normalize_expert_weights",
        "router_normalize_topk_prob",
        "router_norm_topk_prob",
        "routed_norm_topk_prob",
        "routed_normalize_topk_prob"
    };
    const char* source = "default";
    if (moe_router_layer_bool_contract_f32(
            engine, layer, norm_keys, sizeof(norm_keys) / sizeof(norm_keys[0]),
            &value, &source)) {
        result = value ? 1 : 0;
        moe_profile_log(
            "router_norm_topk_contract",
            "layer=%d norm_topk=%d source=%s",
            layer, result, source ? source : "unknown");
        return result;
    }
    if (!use_sigmoid && moe_engine_uses_hybrid_local_global_attention(engine)) {
        result = 1;
    }
    if (layer < 0) {
        const int cached = engine->cached_router_norm_topk_prob.load(std::memory_order_acquire);
        if (cached >= 0) {
            return cached;
        }
        engine->cached_router_norm_topk_prob.store(result, std::memory_order_release);
    }
    moe_profile_log("router_norm_topk_contract", "layer=%d norm_topk=%d source=default", layer, result);
    return result;
}

static int moe_router_norm_topk_prob(const moe_pc_engine* engine, int use_sigmoid) {
    return moe_router_norm_topk_prob_for_layer(engine, -1, use_sigmoid);
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
    // BUGFIX 668: Remove duplicate limited_groups == 0 check
    if (groups <= 1 || limited_groups == 0 || limited_groups >= groups || experts % groups != 0) {
        return;
    }

    const uint32_t experts_per_group = experts / groups;
    // CPU-only hot path: group counts are small for routed MoE, so avoid
    // allocating three vectors every router call.  Fall back only for very
    // unusual models with huge group counts.
    float group_scores_stack[512];
    uint32_t group_order_stack[512];
    uint8_t keep_stack[512];
    std::vector<float> group_scores_heap;
    std::vector<uint32_t> group_order_heap;
    std::vector<uint8_t> keep_heap;
    float* group_scores = group_scores_stack;
    uint32_t* group_order = group_order_stack;
    uint8_t* keep = keep_stack;
    if (groups > 512u) {
        group_scores_heap.assign(groups, 0.0f);
        group_order_heap.resize(groups);
        keep_heap.assign(groups, 0u);
        group_scores = group_scores_heap.data();
        group_order = group_order_heap.data();
        keep = keep_heap.data();
    } else {
        std::memset(keep_stack, 0, groups * sizeof(uint8_t));
    }
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
    std::nth_element(group_order, group_order + limited_groups, group_order + groups, group_cmp);
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
            // Group-limited routers use this tensor as an execution mask, not as
            // a probability floor.  0.0 is a valid sigmoid score and can re-enter
            // top-k on ties or very small kept scores, so push masked experts
            // below any real router probability.
            (*scores)[e] = -std::numeric_limits<float>::max();
        }
    }
}


// CPU-storage router fast path: top_k is tiny for MoE decode (usually 2..8),
// while expert count can be 64..256+.  Do not allocate/fill/sort a full order
// vector in the hot path.  Keep a sorted fixed-size top-k window instead.
// This removes per-layer scratch_order writes and nth_element/sort overhead.
static int moe_router_select_topk_small_f32(
    const std::vector<float>& scores,
    uint32_t experts,
    uint32_t top_k,
    std::vector<int>* out_experts,
    std::vector<float>* out_weights,
    float* out_denom
) {
    if (!out_experts || !out_weights || !out_denom || top_k == 0 || experts == 0 || scores.size() < experts) {
        return 0;
    }
    if (top_k > 64u) {
        return 0;
    }
    uint32_t ids[64];
    float vals[64];
    uint32_t count = 0;
    for (uint32_t e = 0; e < experts; ++e) {
        const float v = scores[e];
        if (!std::isfinite(v)) {
            return 0;
        }
        uint32_t pos = count;
        while (pos > 0) {
            const float pv = vals[pos - 1u];
            const uint32_t pid = ids[pos - 1u];
            if (pv > v || (pv == v && pid < e)) {
                break;
            }
            --pos;
        }
        if (pos >= top_k) {
            continue;
        }
        if (count < top_k) {
            ++count;
        }
        for (uint32_t j = count - 1u; j > pos; --j) {
            vals[j] = vals[j - 1u];
            ids[j] = ids[j - 1u];
        }
        vals[pos] = v;
        ids[pos] = e;
    }
    if (count != top_k) {
        return 0;
    }
    out_experts->resize(top_k);
    out_weights->resize(top_k);
    float denom = 0.0f;
    for (uint32_t i = 0; i < top_k; ++i) {
        (*out_experts)[i] = (int)ids[i];
        (*out_weights)[i] = vals[i];
        denom += vals[i];
    }
    *out_denom = denom;
    return 1;
}

// CPU-storage softmax router shortcut.  For routers that normalize the selected
// top-k probabilities (norm_topk contract), full softmax over all
// experts is mathematically unnecessary: top-k is monotonic in logits and the
// final normalized weights are exp(selected_logits) / sum(exp(selected_logits)).
// This removes per-layer exp() over every expert and avoids scratch_order.
static int moe_router_select_topk_logits_norm_softmax_f32(
    const std::vector<float>& logits,
    uint32_t experts,
    uint32_t top_k,
    std::vector<int>* out_experts,
    std::vector<float>* out_weights
) {
    if (!out_experts || !out_weights || top_k == 0 || experts == 0 || logits.size() < experts) {
        return 0;
    }
    if (top_k > 64u) {
        return 0;
    }
    uint32_t ids[64];
    float vals[64];
    uint32_t count = 0;
    for (uint32_t e = 0; e < experts; ++e) {
        const float v = logits[e];
        if (!std::isfinite(v)) {
            return 0;
        }
        uint32_t pos = count;
        while (pos > 0) {
            const float pv = vals[pos - 1u];
            const uint32_t pid = ids[pos - 1u];
            if (pv > v || (pv == v && pid < e)) {
                break;
            }
            --pos;
        }
        if (pos >= top_k) {
            continue;
        }
        if (count < top_k) {
            ++count;
        }
        for (uint32_t j = count - 1u; j > pos; --j) {
            vals[j] = vals[j - 1u];
            ids[j] = ids[j - 1u];
        }
        vals[pos] = v;
        ids[pos] = e;
    }
    if (count != top_k) {
        return 0;
    }
    const float max_selected = vals[0];
    double denom = 0.0;
    float tmp_weights[64];
    for (uint32_t i = 0; i < top_k; ++i) {
        const float ev = (float)std::exp((double)vals[i] - (double)max_selected);
        if (!std::isfinite(ev)) {
            return 0;
        }
        tmp_weights[i] = ev;
        denom += (double)ev;
    }
    if (!(denom > 0.0) || !std::isfinite(denom)) {
        return 0;
    }
    const float inv = (float)(1.0 / denom);
    out_experts->resize(top_k);
    out_weights->resize(top_k);
    for (uint32_t i = 0; i < top_k; ++i) {
        (*out_experts)[i] = (int)ids[i];
        (*out_weights)[i] = tmp_weights[i] * inv;
    }
    return 1;
}


// Sigmoid routers are monotonic in the raw router logits.  Selecting top-k after
// materializing sigmoid(logit) is numerically wrong for large positive logits:
// sigmoid() saturates to exactly 1.0f, causing ties and deterministic low expert
// ids even when the raw logits are distinct.  Keep selection in raw-logit space,
// then apply sigmoid only to the selected experts for the final routing weights.
static int moe_router_select_topk_logits_sigmoid_f32(
    const std::vector<float>& logits,
    uint32_t experts,
    uint32_t top_k,
    std::vector<int>* out_experts,
    std::vector<float>* out_weights,
    float* out_denom
) {
    if (!out_experts || !out_weights || !out_denom || top_k == 0 || experts == 0 || logits.size() < experts) {
        return 0;
    }
    if (top_k > 64u) {
        return 0;
    }
    uint32_t ids[64];
    float vals[64];
    uint32_t count = 0;
    for (uint32_t e = 0; e < experts; ++e) {
        const float v = logits[e];
        if (!std::isfinite(v)) {
            return 0;
        }
        uint32_t pos = count;
        while (pos > 0) {
            const float pv = vals[pos - 1u];
            const uint32_t pid = ids[pos - 1u];
            if (pv > v || (pv == v && pid < e)) {
                break;
            }
            --pos;
        }
        if (pos >= top_k) {
            continue;
        }
        if (count < top_k) {
            ++count;
        }
        for (uint32_t j = count - 1u; j > pos; --j) {
            vals[j] = vals[j - 1u];
            ids[j] = ids[j - 1u];
        }
        vals[pos] = v;
        ids[pos] = e;
    }
    if (count != top_k) {
        return 0;
    }
    out_experts->resize(top_k);
    out_weights->resize(top_k);
    float denom = 0.0f;
    for (uint32_t i = 0; i < top_k; ++i) {
        const float w = moe_router_sigmoid(vals[i]);
        if (!std::isfinite(w)) {
            return 0;
        }
        (*out_experts)[i] = (int)ids[i];
        (*out_weights)[i] = w;
        denom += w;
    }
    *out_denom = denom;
    return std::isfinite(denom) && denom > 0.0f ? 1 : 0;
}

static float moe_router_normalized_entropy_from_raw_sigmoid_logits(
    const std::vector<float>& logits,
    uint32_t experts
) {
    if (experts <= 1u || logits.size() < experts) {
        return 0.0f;
    }
    double sum = 0.0;
    for (uint32_t i = 0; i < experts; ++i) {
        const float p = moe_router_sigmoid(logits[i]);
        if (p > 0.0f && std::isfinite(p)) {
            sum += (double)p;
        }
    }
    if (!(sum > 0.0) || !std::isfinite(sum)) {
        return 0.0f;
    }
    double entropy = 0.0;
    for (uint32_t i = 0; i < experts; ++i) {
        const float p = moe_router_sigmoid(logits[i]);
        if (p <= 0.0f || !std::isfinite(p)) {
            continue;
        }
        const double q = (double)p / sum;
        entropy -= q * (double)fast_log2f((float)q);
    }
    const double denom = (double)fast_log2f((float)experts);
    if (!(denom > 0.0) || !std::isfinite(denom)) {
        return 0.0f;
    }
    return (float)std::max(0.0, std::min(1.0, entropy / denom));
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
    const int router_scale_declared = moe_engine_contract_has_router_input_scale(engine, layer);
    if (moe_prepare_scaled_router_input_f32(engine, layer, hidden, hidden_size, &scaled_router_input)) {
        router_input = scaled_router_input.data();
    } else if (router_scale_declared) {
        std::fprintf(stderr,
            "[storagellm router] declared router input scale could not be applied layer=%d hidden=%u\n",
            layer, hidden_size);
        return 0;
    }
    if (!moe_router_logits_f32(engine, layer, router_input, hidden_size, experts, scratch_logits)) {
        return 0;
    }
    if (!moe_router_apply_bias_f32(engine, layer, experts, scratch_logits)) {
        return 0;
    }

    top_k = std::min(top_k, experts);

    const int use_sigmoid = moe_router_uses_sigmoid_scores_for_layer(engine, layer);
    const int cpu_storage_router_shortcut =
        engine->cpu_storage_engine_ready.load(std::memory_order_relaxed) ||
        engine->cpu_only_runtime.load(std::memory_order_relaxed);
    const int norm_topk_pref = moe_router_norm_topk_prob_for_layer(engine, layer, use_sigmoid);
    if (!use_sigmoid && norm_topk_pref && cpu_storage_router_shortcut && top_k <= 64u &&
        moe_router_select_topk_logits_norm_softmax_f32(*scratch_logits, experts, top_k, out_experts, out_weights)) {
        const float route_scale = moe_router_effective_routed_scaling_factor_for_layer(engine, layer, use_sigmoid);
        if (route_scale != 1.0f) {
            for (float& w : *out_weights) {
                w *= route_scale;
            }
        }
        if (session) {
            session->session_router_entropy_milli = 500u;
        }
        engine->last_router_entropy_milli.store(500u, std::memory_order_relaxed);
        moe_profile_log(
            "router_topk",
            "layer=%d experts=%u top_k=%u use_sigmoid=0 norm_topk=1 entropy=0.500 scale=%.6g first_expert=%d first_weight=%.6g fast_logits_topk=1",
            layer, experts, top_k, (double)route_scale,
            out_experts->empty() ? -1 : (*out_experts)[0],
            out_weights->empty() ? 0.0 : (double)(*out_weights)[0]);
        return 1;
    }
    float raw_min_logit = std::numeric_limits<float>::infinity();
    float raw_max_logit = -std::numeric_limits<float>::infinity();
    for (float v : *scratch_logits) {
        if (!std::isfinite(v)) {
            return 0;
        }
        if (v < raw_min_logit) raw_min_logit = v;
        if (v > raw_max_logit) raw_max_logit = v;
    }
    if (use_sigmoid) {
        // Mask group-limited experts in raw-logit space.  Sigmoid is monotonic,
        // so the selected experts are unchanged versus ideal real-valued sigmoid
        // scores, but we avoid float32 saturation to exactly 1.0f before top-k.
        moe_router_apply_group_limit(engine, scratch_logits, experts);
    } else {
        // BUGFIX 665: Check for NaN/Inf before softmax computation ★★★
        // BUGFIX 667: Prevent NaN propagation in router logits ★★
        const float max_logit = raw_max_logit;
        if (!std::isfinite(max_logit)) {
            return 0;
        } else {
            double exp_denom = 0.0;
            // CPU hot path: normalize in-place.  The previous code allocated an
            // exp_values vector every router call, which is pure overhead in decode.
            for (size_t i = 0; i < scratch_logits->size(); ++i) {
                const float ev = (float)std::exp((double)(*scratch_logits)[i] - (double)max_logit);
                (*scratch_logits)[i] = ev;
                exp_denom += (double)ev;
            }
            if (!(exp_denom > 0.0) || !std::isfinite(exp_denom)) {
                return 0;
            } else {
                const float inv = (float)(1.0 / exp_denom);
                for (float& v : *scratch_logits) {
                    v *= inv;
                }
            }
        }
    }

    float router_entropy = 0.5f;
    if (use_sigmoid) {
        router_entropy = moe_router_normalized_entropy_from_raw_sigmoid_logits(*scratch_logits, experts);
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

    float denom = 0.0f;
    const int cpu_storage_router_fast =
        engine->cpu_storage_engine_ready.load(std::memory_order_relaxed) ||
        engine->cpu_only_runtime.load(std::memory_order_relaxed);
    if (use_sigmoid && cpu_storage_router_fast && top_k <= 64u &&
        moe_router_select_topk_logits_sigmoid_f32(*scratch_logits, experts, top_k, out_experts, out_weights, &denom)) {
        // Top-k was selected by raw logits to avoid sigmoid saturation ties; the
        // returned weights are sigmoid(selected_logit) values.
    } else if (!use_sigmoid && cpu_storage_router_fast && top_k <= 64u &&
        moe_router_select_topk_small_f32(*scratch_logits, experts, top_k, out_experts, out_weights, &denom)) {
        // Small top-k selection succeeded without scratch_order allocation.
    } else {
        scratch_order->resize(experts);
        uint32_t* order = scratch_order->data();
        for (uint32_t i = 0; i < experts; ++i) {
            order[i] = i;
        }
        auto cmp = [&](uint32_t a, uint32_t b) {
            const float av = (*scratch_logits)[a];
            const float bv = (*scratch_logits)[b];
            if (av > bv) return true;
            if (av < bv) return false;
            return a < b;  // deterministic source-compatible tie break for exact PPL
        };
        if (top_k < experts) {
            std::nth_element(order, order + top_k, order + experts, cmp);
        }
        std::sort(order, order + top_k, cmp);

        out_experts->clear();
        out_weights->clear();
        out_experts->reserve(top_k);
        out_weights->reserve(top_k);
        for (uint32_t i = 0; i < top_k; ++i) {
            float w = (*scratch_logits)[order[i]];
            if (use_sigmoid) {
                w = moe_router_sigmoid(w);
            }
            if (!std::isfinite(w)) {
                return 0;
            }
            out_experts->push_back((int)order[i]);
            out_weights->push_back(w);
            denom += w;
        }
    }
    const int norm_topk = moe_router_norm_topk_prob_for_layer(engine, layer, use_sigmoid);
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
    const float scale = moe_router_effective_routed_scaling_factor_for_layer(engine, layer, use_sigmoid);
    if (scale != 1.0f) {
        for (float& w : *out_weights) {
            w *= scale;
        }
    }
    moe_profile_log(
        "router_topk",
        "layer=%d experts=%u top_k=%u use_sigmoid=%d norm_topk=%d entropy=%.3f scale=%.6g raw_min=%.6g raw_max=%.6g raw_span=%.6g sigmoid_topk_raw=1 first_expert=%d first_weight=%.6g",
        layer, experts, top_k, use_sigmoid, norm_topk, (double)router_entropy, (double)scale,
        (double)raw_min_logit, (double)raw_max_logit, (double)(raw_max_logit - raw_min_logit),
        out_experts->empty() ? -1 : (*out_experts)[0],
        out_weights->empty() ? 0.0 : (double)(*out_weights)[0]);
    return 1;
}

