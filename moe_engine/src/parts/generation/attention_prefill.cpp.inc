static int moe_apply_rmsnorm_weight_entry_f32(
    const moe_pc_engine* engine,
    const moe_raw_tensor_cache_entry& raw,
    const float* input,
    uint32_t count,
    float* output,
    float eps
) {
    // BUGFIX 214: Division by zero clarification (same pattern as BUGFIX 208)
    // Problem: count == 0 check exists but no comment explaining division safety
    // Solution: Add explicit comment for code clarity
    if (!raw.base || !input || !output || count == 0 || raw.cols != 0 || raw.rows != count) {
        return 0;
    }
    const uint32_t elem = moe_raw_scalar_size(raw.scalar_encoding);
    if (raw.bytes < (uint64_t)count * (uint64_t)elem) {
        return 0;
    }
    double ss = 0.0;
    for (uint32_t i = 0; i < count; ++i) {
        const double val = (double)input[i];
        if (!std::isfinite(val)) {
            return 0;
        }
        ss += val * val;
    }
    // Safe: count > 0 guaranteed by check above
    const float norm = 1.0f / std::sqrt((float)(ss / (double)count) + (eps > 0.0f ? eps : 1.0e-6f));
    const int unit_offset = moe_engine_rmsnorm_unit_offset(engine, raw);
    for (uint32_t i = 0; i < count; ++i) {
        const float w = moe_raw_scalar_to_f32(raw.base + (uint64_t)i * (uint64_t)elem, raw.scalar_encoding);
        if (!std::isfinite(w)) {
            return 0;
        }
        const float result = input[i] * norm * (unit_offset ? (1.0f + w) : w);
        if (!std::isfinite(result)) {
            return 0;
        }
        output[i] = result;
    }
    return 1;
}

static int moe_apply_optional_head_rmsnorm_f32(
    moe_pc_engine* engine,
    int32_t layer,
    moe_raw_tensor_kind_t kind,
    float* values,
    uint32_t head_count,
    uint32_t head_dim,
    float eps
) {
    if (!engine || !values || head_count == 0 || head_dim == 0) {
        return 0;
    }
    moe_raw_tensor_cache_entry raw{};
    if (!get_cached_raw_tensor(engine, layer, kind, &raw)) {
        return 1;
    }
    if (raw.rows != head_dim || raw.cols != 0) {
        return 0;
    }
    std::vector<float> tmp(head_dim);
    for (uint32_t h = 0; h < head_count; ++h) {
        float* head = values + (uint64_t)h * (uint64_t)head_dim;
        if (!moe_apply_rmsnorm_weight_entry_f32(engine, raw, head, head_dim, tmp.data(), eps)) {
            return 0;
        }
        std::memcpy(head, tmp.data(), (size_t)head_dim * sizeof(float));
    }
    return 1;
}

static int moe_apply_unweighted_head_rmsnorm_f32(
    float* values,
    uint32_t head_count,
    uint32_t head_dim,
    float eps
) {
    // BUGFIX 215: Division by zero clarification
    // Problem: head_dim == 0 check exists but no comment explaining division safety
    // Solution: Add explicit comment for code clarity
    if (!values || head_count == 0 || head_dim == 0) {
        return 0;
    }
    const float norm_eps = eps > 0.0f ? eps : 1.0e-6f;
    for (uint32_t h = 0; h < head_count; ++h) {
        float* head = values + (uint64_t)h * (uint64_t)head_dim;
        double ss = 0.0;
        for (uint32_t i = 0; i < head_dim; ++i) {
            const double val = (double)head[i];
            if (!std::isfinite(val)) {
                return 0;
            }
            ss += val * val;
        }
        // Safe: head_dim > 0 guaranteed by check above
        const float norm = 1.0f / std::sqrt((float)(ss / (double)head_dim) + norm_eps);
        for (uint32_t i = 0; i < head_dim; ++i) {
            const float result = head[i] * norm;
            if (!std::isfinite(result)) {
                return 0;
            }
            head[i] = result;
        }
    }
    return 1;
}

static size_t moe_json_find_next_array_for_key_local(
    const std::string& text,
    const char* key,
    size_t search_pos,
    size_t* begin,
    size_t* end
) {
    if (begin) {
        *begin = 0;
    }
    if (end) {
        *end = 0;
    }
    if (!key || !key[0] || !begin || !end || search_pos >= text.size()) {
        return std::string::npos;
    }
    const size_t key_len = std::strlen(key);
    size_t pos = search_pos;
    while ((pos = text.find('"', pos)) != std::string::npos) {
        if (pos + 1u + key_len < text.size() &&
            text.compare(pos + 1u, key_len, key) == 0 &&
            text[pos + 1u + key_len] == '"') {
            const size_t colon = text.find_first_not_of(" \t\r\n", pos + key_len + 2u);
            if (colon != std::string::npos && colon < text.size() && text[colon] == ':') {
                const size_t value = text.find_first_not_of(" \t\r\n", colon + 1u);
                if (value != std::string::npos && value < text.size() && text[value] == '[') {
                    const size_t close = moe_json_match_bracket_local(text, value, '[', ']');
                    if (close != std::string::npos) {
                        *begin = value;
                        *end = close;
                        return pos;
                    }
                }
            }
        }
        ++pos;
    }
    return std::string::npos;
}

static int moe_layer_contract_declares_unweighted_v_norm(
    const std::string& doc,
    const char* array_key,
    int32_t layer
) {
    if (doc.empty() || !array_key || layer < 0) {
        return 0;
    }
    size_t search_pos = 0;
    while (search_pos < doc.size()) {
        size_t arr_begin = 0;
        size_t arr_end = 0;
        const size_t key_pos = moe_json_find_next_array_for_key_local(
            doc, array_key, search_pos, &arr_begin, &arr_end);
        if (key_pos == std::string::npos) {
            break;
        }
        search_pos = arr_end > key_pos ? arr_end : key_pos + 1u;
        size_t cursor = arr_begin + 1u;
        while (cursor < arr_end) {
            size_t obj_begin = 0;
            size_t obj_end = 0;
            if (!moe_json_next_object_in_array(doc, &cursor, arr_end, &obj_begin, &obj_end)) {
                break;
            }
            uint64_t parsed_layer = UINT64_MAX;
            if (!moe_json_get_u64_slice(doc, obj_begin, obj_end, "layer", &parsed_layer) ||
                parsed_layer != (uint64_t)layer) {
                continue;
            }
            int declared = 0;
            if (moe_json_get_bool_slice(
                    doc, obj_begin, obj_end,
                    "unweighted_value_norm_is_contractual_when_declared", &declared) &&
                declared) {
                return 1;
            }
            std::string mode;
            if (moe_json_get_string_slice(doc, obj_begin, obj_end, "value_norm_mode", &mode) &&
                mode == "unweighted_rmsnorm_contract") {
                return 1;
            }
            if (moe_json_get_string_slice(doc, obj_begin, obj_end, "norm_mode", &mode) &&
                mode == "unweighted_rmsnorm_contract") {
                return 1;
            }
        }
    }
    return 0;
}

static int moe_layer_object_contract_declares_unweighted_v_norm(
    const std::string& doc,
    const char* object_key,
    const char* array_key,
    int32_t layer
) {
    if (doc.empty() || !object_key || !array_key) {
        return 0;
    }
    size_t object_begin = 0;
    size_t object_end = 0;
    if (!moe_json_find_object_local(doc, object_key, &object_begin, &object_end)) {
        return 0;
    }
    return moe_layer_contract_declares_unweighted_v_norm(
        doc.substr(object_begin, object_end - object_begin), array_key, layer);
}

static int moe_layer_uses_unweighted_v_norm(
    moe_pc_engine* engine,
    int32_t layer
) {
    if (!engine || layer < 0) {
        return 0;
    }
    static thread_local const moe_pc_engine* tls_cached_engine = nullptr;
    static thread_local uint64_t tls_cached_generation = 0;
    static thread_local size_t tls_cached_graph_size = 0;
    static thread_local size_t tls_cached_metadata_size = 0;
    static thread_local std::vector<int8_t> tls_cached_results;
    const uint64_t generation = engine->engine_generation;
    const size_t graph_size = engine->offload_graph_ir_json.size();
    const size_t metadata_size = engine->offload_gguf_metadata_json.size();
    const size_t layer_index = (size_t)layer;
    int8_t* cached = nullptr;
    if (layer_index < 10000u) {
        if (tls_cached_engine != engine ||
            tls_cached_generation != generation ||
            tls_cached_graph_size != graph_size ||
            tls_cached_metadata_size != metadata_size) {
            tls_cached_engine = engine;
            tls_cached_generation = generation;
            tls_cached_graph_size = graph_size;
            tls_cached_metadata_size = metadata_size;
            tls_cached_results.clear();
        }
        if (tls_cached_results.size() <= layer_index) {
            tls_cached_results.resize(layer_index + 1u, -1);
        }
        cached = &tls_cached_results[layer_index];
        if (*cached >= 0) {
            return *cached ? 1 : 0;
        }
    }
    int result = 0;
    // Weighted V norm tensors are handled through moe_RAW_TENSOR_V_A_LAYER_NORM.
    // This fallback is only for contracts that explicitly imply an unweighted
    // value RMSNorm.
    if (moe_engine_has_common_tensor_suffix(engine, layer, "attn_v_norm.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "value_norm.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "v_norm.weight")) {
        if (cached) {
            *cached = 0;
        }
        return 0;
    }
    const std::string* docs[] = {
        &engine->offload_graph_ir_json,
        &engine->offload_gguf_metadata_json,
    };
    const char* doc_names[] = {
        "graph_ir",
        "metadata",
    };
    const char* array_keys[] = {
        "value_norm_contract_table",
        "layer_execution_contract_table",
        "attention_layer_contract_table",
        "attention_layer_table",
        "layers",
    };
    const char* matched_doc = "";
    const char* matched_array = "";
    for (size_t doc_i = 0; doc_i < sizeof(docs) / sizeof(docs[0]); ++doc_i) {
        const std::string* doc = docs[doc_i];
        if (!doc || doc->empty()) {
            continue;
        }
        for (size_t key_i = 0; key_i < sizeof(array_keys) / sizeof(array_keys[0]); ++key_i) {
            if (moe_layer_contract_declares_unweighted_v_norm(*doc, array_keys[key_i], layer)) {
                result = 1;
                matched_doc = doc_names[doc_i];
                matched_array = array_keys[key_i];
                break;
            }
        }
        if (result) {
            break;
        }
    }
    // Do not infer a layer-local V norm from graph-wide substring matches.
    // Unweighted V norm is applied only when a per-layer contract declares it,
    // regardless of whether the value source is v_projection or raw K.
    if (result) {
        moe_profile_log(
            "attn_value_norm_contract",
            "layer=%d mode=unweighted_rmsnorm_contract source=%s array=%s",
            layer,
            matched_doc,
            matched_array);
    }
    if (cached) {
        *cached = result ? 1 : 0;
    }
    return result;
}

static int moe_layer_has_qk_norm_scaled_standard_layout(
    const moe_pc_engine* engine,
    int32_t layer
) {
    if (!engine || layer < 0) {
        return 0;
    }
    const int has_q_norm =
        moe_engine_has_common_tensor_suffix(engine, layer, "attn_q_norm.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "q_norm.weight");
    const int has_k_norm =
        moe_engine_has_common_tensor_suffix(engine, layer, "attn_k_norm.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "k_norm.weight");
    const int has_scaled_block_norms =
        moe_engine_has_common_tensor_suffix(engine, layer, "post_attention_norm.weight") &&
        moe_engine_has_common_tensor_suffix(engine, layer, "post_ffw_norm.weight") &&
        moe_engine_has_common_tensor_suffix(engine, layer, "layer_output_scale.weight");
    return has_q_norm && has_k_norm && has_scaled_block_norms;
}

static uint32_t moe_attention_float_to_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float moe_attention_bits_to_float(uint32_t bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

static const char* moe_attention_scale_source_name(int source) {
    switch (source) {
    case 1: return "graph_ir_scale";
    case 2: return "metadata_scale";
    case 3: return "query_pre_attn_scalar";
    case 4: return "gemma4_unit_qk_norm";
    default: return "head_dim";
    }
}

struct moe_attention_runtime_cache_t {
    const moe_pc_engine* engine = nullptr;
    size_t graph_size = 0;
    size_t metadata_size = 0;
    uint32_t query_pre_attn_scalar_bits = 0;
    int scale_source = -1;
    uint32_t scale_bits = 0;
    int softcap_state = -1;
    uint32_t softcap_bits = 0;
};

static std::mutex g_moe_attention_runtime_cache_mutex;
static moe_attention_runtime_cache_t g_moe_attention_runtime_cache;

static void moe_attention_runtime_cache_prepare_locked(const moe_pc_engine* engine) {
    const size_t graph_size = engine ? engine->offload_graph_ir_json.size() : 0;
    const size_t metadata_size = engine ? engine->offload_gguf_metadata_json.size() : 0;
    const uint32_t query_bits = engine
        ? moe_attention_float_to_bits(engine->model_config_query_pre_attn_scalar)
        : 0;
    if (g_moe_attention_runtime_cache.engine != engine ||
        g_moe_attention_runtime_cache.graph_size != graph_size ||
        g_moe_attention_runtime_cache.metadata_size != metadata_size ||
        g_moe_attention_runtime_cache.query_pre_attn_scalar_bits != query_bits) {
        g_moe_attention_runtime_cache = moe_attention_runtime_cache_t{};
        g_moe_attention_runtime_cache.engine = engine;
        g_moe_attention_runtime_cache.graph_size = graph_size;
        g_moe_attention_runtime_cache.metadata_size = metadata_size;
        g_moe_attention_runtime_cache.query_pre_attn_scalar_bits = query_bits;
    }
}

static int moe_attention_cached_global_scale(
    const moe_pc_engine* engine,
    int* out_source,
    float* out_value
) {
    std::lock_guard<std::mutex> lock(g_moe_attention_runtime_cache_mutex);
    moe_attention_runtime_cache_prepare_locked(engine);
    if (g_moe_attention_runtime_cache.scale_source < 0) {
        return 0;
    }
    if (out_source) {
        *out_source = g_moe_attention_runtime_cache.scale_source;
    }
    if (out_value) {
        *out_value = moe_attention_bits_to_float(g_moe_attention_runtime_cache.scale_bits);
    }
    return 1;
}

static void moe_attention_store_global_scale(
    const moe_pc_engine* engine,
    int source,
    float value
) {
    std::lock_guard<std::mutex> lock(g_moe_attention_runtime_cache_mutex);
    moe_attention_runtime_cache_prepare_locked(engine);
    g_moe_attention_runtime_cache.scale_source = source;
    g_moe_attention_runtime_cache.scale_bits = source > 0 ? moe_attention_float_to_bits(value) : 0;
}

static int moe_attention_cached_softcap(
    const moe_pc_engine* engine,
    int* out_state,
    float* out_value
) {
    std::lock_guard<std::mutex> lock(g_moe_attention_runtime_cache_mutex);
    moe_attention_runtime_cache_prepare_locked(engine);
    if (g_moe_attention_runtime_cache.softcap_state < 0) {
        return 0;
    }
    if (out_state) {
        *out_state = g_moe_attention_runtime_cache.softcap_state;
    }
    if (out_value) {
        *out_value = moe_attention_bits_to_float(g_moe_attention_runtime_cache.softcap_bits);
    }
    return 1;
}

static void moe_attention_store_softcap(
    const moe_pc_engine* engine,
    int state,
    float value
) {
    std::lock_guard<std::mutex> lock(g_moe_attention_runtime_cache_mutex);
    moe_attention_runtime_cache_prepare_locked(engine);
    g_moe_attention_runtime_cache.softcap_state = state;
    g_moe_attention_runtime_cache.softcap_bits = state == 1 ? moe_attention_float_to_bits(value) : 0;
}

static float moe_standard_attention_score_scale(
    const moe_pc_engine* engine,
    int32_t layer,
    uint32_t head_dim,
    const char** source_out = nullptr
) {
    if (source_out) {
        *source_out = "head_dim";
    }
    if (head_dim == 0) {
        if (source_out) {
            *source_out = "fallback";
        }
        return 1.0f;
    }
    if (engine) {
        int cached_source = -1;
        float cached_value = 0.0f;
        if (moe_attention_cached_global_scale(engine, &cached_source, &cached_value)) {
            if (cached_source > 0) {
                if (cached_value > 0.0f && std::isfinite(cached_value)) {
                    if (source_out) {
                        *source_out = moe_attention_scale_source_name(cached_source);
                    }
                    return cached_value;
                }
            }
        } else {
            int resolved_source = 0;
            float resolved_value = 0.0f;
            if (moe_engine_is_gemma4_text_contract(engine) &&
                moe_layer_has_qk_norm_scaled_standard_layout(engine, layer)) {
                // Gemma4 text attention uses Q/K RMSNorm with unit attention
                // scaling.  query_pre_attn_scalar may still be present in
                // JUJU metadata, but applying 1/sqrt(qpre) double-scales the
                // normalized Q/K path and diverges from HF/vLLM.
                resolved_source = 4;
                resolved_value = 1.0f;
            }
            float query_pre_attn_scalar = engine->model_config_query_pre_attn_scalar;
            if (resolved_source == 0 && query_pre_attn_scalar > 0.0f && std::isfinite(query_pre_attn_scalar)) {
                const float result = 1.0f / std::sqrt(query_pre_attn_scalar);
                if (std::isfinite(result) && result > 0.0f) {
                    resolved_source = 3;
                    resolved_value = result;
                }
            }
            if (resolved_source == 0 && !engine->offload_graph_ir_json.empty()) {
                double direct_scale = 0.0;
                if ((moe_json_get_double_local(engine->offload_graph_ir_json, "attention_scale", &direct_scale) ||
                     moe_json_get_double_local(engine->offload_graph_ir_json, "attn_scale", &direct_scale) ||
                     moe_json_get_double_local(engine->offload_graph_ir_json, "attention_score_scale", &direct_scale) ||
                     moe_json_get_double_local(engine->offload_graph_ir_json, "f_attn_scale", &direct_scale) ||
                     moe_json_get_double_local(engine->offload_graph_ir_json, "gemma4.attention.scale", &direct_scale) ||
                     moe_json_get_double_local(engine->offload_graph_ir_json, "qk_scale", &direct_scale)) &&
                    direct_scale > 0.0 && direct_scale <= 1024.0 && std::isfinite(direct_scale)) {
                    resolved_source = 1;
                    resolved_value = (float)direct_scale;
                }
            }
            if (resolved_source == 0 && !engine->offload_gguf_metadata_json.empty()) {
                double direct_scale = 0.0;
                if ((moe_json_get_double_local(engine->offload_gguf_metadata_json, "attention_scale", &direct_scale) ||
                     moe_json_get_double_local(engine->offload_gguf_metadata_json, "attn_scale", &direct_scale) ||
                     moe_json_get_double_local(engine->offload_gguf_metadata_json, "attention_score_scale", &direct_scale) ||
                     moe_json_get_double_local(engine->offload_gguf_metadata_json, "f_attn_scale", &direct_scale) ||
                     moe_json_get_double_local(engine->offload_gguf_metadata_json, "gemma4.attention.scale", &direct_scale) ||
                     moe_json_get_double_local(engine->offload_gguf_metadata_json, "qk_scale", &direct_scale)) &&
                    direct_scale > 0.0 && direct_scale <= 1024.0 && std::isfinite(direct_scale)) {
                    resolved_source = 2;
                    resolved_value = (float)direct_scale;
                }
            }
            if (resolved_source == 0 && !(query_pre_attn_scalar > 0.0f) && !engine->offload_graph_ir_json.empty()) {
                double value = 0.0;
                if (moe_json_get_double_local(engine->offload_graph_ir_json, "query_pre_attn_scalar", &value) ||
                    moe_json_get_double_local(engine->offload_graph_ir_json, "attention_query_pre_attn_scalar", &value) ||
                    moe_json_get_double_local(engine->offload_graph_ir_json, "attn_query_pre_attn_scalar", &value)) {
                    if (value > 0.0 && value <= 65536.0 && std::isfinite(value)) {
                        query_pre_attn_scalar = (float)value;
                    }
                }
            }
            if (resolved_source == 0 && !(query_pre_attn_scalar > 0.0f) && !engine->offload_gguf_metadata_json.empty()) {
                double value = 0.0;
                if (moe_json_get_double_local(engine->offload_gguf_metadata_json, "query_pre_attn_scalar", &value) ||
                    moe_json_get_double_local(engine->offload_gguf_metadata_json, "attention_query_pre_attn_scalar", &value) ||
                    moe_json_get_double_local(engine->offload_gguf_metadata_json, "attn_query_pre_attn_scalar", &value)) {
                    if (value > 0.0 && value <= 65536.0 && std::isfinite(value)) {
                        query_pre_attn_scalar = (float)value;
                    }
                }
            }
            if (resolved_source == 0 && query_pre_attn_scalar > 0.0f && std::isfinite(query_pre_attn_scalar)) {
                const float result = 1.0f / std::sqrt(query_pre_attn_scalar);
                if (std::isfinite(result)) {
                    resolved_source = 3;
                    resolved_value = result;
                }
            }
            if (resolved_source > 0) {
                moe_attention_store_global_scale(engine, resolved_source, resolved_value);
                if (source_out) {
                    *source_out = moe_attention_scale_source_name(resolved_source);
                }
                return resolved_value;
            }
            moe_attention_store_global_scale(engine, 0, 0.0f);
        }
    }
    float result = 1.0f / std::sqrt((float)head_dim);
    if (!std::isfinite(result) || !(result > 0.0f)) {
        if (source_out) {
            *source_out = "fallback";
        }
        return 1.0f;
    }
    if (moe_layer_has_qk_norm_scaled_standard_layout(engine, layer)) {
        if (moe_engine_is_gemma4_text_contract(engine)) {
            if (source_out) {
                *source_out = "gemma4_unit_qk_norm";
            }
            return 1.0f;
        }
        if (source_out) {
            *source_out = "qk_norm_head_dim_fallback";
        }
        return result;
    }
    return result;
}

static int moe_attention_score_scale_source_is_explicit(
    const char* source
) {
    return source &&
        (std::strcmp(source, "graph_ir_scale") == 0 ||
         std::strcmp(source, "metadata_scale") == 0 ||
         std::strcmp(source, "query_pre_attn_scalar") == 0 ||
         std::strcmp(source, "gemma4_unit_qk_norm") == 0);
}

static int moe_attention_score_scale_violates_qk_norm_contract(
    const moe_pc_engine* engine,
    int32_t layer,
    float scale,
    const char* source
) {
    if (!moe_layer_has_qk_norm_scaled_standard_layout(engine, layer) ||
        moe_attention_score_scale_source_is_explicit(source)) {
        return 0;
    }
    return !(std::isfinite(scale) && scale > 0.0f && scale <= 1.0f);
}

static float moe_attention_logit_softcap_value(
    const moe_pc_engine* engine
) {
    if (engine) {
        int cached_state = -1;
        float cached_value = 0.0f;
        if (moe_attention_cached_softcap(engine, &cached_state, &cached_value)) {
            return cached_state == 1 ? cached_value : 0.0f;
        }
    }
    double value = 0.0;
    if (engine &&
        !engine->offload_graph_ir_json.empty() &&
        (moe_json_get_double_local(engine->offload_graph_ir_json, "attn_logit_softcapping", &value) ||
         moe_json_get_double_local(engine->offload_graph_ir_json, "attn_logit_softcap", &value) ||
         moe_json_get_double_local(engine->offload_graph_ir_json, "attention_logit_softcapping", &value) ||
         moe_json_get_double_local(engine->offload_graph_ir_json, "attention_logit_softcap", &value)) &&
        value > 0.0 && value <= 100000.0 && std::isfinite(value)) {
        moe_attention_store_softcap(engine, 1, (float)value);
        return (float)value;
    }
    if (engine &&
        !engine->offload_gguf_metadata_json.empty() &&
        (moe_json_get_double_local(engine->offload_gguf_metadata_json, "attn_logit_softcapping", &value) ||
         moe_json_get_double_local(engine->offload_gguf_metadata_json, "attn_logit_softcap", &value) ||
         moe_json_get_double_local(engine->offload_gguf_metadata_json, "attention_logit_softcapping", &value) ||
         moe_json_get_double_local(engine->offload_gguf_metadata_json, "attention_logit_softcap", &value)) &&
        value > 0.0 && value <= 100000.0 && std::isfinite(value)) {
        moe_attention_store_softcap(engine, 1, (float)value);
        return (float)value;
    }
    if (engine) {
        moe_attention_store_softcap(engine, 0, 0.0f);
    }
    return 0.0f;
}

static float moe_apply_attention_logit_softcap(
    float cap,
    float score
) {
    if (!(cap > 0.0f) || !std::isfinite(cap) || !std::isfinite(score)) {
        return score;
    }
    return std::tanh(score / cap) * cap;
}

static void moe_standard_attention_rope_params(
    const moe_pc_engine* engine,
    int32_t layer,
    int has_v,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t* io_rope_dim,
    float* io_rope_theta
) {
    if (!io_rope_dim || !io_rope_theta || head_dim == 0) {
        return;
    }
    if (!moe_layer_has_qk_norm_scaled_standard_layout(engine, layer)) {
        if (engine &&
            engine->model_config_attention_valid.load(std::memory_order_relaxed) &&
            *io_rope_dim > 0 && *io_rope_dim <= head_dim &&
            *io_rope_theta > 0.0f) {
            return;
        }
        return;
    }
    double partial_value = 0.0;
    if (!engine || engine->offload_gguf_metadata_json.empty() ||
        !moe_json_get_double_local(engine->offload_gguf_metadata_json, "partial_rotary_factor", &partial_value) ||
        partial_value <= 0.0 || partial_value > 1.0) {
        return;
    }
    double full_theta = 0.0;
    double sliding_theta = 0.0;
    const int has_full_theta =
        moe_json_get_double_local(engine->offload_gguf_metadata_json, "full_rope_theta", &full_theta) ||
        moe_json_get_double_local(engine->offload_gguf_metadata_json, "full_attention_rope_theta", &full_theta) ||
        moe_json_get_double_local(engine->offload_gguf_metadata_json, "gemma4.rope.freq_base", &full_theta) ||
        moe_json_get_double_local(engine->offload_gguf_metadata_json, "rope.freq_base", &full_theta);
    const int has_sliding_theta =
        moe_json_get_double_local(engine->offload_gguf_metadata_json, "sliding_rope_theta", &sliding_theta) ||
        moe_json_get_double_local(engine->offload_gguf_metadata_json, "sliding_attention_rope_theta", &sliding_theta) ||
        moe_json_get_double_local(engine->offload_gguf_metadata_json, "gemma4.rope.freq_base_swa", &sliding_theta) ||
        moe_json_get_double_local(engine->offload_gguf_metadata_json, "rope.freq_base_swa", &sliding_theta);
    const int full_attention_layout =
        !has_v ||
        (head_dim > 256u) ||
        (num_heads > 0 && num_kv_heads > 0 && num_kv_heads * 4u <= num_heads);
    if (full_attention_layout) {
        uint32_t partial_dim = (uint32_t)std::round((float)head_dim * (float)partial_value);
        if (partial_dim < 2u) partial_dim = 2u;
        if (partial_dim > head_dim) partial_dim = head_dim;
        *io_rope_dim = partial_dim;
        if (has_full_theta && full_theta > 0.0) {
            *io_rope_theta = (float)full_theta;
        }
    } else if (has_sliding_theta && sliding_theta > 0.0) {
        *io_rope_dim = head_dim;
        *io_rope_theta = (float)sliding_theta;
    }
}

static int moe_standard_kv_cache_ensure(
    moe_pc_engine* engine,
    moe_decode_scratch* s,
    uint32_t position,
    uint32_t kv_entry_dim
) {
    if (!engine || !s || kv_entry_dim == 0) {
        return 0;
    }
    if (moe_engine_qkv_only_required(engine)) {
        moe_qkv_only_plain_touch(engine, "capacity");
        return 0;
    }
    // BUGFIX 778: Check position for overflow ★★★
    if (position > UINT32_MAX / 2) {
        return 0;
    }
    if (position < s->kv_cache.max_seq_len) {
        return 1;
    }
    if (!s->kv_cache.ensure_capacity(position)) {
        return 0;
    }
    moe_kv_cache_track_ram(engine, &s->kv_cache);
    return position < s->kv_cache.max_seq_len ? 1 : 0;
}

static int moe_layer_standard_attention_f32(
    moe_pc_engine* engine,
    int32_t layer,
    moe_decode_scratch* s,
    float* hidden,
    uint32_t hidden_size,
    uint32_t position,
    const moe_attn_config_t& cfg,
    const float* norm_input = nullptr,
    int add_residual = 1
) {
    const uint64_t total_start_us = moe_profile_now_us();
    const uint64_t map_start_us = moe_profile_now_us();
    uint32_t num_heads = cfg.num_heads;
    uint32_t num_kv_heads = cfg.num_kv_heads;
    uint32_t head_dim = cfg.head_dim;
    uint32_t v_head_dim = cfg.v_head_dim ? cfg.v_head_dim : cfg.head_dim;
    moe_raw_tensor_cache_entry q_shape{};
    moe_raw_tensor_cache_entry k_shape{};
    moe_raw_tensor_cache_entry v_shape{};
    moe_raw_tensor_cache_entry o_shape{};
    moe_raw_tensor_cache_entry q_norm_shape{};
    moe_raw_tensor_cache_entry k_norm_shape{};
    const int has_q = get_cached_raw_tensor(engine, layer, moe_RAW_TENSOR_Q_A_PROJ, &q_shape);
    const int has_k = get_cached_raw_tensor(engine, layer, moe_RAW_TENSOR_KV_A_PROJ_WITH_MQA, &k_shape);
    const int has_v = get_cached_raw_tensor(engine, layer, moe_RAW_TENSOR_KV_B_PROJ, &v_shape);
    const int has_o = get_cached_raw_tensor(engine, layer, moe_RAW_TENSOR_O_PROJ, &o_shape);
    const int has_q_norm = get_cached_raw_tensor(engine, layer, moe_RAW_TENSOR_Q_A_LAYER_NORM, &q_norm_shape);
    const int has_k_norm = get_cached_raw_tensor(engine, layer, moe_RAW_TENSOR_KV_A_LAYER_NORM, &k_norm_shape);
    if (!has_q || !has_k || !has_o) {
        std::fprintf(stderr, "[storagellm attn] missing q/k/o layer=%d q=%d k=%d v=%d o=%d\n",
            layer, has_q, has_k, has_v, has_o);
        return 0;
    }
    if (has_q_norm && has_k_norm &&
        q_norm_shape.rows && q_norm_shape.rows == k_norm_shape.rows &&
        q_shape.rows % q_norm_shape.rows == 0 &&
        k_shape.rows % q_norm_shape.rows == 0) {
        head_dim = q_norm_shape.rows;
    } else if (head_dim == 0 && num_heads && q_shape.rows % num_heads == 0) {
        head_dim = q_shape.rows / num_heads;
    }
    if (head_dim && q_shape.rows % head_dim == 0) {
        num_heads = q_shape.rows / head_dim;
    }
    if (head_dim && k_shape.rows % head_dim == 0) {
        num_kv_heads = k_shape.rows / head_dim;
    }
    if (has_v && num_kv_heads && v_shape.rows % num_kv_heads == 0) {
        v_head_dim = v_shape.rows / num_kv_heads;
    } else if (num_heads && o_shape.cols % num_heads == 0) {
        v_head_dim = o_shape.cols / num_heads;
    }
    const uint32_t q_dim = q_shape.rows;
    const uint32_t k_dim = k_shape.rows;
    const uint32_t kv_v_dim = has_v ? v_shape.rows : num_kv_heads * v_head_dim;
    const uint32_t v_total_dim = o_shape.cols ? o_shape.cols : num_heads * v_head_dim;
    const uint32_t kv_entry_dim = k_dim + kv_v_dim;
    const uint32_t base_rope_dim = std::min<uint32_t>(cfg.qk_rope_head_dim ? cfg.qk_rope_head_dim : head_dim, head_dim);
    uint32_t rope_dim = base_rope_dim;
    float rope_theta = cfg.rope_theta;
    moe_standard_attention_rope_params(
        engine, layer, has_v, num_heads, num_kv_heads, head_dim, &rope_dim, &rope_theta);
    const int use_dynamic_rope =
        moe_engine_uses_hybrid_local_global_attention(engine) ||
        rope_dim != base_rope_dim ||
        std::fabs(rope_theta - cfg.rope_theta) > 1.0e-6f;
    moe_profile_log(
        "attn_standard_cfg",
        "layer=%d pos=%u heads=%u kv_heads=%u head_dim=%u v_head_dim=%u q_dim=%u k_dim=%u kv_v_dim=%u v_total=%u rope_dim=%u theta=%.6g has_v=%d dynamic_rope=%d",
        layer,
        position,
        num_heads,
        num_kv_heads,
        head_dim,
        v_head_dim,
        q_dim,
        k_dim,
