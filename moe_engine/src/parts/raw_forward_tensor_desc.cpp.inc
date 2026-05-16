// Raw forward - Tensor descriptor definitions

struct moe_raw_tensor_desc {
    uint32_t role;
    uint64_t span_bytes;
    uint32_t occurrence;
    uint64_t rel_offset;
    uint64_t bytes;
    uint32_t rows;
    uint32_t cols;
    uint8_t math_transposed;
};

static uint8_t moe_raw_scalar_encoding_from_common_tensor(
    const moe_gguf_common_tensor_record& rec,
    uint32_t rows,
    uint32_t cols
) {
    if (rec.weight_encoding == moe_WEIGHT_ENCODING_RAW_FP32 || rec.type == 0u) {
        return moe_RAW_SCALAR_F32;
    }
    if (rec.weight_encoding == moe_WEIGHT_ENCODING_RAW_FP16 || rec.type == 1u) {
        return moe_RAW_SCALAR_F16;
    }
    if (rec.weight_encoding == moe_WEIGHT_ENCODING_RAW_BF16 || rec.type == 30u) {
        return moe_RAW_SCALAR_BF16;
    }
    (void)rows;
    (void)cols;
    return moe_RAW_SCALAR_UNKNOWN;
}

static uint8_t moe_raw_weight_encoding_from_common_tensor(
    const moe_gguf_common_tensor_record& rec,
    uint32_t rows,
    uint32_t cols,
    uint8_t scalar_encoding
) {
    (void)rows;
    (void)cols;
    (void)scalar_encoding;
    if (rec.type == 0u || rec.weight_encoding == moe_WEIGHT_ENCODING_RAW_FP32) {
        return moe_WEIGHT_ENCODING_RAW_FP32;
    }
    if (rec.type == 1u || rec.weight_encoding == moe_WEIGHT_ENCODING_RAW_FP16) {
        return moe_WEIGHT_ENCODING_RAW_FP16;
    }
    if (rec.type == 30u || rec.weight_encoding == moe_WEIGHT_ENCODING_RAW_BF16) {
        return moe_WEIGHT_ENCODING_RAW_BF16;
    }
    return moe_WEIGHT_ENCODING_UNKNOWN;
}

static void moe_trace_common_tensor_unknown_weight_format(
    const moe_gguf_common_tensor_record& rec,
    const moe_raw_tensor_desc& desc
) {
    static std::atomic<uint32_t> logged{0};
    const uint32_t n = logged.fetch_add(1, std::memory_order_relaxed);
    if (n >= 32u) {
        return;
    }
    moe_profile_log(
        "common_tensor_unknown_weight_format",
        "name=%s type=%u encoding=%u bytes=%llu rows=%u cols=%u",
        rec.name.c_str(),
        rec.type,
        (uint32_t)rec.weight_encoding,
        (unsigned long long)rec.bytes,
        desc.rows,
        desc.cols);
}

static void moe_trace_common_tensor_unsupported_weight_format(
    const moe_gguf_common_tensor_record& rec,
    const moe_raw_tensor_desc& desc,
    uint8_t weight_format
) {
    static std::atomic<uint32_t> logged{0};
    const uint32_t n = logged.fetch_add(1, std::memory_order_relaxed);
    if (n >= 32u) {
        return;
    }
    moe_profile_log(
        "common_tensor_unsupported_weight_format",
        "name=%s type=%u encoding=%u format=%u bytes=%llu rows=%u cols=%u",
        rec.name.c_str(),
        rec.type,
        (uint32_t)rec.weight_encoding,
        (uint32_t)weight_format,
        (unsigned long long)rec.bytes,
        desc.rows,
        desc.cols);
}

static uint8_t moe_common_tensor_effective_weight_format(
    const moe_gguf_common_tensor_record& rec,
    const moe_raw_tensor_desc& desc
) {
    const uint8_t scalar_encoding = moe_raw_scalar_encoding_from_common_tensor(rec, desc.rows, desc.cols);
    uint8_t weight_format = rec.weight_encoding != moe_WEIGHT_ENCODING_UNKNOWN ?
        rec.weight_encoding : moe_weight_encoding_from_gguf_type(rec.type, moe_offload_gguf_contract_t{});
    if (weight_format == moe_WEIGHT_ENCODING_UNKNOWN) {
        weight_format = moe_raw_weight_encoding_from_common_tensor(
            rec, desc.rows, desc.cols, scalar_encoding);
    }
    return weight_format;
}

static int moe_common_tensor_raw_weight_format_supported(uint8_t weight_format) {
    return weight_format == moe_WEIGHT_ENCODING_RAW_FP16 ||
        weight_format == moe_WEIGHT_ENCODING_RAW_BF16 ||
        weight_format == moe_WEIGHT_ENCODING_RAW_FP32 ||
        weight_format == moe_WEIGHT_ENCODING_MXFP4 ||
        weight_format == moe_WEIGHT_ENCODING_GGUF_Q4_0 ||
        weight_format == moe_WEIGHT_ENCODING_GGUF_Q4_1 ||
        weight_format == moe_WEIGHT_ENCODING_GGUF_Q5_0 ||
        weight_format == moe_WEIGHT_ENCODING_GGUF_Q5_1 ||
        weight_format == moe_WEIGHT_ENCODING_GGUF_Q8_0 ||
        weight_format == moe_WEIGHT_ENCODING_GGUF_Q8_1 ||
        weight_format == moe_WEIGHT_ENCODING_GGUF_Q8_K ||
        weight_format == moe_WEIGHT_ENCODING_GGUF_Q2_K ||
        weight_format == moe_WEIGHT_ENCODING_GGUF_Q3_K ||
        weight_format == moe_WEIGHT_ENCODING_GGUF_Q4_K ||
        weight_format == moe_WEIGHT_ENCODING_GGUF_Q5_K ||
        weight_format == moe_WEIGHT_ENCODING_GGUF_Q6_K ||
        weight_format == moe_WEIGHT_ENCODING_GGUF_IQ2_XXS ||
        weight_format == moe_WEIGHT_ENCODING_GGUF_IQ2_XS ||
        weight_format == moe_WEIGHT_ENCODING_GGUF_IQ2_S ||
        weight_format == moe_WEIGHT_ENCODING_GGUF_IQ3_XXS ||
        weight_format == moe_WEIGHT_ENCODING_GGUF_IQ3_S ||
        weight_format == moe_WEIGHT_ENCODING_GGUF_IQ4_NL ||
        weight_format == moe_WEIGHT_ENCODING_GGUF_IQ4_XS ||
        weight_format == moe_WEIGHT_ENCODING_GGUF_IQ1_S ||
        weight_format == moe_WEIGHT_ENCODING_GGUF_IQ1_M;
}

static int moe_common_tensor_record_raw_compatible(
    const moe_gguf_common_tensor_record& rec,
    const moe_raw_tensor_desc& desc,
    uint8_t* out_weight_format = nullptr
) {
    const uint8_t weight_format = moe_common_tensor_effective_weight_format(rec, desc);
    if (out_weight_format) {
        *out_weight_format = weight_format;
    }
    return moe_common_tensor_raw_weight_format_supported(weight_format);
}

static void moe_trace_raw_tensor_static_fallback_blocked(int32_t layer, moe_raw_tensor_kind_t kind) {
    static std::atomic<uint32_t> logged{0};
    const uint32_t n = logged.fetch_add(1, std::memory_order_relaxed);
    if (n >= 64u) {
        return;
    }
    moe_profile_log(
        "raw_tensor_static_fallback_blocked",
        "layer=%d kind=%u",
        (int)layer,
        (uint32_t)kind);
}

static int moe_common_tensor_record_to_raw_desc(
    const moe_gguf_common_tensor_record& rec,
    moe_raw_tensor_desc* out_desc
) {
    if (!out_desc || rec.bytes == 0 || rec.dims == 0 || rec.shape[0] == 0) {
        return 0;
    }
    moe_raw_tensor_desc desc{};
    desc.rel_offset = 0;
    desc.bytes = rec.bytes;
    desc.math_transposed = rec.math_transposed;
    if (rec.dims == 1) {
        desc.rows = (uint32_t)std::min<uint64_t>(rec.shape[0], UINT32_MAX);
        desc.cols = 0;
    } else if (rec.math_rows > 0 && rec.math_cols > 0) {
        desc.rows = rec.math_rows;
        desc.cols = rec.math_cols;
    } else if (rec.math_rows > 0) {
        desc.rows = rec.math_rows;
        desc.cols = 0;
    } else {
        if (!rec.shape[1]) {
            return 0;
        }
        desc.cols = (uint32_t)std::min<uint64_t>(rec.shape[0], UINT32_MAX);
        desc.rows = (uint32_t)std::min<uint64_t>(rec.shape[1], UINT32_MAX);
    }
    if (!desc.rows) {
        return 0;
    }
    *out_desc = desc;
    return 1;
}

static const moe_gguf_common_tensor_record* moe_common_tensor_find(
    const moe_pc_engine* engine,
    const std::string& name
) {
    if (!engine || name.empty()) {
        return nullptr;
    }
    const std::string key = moe_lower_ascii_copy(name);
    auto it = engine->gguf_common_tensor_slots.find(key);
    if (it == engine->gguf_common_tensor_slots.end() ||
        it->second >= engine->gguf_common_tensors.size()) {
        return nullptr;
    }
    return &engine->gguf_common_tensors[it->second];
}

static void moe_push_layer_tensor_candidate(
    std::vector<std::string>* out,
    int32_t layer,
    const char* suffix
) {
    if (!out || layer < 0 || !suffix || !suffix[0]) {
        return;
    }
    char buf[256];
    std::snprintf(buf, sizeof(buf), "blk.%d.%s", (int)layer, suffix);
    out->push_back(buf);
    std::snprintf(buf, sizeof(buf), "model.layers.%d.%s", (int)layer, suffix);
    out->push_back(buf);
    std::snprintf(buf, sizeof(buf), "layers.%d.%s", (int)layer, suffix);
    out->push_back(buf);
    std::snprintf(buf, sizeof(buf), "transformer.h.%d.%s", (int)layer, suffix);
    out->push_back(buf);
    std::snprintf(buf, sizeof(buf), "h.%d.%s", (int)layer, suffix);
    out->push_back(buf);
}

static void moe_push_layer_tensor_candidates(
    std::vector<std::string>* out,
    int32_t layer,
    const char* const* suffixes,
    size_t suffix_count
) {
    if (!out || !suffixes) {
        return;
    }
    for (size_t i = 0; i < suffix_count; ++i) {
        moe_push_layer_tensor_candidate(out, layer, suffixes[i]);
    }
}

static void moe_raw_tensor_kind_common_names(
    int32_t layer,
    moe_raw_tensor_kind_t kind,
    std::vector<std::string>* out
) {
    if (!out) {
        return;
    }
    out->clear();
    switch (kind) {
        case moe_RAW_TENSOR_INPUT_LAYER_NORM: {
            static const char* const suffixes[] = {
                "attn_norm.weight",
                "pre_attention_norm.weight",
                "input_layernorm.weight",
                "input_layer_norm.weight",
                "attention_norm.weight",
                "self_attn_layer_norm.weight",
                "ln_1.weight"
            };
            moe_push_layer_tensor_candidates(out, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]));
            // Do not fall back to final_norm/global norm for a layer-local input norm.
            // That silently applies the final layer scale to every attention block.
            break;
        }
        case moe_RAW_TENSOR_POST_ATTENTION_LAYER_NORM: {
            static const char* const suffixes[] = {
                "post_attention_norm.weight",
                "post_attention_layernorm.weight",
                "post_attention_layer_norm.weight",
                "post_attn_norm.weight"
            };
            // GraphIR-required artifacts can carry separate ffn_norm,
            // pre_ffw_norm and post_ffw_norm tensors.  Treating those aliases as
            // post-attention norms made the hand-coded forward bind an FFN norm
            // into the attention residual branch.  Keep this raw role exact; old
            // naming aliases such as ln_2 must be handled by explicit graph-role
            // binding rather than a broad fallback.
            moe_push_layer_tensor_candidates(out, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]));
            break;
        }
        case moe_RAW_TENSOR_Q_A_LAYER_NORM: {
            static const char* const suffixes[] = {
                "attn_q_norm.weight",
                "q_norm.weight",
                "attn_q_a_norm.weight",
                "q_a_norm.weight",
                "q_a_layernorm.weight",
                "self_attn.q_norm.weight",
                "self_attn.q_a_norm.weight",
                "self_attn.q_a_layernorm.weight"
            };
            moe_push_layer_tensor_candidates(out, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]));
            break;
        }
        case moe_RAW_TENSOR_KV_A_LAYER_NORM: {
            static const char* const suffixes[] = {
                "attn_k_norm.weight",
                "k_norm.weight",
                "attn_kv_a_norm.weight",
                "kv_a_norm.weight",
                "kv_a_layernorm.weight",
                "self_attn.k_norm.weight",
                "self_attn.kv_a_norm.weight",
                "self_attn.kv_a_layernorm.weight"
            };
            moe_push_layer_tensor_candidates(out, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]));
            break;
        }
        case moe_RAW_TENSOR_V_A_LAYER_NORM: {
            static const char* const suffixes[] = {
                "attn_v_norm.weight",
                "v_norm.weight",
                "attn_value_norm.weight",
                "value_norm.weight",
                "attn_kv_b_norm.weight",
                "kv_b_norm.weight",
                "self_attn.v_norm.weight",
                "self_attn.value_norm.weight",
                "self_attn.kv_b_norm.weight"
            };
            moe_push_layer_tensor_candidates(out, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]));
            break;
        }
        case moe_RAW_TENSOR_Q_A_PROJ: {
            static const char* const suffixes[] = {
                "attn_q.weight",
                "q_proj.weight",
                "q_projection.weight",
                "query_proj.weight",
                "self_attn.q_proj.weight",
                "attention.wq.weight",
                "attn_q_a.weight",
                "attn_q_a_proj.weight",
                "q_a_proj.weight",
                "self_attn.q_a_proj.weight"
            };
            moe_push_layer_tensor_candidates(out, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]));
            break;
        }
        case moe_RAW_TENSOR_Q_B_PROJ: {
            static const char* const suffixes[] = {
                "attn_q_b.weight",
                "attn_q_b_proj.weight",
                "q_b_proj.weight",
                "self_attn.q_b_proj.weight"
            };
            moe_push_layer_tensor_candidates(out, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]));
            break;
        }
        case moe_RAW_TENSOR_KV_A_PROJ_WITH_MQA: {
            static const char* const suffixes[] = {
                "attn_k.weight",
                "k_proj.weight",
                "k_projection.weight",
                "key_proj.weight",
                "self_attn.k_proj.weight",
                "attention.wk.weight",
                "attn_kv_a_mqa.weight",
                "attn_kv_a_proj_with_mqa.weight",
                "kv_a_proj_with_mqa.weight",
                "kv_a_proj.weight",
                "self_attn.kv_a_proj_with_mqa.weight",
                "self_attn.kv_a_proj.weight"
            };
            moe_push_layer_tensor_candidates(out, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]));
            break;
        }
        case moe_RAW_TENSOR_KV_B_PROJ: {
            static const char* const suffixes[] = {
                "attn_v.weight",
                "v_proj.weight",
                "v_projection.weight",
                "value_proj.weight",
                "self_attn.v_proj.weight",
                "attention.wv.weight",
                "attn_kv_b.weight",
                "attn_kv_b_proj.weight",
                "kv_b_proj.weight",
                "self_attn.kv_b_proj.weight"
            };
            moe_push_layer_tensor_candidates(out, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]));
            break;
        }
        case moe_RAW_TENSOR_O_PROJ: {
            static const char* const suffixes[] = {
                "attn_output.weight",
                "attn_o.weight",
                "o_proj.weight",
                "out_proj.weight",
                "dense.weight",
                "self_attn.o_proj.weight",
                "attention.wo.weight"
            };
            moe_push_layer_tensor_candidates(out, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]));
            break;
        }
        default:
            break;
    }
}

static const moe_gguf_common_tensor_record* moe_common_tensor_find_first_raw_compatible(
    const moe_pc_engine* engine,
    const std::vector<std::string>& names,
    moe_raw_tensor_desc* out_desc
) {
    for (const std::string& name : names) {
        const moe_gguf_common_tensor_record* rec = moe_common_tensor_find(engine, name);
        if (!rec) {
            continue;
        }
        moe_raw_tensor_desc desc{};
        if (!moe_common_tensor_record_to_raw_desc(*rec, &desc)) {
            continue;
        }
        if (!moe_common_tensor_record_raw_compatible(*rec, desc)) {
            continue;
        }
        if (out_desc) {
            *out_desc = desc;
        }
        return rec;
    }
    return nullptr;
}

static void moe_raw_push_unique_tensor_name(
    std::vector<std::string>* out,
    const std::string& name
) {
    if (!out || name.empty()) {
        return;
    }
    for (const std::string& existing : *out) {
        if (existing == name) {
            return;
        }
    }
    out->push_back(name);
}

static int moe_json_find_next_array_key_raw(
    const std::string& text,
    const char* key,
    size_t start_pos,
    size_t* begin,
    size_t* end
) {
    if (!key || !begin || !end || start_pos >= text.size()) {
        return 0;
    }
    const size_t key_len = std::strlen(key);
    size_t pos = start_pos;
    while ((pos = text.find('"', pos)) != std::string::npos) {
        if (pos + 1 + key_len < text.size() &&
            text.compare(pos + 1, key_len, key) == 0 &&
            text[pos + 1 + key_len] == '"') {
            const size_t colon = text.find_first_not_of(" \t\r\n", pos + key_len + 2);
            if (colon != std::string::npos && colon < text.size() && text[colon] == ':') {
                const size_t value = text.find_first_not_of(" \t\r\n", colon + 1);
                if (value != std::string::npos && value < text.size() && text[value] == '[') {
                    const size_t close = moe_json_match_bracket_local(text, value, '[', ']');
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

static void moe_raw_tensor_kind_contract_fields(
    moe_raw_tensor_kind_t kind,
    const char** out_section,
    std::vector<const char*>* out_keys
) {
    if (out_section) {
        *out_section = nullptr;
    }
    if (!out_keys) {
        return;
    }
    out_keys->clear();
    switch (kind) {
        case moe_RAW_TENSOR_INPUT_LAYER_NORM:
            if (out_section) *out_section = "norms";
            out_keys->push_back("attention_input_norm");
            out_keys->push_back("input_layer_norm");
            break;
        case moe_RAW_TENSOR_POST_ATTENTION_LAYER_NORM:
            if (out_section) *out_section = "norms";
            out_keys->push_back("post_attention_norm");
            out_keys->push_back("post_attn_norm");
            break;
        case moe_RAW_TENSOR_Q_A_LAYER_NORM:
            if (out_section) *out_section = "attention";
            out_keys->push_back("q_norm");
            break;
        case moe_RAW_TENSOR_KV_A_LAYER_NORM:
            if (out_section) *out_section = "attention";
            out_keys->push_back("k_norm");
            break;
        case moe_RAW_TENSOR_V_A_LAYER_NORM:
            if (out_section) *out_section = "attention";
            out_keys->push_back("v_norm");
            out_keys->push_back("value_norm");
            break;
        case moe_RAW_TENSOR_Q_A_PROJ:
            if (out_section) *out_section = "attention";
            out_keys->push_back("q_projection");
            out_keys->push_back("query_projection");
            break;
        case moe_RAW_TENSOR_Q_B_PROJ:
            if (out_section) *out_section = "attention";
            out_keys->push_back("q_b_projection");
            out_keys->push_back("query_b_projection");
            break;
        case moe_RAW_TENSOR_KV_A_PROJ_WITH_MQA:
            if (out_section) *out_section = "attention";
            out_keys->push_back("k_projection");
            out_keys->push_back("key_projection");
            out_keys->push_back("kv_a_projection");
            break;
        case moe_RAW_TENSOR_KV_B_PROJ:
            if (out_section) *out_section = "attention";
            out_keys->push_back("v_projection");
            out_keys->push_back("value_projection");
            out_keys->push_back("kv_b_projection");
            break;
        case moe_RAW_TENSOR_O_PROJ:
            if (out_section) *out_section = "attention";
            out_keys->push_back("o_projection");
            out_keys->push_back("attention_output");
            out_keys->push_back("out_projection");
            break;
        default:
            break;
    }
}

static void moe_raw_contract_tensor_names_from_doc(
    const std::string& doc,
    int32_t layer,
    moe_raw_tensor_kind_t kind,
    std::vector<std::string>* out_names
) {
    if (!out_names || doc.empty() || layer < 0) {
        return;
    }
    const char* section_key = nullptr;
    std::vector<const char*> field_keys;
    moe_raw_tensor_kind_contract_fields(kind, &section_key, &field_keys);
    if (field_keys.empty()) {
        return;
    }
    size_t search_pos = 0;
    size_t arr_begin = 0;
    size_t arr_end = 0;
    while (moe_json_find_next_array_key_raw(
        doc, "layer_execution_contract_table", search_pos, &arr_begin, &arr_end)) {
        size_t cursor = arr_begin + 1;
        size_t obj_begin = 0;
        size_t obj_end = 0;
        while (moe_json_next_object_in_array(doc, &cursor, arr_end, &obj_begin, &obj_end)) {
            uint64_t row_layer = UINT64_MAX;
            if (!moe_json_get_u64_slice(doc, obj_begin, obj_end, "layer", &row_layer) ||
                row_layer != (uint64_t)layer) {
                continue;
            }
            size_t sec_begin = 0;
            size_t sec_end = 0;
            const int has_section = section_key &&
                moe_json_find_object_local(doc.substr(obj_begin, obj_end - obj_begin), section_key, &sec_begin, &sec_end);
            for (const char* key : field_keys) {
                if (!key || !key[0]) {
                    continue;
                }
                std::string name;
                if (has_section) {
                    const size_t abs_sec_begin = obj_begin + sec_begin;
                    const size_t abs_sec_end = obj_begin + sec_end;
                    if (moe_json_get_string_slice(doc, abs_sec_begin, abs_sec_end, key, &name) &&
                        !name.empty()) {
                        moe_raw_push_unique_tensor_name(out_names, name);
                    }
                }
                name.clear();
                if (moe_json_get_string_slice(doc, obj_begin, obj_end, key, &name) && !name.empty()) {
                    moe_raw_push_unique_tensor_name(out_names, name);
                }
            }
        }
        search_pos = arr_end;
    }
}

static const moe_gguf_common_tensor_record* find_common_raw_tensor_record_by_contract(
    const moe_pc_engine* engine,
    int32_t layer,
    moe_raw_tensor_kind_t kind,
    moe_raw_tensor_desc* out_desc
) {
    if (!engine || layer < 0) {
        return nullptr;
    }
    std::vector<std::string> names;
    moe_raw_contract_tensor_names_from_doc(engine->offload_graph_ir_json, layer, kind, &names);
    moe_raw_contract_tensor_names_from_doc(engine->offload_gguf_metadata_json, layer, kind, &names);
    if (names.empty()) {
        return nullptr;
    }
    const moe_gguf_common_tensor_record* rec =
        moe_common_tensor_find_first_raw_compatible(engine, names, out_desc);
    if (rec) {
        moe_profile_log(
            "raw_tensor_contract_map",
            "layer=%d kind=%u name=%s source=layer_execution_contract_table",
            (int)layer,
            (unsigned)kind,
            rec->name.c_str());
    }
    return rec;
}

static int moe_common_tensor_layer_matches(
    const moe_gguf_common_tensor_record& rec,
    int32_t layer
) {
    if (rec.execution_layer != INT32_MIN) {
        return rec.execution_layer == layer ? 1 : 0;
    }
    uint32_t parsed_layer = 0;
    return moe_parse_layer_from_gguf_name(rec.name, &parsed_layer) &&
        (int32_t)parsed_layer == layer ? 1 : 0;
}

static int moe_common_tensor_role_text_matches(
    const std::string& text,
    const char* role
) {
    if (!role || !role[0] || text.empty()) {
        return 0;
    }
    const std::string lhs = moe_lower_ascii_copy(text);
    const std::string rhs = moe_lower_ascii_copy(role);
    return lhs == rhs ? 1 : 0;
}

static int moe_common_tensor_op_role_matches(
    const moe_gguf_common_tensor_record& rec,
    const char* role
) {
    if (!role || !role[0]) {
        return 0;
    }
    return
        moe_common_tensor_role_text_matches(rec.op_role, role) ||
        moe_common_tensor_role_text_matches(rec.execution_op, role) ||
        moe_common_tensor_role_text_matches(rec.bundle_member_role, role) ||
        moe_common_tensor_role_text_matches(rec.graph_role_text, role);
}

static void moe_raw_tensor_kind_op_roles(
    moe_raw_tensor_kind_t kind,
    std::vector<const char*>* out
) {
    if (!out) {
        return;
    }
    out->clear();
    switch (kind) {
        case moe_RAW_TENSOR_INPUT_LAYER_NORM:
            out->push_back("attention_input_norm");
            break;
        case moe_RAW_TENSOR_POST_ATTENTION_LAYER_NORM:
            out->push_back("post_attention_norm");
            break;
        case moe_RAW_TENSOR_Q_A_LAYER_NORM:
            out->push_back("q_norm");
            break;
        case moe_RAW_TENSOR_KV_A_LAYER_NORM:
            out->push_back("k_norm");
            break;
        case moe_RAW_TENSOR_V_A_LAYER_NORM:
            out->push_back("v_norm");
            break;
        case moe_RAW_TENSOR_Q_A_PROJ:
            out->push_back("q_projection");
            break;
        case moe_RAW_TENSOR_Q_B_PROJ:
            out->push_back("q_b_projection");
            break;
        case moe_RAW_TENSOR_KV_A_PROJ_WITH_MQA:
            out->push_back("k_projection");
            out->push_back("kv_a_projection");
            break;
        case moe_RAW_TENSOR_KV_B_PROJ:
            out->push_back("v_projection");
            out->push_back("kv_b_projection");
            break;
        case moe_RAW_TENSOR_O_PROJ:
            out->push_back("attention_output");
            out->push_back("o_projection");
            break;
        default:
            break;
    }
}

static const moe_gguf_common_tensor_record* find_common_raw_tensor_record_by_role(
    const moe_pc_engine* engine,
    int32_t layer,
    moe_raw_tensor_kind_t kind,
    moe_raw_tensor_desc* out_desc
) {
    if (!engine) {
        return nullptr;
    }
    std::vector<const char*> roles;
    moe_raw_tensor_kind_op_roles(kind, &roles);
    if (roles.empty()) {
        return nullptr;
    }
    for (const moe_gguf_common_tensor_record& rec : engine->gguf_common_tensors) {
        if (!moe_common_tensor_layer_matches(rec, layer)) {
            continue;
        }
        int role_match = 0;
        for (const char* role : roles) {
            if (moe_common_tensor_op_role_matches(rec, role)) {
                role_match = 1;
                break;
            }
        }
        if (!role_match) {
            continue;
        }
        moe_raw_tensor_desc desc{};
        if (!moe_common_tensor_record_to_raw_desc(rec, &desc)) {
            continue;
        }
        if (!moe_common_tensor_record_raw_compatible(rec, desc)) {
            continue;
        }
        if (out_desc) {
            *out_desc = desc;
        }
        return &rec;
    }
    return nullptr;
}

static const moe_gguf_common_tensor_record* find_common_raw_tensor_record(
    const moe_pc_engine* engine,
    int32_t layer,
    moe_raw_tensor_kind_t kind,
    moe_raw_tensor_desc* out_desc
) {
    if (const moe_gguf_common_tensor_record* rec =
            find_common_raw_tensor_record_by_contract(engine, layer, kind, out_desc)) {
        return rec;
    }
    if (const moe_gguf_common_tensor_record* rec =
            find_common_raw_tensor_record_by_role(engine, layer, kind, out_desc)) {
        return rec;
    }
    std::vector<std::string> names;
    moe_raw_tensor_kind_common_names(layer, kind, &names);
    return moe_common_tensor_find_first_raw_compatible(engine, names, out_desc);
}

static const moe_gguf_common_tensor_record* find_common_tensor_by_names(
    const moe_pc_engine* engine,
    const char* const* names,
    size_t name_count,
    moe_raw_tensor_desc* out_desc
) {
    if (!names) {
        return nullptr;
    }
    for (size_t i = 0; i < name_count; ++i) {
        if (const moe_gguf_common_tensor_record* rec = moe_common_tensor_find(engine, names[i])) {
            moe_raw_tensor_desc desc{};
            if (!moe_common_tensor_record_to_raw_desc(*rec, &desc)) {
                continue;
            }
            if (!moe_common_tensor_record_raw_compatible(*rec, desc)) {
                continue;
            }
            if (out_desc) {
                *out_desc = desc;
            }
            return rec;
        }
    }
    return nullptr;
}

struct moe_common_op_role_lookup_cache_entry {
    const moe_pc_engine* engine = nullptr;
    uint64_t generation = 0;
    int32_t layer = INT32_MIN;
    std::string role;
    const moe_gguf_common_tensor_record* rec = nullptr;
    moe_raw_tensor_desc desc{};
    int found = 0;
};

static const moe_gguf_common_tensor_record* find_common_tensor_by_op_role_uncached(
    const moe_pc_engine* engine,
    const char* role,
    int32_t layer,
    moe_raw_tensor_desc* out_desc
) {
    if (!engine || !role || !role[0]) {
        return nullptr;
    }
    for (const moe_gguf_common_tensor_record& rec : engine->gguf_common_tensors) {
        if (!moe_common_tensor_op_role_matches(rec, role)) {
            continue;
        }
        if (layer >= 0) {
            if (!moe_common_tensor_layer_matches(rec, layer)) {
                continue;
            }
        } else if (layer != INT32_MIN && rec.execution_layer != INT32_MIN &&
            rec.execution_layer != layer) {
            continue;
        }
        moe_raw_tensor_desc desc{};
        if (!moe_common_tensor_record_to_raw_desc(rec, &desc)) {
            continue;
        }
        if (!moe_common_tensor_record_raw_compatible(rec, desc)) {
            continue;
        }
        if (out_desc) {
            *out_desc = desc;
        }
        return &rec;
    }
    return nullptr;
}

static const moe_gguf_common_tensor_record* find_common_tensor_by_op_role(
    const moe_pc_engine* engine,
    const char* role,
    int32_t layer,
    moe_raw_tensor_desc* out_desc
) {
    if (!engine || !role || !role[0]) {
        return nullptr;
    }
    static thread_local std::vector<moe_common_op_role_lookup_cache_entry> tls_cache;
    const uint64_t generation = engine->engine_generation;
    for (const moe_common_op_role_lookup_cache_entry& e : tls_cache) {
        if (e.engine == engine && e.generation == generation && e.layer == layer && e.role == role) {
            if (e.found && out_desc) {
                *out_desc = e.desc;
            }
            return e.found ? e.rec : nullptr;
        }
    }
    moe_raw_tensor_desc desc{};
    const moe_gguf_common_tensor_record* rec =
        find_common_tensor_by_op_role_uncached(engine, role, layer, &desc);
    if (tls_cache.size() > 256u) {
        tls_cache.erase(tls_cache.begin(), tls_cache.begin() + 64);
    }
    moe_common_op_role_lookup_cache_entry entry{};
    entry.engine = engine;
    entry.generation = generation;
    entry.layer = layer;
    entry.role = role;
    entry.rec = rec;
    entry.desc = desc;
    entry.found = rec ? 1 : 0;
    tls_cache.push_back(entry);
    if (rec && out_desc) {
        *out_desc = desc;
    }
    return rec;
}

static int common_tensor_backing_file_ready(
    const moe_pc_engine* engine,
    const moe_gguf_common_tensor_record* rec
) {
    if (!engine || !rec || rec->resolved_path_index >= engine->tensor_paths.size() || rec->bytes == 0) {
        return 0;
    }
    uint64_t file_bytes = 0;
    const std::string& path = engine->tensor_paths[rec->resolved_path_index];
    return file_size_bytes(path, &file_bytes) &&
        rec->file_offset <= file_bytes &&
        rec->bytes <= file_bytes - rec->file_offset ? 1 : 0;
}

static int map_common_tensor_record(
    moe_pc_engine* engine,
    const moe_gguf_common_tensor_record& rec,
    const moe_raw_tensor_desc& desc,
    moe_raw_tensor_cache_entry* out_entry
) {
    if (!engine || !out_entry || rec.resolved_path_index >= engine->tensor_paths.size() ||
        rec.bytes == 0 || desc.bytes == 0) {
        return 0;
    }
    const std::string& path = engine->tensor_paths[rec.resolved_path_index];
    mmap_context_t* mapped = get_or_map_shard_partial(engine, path, rec.file_offset, rec.bytes);
    if (!mapped || !mmap_get_ptr(mapped, 0) || rec.bytes > (uint64_t)mmap_get_size(mapped)) {
        return 0;
    }
    moe_raw_tensor_cache_entry entry{};
    entry.span = nullptr;
    entry.mapped = mapped;
    entry.base = static_cast<const uint8_t*>(mmap_get_ptr(mapped, 0));
    entry.rel_offset = 0;
    entry.bytes = rec.bytes;
    entry.rows = desc.rows;
    entry.cols = desc.cols;
    entry.scalar_encoding = moe_raw_scalar_encoding_from_common_tensor(rec, desc.rows, desc.cols);
    entry.weight_format = moe_common_tensor_effective_weight_format(rec, desc);
    entry.math_transposed = rec.math_transposed;
    entry.physical_rows = entry.rows;
    entry.physical_cols = entry.cols;
    if (rec.math_transposed) {
        if (rec.source_dims >= 2 && rec.source_shape[0] && rec.source_shape[1]) {
            entry.physical_cols = (uint32_t)std::min<uint64_t>(rec.source_shape[0], UINT32_MAX);
            entry.physical_rows = (uint32_t)std::min<uint64_t>(rec.source_shape[1], UINT32_MAX);
        } else {
            entry.physical_rows = desc.cols;
            entry.physical_cols = desc.rows;
        }
    }
    if (entry.weight_format == moe_WEIGHT_ENCODING_UNKNOWN) {
        moe_trace_common_tensor_unknown_weight_format(rec, desc);
        return 0;
    }
    if (!moe_common_tensor_raw_weight_format_supported(entry.weight_format)) {
        moe_trace_common_tensor_unsupported_weight_format(rec, desc, entry.weight_format);
        return 0;
    }
    moe_tensor_info_t info{};
    info.rows = desc.rows;
    info.cols = desc.cols;
    info.weight_byte_length = rec.bytes;
    entry.weight_row_bytes = infer_weight_row_bytes(info, entry.weight_format);
    if (rec.weight_row_bytes >= entry.weight_row_bytes &&
        entry.rows > 0 &&
        rec.weight_row_bytes <= entry.bytes &&
        rec.weight_row_bytes <= UINT64_MAX / (uint64_t)entry.rows &&
        (uint64_t)entry.rows * rec.weight_row_bytes <= entry.bytes) {
        entry.weight_row_bytes = rec.weight_row_bytes;
    }
    if (entry.rows) {
        const uint64_t rows64 = (uint64_t)entry.rows;
        const int stride_out_of_range =
            entry.weight_row_bytes &&
            (entry.weight_row_bytes > UINT64_MAX / rows64 ||
             entry.weight_row_bytes * rows64 > entry.bytes);
        if ((!entry.weight_row_bytes || stride_out_of_range) &&
            entry.bytes % rows64 == 0) {
            entry.weight_row_bytes = entry.bytes / rows64;
        }
    }
    *out_entry = entry;
    return 1;
}

static int raw_tensor_desc_for_kind(moe_raw_tensor_kind_t kind, moe_raw_tensor_desc* out) {
    if (!out) {
        return 0;
    }
    switch (kind) {
        case moe_RAW_TENSOR_INPUT_LAYER_NORM:
            *out = {moe_RAW_ROLE_NORM, 12288ull, 0u, 0ull, 12288ull, 6144u, 0u};
            return 1;
        case moe_RAW_TENSOR_POST_ATTENTION_LAYER_NORM:
            *out = {moe_RAW_ROLE_NORM, 12288ull, 1u, 0ull, 12288ull, 6144u, 0u};
            return 1;
        case moe_RAW_TENSOR_Q_A_LAYER_NORM:
            *out = {moe_RAW_ROLE_NORM, 4096ull, 0u, 0ull, 4096ull, 2048u, 0u};
            return 1;
        case moe_RAW_TENSOR_KV_A_LAYER_NORM:
            *out = {moe_RAW_ROLE_NORM, 1024ull, 0u, 0ull, 1024ull, 512u, 0u};
            return 1;
        case moe_RAW_TENSOR_V_A_LAYER_NORM:
            return 0;
        case moe_RAW_TENSOR_Q_A_PROJ:
            *out = {moe_RAW_ROLE_ATTENTION, 92274688ull, 0u, 0ull, 25165824ull, 2048u, 6144u};
            return 1;
        case moe_RAW_TENSOR_Q_B_PROJ:
            *out = {moe_RAW_ROLE_ATTENTION, 92274688ull, 0u, 25165824ull, 67108864ull, 16384u, 2048u};
            return 1;
        case moe_RAW_TENSOR_KV_A_PROJ_WITH_MQA:
            *out = {moe_RAW_ROLE_ATTENTION, 237764608ull, 0u, 0ull, 7077888ull, 576u, 6144u};
            return 1;
        case moe_RAW_TENSOR_KV_B_PROJ:
            *out = {moe_RAW_ROLE_ATTENTION, 237764608ull, 0u, 7077888ull, 29360128ull, 28672u, 512u};
            return 1;
        case moe_RAW_TENSOR_O_PROJ:
            *out = {moe_RAW_ROLE_ATTENTION, 237764608ull, 0u, 36438016ull, 201326592ull, 6144u, 16384u};
            return 1;
        case moe_RAW_TENSOR_INDEXER_K_NORM_BIAS:
            *out = {moe_RAW_ROLE_ATTENTION, 18743808ull, 0u, 0ull, 256ull, 128u, 0u};
            return 1;
        case moe_RAW_TENSOR_INDEXER_K_NORM_WEIGHT:
            *out = {moe_RAW_ROLE_ATTENTION, 18743808ull, 0u, 256ull, 256ull, 128u, 0u};
            return 1;
        case moe_RAW_TENSOR_INDEXER_WEIGHTS_PROJ:
            *out = {moe_RAW_ROLE_ATTENTION, 18743808ull, 0u, 512ull, 393216ull, 32u, 6144u};
            return 1;
        case moe_RAW_TENSOR_INDEXER_WK:
            *out = {moe_RAW_ROLE_ATTENTION, 18743808ull, 0u, 393728ull, 1572864ull, 128u, 6144u};
            return 1;
        case moe_RAW_TENSOR_INDEXER_WQ_B:
            *out = {moe_RAW_ROLE_ATTENTION, 18743808ull, 0u, 1966592ull, 16777216ull, 4096u, 2048u};
            return 1;
        default:
            return 0;
    }
}

static int find_raw_tensor_desc_and_span(
    int32_t layer,
    moe_raw_tensor_kind_t kind,
    moe_raw_tensor_desc* out_desc,
    const moe_storage_raw_span_spec_t** out_span
) {
    moe_raw_tensor_desc desc{};
    if (!raw_tensor_desc_for_kind(kind, &desc)) {
        return 0;
    }
    const moe_storage_raw_span_spec_t* span =
        find_raw_role_span_bytes(layer, desc.role, desc.span_bytes, desc.occurrence);
    if (!span && desc.role == moe_RAW_ROLE_ATTENTION) {
        // Boundary layers can pack the 18,743,808-byte indexer span and the
        // 92,274,688-byte Q span into one 111,018,496-byte attention span.
        const uint64_t combined_indexer_q_bytes = 18743808ull + 92274688ull;
        if (desc.span_bytes == 92274688ull) {
            const moe_storage_raw_span_spec_t* combined =
                find_raw_role_span_bytes(layer, moe_RAW_ROLE_ATTENTION, combined_indexer_q_bytes, 0);
            if (combined) {
                desc.rel_offset += 18743808ull;
                span = combined;
            }
        } else if (desc.span_bytes == 18743808ull) {
            span = find_raw_role_span_bytes(layer, moe_RAW_ROLE_ATTENTION, combined_indexer_q_bytes, 0);
        }
    }
    if (!span || desc.rel_offset > span->bytes || desc.bytes > span->bytes - desc.rel_offset) {
        return 0;
    }
    if (out_desc) {
        *out_desc = desc;
    }
    if (out_span) {
        *out_span = span;
    }
    return 1;
}
