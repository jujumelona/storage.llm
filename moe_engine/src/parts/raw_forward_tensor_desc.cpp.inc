// Raw forward - Tensor descriptor definitions

struct moe_raw_tensor_desc {
    uint32_t role;
    uint64_t span_bytes;
    uint32_t occurrence;
    uint64_t rel_offset;
    uint64_t bytes;
    uint32_t rows;
    uint32_t cols;
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
    if (rec.dims == 1) {
        desc.rows = (uint32_t)std::min<uint64_t>(rec.shape[0], UINT32_MAX);
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
            out->push_back("output_norm.weight");
            out->push_back("model.norm.weight");
            out->push_back("norm.weight");
            out->push_back("transformer.ln_f.weight");
            break;
        }
        case moe_RAW_TENSOR_POST_ATTENTION_LAYER_NORM: {
            static const char* const suffixes[] = {
                "post_attention_norm.weight",
                "post_attention_layernorm.weight",
                "post_attention_layer_norm.weight",
                "post_attn_norm.weight",
                "ffn_norm.weight",
                "mlp_norm.weight",
                "post_attention_layernorm.weight",
                "pre_ffw_norm.weight",
                "ln_2.weight"
            };
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

static const moe_gguf_common_tensor_record* find_common_raw_tensor_record(
    const moe_pc_engine* engine,
    int32_t layer,
    moe_raw_tensor_kind_t kind,
    moe_raw_tensor_desc* out_desc
) {
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
    if ((!entry.weight_row_bytes ||
         entry.weight_row_bytes * (uint64_t)entry.rows > entry.bytes) &&
        entry.rows && entry.bytes % (uint64_t)entry.rows == 0) {
        entry.weight_row_bytes = entry.bytes / (uint64_t)entry.rows;
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
