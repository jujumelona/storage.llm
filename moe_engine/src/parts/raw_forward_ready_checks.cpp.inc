// Raw forward - Ready state checks

static int raw_span_backing_file_ready(
    const moe_pc_engine* engine,
    const moe_storage_raw_span_spec_t* span
) {
    if (!engine || engine->model_root.empty() || !span) {
        return 0;
    }
    const moe_storage_part_spec_t* part = moe_storage_part_by_id(span->part);
    if (!part || !part->path) {
        return 0;
    }
    uint64_t file_bytes = 0;
    {
        std::lock_guard<std::mutex> lock(engine->raw_file_size_cache_mutex);
        auto it = engine->raw_file_size_cache.find(span->part);
        if (it != engine->raw_file_size_cache.end()) {
            file_bytes = it->second;
        } else {
            const std::string path = resolve_model_file_path(engine->model_root, part->path, "");
            if (!file_size_bytes(path, &file_bytes)) {
                return 0;
            }
            engine->raw_file_size_cache[span->part] = file_bytes;
        }
    }
    return span->offset <= file_bytes && span->bytes <= file_bytes - span->offset ? 1 : 0;
}

static int raw_tensor_kind_available(
    const moe_pc_engine* engine,
    int32_t layer,
    moe_raw_tensor_kind_t kind
) {
    if (const moe_gguf_common_tensor_record* rec =
            find_common_raw_tensor_record(engine, layer, kind, nullptr)) {
        return common_tensor_backing_file_ready(engine, rec);
    }
    const moe_storage_raw_span_spec_t* span = nullptr;
    return find_raw_tensor_desc_and_span(layer, kind, nullptr, &span) &&
        raw_span_backing_file_ready(engine, span) ? 1 : 0;
}

static int moe_attention_projectors_ready(const moe_pc_engine* engine) {
    const moe_storage_constants_t runtime_c = moe_engine_storage_constants_or_static(engine);
    const moe_storage_constants_t* c = &runtime_c;
    if (!engine || engine->model_root.empty() || !c || c->num_hidden_layers == 0) {
        return 0;
    }
    const int graph_ir_ready = engine->offload_graph_ir_ready.load(std::memory_order_relaxed);
    if (engine->offload_gguf_valid.load(std::memory_order_relaxed) &&
        !graph_ir_ready &&
        !engine->forward_adapter_executable.load(std::memory_order_relaxed)) {
        return 0;
    }
    if (graph_ir_ready &&
        engine->forward_adapter_executable.load(std::memory_order_relaxed) &&
        engine->attn_config_valid.load(std::memory_order_relaxed) &&
        engine->dynamic_shape_valid.load(std::memory_order_relaxed) &&
        !engine->gguf_common_tensor_slots.empty()) {
        return 1;
    }
    moe_attn_type_t attn_type = moe_ATTN_TYPE_MLA;
    if (engine->attn_config_valid.load(std::memory_order_relaxed)) {
        attn_type = engine->attn_config.type;
    } else if (graph_ir_ready) {
        attn_type = moe_ATTN_TYPE_MHA;
    }
    const moe_raw_tensor_kind_t required_mla[] = {
        moe_RAW_TENSOR_INPUT_LAYER_NORM,
        moe_RAW_TENSOR_POST_ATTENTION_LAYER_NORM,
        moe_RAW_TENSOR_Q_A_LAYER_NORM,
        moe_RAW_TENSOR_KV_A_LAYER_NORM,
        moe_RAW_TENSOR_Q_A_PROJ,
        moe_RAW_TENSOR_Q_B_PROJ,
        moe_RAW_TENSOR_KV_A_PROJ_WITH_MQA,
        moe_RAW_TENSOR_KV_B_PROJ,
        moe_RAW_TENSOR_O_PROJ
    };
    const moe_raw_tensor_kind_t required_standard[] = {
        moe_RAW_TENSOR_INPUT_LAYER_NORM,
        moe_RAW_TENSOR_POST_ATTENTION_LAYER_NORM,
        moe_RAW_TENSOR_Q_A_PROJ,
        moe_RAW_TENSOR_KV_A_PROJ_WITH_MQA,
        moe_RAW_TENSOR_KV_B_PROJ,
        moe_RAW_TENSOR_O_PROJ
    };
    const moe_raw_tensor_kind_t* required =
        attn_type == moe_ATTN_TYPE_MLA ? required_mla : required_standard;
    const size_t required_count = attn_type == moe_ATTN_TYPE_MLA ?
        sizeof(required_mla) / sizeof(required_mla[0]) :
        sizeof(required_standard) / sizeof(required_standard[0]);
    for (uint32_t layer = 0; layer < c->num_hidden_layers; ++layer) {
        for (size_t i = 0; i < required_count; ++i) {
            const moe_raw_tensor_kind_t kind = required[i];
            if (!raw_tensor_kind_available(engine, (int32_t)layer, kind)) {
                return 0;
            }
        }
    }
    return 1;
}

static int moe_gguf_common_tensor_ready(const moe_pc_engine* engine, const char* name) {
    if (!engine || !name || !name[0]) {
        return 0;
    }
    return engine->gguf_common_tensor_slots.find(moe_lower_ascii_copy(name)) !=
        engine->gguf_common_tensor_slots.end() ? 1 : 0;
}

static int moe_raw_embedding_ready(const moe_pc_engine* engine) {
    const moe_storage_constants_t runtime_c = moe_engine_storage_constants_or_static(engine);
    const moe_storage_constants_t* c = &runtime_c;
    const moe_storage_raw_span_spec_t* span = find_raw_role_span(-1, moe_RAW_ROLE_EMBED);
    static const char* const embedding_names[] = {
        "token_embd.weight",
        "model.embed_tokens.weight",
        "embed_tokens.weight",
        "transformer.wte.weight"
    };
    if (const moe_gguf_common_tensor_record* rec =
            find_common_tensor_by_names(engine, embedding_names, sizeof(embedding_names) / sizeof(embedding_names[0]), nullptr)) {
        return common_tensor_backing_file_ready(engine, rec);
    }
    if (engine && engine->offload_gguf_valid.load(std::memory_order_relaxed)) {
        return 0;
    }
    if (!engine || engine->model_root.empty() || !c || !span ||
        c->hidden_size == 0 || c->vocab_size == 0) {
        return 0;
    }
    return span->bytes >= (uint64_t)c->hidden_size * (uint64_t)c->vocab_size * 2ull &&
        raw_span_backing_file_ready(engine, span) ? 1 : 0;
}

static int moe_raw_lm_head_ready(const moe_pc_engine* engine) {
    const moe_storage_constants_t runtime_c = moe_engine_storage_constants_or_static(engine);
    const moe_storage_constants_t* c = &runtime_c;
    const int32_t lm_head_layer = c ? (int32_t)c->num_hidden_layers + 1 : 80;
    const moe_storage_raw_span_spec_t* span = find_raw_role_span(lm_head_layer, moe_RAW_ROLE_LM_HEAD);
    static const char* const lm_head_names[] = {
        "output.weight",
        "lm_head.weight",
        "token_embd.weight",
        "model.embed_tokens.weight",
        "embed_tokens.weight"
    };
    if (const moe_gguf_common_tensor_record* rec =
            find_common_tensor_by_names(engine, lm_head_names, sizeof(lm_head_names) / sizeof(lm_head_names[0]), nullptr)) {
        return common_tensor_backing_file_ready(engine, rec);
    }
    if (engine && engine->offload_gguf_valid.load(std::memory_order_relaxed)) {
        return 0;
    }
    if (!engine || engine->model_root.empty() || !c || !span ||
        c->hidden_size == 0 || c->vocab_size == 0) {
        return 0;
    }
    return span->bytes >= (uint64_t)c->hidden_size * (uint64_t)c->vocab_size * 2ull &&
        raw_span_backing_file_ready(engine, span) ? 1 : 0;
}
