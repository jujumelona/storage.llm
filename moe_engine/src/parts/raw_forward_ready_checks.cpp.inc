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

static int moe_attention_projectors_ready(const moe_pc_engine* engine) {
    const moe_storage_constants_t* c = moe_storage_constants();
    if (!engine || engine->model_root.empty() || !c || c->num_hidden_layers == 0) {
        return 0;
    }
    const moe_raw_tensor_kind_t required[] = {
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
    for (uint32_t layer = 0; layer < c->num_hidden_layers; ++layer) {
        for (moe_raw_tensor_kind_t kind : required) {
            const moe_storage_raw_span_spec_t* span = nullptr;
            if (!find_raw_tensor_desc_and_span((int32_t)layer, kind, nullptr, &span) ||
                !raw_span_backing_file_ready(engine, span)) {
                return 0;
            }
        }
    }
    return 1;
}

static int moe_raw_embedding_ready(const moe_pc_engine* engine) {
    const moe_storage_constants_t* c = moe_storage_constants();
    const moe_storage_raw_span_spec_t* span = find_raw_role_span(-1, moe_RAW_ROLE_EMBED);
    if (!engine || engine->model_root.empty() || !c || !span ||
        c->hidden_size == 0 || c->vocab_size == 0) {
        return 0;
    }
    return span->bytes >= (uint64_t)c->hidden_size * (uint64_t)c->vocab_size * 2ull &&
        raw_span_backing_file_ready(engine, span) ? 1 : 0;
}

static int moe_raw_lm_head_ready(const moe_pc_engine* engine) {
    const moe_storage_constants_t* c = moe_storage_constants();
    const int32_t lm_head_layer = c ? (int32_t)c->num_hidden_layers + 1 : 80;
    const moe_storage_raw_span_spec_t* span = find_raw_role_span(lm_head_layer, moe_RAW_ROLE_LM_HEAD);
    if (!engine || engine->model_root.empty() || !c || !span ||
        c->hidden_size == 0 || c->vocab_size == 0) {
        return 0;
    }
    return span->bytes >= (uint64_t)c->hidden_size * (uint64_t)c->vocab_size * 2ull &&
        raw_span_backing_file_ready(engine, span) ? 1 : 0;
}
