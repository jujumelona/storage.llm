static int glm5_forward_tokenizer_asset_ready(const glm5_pc_engine* engine) {
    if (!engine || engine->model_root.empty()) {
        return 0;
    }
    static const char* kTokenizerFiles[] = {
        "tokenizer.json",
        "tokenizer.model"
    };
    for (const char* file : kTokenizerFiles) {
        uint64_t bytes = 0;
        if (file_size_bytes(resolve_model_file_path(engine->model_root, file, ""), &bytes) && bytes > 0) {
            return 1;
        }
    }
    return 0;
}

static int glm5_forward_kv_cache_ready(const glm5_pc_engine* engine, int attention_decode_ready) {
    if (!engine || !attention_decode_ready) {
        return 0;
    }
    return engine->config.kv_mode == GLM5_KV_MODE_PLAIN ||
        engine->config.kv_mode == GLM5_KV_MODE_QKV;
}

int glm5_pc_engine_get_forward_status(
    const glm5_pc_engine_t* engine,
    glm5_forward_status_t* out_status
) {
    if (!engine || !out_status) {
        return 0;
    }
    std::memset(out_status, 0, sizeof(*out_status));
    out_status->storage_state_valid = glm5_storage_model_state_valid();
    // Fix 59: Use cached model_root_valid flag instead of 21× stat() per call
    if (!engine->model_root.empty() && !engine->model_root_valid.load(std::memory_order_relaxed)) {
        out_status->storage_state_valid = 0;
    }
    out_status->tensor_count = glm5_pc_engine_tensor_count(engine);
    out_status->tensor_table_loaded = out_status->tensor_count > 0;
    out_status->expert_triplet_available =
        out_status->storage_state_valid && out_status->tensor_table_loaded;
    {
        int cached = engine->tokenizer_valid.load(std::memory_order_relaxed);
        if (cached < 0) {
            cached = glm5_forward_tokenizer_asset_ready(engine);
            engine->tokenizer_valid.store(cached, std::memory_order_relaxed);
        }
        out_status->tokenizer_ready = cached;
    }
    out_status->embedding_ready = glm5_raw_embedding_ready(engine);
    out_status->lm_head_ready = glm5_raw_lm_head_ready(engine);
    // Fix 72: Cache attention projector readiness (720× stat() → atomic load)
    // -1=unchecked, recheck needed; 0=not ready; 1=ready (sticky until model_root change)
    {
        int cached = engine->attention_projector_valid.load(std::memory_order_relaxed);
        if (cached < 0) {
            cached = glm5_attention_projectors_ready(engine);
            engine->attention_projector_valid.store(cached, std::memory_order_relaxed);
        }
        out_status->attention_projector_ready = cached;
    }
    out_status->attention_decode_ready = out_status->attention_projector_ready;
    out_status->kv_cache_ready = glm5_forward_kv_cache_ready(engine, out_status->attention_decode_ready);
    out_status->attention_ready =
        out_status->attention_projector_ready &&
        out_status->attention_decode_ready;
    out_status->sampler_ready = out_status->lm_head_ready;
    out_status->decode_loop_ready =
        out_status->tokenizer_ready &&
        out_status->embedding_ready &&
        out_status->lm_head_ready &&
        out_status->expert_triplet_available &&
        out_status->attention_ready &&
        out_status->kv_cache_ready &&
        out_status->sampler_ready;
    out_status->chat_graph_ready =
        out_status->storage_state_valid &&
        out_status->tensor_table_loaded &&
        out_status->expert_triplet_available &&
        out_status->tokenizer_ready &&
        out_status->embedding_ready &&
        out_status->lm_head_ready &&
        out_status->attention_ready &&
        out_status->sampler_ready &&
        out_status->decode_loop_ready;
    return 1;
}
