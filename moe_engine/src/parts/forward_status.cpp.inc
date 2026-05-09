static int moe_forward_tokenizer_asset_ready(const moe_pc_engine* engine) {
    if (!engine || engine->model_root.empty()) {
        return 0;
    }
    static const char* kTokenizerFiles[] = {
        "tokenizer.json",
        "tokenizer.model",
        "sentencepiece.bpe.model",
        "tiktoken.model",
        "vocab.json",
        "merges.txt"
    };
    static const char* kTokenizerDirs[] = {
        "",
        "tokenizer"
    };
    for (const char* dir : kTokenizerDirs) {
        for (const char* file : kTokenizerFiles) {
            const std::string rel = dir && dir[0] ?
                storagellm::path_join(std::string(dir), std::string(file)) :
                std::string(file);
            uint64_t bytes = 0;
            if (file_size_bytes(resolve_model_file_path(engine->model_root, rel, ""), &bytes) && bytes > 0) {
                return 1;
            }
        }
    }
    return 0;
}

static int moe_forward_kv_cache_ready(const moe_pc_engine* engine, int attention_decode_ready) {
    if (!engine || !attention_decode_ready) {
        return 0;
    }
    return engine->config.kv_mode == moe_KV_MODE_PLAIN ||
        engine->config.kv_mode == moe_KV_MODE_QKV;
}

static int moe_forward_expert_triplet_records_ready(const moe_pc_engine* engine) {
    if (!engine || engine->tensors.empty()) {
        return 0;
    }
    const moe_storage_constants_t storage = moe_engine_storage_constants_or_static(engine);
    const uint32_t hidden_size = storage.hidden_size;
    const uint32_t intermediate_size = storage.expert_intermediate_size;
    if (hidden_size == 0 || intermediate_size == 0) {
        return 0;
    }

    for (const moe_tensor_record& gate : engine->tensors) {
        if (gate.info.proj != moe_PROJ_GATE || gate.info.weight_byte_length == 0 ||
            gate.info.rows != intermediate_size || gate.info.cols != hidden_size) {
            continue;
        }
        const moe_tensor_record* up = find_tensor_record(
            const_cast<moe_pc_engine*>(engine), gate.info.layer, gate.info.expert, moe_PROJ_UP);
        const moe_tensor_record* down = find_tensor_record(
            const_cast<moe_pc_engine*>(engine), gate.info.layer, gate.info.expert, moe_PROJ_DOWN);
        if (up && down &&
            up->info.weight_byte_length > 0 &&
            down->info.weight_byte_length > 0 &&
            up->info.rows == intermediate_size &&
            up->info.cols == hidden_size &&
            down->info.rows == hidden_size &&
            down->info.cols == intermediate_size) {
            return 1;
        }
    }
    return 0;
}

int moe_pc_engine_get_forward_status(
    const moe_pc_engine_t* engine,
    moe_forward_status_t* out_status
) {
    if (!engine || !out_status) {
        return 0;
    }
    std::memset(out_status, 0, sizeof(*out_status));
    out_status->storage_state_valid = moe_storage_model_state_valid();
    if (!out_status->storage_state_valid &&
        engine->offload_gguf_valid.load(std::memory_order_relaxed) &&
        engine->model_root_valid.load(std::memory_order_relaxed)) {
        out_status->storage_state_valid = 1;
    }
    // Fix 59: Use cached model_root_valid flag instead of 21× stat() per call
    if (!engine->model_root.empty() && !engine->model_root_valid.load(std::memory_order_relaxed)) {
        out_status->storage_state_valid = 0;
    }
    out_status->tensor_count = moe_pc_engine_tensor_count(engine);
    out_status->forward_adapter = engine->forward_adapter.load(std::memory_order_relaxed);
    out_status->forward_adapter_executable =
        engine->forward_adapter_executable.load(std::memory_order_relaxed);
    out_status->dynamic_shape_ready = engine->dynamic_shape_valid.load(std::memory_order_relaxed);
    out_status->graph_ir_ready = engine->offload_graph_ir_ready.load(std::memory_order_relaxed);
    out_status->graph_ir_required = engine->offload_graph_ir_required.load(std::memory_order_relaxed);
    out_status->graph_ir_version = engine->offload_graph_ir_version.load(std::memory_order_relaxed);
    out_status->graph_ir_layer_count = engine->offload_graph_ir_layer_count.load(std::memory_order_relaxed);
    out_status->graph_ir_op_count = engine->offload_graph_ir_op_count.load(std::memory_order_relaxed);
    out_status->runtime_execution_manifest_ready =
        engine->juju_runtime_execution_manifest_ready.load(std::memory_order_relaxed) ? 1 : 0;
    out_status->runtime_access_plan_ready =
        engine->juju_runtime_access_plan_ready.load(std::memory_order_relaxed) ? 1 : 0;
    out_status->runtime_access_hotset_count =
        engine->juju_runtime_access_hotset_count.load(std::memory_order_relaxed);
    out_status->runtime_access_file_group_count =
        engine->juju_runtime_access_file_group_count.load(std::memory_order_relaxed);
    out_status->runtime_access_executor_tensor_count =
        engine->juju_runtime_executor_tensor_count.load(std::memory_order_relaxed);
    out_status->runtime_access_layer_prefetch_plan_count =
        engine->juju_runtime_layer_prefetch_plan_count.load(std::memory_order_relaxed);
    out_status->kv_layout_contract_ready =
        engine->juju_kv_layout_contract_ready.load(std::memory_order_relaxed) ? 1 : 0;
    out_status->kv_layout_max_entry_dim =
        engine->juju_kv_layout_max_entry_dim.load(std::memory_order_relaxed);
    out_status->kv_layout_page_size_tokens =
        engine->juju_kv_layout_page_size_tokens.load(std::memory_order_relaxed);
    if (out_status->dynamic_shape_ready) {
        out_status->dynamic_num_hidden_layers = engine->dynamic_shape.num_hidden_layers;
        out_status->dynamic_hidden_size = engine->dynamic_shape.hidden_size;
        out_status->dynamic_vocab_size = engine->dynamic_shape.vocab_size;
    }
    if (engine->attn_config_valid.load(std::memory_order_relaxed)) {
        out_status->dynamic_max_seq_len = engine->attn_config.max_seq_len;
    }
    out_status->tensor_table_loaded = out_status->tensor_count > 0;
    out_status->expert_triplet_available =
        out_status->tensor_table_loaded &&
        moe_forward_expert_triplet_records_ready(engine);
    {
        int cached = engine->tokenizer_valid.load(std::memory_order_relaxed);
        if (cached < 0) {
            cached = moe_forward_tokenizer_asset_ready(engine);
            engine->tokenizer_valid.store(cached, std::memory_order_relaxed);
        }
        out_status->tokenizer_ready = cached;
    }
    out_status->embedding_ready = moe_raw_embedding_ready(engine);
    out_status->lm_head_ready = moe_raw_lm_head_ready(engine);
    // Fix 72: Cache attention projector readiness (720× stat() ??atomic load)
    // -1=unchecked, recheck needed; 0=not ready; 1=ready (sticky until model_root change)
    {
        int cached = engine->attention_projector_valid.load(std::memory_order_relaxed);
        if (cached < 0) {
            cached = moe_attention_projectors_ready(engine);
            engine->attention_projector_valid.store(cached, std::memory_order_relaxed);
        }
        out_status->attention_projector_ready = cached;
    }
    out_status->attention_decode_ready = out_status->attention_projector_ready;
    out_status->kv_cache_ready = moe_forward_kv_cache_ready(engine, out_status->attention_decode_ready);
    out_status->attention_ready =
        out_status->attention_projector_ready &&
        out_status->attention_decode_ready;
    out_status->sampler_ready = out_status->lm_head_ready;
    out_status->decode_loop_ready =
        out_status->forward_adapter_executable &&
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
        out_status->forward_adapter_executable &&
        out_status->tokenizer_ready &&
        out_status->embedding_ready &&
        out_status->lm_head_ready &&
        out_status->attention_ready &&
        out_status->sampler_ready &&
        out_status->decode_loop_ready;
    return 1;
}
