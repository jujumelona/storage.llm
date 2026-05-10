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
    uint64_t gate = 0;
    uint64_t up = 0;
    uint64_t down = 0;
    for (const moe_tensor_record& rec : engine->tensors) {
        if (rec.info.weight_byte_length == 0) {
            continue;
        }
        if (rec.info.proj == moe_PROJ_GATE) {
            ++gate;
        } else if (rec.info.proj == moe_PROJ_UP) {
            ++up;
        } else if (rec.info.proj == moe_PROJ_DOWN) {
            ++down;
        }
    }
    return gate > 0 && up > 0 && down > 0 ? 1 : 0;
}

int moe_pc_engine_get_forward_status(
    const moe_pc_engine_t* engine,
    moe_forward_status_t* out_status
) {
    if (!engine || !out_status) {
        return 0;
    }
    std::memset(out_status, 0, sizeof(*out_status));
    moe_cpu_storage_ensure_runtime(const_cast<moe_pc_engine_t*>(engine));
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
    out_status->cpu_only_runtime = engine->cpu_only_runtime.load(std::memory_order_relaxed);
    out_status->cpu_storage_engine_ready = engine->cpu_storage_engine_ready.load(std::memory_order_relaxed);
    out_status->cpu_storage_execution_plan_ready = engine->cpu_storage_execution_plan_ready.load(std::memory_order_relaxed);
    out_status->cpu_quant_registry_ready = engine->cpu_quant_registry_ready.load(std::memory_order_relaxed);
    out_status->graph_ir_used_for_forward = engine->graph_ir_used_for_forward.load(std::memory_order_relaxed);
    out_status->gpu_path_disabled_for_cpu_only = engine->gpu_path_disabled_for_cpu_only.load(std::memory_order_relaxed);
    out_status->cpu_execution_plan_layers = engine->cpu_execution_plan_layers.load(std::memory_order_relaxed);
    out_status->cpu_execution_plan_tensors = engine->cpu_execution_plan_tensors.load(std::memory_order_relaxed);
    out_status->cpu_execution_plan_expert_tensors = engine->cpu_execution_plan_expert_tensors.load(std::memory_order_relaxed);
    out_status->cpu_execution_plan_bytes = engine->cpu_execution_plan_bytes.load(std::memory_order_relaxed);
    out_status->cpu_prefetch_scheduled = engine->cpu_prefetch_scheduled.load(std::memory_order_relaxed);
    out_status->cpu_selected_expert_prefetch_scheduled = engine->cpu_selected_expert_prefetch_scheduled.load(std::memory_order_relaxed);
    out_status->cpu_layer_begin_count = engine->cpu_layer_begin_count.load(std::memory_order_relaxed);
out_status->cpu_hot_tensor_views_ready = engine->cpu_hot_tensor_views_ready.load(std::memory_order_relaxed);
out_status->cpu_hot_tensor_view_count = engine->cpu_hot_tensor_view_count.load(std::memory_order_relaxed);
out_status->cpu_hot_tensor_simple_dot_count = engine->cpu_hot_tensor_simple_dot_count.load(std::memory_order_relaxed);
out_status->cpu_hot_tensor_bytes = engine->cpu_hot_tensor_bytes.load(std::memory_order_relaxed);
out_status->cpu_hot_tensor_reused = engine->cpu_hot_tensor_reused.load(std::memory_order_relaxed);
out_status->cpu_layer_hot_plan_ready = engine->cpu_layer_hot_plan_ready.load(std::memory_order_relaxed);
out_status->cpu_layer_hot_tensor_count = engine->cpu_layer_hot_tensor_count.load(std::memory_order_relaxed);
out_status->cpu_layer_hot_prefetch_count = engine->cpu_layer_hot_prefetch_count.load(std::memory_order_relaxed);
out_status->cpu_layer_hot_prefetch_bytes = engine->cpu_layer_hot_prefetch_bytes.load(std::memory_order_relaxed);
out_status->cpu_layer_prefetch_duplicate_skips = engine->cpu_layer_prefetch_duplicate_skips.load(std::memory_order_relaxed);
out_status->storage_io_plan_ready = engine->storage_io_plan_ready.load(std::memory_order_relaxed);
out_status->storage_profile = engine->storage_profile.load(std::memory_order_relaxed);
out_status->storage_prefetch_workers = engine->storage_prefetch_workers.load(std::memory_order_relaxed);
out_status->storage_prefetch_queue_depth = engine->storage_prefetch_queue_depth.load(std::memory_order_relaxed);
out_status->storage_io_span_count = engine->storage_io_span_count.load(std::memory_order_relaxed);
out_status->storage_io_merged_span_count = engine->storage_io_merged_span_count.load(std::memory_order_relaxed);
out_status->storage_io_prefetch_count = engine->storage_io_prefetch_count.load(std::memory_order_relaxed);
out_status->storage_io_prefetch_bytes = engine->storage_io_prefetch_bytes.load(std::memory_order_relaxed);
out_status->storage_io_fadvise_count = engine->storage_io_fadvise_count.load(std::memory_order_relaxed);
out_status->storage_io_readahead_count = engine->storage_io_readahead_count.load(std::memory_order_relaxed);
out_status->storage_io_madvise_count = engine->storage_io_madvise_count.load(std::memory_order_relaxed);
out_status->storage_io_page_touch_count = engine->storage_io_page_touch_count.load(std::memory_order_relaxed);
out_status->storage_io_duplicate_skips = engine->storage_io_duplicate_skips.load(std::memory_order_relaxed);
out_status->storage_iouring_available = engine->storage_iouring_available.load(std::memory_order_relaxed);
out_status->storage_iouring_enabled = engine->storage_iouring_enabled.load(std::memory_order_relaxed);
out_status->storage_iouring_submit_count = engine->storage_iouring_submit_count.load(std::memory_order_relaxed);
out_status->storage_iouring_bytes = engine->storage_iouring_bytes.load(std::memory_order_relaxed);
out_status->storage_iouring_fallback_count = engine->storage_iouring_fallback_count.load(std::memory_order_relaxed);
out_status->storage_async_prefetch_enabled = engine->storage_async_prefetch_enabled.load(std::memory_order_relaxed);
out_status->storage_async_prefetch_enqueued = engine->storage_async_prefetch_enqueued.load(std::memory_order_relaxed);
out_status->storage_async_prefetch_completed = engine->storage_async_prefetch_completed.load(std::memory_order_relaxed);
out_status->storage_async_prefetch_dropped = engine->storage_async_prefetch_dropped.load(std::memory_order_relaxed);
out_status->storage_async_prefetch_inline = engine->storage_async_prefetch_inline.load(std::memory_order_relaxed);
out_status->storage_async_prefetch_pending = engine->storage_async_prefetch_pending.load(std::memory_order_relaxed);
out_status->storage_gpu_staging_plan_ready = engine->storage_gpu_staging_plan_ready.load(std::memory_order_relaxed);
out_status->storage_gpu_staging_hook_count = engine->storage_gpu_staging_hook_count.load(std::memory_order_relaxed);
out_status->storage_gpu_staging_bytes = engine->storage_gpu_staging_bytes.load(std::memory_order_relaxed);
out_status->storage_cpu_pagecache_hook_count = engine->storage_cpu_pagecache_hook_count.load(std::memory_order_relaxed);
out_status->storage_cpu_pagecache_bytes = engine->storage_cpu_pagecache_bytes.load(std::memory_order_relaxed);
out_status->storage_expert_index_ready = engine->storage_expert_index_ready.load(std::memory_order_relaxed);
out_status->storage_expert_index_entry_count = engine->storage_expert_index_entry_count.load(std::memory_order_relaxed);
out_status->storage_expert_index_view_count = engine->storage_expert_index_view_count.load(std::memory_order_relaxed);
out_status->storage_selected_expert_index_hits = engine->storage_selected_expert_index_hits.load(std::memory_order_relaxed);
out_status->storage_selected_expert_linear_fallbacks = engine->storage_selected_expert_linear_fallbacks.load(std::memory_order_relaxed);
out_status->storage_selected_expert_prefetch_bytes = engine->storage_selected_expert_prefetch_bytes.load(std::memory_order_relaxed);
out_status->storage_selected_expert_duplicate_skips = engine->storage_selected_expert_duplicate_skips.load(std::memory_order_relaxed);
out_status->storage_format_plan_available = engine->storage_format_plan_available.load(std::memory_order_relaxed);
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
        out_status->storage_state_valid &&
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
