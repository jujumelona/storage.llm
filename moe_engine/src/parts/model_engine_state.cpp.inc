// Engine state management
// QKV cache reset and contract clearing

static void stop_io_workers(moe_pc_engine* engine);
static void stop_cpu_row_workers(moe_pc_engine* engine);


static void moe_engine_stop_storage_prefetch_workers_for_reset(moe_pc_engine* engine) {
    if (!engine) return;
    engine->storage_prefetch_stop.store(1, std::memory_order_release);
    engine->storage_prefetch_queue_cv.notify_all();
    for (std::thread& t : engine->storage_prefetch_worker_threads) {
        if (t.joinable()) t.join();
    }
    engine->storage_prefetch_worker_threads.clear();
    {
        std::lock_guard<std::mutex> lock(engine->storage_prefetch_queue_mutex);
        engine->storage_prefetch_queue.clear();
    }
    engine->storage_prefetch_stop.store(0, std::memory_order_release);
    engine->storage_async_prefetch_pending.store(0, std::memory_order_relaxed);
    engine->storage_async_prefetch_enabled.store(0, std::memory_order_relaxed);
}

static void moe_engine_reset_storage_execution_state(moe_pc_engine* engine) {
    if (!engine) return;
    // These plans contain raw mmap pointers and tensor_record addresses.  They
    // must be cleared on model-root reload before shard_mmaps/tensors are
    // invalidated, otherwise the storage prefetch workers can keep warming stale
    // spans and the next model may accidentally reuse an old fast path.
    moe_engine_stop_storage_prefetch_workers_for_reset(engine);

    engine->cpu_hot_tensor_views.clear();
    engine->cpu_hot_tensor_views_ready.store(0, std::memory_order_relaxed);
    engine->cpu_hot_tensor_view_count.store(0, std::memory_order_relaxed);
    engine->cpu_hot_tensor_simple_dot_count.store(0, std::memory_order_relaxed);
    engine->cpu_hot_tensor_bytes.store(0, std::memory_order_relaxed);
    engine->cpu_hot_tensor_reused.store(0, std::memory_order_relaxed);

    engine->cpu_layer_hot_offsets.clear();
    engine->cpu_layer_hot_tensor_indices.clear();
    engine->cpu_layer_prefetch_epoch.clear();
    engine->cpu_prefetch_epoch.store(0, std::memory_order_relaxed);
    engine->cpu_layer_hot_plan_ready.store(0, std::memory_order_relaxed);
    engine->cpu_layer_hot_tensor_count.store(0, std::memory_order_relaxed);
    engine->cpu_layer_hot_prefetch_count.store(0, std::memory_order_relaxed);
    engine->cpu_layer_hot_prefetch_bytes.store(0, std::memory_order_relaxed);
    engine->cpu_layer_prefetch_duplicate_skips.store(0, std::memory_order_relaxed);

    engine->storage_io_spans.clear();
    engine->storage_layer_span_offsets.clear();
    engine->storage_layer_prefetch_epoch.clear();
    engine->storage_io_plan_ready.store(0, std::memory_order_relaxed);
    engine->storage_profile.store(moe_STORAGE_PROFILE_AUTO, std::memory_order_relaxed);
    engine->storage_prefetch_workers.store(0, std::memory_order_relaxed);
    engine->storage_prefetch_queue_depth.store(0, std::memory_order_relaxed);
    engine->storage_io_span_count.store(0, std::memory_order_relaxed);
    engine->storage_io_merged_span_count.store(0, std::memory_order_relaxed);
    engine->storage_io_prefetch_count.store(0, std::memory_order_relaxed);
    engine->storage_io_prefetch_bytes.store(0, std::memory_order_relaxed);
    engine->storage_io_fadvise_count.store(0, std::memory_order_relaxed);
    engine->storage_io_readahead_count.store(0, std::memory_order_relaxed);
    engine->storage_io_madvise_count.store(0, std::memory_order_relaxed);
    engine->storage_io_page_touch_count.store(0, std::memory_order_relaxed);
    engine->storage_io_duplicate_skips.store(0, std::memory_order_relaxed);

    engine->storage_iouring_available.store(0, std::memory_order_relaxed);
    engine->storage_iouring_enabled.store(0, std::memory_order_relaxed);
    engine->storage_iouring_submit_count.store(0, std::memory_order_relaxed);
    engine->storage_iouring_bytes.store(0, std::memory_order_relaxed);
    engine->storage_iouring_fallback_count.store(0, std::memory_order_relaxed);
    engine->storage_async_prefetch_enqueued.store(0, std::memory_order_relaxed);
    engine->storage_async_prefetch_completed.store(0, std::memory_order_relaxed);
    engine->storage_async_prefetch_dropped.store(0, std::memory_order_relaxed);
    engine->storage_async_prefetch_inline.store(0, std::memory_order_relaxed);

    engine->storage_gpu_staging_plan_ready.store(0, std::memory_order_relaxed);
    engine->storage_gpu_staging_hook_count.store(0, std::memory_order_relaxed);
    engine->storage_gpu_staging_bytes.store(0, std::memory_order_relaxed);
    engine->storage_cpu_pagecache_hook_count.store(0, std::memory_order_relaxed);
    engine->storage_cpu_pagecache_bytes.store(0, std::memory_order_relaxed);

    engine->storage_expert_view_offsets.clear();
    engine->storage_expert_view_indices.clear();
    engine->storage_expert_prefetch_epoch.clear();
    engine->storage_expert_index_ready.store(0, std::memory_order_relaxed);
    engine->storage_expert_index_entry_count.store(0, std::memory_order_relaxed);
    engine->storage_expert_index_view_count.store(0, std::memory_order_relaxed);
    engine->storage_selected_expert_index_hits.store(0, std::memory_order_relaxed);
    engine->storage_selected_expert_linear_fallbacks.store(0, std::memory_order_relaxed);
    engine->storage_selected_expert_prefetch_bytes.store(0, std::memory_order_relaxed);
    engine->storage_selected_expert_duplicate_skips.store(0, std::memory_order_relaxed);
    engine->storage_format_plan_available.store(0, std::memory_order_relaxed);
}

static void moe_engine_reset_qkv_cache(moe_pc_engine* engine) {
    if (!engine) {
        return;
    }
    std::lock_guard<std::mutex> lock(engine->qkv_mutex);
    for (size_t i = 0; i < engine->qkv_head_state_slots.size(); ++i) {
        if (i < engine->qkv_head_state_valid.size() && engine->qkv_head_state_valid[i]) {
            qkv_free(&engine->qkv_head_state_slots[i]);
        }
    }
    engine->qkv_head_state_slots.clear();
    engine->qkv_head_state_valid.clear();
    engine->qkv_head_state_locks.clear();
    engine->qkv_head_state_layers.clear();
    engine->qkv_head_state_heads.clear();
    engine->qkv_head_state_fingerprints.clear();
    engine->qkv_head_state_epochs.clear();
    engine->qkv_cfg_initialized.store(false, std::memory_order_release);
    engine->qkv_last_ctx = 0;
}

static void moe_engine_clear_offload_gguf_contract(moe_pc_engine* engine) {
    if (!engine) {
        return;
    }
    // BUGFIX 170: Stop workers before clearing state to prevent use-after-free
    // Problem: IO workers and CPU workers may be accessing expert_states, tensors, etc.
    // while clear() invalidates them. This causes iterator invalidation and crashes.
    // Solution: Stop all workers first, then clear state, then restart workers if needed.
    // Note: Caller (set_model_root) will restart workers after loading new model.

    // Stop IO workers (disk, pinned, gpu workers)
    stop_io_workers(engine);

    // Stop CPU row workers (parallel computation threads)
    stop_cpu_row_workers(engine);

    // Stop storage-required prefetch workers and clear all span/row-view plans
    // before tensors and mmaps are invalidated.
    moe_engine_reset_storage_execution_state(engine);

    // Now safe to clear all state
    engine->offload_gguf_valid.store(0, std::memory_order_relaxed);
    engine->offload_gguf_file_count.store(0, std::memory_order_relaxed);
    engine->offload_gguf_tensor_count.store(0, std::memory_order_relaxed);
    engine->offload_gguf_executable_tensor_count.store(0, std::memory_order_relaxed);
    engine->qkv_forced_by_format.store(0, std::memory_order_relaxed);
    engine->offload_qkv_cfg_available.store(false, std::memory_order_release);
    engine->offload_qkv_cfg = qkv_config_default(128);
    engine->offload_weight_quant_bits.store(0, std::memory_order_relaxed);
    engine->offload_weight_encoding.store(0, std::memory_order_relaxed);
    engine->offload_weight_block_size.store(0, std::memory_order_relaxed);
    engine->offload_optimization_catalog_count.store(0, std::memory_order_relaxed);
    engine->juju_strategy_valid.store(0, std::memory_order_relaxed);
    engine->juju_execution_path = moe_JUJU_EXEC_STORAGE_STREAMING;
    std::memset(&engine->juju_pipeline_config, 0, sizeof(engine->juju_pipeline_config));
    engine->offload_graph_ir_json.clear();
    engine->offload_graph_ir_ready.store(0, std::memory_order_relaxed);
    engine->offload_graph_ir_required.store(0, std::memory_order_relaxed);
    engine->offload_graph_ir_version.store(0, std::memory_order_relaxed);
    engine->offload_graph_ir_op_count.store(0, std::memory_order_relaxed);
    engine->offload_graph_ir_layer_count.store(0, std::memory_order_relaxed);
    engine->juju_runtime_execution_manifest_ready.store(0, std::memory_order_relaxed);
    engine->juju_runtime_access_plan_ready.store(0, std::memory_order_relaxed);
    engine->juju_runtime_access_hotset_count.store(0, std::memory_order_relaxed);
    engine->juju_runtime_access_file_group_count.store(0, std::memory_order_relaxed);
    engine->juju_runtime_executor_tensor_count.store(0, std::memory_order_relaxed);
    engine->juju_runtime_layer_prefetch_plan_count.store(0, std::memory_order_relaxed);
    engine->juju_kv_layout_contract_ready.store(0, std::memory_order_relaxed);
    engine->juju_kv_layout_max_entry_dim.store(0, std::memory_order_relaxed);
    engine->juju_kv_layout_page_size_tokens.store(0, std::memory_order_relaxed);
    engine->cached_gelu_activation_mode.store(-1, std::memory_order_relaxed);
    engine->cached_router_use_sigmoid.store(-1, std::memory_order_relaxed);
    engine->cached_router_norm_topk_prob.store(-1, std::memory_order_relaxed);
    engine->cached_router_top_k_hint.store(-1, std::memory_order_relaxed);
    engine->cached_router_group_count.store(-1, std::memory_order_relaxed);
    engine->cached_router_limited_group_count.store(-1, std::memory_order_relaxed);
    engine->cached_router_route_scale_bits.store(0, std::memory_order_relaxed);
    engine->cached_embedding_scale_hidden.store(0, std::memory_order_relaxed);
    engine->cached_embedding_scale_bits.store(0, std::memory_order_relaxed);
    engine->cached_rmsnorm_unit_offset.store(-1, std::memory_order_relaxed);
    {
        std::unique_lock<std::shared_mutex> lock(engine->rmsnorm_unit_offset_cache_mutex);
        engine->rmsnorm_unit_offset_cache.clear();
    }
    {
        std::unique_lock<std::shared_mutex> lock(engine->common_tensor_suffix_cache_mutex);
        engine->common_tensor_suffix_cache.clear();
    }
    {
        std::unique_lock<std::shared_mutex> lock(engine->text_mention_cache_mutex);
        engine->graph_ir_mention_cache.clear();
        engine->metadata_mention_cache.clear();
    }
    engine->model_config_full_attention_layers.clear();
    engine->model_config_layer_types_valid.store(0, std::memory_order_relaxed);
    engine->model_config_attention_valid.store(0, std::memory_order_relaxed);
    engine->model_config_num_attention_heads = 0;
    engine->model_config_num_key_value_heads = 0;
    engine->model_config_num_global_key_value_heads = 0;
    engine->model_config_head_dim = 0;
    engine->model_config_global_head_dim = 0;
    engine->model_config_sliding_window = 0;
    engine->model_config_query_pre_attn_scalar = 0.0f;
    engine->model_config_sliding_rope_theta = 0.0f;
    engine->model_config_full_rope_theta = 0.0f;
    engine->model_config_partial_rotary_factor = 0.0f;
    engine->juju_idx_schema_version.store(0, std::memory_order_relaxed);
    engine->juju_split_group_count.store(0, std::memory_order_relaxed);
    engine->juju_split_missing_count.store(0, std::memory_order_relaxed);
    engine->juju_runtime_hint_tensor_count.store(0, std::memory_order_relaxed);
    engine->juju_priority_tensor_count.store(0, std::memory_order_relaxed);
    engine->juju_fastmem_tensor_count.store(0, std::memory_order_relaxed);
    engine->juju_slowmem_tensor_count.store(0, std::memory_order_relaxed);
    engine->juju_predictor_metadata_ready.store(0, std::memory_order_relaxed);
    engine->juju_predictor_layer_count.store(0, std::memory_order_relaxed);
    engine->juju_predictor_expert_tensor_count.store(0, std::memory_order_relaxed);
    engine->juju_buddy_map_ready.store(0, std::memory_order_relaxed);
    engine->juju_buddy_unit_count.store(0, std::memory_order_relaxed);
    engine->juju_buddy_tensor_count.store(0, std::memory_order_relaxed);
    engine->juju_buddy_explicit_edge_count.store(0, std::memory_order_relaxed);
    engine->juju_tier_hint_ready.store(0, std::memory_order_relaxed);
    engine->juju_tier_hot_expert_tensor_count.store(0, std::memory_order_relaxed);
    engine->juju_tier_warm_expert_tensor_count.store(0, std::memory_order_relaxed);
    engine->juju_tier_cold_expert_tensor_count.store(0, std::memory_order_relaxed);
    engine->juju_partial_expert_tensor_count.store(0, std::memory_order_relaxed);
    engine->juju_moe_layer_bitmask_ready.store(0, std::memory_order_relaxed);
    engine->juju_moe_layer_count.store(0, std::memory_order_relaxed);
    engine->juju_moe_layer_bitmask_words.clear();
    engine->juju_io_section_hints_ready.store(0, std::memory_order_relaxed);
    engine->juju_runtime_contract_io_policy_ready.store(0, std::memory_order_relaxed);
    engine->juju_runtime_contract_mmap_allowed.store(1, std::memory_order_relaxed);
    engine->juju_shared_mmap_friendly.store(1, std::memory_order_relaxed);
    engine->juju_hot_mmap_friendly.store(1, std::memory_order_relaxed);
    engine->juju_warm_mmap_friendly.store(1, std::memory_order_relaxed);
    engine->juju_cold_mmap_friendly.store(1, std::memory_order_relaxed);
    engine->juju_shared_sequential_block_size.store(0, std::memory_order_relaxed);
    engine->juju_hot_sequential_block_size.store(0, std::memory_order_relaxed);
    engine->juju_warm_sequential_block_size.store(0, std::memory_order_relaxed);
    engine->juju_cold_sequential_block_size.store(0, std::memory_order_relaxed);
    engine->juju_max_prefetch_distance.store(0, std::memory_order_relaxed);
    engine->juju_shared_section_bytes.store(0, std::memory_order_relaxed);
    engine->juju_hot_section_bytes.store(0, std::memory_order_relaxed);
    engine->juju_warm_section_bytes.store(0, std::memory_order_relaxed);
    engine->juju_cold_section_bytes.store(0, std::memory_order_relaxed);
    engine->expert_segment_policies.clear();
    engine->expert_partial_exec_flags.clear();
    engine->expert_partial_accuracy_milli.clear();
    engine->juju_buddy_expert_ids.clear();
    engine->offload_gguf_first_file.clear();
    engine->offload_gguf_metadata_json.clear();
    engine->offload_weight_quant_family.clear();
    engine->offload_weight_kernel_family.clear();
    engine->dynamic_shape = moe_model_shape_t{};
    engine->dynamic_architecture.clear();
    engine->dynamic_model_id.clear();
    engine->dynamic_shape_valid.store(0, std::memory_order_relaxed);
    engine->attn_config = moe_attn_config_t{};
    engine->attn_config_valid.store(0, std::memory_order_relaxed);
    moe_engine_reset_qkv_cache(engine);
    engine->forward_adapter.store(moe_FORWARD_ADAPTER_NONE, std::memory_order_relaxed);
    engine->forward_adapter_executable.store(0, std::memory_order_relaxed);
    engine->cpu_only_runtime.store(0, std::memory_order_relaxed);
    engine->storage_runtime_path.store(0, std::memory_order_relaxed);
    engine->cpu_storage_engine_ready.store(0, std::memory_order_relaxed);
    engine->cpu_storage_execution_plan_ready.store(0, std::memory_order_relaxed);
    engine->cpu_quant_registry_ready.store(0, std::memory_order_relaxed);
    engine->graph_ir_used_for_forward.store(0, std::memory_order_relaxed);
    engine->gpu_path_disabled_for_cpu_only.store(0, std::memory_order_relaxed);
    engine->cpu_execution_plan_layers.store(0, std::memory_order_relaxed);
    engine->cpu_execution_plan_tensors.store(0, std::memory_order_relaxed);
    engine->cpu_execution_plan_expert_tensors.store(0, std::memory_order_relaxed);
    engine->cpu_execution_plan_bytes.store(0, std::memory_order_relaxed);
    engine->cpu_prefetch_scheduled.store(0, std::memory_order_relaxed);
    engine->cpu_selected_expert_prefetch_scheduled.store(0, std::memory_order_relaxed);
    engine->cpu_layer_begin_count.store(0, std::memory_order_relaxed);
}
