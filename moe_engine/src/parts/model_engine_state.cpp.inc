// Engine state management
// QKV cache reset and contract clearing

static void stop_io_workers(moe_pc_engine* engine);
static void stop_cpu_row_workers(moe_pc_engine* engine);

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
    engine->juju_idx_schema_version.store(0, std::memory_order_relaxed);
    engine->juju_split_group_count.store(0, std::memory_order_relaxed);
    engine->juju_split_missing_count.store(0, std::memory_order_relaxed);
    engine->juju_runtime_hint_tensor_count.store(0, std::memory_order_relaxed);
    engine->juju_priority_tensor_count.store(0, std::memory_order_relaxed);
    engine->juju_fastmem_tensor_count.store(0, std::memory_order_relaxed);
    engine->juju_slowmem_tensor_count.store(0, std::memory_order_relaxed);
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
}
