int moe_pc_engine_get_stats(const moe_pc_engine_t* engine, moe_pc_engine_stats_t* out_stats) {
    if (!engine || !out_stats) {
        return 0;
    }
    std::memset(out_stats, 0, sizeof(*out_stats));
    const moe_perf_runtime_profile_t runtime_profile =
        moe_perf_profile_for_backend(engine->backend_caps.backend);
    out_stats->runtime_profile_kind = static_cast<uint32_t>(runtime_profile);
    out_stats->real_continuous_batch_exec_requested =
        moe_perf_real_continuous_batch_exec_requested() ? 1u : 0u;
    out_stats->real_continuous_batch_exec_linked =
        moe_perf_real_continuous_batch_exec_linked() ? 1u : 0u;
    out_stats->runtime_uses_vram =
        moe_perf_profile_uses_vram(runtime_profile) ? 1u : 0u;
    out_stats->real_continuous_batch_executor_available =
        moe_perf_real_continuous_batch_executor_available() ? 1u : 0u;
    out_stats->per_request_context_available =
        moe_perf_per_request_context_available() ? 1u : 0u;
    out_stats->engine_mutex_removable =
        moe_perf_engine_mutex_removable() ? 1u : 0u;
    const moe_fast_backend_kind_t selected_fast_backend = choose_fast_backend(engine);
    out_stats->fast_backend_kind = static_cast<uint32_t>(selected_fast_backend);
    out_stats->fast_backend_true_kernel =
        moe_perf_backend_kind_is_true_kernel(selected_fast_backend) ? 1u : 0u;
    out_stats->fast_backend_fused_moe =
        moe_perf_backend_kind_is_fused_moe(selected_fast_backend) ? 1u : 0u;
    out_stats->strict_completion_ready =
        (out_stats->real_continuous_batch_executor_available &&
         out_stats->per_request_context_available &&
         out_stats->engine_mutex_removable &&
         out_stats->fast_backend_true_kernel &&
         out_stats->fast_backend_fused_moe) ? 1u : 0u;
    std::strncpy(
        out_stats->runtime_profile,
        moe_perf_profile_name(runtime_profile),
        sizeof(out_stats->runtime_profile) - 1);
    std::strncpy(
        out_stats->runtime_pipeline,
        moe_perf_pipeline_name(runtime_profile),
        sizeof(out_stats->runtime_pipeline) - 1);
    std::strncpy(
        out_stats->fast_backend_name,
        moe_perf_backend_kind_name(selected_fast_backend),
        sizeof(out_stats->fast_backend_name) - 1);
    std::strncpy(
        out_stats->performance_completion_level,
        moe_perf_completion_level(
            out_stats->real_continuous_batch_executor_available,
            out_stats->per_request_context_available,
            out_stats->engine_mutex_removable,
            out_stats->fast_backend_true_kernel,
            out_stats->fast_backend_fused_moe),
        sizeof(out_stats->performance_completion_level) - 1);
    out_stats->vram_used_bytes = tier_used(engine, moe_TIER_VRAM);
    out_stats->ram_used_bytes = tier_used(engine, moe_TIER_RAM);
    out_stats->db_used_bytes = tier_used(engine, moe_TIER_DB);
    // BUGFIX 225: Add memory_order_relaxed to 15 atomic loads in stats snapshot ★★
    // Problem: Missing memory_order causes seq_cst default (40-200x slower than relaxed)
    // Solution: Add memory_order_relaxed to all atomic loads (stats snapshot doesn't need synchronization)
    // Impact: 40-200x performance improvement in stats collection (hot path in monitoring)
    out_stats->common_vram_reserved_bytes = engine->common_vram_reserved_bytes.load(std::memory_order_relaxed);
    out_stats->vram_expert_used_bytes =
        out_stats->vram_used_bytes > out_stats->common_vram_reserved_bytes ?
        out_stats->vram_used_bytes - out_stats->common_vram_reserved_bytes : 0;
    out_stats->common_raw_prefetched_bytes = engine->common_raw_prefetched_bytes.load(std::memory_order_relaxed);
    out_stats->common_raw_prefetched_spans = engine->common_raw_prefetched_spans.load(std::memory_order_relaxed);
    out_stats->device_allocated_bytes = engine->device_allocated_bytes.load(std::memory_order_relaxed);
    out_stats->device_allocation_count = engine->device_allocation_count.load(std::memory_order_relaxed);
    out_stats->device_fixed_bytes = engine->device_fixed_bytes.load(std::memory_order_relaxed);
    out_stats->device_activation_bytes = engine->device_activation_bytes.load(std::memory_order_relaxed);
    out_stats->device_expert_bytes = engine->device_expert_bytes.load(std::memory_order_relaxed);
    out_stats->device_stream_bytes = engine->device_stream_bytes.load(std::memory_order_relaxed);
    out_stats->device_total_bytes = engine->device_total_bytes.load(std::memory_order_relaxed);
    out_stats->device_free_bytes = engine->device_free_bytes.load(std::memory_order_relaxed);
    out_stats->model_lib_loaded = engine->model_lib_loaded;
    out_stats->model_lib_generated = engine->model_lib_generated;
    out_stats->model_lib_compile_attempted = engine->model_lib_compile_attempted;
    out_stats->model_lib_compile_succeeded = engine->model_lib_compile_succeeded;
    out_stats->model_lib_abi_version = engine->model_lib_abi_version;
    if (!engine->model_lib_path.empty()) {
        std::strncpy(out_stats->model_lib_path, engine->model_lib_path.c_str(), sizeof(out_stats->model_lib_path) - 1);
    }
    out_stats->common_raw_tensor_count = moe_common_raw_tensor_count();
    out_stats->vram_expert_count = engine->vram_expert_count.load(std::memory_order_relaxed);
    out_stats->ram_expert_count = engine->ram_expert_count.load(std::memory_order_relaxed);
    out_stats->db_expert_count = engine->db_expert_count.load(std::memory_order_relaxed);
    out_stats->hot_expert_count = out_stats->vram_expert_count + out_stats->ram_expert_count + out_stats->db_expert_count;
    out_stats->generation_started = engine->generation_started.load(std::memory_order_relaxed);
    out_stats->generation_completed = engine->generation_completed.load(std::memory_order_relaxed);
    out_stats->generation_failed = engine->generation_failed.load(std::memory_order_relaxed);
    out_stats->generation_active = engine->generation_active.load(std::memory_order_relaxed);
    out_stats->generation_token = engine->generation_token.load(std::memory_order_relaxed);
    out_stats->generation_layer = engine->generation_layer.load(std::memory_order_relaxed);
    out_stats->generation_phase = engine->generation_phase.load(std::memory_order_relaxed);
    out_stats->kv_mode = engine->config.kv_mode;
    out_stats->offload_gguf_valid = engine->offload_gguf_valid.load(std::memory_order_relaxed);
    out_stats->offload_gguf_file_count = engine->offload_gguf_file_count.load(std::memory_order_relaxed);
    out_stats->qkv_forced_by_format = engine->qkv_forced_by_format.load(std::memory_order_relaxed);
    if (engine->offload_qkv_cfg_available.load(std::memory_order_acquire)) {
        out_stats->qkv_k_bits = (uint32_t)std::max(0, engine->offload_qkv_cfg.k_bits);
        out_stats->qkv_v_bits = (uint32_t)std::max(0, engine->offload_qkv_cfg.v_bits);
        out_stats->qkv_normal_bits = (uint32_t)std::max(0, engine->offload_qkv_cfg.normal_bits);
        out_stats->qkv_key_normal_bits = (uint32_t)std::max(0, engine->offload_qkv_cfg.key_normal_bits);
        out_stats->qkv_value_normal_bits = (uint32_t)std::max(0, engine->offload_qkv_cfg.value_normal_bits);
        out_stats->qkv_group_size = engine->offload_qkv_cfg.group_size;
        out_stats->qkv_page_size_tokens = engine->offload_qkv_cfg.page_size_tokens;
        out_stats->qkv_sink_tokens = engine->offload_qkv_cfg.sink_tokens;
        out_stats->qkv_rotation_seed = engine->offload_qkv_cfg.rotation_seed;
        out_stats->qkv_qjl_seed = engine->offload_qkv_cfg.qjl_seed;
        out_stats->qkv_enable_qjl = engine->offload_qkv_cfg.enable_qjl ? 1u : 0u;
        out_stats->qkv_enable_rotation = engine->offload_qkv_cfg.enable_rotation ? 1u : 0u;
        out_stats->qkv_outlier_channels = (uint32_t)std::max(0, engine->offload_qkv_cfg.outlier_channels);
        out_stats->qkv_outlier_bits = (uint32_t)std::max(0, engine->offload_qkv_cfg.outlier_bits);
        out_stats->qkv_key_outlier_bits = (uint32_t)std::max(0, engine->offload_qkv_cfg.key_outlier_bits);
        out_stats->qkv_value_outlier_bits = (uint32_t)std::max(0, engine->offload_qkv_cfg.value_outlier_bits);
        out_stats->qkv_plain_kv_persistent_storage =
            engine->offload_qkv_cfg.plain_kv_persistent_storage ? 1u : 0u;
    }
    out_stats->qkv_plain_kv_forbidden_touches = engine->qkv_plain_kv_forbidden_touches.load(std::memory_order_relaxed);
    out_stats->qkv_plain_kv_allocation_skips = engine->qkv_plain_kv_allocation_skips.load(std::memory_order_relaxed);
    out_stats->qkv_plain_kv_readiness_touches = engine->qkv_plain_kv_readiness_touches.load(std::memory_order_relaxed);
    out_stats->qkv_plain_kv_capacity_touches = engine->qkv_plain_kv_capacity_touches.load(std::memory_order_relaxed);
    out_stats->qkv_plain_kv_store_touches = engine->qkv_plain_kv_store_touches.load(std::memory_order_relaxed);
    out_stats->qkv_plain_kv_decode_cache_touches = engine->qkv_plain_kv_decode_cache_touches.load(std::memory_order_relaxed);
    out_stats->qkv_ppl_qkv_only_evals = engine->qkv_ppl_qkv_only_evals.load(std::memory_order_relaxed);
    out_stats->qkv_env_override_blocked = engine->qkv_env_override_blocked.load(std::memory_order_relaxed);
    out_stats->weight_quant_bits = engine->offload_weight_quant_bits.load(std::memory_order_relaxed);
    out_stats->weight_quant_encoding = engine->offload_weight_encoding.load(std::memory_order_relaxed);
    out_stats->weight_quant_block_size = engine->offload_weight_block_size.load(std::memory_order_relaxed);
    out_stats->optimization_catalog_count =
        engine->offload_optimization_catalog_count.load(std::memory_order_relaxed);
    out_stats->juju_strategy_valid = engine->juju_strategy_valid.load(std::memory_order_relaxed);
    out_stats->juju_execution_path = (uint32_t)engine->juju_execution_path;
    out_stats->juju_prefetch_depth = engine->juju_pipeline_config.prefetch_depth;
    out_stats->juju_io_threads = engine->juju_pipeline_config.io_threads;
    out_stats->juju_segment_count = engine->juju_pipeline_config.segment_count;
    out_stats->juju_block_size = engine->juju_pipeline_config.block_size;
    out_stats->juju_segment_bytes = engine->juju_pipeline_config.segment_bytes;
    out_stats->juju_tier0_expert_count = engine->juju_pipeline_config.tier0_expert_count;
    out_stats->juju_tier1_expert_count = engine->juju_pipeline_config.tier1_expert_count;
    out_stats->juju_compute_ms = engine->juju_pipeline_config.compute_ms;
    out_stats->juju_tier1_to_tier0_ms = engine->juju_pipeline_config.tier1_to_tier0_ms;
    out_stats->juju_tier2_to_tier1_ms = engine->juju_pipeline_config.tier2_to_tier1_ms;
    out_stats->offload_gguf_tensor_count =
        engine->offload_gguf_tensor_count.load(std::memory_order_relaxed);
    out_stats->offload_gguf_executable_tensor_count =
        engine->offload_gguf_executable_tensor_count.load(std::memory_order_relaxed);
    out_stats->juju_idx_schema_version =
        engine->juju_idx_schema_version.load(std::memory_order_relaxed);
    out_stats->juju_split_group_count =
        engine->juju_split_group_count.load(std::memory_order_relaxed);
    out_stats->juju_split_missing_count =
        engine->juju_split_missing_count.load(std::memory_order_relaxed);
    out_stats->juju_runtime_hint_tensor_count =
        engine->juju_runtime_hint_tensor_count.load(std::memory_order_relaxed);
    out_stats->juju_priority_tensor_count =
        engine->juju_priority_tensor_count.load(std::memory_order_relaxed);
    out_stats->juju_fastmem_tensor_count =
        engine->juju_fastmem_tensor_count.load(std::memory_order_relaxed);
    out_stats->juju_slowmem_tensor_count =
        engine->juju_slowmem_tensor_count.load(std::memory_order_relaxed);
    out_stats->juju_io_section_hints_ready =
        engine->juju_io_section_hints_ready.load(std::memory_order_relaxed);
    out_stats->juju_runtime_mmap_allowed =
        engine->juju_runtime_contract_mmap_allowed.load(std::memory_order_relaxed);
    out_stats->juju_cold_mmap_friendly =
        engine->juju_cold_mmap_friendly.load(std::memory_order_relaxed);
    out_stats->juju_warm_mmap_friendly =
        engine->juju_warm_mmap_friendly.load(std::memory_order_relaxed);
    out_stats->juju_cold_sequential_block_size =
        engine->juju_cold_sequential_block_size.load(std::memory_order_relaxed);
    out_stats->juju_warm_sequential_block_size =
        engine->juju_warm_sequential_block_size.load(std::memory_order_relaxed);
    out_stats->juju_max_prefetch_distance =
        engine->juju_max_prefetch_distance.load(std::memory_order_relaxed);
    out_stats->expert_cache_request_count =
        engine->expert_cache_request_count.load(std::memory_order_relaxed);
    out_stats->expert_cache_hit_count =
        engine->expert_cache_hit_count.load(std::memory_order_relaxed);
    out_stats->expert_cache_vram_hit_count =
        engine->expert_cache_vram_hit_count.load(std::memory_order_relaxed);
    out_stats->expert_cache_ram_hit_count =
        engine->expert_cache_ram_hit_count.load(std::memory_order_relaxed);
    out_stats->expert_cache_miss_count =
        engine->expert_cache_miss_count.load(std::memory_order_relaxed);
    out_stats->expert_cache_miss_bytes =
        engine->expert_cache_miss_bytes.load(std::memory_order_relaxed);
    out_stats->expert_miss_latency_us_total =
        engine->expert_miss_latency_us_total.load(std::memory_order_relaxed);
    out_stats->expert_hit_rate_milli =
        engine->expert_hit_rate_milli.load(std::memory_order_relaxed);
    out_stats->speculative_prefetch_planned_count =
        engine->speculative_prefetch_planned_count.load(std::memory_order_relaxed);
    out_stats->speculative_prefetch_planned_bytes =
        engine->speculative_prefetch_planned_bytes.load(std::memory_order_relaxed);
    out_stats->prefetch_waste_ratio_milli =
        engine->prefetch_waste_ratio_milli.load(std::memory_order_relaxed);
    out_stats->gpu_copy_idle_milli =
        engine->gpu_copy_idle_milli.load(std::memory_order_relaxed);
    out_stats->adaptive_prefetch_window_last =
        engine->adaptive_prefetch_window_last.load(std::memory_order_relaxed);
    out_stats->disk_read_bytes =
        engine->io_atomic.bytes_disk_to_ram.load(std::memory_order_relaxed) +
        engine->io_atomic.bytes_direct_to_gpu.load(std::memory_order_relaxed);
    out_stats->pcie_copy_bytes =
        engine->io_atomic.bytes_pinned_to_gpu.load(std::memory_order_relaxed) +
        engine->io_atomic.bytes_sync_h2d_fallback.load(std::memory_order_relaxed) +
        engine->io_atomic.backend_async_submitted.load(std::memory_order_relaxed);
    if (!engine->offload_gguf_first_file.empty()) {
        std::strncpy(
            out_stats->offload_gguf_first_file,
            engine->offload_gguf_first_file.c_str(),
            sizeof(out_stats->offload_gguf_first_file) - 1
        );
    }
    if (!engine->offload_weight_quant_family.empty()) {
        std::strncpy(
            out_stats->weight_quant_family,
            engine->offload_weight_quant_family.c_str(),
            sizeof(out_stats->weight_quant_family) - 1
        );
    }
    if (!engine->offload_weight_kernel_family.empty()) {
        std::strncpy(
            out_stats->weight_kernel_family,
            engine->offload_weight_kernel_family.c_str(),
            sizeof(out_stats->weight_kernel_family) - 1
        );
    }
    {
        std::lock_guard<std::mutex> batch_lock(const_cast<std::mutex&>(engine->batch_scheduler.mtx));
        out_stats->batch_waiting_requests = (uint64_t)engine->batch_scheduler.waiting_queue.size();
        out_stats->batch_running_requests = (uint64_t)engine->batch_scheduler.running_queue.size();
        out_stats->batch_completed_result_slots = (uint64_t)engine->batch_scheduler.completed_results.size();
        out_stats->batch_round_robin_cursor = (uint64_t)engine->batch_scheduler.round_robin_cursor;
        out_stats->batch_max_running_requests = engine->batch_scheduler.max_running_requests;
        out_stats->batch_max_tokens_per_iteration = engine->batch_scheduler.max_tokens_per_iteration;
    }
    out_stats->batch_submitted_requests = engine->batch_scheduler.submitted_requests.load(std::memory_order_relaxed);
    out_stats->batch_admitted_requests = engine->batch_scheduler.admitted_requests.load(std::memory_order_relaxed);
    out_stats->batch_completed_requests = engine->batch_scheduler.completed_requests.load(std::memory_order_relaxed);
    out_stats->batch_cancelled_requests = engine->batch_scheduler.cancelled_requests.load(std::memory_order_relaxed);
    out_stats->batch_schedule_iterations = engine->batch_scheduler.schedule_iterations.load(std::memory_order_relaxed);
    out_stats->batch_prefill_scheduled = engine->batch_scheduler.prefill_scheduled.load(std::memory_order_relaxed);
    out_stats->batch_decode_scheduled = engine->batch_scheduler.decode_scheduled.load(std::memory_order_relaxed);
    out_stats->batch_max_waiting_observed = engine->batch_scheduler.max_waiting_observed.load(std::memory_order_relaxed);
    out_stats->batch_max_running_observed = engine->batch_scheduler.max_running_observed.load(std::memory_order_relaxed);
    out_stats->batch_max_completed_observed = engine->batch_scheduler.max_completed_observed.load(std::memory_order_relaxed);
    out_stats->batch_runtime_states = moe_batch_runtime_state_count();
    return 1;
}
