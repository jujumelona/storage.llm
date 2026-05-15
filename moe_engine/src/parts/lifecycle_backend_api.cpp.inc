static void moe_storage_async_prefetch_shutdown(moe_pc_engine* engine);
moe_pc_engine_t* moe_pc_engine_create(const moe_pc_engine_config_t* config) {
    if (!config) return nullptr;
    moe_platform_init_limits();
    auto* engine = new (std::nothrow) moe_pc_engine();
    if (!engine) return nullptr;
    engine->engine_generation = g_moe_engine_generation.fetch_add(1u, std::memory_order_relaxed) + 1u;
    engine->config = config ? *config : moe_pc_default_config();
    if (engine->config.kv_offload_min_context_tokens == 0) {
        engine->config.kv_offload_min_context_tokens = 32768;
    }
    if (engine->io_config.prefill_group_wait_us == 0) {
        engine->io_config.prefill_group_wait_us = 200;
    }
    if (engine->io_config.internal_require_gpu_expert_batch_kernel == 0) {
        engine->io_config.internal_require_gpu_expert_batch_kernel = 1;
    }
    if (engine->io_config.internal_verified_backend_only == 0) {
        engine->io_config.internal_verified_backend_only = 1;
    }
    if (engine->config.kv_offload_mode != moe_KV_OFFLOAD_DISABLED &&
        engine->config.kv_offload_mode != moe_KV_OFFLOAD_COMPAT) {
        engine->config.kv_offload_mode = moe_KV_OFFLOAD_DISABLED;
    }
    qkv_set_mode(engine->config.kv_mode == moe_KV_MODE_QKV);
    engine->expert_states.assign(expert_slot_count(engine), moe_block_state{});
    tensor_slots_reset(engine);
    engine->tensors.reserve(tensor_record_capacity(engine));
    engine->shard_mmaps.reserve(105);
    engine->topology_edges.reserve(19456);
    engine->prefetch_plan_seen_gen.reserve(24576);
    fill_backend_caps(engine->config.preferred_backend, engine->config.platform, &engine->backend_caps, &engine->config);
    model_lib_prepare(engine);
    if (engine->config.prefer_gpu && engine->config.vram_budget_bytes == 0 &&
        caps_gpu_backend_available(engine->backend_caps)) {
        engine->config.vram_budget_bytes = engine->backend_caps.recommended_vram_cache_bytes;
    }
    moe_storage_runtime_path_refresh(engine);
    device_allocator_init(engine);
    apply_common_vram_reservation(engine);
    moe_pc_engine_configure_io(engine, nullptr);
    engine->last_dropped_requests_snapshot.store(engine->io_atomic.dropped_requests.load(std::memory_order_relaxed), std::memory_order_relaxed);
    ensure_cpu_row_workers(engine, 0);
    device_stream_pool_init(engine);
    return engine;
}

void moe_pc_engine_destroy(moe_pc_engine_t* engine) {
    if (!engine) {
        return;
    }
    if (!engine->topology_path.empty()) {
        moe_pc_engine_save_topology(engine, nullptr);
    }
    // BUGFIX 971: Persist EMA residency stats on shutdown (atomic write-back)
    flush_residency_ema_to_idx(engine);
    moe_storage_async_prefetch_shutdown(engine);
    stop_cpu_row_workers(engine);
    stop_io_workers(engine);
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
    directstorage_queue_cleanup(engine);
    device_stream_pool_destroy(engine);
    device_cufile_cleanup_caches(engine);
    device_allocator_shutdown(engine);
    model_lib_unload(engine);
    {
        std::unique_lock<std::shared_mutex> lock(engine->shard_mutex);
        for (auto& item : engine->shard_mmaps) {
            unload_mapped_shard(item.second);
        }
        engine->shard_mmaps.clear();
        engine->shard_mmaps_by_id.clear();
        engine->shard_mmap_cvs.clear();
        engine->locked_mmaps.clear();
    }
    reset_source_mmap_slots(engine);
    {
        std::lock_guard<std::mutex> lock(engine->raw_file_size_cache_mutex);
        engine->raw_file_size_cache.clear();
    }
    {
        std::unique_lock<std::shared_mutex> lock(engine->raw_tensor_cache_mutex);
        engine->raw_tensor_cache.clear();
    }
    {
        std::unique_lock<std::shared_mutex> lock(engine->router_layer_contract_slice_cache_mutex);
        engine->router_layer_contract_slice_cache.clear();
    }
    engine->cached_rmsnorm_unit_offset.store(-1, std::memory_order_relaxed);
    {
        std::unique_lock<std::shared_mutex> lock(engine->rmsnorm_unit_offset_cache_mutex);
        engine->rmsnorm_unit_offset_cache.clear();
    }
    if (engine->scale4) {
        moe_scale4_close(engine->scale4);
        engine->scale4 = nullptr;
    }
    // Bug 13 Fix: Call reset_staging_slots BEFORE free_staging to prevent use-after-free.
    // Workers may still reference staging_slots[i].buf in 5ms race window.
    // reset_staging_slots clears in_use flags and waits for workers to exit.
    // BUGFIX 988: Drain orphaned_events on shutdown ★★ IMPORTANT
    // Problem: orphaned_events only drained in release_staging_slot.
    // If workers terminated via exception (no slot return), orphaned_events
    // accumulate permanently → CUDA event handle leak until process exit.
    // Solution: Drain all orphaned events before reset to prevent driver handle leak.
    {
        std::lock_guard<std::mutex> orphan_lock(engine->orphaned_events_mutex);
        for (CUevent ev : engine->orphaned_events) {
            device_backend_event_destroy(engine, ev);
        }
        engine->orphaned_events.clear();
    }
    reset_staging_slots(engine);
    free_staging_for_engine(engine, engine->staging, engine->staging_size, engine->staging_pinned,
                            engine->staging_numa_allocated, engine->staging_device_host_allocated);
    engine->staging = nullptr;
    engine->staging_size = 0;
    engine->staging_pinned = false;
    engine->staging_numa_allocated = false;
    engine->staging_device_host_allocated = false;
    delete engine;
}

int moe_pc_detect_backend_caps(
    moe_backend_t preferred_backend,
    moe_platform_t platform,
    moe_backend_caps_t* out_caps
) {
    if (!out_caps) {
        return 0;
    }
    fill_backend_caps(preferred_backend, platform, out_caps, nullptr);
    return 1;
}

int moe_pc_engine_get_backend_caps(const moe_pc_engine_t* engine, moe_backend_caps_t* out_caps) {
    if (!engine || !out_caps) {
        return 0;
    }
    *out_caps = engine->backend_caps;
    const uint64_t recommended_staging = moe_recommended_staging_bytes(engine);
    if (recommended_staging > out_caps->recommended_pinned_bytes) {
        out_caps->recommended_pinned_bytes = recommended_staging;
    }
    return 1;
}
