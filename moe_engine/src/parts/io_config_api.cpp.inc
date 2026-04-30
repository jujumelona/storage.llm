int moe_pc_engine_configure_io(moe_pc_engine_t* engine, const moe_io_config_t* config) {
    if (!engine) {
        return 0;
    }
    stop_io_workers(engine);
    reset_staging_slots(engine);
    free_staging(engine->staging, engine->staging_size, engine->staging_pinned);
    engine->staging = nullptr;
    engine->staging_size = 0;
    engine->staging_pinned = false;

    engine->io_config = config ? *config : moe_pc_default_io_config();
    if (engine->io_config.io_worker_threads > disk_worker_cap()) {
        engine->io_config.io_worker_threads = disk_worker_cap();
    }
    if (engine->io_config.max_prefetch_queue == 0) {
        engine->io_config.max_prefetch_queue = 4096;
    } else if (engine->io_config.max_prefetch_queue < 4096) {
        engine->io_config.max_prefetch_queue = 4096;
    }
    if (engine->io_config.topology_max_candidates == 0) {
        engine->io_config.topology_max_candidates = 1;
    }
    if (engine->io_config.topology_max_edges_per_source == 0) {
        engine->io_config.topology_max_edges_per_source = 1;
    }
    if (engine->io_config.disk_worker_threads == 0) {
        engine->io_config.disk_worker_threads = engine->io_config.io_worker_threads ?
            engine->io_config.io_worker_threads : 1;
    }
    engine->io_config.disk_worker_threads =
        clamp_to_cap(engine->io_config.disk_worker_threads, disk_worker_cap());
    if (engine->io_config.pinned_worker_threads == 0) {
        engine->io_config.pinned_worker_threads = 1;
    }
    engine->io_config.pinned_worker_threads =
        clamp_to_cap(engine->io_config.pinned_worker_threads, pinned_worker_cap(engine));
    if (engine->io_config.gpu_worker_threads == 0) {
        engine->io_config.gpu_worker_threads = engine->backend_caps.copy_streams > 1 ? 2u : 1u;
    }
    engine->io_config.gpu_worker_threads =
        clamp_to_cap(engine->io_config.gpu_worker_threads, gpu_worker_cap(engine));
    engine->runtime_prefetch_window_layers.store(
        engine->io_config.prefetch_window_layers, std::memory_order_relaxed);
    runtime_store_io_workers(
        engine,
        engine->io_config.disk_worker_threads,
        engine->io_config.pinned_worker_threads,
        engine->io_config.gpu_worker_threads);
    if (engine->io_config.direct_upload_min_bytes == 0) {
        const moe_storage_constants_t* storage = moe_storage_constants();
        engine->io_config.direct_upload_min_bytes =
            storage ? storage->normal_bundle_min_bytes : 1;
    }
    if (engine_bypass_staging_pipeline(engine)) {
        engine->io_config.enable_directstorage = 0;
        engine->io_config.enable_gpudirect_storage = 0;
        engine->io_config.enable_bam_nvme = 0;
        engine->io_config.enable_pinned_host = 0;
        engine->io_config.enable_multistage_pipeline = 0;
        engine->io_config.pinned_staging_bytes = 0;
        engine->io_config.pinned_worker_threads = 0;
        engine->io_config.gpu_worker_threads = 0;
    }
    if (engine->io_config.enable_pinned_host && engine->io_config.pinned_staging_bytes > 0) {
        const uint64_t requested_staging_bytes = engine->io_config.pinned_staging_bytes;
        engine->staging = allocate_best_staging(
            requested_staging_bytes,
            &engine->staging_size,
            &engine->staging_pinned
        );
        if (!engine->staging) {
            engine->io_config.enable_pinned_host = 0;
            engine->io_config.pinned_staging_bytes = 0;
            engine->io_config.enable_multistage_pipeline = 0;
            fprintf(stderr,
                "[storagellm] pinned staging unavailable; continuing without pinned staging "
                "(requested=%llu bytes)\n",
                (unsigned long long)requested_staging_bytes);
        }
        // BUG 1 FIX: Non-pinned staging mistaken for pinned ?…â˜…??        // allocate_best_staging returns non-null + non-pinned when mlock fails.
        // Without this check, enable_pinned_host=1 and enable_multistage_pipeline=1 remain set,
        // causing pinned_worker threads to launch and use non-pinned memory for "async" DMA.
        // CUDA driver internally copies non-pinned memory to temp pinned buffer ??sync blocking.
        // Result: entire pipeline blocks on every transfer with no stderr warning.
        if (engine->staging && !engine->staging_pinned) {
            // Memory allocated but not pinned ??async DMA impossible, pipeline has no effect
            engine->io_config.enable_pinned_host = 0;
            engine->io_config.enable_multistage_pipeline = 0;
            fprintf(stderr,
                "[storagellm] staging allocated but not pinned; "
                "disabling multistage pipeline (sync DMA fallback).\n");
        }
        if (engine->staging) {
            std::memset(engine->staging, 0, static_cast<size_t>(std::min<uint64_t>(engine->staging_size, 4096)));
            configure_staging_slots(engine);
        }
    }

    reset_io_stats_for_config(engine);
    start_io_workers(engine);
    return 1;
}

int moe_pc_engine_get_io_stats(const moe_pc_engine_t* engine, moe_io_stats_t* out_stats) {
    if (!engine || !out_stats) {
        return 0;
    }
    snapshot_io_stats(engine, out_stats);
    return 1;
}

int moe_pc_engine_wait_io(moe_pc_engine_t* engine) {
    if (!engine) {
        return 0;
    }
    std::unique_lock<std::mutex> lock(engine->done_mutex);
    engine->io_done_cv.wait(lock, [&] {
        return stage_idle(engine->disk_stage) &&
               stage_idle(engine->pinned_stage) &&
               stage_idle(engine->gpu_stage);
    });
    return 1;
}
