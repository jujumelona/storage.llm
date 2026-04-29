int glm5_pc_engine_configure_io(glm5_pc_engine_t* engine, const glm5_io_config_t* config) {
    if (!engine) {
        return 0;
    }
    stop_io_workers(engine);
    reset_staging_slots(engine);
    free_staging(engine->staging, engine->staging_size, engine->staging_pinned);
    engine->staging = nullptr;
    engine->staging_size = 0;
    engine->staging_pinned = false;

    engine->io_config = config ? *config : glm5_pc_default_io_config();
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
        const glm5_storage_constants_t* storage = glm5_storage_constants();
        engine->io_config.direct_upload_min_bytes =
            storage ? storage->normal_bundle_min_bytes : 1;
    }
    if (engine->backend_caps.backend == GLM5_BACKEND_METAL) {
        engine->io_config.enable_directstorage = 0;
        engine->io_config.enable_gpudirect_storage = 0;
        engine->io_config.enable_bam_nvme = 0;
        engine->io_config.enable_pinned_host = 0;
        engine->io_config.pinned_staging_bytes = 0;
    }
    if (engine->io_config.enable_pinned_host && engine->io_config.pinned_staging_bytes > 0) {
        engine->staging = allocate_best_staging(
            engine->io_config.pinned_staging_bytes,
            &engine->staging_size,
            &engine->staging_pinned
        );
        if (engine->staging) {
            std::memset(engine->staging, 0, static_cast<size_t>(std::min<uint64_t>(engine->staging_size, 4096)));
            configure_staging_slots(engine);
        }
    }

    reset_io_stats_for_config(engine);
    start_io_workers(engine);
    return 1;
}

int glm5_pc_engine_get_io_stats(const glm5_pc_engine_t* engine, glm5_io_stats_t* out_stats) {
    if (!engine || !out_stats) {
        return 0;
    }
    snapshot_io_stats(engine, out_stats);
    return 1;
}

int glm5_pc_engine_wait_io(glm5_pc_engine_t* engine) {
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

