static void stop_io_workers(glm5_pc_engine* engine) {
    if (!engine) {
        return;
    }
    engine->io_stop.store(true);
    engine->disk_worker_target.store(0);
    engine->pinned_worker_target.store(0);
    engine->gpu_worker_target.store(0);
    engine->disk_stage.cv.notify_all();
    engine->pinned_stage.cv.notify_all();
    engine->gpu_stage.cv.notify_all();
    engine->staging_slot_cv.notify_all();
    for (std::thread& worker : engine->io_workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    for (std::thread& worker : engine->pinned_workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    for (std::thread& worker : engine->gpu_workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    device_force_clear_pending_copies(engine);
    // Fix 71: force_blocking=true on shutdown to ensure all threads joined
    flush_deferred_joins(engine, true);
    engine->io_workers.clear();
    engine->pinned_workers.clear();
    engine->gpu_workers.clear();
    // Fix: Clear stages BEFORE setting io_stop = false to prevent race condition
    // where new tasks are enqueued and immediately cleared
    clear_stage(engine, engine->disk_stage);
    clear_stage(engine, engine->pinned_stage);
    clear_stage(engine, engine->gpu_stage);
    engine->total_queue_depth.store(0, std::memory_order_relaxed);
    engine->io_stop.store(false);
    engine->io_done_cv.notify_all();
}

static void start_io_workers(glm5_pc_engine* engine) {
    if (!engine || !engine->io_config.enable_compute_io_overlap) {
        return;
    }
    uint32_t workers = engine->io_config.disk_worker_threads ?
        engine->io_config.disk_worker_threads : engine->io_config.io_worker_threads;
    if (workers == 0) {
        workers = 1;
    }
    workers = clamp_to_cap(workers, disk_worker_cap());
    engine->disk_worker_target.store(workers);
    for (uint32_t i = 0; i < workers; ++i) {
        engine->io_workers.emplace_back(io_worker_loop, engine, i);
    }
    if (engine->io_config.enable_multistage_pipeline) {
        uint32_t pinned_workers = engine->io_config.pinned_worker_threads;
        if (pinned_workers == 0) {
            pinned_workers = 1;
        }
        pinned_workers = clamp_to_cap(pinned_workers, pinned_worker_cap(engine));
        engine->pinned_worker_target.store(pinned_workers);
        for (uint32_t i = 0; i < pinned_workers; ++i) {
            engine->pinned_workers.emplace_back(pinned_worker_loop, engine, i);
        }

        uint32_t gpu_workers = engine->io_config.gpu_worker_threads;
        if (gpu_workers == 0) {
            gpu_workers = 1;
        }
        gpu_workers = clamp_to_cap(gpu_workers, gpu_worker_cap(engine));
        engine->gpu_worker_target.store(gpu_workers);
        for (uint32_t i = 0; i < gpu_workers; ++i) {
            engine->gpu_workers.emplace_back(gpu_worker_loop, engine, i);
        }
    }
}

static int ensure_io_ready(glm5_pc_engine* engine) {
    if (!engine) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(engine->io_start_mutex);
    if (!engine->io_workers.empty() || !engine->io_config.enable_compute_io_overlap) {
        return 1;
    }
    start_io_workers(engine);
    return !engine->io_config.enable_compute_io_overlap || !engine->io_workers.empty();
}
