// BUGFIX 55: Issue 30 - Protect stop_io_workers with io_start_mutex ?�★
// Problem: stop_io_workers sets io_stop=false without holding io_start_mutex.
// If ensure_io_ready (holds io_start_mutex) and set_model_root (calls stop_io_workers)
// run concurrently, io_stop can flip while workers are being created, causing race.
// Solution: Callers of stop_io_workers must hold io_start_mutex to serialize with
// ensure_io_ready. This prevents concurrent worker start/stop.
// Note: stop_io_workers itself doesn't take the mutex to allow flexible locking by callers.
static void stop_io_workers_locked(moe_pc_engine* engine) {
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
    engine->disk_stage.pop_cv.notify_all();
    engine->pinned_stage.pop_cv.notify_all();
    engine->gpu_stage.pop_cv.notify_all();
    engine->staging_slot_cv.notify_all();

    // Fix: Clear stages BEFORE joining to prevent workers getting stuck
    clear_stage(engine, engine->disk_stage);
    clear_stage(engine, engine->pinned_stage);
    clear_stage(engine, engine->gpu_stage);

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
    if (engine->eviction_worker.joinable()) {
        engine->eviction_worker.join();
    }
    device_force_clear_pending_copies(engine);
    // Fix 71: force_blocking=true on shutdown to ensure all threads joined
    flush_deferred_joins_locked(engine, true);
    engine->io_workers.clear();
    engine->pinned_workers.clear();
    engine->gpu_workers.clear();
    engine->total_queue_depth.store(0, std::memory_order_relaxed);
    engine->io_stop.store(false);
    engine->io_done_cv.notify_all();
}

static void stop_io_workers(moe_pc_engine* engine) {
    if (!engine) {
        return;
    }
    std::lock_guard<std::mutex> lock(engine->io_start_mutex);
    stop_io_workers_locked(engine);
}
static void start_io_workers(moe_pc_engine* engine) {
    if (!engine || !engine->io_config.enable_compute_io_overlap) {
        return;
    }
    uint32_t workers = runtime_disk_workers(engine);
    if (workers == 0) {
        workers = 1;
    }
    workers = clamp_to_cap(workers, disk_worker_cap());
    // BUGFIX 553: Check workers > 0 after clamp_to_cap ★
    // Problem: clamp_to_cap can return 0 if disk_worker_cap() returns 0
    // Loop creates 0 threads but disk_worker_target is set to non-zero
    // Solution: Recheck workers after clamp
    if (workers == 0) {
        return;
    }
    engine->disk_worker_target.store(workers);
    for (uint32_t i = 0; i < workers; ++i) {
        engine->io_workers.emplace_back(io_worker_loop, engine, i);
    }
    if (engine->io_config.enable_multistage_pipeline && pinned_worker_cap(engine) > 0) {
        uint32_t pinned_workers = runtime_pinned_workers(engine);
        pinned_workers = clamp_to_cap(pinned_workers, pinned_worker_cap(engine));
        // BUGFIX 554: Check pinned_workers > 0 after clamp ★
        // Problem: clamp_to_cap can return 0, loop creates 0 threads
        // Solution: Skip if workers == 0
        if (pinned_workers > 0) {
            engine->pinned_worker_target.store(pinned_workers);
            for (uint32_t i = 0; i < pinned_workers; ++i) {
                engine->pinned_workers.emplace_back(pinned_worker_loop, engine, i);
            }
        }
    }

    if ((engine->io_config.enable_multistage_pipeline ||
            (engine->io_config.enable_pinned_host && engine->staging)) &&
        gpu_worker_cap(engine) > 0) {
        uint32_t gpu_workers = runtime_gpu_workers(engine);
        gpu_workers = clamp_to_cap(gpu_workers, gpu_worker_cap(engine));
        // BUGFIX 555: Check gpu_workers > 0 after clamp ★
        // Problem: clamp_to_cap can return 0, loop creates 0 threads
        // Solution: Skip if workers == 0
        if (gpu_workers > 0) {
            engine->gpu_worker_target.store(gpu_workers);
            for (uint32_t i = 0; i < gpu_workers; ++i) {
                engine->gpu_workers.emplace_back(gpu_worker_loop, engine, i);
            }
        }
    }
    if (!engine->eviction_worker.joinable()) {
        engine->eviction_worker = std::thread(proactive_eviction_worker_loop, engine);
    }
}

static int ensure_io_ready(moe_pc_engine* engine) {
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
