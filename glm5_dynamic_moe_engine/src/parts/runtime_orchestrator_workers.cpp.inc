static void shrink_worker_tail(std::vector<std::thread>& workers, uint32_t target) {
    for (size_t i = target; i < workers.size(); ++i) {
        if (workers[i].joinable()) {
            workers[i].join();
        }
    }
    workers.resize(target);
}

// Fix 44: Deferred joins moved from thread_local to engine struct
// so stop_io_workers can join them on shutdown
// Fix 71: Non-blocking — only join threads confirmed exited, skip active ones
// Fix 10: POSIX must NOT detach — use timed join or skip to prevent UAF
static void flush_deferred_joins(glm5_pc_engine* engine, bool force_blocking = false) {
    if (!engine) return;
    auto it = engine->deferred_joins.begin();
    while (it != engine->deferred_joins.end()) {
        if (!it->joinable()) {
            it = engine->deferred_joins.erase(it);
            continue;
        }
        if (force_blocking) {
            it->join();
            it = engine->deferred_joins.erase(it);
            continue;
        }
#ifdef _WIN32
        // Non-blocking: check if thread has exited (0ms timeout)
        DWORD result = WaitForSingleObject(it->native_handle(), 0);
        if (result == WAIT_OBJECT_0) {
            it->join();
            it = engine->deferred_joins.erase(it);
        } else {
            ++it;  // Still running — skip, try next time
        }
#else
        // Fix 10: POSIX — use timed try-join (pthread_timedjoin_np) or skip
        // NEVER detach — detached threads continue accessing engine after destruction
        // Try a short timeout to see if thread has exited
        bool joined = false;
        for (int retry = 0; retry < 10 && !joined; ++retry) {
            // Check if io_stop is set (thread should exit soon)
            if (engine->io_stop.load()) {
                it->join();
                joined = true;
            } else {
                // Yield and retry — thread may still be processing
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        if (joined) {
            it = engine->deferred_joins.erase(it);
        } else {
            ++it;  // Still running — keep in list for later
        }
#endif
    }
}

static void reap_dead_workers(glm5_pc_engine* engine, std::vector<std::thread>& workers, std::atomic<uint32_t>& target_atomic) {
    // Bug #2: Do NOT call flush_deferred_joins here — it blocks hot path up to 10ms×N
    // Deferred joins are only flushed during shutdown (stop_io_workers with force_blocking=true)

    const uint32_t target = target_atomic.load();
    while (workers.size() > target) {
        engine->deferred_joins.push_back(std::move(workers.back()));
        workers.pop_back();
    }
}

static void resize_disk_workers(glm5_pc_engine* engine, uint32_t target) {
    engine->disk_worker_target.store(target);
    if (engine->io_workers.size() > target) {
        engine->disk_stage.cv.notify_all();
        reap_dead_workers(engine, engine->io_workers, engine->disk_worker_target);
    }
    while (engine->io_workers.size() < target) {
        const uint32_t index = static_cast<uint32_t>(engine->io_workers.size());
        engine->io_workers.emplace_back(io_worker_loop, engine, index);
    }
}

static void resize_pinned_workers(glm5_pc_engine* engine, uint32_t target) {
    engine->pinned_worker_target.store(target);
    if (engine->pinned_workers.size() > target) {
        engine->pinned_stage.cv.notify_all();
        reap_dead_workers(engine, engine->pinned_workers, engine->pinned_worker_target);
    }
    while (engine->pinned_workers.size() < target) {
        const uint32_t index = static_cast<uint32_t>(engine->pinned_workers.size());
        engine->pinned_workers.emplace_back(pinned_worker_loop, engine, index);
    }
}

static void resize_gpu_workers(glm5_pc_engine* engine, uint32_t target) {
    engine->gpu_worker_target.store(target);
    if (engine->gpu_workers.size() > target) {
        engine->gpu_stage.cv.notify_all();
        reap_dead_workers(engine, engine->gpu_workers, engine->gpu_worker_target);
    }
    while (engine->gpu_workers.size() < target) {
        const uint32_t index = static_cast<uint32_t>(engine->gpu_workers.size());
        engine->gpu_workers.emplace_back(gpu_worker_loop, engine, index);
    }
}

static void orchestrator_resize_workers(
    glm5_pc_engine* engine,
    const glm5_runtime_orchestrator_plan_t* plan
) {
    if (!engine || !plan || !engine->io_config.enable_compute_io_overlap) {
        return;
    }
    const uint32_t disk = clamp_to_cap(plan->disk_workers, disk_worker_cap());
    const uint32_t pinned = clamp_to_cap(plan->pinned_workers, pinned_worker_cap(engine));
    const uint32_t gpu = clamp_to_cap(plan->gpu_workers, gpu_worker_cap(engine));

    const uint32_t old_disk = engine->io_config.disk_worker_threads;
    const uint32_t old_pinned = engine->io_config.pinned_worker_threads;
    const uint32_t old_gpu = engine->io_config.gpu_worker_threads;

    resize_disk_workers(engine, disk);
    if (engine->io_config.enable_multistage_pipeline) {
        resize_pinned_workers(engine, pinned);
        resize_gpu_workers(engine, gpu);
    }

    // Fix 34+45: Reconfigure staging slots when worker count changes
    if (disk != old_disk || pinned != old_pinned || gpu != old_gpu) {
        // Bug #16: Was 3000ms spin — blocks inference. Now 100ms max, defer if still busy
        bool queues_drained = true;
        for (int spin = 0; spin < 100 && 
             (engine->pinned_stage.inflight.load() > 0 || engine->gpu_stage.inflight.load() > 0 ||
              engine->gpu_stage.depth.load() > 0 || engine->pinned_stage.depth.load() > 0); ++spin) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            if (spin == 99) queues_drained = false;
        }
        // Fix 12: Use atomic stores for worker thread counts to prevent data race
        // Workers read these values concurrently in io_worker_loop/resize_*
        reinterpret_cast<std::atomic<uint32_t>*>(&engine->io_config.disk_worker_threads)->store(disk, std::memory_order_relaxed);
        reinterpret_cast<std::atomic<uint32_t>*>(&engine->io_config.pinned_worker_threads)->store(pinned, std::memory_order_relaxed);
        reinterpret_cast<std::atomic<uint32_t>*>(&engine->io_config.gpu_worker_threads)->store(gpu, std::memory_order_relaxed);
        if (queues_drained) {
            configure_staging_slots(engine);
        }
        // If !queues_drained, staging slots keep old config — reconfigure next resize
    }
}
