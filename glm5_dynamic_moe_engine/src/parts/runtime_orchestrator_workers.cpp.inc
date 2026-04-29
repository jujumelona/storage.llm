static void shrink_worker_tail(std::vector<std::thread>& workers, uint32_t target) {
    for (size_t i = target; i < workers.size(); ++i) {
        if (workers[i].joinable()) {
            workers[i].join();
        }
    }
    workers.resize(target);
}

// Fix 44: Deferred joins moved from thread_local to engine struct
// so stop_io_workers can join them on shutdown.
// Fix 71: Non-blocking: only join threads confirmed exited, skip active ones.
// Fix 10: POSIX must not detach; use timed join or skip to prevent UAF.
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
        // Non-blocking: check if thread has exited (0ms timeout).
        DWORD result = WaitForSingleObject(it->native_handle(), 0);
        if (result == WAIT_OBJECT_0) {
            it->join();
            it = engine->deferred_joins.erase(it);
        } else {
            ++it;
        }
#else
        bool joined = false;
        for (int retry = 0; retry < 10 && !joined; ++retry) {
            if (engine->io_stop.load()) {
                it->join();
                joined = true;
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        if (joined) {
            it = engine->deferred_joins.erase(it);
        } else {
            ++it;
        }
#endif
    }
}

static void reap_dead_workers(glm5_pc_engine* engine, std::vector<std::thread>& workers, std::atomic<uint32_t>& target_atomic) {
    // Deferred joins are only flushed during shutdown.
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

    const uint32_t old_disk = runtime_disk_workers(engine);
    const uint32_t old_pinned = runtime_pinned_workers(engine);
    const uint32_t old_gpu = runtime_gpu_workers(engine);

    resize_disk_workers(engine, disk);
    if (engine->io_config.enable_multistage_pipeline) {
        resize_pinned_workers(engine, pinned);
    }
    if (engine->io_config.enable_multistage_pipeline || (engine->io_config.enable_pinned_host && engine->staging)) {
        resize_gpu_workers(engine, gpu);
    }

    if (disk != old_disk || pinned != old_pinned || gpu != old_gpu) {
        runtime_store_io_workers(engine, disk, pinned, gpu);
        update_staging_slot_split_for_workers(engine);
        if (engine->cpu_row_active.load(std::memory_order_acquire) == 0) {
            std::lock_guard<std::mutex> row_lock(engine->cpu_row_mutex);
            if (engine->cpu_row_active.load(std::memory_order_relaxed) == 0 &&
                engine->cpu_row_done_generation == engine->cpu_row_generation) {
                engine->cpu_row_target_workers.store(0, std::memory_order_relaxed);
            }
        }
    }
}
