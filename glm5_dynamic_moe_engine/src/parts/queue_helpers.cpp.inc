static void queue_total_depth_sub(glm5_pc_engine* engine, uint32_t count) {
    if (!engine || count == 0) {
        return;
    }
    uint32_t current = engine->total_queue_depth.load(std::memory_order_relaxed);
    for (;;) {
        const uint32_t next = current > count ? current - count : 0;
        if (engine->total_queue_depth.compare_exchange_weak(
                current,
                next,
                std::memory_order_relaxed,
                std::memory_order_relaxed)) {
            return;
        }
    }
}

static int enqueue_stage(
    glm5_pc_engine* engine,
    glm5_stage_queue& stage,
    const glm5_prefetch_task& task
) {
    if (!engine) {
        return 0;
    }
    {
        std::lock_guard<std::mutex> lock(stage.mtx);
        const size_t total = stage.queue.size() + stage.urgent_queue.size();
        if (total >= engine->io_config.max_prefetch_queue) {
            engine->io_atomic.dropped_requests.fetch_add(1);
            return 0;
        }
        // Fix 12: O(1) insertion via dual-deque
        if (task.urgent) {
            stage.urgent_queue.push_back(task);
        } else {
            stage.queue.push_back(task);
        }
        stage.depth.store(static_cast<uint32_t>(stage.queue.size() + stage.urgent_queue.size()));
        engine->total_queue_depth.fetch_add(1, std::memory_order_relaxed);
    }
    return 1;
}

static int enqueue_stage_or_retry(
    glm5_pc_engine* engine,
    glm5_stage_queue& stage,
    const glm5_prefetch_task& task
) {
    if (!engine) {
        return 0;
    }
    glm5_prefetch_task queued = task;
    {
        std::lock_guard<std::mutex> lock(stage.mtx);
        const size_t total = stage.queue.size() + stage.urgent_queue.size();
        const size_t normal_cap = engine->io_config.max_prefetch_queue;
        const size_t retry_room = std::max<size_t>(64u, normal_cap / 2u);
        const size_t retry_cap = normal_cap + retry_room;
        if (total >= retry_cap) {
            engine->io_atomic.dropped_requests.fetch_add(1);
            return 0;
        }
        // Bug 9: Remove requeue_count-based urgent promotion
        // Requeued speculative tasks should not jump ahead of router-selected tasks
        // Original caller's urgent flag is the only priority signal
        // if (total >= normal_cap && queued.requeue_count > 0) {
        //     queued.urgent = 1;
        // }
        if (queued.urgent) {
            stage.urgent_queue.push_back(queued);
        } else {
            stage.queue.push_back(queued);
        }
        stage.depth.store(static_cast<uint32_t>(stage.queue.size() + stage.urgent_queue.size()));
        engine->total_queue_depth.fetch_add(1, std::memory_order_relaxed);
    }
    return 1;
}

static bool pop_stage_task(
    glm5_pc_engine* engine,
    glm5_stage_queue& stage,
    glm5_prefetch_task* out_task,
    const std::atomic<uint32_t>* worker_target,
    uint32_t worker_index
) {
    std::unique_lock<std::mutex> lock(stage.mtx);
    stage.cv.wait(lock, [&] {
        const bool retiring = worker_target && worker_index >= worker_target->load();
        return engine->io_stop.load() || retiring || !stage.queue.empty() || !stage.urgent_queue.empty();
    });
    if (engine->io_stop.load() || (worker_target && worker_index >= worker_target->load())) {
        return false;
    }
    // Pop urgent first
    if (!stage.urgent_queue.empty()) {
        *out_task = stage.urgent_queue.front();
        stage.urgent_queue.pop_front();
    } else {
        *out_task = stage.queue.front();
        stage.queue.pop_front();
    }
    stage.depth.store(static_cast<uint32_t>(stage.queue.size() + stage.urgent_queue.size()));
    stage.pop_cv.notify_one();
    queue_total_depth_sub(engine, 1);
    stage.inflight.fetch_add(1);
    return true;
}

static void finish_stage_task(glm5_pc_engine* engine, glm5_stage_queue& stage, int ok, int count_completed) {
    if (ok && count_completed) {
        engine->io_atomic.completed_requests.fetch_add(1);
    } else if (!ok) {
        engine->io_atomic.failed_requests.fetch_add(1);
    }
    stage.inflight.fetch_sub(1);
    engine->io_done_cv.notify_one();
}

static bool stage_idle(glm5_stage_queue& stage) {
    return stage.depth.load() == 0 && stage.inflight.load() == 0;
}

static void clear_stage(glm5_pc_engine* engine, glm5_stage_queue& stage) {
    std::lock_guard<std::mutex> lock(stage.mtx);
    const uint32_t removed = static_cast<uint32_t>(stage.queue.size() + stage.urgent_queue.size());
    stage.queue.clear();
    stage.urgent_queue.clear();
    stage.depth.store(0);
    stage.inflight.store(0);
    queue_total_depth_sub(engine, removed);
}

static void record_dropped_request(glm5_pc_engine* engine) {
    if (engine) {
        engine->io_atomic.dropped_requests.fetch_add(1);
    }
}

static int should_use_direct_device_path(
    glm5_pc_engine* engine,
    const glm5_tensor_record& rec,
    glm5_tier_t target,
    glm5_io_path_t path
) {
    if (!engine || target != GLM5_TIER_VRAM) {
        return 0;
    }
    if (path != GLM5_IO_PATH_DIRECTSTORAGE &&
        path != GLM5_IO_PATH_GPUDIRECT_STORAGE &&
        path != GLM5_IO_PATH_BAM_NVME) {
        return 0;
    }
    return tensor_stream_bytes(rec) >= engine->io_config.direct_upload_min_bytes ? 1 : 0;
}

