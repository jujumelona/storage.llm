static void queue_total_depth_sub(moe_pc_engine* engine, uint32_t count) {
    if (!engine || count == 0) {
        return;
    }
    const uint32_t old_val = engine->total_queue_depth.fetch_sub(count, std::memory_order_relaxed);
    if (old_val < count) {
        engine->total_queue_depth.fetch_add(count - old_val, std::memory_order_relaxed);
    }
}

static int enqueue_stage(
    moe_pc_engine* engine,
    moe_stage_queue& stage,
    const moe_prefetch_task& task
) {
    if (!engine) {
        return 0;
    }
    {
        std::lock_guard<std::mutex> lock(stage.mtx);
        // BUGFIX 97: Issue 18 - Cache queue size calculations ★
        // Problem: size() called twice per enqueue (total check + depth store)
        // Solution: Cache size calculation to avoid repeated calls
        const size_t queue_sz = stage.queue.size();
        const size_t urgent_sz = stage.urgent_queue.size();
        const size_t total = queue_sz + urgent_sz;
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
        stage.depth.store(static_cast<uint32_t>(queue_sz + urgent_sz + 1));  // +1 for just-added task
        engine->total_queue_depth.fetch_add(1, std::memory_order_relaxed);
    }
    return 1;
}

static int enqueue_stage_or_retry(
    moe_pc_engine* engine,
    moe_stage_queue& stage,
    const moe_prefetch_task& task
) {
    if (!engine) {
        return 0;
    }
    moe_prefetch_task queued = task;
    {
        std::lock_guard<std::mutex> lock(stage.mtx);
        // BUGFIX 97: Cache queue size calculations
        const size_t queue_sz = stage.queue.size();
        const size_t urgent_sz = stage.urgent_queue.size();
        const size_t total = queue_sz + urgent_sz;
        const size_t normal_cap = engine->io_config.max_prefetch_queue;
        // PIPELINE FIX 18: Reduce retry_cap from 1.5x to 1.25x ??�쁾
        // Old: retry_room = normal_cap/2 ??queue can grow to 150%
        // New: retry_room = normal_cap/4 ??queue can grow to 125%
        // Reduces queue inflation and makes dropped_requests metric more meaningful
        const size_t retry_room = std::max<size_t>(32u, normal_cap / 4u);
        const size_t retry_cap = normal_cap + retry_room;
        if (total >= retry_cap) {
            engine->io_atomic.dropped_requests.fetch_add(1);
            return 0;
        }

        // BUGFIX 40: Add urgent_queue size limit to prevent unbounded growth ??�쁾??        // Old: urgent_queue has no cap ??can grow infinitely if all requeues become urgent
        // New: urgent_queue capped at normal_cap/2
        // Scenario: All tasks requeue with urgent=1 ??urgent_queue grows without bound
        //           ??memory exhaustion, priority inversion (everything is "urgent")
        // Solution: Cap urgent_queue separately. If urgent_queue full, drop or demote to normal
        const size_t urgent_cap = std::max<size_t>(16u, normal_cap / 2u);

        // Bug 9: Remove requeue_count-based urgent promotion
        // Requeued speculative tasks should not jump ahead of router-selected tasks
        // Original caller's urgent flag is the only priority signal
        // if (total >= normal_cap && queued.requeue_count > 0) {
        //     queued.urgent = 1;
        // }

        if (queued.urgent) {
            if (stage.urgent_queue.size() >= urgent_cap) {
                // Urgent queue full: demote to normal queue if space, else drop
                if (total < retry_cap) {
                    queued.urgent = 0;  // Demote to normal priority
                    stage.queue.push_back(queued);
                } else {
                    engine->io_atomic.dropped_requests.fetch_add(1);
                    return 0;
                }
            } else {
                stage.urgent_queue.push_back(queued);
            }
        } else {
            stage.queue.push_back(queued);
        }
        // BUGFIX 113: Issue 25 - Fix depth calculation after conditional enqueue ★★★
        // Problem: stage.depth.store(queue_sz + urgent_sz + 1) assumes task always added
        // But if urgent queue full and demoted to normal, depth calculation is wrong
        // queue_sz and urgent_sz are OLD sizes before push_back
        // After demotion: normal queue grows by 1, urgent stays same
        // Correct depth = (queue_sz + 1) + urgent_sz = queue_sz + urgent_sz + 1 ✓
        // After urgent add: normal stays same, urgent grows by 1
        // Correct depth = queue_sz + (urgent_sz + 1) = queue_sz + urgent_sz + 1 ✓
        // After normal add: normal grows by 1, urgent stays same
        // Correct depth = (queue_sz + 1) + urgent_sz = queue_sz + urgent_sz + 1 ✓
        // Actually the formula is correct! But let's verify with actual sizes for clarity
        const size_t final_depth = stage.queue.size() + stage.urgent_queue.size();
        stage.depth.store(static_cast<uint32_t>(final_depth));
        engine->total_queue_depth.fetch_add(1, std::memory_order_relaxed);
    }
    return 1;
}

static bool pop_stage_task(
    moe_pc_engine* engine,
    moe_stage_queue& stage,
    moe_prefetch_task* out_task,
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
    // BUGFIX 97: Cache size after pop
    const size_t new_depth = stage.queue.size() + stage.urgent_queue.size();
    stage.depth.store(static_cast<uint32_t>(new_depth));
    // PIPELINE FIX 6: Increment inflight BEFORE notify to prevent race condition ??�쁾??    // Old order: notify ??inflight++ ??awakened thread sees inflight==0 ??false "idle"
    // New order: inflight++ ??notify ??awakened thread always sees inflight>=1
    // Prevents premature "pipeline done" signal when tasks are still processing
    stage.inflight.fetch_add(1);
    stage.pop_cv.notify_one();
    queue_total_depth_sub(engine, 1);
    return true;
}

// Bug 2 Fix: Non-destructive peek to enable lookahead touch_mapped_range
// Allows io_worker to prefetch next task's mmap pages before processing current task
static bool peek_stage_task(
    moe_pc_engine* engine,
    moe_stage_queue& stage,
    moe_prefetch_task* out_task
) {
    if (!engine || !out_task) {
        return false;
    }
    std::lock_guard<std::mutex> lock(stage.mtx);
    if (!stage.urgent_queue.empty()) {
        *out_task = stage.urgent_queue.front();
        return true;
    } else if (!stage.queue.empty()) {
        *out_task = stage.queue.front();
        return true;
    }
    return false;
}

static void finish_stage_task(moe_pc_engine* engine, moe_stage_queue& stage, int ok, int count_completed) {
    if (ok && count_completed) {
        engine->io_atomic.completed_requests.fetch_add(1);
    } else if (!ok) {
        engine->io_atomic.failed_requests.fetch_add(1);
    }
    stage.inflight.fetch_sub(1);
    // BUGFIX 26: Only notify on terminal completion ??�쁾
    // Old: notify on every finish, including requeues ??spurious wakeups
    // New: notify only when count_completed=true ??accurate completion signal
    // Prevents premature "all done" detection in startup_warm and other waiters
    if (count_completed) {
        engine->io_done_cv.notify_one();
    }
}

static bool stage_idle(moe_stage_queue& stage) {
    return stage.depth.load() == 0 && stage.inflight.load() == 0;
}

static void clear_stage(moe_pc_engine* engine, moe_stage_queue& stage) {
    std::lock_guard<std::mutex> lock(stage.mtx);
    // BUGFIX 97: Cache size before clear
    const uint32_t removed = static_cast<uint32_t>(stage.queue.size() + stage.urgent_queue.size());
    stage.queue.clear();
    stage.urgent_queue.clear();
    stage.depth.store(0);
    stage.inflight.store(0);
    queue_total_depth_sub(engine, removed);
}

static void record_dropped_request(moe_pc_engine* engine) {
    if (engine) {
        engine->io_atomic.dropped_requests.fetch_add(1);
    }
}

static int should_use_direct_device_path(
    moe_pc_engine* engine,
    const moe_tensor_record& rec,
    moe_tier_t target,
    moe_io_path_t path
) {
    if (!engine || target != moe_TIER_VRAM) {
        return 0;
    }
    if (path != moe_IO_PATH_DIRECTSTORAGE &&
        path != moe_IO_PATH_GPUDIRECT_STORAGE &&
        path != moe_IO_PATH_BAM_NVME) {
        return 0;
    }
    return tensor_stream_bytes(rec) >= engine->io_config.direct_upload_min_bytes ? 1 : 0;
}

