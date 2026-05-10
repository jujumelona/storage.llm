
static uint64_t moe_prefetch_task_estimated_bytes(
    const moe_pc_engine* engine,
    const moe_prefetch_task& task
) {
    if (!engine) return 0;
    if (task.bundle_triplet && task.bundle_layer >= 0 && task.bundle_expert >= 0) {
        uint64_t bundle_sum = 0;
        for (uint32_t i = 0; i < 3u; ++i) bundle_sum += task.bundle_bytes[i];
        if (bundle_sum > 0) return bundle_sum;
        return expert_triplet_or_hardcoded_bytes(engine, task.bundle_layer, task.bundle_expert);
    }
    if (task.staging_bytes > 0) return task.staging_bytes;
    if (task.tensor_index < engine->tensors.size()) {
        return tensor_stream_bytes(engine->tensors[task.tensor_index]);
    }
    if (task.raw_span_index >= 0) {
        const moe_storage_raw_span_spec_t* span = moe_storage_raw_span_at((uint32_t)task.raw_span_index);
        return span ? span->bytes : 0;
    }
    return 0;
}

static void moe_atomic_u64_saturating_sub(std::atomic<uint64_t>& value, uint64_t bytes) {
    if (bytes == 0) return;
    uint64_t cur = value.load(std::memory_order_relaxed);
    while (cur > 0) {
        const uint64_t next = cur > bytes ? cur - bytes : 0;
        if (value.compare_exchange_weak(
                cur, next,
                std::memory_order_relaxed,
                std::memory_order_relaxed)) {
            break;
        }
    }
}

static void moe_stage_queued_bytes_sub(moe_stage_queue& stage, uint64_t bytes) {
    moe_atomic_u64_saturating_sub(stage.queued_bytes, bytes);
}

static void moe_stage_inflight_bytes_sub(moe_stage_queue& stage, uint64_t bytes) {
    moe_atomic_u64_saturating_sub(stage.inflight_bytes, bytes);
}

static uint64_t moe_stage_byte_cap(
    const moe_pc_engine* engine,
    const moe_stage_queue& stage
) {
    if (!engine) return 0;
    const uint64_t max_tensor = std::max<uint64_t>(
        engine->max_tensor_stream_bytes.load(std::memory_order_relaxed),
        normal_bundle_max_or(32ull * 1024ull * 1024ull));
    if (&stage == &engine->gpu_stage) {
        const uint64_t stream_cap = engine->backend_caps.copy_stream_bytes_limit > 0
            ? engine->backend_caps.copy_stream_bytes_limit
            : std::max<uint64_t>(64ull * 1024ull * 1024ull,
                  (uint64_t)std::max<uint32_t>(1u, engine->backend_caps.copy_streams) *
                  32ull * 1024ull * 1024ull);
        const uint64_t workers = std::max<uint32_t>(1u, runtime_gpu_workers(engine));
        const uint64_t staging_cap = engine->staging_size > 0 ? engine->staging_size : stream_cap * workers;
        return std::max<uint64_t>(max_tensor * workers, std::min<uint64_t>(staging_cap, stream_cap * workers * 2ull));
    }
    if (&stage == &engine->pinned_stage) {
        const uint64_t workers = std::max<uint32_t>(1u, runtime_pinned_workers(engine));
        const uint64_t staging_cap = engine->staging_size > 0 ? engine->staging_size * 2ull : max_tensor * workers * 4ull;
        return std::max<uint64_t>(max_tensor * workers * 2ull, staging_cap);
    }
    if (&stage == &engine->disk_stage) {
        const uint64_t workers = std::max<uint32_t>(1u, runtime_disk_workers(engine));
        const uint64_t window = std::max<uint32_t>(
            1u, engine->runtime_prefetch_window_layers.load(std::memory_order_relaxed));
        const uint64_t storage_bw = engine->backend_caps.measured_storage_bw_bytes_per_sec;
        const uint64_t bw_window = storage_bw > 0
            ? std::max<uint64_t>(max_tensor * workers, storage_bw / 4ull)
            : max_tensor * workers * std::max<uint64_t>(2ull, window);
        const uint64_t lookahead_bytes = max_tensor * workers * std::max<uint64_t>(2ull, window) * 2ull;
        uint64_t ram_guard = engine->config.ram_budget_bytes > 0
            ? engine->config.ram_budget_bytes / 8ull
            : 2ull * 1024ull * 1024ull * 1024ull;
        ram_guard = std::max<uint64_t>(ram_guard, max_tensor * workers * 2ull);
        return std::max<uint64_t>(max_tensor * workers * 2ull,
            std::min<uint64_t>(std::max<uint64_t>(bw_window, lookahead_bytes), ram_guard));
    }
    return 0;
}

static bool moe_stage_byte_room(
    moe_pc_engine* engine,
    const moe_stage_queue& stage,
    const moe_prefetch_task& task,
    uint64_t task_bytes
) {
    if (!engine || task_bytes == 0) return true;
    const uint64_t cap = moe_stage_byte_cap(engine, stage);
    if (cap == 0) return true;
    const uint64_t queued = stage.queued_bytes.load(std::memory_order_relaxed);
    const uint64_t inflight = stage.inflight_bytes.load(std::memory_order_relaxed);
    const uint64_t used = queued > UINT64_MAX - inflight ? UINT64_MAX : queued + inflight;
    const uint64_t allowed = (task.urgent || task.urgent_required || task.required_now)
        ? cap + cap / 4ull
        : cap;
    return used <= allowed && task_bytes <= allowed - used;
}

// BUGFIX 288: Fix underflow recovery with CAS loop (same pattern as BUGFIX 256) ★
// Problem: fetch_sub then fetch_add for recovery is not atomic
// If old_val < count, underflow already occurred. Another thread may have
// modified total_queue_depth between fetch_sub and fetch_add.
// Solution: Use CAS loop to atomically prevent underflow
static void queue_total_depth_sub(moe_pc_engine* engine, uint32_t count) {
    if (!engine || count == 0) {
        return;
    }
    uint32_t expected = engine->total_queue_depth.load(std::memory_order_relaxed);
    while (true) {
        if (expected < count) {
            // Underflow would occur - clamp to 0
            if (engine->total_queue_depth.compare_exchange_weak(
                    expected, 0,
                    std::memory_order_relaxed,
                    std::memory_order_relaxed)) {
                break;
            }
        } else {
            // Normal case - subtract count
            if (engine->total_queue_depth.compare_exchange_weak(
                    expected, expected - count,
                    std::memory_order_relaxed,
                    std::memory_order_relaxed)) {
                break;
            }
        }
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
    if (engine->io_reconfiguring.load(std::memory_order_acquire)) {
        engine->io_atomic.dropped_requests.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    moe_prefetch_task queued = task;
    queued.stage_accounted_bytes = 0;
    {
        std::lock_guard<std::mutex> lock(stage.mtx);
        // BUGFIX 97: Issue 18 - Cache queue size calculations ★
        // Problem: size() called twice per enqueue (total check + depth store)
        // Solution: Cache size calculation to avoid repeated calls
        const size_t queue_sz = stage.queue.size();
        const size_t urgent_sz = stage.urgent_queue.size();
        const size_t total = queue_sz + urgent_sz;

        // BUGFIX 645: 단계별 큐 깊이 cap 적용 ★★★
        uint32_t stage_cap = engine->io_config.max_prefetch_queue;
        if (&stage == &engine->disk_stage) {
            stage_cap = engine->disk_stage_depth_cap;
        } else if (&stage == &engine->pinned_stage) {
            stage_cap = engine->pinned_stage_depth_cap;
        } else if (&stage == &engine->gpu_stage) {
            stage_cap = engine->gpu_stage_depth_cap;
        }

        const uint64_t task_bytes = moe_prefetch_task_estimated_bytes(engine, queued);
        if (total >= stage_cap || !moe_stage_byte_room(engine, stage, queued, task_bytes)) {
            // BUGFIX 154: Add memory_order_relaxed for statistics counter
            engine->io_atomic.dropped_requests.fetch_add(1, std::memory_order_relaxed);
            return 0;
        }
        // Fix 12: O(1) insertion via dual-deque
        if (queued.urgent) {
            stage.urgent_queue.push_back(queued);
        } else {
            stage.queue.push_back(queued);
        }
        // BUGFIX 205: Add memory_order_relaxed for depth counter ★
        stage.depth.store(static_cast<uint32_t>(queue_sz + urgent_sz + 1), std::memory_order_relaxed);  // +1 for just-added task
        if (task_bytes > 0) stage.queued_bytes.fetch_add(task_bytes, std::memory_order_relaxed);
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
    if (engine->io_reconfiguring.load(std::memory_order_acquire)) {
        engine->io_atomic.dropped_requests.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    moe_prefetch_task queued = task;
    queued.stage_accounted_bytes = 0;
    size_t final_depth_after = 0;  // Fix 30: declared outside lock scope
    {
        std::lock_guard<std::mutex> lock(stage.mtx);
        // BUGFIX 97: Cache queue size calculations
        const size_t queue_sz = stage.queue.size();
        const size_t urgent_sz = stage.urgent_queue.size();
        const size_t total = queue_sz + urgent_sz;

        // BUGFIX 645: 단계별 큐 깊이 cap 적용 ★★★
        size_t normal_cap = engine->io_config.max_prefetch_queue;
        if (&stage == &engine->disk_stage) {
            normal_cap = engine->disk_stage_depth_cap;
        } else if (&stage == &engine->pinned_stage) {
            normal_cap = engine->pinned_stage_depth_cap;
        } else if (&stage == &engine->gpu_stage) {
            normal_cap = engine->gpu_stage_depth_cap;
        }

        // PIPELINE FIX 18: Reduce retry_cap from 1.5x to 1.25x ??�쁾
        // Old: retry_room = normal_cap/2 ??queue can grow to 150%
        // New: retry_room = normal_cap/4 ??queue can grow to 125%
        // Reduces queue inflation and makes dropped_requests metric more meaningful
        const size_t retry_room = std::max<size_t>(32u, normal_cap / 4u);
        const size_t retry_cap = normal_cap + retry_room;
        const uint64_t task_bytes = moe_prefetch_task_estimated_bytes(engine, queued);
        if (total >= retry_cap || !moe_stage_byte_room(engine, stage, queued, task_bytes)) {
            // BUGFIX 154: Add memory_order_relaxed for statistics counter
            engine->io_atomic.dropped_requests.fetch_add(1, std::memory_order_relaxed);
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
                    // BUGFIX 201: Track urgent drops separately ★
                    engine->io_atomic.urgent_dropped_requests.fetch_add(1, std::memory_order_relaxed);
                    // BUGFIX 154: Add memory_order_relaxed for statistics counter
                    engine->io_atomic.dropped_requests.fetch_add(1, std::memory_order_relaxed);
                    return 0;
                }
            } else {
                stage.urgent_queue.push_back(queued);
            }
        } else {
            stage.queue.push_back(queued);
        }
        // BUGFIX Issue 7: Move depth.store inside lock to prevent stale-store race
        // Problem: Thread A computes final=5, releases lock, Thread B stores 6, Thread A stores 5
        // Result: depth shows 5 but actual queue has 6 items → admission underestimates pressure
        // Solution: Store depth inside lock, same pattern as enqueue_stage
        final_depth_after = stage.queue.size() + stage.urgent_queue.size();
        stage.depth.store(static_cast<uint32_t>(final_depth_after), std::memory_order_relaxed);
        if (task_bytes > 0) stage.queued_bytes.fetch_add(task_bytes, std::memory_order_relaxed);
    }
    // total_queue_depth is safe to update outside lock (atomic, no ordering dependency)
    engine->total_queue_depth.fetch_add(1, std::memory_order_relaxed);
    return 1;
}

static int enqueue_directstorage_staging_pending(
    moe_pc_engine* engine,
    const moe_dstorage_pending_copy& pending,
    uint32_t staging_slot,
    const moe_prefetch_task& next_task,
    uint64_t expected_bytes
) {
    if (!engine || staging_slot == UINT32_MAX || expected_bytes == 0 || !pending.status_array) {
        return 0;
    }
    moe_directstorage_staging_pending item{};
    item.io = pending;
    item.staging_slot = staging_slot;
    item.next_task = next_task;
    item.expected_bytes = expected_bytes;
    std::lock_guard<std::mutex> lock(engine->directstorage_staging_pending_mutex);
    engine->directstorage_pending_staging_copies.push_back(item);
    engine->directstorage_pending_count.fetch_add(1, std::memory_order_release);
    return 1;
}

static size_t drain_directstorage_staging_copies(moe_pc_engine* engine) {
    if (!engine) {
        return 0;
    }
    size_t drained = 0;
    std::vector<moe_directstorage_staging_pending> ready;
    std::vector<moe_directstorage_staging_pending> failed;
    {
        std::lock_guard<std::mutex> lock(engine->directstorage_staging_pending_mutex);
        auto& pending = engine->directstorage_pending_staging_copies;
        for (size_t i = 0; i < pending.size();) {
            moe_directstorage_staging_pending& item = pending[i];
            const int state = directstorage_poll_pending_copy(engine, &item.io);
            if (state == 0) {
                ++i;
                continue;
            }
            if (state > 0) {
                ready.push_back(item);
            } else {
                failed.push_back(item);
            }
            pending[i] = pending.back();
            pending.pop_back();
            engine->directstorage_pending_count.fetch_sub(1, std::memory_order_acq_rel);
            ++drained;
        }
    }
    for (moe_directstorage_staging_pending& item : ready) {
        moe_prefetch_task next = item.next_task;
        next.staging_slot = item.staging_slot;
        next.staging_bytes = item.expected_bytes;
        next.staging_valid = 1;
        if (enqueue_stage_or_retry(engine, engine->gpu_stage, next)) {
            engine->io_atomic.bytes_prefetched.fetch_add(item.expected_bytes, std::memory_order_relaxed);
            engine->io_atomic.bytes_disk_to_ram.fetch_add(item.expected_bytes, std::memory_order_relaxed);
            engine->io_atomic.pinned_stage_done.fetch_add(1, std::memory_order_relaxed);
            engine->gpu_stage.cv.notify_one();
        } else {
            release_staging_slot(engine, item.staging_slot);
            if (next.vram_token.valid) {
                device_vram_release_reservation_token(engine, next.vram_token);
                next.vram_token.valid = false;
            }
            engine->io_atomic.dropped_requests.fetch_add(1, std::memory_order_relaxed);
        }
    }
    for (moe_directstorage_staging_pending& item : failed) {
        release_staging_slot(engine, item.staging_slot);
        if (item.next_task.vram_token.valid) {
            device_vram_release_reservation_token(engine, item.next_task.vram_token);
            item.next_task.vram_token.valid = false;
        }
        engine->io_atomic.dropped_requests.fetch_add(1, std::memory_order_relaxed);
    }
    return drained;
}

static bool pop_stage_task(
    moe_pc_engine* engine,
    moe_stage_queue& stage,
    moe_prefetch_task* out_task,
    const std::atomic<uint32_t>* worker_target,
    uint32_t worker_index
) {
    std::unique_lock<std::mutex> lock(stage.mtx);
    // BUGFIX 709: GPU Worker periodic drain during empty queue ★★★
    // Problem: When queue is empty, gpu_worker blocks indefinitely in cv.wait.
    // If there are pending deferred copies (event completions), their slots are
    // locked until the next task arrives and wakes the worker, causing staging
    // memory leaks/starvation.
    // Solution: Use adaptive micro-polling while DirectStorage work is pending
    // and fall back to the old low-frequency wait when queues are empty.
    if (&stage == &engine->gpu_stage) {
        for (;;) {
            const bool retiring = worker_target && worker_index >= worker_target->load();
            if (engine->io_stop.load() || retiring || !stage.queue.empty() || !stage.urgent_queue.empty()) {
                break;
            }
            const uint32_t wait_us =
                engine->directstorage_pending_count.load(std::memory_order_acquire) > 0 ? 100u : 2000u;
            stage.cv.wait_for(lock, std::chrono::microseconds(wait_us));
            lock.unlock();
            device_drain_deferred_staging_copies(engine, false);
            device_drain_directstorage_device_copies(engine);
            device_refresh_mem_info(engine, 0);
            lock.lock();
        }
    } else if (&stage == &engine->disk_stage) {
        for (;;) {
            const bool retiring = worker_target && worker_index >= worker_target->load();
            if (engine->io_stop.load() || retiring || !stage.queue.empty() || !stage.urgent_queue.empty()) {
                break;
            }
            const uint32_t wait_us =
                engine->directstorage_pending_count.load(std::memory_order_acquire) > 0 ? 100u : 2000u;
            stage.cv.wait_for(lock, std::chrono::microseconds(wait_us));
            lock.unlock();
            device_drain_deferred_staging_copies(engine, false);
            drain_directstorage_staging_copies(engine);
            lock.lock();
        }
    } else {
        stage.cv.wait(lock, [&] {
            const bool retiring = worker_target && worker_index >= worker_target->load();
            return engine->io_stop.load() || retiring || !stage.queue.empty() || !stage.urgent_queue.empty();
        });
    }
    if (engine->io_stop.load() || (worker_target && worker_index >= worker_target->load())) {
        return false;
    }
    // BUGFIX: Prevent urgent_queue from indefinitely starving normal queue
    static thread_local uint32_t tl_normal_turn = 0;
    const bool force_normal = (++tl_normal_turn % 8 == 0) && !stage.queue.empty();

    if (!stage.urgent_queue.empty() && !force_normal) {
        *out_task = stage.urgent_queue.front();
        stage.urgent_queue.pop_front();
    } else if (!stage.queue.empty()) {
        *out_task = stage.queue.front();
        stage.queue.pop_front();
    } else if (!stage.urgent_queue.empty()) {
        *out_task = stage.urgent_queue.front();
        stage.urgent_queue.pop_front();
    }
    const uint64_t popped_bytes = moe_prefetch_task_estimated_bytes(engine, *out_task);
    out_task->stage_accounted_bytes = popped_bytes;
    moe_stage_queued_bytes_sub(stage, popped_bytes);
    if (popped_bytes > 0) {
        stage.inflight_bytes.fetch_add(popped_bytes, std::memory_order_relaxed);
    }
    // BUGFIX 97: Cache size after pop
    const size_t new_depth = stage.queue.size() + stage.urgent_queue.size();
    // BUGFIX 205: Add memory_order_relaxed for depth counter ★
    stage.depth.store(static_cast<uint32_t>(new_depth), std::memory_order_relaxed);
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

static void finish_stage_task(
    moe_pc_engine* engine,
    moe_stage_queue& stage,
    int ok,
    int count_completed,
    uint64_t accounted_bytes = 0
) {
    if (accounted_bytes > 0) {
        moe_stage_inflight_bytes_sub(stage, accounted_bytes);
    }
    if (ok && count_completed) {
        // BUGFIX 154: Add memory_order_relaxed for statistics counters
        engine->io_atomic.completed_requests.fetch_add(1, std::memory_order_relaxed);
    } else if (!ok) {
        engine->io_atomic.failed_requests.fetch_add(1, std::memory_order_relaxed);
    }
    // BUGFIX 616: Check stage.inflight > 0 before fetch_sub to prevent underflow ★
    // Problem: If stage.inflight is 0, fetch_sub causes underflow
    // Solution: Only decrement if > 0
    uint32_t current = stage.inflight.load(std::memory_order_acquire);
    while (current > 0) {
        if (stage.inflight.compare_exchange_weak(
                current, current - 1,
                std::memory_order_release,
                std::memory_order_acquire)) {
            break;
        }
    }
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
    // BUGFIX 205: Add memory_order_relaxed for depth/inflight counters ★
    stage.depth.store(0, std::memory_order_relaxed);
    stage.inflight.store(0, std::memory_order_relaxed);
    queue_total_depth_sub(engine, removed);
}

static void record_dropped_request(moe_pc_engine* engine) {
    if (engine) {
        // BUGFIX 154: Add memory_order_relaxed for statistics counter
        engine->io_atomic.dropped_requests.fetch_add(1, std::memory_order_relaxed);
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
    if (path != moe_IO_PATH_DIRECTSTORAGE_GPU &&
        path != moe_IO_PATH_GPUDIRECT_STORAGE &&
        path != moe_IO_PATH_BAM_NVME) {
        return 0;
    }
    return tensor_stream_bytes(rec) >= engine->io_config.direct_upload_min_bytes ? 1 : 0;
}
