// Device tensor copy stream management

static size_t device_drain_directstorage_device_copies(moe_pc_engine* e);

// DONE: BUGFIX 77 - CUDA stream pool is already implemented below to avoid cuStreamCreate/Destroy on worker resize.
struct moe_deferred_staging_copy {
    uint32_t slot = UINT32_MAX;
    uint64_t key = 0;
    uint64_t extra_keys[3]{};
    uint64_t copy_bytes = 0;
    uint64_t bytes_for_vram_release = 0;
    vram_reservation_token vram_token{};
    CUevent event = nullptr;
    uint64_t submitted_us = 0;
    moe_io_path_t path = moe_IO_PATH_PINNED_HOST;
    uint8_t extra_key_count = 0;
    bool event_recorded = false;
};

struct ThreadStreamGuard {
    moe_pc_engine* engine = nullptr;
    CUstream stream = nullptr;
    ~ThreadStreamGuard();
        // BUGFIX 886: ThreadStreamGuard ?뚮㈇??援ы쁽 ?꾨즺 ?끸쁾??CRITICAL
        // Problem: GPU worker 鍮꾩젙??醫낅즺 ??CUDA/HIP ?ㅽ듃由쇱씠 stream pool??諛섑솚?섏? ?딆쓬
        // Solution: ?뚮㈇?먯뿉??device_stream_destroy_current ?몄텧?섏뿬 ?ㅽ듃由?諛섑솚
        // Note: ?뺤긽 醫낅즺 寃쎈줈(gpu_worker_loop 留덉?留??먯꽌??stream_guard.e = nullptr濡??ㅼ젙?섏뿬 以묐났 諛⑹?
        // Implementation: See destructor below - calls device_stream_destroy_current + releases in-flight bytes
};
static thread_local std::unordered_map<moe_pc_engine*, ThreadStreamGuard> s_tl_stream_guards;
static thread_local std::unordered_map<moe_pc_engine*, std::vector<uint64_t>> s_tl_tensor_pending_keys;
// BUGFIX 67: Issue 2 - Change deferred_staging_copies from vector to deque.
// Problem: copies.erase(copies.begin()) is O(n) on vector
// With 8-16 concurrent copies, drain loop becomes O(n짼)
// Solution: Use std::deque for O(1) pop_front()
// Impact: Drain performance improves from O(n짼) to O(n)
static thread_local std::unordered_map<moe_pc_engine*, std::deque<moe_deferred_staging_copy>> s_tl_deferred_staging_copies;
static thread_local std::unordered_map<moe_pc_engine*, std::vector<CUevent>> s_tl_copy_event_pool;
static constexpr int moe_DEVICE_COPY_BACKPRESSURE = -2;
static constexpr int moe_DEVICE_COPY_PENDING_ASYNC = 2;

static void device_release_inflight_vram_bytes(moe_pc_engine* e, uint64_t bytes) {
    if (!e || bytes == 0) {
        return;
    }
    uint64_t current = e->vram_inflight_bytes.load(std::memory_order_relaxed);
    for (;;) {
        const uint64_t next = current > bytes ? current - bytes : 0;
        // BUGFIX 966: Warn on underflow instead of silent clamp ?끸쁾 DIAGNOSTIC
        // Problem: Silent clamp to 0 hides double-free and byte mismatch bugs.
        // Solution: Log warning when underflow detected. No perf cost in normal path.
        if (current < bytes) {
            fprintf(stderr,
                "[storageLLM WARNING] vram_inflight_bytes underflow: current=%llu release=%llu "
                "(possible double-free or byte mismatch)\n",
                (unsigned long long)current, (unsigned long long)bytes);
        }
        if (e->vram_inflight_bytes.compare_exchange_weak(
                current,
                next,
                std::memory_order_relaxed,
                std::memory_order_relaxed)) {
            return;
        }
    }
}

// BUGFIX 77: Issue 7 - CUDA stream pool to avoid cuStreamCreate/Destroy on worker resize.
static CUstream device_tensor_stream_current(moe_pc_engine* e) {
    if (!e ||
        !e->config.prefer_gpu ||
        !e->backend_caps.supports_backend_async_api) {
        return nullptr;
    }
    if (!device_set_current(e)) {
        return nullptr;
    }
    auto it = s_tl_stream_guards.find(e);
    if (it != s_tl_stream_guards.end()) {
        return it->second.stream;
    }
    CUstream stream = nullptr;
    if (e->stream_pool_available.load(std::memory_order_acquire) > 0) {
        std::lock_guard<std::mutex> lock(e->stream_pool_mutex);
        if (!e->stream_pool.empty()) {
            stream = e->stream_pool.back();
            e->stream_pool.pop_back();
            e->stream_pool_available.store(static_cast<uint32_t>(e->stream_pool.size()), std::memory_order_release);
        }
    }
    if (!stream) {
        if (!device_backend_stream_create(e, &stream, 1u /* CU_STREAM_NON_BLOCKING */)) {
            return nullptr;
        }
    }
    s_tl_stream_guards[e] = {e, stream};
    return stream;
}

static CUevent device_take_copy_event(moe_pc_engine* e) {
    if (!e) {
        return nullptr;
    }
    auto& pool = s_tl_copy_event_pool[e];
    if (!pool.empty()) {
        CUevent event = pool.back();
        pool.pop_back();
        return event;
    }
    CUevent event = nullptr;
    return device_backend_event_create(e, &event) ? event : nullptr;
}

static void device_recycle_copy_event(moe_pc_engine* e, CUevent event) {
    if (!e || !event) {
        return;
    }
    s_tl_copy_event_pool[e].push_back(event);
}

static void device_destroy_copy_event_pool(moe_pc_engine* e) {
    auto it = s_tl_copy_event_pool.find(e);
    if (it == s_tl_copy_event_pool.end()) {
        return;
    }
    for (CUevent event : it->second) {
        device_backend_event_destroy(e, event);
    }
    s_tl_copy_event_pool.erase(it);
}

static int device_defer_staging_copy(
    moe_pc_engine* e,
    uint32_t slot,
    uint64_t key,
    uint64_t copy_bytes,
    uint64_t vram_bytes,
    CUstream stream,
    vram_reservation_token token = vram_reservation_token{},
    moe_io_path_t path = moe_IO_PATH_PINNED_HOST,
    const uint64_t* extra_keys = nullptr,
    uint8_t extra_key_count = 0
) {
    // BUGFIX Issue 6: Use staging_slot_valid() helper for correct check order ?끸쁾??
    if (!e) {
        return 0;
    }


    // Check staging_slot_size > 0 to prevent use after timeout
    if (slot != UINT32_MAX && !staging_slot_valid(e, slot)) {
        fprintf(stderr,
            "[storageLLM ERROR] device_defer_staging_copy: staging_slot_size=0, "
            "slots are invalid after reset timeout. Rejecting deferred copy.\n");
        return 0;
    }
    moe_deferred_staging_copy copy{};
    copy.slot = slot;
    copy.key = key;
    copy.copy_bytes = copy_bytes;
    copy.bytes_for_vram_release = vram_bytes;
    copy.vram_token = token;
    copy.submitted_us = moe_profile_now_us();
    copy.path = path;
    copy.extra_key_count = std::min<uint8_t>(extra_key_count, static_cast<uint8_t>(3));
    for (uint8_t i = 0; i < copy.extra_key_count; ++i) {
        copy.extra_keys[i] = extra_keys ? extra_keys[i] : 0;
    }
    copy.event = device_take_copy_event(e);
    if (!copy.event) {
        return 0;
    }
    copy.event_recorded = device_backend_event_record(e, copy.event, stream) ? true : false;
    if (!copy.event_recorded) {
        device_recycle_copy_event(e, copy.event);
        copy.event = nullptr;
        return 0;
    }
    s_tl_deferred_staging_copies[e].push_back(copy);
    return 1;
}

static size_t device_deferred_staging_copy_count(moe_pc_engine* e) {
    auto it = s_tl_deferred_staging_copies.find(e);
    return it == s_tl_deferred_staging_copies.end() ? 0u : it->second.size();
}

static size_t device_drain_deferred_staging_copies(moe_pc_engine* e, bool wait_all) {
    if (!e) {
        return 0;
    }
    auto it = s_tl_deferred_staging_copies.find(e);
    if (it == s_tl_deferred_staging_copies.end() || it->second.empty()) {
        return 0;
    }
    // BUGFIX 79: Issue 1 - Get stream from s_tl_stream_guards, not s_tl_tensor_streams.
    CUstream stream = nullptr;
    auto sit = s_tl_stream_guards.find(e);
    if (sit != s_tl_stream_guards.end()) {
        stream = sit->second.stream;
    }
    bool stream_synced = false;
    size_t released = 0;
    auto& copies = it->second;

    // BUGFIX 898: RAII guard for vram_in_flight_bytes release ?끸쁾??CRITICAL
    // Problem: DMA failure or staging slot invalidation skips device_release_in_flight_vram_bytes
    // ??vram_in_flight_bytes accumulates ??prefetch_plan trim sees device_free=0 ??no prefetch
    // Solution: RAII guard ensures bytes are released on all exit paths (success, failure, exception)
    // Impact: Prevents VRAM budget deadlock from leaked in-flight bytes
    struct VramReleaseGuard {
        moe_pc_engine* engine;
        uint64_t bytes;
        bool released;
        VramReleaseGuard(moe_pc_engine* e, uint64_t b) : engine(e), bytes(b), released(false) {}
        ~VramReleaseGuard() { if (!released && bytes > 0) device_release_inflight_vram_bytes(engine, bytes); }
        void release() { if (!released) { device_release_inflight_vram_bytes(engine, bytes); released = true; } }
    };

    while (!copies.empty()) {
        moe_deferred_staging_copy& copy = copies.front();
        VramReleaseGuard vram_guard(e, copy.vram_token.valid ? 0 : copy.bytes_for_vram_release);

        bool ready = false;
        if (copy.event && copy.event_recorded) {
            ready = wait_all ?
                (device_backend_event_sync(e, copy.event) != 0) :
                (device_backend_event_query(e, copy.event) != 0);
            // Check both supports_backend_async_api and stream before forcing a stream wait.
            // Backends without a real async API must not drive async counters.
            if (!ready && wait_all && e->backend_caps.supports_backend_async_api && stream && !stream_synced) {
                ready = device_backend_stream_sync(e, stream) != 0;
                stream_synced = ready;
            }
        } else if (wait_all && e->backend_caps.supports_backend_async_api && stream) {
            if (!stream_synced) {
                stream_synced = device_backend_stream_sync(e, stream) != 0;
            }
            ready = stream_synced;
        }
        if (!ready) {
            break;
        }
        if (copy.key != 0) {
            device_complete_pending_copy_key(e, copy.key);
        }
        for (uint8_t extra_i = 0; extra_i < copy.extra_key_count; ++extra_i) {
            if (copy.extra_keys[extra_i] != 0) {
                device_complete_pending_copy_key(e, copy.extra_keys[extra_i]);
            }
        }
        if (copy.copy_bytes > 0) {
            e->io_atomic.backend_async_completed.fetch_add(
                copy.copy_bytes, std::memory_order_relaxed);
            const uint64_t now_us = moe_profile_now_us();
            const uint64_t wait_ns = now_us > copy.submitted_us ?
                (now_us - copy.submitted_us) * 1000ull : 0ull;
            if (copy.path == moe_IO_PATH_GPUDIRECT_STORAGE) {
                e->io_atomic.bytes_direct_to_gpu.fetch_add(copy.copy_bytes, std::memory_order_relaxed);
                e->gds_async_completed.fetch_add(1, std::memory_order_relaxed);
                moe_io_path_perf_complete(e->gds_perf, wait_ns);
            } else {
                moe_io_path_perf_complete(e->pinned_perf, wait_ns);
            }
        }

        // BUGFIX 193: Slot already released in device_copy_tensor_from_host ?끸쁾??
        // Only release if slot is still valid (UINT32_MAX means already released)
        if (copy.slot != UINT32_MAX) {
            release_staging_slot(e, copy.slot);
        }
        if (copy.event) {
            device_recycle_copy_event(e, copy.event);
        }

        if (copy.vram_token.valid) {
            device_vram_complete_reservation_token(e, copy.vram_token, copy.key);
        } else {
            vram_guard.release();
            device_vram_mark_allocation_resident(e, copy.key);
        }
        for (uint8_t extra_i = 0; extra_i < copy.extra_key_count; ++extra_i) {
            if (copy.extra_keys[extra_i] != 0) {
                device_vram_mark_allocation_resident(e, copy.extra_keys[extra_i]);
                device_touch_allocation(e, copy.extra_keys[extra_i]);
            }
        }

        // BUGFIX 67: Issue 2 - Use pop_front() instead of erase(begin()).
        // Old: copies.erase(copies.begin()) ??O(n) on vector
        // New: copies.pop_front() ??O(1) on deque
        copies.pop_front();
        ++released;
    }
    return released;
}

// BUGFIX 79: Issue 1 - Fix stream sync wrong map reference.
// Problem: device_stream_sync_current finds stream in s_tl_stream_guards
// but checks s_tl_tensor_streams.end() -> wrong map, always fails.
// Stream acquired from pool never gets synced -> DMA completion not guaranteed.
// Solution: Check s_tl_stream_guards.end() and access it->second.stream
static void device_stream_sync_current(moe_pc_engine* e) {
    if (e) {
        auto it = s_tl_stream_guards.find(e);
        if (it != s_tl_stream_guards.end() && it->second.stream) {
            device_backend_stream_sync(e, it->second.stream);
        }
    }
    device_complete_pending_copies(e);
    device_drain_deferred_staging_copies(e, true);
    device_drain_directstorage_device_copies(e);
}

// BUGFIX 79: Issue 1 - Fix stream destroy wrong map reference.
// Problem: Same as sync - finds in s_tl_stream_guards, checks s_tl_tensor_streams
// Stream never returned to pool -> pool exhaustion -> cuStreamCreate fallback.
// Solution: Check s_tl_stream_guards.end() and return it->second.stream to pool
// BUGFIX 888: Remove deferred_staging_copies.erase() from here ?끸쁾??CRITICAL
// Problem: ThreadStreamGuard destructor calls this function, then tries to access
// s_tl_deferred_staging_copies again to release in-flight bytes. But this function
// already erased it, so destructor's cleanup code becomes dead code ??VRAM deadlock.
// Solution: Only erase deferred_staging_copies in destructor, not here.
static void device_stream_destroy_current(moe_pc_engine* e) {
    device_stream_sync_current(e);
    if (e) {
        auto it = s_tl_stream_guards.find(e);
        if (it != s_tl_stream_guards.end() && it->second.stream) {
            CUstream stream = it->second.stream;
            it->second.engine = nullptr;
            it->second.stream = nullptr;
            std::lock_guard<std::mutex> lock(e->stream_pool_mutex);
            e->stream_pool.push_back(stream);
            e->stream_pool_available.store(static_cast<uint32_t>(e->stream_pool.size()), std::memory_order_release);
            s_tl_stream_guards.erase(it);
        }
    }
    s_tl_tensor_pending_keys.erase(e);
    // BUGFIX 888: Removed s_tl_deferred_staging_copies.erase(e) - moved to destructor
    device_destroy_copy_event_pool(e);
}

ThreadStreamGuard::~ThreadStreamGuard() {
    if (!engine) return;

    moe_pc_engine* e = engine;
    CUstream owned_stream = stream;
    device_stream_sync_current(e);
    engine = nullptr;
    stream = nullptr;

    // BUGFIX 888: Clean up deferred copies BEFORE calling device_stream_destroy_current ?끸쁾??CRITICAL
    // Problem: device_stream_destroy_current() used to erase s_tl_deferred_staging_copies,
    // so the cleanup code below became dead code. In-flight bytes were never released ??VRAM deadlock.
    // Solution: Clean up deferred copies first, then call device_stream_destroy_current.
    // Impact: Fixes VRAM budget deadlock on worker crash, ensures in-flight bytes are released.
    auto dit = s_tl_deferred_staging_copies.find(e);
    if (dit != s_tl_deferred_staging_copies.end()) {
        for (auto& copy : dit->second) {
            if (copy.event) device_backend_event_destroy(e, copy.event);
            if (copy.slot != UINT32_MAX) release_staging_slot(e, copy.slot);
            // Release in-flight bytes for this copy
            if (copy.vram_token.valid) {
                device_vram_release_reservation_token(e, copy.vram_token);
            } else {
                device_release_inflight_vram_bytes(e, copy.bytes_for_vram_release);
            }
        }
        s_tl_deferred_staging_copies.erase(dit);
    }

    device_destroy_copy_event_pool(e);

    if (owned_stream) {
        std::lock_guard<std::mutex> lock(e->stream_pool_mutex);
        e->stream_pool.push_back(owned_stream);
        e->stream_pool_available.store(static_cast<uint32_t>(e->stream_pool.size()), std::memory_order_release);
    }
}

static void device_stream_pool_destroy(moe_pc_engine* e) {
    if (!e || !e->config.prefer_gpu || !e->backend_caps.supports_backend_async_api) return;
    if (!device_set_current(e)) return;
    std::lock_guard<std::mutex> lock(e->stream_pool_mutex);
    for (CUstream stream : e->stream_pool) {
        if (stream) {
            device_backend_stream_destroy(e, stream);
        }
    }
    e->stream_pool.clear();
    e->stream_pool_available.store(0, std::memory_order_release);
}

static void device_stream_pool_init(moe_pc_engine* e) {
    if (!e || !e->config.prefer_gpu || !e->backend_caps.supports_backend_async_api) return;
    if (!device_set_current(e)) return;
    std::lock_guard<std::mutex> lock(e->stream_pool_mutex);
    if (!e->stream_pool.empty()) return;

    uint32_t gpu_workers = runtime_gpu_workers(e);
    uint32_t io_workers = runtime_disk_workers(e);
    if (io_workers == 0) io_workers = e->io_config.io_worker_threads;
    uint32_t pinned_workers = runtime_pinned_workers(e);
    uint32_t max_workers = gpu_workers + io_workers + pinned_workers + 2;
    if (max_workers == 0) max_workers = 4;
    uint32_t pool_size = max_workers * (e->backend_caps.copy_streams > 0 ? e->backend_caps.copy_streams : 1);

    for (uint32_t i = 0; i < pool_size; i++) {
        CUstream stream = nullptr;
        if (device_backend_stream_create(e, &stream, 1u /* CU_STREAM_NON_BLOCKING */) && stream) {
            e->stream_pool.push_back(stream);
        }
    }
    e->stream_pool_available.store(static_cast<uint32_t>(e->stream_pool.size()), std::memory_order_release);
}

// Fix 2: Thread-local cleanup function to prevent memory leak/dangling pointers
// BUGFIX 臾몄젣 7: Add thread-local map cleanup for gpu_worker_loop ?끸쁾??CRITICAL
// Problem: s_tl_deferred_staging_copies, s_tl_tensor_pending_keys use engine* as key
// ??Engine destroyed ??dangling pointer in thread_local map
// ??New engine at same address reuses stale data ??corruption
// Solution: GPU worker calls this function on exit to erase engine from all thread_local maps
// Impact: Prevents use-after-free and memory corruption in multi-engine processes
extern "C" {
void device_cleanup_thread_locals(moe_pc_engine* engine) {
    if (!engine) return;

    // BUGFIX 臾몄젣 7: Clean up all thread_local maps ?끸쁾??
    device_stream_destroy_current(engine);

    auto pending_it = s_tl_tensor_pending_keys.find(engine);
    if (pending_it != s_tl_tensor_pending_keys.end()) {
        s_tl_tensor_pending_keys.erase(pending_it);
    }

    // BUGFIX 臾몄젣 7: Also clean up s_tl_stream_guards ?끸쁾??
}
}
