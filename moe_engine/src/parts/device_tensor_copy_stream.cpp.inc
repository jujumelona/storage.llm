// Device tensor copy stream management

// DONE: BUGFIX 77 - CUDA stream pool is already implemented below to avoid cuStreamCreate/Destroy on worker resize.

struct moe_deferred_staging_copy {
    uint32_t slot = UINT32_MAX;
    uint64_t key = 0;
    CUevent event = nullptr;
    bool event_recorded = false;
};

struct ThreadStreamGuard { moe_pc_engine* engine = nullptr; CUstream stream = nullptr; ~ThreadStreamGuard() { if (engine && stream && engine->device_allocator_ready.load(std::memory_order_acquire)) { std::lock_guard<std::mutex> lock(engine->stream_pool_mutex); engine->stream_pool.push_back(stream); engine->stream_pool_available.fetch_add(1, std::memory_order_release); } } }; static thread_local std::unordered_map<moe_pc_engine*, ThreadStreamGuard> s_tl_stream_guards;
static thread_local std::unordered_map<moe_pc_engine*, std::vector<uint64_t>> s_tl_tensor_pending_keys;
// BUGFIX 67: Issue 2 - Change deferred_staging_copies from vector to deque.
// Problem: copies.erase(copies.begin()) is O(n) on vector
// With 8-16 concurrent copies, drain loop becomes O(n²)
// Solution: Use std::deque for O(1) pop_front()
// Impact: Drain performance improves from O(n²) to O(n)
static thread_local std::unordered_map<moe_pc_engine*, std::deque<moe_deferred_staging_copy>> s_tl_deferred_staging_copies;
static thread_local std::unordered_map<moe_pc_engine*, std::vector<CUevent>> s_tl_copy_event_pool;
static constexpr int moe_DEVICE_COPY_BACKPRESSURE = -2;

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

static void device_defer_staging_copy(moe_pc_engine* e, uint32_t slot, uint64_t key, CUstream stream) {
    if (!e || slot == UINT32_MAX) {
        return;
    }
    moe_deferred_staging_copy copy{};
    copy.slot = slot;
    copy.key = key;
    copy.event = device_take_copy_event(e);
    if (copy.event) {
        copy.event_recorded = device_backend_event_record(e, copy.event, stream) ? true : false;
        if (!copy.event_recorded) {
            device_recycle_copy_event(e, copy.event);
            copy.event = nullptr;
        }
    }
    s_tl_deferred_staging_copies[e].push_back(copy);
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
    while (!copies.empty()) {
        moe_deferred_staging_copy& copy = copies.front();
        bool ready = false;
        if (copy.event && copy.event_recorded) {
            ready = wait_all ?
                (device_backend_event_sync(e, copy.event) != 0) :
                (device_backend_event_query(e, copy.event) != 0);
            if (!ready && wait_all && stream && !stream_synced) {
                ready = device_backend_stream_sync(e, stream) != 0;
                stream_synced = ready;
            }
        } else if (wait_all && stream) {
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
        release_staging_slot(e, copy.slot);
        if (copy.event) {
            device_recycle_copy_event(e, copy.event);
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
}

// BUGFIX 79: Issue 1 - Fix stream destroy wrong map reference.
// Problem: Same as sync - finds in s_tl_stream_guards, checks s_tl_tensor_streams
// Stream never returned to pool -> pool exhaustion -> cuStreamCreate fallback.
// Solution: Check s_tl_stream_guards.end() and return it->second.stream to pool
static void device_stream_destroy_current(moe_pc_engine* e) {
    device_stream_sync_current(e);
    if (e) {
        auto it = s_tl_stream_guards.find(e);
        if (it != s_tl_stream_guards.end() && it->second.stream) {
            std::lock_guard<std::mutex> lock(e->stream_pool_mutex);
            e->stream_pool.push_back(it->second.stream);
            e->stream_pool_available.store(static_cast<uint32_t>(e->stream_pool.size()), std::memory_order_release);
            s_tl_stream_guards.erase(it);
        }
    }
    s_tl_tensor_pending_keys.erase(e);
    s_tl_deferred_staging_copies.erase(e);
    device_destroy_copy_event_pool(e);
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

    // Issue 7: Pre-allocate stream pool to avoid cuStreamCreate overhead
    uint32_t gpu_workers = runtime_gpu_workers(e);
    uint32_t io_workers = runtime_disk_workers(e);
    if (io_workers == 0) io_workers = e->io_config.io_worker_threads;
    uint32_t pinned_workers = runtime_pinned_workers(e);
    uint32_t max_workers = gpu_workers + io_workers + pinned_workers + 2; // +2 for main thread and safety margin
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
extern "C" {
void device_cleanup_thread_locals(moe_pc_engine* engine) {
    if (!engine) return;
    
    auto it = s_tl_deferred_staging_copies.find(engine);
    if (it != s_tl_deferred_staging_copies.end()) {
        it->second.clear();
        s_tl_deferred_staging_copies.erase(it);
    }
    
    device_destroy_copy_event_pool(engine);
    
    auto pending_it = s_tl_tensor_pending_keys.find(engine);
    if (pending_it != s_tl_tensor_pending_keys.end()) {
        s_tl_tensor_pending_keys.erase(pending_it);
    }
}
}
