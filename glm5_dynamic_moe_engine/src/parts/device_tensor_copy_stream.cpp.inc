// Device tensor copy stream management

struct glm5_deferred_staging_copy {
    uint32_t slot = UINT32_MAX;
    uint64_t key = 0;
    CUevent event = nullptr;
    bool event_recorded = false;
};

static thread_local std::unordered_map<glm5_pc_engine*, CUstream> s_tl_tensor_streams;
static thread_local std::unordered_map<glm5_pc_engine*, std::vector<uint64_t>> s_tl_tensor_pending_keys;
static thread_local std::unordered_map<glm5_pc_engine*, std::vector<glm5_deferred_staging_copy>> s_tl_deferred_staging_copies;
static thread_local std::unordered_map<glm5_pc_engine*, std::vector<CUevent>> s_tl_copy_event_pool;
static constexpr int GLM5_DEVICE_COPY_BACKPRESSURE = -2;

static CUstream device_tensor_stream_current(glm5_pc_engine* e) {
    if (!e ||
        !e->config.prefer_gpu ||
        e->backend_caps.backend != GLM5_BACKEND_CUDA ||
        !e->backend_caps.supports_backend_async_api) {
        return nullptr;
    }
    if (!device_set_current(e)) {
        return nullptr;
    }
    auto it = s_tl_tensor_streams.find(e);
    if (it != s_tl_tensor_streams.end()) {
        return it->second;
    }
    CUstream stream = nullptr;
    // Fix 4: Use CU_STREAM_NON_BLOCKING (=1) to prevent implicit sync with default stream.
    // This allows multiple HtoD copy streams to run truly in parallel.
    if (!device_backend_stream_create(e, &stream, 1u /* CU_STREAM_NON_BLOCKING */)) {
        return nullptr;
    }
    s_tl_tensor_streams[e] = stream;
    return stream;
}

static CUevent device_take_copy_event(glm5_pc_engine* e) {
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

static void device_recycle_copy_event(glm5_pc_engine* e, CUevent event) {
    if (!e || !event) {
        return;
    }
    s_tl_copy_event_pool[e].push_back(event);
}

static void device_destroy_copy_event_pool(glm5_pc_engine* e) {
    auto it = s_tl_copy_event_pool.find(e);
    if (it == s_tl_copy_event_pool.end()) {
        return;
    }
    for (CUevent event : it->second) {
        device_backend_event_destroy(e, event);
    }
    s_tl_copy_event_pool.erase(it);
}

static void device_defer_staging_copy(glm5_pc_engine* e, uint32_t slot, uint64_t key, CUstream stream) {
    if (!e || slot == UINT32_MAX) {
        return;
    }
    glm5_deferred_staging_copy copy{};
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

static size_t device_deferred_staging_copy_count(glm5_pc_engine* e) {
    auto it = s_tl_deferred_staging_copies.find(e);
    return it == s_tl_deferred_staging_copies.end() ? 0u : it->second.size();
}

static size_t device_drain_deferred_staging_copies(glm5_pc_engine* e, bool wait_all) {
    if (!e) {
        return 0;
    }
    auto it = s_tl_deferred_staging_copies.find(e);
    if (it == s_tl_deferred_staging_copies.end() || it->second.empty()) {
        return 0;
    }
    CUstream stream = nullptr;
    auto sit = s_tl_tensor_streams.find(e);
    if (sit != s_tl_tensor_streams.end()) {
        stream = sit->second;
    }
    bool stream_synced = false;
    size_t released = 0;
    auto& copies = it->second;
    while (!copies.empty()) {
        glm5_deferred_staging_copy& copy = copies.front();
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
        copies.erase(copies.begin());
        ++released;
    }
    return released;
}

static void device_stream_sync_current(glm5_pc_engine* e) {
    if (e) {
        auto it = s_tl_tensor_streams.find(e);
        if (it != s_tl_tensor_streams.end() && it->second) {
            device_backend_stream_sync(e, it->second);
        }
    }
    device_complete_pending_copies(e);
    device_drain_deferred_staging_copies(e, true);
}

static void device_stream_destroy_current(glm5_pc_engine* e) {
    device_stream_sync_current(e);
    if (e) {
        auto it = s_tl_tensor_streams.find(e);
        if (it != s_tl_tensor_streams.end() && it->second) {
            device_backend_stream_destroy(e, it->second);
            s_tl_tensor_streams.erase(it);
        }
    }
    s_tl_tensor_pending_keys.erase(e);
    s_tl_deferred_staging_copies.erase(e);
    device_destroy_copy_event_pool(e);
}
