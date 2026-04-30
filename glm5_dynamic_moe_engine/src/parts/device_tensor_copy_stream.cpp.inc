// Device tensor copy stream management

static thread_local std::unordered_map<glm5_pc_engine*, CUstream> s_tl_tensor_streams;
static thread_local std::unordered_map<glm5_pc_engine*, std::vector<uint64_t>> s_tl_tensor_pending_keys;
// Fix 63: Deferred staging slot release — freed after batch stream sync
static thread_local std::unordered_map<glm5_pc_engine*, std::vector<uint32_t>> s_tl_deferred_staging_slots;
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

static void device_stream_sync_current(glm5_pc_engine* e) {
    if (e) {
        auto it = s_tl_tensor_streams.find(e);
        if (it != s_tl_tensor_streams.end() && it->second) {
            device_backend_stream_sync(e, it->second);
        }
    }
    device_complete_pending_copies(e);
    // Fix 63: Release deferred staging slots after stream sync (GPU done reading)
    auto sit = s_tl_deferred_staging_slots.find(e);
    if (sit != s_tl_deferred_staging_slots.end() && !sit->second.empty()) {
        for (uint32_t slot : sit->second) {
            release_staging_slot(e, slot);
        }
        sit->second.clear();
        // Bug #2: Wake all waiters after batch deferred slot release
        e->staging_slot_cv.notify_all();
    }
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
    s_tl_deferred_staging_slots.erase(e);
}
