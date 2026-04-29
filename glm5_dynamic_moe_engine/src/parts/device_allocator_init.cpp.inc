// Device allocator initialization and shutdown

static std::atomic<uint64_t>* device_pool_counter(glm5_pc_engine* e, uint8_t pool) {
    if (!e) return nullptr;
    if (pool == GLM5_DEVICE_POOL_FIXED) return &e->device_fixed_bytes;
    if (pool == GLM5_DEVICE_POOL_ACTIVATION) return &e->device_activation_bytes;
    if (pool == GLM5_DEVICE_POOL_EXPERT) return &e->device_expert_bytes;
    return &e->device_stream_bytes;
}

static int device_allocator_init(glm5_pc_engine* e) {
    if (!e || e->device_allocator_ready.load() != 0) return e && e->device_allocator_ready.load() > 0;
    if (!e->config.prefer_gpu || e->backend_caps.backend != GLM5_BACKEND_CUDA) return 0;
    std::lock_guard<std::mutex> lock(e->device_mutex);
    if (e->device_allocator_ready.load() != 0) return e->device_allocator_ready.load() > 0;
    if (!cuda_driver_load(&e->cuda) || e->cuda.cuInit(0) != 0 ||
        e->cuda.cuDeviceGet(&e->cuda.device, (int)e->backend_caps.device_id) != 0 ||
        e->cuda.cuDevicePrimaryCtxRetain(&e->cuda.context, e->cuda.device) != 0 ||
        e->cuda.cuCtxSetCurrent(e->cuda.context) != 0) {
        glm5_close_shared_library(e->cuda.library);
        e->cuda = glm5_cuda_driver_api{};
        e->device_allocator_ready.store(-1);
        return 0;
    }
    size_t free_b = 0, total_b = 0;
    if (e->cuda.cuMemGetInfo(&free_b, &total_b) == 0) {
        e->device_free_bytes.store((uint64_t)free_b);
        e->device_total_bytes.store((uint64_t)total_b);
        if (e->backend_caps.device_memory_bytes == 0) e->backend_caps.device_memory_bytes = (uint64_t)free_b;
    }
    e->device_allocator_ready.store(1);
    return 1;
}

static int device_set_current(glm5_pc_engine* e) {
    static thread_local CUcontext tls_ctx = nullptr;
    if (!e || !e->cuda.cuCtxSetCurrent) return 0;
    if (tls_ctx == e->cuda.context) return 1;
    if (e->cuda.cuCtxSetCurrent(e->cuda.context) != 0) return 0;
    tls_ctx = e->cuda.context;
    return 1;
}

static void device_allocator_shutdown(glm5_pc_engine* e) {
    if (!e) return;
    std::vector<std::pair<CUdeviceptr, uint64_t>> deferred_free_ptrs;
    std::vector<uint64_t> keys;
    {
        std::lock_guard<std::mutex> lock(e->device_mutex);
        for (const auto& it : e->device_allocations) keys.push_back(it.first);
        for (uint64_t key : keys) device_free_locked(e, key, &deferred_free_ptrs);
        decltype(e->activation_evict_heap) empty;
        e->activation_evict_heap.swap(empty);
    }
    device_free_deferred_ptrs(e, deferred_free_ptrs);
    if (e->cuda.cuDevicePrimaryCtxRelease && e->cuda.context) e->cuda.cuDevicePrimaryCtxRelease(e->cuda.device);
    glm5_close_shared_library(e->cuda.library);
    e->cuda = glm5_cuda_driver_api{};
    e->device_allocator_ready.store(0);
}
