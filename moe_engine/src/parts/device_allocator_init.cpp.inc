// Device allocator initialization and shutdown

static std::atomic<uint64_t>* device_pool_counter(moe_pc_engine* e, uint8_t pool) {
    if (!e) return nullptr;
    if (pool == moe_DEVICE_POOL_FIXED) return &e->device_fixed_bytes;
    if (pool == moe_DEVICE_POOL_ACTIVATION) return &e->device_activation_bytes;
    if (pool == moe_DEVICE_POOL_EXPERT) return &e->device_expert_bytes;
    return &e->device_stream_bytes;
}

static void device_stream_pool_init(moe_pc_engine* e);

static int device_allocator_init(moe_pc_engine* e) {
    if (!e || e->device_allocator_ready.load() != 0) return e && e->device_allocator_ready.load() > 0;
    if (!e->config.prefer_gpu) return 0;
    if (e->backend_caps.backend != moe_BACKEND_CUDA && e->backend_caps.backend != moe_BACKEND_HIP && e->backend_caps.backend != moe_BACKEND_METAL && e->backend_caps.backend != moe_BACKEND_VULKAN && e->backend_caps.backend != moe_BACKEND_LEVEL_ZERO && e->backend_caps.backend != moe_BACKEND_SYCL) return 0;
    std::lock_guard<std::mutex> lock(e->device_mutex);
    if (e->device_allocator_ready.load() != 0) return e->device_allocator_ready.load() > 0;

    // QKV State (~0.69 GiB) is now tracked via add_tier_usage dynamically,
    // so we no longer pre-reserve it here to avoid double-counting.

    if (e->backend_caps.backend == moe_BACKEND_CUDA) {
        if (!cuda_driver_load(&e->cuda) || e->cuda.cuInit(0) != 0 ||
            e->cuda.cuDeviceGet(&e->cuda.device, (int)e->backend_caps.device_id) != 0 ||
            e->cuda.cuDevicePrimaryCtxRetain(&e->cuda.context, e->cuda.device) != 0 ||
            e->cuda.cuCtxSetCurrent(e->cuda.context) != 0) {
            moe_close_shared_library(e->cuda.library);
            e->cuda = moe_cuda_driver_api{};
            e->device_allocator_ready.store(-1);
            return 0;
        }
        size_t free_b = 0, total_b = 0;
        if (e->cuda.cuMemGetInfo(&free_b, &total_b) == 0) {
            e->device_free_bytes.store((uint64_t)free_b);
            e->device_total_bytes.store((uint64_t)total_b);
            if (e->backend_caps.device_memory_bytes == 0) e->backend_caps.device_memory_bytes = (uint64_t)free_b;
        }
    } else if (e->backend_caps.backend == moe_BACKEND_HIP) {
        if (!hip_driver_load(&e->hip) || e->hip.hipInit(0) != 0 ||
            e->hip.hipDeviceGet(&e->hip.device, (int)e->backend_caps.device_id) != 0 ||
            e->hip.hipDevicePrimaryCtxRetain(&e->hip.context, e->hip.device) != 0 ||
            e->hip.hipCtxSetCurrent(e->hip.context) != 0) {
            moe_close_shared_library(e->hip.library);
            e->hip = moe_hip_driver_api{};
            e->device_allocator_ready.store(-1);
            return 0;
        }
        size_t free_b = 0, total_b = 0;
        if (e->hip.hipMemGetInfo(&free_b, &total_b) == 0) {
            e->device_free_bytes.store((uint64_t)free_b);
            e->device_total_bytes.store((uint64_t)total_b);
            if (e->backend_caps.device_memory_bytes == 0) e->backend_caps.device_memory_bytes = (uint64_t)free_b;
        }
    // BUGFIX 80: Issue 2 - Fix Metal budget calculation and remove dead code.
    // Problem: Metal uses full system_memory_bytes as VRAM budget
    // 32GB Mac -> 32GB VRAM budget -> OS pressure and excessive expert caching.
    // Also: if (false) block is dead code from refactoring
    // Solution: Reserve 25% for OS, remove dead code
    } else if (e->backend_caps.backend == moe_BACKEND_METAL) {
        // Apple Silicon unified memory: reserve 25% for OS
        const uint64_t sys = e->backend_caps.system_memory_bytes > 0
            ? e->backend_caps.system_memory_bytes : 16ull * 1024 * 1024 * 1024;
        const uint64_t usable = (sys * 3ull) / 4ull;  // 75% usable, 25% OS reserve
        if (e->backend_caps.device_memory_bytes == 0) {
            e->backend_caps.device_memory_bytes = usable;
        }
        e->device_free_bytes.store(e->backend_caps.device_memory_bytes);
        e->device_total_bytes.store(e->backend_caps.device_memory_bytes);
    } else if (e->backend_caps.backend == moe_BACKEND_VULKAN ||
               e->backend_caps.backend == moe_BACKEND_LEVEL_ZERO ||
               e->backend_caps.backend == moe_BACKEND_SYCL) {
        // Vulkan/Level Zero/SYCL: use reported device memory or fallback
        if (e->backend_caps.device_memory_bytes == 0) {
            e->backend_caps.device_memory_bytes = e->backend_caps.system_memory_bytes > 0
                ? e->backend_caps.system_memory_bytes : 16ull * 1024 * 1024 * 1024;
        }

        uint64_t free_mem = 0;
        int exact_queried = 0;

        // BUGFIX 115: Implement actual VRAM query for Vulkan and Level Zero
        if (e->backend_caps.backend == moe_BACKEND_VULKAN && e->vk.vkGetPhysicalDeviceMemoryProperties2 && e->vk.physicalDevice) {
            // Vulkan: Query memory budget using VK_EXT_memory_budget
            // VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{};
            // budget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
            // VkPhysicalDeviceMemoryProperties2 props{};
            // props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
            // props.pNext = &budget;
            // e->vk.vkGetPhysicalDeviceMemoryProperties2(e->vk.physicalDevice, &props);
            // Sum budget.heapBudget[i] for device-local heaps to get free_mem
            // exact_queried = 1;
            // TODO: Requires VK_EXT_memory_budget extension and proper struct definitions
        } else if (e->backend_caps.backend == moe_BACKEND_LEVEL_ZERO && e->ze.zeDeviceGetMemoryProperties && e->ze.device) {
            // Level Zero: Query memory properties
            // ze_device_memory_properties_t mem_props{};
            // mem_props.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
            // uint32_t count = 1;
            // e->ze.zeDeviceGetMemoryProperties(e->ze.device, &count, &mem_props);
            // free_mem = mem_props.totalSize; // May need zeDeviceGetMemoryAccessProperties for free estimate
            // exact_queried = 1;
            // TODO: Requires proper Level Zero API integration
        }
        
        if (!exact_queried) {
            // Use 85% as a conservative free memory estimate to avoid OOM
            free_mem = (e->backend_caps.device_memory_bytes * 85ull) / 100ull;
        }
        e->device_free_bytes.store(free_mem);
        e->device_total_bytes.store(e->backend_caps.device_memory_bytes);
    }
    e->device_allocator_ready.store(1);
    // Stream pool init moved to engine start
    return 1;
}

static int device_set_current(moe_pc_engine* e) {
    static thread_local CUcontext tls_ctx = nullptr;
    if (!e) return 0;
    if (e->backend_caps.backend == moe_BACKEND_CUDA) {
        if (!e->cuda.cuCtxSetCurrent) return 0;
        if (tls_ctx == e->cuda.context) return 1;
        if (e->cuda.cuCtxSetCurrent(e->cuda.context) != 0) return 0;
        tls_ctx = e->cuda.context;
        return 1;
    } else if (e->backend_caps.backend == moe_BACKEND_HIP) {
        if (!e->hip.hipCtxSetCurrent) return 0;
        if (tls_ctx == e->hip.context) return 1;
        if (e->hip.hipCtxSetCurrent(e->hip.context) != 0) return 0;
        tls_ctx = e->hip.context;
        return 1;
    }
    return 0;
}

static void device_allocator_shutdown(moe_pc_engine* e) {
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
    if (e->backend_caps.backend == moe_BACKEND_CUDA) {
        if (e->cuda.cuDevicePrimaryCtxRelease && e->cuda.context) e->cuda.cuDevicePrimaryCtxRelease(e->cuda.device);
        moe_close_shared_library(e->cuda.library);
        e->cuda = moe_cuda_driver_api{};
    } else if (e->backend_caps.backend == moe_BACKEND_HIP) {
        if (e->hip.hipDevicePrimaryCtxRelease && e->hip.context) e->hip.hipDevicePrimaryCtxRelease(e->hip.device);
        moe_close_shared_library(e->hip.library);
        e->hip = moe_hip_driver_api{};
    }
    e->device_allocator_ready.store(0);
}
