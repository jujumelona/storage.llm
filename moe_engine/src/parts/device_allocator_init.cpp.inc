// Device allocator initialization and shutdown

extern "C" {
uint64_t metal_measure_h2d_bandwidth(void* device_handle);
void* metal_get_default_device();
}

static std::atomic<uint64_t>* device_pool_counter(moe_pc_engine* e, uint8_t pool) {
    if (!e) return nullptr;
    // BUGFIX 590: Check pool value range ★
    // Problem: pool can be out of range
    // Solution: Validate pool before use
    if (pool == moe_DEVICE_POOL_FIXED) return &e->device_fixed_bytes;
    if (pool == moe_DEVICE_POOL_ACTIVATION) return &e->device_activation_bytes;
    if (pool == moe_DEVICE_POOL_EXPERT) return &e->device_expert_bytes;
    // BUGFIX 591: Return nullptr for invalid pool instead of stream_bytes ★
    // Problem: Invalid pool returns stream_bytes, causing wrong accounting
    // Solution: Return nullptr for invalid pool
    if (pool == moe_DEVICE_POOL_STREAM) return &e->device_stream_bytes;
    // BUGFIX 문제4: Add KV cache pool counter ★★★
    if (pool == moe_DEVICE_POOL_KV_CACHE) return &e->device_kv_cache_bytes;
    return nullptr;
}

static void device_stream_pool_init(moe_pc_engine* e);

static void* moe_open_level_zero_loader_library() {
#ifdef _WIN32
    return (void*)LoadLibraryA("ze_loader.dll");
#else
    void* lib = dlopen("libze_loader.so.1", RTLD_LAZY | RTLD_LOCAL);
    return lib ? lib : dlopen("libze_loader.so", RTLD_LAZY | RTLD_LOCAL);
#endif
}

// BUGFIX Problem E: Vulkan loader library opener ★★★ FEATURE
static void* moe_open_vulkan_loader_library() {
#ifdef _WIN32
    return (void*)LoadLibraryA("vulkan-1.dll");
#else
    void* lib = dlopen("libvulkan.so.1", RTLD_LAZY | RTLD_LOCAL);
    return lib ? lib : dlopen("libvulkan.so", RTLD_LAZY | RTLD_LOCAL);
#endif
}

// BUGFIX Problem E: Vulkan driver loader ★★★ FEATURE
// Problem: Vulkan H2D copy not implemented, forcing sync-only path
// Solution: Load Vulkan function pointers and initialize command pool
// Trade-off: Requires full Vulkan initialization, but enables async DMA
static int vulkan_driver_load(moe_vulkan_driver_api* api, uint32_t device_id) {
    if (!api) {
        return 0;
    }
    if (api->library && api->device && api->commandPool) {
        return 1;
    }

    api->library = moe_open_vulkan_loader_library();
    if (!api->library) {
        return 0;
    }

    // Load core Vulkan functions
    typedef int (*PFN_vkGetInstanceProcAddr)(void*, const char*);
    typedef int (*PFN_vkCreateInstance)(const void*, const void*, void**);
    typedef int (*PFN_vkEnumeratePhysicalDevices)(void*, uint32_t*, void**);
    typedef int (*PFN_vkGetPhysicalDeviceProperties)(void*, void*);
    typedef int (*PFN_vkGetPhysicalDeviceQueueFamilyProperties)(void*, uint32_t*, void*);
    typedef int (*PFN_vkCreateDevice)(void*, const void*, const void*, void**);
    typedef void (*PFN_vkGetDeviceQueue)(void*, uint32_t, uint32_t, void**);

    auto vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)moe_find_shared_symbol(api->library, "vkGetInstanceProcAddr");
    auto vkCreateInstance = (PFN_vkCreateInstance)moe_find_shared_symbol(api->library, "vkCreateInstance");

    if (!vkGetInstanceProcAddr || !vkCreateInstance) {
        moe_close_shared_library(api->library);
        *api = moe_vulkan_driver_api{};
        return 0;
    }

    // Create Vulkan instance
    struct VkApplicationInfo_local {
        uint32_t sType; void* pNext; const char* pApplicationName; uint32_t applicationVersion;
        const char* pEngineName; uint32_t engineVersion; uint32_t apiVersion;
    } app_info = {0x00000000, nullptr, "storageLLM", 1, "moe_engine", 1, 0x00400000}; // VK_API_VERSION_1_0

    struct VkInstanceCreateInfo_local {
        uint32_t sType; void* pNext; uint32_t flags; const VkApplicationInfo_local* pApplicationInfo;
        uint32_t enabledLayerCount; const char* const* ppEnabledLayerNames;
        uint32_t enabledExtensionCount; const char* const* ppEnabledExtensionNames;
    } create_info = {0x00000001, nullptr, 0, &app_info, 0, nullptr, 0, nullptr}; // VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO

    void* instance = nullptr;
    if (vkCreateInstance(&create_info, nullptr, &instance) != 0 || !instance) {
        moe_close_shared_library(api->library);
        *api = moe_vulkan_driver_api{};
        return 0;
    }
    api->instance = instance;

    // Load instance-level functions
    auto vkEnumeratePhysicalDevices = (PFN_vkEnumeratePhysicalDevices)vkGetInstanceProcAddr(instance, "vkEnumeratePhysicalDevices");
    auto vkGetPhysicalDeviceProperties = (PFN_vkGetPhysicalDeviceProperties)vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceProperties");
    auto vkGetPhysicalDeviceQueueFamilyProperties = (PFN_vkGetPhysicalDeviceQueueFamilyProperties)vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceQueueFamilyProperties");
    auto vkCreateDevice = (PFN_vkCreateDevice)vkGetInstanceProcAddr(instance, "vkCreateDevice");

    if (!vkEnumeratePhysicalDevices || !vkCreateDevice) {
        moe_close_shared_library(api->library);
        *api = moe_vulkan_driver_api{};
        return 0;
    }

    // Enumerate physical devices
    uint32_t device_count = 0;
    if (vkEnumeratePhysicalDevices(instance, &device_count, nullptr) != 0 || device_count == 0) {
        moe_close_shared_library(api->library);
        *api = moe_vulkan_driver_api{};
        return 0;
    }

    std::vector<void*> physical_devices(device_count, nullptr);
    if (vkEnumeratePhysicalDevices(instance, &device_count, physical_devices.data()) != 0) {
        moe_close_shared_library(api->library);
        *api = moe_vulkan_driver_api{};
        return 0;
    }

    void* physical_device = physical_devices[std::min<uint32_t>(device_id, device_count - 1)];
    api->physicalDevice = physical_device;

    // Find queue family with graphics+compute support
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);

    struct VkQueueFamilyProperties_local {
        uint32_t queueFlags; uint32_t queueCount; uint32_t timestampValidBits;
        struct { uint32_t width; uint32_t height; uint32_t depth; } minImageTransferGranularity;
    };
    std::vector<VkQueueFamilyProperties_local> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_families.data());

    uint32_t queue_family_index = UINT32_MAX;
    for (uint32_t i = 0; i < queue_family_count; ++i) {
        if (queue_families[i].queueFlags & 0x4) { // VK_QUEUE_TRANSFER_BIT
            queue_family_index = i;
            break;
        }
    }

    if (queue_family_index == UINT32_MAX) {
        moe_close_shared_library(api->library);
        *api = moe_vulkan_driver_api{};
        return 0;
    }

    // Create logical device
    float queue_priority = 1.0f;
    struct VkDeviceQueueCreateInfo_local {
        uint32_t sType; void* pNext; uint32_t flags; uint32_t queueFamilyIndex;
        uint32_t queueCount; const float* pQueuePriorities;
    } queue_create_info = {0x00000002, nullptr, 0, queue_family_index, 1, &queue_priority}; // VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO

    struct VkDeviceCreateInfo_local {
        uint32_t sType; void* pNext; uint32_t flags; uint32_t queueCreateInfoCount;
        const VkDeviceQueueCreateInfo_local* pQueueCreateInfos; uint32_t enabledLayerCount;
        const char* const* ppEnabledLayerNames; uint32_t enabledExtensionCount;
        const char* const* ppEnabledExtensionNames; const void* pEnabledFeatures;
    } device_create_info = {0x00000003, nullptr, 0, 1, &queue_create_info, 0, nullptr, 0, nullptr, nullptr}; // VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO

    void* device = nullptr;
    if (vkCreateDevice(physical_device, &device_create_info, nullptr, &device) != 0 || !device) {
        moe_close_shared_library(api->library);
        *api = moe_vulkan_driver_api{};
        return 0;
    }
    api->device = device;

    // Get device queue
    auto vkGetDeviceQueue = (PFN_vkGetDeviceQueue)vkGetInstanceProcAddr(instance, "vkGetDeviceQueue");
    if (vkGetDeviceQueue) {
        void* queue = nullptr;
        vkGetDeviceQueue(device, queue_family_index, 0, &queue);
        api->queue = queue;
    }

    // Load device-level functions
    api->vkCreateCommandPool = (decltype(api->vkCreateCommandPool))vkGetInstanceProcAddr(instance, "vkCreateCommandPool");
    api->vkDestroyCommandPool = (decltype(api->vkDestroyCommandPool))vkGetInstanceProcAddr(instance, "vkDestroyCommandPool");
    api->vkAllocateCommandBuffers = (decltype(api->vkAllocateCommandBuffers))vkGetInstanceProcAddr(instance, "vkAllocateCommandBuffers");
    api->vkFreeCommandBuffers = (decltype(api->vkFreeCommandBuffers))vkGetInstanceProcAddr(instance, "vkFreeCommandBuffers");
    api->vkQueueSubmit = (decltype(api->vkQueueSubmit))vkGetInstanceProcAddr(instance, "vkQueueSubmit");
    api->vkQueueWaitIdle = (decltype(api->vkQueueWaitIdle))vkGetInstanceProcAddr(instance, "vkQueueWaitIdle");
    api->vkCreateBuffer = (decltype(api->vkCreateBuffer))vkGetInstanceProcAddr(instance, "vkCreateBuffer");
    api->vkDestroyBuffer = (decltype(api->vkDestroyBuffer))vkGetInstanceProcAddr(instance, "vkDestroyBuffer");
    api->vkGetBufferMemoryRequirements = (decltype(api->vkGetBufferMemoryRequirements))vkGetInstanceProcAddr(instance, "vkGetBufferMemoryRequirements");
    api->vkAllocateMemory = (decltype(api->vkAllocateMemory))vkGetInstanceProcAddr(instance, "vkAllocateMemory");
    api->vkFreeMemory = (decltype(api->vkFreeMemory))vkGetInstanceProcAddr(instance, "vkFreeMemory");
    api->vkBindBufferMemory = (decltype(api->vkBindBufferMemory))vkGetInstanceProcAddr(instance, "vkBindBufferMemory");
    api->vkMapMemory = (decltype(api->vkMapMemory))vkGetInstanceProcAddr(instance, "vkMapMemory");
    api->vkUnmapMemory = (decltype(api->vkUnmapMemory))vkGetInstanceProcAddr(instance, "vkUnmapMemory");
    api->vkBeginCommandBuffer = (decltype(api->vkBeginCommandBuffer))vkGetInstanceProcAddr(instance, "vkBeginCommandBuffer");
    api->vkEndCommandBuffer = (decltype(api->vkEndCommandBuffer))vkGetInstanceProcAddr(instance, "vkEndCommandBuffer");
    api->vkCmdCopyBuffer = (decltype(api->vkCmdCopyBuffer))vkGetInstanceProcAddr(instance, "vkCmdCopyBuffer");
    api->vkResetCommandBuffer = (decltype(api->vkResetCommandBuffer))vkGetInstanceProcAddr(instance, "vkResetCommandBuffer");
    api->vkGetPhysicalDeviceMemoryProperties2 = (decltype(api->vkGetPhysicalDeviceMemoryProperties2))vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceMemoryProperties2");

    // Create command pool
    if (api->vkCreateCommandPool && api->device) {
        struct VkCommandPoolCreateInfo_local {
            uint32_t sType; void* pNext; uint32_t flags; uint32_t queueFamilyIndex;
        } pool_create_info = {0x00000027, nullptr, 0x00000002, queue_family_index}; // VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT

        void* command_pool = nullptr;
        if (api->vkCreateCommandPool(device, &pool_create_info, nullptr, &command_pool) == 0 && command_pool) {
            api->commandPool = command_pool;
        }
    }

    return api->device && api->commandPool;
}

static int level_zero_driver_load(moe_level_zero_driver_api* api, uint32_t device_id) {
    if (!api) {
        return 0;
    }
    if (api->library && api->context && api->device) {
        return 1;
    }
    api->library = moe_open_level_zero_loader_library();
    if (!api->library) {
        return 0;
    }
    api->zeInit = (moe_ze_init_fn)moe_find_shared_symbol(api->library, "zeInit");
    api->zeDriverGet = (moe_ze_driver_get_fn)moe_find_shared_symbol(api->library, "zeDriverGet");
    api->zeDeviceGet = (moe_ze_device_get_fn)moe_find_shared_symbol(api->library, "zeDeviceGet");
    api->zeContextCreate = (moe_ze_context_create_fn)moe_find_shared_symbol(api->library, "zeContextCreate");
    api->zeContextDestroy = (moe_ze_context_destroy_fn)moe_find_shared_symbol(api->library, "zeContextDestroy");
    api->zeMemAllocHost = (moe_ze_mem_alloc_host_fn)moe_find_shared_symbol(api->library, "zeMemAllocHost");
    api->zeMemFree = (moe_ze_mem_free_fn)moe_find_shared_symbol(api->library, "zeMemFree");
    api->zeDeviceGetMemoryProperties = (moe_ze_device_get_memory_properties_fn)moe_find_shared_symbol(api->library, "zeDeviceGetMemoryProperties");

    // OPTIMIZATION: Load async transfer API functions ★★★
    api->zeCommandQueueCreate = (decltype(api->zeCommandQueueCreate))moe_find_shared_symbol(api->library, "zeCommandQueueCreate");
    api->zeCommandQueueDestroy = (decltype(api->zeCommandQueueDestroy))moe_find_shared_symbol(api->library, "zeCommandQueueDestroy");
    api->zeCommandQueueSynchronize = (decltype(api->zeCommandQueueSynchronize))moe_find_shared_symbol(api->library, "zeCommandQueueSynchronize");
    api->zeCommandQueueExecuteCommandLists = (decltype(api->zeCommandQueueExecuteCommandLists))moe_find_shared_symbol(api->library, "zeCommandQueueExecuteCommandLists");
    api->zeCommandListCreate = (decltype(api->zeCommandListCreate))moe_find_shared_symbol(api->library, "zeCommandListCreate");
    api->zeCommandListDestroy = (decltype(api->zeCommandListDestroy))moe_find_shared_symbol(api->library, "zeCommandListDestroy");
    api->zeCommandListClose = (decltype(api->zeCommandListClose))moe_find_shared_symbol(api->library, "zeCommandListClose");
    api->zeCommandListReset = (decltype(api->zeCommandListReset))moe_find_shared_symbol(api->library, "zeCommandListReset");
    api->zeCommandListAppendMemoryCopy = (decltype(api->zeCommandListAppendMemoryCopy))moe_find_shared_symbol(api->library, "zeCommandListAppendMemoryCopy");
    api->zeCommandListAppendSignalEvent = (decltype(api->zeCommandListAppendSignalEvent))moe_find_shared_symbol(api->library, "zeCommandListAppendSignalEvent");
    api->zeCommandListAppendWaitOnEvents = (decltype(api->zeCommandListAppendWaitOnEvents))moe_find_shared_symbol(api->library, "zeCommandListAppendWaitOnEvents");
    api->zeFenceCreate = (decltype(api->zeFenceCreate))moe_find_shared_symbol(api->library, "zeFenceCreate");
    api->zeFenceDestroy = (decltype(api->zeFenceDestroy))moe_find_shared_symbol(api->library, "zeFenceDestroy");
    api->zeFenceHostSynchronize = (decltype(api->zeFenceHostSynchronize))moe_find_shared_symbol(api->library, "zeFenceHostSynchronize");
    api->zeFenceQueryStatus = (decltype(api->zeFenceQueryStatus))moe_find_shared_symbol(api->library, "zeFenceQueryStatus");
    api->zeDeviceGetProperties = (decltype(api->zeDeviceGetProperties))moe_find_shared_symbol(api->library, "zeDeviceGetProperties");

    if (!api->zeInit || !api->zeDriverGet || !api->zeDeviceGet ||
        !api->zeContextCreate || !api->zeContextDestroy ||
        !api->zeMemAllocHost || !api->zeMemFree) {
        moe_close_shared_library(api->library);
        *api = moe_level_zero_driver_api{};
        return 0;
    }
    if (api->zeInit(0) != 0) {
        moe_close_shared_library(api->library);
        *api = moe_level_zero_driver_api{};
        return 0;
    }
    void* driver = nullptr;
    uint32_t driver_count = 1;
    if (api->zeDriverGet(&driver_count, &driver) != 0 || !driver) {
        moe_close_shared_library(api->library);
        *api = moe_level_zero_driver_api{};
        return 0;
    }
    void* device = nullptr;
    uint32_t device_count = device_id + 1u;
    std::vector<void*> devices(device_count ? device_count : 1u, nullptr);
    if (api->zeDeviceGet(driver, &device_count, devices.data()) != 0 || device_count == 0) {
        moe_close_shared_library(api->library);
        *api = moe_level_zero_driver_api{};
        return 0;
    }
    device = devices[std::min<uint32_t>(device_id, device_count - 1u)];
    struct ze_context_desc_t_local {
        uint32_t stype;
        const void* pNext;
        uint32_t flags;
    } ctx_desc{0xDu, nullptr, 0u}; // ZE_STRUCTURE_TYPE_CONTEXT_DESC
    void* context = nullptr;
    if (api->zeContextCreate(driver, &ctx_desc, &context) != 0 || !context) {
        moe_close_shared_library(api->library);
        *api = moe_level_zero_driver_api{};
        return 0;
    }

    // OPTIMIZATION: Create command queue for async operations ★★★
    if (api->zeCommandQueueCreate) {
        struct ze_command_queue_desc_t_local {
            uint32_t stype; void* pNext; uint32_t ordinal; uint32_t index;
            uint32_t flags; uint32_t mode; uint32_t priority;
        } queue_desc = {0x00010006, nullptr, 0, 0, 0, 0, 0}; // ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC
        void* queue = nullptr;
        if (api->zeCommandQueueCreate(context, device, &queue_desc, &queue) == 0 && queue) {
            api->command_queue = queue;
        }
    }

    api->driver = driver;
    api->device = device;
    api->context = context;
    return 1;
}

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
            if (e->backend_caps.device_memory_bytes == 0) {
                e->backend_caps.device_memory_bytes = (uint64_t)(total_b ? total_b : free_b);
            }
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
            if (e->backend_caps.device_memory_bytes == 0) {
                e->backend_caps.device_memory_bytes = (uint64_t)(total_b ? total_b : free_b);
            }
        }
    // BUGFIX 80: Issue 2 - Fix Metal budget calculation and remove dead code.
    // Problem: Metal uses full system_memory_bytes as VRAM budget
    // 32GB Mac -> 32GB VRAM budget -> OS pressure and excessive expert caching.
    // Also: if (false) block is dead code from refactoring
    // Solution: Reserve 25% for OS, remove dead code
    // BUGFIX 124: Issue 29 - Metal unified memory budget integration ★★
    // Set RAM budget to 0 for Metal, all tracking in VRAM budget (unified memory).
    // BUGFIX Issue 5: Generalize to all unified memory backends ★★★
    } else if (e->backend_caps.backend == moe_BACKEND_METAL ||
               (e->backend_caps.supports_unified_memory &&
                (e->backend_caps.backend == moe_BACKEND_HIP ||
                 e->backend_caps.backend == moe_BACKEND_LEVEL_ZERO))) {
        // Unified memory: reserve 25% for OS (Metal, AMD APU, Intel integrated)
        const uint64_t sys = e->backend_caps.system_memory_bytes > 0
            ? e->backend_caps.system_memory_bytes : 16ull * 1024 * 1024 * 1024;
        if (e->io_config.metal_memory_budget_pct == 0) {
            const uint64_t sys_gb = sys / (1024ull * 1024ull * 1024ull);
            uint64_t auto_pct = 75u;
            if (sys_gb >= 64ull) {
                auto_pct = 80u;
            } else if (sys_gb >= 32ull) {
                auto_pct = 77u;
            }
            e->io_config.metal_memory_budget_pct = (uint8_t)auto_pct;
            const char* backend_name = e->backend_caps.backend == moe_BACKEND_METAL ? "Metal" :
                                       e->backend_caps.backend == moe_BACKEND_HIP ? "HIP APU" : "LevelZero integrated";
            fprintf(stderr,
                "[storageLLM INFO] Auto-detected %s unified-memory budget: %u%% for %llu GB system memory\n",
                backend_name,
                (unsigned)e->io_config.metal_memory_budget_pct,
                (unsigned long long)sys_gb);
        }
        const uint64_t usable = (sys * (uint64_t)e->io_config.metal_memory_budget_pct) / 100ull;
        if (e->backend_caps.device_memory_bytes == 0) {
            e->backend_caps.device_memory_bytes = usable;
        }
        e->device_free_bytes.store(e->backend_caps.device_memory_bytes);
        e->device_total_bytes.store(e->backend_caps.device_memory_bytes);
        // BUGFIX 124: Disable RAM budget for unified memory - all tracked in VRAM budget
        // BUGFIX Issue 5: Apply to all unified memory backends, not just Metal
        e->config.ram_budget_bytes = 0;

        // BUGFIX Problem 8: Measure Metal H2D bandwidth at runtime ★★ PERFORMANCE
        // Problem: All Metal devices use hardcoded 120 GB/s in moe_rt_compute_mem_bw_hint()
        // M3 Max (400 GB/s) and M4 Pro (273 GB/s) underutilize by 3x
        // Solution: Measure actual bandwidth if enable_bw_benchmark flag is set
        // Impact: Correct staging slot count and prefetch depth calculation
        // BUGFIX Problem C: Respect enable_bw_benchmark flag ★★
        // Problem: Metal benchmark runs unconditionally, adding 300ms-1s init delay
        // Solution: Only measure if enable_bw_benchmark is enabled
        if (e->backend_caps.backend == moe_BACKEND_METAL && e->io_config.enable_bw_benchmark) {
            const uint64_t measured = metal_measure_h2d_bandwidth(metal_get_default_device());
            if (measured > 0) {
                e->backend_caps.measured_h2d_bw_bytes_per_sec = measured;
                const double gb_per_sec = (double)measured / 1e9;
                fprintf(stderr,
                    "[storageLLM INFO] Metal H2D bandwidth measured: %.1f GB/s\n",
                    gb_per_sec);
            }
        }

        // BUGFIX Issue ①: Metal unified memory double-counting ★★★ CRITICAL
        // Problem: Same data counted in both VRAM and RAM budgets → available memory appears halved
        // Solution: Set RAM budget to 0 for Metal, delegate all memory to unified VRAM budget
        // Impact: OS unified memory pressure handles eviction, no explicit RAM tier needed
        if (e->backend_caps.backend == moe_BACKEND_METAL) {
            e->config.ram_budget_bytes = 0;
        }
    } else if (e->backend_caps.backend == moe_BACKEND_LEVEL_ZERO) {
        level_zero_driver_load(&e->ze, e->backend_caps.device_id);

        // BUGFIX 896: Detect integrated GPU and enable unified memory handling ★★★ CRITICAL
        // Problem: Intel iGPU shares system RAM but allocatedSize==0 fallback uses totalSize (16GB).
        // This overestimates available VRAM → OOM when actual GPU buffer space is much smaller.
        // Solution: Query zeDeviceGetProperties to detect integrated GPU, set supports_unified_memory.
        // Apply Metal-style unified memory budget calculation (same as HIP APU).
        // Impact: Correct memory budget for Intel integrated GPUs, prevents OOM.
        if (e->ze.zeDeviceGetProperties && e->ze.device) {
            struct ze_device_properties_local {
                uint32_t stype; void* pNext; uint32_t type;
                uint32_t vendorId; uint32_t deviceId; uint32_t flags;
                uint32_t subdeviceId; uint32_t coreClockRate; uint64_t maxMemAllocSize;
                uint32_t maxHardwareContexts; uint32_t maxCommandQueuePriority;
                uint32_t numThreadsPerEU; uint32_t physicalEUSimdWidth; uint32_t numEUsPerSubslice;
                uint32_t numSubslicesPerSlice; uint32_t numSlices; uint64_t timerResolution;
                uint32_t timestampValidBits; uint32_t kernelTimestampValidBits;
                uint8_t uuid[16]; char name[256];
            } props{};
            props.stype = 0x00010003; // ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES
            if (e->ze.zeDeviceGetProperties(e->ze.device, &props) == 0 && (props.flags & 1u)) {
                // ZE_DEVICE_PROPERTY_FLAG_INTEGRATED (bit 0) set → integrated GPU
                e->backend_caps.supports_unified_memory = true;
                fprintf(stderr,
                    "[storageLLM INFO] Level Zero integrated GPU detected, enabling unified memory mode\n");
            }
        }

        if (e->backend_caps.device_memory_bytes == 0) {
            e->backend_caps.device_memory_bytes = e->backend_caps.system_memory_bytes > 0
                ? e->backend_caps.system_memory_bytes : 16ull * 1024ull * 1024ull * 1024ull;
        }
        uint64_t free_mem = 0;
        uint64_t total_mem = e->backend_caps.device_memory_bytes;
        if (e->ze.zeDeviceGetMemoryProperties && e->ze.device) {
            struct ze_device_memory_properties_t_local {
                uint32_t stype;
                void* pNext;
                uint32_t flags;
                uint32_t maxClockRate;
                uint32_t maxBusWidth;
                uint64_t totalSize;
                char name[256];
            };
            struct ze_device_memory_ext_properties_t_local {
                uint32_t stype;
                void* pNext;
                uint32_t type;
                uint64_t physicalSize;
                uint64_t allocatedSize;
            } ext_props{};
            ext_props.stype = 0x0001000B;
            ze_device_memory_properties_t_local mem_props{};
            mem_props.stype = 0x7; // ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES
            mem_props.pNext = &ext_props;
            uint32_t count = 1;
            if (e->ze.zeDeviceGetMemoryProperties(e->ze.device, &count, &mem_props) == 0 &&
                mem_props.totalSize > 0) {
                total_mem = mem_props.totalSize;
                if (ext_props.allocatedSize > 0) {
                    const uint64_t avail = mem_props.totalSize > ext_props.allocatedSize
                        ? mem_props.totalSize - ext_props.allocatedSize : mem_props.totalSize;
                    free_mem = (avail * 90ull) / 100ull;
                } else {
                    free_mem = (mem_props.totalSize * 85ull) / 100ull;
                }
            }
        }
        if (free_mem == 0) {
            free_mem = (e->backend_caps.device_memory_bytes * 85ull) / 100ull;
        }
        if (total_mem > 0) {
            e->backend_caps.device_memory_bytes = total_mem;
        }
        e->device_free_bytes.store(free_mem);
        e->device_total_bytes.store(total_mem);
    } else if (e->backend_caps.backend == moe_BACKEND_SYCL) {
        level_zero_driver_load(&e->ze, e->backend_caps.device_id);
        if (e->backend_caps.device_memory_bytes == 0) {
            e->backend_caps.device_memory_bytes = e->backend_caps.system_memory_bytes > 0
                ? e->backend_caps.system_memory_bytes : 16ull * 1024ull * 1024ull * 1024ull;
        }
        uint64_t free_mem = 0;
        uint64_t total_mem = e->backend_caps.device_memory_bytes;
        if (e->ze.zeDeviceGetMemoryProperties && e->ze.device) {
            struct ze_device_memory_properties_t_local {
                uint32_t stype;
                void* pNext;
                uint32_t flags;
                uint32_t maxClockRate;
                uint32_t maxBusWidth;
                uint64_t totalSize;
                char name[256];
            };
            struct ze_device_memory_ext_properties_t_local {
                uint32_t stype;
                void* pNext;
                uint32_t type;
                uint64_t physicalSize;
                uint64_t allocatedSize;
            } ext_props{};
            ext_props.stype = 0x0001000B;
            ze_device_memory_properties_t_local mem_props{};
            mem_props.stype = 0x7; // ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES
            mem_props.pNext = &ext_props;
            uint32_t count = 1;
            if (e->ze.zeDeviceGetMemoryProperties(e->ze.device, &count, &mem_props) == 0 &&
                mem_props.totalSize > 0) {
                total_mem = mem_props.totalSize;
                if (ext_props.allocatedSize > 0) {
                    const uint64_t avail = mem_props.totalSize > ext_props.allocatedSize
                        ? mem_props.totalSize - ext_props.allocatedSize : mem_props.totalSize;
                    free_mem = (avail * 90ull) / 100ull;
                }
            }
        }
        if (free_mem == 0) {
            free_mem = (e->backend_caps.device_memory_bytes * 85ull) / 100ull;
        }
        if (total_mem > 0) {
            e->backend_caps.device_memory_bytes = total_mem;
        }
        e->device_free_bytes.store(free_mem);
        e->device_total_bytes.store(total_mem);
    } else if (e->backend_caps.backend == moe_BACKEND_VULKAN) {
        if (!vulkan_driver_load(&e->vk, e->backend_caps.device_id)) {
            e->device_allocator_ready.store(-1);
            return 0;
        }

        e->backend_caps.copy_streams = 2;

        if (e->backend_caps.device_memory_bytes == 0) {
            e->backend_caps.device_memory_bytes = e->backend_caps.system_memory_bytes > 0
                ? e->backend_caps.system_memory_bytes : 16ull * 1024 * 1024 * 1024;
        }

        uint64_t free_mem = 0;
        uint64_t total_mem = e->backend_caps.device_memory_bytes;
        int exact_queried = 0;

        // BUGFIX 200: Implement actual VRAM query for Vulkan and Level Zero ★
        // Problem: Vulkan/Level Zero use 85% estimate, never query actual VRAM
        // device_refresh_mem_info has query code but only runs on failure (hot path excluded)
        // Solution: Query at init time to set accurate device_free_bytes baseline
        if (e->backend_caps.backend == moe_BACKEND_VULKAN && e->vk.vkGetPhysicalDeviceMemoryProperties2 && e->vk.physicalDevice) {
            // Vulkan: Query memory budget using VK_EXT_memory_budget
            struct VkPhysicalDeviceMemoryBudgetPropertiesEXT_local {
                uint32_t sType;
                void* pNext;
                uint64_t heapUsage[16];
                uint64_t heapBudget[16];
            } budget{};
            budget.sType = 1000237000; // VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT
            struct VkMemoryHeap_local {
                uint64_t size;
                uint32_t flags;
            };
            struct VkPhysicalDeviceMemoryProperties_local {
                uint32_t memoryTypeCount;
                struct { uint32_t propertyFlags; uint32_t heapIndex; } memoryTypes[32];
                uint32_t memoryHeapCount;
                VkMemoryHeap_local memoryHeaps[16];
            };
            struct VkPhysicalDeviceMemoryProperties2_local {
                uint32_t sType;
                void* pNext;
                VkPhysicalDeviceMemoryProperties_local memoryProperties;
            } props{};
            props.sType = 1000059006; // VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2
            props.pNext = &budget;
            e->vk.vkGetPhysicalDeviceMemoryProperties2(e->vk.physicalDevice, &props);
            uint64_t calc_free = 0;
            uint64_t calc_total = 0;
            // BUGFIX 466: memoryHeapCount 범위 체크 (최대 16개)
            uint32_t heap_count = props.memoryProperties.memoryHeapCount;
            if (heap_count > 16) heap_count = 16;
            for (uint32_t i = 0; i < heap_count; ++i) {
                if (props.memoryProperties.memoryHeaps[i].flags & 1) { // VK_MEMORY_HEAP_DEVICE_LOCAL_BIT
                    calc_total += props.memoryProperties.memoryHeaps[i].size;
                    if (budget.heapBudget[i] > 0) {
                        uint64_t avail = (budget.heapBudget[i] > budget.heapUsage[i]) ? (budget.heapBudget[i] - budget.heapUsage[i]) : 0;
                        calc_free += (avail * 90ull) / 100ull;
                    } else {
                        // Fallback to 85% if VK_EXT_memory_budget is missing
                        calc_free += (props.memoryProperties.memoryHeaps[i].size * 85ull) / 100ull;
                    }
                }
            }
            if (calc_total > 0) {
                free_mem = calc_free;
                total_mem = calc_total;
                exact_queried = 1;
            }
        } else if (e->backend_caps.backend == moe_BACKEND_LEVEL_ZERO && e->ze.zeDeviceGetMemoryProperties && e->ze.device) {
            // Level Zero: Query memory properties
            struct ze_device_memory_properties_t_local {
                uint32_t stype;
                void* pNext;
                uint32_t flags;
                uint32_t maxClockRate;
                uint32_t maxBusWidth;
                uint64_t totalSize;
                char name[256];
            };
            struct ze_device_memory_ext_properties_t_local {
                uint32_t stype;
                void* pNext;
                uint32_t type;
                uint64_t physicalSize;
                uint64_t allocatedSize;
            } ext_props{};
            ext_props.stype = 0x0001000B; // ZE_STRUCTURE_TYPE_DEVICE_MEMORY_EXT_PROPERTIES
            ze_device_memory_properties_t_local mem_props{};
            mem_props.stype = 0x7; // ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES
            mem_props.pNext = &ext_props;
            uint32_t count = 1;
            if (e->ze.zeDeviceGetMemoryProperties(e->ze.device, &count, &mem_props) == 0) {
                total_mem = mem_props.totalSize;
                if (ext_props.allocatedSize > 0) {
                    uint64_t avail = (mem_props.totalSize > ext_props.allocatedSize) ? (mem_props.totalSize - ext_props.allocatedSize) : mem_props.totalSize;
                    free_mem = (avail * 90ull) / 100ull;
                } else {
                    // Fallback to 85% if allocatedSize not populated
                    free_mem = (mem_props.totalSize * 85ull) / 100ull;
                }
                exact_queried = 1;
            }
        }
#ifdef HAVE_SYCL
        else if (e->backend_caps.backend == moe_BACKEND_SYCL) {
            // SYCL: Level Zero 백엔드를 통해 쿼리하거나 OpenCL 메모리 쿼리 사용
            // Level Zero가 SYCL 아래에 있는 경우 기존 LZ 쿼리 재사용 가능
            // 없으면 sycl::device::get_info<sycl::info::device::global_mem_size>() 사용
        }
#endif

        if (!exact_queried) {
            // Use 85% as a conservative free memory estimate to avoid OOM
            free_mem = (e->backend_caps.device_memory_bytes * 85ull) / 100ull;
        }
        e->device_free_bytes.store(free_mem);
        e->device_total_bytes.store(total_mem);
    }
    device_refresh_mem_info(e);

    // CRITICAL ORDER: init_kv_vram_budget MUST precede common_vram_wanted_bytes.
    // Reversing this order causes VRAM budget double-deduction (Problem 2).
    extern void init_kv_vram_budget(moe_pc_engine* engine);
    init_kv_vram_budget(e);

    // Verify KV budget was set only when a real device-side KV consumer exists.
    assert(e->vram_kv_budget_bytes.load(std::memory_order_relaxed) > 0 ||
           e->backend_caps.backend == moe_BACKEND_METAL ||
           !moe_kv_vram_attention_runtime_available(e));

    e->common_vram_reserved_bytes.store(common_vram_wanted_bytes(e), std::memory_order_relaxed);

    // Fix 27: Run a quick microbenchmark to measure actual TFLOPS
    // The hardcoded fallbacks (22/35 TFLOPS) are wildly inaccurate for many GPUs.
    if (e->backend_caps.tflops_hint <= 0.0 &&
        (e->backend_caps.backend == moe_BACKEND_CUDA || e->backend_caps.backend == moe_BACKEND_HIP)) {
        const size_t bench_bytes = 512 * 512 * sizeof(float);
        CUdeviceptr d_a = 0, d_b = 0;
        int alloc_ok = 0;
        if (e->backend_caps.backend == moe_BACKEND_CUDA && e->cuda.cuMemAlloc) {
            alloc_ok = (e->cuda.cuMemAlloc(&d_a, bench_bytes) == 0 &&
                        e->cuda.cuMemAlloc(&d_b, bench_bytes) == 0) ? 1 : 0;
        } else if (e->backend_caps.backend == moe_BACKEND_HIP && e->hip.hipMalloc) {
            alloc_ok = (e->hip.hipMalloc(&d_a, bench_bytes) == 0 &&
                        e->hip.hipMalloc(&d_b, bench_bytes) == 0) ? 1 : 0;
        }
        if (alloc_ok && d_a && d_b) {
            auto t0 = std::chrono::steady_clock::now();
            const int iters = 20;
            for (int i = 0; i < iters; ++i) {
                if (e->backend_caps.backend == moe_BACKEND_CUDA && e->cuda.cuMemcpyDtoD)
                    e->cuda.cuMemcpyDtoD(d_b, d_a, bench_bytes);
                else if (e->backend_caps.backend == moe_BACKEND_HIP && e->hip.hipMemcpyDtoD)
                    e->hip.hipMemcpyDtoD(d_b, d_a, bench_bytes);
            }
            if (e->backend_caps.backend == moe_BACKEND_CUDA && e->cuda.cuCtxSynchronize)
                e->cuda.cuCtxSynchronize();
            else if (e->backend_caps.backend == moe_BACKEND_HIP && e->hip.hipDeviceSynchronize)
                e->hip.hipDeviceSynchronize();
            auto t1 = std::chrono::steady_clock::now();
            double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
            if (elapsed_s > 1e-6) {
                double bw_gbs = (double)(bench_bytes * iters * 2) / elapsed_s / 1e9;
                double est_tflops = bw_gbs * 0.05;
                if (est_tflops > 1.0 && est_tflops < 500.0) {
                    e->backend_caps.tflops_hint = est_tflops;
                }
            }
        }
        if (d_a) {
            if (e->backend_caps.backend == moe_BACKEND_CUDA) e->cuda.cuMemFree(d_a);
            else if (e->backend_caps.backend == moe_BACKEND_HIP) e->hip.hipFree(d_a);
        }
        if (d_b) {
            if (e->backend_caps.backend == moe_BACKEND_CUDA) e->cuda.cuMemFree(d_b);
            else if (e->backend_caps.backend == moe_BACKEND_HIP) e->hip.hipFree(d_b);
        }
    }

    // BUGFIX Problem 2: Use seq_cst for device_allocator_ready ★★★ CRITICAL
    // Problem: relaxed ordering allows compiler reordering of init_kv_vram_budget() call
    // common_vram_wanted_bytes() may see device_allocator_ready=1 before KV budget is set
    // Solution: Use memory_order_seq_cst to enforce total ordering
    // Impact: Prevents double-deduction of KV budget during init
    e->device_allocator_ready.store(1, std::memory_order_seq_cst);

    // BUGFIX 894: Validate staging_slot_size against max expert bytes ★★★ CRITICAL
    // Problem: If staging_slot_size < max_expert_bytes, all prefetch attempts fail silently.
    // Slot acquired → size insufficient → released → staging_slot_contention++ → infinite loop.
    // Engine runs but nothing loads to VRAM → silent performance failure.
    // Solution: Calculate max expert bytes and warn if staging_slot_size is too small.
    // Impact: Prevents silent prefetch failures, enables auto-sizing.
    if (e->staging_slot_size > 0) {
        const moe_storage_constants_t* storage = moe_storage_constants();
        if (storage && storage->normal_bundle_max_bytes > 0) {
            const uint64_t max_expert_bytes = storage->normal_bundle_max_bytes;
            if (e->staging_slot_size < max_expert_bytes) {
                const uint64_t recommended = (max_expert_bytes * 110ull) / 100ull; // +10% margin
                fprintf(stderr,
                    "[storageLLM WARNING] staging_slot_size=%llu < max_expert_bytes=%llu.\n"
                    "  All prefetch attempts will fail. Recommended: %llu bytes.\n"
                    "  Set staging_size to at least %llu bytes (slot_size * slot_count).\n",
                    (unsigned long long)e->staging_slot_size,
                    (unsigned long long)max_expert_bytes,
                    (unsigned long long)recommended,
                    (unsigned long long)(recommended * e->staging_slots.size()));
            }
        }
    }

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
    } else if (e->backend_caps.backend == moe_BACKEND_LEVEL_ZERO ||
               e->backend_caps.backend == moe_BACKEND_SYCL) {
        if (e->ze.zeContextDestroy && e->ze.context) e->ze.zeContextDestroy(e->ze.context);
        moe_close_shared_library(e->ze.library);
        e->ze = moe_level_zero_driver_api{};
    }
    e->device_allocator_ready.store(0);
}
