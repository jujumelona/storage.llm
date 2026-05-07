#ifndef CU_TYPES_DEFINED
#define CU_TYPES_DEFINED
typedef uint64_t CUdeviceptr;
typedef void* CUstream;
typedef void* CUcontext;
typedef void* CUevent;
#endif

typedef int (*moe_cu_init_fn)(unsigned int);
typedef int (*moe_cu_device_get_fn)(int*, int);
typedef int (*moe_cu_device_get_attribute_fn)(int*, int, int);
typedef int (*moe_cu_device_get_pci_bus_id_fn)(char*, int, int);
typedef int (*moe_cu_ctx_set_current_fn)(void*);
typedef int (*moe_cu_ctx_synchronize_fn)();
typedef int (*moe_cu_primary_ctx_retain_fn)(void**, int);
typedef int (*moe_cu_primary_ctx_release_fn)(int);
typedef int (*moe_cu_mem_alloc_fn)(uint64_t*, size_t);
typedef int (*moe_cu_mem_free_fn)(uint64_t);
typedef int (*moe_cu_memcpy_htod_fn)(uint64_t, const void*, size_t);
typedef int (*moe_cu_memcpy_htod_async_fn)(uint64_t, const void*, size_t, void*);
typedef int (*moe_cu_memcpy_dtod_fn)(uint64_t, uint64_t, size_t);
typedef int (*moe_cu_memcpy_dtoh_fn)(void*, uint64_t, size_t);
typedef int (*moe_cu_mem_get_info_fn)(size_t*, size_t*);
typedef int (*moe_cu_stream_create_fn)(void**, unsigned int);
typedef int (*moe_cu_stream_synchronize_fn)(void*);
typedef int (*moe_cu_stream_destroy_fn)(void*);
typedef int (*moe_cu_event_create_fn)(void**, unsigned int);
typedef int (*moe_cu_event_record_fn)(void*, void*);
typedef int (*moe_cu_event_query_fn)(void*);
typedef int (*moe_cu_event_synchronize_fn)(void*);
typedef int (*moe_cu_event_destroy_fn)(void*);
struct moe_cufile_error_t {
    int err;
    int cu_err;
};
typedef long long (*moe_cufile_read_fn)(void*, void*, size_t, int64_t, int64_t);
typedef moe_cufile_error_t (*moe_cufile_read_async_fn)(
    void*,
    void*,
    size_t*,
    int64_t*,
    int64_t*,
    void*
);
typedef moe_cufile_error_t (*moe_cufile_stream_register_fn)(void*, unsigned int);
typedef int (*moe_hip_device_get_attribute_fn)(int*, int, int);
typedef int (*moe_ze_init_fn)(uint32_t);
typedef int (*moe_ze_driver_get_fn)(uint32_t*, void**);
typedef int (*moe_ze_device_get_fn)(void*, uint32_t*, void**);
typedef int (*moe_ze_context_create_fn)(void*, const void*, void**);
typedef int (*moe_ze_context_destroy_fn)(void*);
typedef int (*moe_ze_mem_alloc_host_fn)(void*, const void*, size_t, size_t, void**);
typedef int (*moe_ze_mem_alloc_device_fn)(void*, const void*, size_t, size_t, void*, void**);
typedef int (*moe_ze_mem_free_fn)(void*, void*);
typedef int (*moe_ze_device_get_memory_properties_fn)(void*, uint32_t*, void*);

struct moe_cuda_driver_api {
    void* library = nullptr;
    void* context = nullptr;
    int device = 0;
    moe_cu_init_fn cuInit = nullptr;
    moe_cu_device_get_fn cuDeviceGet = nullptr;
    moe_cu_device_get_attribute_fn cuDeviceGetAttribute = nullptr;
    moe_cu_device_get_pci_bus_id_fn cuDeviceGetPCIBusId = nullptr;
    moe_cu_ctx_set_current_fn cuCtxSetCurrent = nullptr;
    moe_cu_ctx_synchronize_fn cuCtxSynchronize = nullptr;
    moe_cu_primary_ctx_retain_fn cuDevicePrimaryCtxRetain = nullptr;
    moe_cu_primary_ctx_release_fn cuDevicePrimaryCtxRelease = nullptr;
    moe_cu_mem_alloc_fn cuMemAlloc = nullptr;
    moe_cu_mem_free_fn cuMemFree = nullptr;
    moe_cu_memcpy_htod_fn cuMemcpyHtoD = nullptr;
    moe_cu_memcpy_htod_async_fn cuMemcpyHtoDAsync = nullptr;
    moe_cu_memcpy_dtod_fn cuMemcpyDtoD = nullptr;
    moe_cu_mem_get_info_fn cuMemGetInfo = nullptr;
    moe_cu_stream_create_fn cuStreamCreate = nullptr;
    moe_cu_stream_synchronize_fn cuStreamSynchronize = nullptr;
    moe_cu_stream_destroy_fn cuStreamDestroy = nullptr;
    moe_cu_event_create_fn cuEventCreate = nullptr;
    moe_cu_event_record_fn cuEventRecord = nullptr;
    moe_cu_event_query_fn cuEventQuery = nullptr;
    moe_cu_event_synchronize_fn cuEventSynchronize = nullptr;
    moe_cu_event_destroy_fn cuEventDestroy = nullptr;
    void* (*cuGraphCreate)(void**, unsigned int) = nullptr;
    void* (*cuGraphDestroy)(void*) = nullptr;
    void* (*cuGraphAddKernelNode)(void**, void*, void**, size_t, void*) = nullptr;
    void* (*cuGraphInstantiate)(void**, void*, void*, char*, size_t) = nullptr;
    void* (*cuGraphLaunch)(void*, void*) = nullptr;
    void* (*cuGraphExecDestroy)(void*) = nullptr;
    void* (*cuStreamBeginCapture)(void*, int) = nullptr;
    void* (*cuStreamEndCapture)(void*, void**) = nullptr;

    // GPUDirect Storage (cuFile) API ??NVMe ??GPU zero-copy DMA
    // Research: GPUDirect Storage (NVIDIA, 2023-25)
    // Eliminates CPU memcpy in the disk ??pinned ??GPU path by reading
    // directly from NVMe into GPU memory via RDMA engine.
    void* cufile_library = nullptr;
    moe_cufile_error_t (*cuFileDriverOpen)() = nullptr;
    void (*cuFileDriverClose)() = nullptr;
    moe_cufile_error_t (*cuFileHandleRegister)(void**, const void*) = nullptr; // (CUfileHandle_t*, CUfileDescr_t*)
    void (*cuFileHandleDeregister)(void*) = nullptr;           // (CUfileHandle_t)
    moe_cufile_error_t (*cuFileBufRegister)(void*, size_t, int) = nullptr; // (devPtr, size, flags)
    moe_cufile_error_t (*cuFileBufDeregister)(void*) = nullptr;            // (devPtr)
    moe_cufile_read_fn cuFileRead = nullptr;  // (handle, devPtr, size, file_off, buf_off)
    moe_cufile_read_async_fn cuFileReadAsync = nullptr;
    moe_cufile_stream_register_fn cuFileStreamRegister = nullptr;
    bool cufile_available = false;

    // cuMemHostRegister ??for io_uring registered buffer integration
    int (*cuMemHostRegister)(void*, size_t, unsigned int) = nullptr;
    int (*cuMemHostUnregister)(void*) = nullptr;
};

struct moe_hip_driver_api {
    void* library = nullptr;
    void* context = nullptr;
    int device = 0;
    moe_cu_init_fn hipInit = nullptr;
    moe_cu_device_get_fn hipDeviceGet = nullptr;
    moe_hip_device_get_attribute_fn hipDeviceGetAttribute = nullptr;
    moe_cu_device_get_pci_bus_id_fn hipDeviceGetPCIBusId = nullptr;
    moe_cu_ctx_set_current_fn hipCtxSetCurrent = nullptr;
    moe_cu_ctx_synchronize_fn hipDeviceSynchronize = nullptr;
    moe_cu_primary_ctx_retain_fn hipDevicePrimaryCtxRetain = nullptr;
    moe_cu_primary_ctx_release_fn hipDevicePrimaryCtxRelease = nullptr;
    moe_cu_mem_alloc_fn hipMalloc = nullptr;
    moe_cu_mem_free_fn hipFree = nullptr;
    moe_cu_memcpy_htod_fn hipMemcpyHtoD = nullptr;
    moe_cu_memcpy_htod_async_fn hipMemcpyHtoDAsync = nullptr;
    moe_cu_memcpy_dtod_fn hipMemcpyDtoD = nullptr;
    moe_cu_memcpy_dtoh_fn hipMemcpyDtoH = nullptr;
    moe_cu_mem_get_info_fn hipMemGetInfo = nullptr;
    moe_cu_stream_create_fn hipStreamCreate = nullptr;
    moe_cu_stream_synchronize_fn hipStreamSynchronize = nullptr;
    moe_cu_stream_destroy_fn hipStreamDestroy = nullptr;
    moe_cu_event_create_fn hipEventCreate = nullptr;
    moe_cu_event_record_fn hipEventRecord = nullptr;
    moe_cu_event_query_fn hipEventQuery = nullptr;
    moe_cu_event_synchronize_fn hipEventSynchronize = nullptr;
    moe_cu_event_destroy_fn hipEventDestroy = nullptr;
    // hipHostMalloc/hipHostFree for IOMMU-registered pinned staging.
    int (*hipHostMalloc)(void**, size_t, unsigned int) = nullptr;
    int (*hipHostFree)(void*) = nullptr;
    int (*hipHostRegister)(void*, size_t, unsigned int) = nullptr;
    int (*hipHostGetDevicePointer)(void**, void*, unsigned int) = nullptr;
    int (*hipHostUnregister)(void*) = nullptr;
};

// Level Zero basic API hooks
struct moe_level_zero_driver_api {
    void* library = nullptr;
    void* driver = nullptr;
    void* context = nullptr;
    void* device = nullptr;
    void* command_queue = nullptr;
    moe_ze_init_fn zeInit = nullptr;
    moe_ze_driver_get_fn zeDriverGet = nullptr;
    moe_ze_device_get_fn zeDeviceGet = nullptr;
    moe_ze_context_create_fn zeContextCreate = nullptr;
    moe_ze_context_destroy_fn zeContextDestroy = nullptr;
    moe_ze_mem_alloc_host_fn zeMemAllocHost = nullptr;
    moe_ze_mem_alloc_device_fn zeMemAllocDevice = nullptr;
    moe_ze_mem_free_fn zeMemFree = nullptr;
    int (*zeCommandQueueCreate)(void*, void*, void*, void**) = nullptr;
    int (*zeCommandQueueDestroy)(void*) = nullptr;
    int (*zeCommandQueueSynchronize)(void*, uint64_t) = nullptr;
    int (*zeCommandQueueExecuteCommandLists)(void*, uint32_t, void**, void*) = nullptr;
    int (*zeCommandListCreate)(void*, void*, void*, void**) = nullptr;
    int (*zeCommandListDestroy)(void*) = nullptr;
    int (*zeCommandListClose)(void*) = nullptr;
    int (*zeCommandListReset)(void*) = nullptr;
    int (*zeCommandListAppendMemoryCopy)(void*, void*, const void*, size_t, void*, uint32_t, void**) = nullptr;
    int (*zeFenceCreate)(void*, void*, void**) = nullptr;
    int (*zeFenceDestroy)(void*) = nullptr;
    int (*zeFenceHostSynchronize)(void*, uint64_t) = nullptr;
    int (*zeFenceQueryStatus)(void*) = nullptr;
    int (*zeFenceReset)(void*) = nullptr;
    int (*zeDeviceGetProperties)(void*, void*) = nullptr;
    moe_ze_device_get_memory_properties_fn zeDeviceGetMemoryProperties = nullptr;
};

// Vulkan basic API hooks
struct moe_vulkan_driver_api {
    void* library = nullptr;
    void* instance = nullptr;
    void* physicalDevice = nullptr;
    void* device = nullptr;
    void* queue = nullptr;
    void* commandPool = nullptr;
    uint32_t queueFamilyIndex = UINT32_MAX;
    void (*vkDestroyInstance)(void*, const void*) = nullptr;
    void (*vkDestroyDevice)(void*, const void*) = nullptr;
    void* (*vkGetDeviceProcAddr)(void*, const char*) = nullptr;
    int (*vkCreateCommandPool)(void*, const void*, const void*, void**) = nullptr;
    void (*vkDestroyCommandPool)(void*, void*, const void*) = nullptr;
    int (*vkAllocateCommandBuffers)(void*, const void*, void**) = nullptr;
    void (*vkFreeCommandBuffers)(void*, void*, uint32_t, const void**) = nullptr;
    int (*vkQueueSubmit)(void*, uint32_t, const void*, void*) = nullptr;
    int (*vkQueueWaitIdle)(void*) = nullptr;
    int (*vkCreateFence)(void*, const void*, const void*, void**) = nullptr;
    int (*vkGetFenceStatus)(void*, void*) = nullptr;
    int (*vkWaitForFences)(void*, uint32_t, const void**, uint32_t, uint64_t) = nullptr;
    int (*vkResetFences)(void*, uint32_t, const void**) = nullptr;
    void (*vkDestroyFence)(void*, void*, const void*) = nullptr;
    void (*vkGetPhysicalDeviceMemoryProperties)(void*, void*) = nullptr;
    void (*vkGetPhysicalDeviceMemoryProperties2)(void*, void*) = nullptr;
    // BUGFIX Problem E: Add functions for H2D copy ?끸쁾??FEATURE
    int (*vkCreateBuffer)(void*, const void*, const void*, void**) = nullptr;
    void (*vkDestroyBuffer)(void*, void*, const void*) = nullptr;
    void (*vkGetBufferMemoryRequirements)(void*, void*, void*) = nullptr;
    int (*vkAllocateMemory)(void*, const void*, const void*, void**) = nullptr;
    void (*vkFreeMemory)(void*, void*, const void*) = nullptr;
    int (*vkBindBufferMemory)(void*, void*, void*, uint64_t) = nullptr;
    int (*vkMapMemory)(void*, void*, uint64_t, uint64_t, uint32_t, void**) = nullptr;
    void (*vkUnmapMemory)(void*, void*) = nullptr;
    int (*vkBeginCommandBuffer)(void*, const void*) = nullptr;
    int (*vkEndCommandBuffer)(void*) = nullptr;
    void (*vkCmdCopyBuffer)(void*, void*, void*, uint32_t, const void*) = nullptr;
    int (*vkResetCommandBuffer)(void*, uint32_t) = nullptr;
};

struct moe_device_allocation {
    uint64_t key = 0;
    uint64_t ptr = 0;
    uint64_t bytes = 0;
    uint64_t last_touch_epoch = 0;
    uint64_t owner_key = 0;
    uint64_t view_offset = 0;
    uint32_t inflight_copies = 0;
    uint32_t bundle_view_ref_count = 0;
    bool pending_free = false;
    bool resident_accounted = false;
    bool is_view = false;
    uint8_t pool = 0;
    uint8_t backend_kind = 0;
    uint64_t backend_aux = 0;
};

struct moe_cufile_handle_key {
    uint64_t dev = 0;
    uint64_t ino = 0;

    bool operator==(const moe_cufile_handle_key& other) const {
        return dev == other.dev && ino == other.ino;
    }
};

struct moe_cufile_handle_key_hash {
    size_t operator()(const moe_cufile_handle_key& key) const {
        const uint64_t mixed = key.dev ^ (key.ino + 0x9e3779b97f4a7c15ull + (key.dev << 6) + (key.dev >> 2));
        return static_cast<size_t>(mixed ^ (mixed >> 32));
    }
};

enum {
    moe_DEVICE_ALLOC_NATIVE_PTR = 0,
    moe_DEVICE_ALLOC_VULKAN_BUFFER = 1,
    moe_DEVICE_ALLOC_D3D12_RESOURCE = 2
};

static constexpr uint32_t moe_DSTORAGE_STATUS_SLOTS = 1024u;

struct moe_dstorage_status_slot {
    std::atomic<bool> in_use{false};
    void* array = nullptr;
    uint32_t index = 0;
    uint64_t generation = 0;
};

struct moe_dstorage_pending_copy {
    void* status_array = nullptr;
    uint32_t status_index = 0;
    moe_dstorage_status_slot* slots = nullptr;
    int slot_index = -1;
    uint64_t generation = 0;
    uint64_t key = 0;
    uint64_t bytes = 0;
    void* resource = nullptr;
    void* file = nullptr;
    uint64_t submitted_us = 0;
    bool direct_to_gpu = false;
};

struct moe_directstorage_state {
    void* factory = nullptr;
    void* queue = nullptr;
    void* gpu_queue = nullptr;
    void* status_array = nullptr;
    void* gpu_status_array = nullptr;
    void* event_handle = nullptr;
    void* completion_wait_handle = nullptr;
    void* d3d12_device = nullptr;
    uint32_t status_index = 0;
    uint32_t gpu_status_index = 0;
    moe_dstorage_status_slot status_slots[moe_DSTORAGE_STATUS_SLOTS]{};
    moe_dstorage_status_slot gpu_status_slots[moe_DSTORAGE_STATUS_SLOTS]{};
    bool initialized = false;
    bool available = false;
    bool gpu_available = false;
};

enum {
    moe_DEVICE_POOL_FIXED = 0,
    moe_DEVICE_POOL_ACTIVATION = 1,
    moe_DEVICE_POOL_EXPERT = 2,
    moe_DEVICE_POOL_STREAM = 3,
    moe_DEVICE_POOL_KV_CACHE = 4
};

struct vram_reservation_token {
    uint64_t bytes = 0;
    uint8_t pool = moe_DEVICE_POOL_EXPERT;
    bool valid = false;
};

enum {
    moe_GDS_BLOCKING_TENSOR = 1,
    moe_GDS_BLOCKING_BUNDLE = 2
};

struct moe_gds_blocking_copy {
    uint32_t kind = 0;
    void* file_handle = nullptr;
    CUdeviceptr dst = 0;
    uint64_t bytes = 0;
    uint64_t ranges[3][2]{};
    uint32_t range_count = 0;
    uint64_t key = 0;
    vram_reservation_token token{};
    uint64_t view_keys[3]{};
    uint32_t view_count = 0;
};
