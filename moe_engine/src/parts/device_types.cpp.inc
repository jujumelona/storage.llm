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
typedef int (*moe_hip_device_get_attribute_fn)(int*, int, int);
typedef int (*moe_ze_init_fn)(uint32_t);
typedef int (*moe_ze_driver_get_fn)(uint32_t*, void**);
typedef int (*moe_ze_device_get_fn)(void*, uint32_t*, void**);
typedef int (*moe_ze_context_create_fn)(void*, const void*, void**);
typedef int (*moe_ze_context_destroy_fn)(void*);
typedef int (*moe_ze_mem_alloc_host_fn)(void*, const void*, size_t, size_t, void**);
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

    // GPUDirect Storage (cuFile) API — NVMe → GPU zero-copy DMA
    // Research: GPUDirect Storage (NVIDIA, 2023-25)
    // Eliminates CPU memcpy in the disk → pinned → GPU path by reading
    // directly from NVMe into GPU memory via RDMA engine.
    void* cufile_library = nullptr;
    moe_cufile_error_t (*cuFileDriverOpen)() = nullptr;
    void (*cuFileDriverClose)() = nullptr;
    moe_cufile_error_t (*cuFileHandleRegister)(void**, const void*) = nullptr; // (CUfileHandle_t*, CUfileDescr_t*)
    void (*cuFileHandleDeregister)(void*) = nullptr;           // (CUfileHandle_t)
    moe_cufile_error_t (*cuFileBufRegister)(void*, size_t, int) = nullptr; // (devPtr, size, flags)
    moe_cufile_error_t (*cuFileBufDeregister)(void*) = nullptr;            // (devPtr)
    moe_cufile_read_fn cuFileRead = nullptr;  // (handle, devPtr, size, file_off, buf_off)
    bool cufile_available = false;

    // cuMemHostRegister — for io_uring registered buffer integration
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
    moe_cu_mem_get_info_fn hipMemGetInfo = nullptr;
    moe_cu_stream_create_fn hipStreamCreate = nullptr;
    moe_cu_stream_synchronize_fn hipStreamSynchronize = nullptr;
    moe_cu_stream_destroy_fn hipStreamDestroy = nullptr;
    moe_cu_event_create_fn hipEventCreate = nullptr;
    moe_cu_event_record_fn hipEventRecord = nullptr;
    moe_cu_event_query_fn hipEventQuery = nullptr;
    moe_cu_event_synchronize_fn hipEventSynchronize = nullptr;
    moe_cu_event_destroy_fn hipEventDestroy = nullptr;
};

// Level Zero basic API hooks
struct moe_level_zero_driver_api {
    void* library = nullptr;
    void* driver = nullptr;
    void* context = nullptr;
    void* device = nullptr;
    moe_ze_init_fn zeInit = nullptr;
    moe_ze_driver_get_fn zeDriverGet = nullptr;
    moe_ze_device_get_fn zeDeviceGet = nullptr;
    moe_ze_context_create_fn zeContextCreate = nullptr;
    moe_ze_context_destroy_fn zeContextDestroy = nullptr;
    moe_ze_mem_alloc_host_fn zeMemAllocHost = nullptr;
    moe_ze_mem_free_fn zeMemFree = nullptr;
    void* (*zeCommandQueueCreate)(void*, void*, void*, void**) = nullptr;
    void* (*zeCommandQueueDestroy)(void*) = nullptr;
    void* (*zeCommandQueueSynchronize)(void*, uint64_t) = nullptr;
    void* (*zeEventCreate)(void*, void*, void**) = nullptr;
    void* (*zeEventDestroy)(void*) = nullptr;
    void* (*zeEventHostSynchronize)(void*, uint64_t) = nullptr;
    void* (*zeEventQueryStatus)(void*) = nullptr;
    void* (*zeCommandListAppendMemoryCopy)(void*, void*, const void*, size_t, void*, uint32_t, void**) = nullptr;
    moe_ze_device_get_memory_properties_fn zeDeviceGetMemoryProperties = nullptr;
};

// Vulkan basic API hooks
struct moe_vulkan_driver_api {
    void* library = nullptr;
    void* instance = nullptr;
    void* physicalDevice = nullptr;
    void* device = nullptr;
    void* (*vkCreateCommandPool)(void*, const void*, const void*, void**) = nullptr;
    void* (*vkDestroyCommandPool)(void*, void*, const void*) = nullptr;
    void* (*vkAllocateCommandBuffers)(void*, const void*, void**) = nullptr;
    void* (*vkFreeCommandBuffers)(void*, void*, uint32_t, const void**) = nullptr;
    void* (*vkQueueSubmit)(void*, uint32_t, const void*, void*) = nullptr;
    void* (*vkQueueWaitIdle)(void*) = nullptr;
    void* (*vkCreateEvent)(void*, const void*, const void*, void**) = nullptr;
    void* (*vkDestroyEvent)(void*, void*, const void*) = nullptr;
    void* (*vkGetEventStatus)(void*, void*) = nullptr;
    void* (*vkGetPhysicalDeviceMemoryProperties2)(void*, void*) = nullptr;
};

struct moe_device_allocation {
    uint64_t key = 0;
    uint64_t ptr = 0;
    uint64_t bytes = 0;
    uint64_t last_touch_epoch = 0;
    uint32_t inflight_copies = 0;
    bool pending_free = false;
    uint8_t pool = 0;
};

enum {
    moe_DEVICE_POOL_FIXED = 0,
    moe_DEVICE_POOL_ACTIVATION = 1,
    moe_DEVICE_POOL_EXPERT = 2,
    moe_DEVICE_POOL_STREAM = 3,
    moe_DEVICE_POOL_KV_CACHE = 4  // BUGFIX 문제4: Add KV cache pool ★★★
};
