#ifndef CU_TYPES_DEFINED
#define CU_TYPES_DEFINED
typedef uint64_t CUdeviceptr;
typedef void* CUstream;
typedef void* CUcontext;
typedef void* CUevent;
#endif

typedef int (*moe_cu_init_fn)(unsigned int);
typedef int (*moe_cu_device_get_fn)(int*, int);
typedef int (*moe_cu_device_get_pci_bus_id_fn)(char*, int, int);
typedef int (*moe_cu_ctx_set_current_fn)(void*);
typedef int (*moe_cu_primary_ctx_retain_fn)(void**, int);
typedef int (*moe_cu_primary_ctx_release_fn)(int);
typedef int (*moe_cu_mem_alloc_fn)(uint64_t*, size_t);
typedef int (*moe_cu_mem_free_fn)(uint64_t);
typedef int (*moe_cu_memcpy_htod_fn)(uint64_t, const void*, size_t);
typedef int (*moe_cu_memcpy_htod_async_fn)(uint64_t, const void*, size_t, void*);
typedef int (*moe_cu_mem_get_info_fn)(size_t*, size_t*);
typedef int (*moe_cu_stream_create_fn)(void**, unsigned int);
typedef int (*moe_cu_stream_synchronize_fn)(void*);
typedef int (*moe_cu_stream_destroy_fn)(void*);
typedef int (*moe_cu_event_create_fn)(void**, unsigned int);
typedef int (*moe_cu_event_record_fn)(void*, void*);
typedef int (*moe_cu_event_query_fn)(void*);
typedef int (*moe_cu_event_synchronize_fn)(void*);
typedef int (*moe_cu_event_destroy_fn)(void*);

struct moe_cuda_driver_api {
    void* library = nullptr;
    void* context = nullptr;
    int device = 0;
    moe_cu_init_fn cuInit = nullptr;
    moe_cu_device_get_fn cuDeviceGet = nullptr;
    moe_cu_device_get_pci_bus_id_fn cuDeviceGetPCIBusId = nullptr;
    moe_cu_ctx_set_current_fn cuCtxSetCurrent = nullptr;
    moe_cu_primary_ctx_retain_fn cuDevicePrimaryCtxRetain = nullptr;
    moe_cu_primary_ctx_release_fn cuDevicePrimaryCtxRelease = nullptr;
    moe_cu_mem_alloc_fn cuMemAlloc = nullptr;
    moe_cu_mem_free_fn cuMemFree = nullptr;
    moe_cu_memcpy_htod_fn cuMemcpyHtoD = nullptr;
    moe_cu_memcpy_htod_async_fn cuMemcpyHtoDAsync = nullptr;
    moe_cu_mem_get_info_fn cuMemGetInfo = nullptr;
    moe_cu_stream_create_fn cuStreamCreate = nullptr;
    moe_cu_stream_synchronize_fn cuStreamSynchronize = nullptr;
    moe_cu_stream_destroy_fn cuStreamDestroy = nullptr;
    moe_cu_event_create_fn cuEventCreate = nullptr;
    moe_cu_event_record_fn cuEventRecord = nullptr;
    moe_cu_event_query_fn cuEventQuery = nullptr;
    moe_cu_event_synchronize_fn cuEventSynchronize = nullptr;
    moe_cu_event_destroy_fn cuEventDestroy = nullptr;
};

struct moe_hip_driver_api {
    void* library = nullptr;
    void* context = nullptr;
    int device = 0;
    moe_cu_init_fn hipInit = nullptr;
    moe_cu_device_get_fn hipDeviceGet = nullptr;
    moe_cu_device_get_pci_bus_id_fn hipDeviceGetPCIBusId = nullptr;
    moe_cu_ctx_set_current_fn hipCtxSetCurrent = nullptr;
    moe_cu_primary_ctx_retain_fn hipDevicePrimaryCtxRetain = nullptr;
    moe_cu_primary_ctx_release_fn hipDevicePrimaryCtxRelease = nullptr;
    moe_cu_mem_alloc_fn hipMalloc = nullptr;
    moe_cu_mem_free_fn hipFree = nullptr;
    moe_cu_memcpy_htod_fn hipMemcpyHtoD = nullptr;
    moe_cu_memcpy_htod_async_fn hipMemcpyHtoDAsync = nullptr;
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
    moe_DEVICE_POOL_STREAM = 3
};
