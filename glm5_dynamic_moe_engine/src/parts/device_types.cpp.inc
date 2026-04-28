#ifndef CU_TYPES_DEFINED
#define CU_TYPES_DEFINED
typedef uint64_t CUdeviceptr;
typedef void* CUstream;
typedef void* CUcontext;
#endif

typedef int (*glm5_cu_init_fn)(unsigned int);
typedef int (*glm5_cu_device_get_fn)(int*, int);
typedef int (*glm5_cu_ctx_set_current_fn)(void*);
typedef int (*glm5_cu_primary_ctx_retain_fn)(void**, int);
typedef int (*glm5_cu_primary_ctx_release_fn)(int);
typedef int (*glm5_cu_mem_alloc_fn)(uint64_t*, size_t);
typedef int (*glm5_cu_mem_free_fn)(uint64_t);
typedef int (*glm5_cu_memcpy_htod_fn)(uint64_t, const void*, size_t);
typedef int (*glm5_cu_memcpy_htod_async_fn)(uint64_t, const void*, size_t, void*);
typedef int (*glm5_cu_mem_get_info_fn)(size_t*, size_t*);
typedef int (*glm5_cu_stream_create_fn)(void**, unsigned int);
typedef int (*glm5_cu_stream_synchronize_fn)(void*);
typedef int (*glm5_cu_stream_destroy_fn)(void*);

struct glm5_cuda_driver_api {
    void* library = nullptr;
    void* context = nullptr;
    int device = 0;
    glm5_cu_init_fn cuInit = nullptr;
    glm5_cu_device_get_fn cuDeviceGet = nullptr;
    glm5_cu_ctx_set_current_fn cuCtxSetCurrent = nullptr;
    glm5_cu_primary_ctx_retain_fn cuDevicePrimaryCtxRetain = nullptr;
    glm5_cu_primary_ctx_release_fn cuDevicePrimaryCtxRelease = nullptr;
    glm5_cu_mem_alloc_fn cuMemAlloc = nullptr;
    glm5_cu_mem_free_fn cuMemFree = nullptr;
    glm5_cu_memcpy_htod_fn cuMemcpyHtoD = nullptr;
    glm5_cu_memcpy_htod_async_fn cuMemcpyHtoDAsync = nullptr;
    glm5_cu_mem_get_info_fn cuMemGetInfo = nullptr;
    glm5_cu_stream_create_fn cuStreamCreate = nullptr;
    glm5_cu_stream_synchronize_fn cuStreamSynchronize = nullptr;
    glm5_cu_stream_destroy_fn cuStreamDestroy = nullptr;
};

struct glm5_device_allocation {
    uint64_t key = 0;
    uint64_t ptr = 0;
    uint64_t bytes = 0;
    uint64_t last_touch_epoch = 0;
    uint32_t inflight_copies = 0;
    bool pending_free = false;
    uint8_t pool = 0;
};

enum {
    GLM5_DEVICE_POOL_FIXED = 0,
    GLM5_DEVICE_POOL_ACTIVATION = 1,
    GLM5_DEVICE_POOL_EXPERT = 2,
    GLM5_DEVICE_POOL_STREAM = 3
};
