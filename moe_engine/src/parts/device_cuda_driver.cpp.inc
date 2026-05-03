static void* moe_open_cuda_driver_library() {
#ifdef _WIN32
    return (void*)LoadLibraryA("nvcuda.dll");
#else
    void* lib = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    return lib ? lib : dlopen("libcuda.so", RTLD_LAZY | RTLD_LOCAL);
#endif
}

static void moe_close_shared_library(void* lib) {
    if (!lib) {
        return;
    }
#ifdef _WIN32
    FreeLibrary((HMODULE)lib);
#else
    dlclose(lib);
#endif
}

static void* moe_find_shared_symbol(void* lib, const char* name) {
    if (!lib || !name) {
        return nullptr;
    }
#ifdef _WIN32
    return (void*)GetProcAddress((HMODULE)lib, name);
#else
    return dlsym(lib, name);
#endif
}

static void* moe_find_cuda_symbol(void* lib, const char* a, const char* b = nullptr) {
    void* sym = moe_find_shared_symbol(lib, a);
    return sym ? sym : moe_find_shared_symbol(lib, b);
}

static int runtime_cuda_driver_available() {
    void* lib = moe_open_cuda_driver_library();
    if (!lib) {
        return 0;
    }
    const int ok = moe_find_shared_symbol(lib, "cuInit") ? 1 : 0;
    moe_close_shared_library(lib);
    return ok;
}

static int cuda_driver_load(moe_cuda_driver_api* api) {
    if (!api) {
        return 0;
    }
    api->library = moe_open_cuda_driver_library();
    if (!api->library) {
        return 0;
    }
    api->cuInit = (moe_cu_init_fn)moe_find_shared_symbol(api->library, "cuInit");
    api->cuDeviceGet = (moe_cu_device_get_fn)moe_find_shared_symbol(api->library, "cuDeviceGet");
    api->cuDeviceGetAttribute = (moe_cu_device_get_attribute_fn)moe_find_shared_symbol(api->library, "cuDeviceGetAttribute");
    api->cuDeviceGetPCIBusId = (moe_cu_device_get_pci_bus_id_fn)moe_find_shared_symbol(api->library, "cuDeviceGetPCIBusId");
    api->cuCtxSetCurrent = (moe_cu_ctx_set_current_fn)moe_find_shared_symbol(api->library, "cuCtxSetCurrent");
    api->cuCtxSynchronize = (moe_cu_ctx_synchronize_fn)moe_find_shared_symbol(api->library, "cuCtxSynchronize");
    api->cuDevicePrimaryCtxRetain = (moe_cu_primary_ctx_retain_fn)moe_find_shared_symbol(api->library, "cuDevicePrimaryCtxRetain");
    api->cuDevicePrimaryCtxRelease = (moe_cu_primary_ctx_release_fn)moe_find_shared_symbol(api->library, "cuDevicePrimaryCtxRelease");
    api->cuMemAlloc = (moe_cu_mem_alloc_fn)moe_find_cuda_symbol(api->library, "cuMemAlloc_v2", "cuMemAlloc");
    api->cuMemFree = (moe_cu_mem_free_fn)moe_find_cuda_symbol(api->library, "cuMemFree_v2", "cuMemFree");
    api->cuMemcpyHtoD = (moe_cu_memcpy_htod_fn)moe_find_cuda_symbol(api->library, "cuMemcpyHtoD_v2", "cuMemcpyHtoD");
    api->cuMemGetInfo = (moe_cu_mem_get_info_fn)moe_find_cuda_symbol(api->library, "cuMemGetInfo_v2", "cuMemGetInfo");
    api->cuMemcpyHtoDAsync = (moe_cu_memcpy_htod_async_fn)moe_find_cuda_symbol(api->library, "cuMemcpyHtoDAsync_v2", "cuMemcpyHtoDAsync");
    api->cuMemcpyDtoD = (moe_cu_memcpy_dtod_fn)moe_find_cuda_symbol(api->library, "cuMemcpyDtoD_v2", "cuMemcpyDtoD");
    api->cuStreamCreate = (moe_cu_stream_create_fn)moe_find_shared_symbol(api->library, "cuStreamCreate");
    api->cuStreamSynchronize = (moe_cu_stream_synchronize_fn)moe_find_shared_symbol(api->library, "cuStreamSynchronize");
    api->cuStreamDestroy = (moe_cu_stream_destroy_fn)moe_find_cuda_symbol(api->library, "cuStreamDestroy_v2", "cuStreamDestroy");
    // CUDA event API is optional and enables non-blocking stream completion checks.
    api->cuEventCreate = (moe_cu_event_create_fn)moe_find_shared_symbol(api->library, "cuEventCreate");
    api->cuEventRecord = (moe_cu_event_record_fn)moe_find_shared_symbol(api->library, "cuEventRecord");
    api->cuEventQuery = (moe_cu_event_query_fn)moe_find_shared_symbol(api->library, "cuEventQuery");
    api->cuEventSynchronize = (moe_cu_event_synchronize_fn)moe_find_shared_symbol(api->library, "cuEventSynchronize");
    api->cuEventDestroy = (moe_cu_event_destroy_fn)moe_find_cuda_symbol(api->library, "cuEventDestroy_v2", "cuEventDestroy");
    
    // CUDA Graph APIs
    api->cuGraphCreate = (void* (*)(void**, unsigned int))moe_find_shared_symbol(api->library, "cuGraphCreate");
    api->cuGraphDestroy = (void* (*)(void*))moe_find_shared_symbol(api->library, "cuGraphDestroy");
    api->cuGraphAddKernelNode = (void* (*)(void**, void*, void**, size_t, void*))moe_find_shared_symbol(api->library, "cuGraphAddKernelNode");
    api->cuGraphInstantiate = (void* (*)(void**, void*, void*, char*, size_t))moe_find_shared_symbol(api->library, "cuGraphInstantiate");
    api->cuGraphLaunch = (void* (*)(void*, void*))moe_find_shared_symbol(api->library, "cuGraphLaunch");
    api->cuGraphExecDestroy = (void* (*)(void*))moe_find_shared_symbol(api->library, "cuGraphExecDestroy");
    api->cuStreamBeginCapture = (void* (*)(void*, int))moe_find_cuda_symbol(api->library, "cuStreamBeginCapture_v2", "cuStreamBeginCapture");
    api->cuStreamEndCapture = (void* (*)(void*, void**))moe_find_shared_symbol(api->library, "cuStreamEndCapture");

    // cuMemHostRegister — needed for io_uring registered buffer integration
    api->cuMemHostRegister = (int (*)(void*, size_t, unsigned int))moe_find_cuda_symbol(api->library, "cuMemHostRegister_v2", "cuMemHostRegister");
    api->cuMemHostUnregister = (int (*)(void*))moe_find_shared_symbol(api->library, "cuMemHostUnregister");

    // GPUDirect Storage (cuFile) — load from separate GDS library
    // cuFile enables NVMe → GPU DMA, bypassing CPU staging entirely.
    // Not available on all systems; graceful fallback to staged pipeline.
#ifdef _WIN32
    api->cufile_library = (void*)LoadLibraryA("cufile.dll");
#else
    api->cufile_library = dlopen("libcufile.so.0", RTLD_LAZY | RTLD_LOCAL);
    if (!api->cufile_library) {
        api->cufile_library = dlopen("libcufile.so", RTLD_LAZY | RTLD_LOCAL);
    }
#endif
    if (api->cufile_library) {
        api->cuFileDriverOpen = (moe_cufile_error_t (*)())moe_find_shared_symbol(api->cufile_library, "cuFileDriverOpen");
        api->cuFileDriverClose = (void (*)())moe_find_shared_symbol(api->cufile_library, "cuFileDriverClose");
        api->cuFileHandleRegister = (moe_cufile_error_t (*)(void**, const void*))moe_find_shared_symbol(api->cufile_library, "cuFileHandleRegister");
        api->cuFileHandleDeregister = (void (*)(void*))moe_find_shared_symbol(api->cufile_library, "cuFileHandleDeregister");
        api->cuFileBufRegister = (moe_cufile_error_t (*)(void*, size_t, int))moe_find_shared_symbol(api->cufile_library, "cuFileBufRegister");
        api->cuFileBufDeregister = (moe_cufile_error_t (*)(void*))moe_find_shared_symbol(api->cufile_library, "cuFileBufDeregister");
        api->cuFileRead = (moe_cufile_read_fn)moe_find_shared_symbol(api->cufile_library, "cuFileRead");
        api->cufile_available = api->cuFileDriverOpen && api->cuFileRead && api->cuFileBufRegister;
        if (api->cufile_available) {
            std::fprintf(stderr, "[storageLLM INFO] GPUDirect Storage (cuFile) API loaded successfully.\n");
        }
    }

    return api->cuInit && api->cuDeviceGet && api->cuCtxSetCurrent &&
        api->cuDevicePrimaryCtxRetain && api->cuDevicePrimaryCtxRelease &&
        api->cuMemAlloc && api->cuMemFree && api->cuMemcpyHtoD && api->cuMemGetInfo;
}

static uint64_t cuda_driver_detect_free_memory_bytes(uint32_t device_id, uint64_t* total_out) {
    if (total_out) {
        *total_out = 0;
    }
    moe_cuda_driver_api api{};
    if (!cuda_driver_load(&api) || api.cuInit(0) != 0) {
        moe_close_shared_library(api.library);
        return 0;
    }
    if (api.cuDeviceGet(&api.device, static_cast<int>(device_id)) != 0 ||
        api.cuDevicePrimaryCtxRetain(&api.context, api.device) != 0 ||
        api.cuCtxSetCurrent(api.context) != 0) {
        if (api.cuDevicePrimaryCtxRelease && api.context) {
            api.cuDevicePrimaryCtxRelease(api.device);
        }
        moe_close_shared_library(api.library);
        return 0;
    }
    size_t free_b = 0;
    size_t total_b = 0;
    const int ok = api.cuMemGetInfo(&free_b, &total_b) == 0;
    if (api.cuDevicePrimaryCtxRelease && api.context) {
        api.cuDevicePrimaryCtxRelease(api.device);
    }
    moe_close_shared_library(api.library);
    if (!ok) {
        return 0;
    }
    if (total_out) {
        *total_out = static_cast<uint64_t>(total_b);
    }
    return static_cast<uint64_t>(free_b);
}

static int cuda_driver_get_numa_node(uint32_t device_id) {
    moe_cuda_driver_api api{};
    if (!cuda_driver_load(&api) || api.cuInit(0) != 0) {
        moe_close_shared_library(api.library);
        return -1;
    }
    int numa_node = -1;
    if (api.cuDeviceGet && api.cuDeviceGetPCIBusId) {
        int dev = 0;
        if (api.cuDeviceGet(&dev, static_cast<int>(device_id)) == 0) {
            char pci_id[32] = {0};
            if (api.cuDeviceGetPCIBusId(pci_id, sizeof(pci_id), dev) == 0) {
#ifdef __linux__
                char path[256];
                snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/numa_node", pci_id);
                FILE* f = fopen(path, "r");
                if (f) {
                    if (fscanf(f, "%d", &numa_node) != 1) numa_node = -1;
                    fclose(f);
                }
#endif
            }
        }
    }
    moe_close_shared_library(api.library);
    return numa_node;
}
