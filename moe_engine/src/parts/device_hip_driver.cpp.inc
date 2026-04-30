static void* moe_open_hip_driver_library() {
#ifdef _WIN32
    return (void*)LoadLibraryA("amdhip64.dll");
#else
    void* lib = dlopen("libamdhip64.so.1", RTLD_LAZY | RTLD_LOCAL);
    return lib ? lib : dlopen("libamdhip64.so", RTLD_LAZY | RTLD_LOCAL);
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

static void* moe_find_hip_symbol(void* lib, const char* a, const char* b = nullptr) {
    void* sym = moe_find_shared_symbol(lib, a);
    return sym ? sym : moe_find_shared_symbol(lib, b);
}

static int runtime_hip_driver_available() {
    void* lib = moe_open_hip_driver_library();
    if (!lib) {
        return 0;
    }
    const int ok = moe_find_shared_symbol(lib, "hipInit") ? 1 : 0;
    moe_close_shared_library(lib);
    return ok;
}

static int hip_driver_load(moe_hip_driver_api* api) {
    if (!api) {
        return 0;
    }
    api->library = moe_open_hip_driver_library();
    if (!api->library) {
        return 0;
    }
    api->hipInit = (moe_hip_init_fn)moe_find_shared_symbol(api->library, "hipInit");
    api->hipDeviceGet = (moe_hip_device_get_fn)moe_find_shared_symbol(api->library, "hipDeviceGet");
    api->hipCtxSetcurrent = (moe_hip_ctx_set_current_fn)moe_find_shared_symbol(api->library, "hipCtxSetcurrent");
    api->hipDevicePrimaryCtxRetain = (moe_hip_primary_ctx_retain_fn)moe_find_shared_symbol(api->library, "hipDevicePrimaryCtxRetain");
    api->hipDevicePrimaryCtxRelease = (moe_hip_primary_ctx_release_fn)moe_find_shared_symbol(api->library, "hipDevicePrimaryCtxRelease");
    api->hipMemAlloc = (moe_hip_mem_alloc_fn)moe_find_hip_symbol(api->library, "hipMemAlloc_v2", "hipMemAlloc");
    api->hipMemFree = (moe_hip_mem_free_fn)moe_find_hip_symbol(api->library, "hipMemFree_v2", "hipMemFree");
    api->hipMemcpyHtoD = (moe_hip_memcpy_htod_fn)moe_find_hip_symbol(api->library, "hipMemcpyHtoD_v2", "hipMemcpyHtoD");
    api->hipMemGetInfo = (moe_hip_mem_get_info_fn)moe_find_hip_symbol(api->library, "hipMemGetInfo_v2", "hipMemGetInfo");
    api->hipMemcpyHtoDAsync = (moe_hip_memcpy_htod_async_fn)moe_find_hip_symbol(api->library, "hipMemcpyHtoDAsync_v2", "hipMemcpyHtoDAsync");
    api->hipStreamCreate = (moe_hip_stream_create_fn)moe_find_shared_symbol(api->library, "hipStreamCreate");
    api->hipStreamSynchronize = (moe_hip_stream_synchronize_fn)moe_find_shared_symbol(api->library, "hipStreamSynchronize");
    api->hipStreamDestroy = (moe_hip_stream_destroy_fn)moe_find_hip_symbol(api->library, "hipStreamDestroy_v2", "hipStreamDestroy");
    // hip event API is optional and enables non-blocking stream completion checks.
    api->hipEventCreate = (moe_hip_event_create_fn)moe_find_shared_symbol(api->library, "hipEventCreate");
    api->hipEventRecord = (moe_hip_event_record_fn)moe_find_shared_symbol(api->library, "hipEventRecord");
    api->hipEventQuery = (moe_hip_event_query_fn)moe_find_shared_symbol(api->library, "hipEventQuery");
    api->hipEventSynchronize = (moe_hip_event_synchronize_fn)moe_find_shared_symbol(api->library, "hipEventSynchronize");
    api->hipEventDestroy = (moe_hip_event_destroy_fn)moe_find_hip_symbol(api->library, "hipEventDestroy_v2", "hipEventDestroy");
    return api->hipInit && api->hipDeviceGet && api->hipCtxSetcurrent &&
        api->hipDevicePrimaryCtxRetain && api->hipDevicePrimaryCtxRelease &&
        api->hipMemAlloc && api->hipMemFree && api->hipMemcpyHtoD && api->hipMemGetInfo;
}

static uint64_t hip_driver_detect_free_memory_bytes(uint32_t device_id, uint64_t* total_out) {
    if (total_out) {
        *total_out = 0;
    }
    moe_hip_driver_api api{};
    if (!hip_driver_load(&api) || api.hipInit(0) != 0) {
        moe_close_shared_library(api.library);
        return 0;
    }
    if (api.hipDeviceGet(&api.device, static_cast<int>(device_id)) != 0 ||
        api.hipDevicePrimaryCtxRetain(&api.context, api.device) != 0 ||
        api.hipCtxSetcurrent(api.context) != 0) {
        if (api.hipDevicePrimaryCtxRelease && api.context) {
            api.hipDevicePrimaryCtxRelease(api.device);
        }
        moe_close_shared_library(api.library);
        return 0;
    }
    size_t free_b = 0;
    size_t total_b = 0;
    const int ok = api.hipMemGetInfo(&free_b, &total_b) == 0;
    if (api.hipDevicePrimaryCtxRelease && api.context) {
        api.hipDevicePrimaryCtxRelease(api.device);
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
