static void* glm5_open_cuda_driver_library() {
#ifdef _WIN32
    return (void*)LoadLibraryA("nvcuda.dll");
#else
    void* lib = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    return lib ? lib : dlopen("libcuda.so", RTLD_LAZY | RTLD_LOCAL);
#endif
}

static void glm5_close_shared_library(void* lib) {
    if (!lib) {
        return;
    }
#ifdef _WIN32
    FreeLibrary((HMODULE)lib);
#else
    dlclose(lib);
#endif
}

static void* glm5_find_shared_symbol(void* lib, const char* name) {
    if (!lib || !name) {
        return nullptr;
    }
#ifdef _WIN32
    return (void*)GetProcAddress((HMODULE)lib, name);
#else
    return dlsym(lib, name);
#endif
}

static void* glm5_find_cuda_symbol(void* lib, const char* a, const char* b = nullptr) {
    void* sym = glm5_find_shared_symbol(lib, a);
    return sym ? sym : glm5_find_shared_symbol(lib, b);
}

static int runtime_cuda_driver_available() {
    void* lib = glm5_open_cuda_driver_library();
    if (!lib) {
        return 0;
    }
    const int ok = glm5_find_shared_symbol(lib, "cuInit") ? 1 : 0;
    glm5_close_shared_library(lib);
    return ok;
}

static int cuda_driver_load(glm5_cuda_driver_api* api) {
    if (!api) {
        return 0;
    }
    api->library = glm5_open_cuda_driver_library();
    if (!api->library) {
        return 0;
    }
    api->cuInit = (glm5_cu_init_fn)glm5_find_shared_symbol(api->library, "cuInit");
    api->cuDeviceGet = (glm5_cu_device_get_fn)glm5_find_shared_symbol(api->library, "cuDeviceGet");
    api->cuCtxSetCurrent = (glm5_cu_ctx_set_current_fn)glm5_find_shared_symbol(api->library, "cuCtxSetCurrent");
    api->cuDevicePrimaryCtxRetain = (glm5_cu_primary_ctx_retain_fn)glm5_find_shared_symbol(api->library, "cuDevicePrimaryCtxRetain");
    api->cuDevicePrimaryCtxRelease = (glm5_cu_primary_ctx_release_fn)glm5_find_shared_symbol(api->library, "cuDevicePrimaryCtxRelease");
    api->cuMemAlloc = (glm5_cu_mem_alloc_fn)glm5_find_cuda_symbol(api->library, "cuMemAlloc_v2", "cuMemAlloc");
    api->cuMemFree = (glm5_cu_mem_free_fn)glm5_find_cuda_symbol(api->library, "cuMemFree_v2", "cuMemFree");
    api->cuMemcpyHtoD = (glm5_cu_memcpy_htod_fn)glm5_find_cuda_symbol(api->library, "cuMemcpyHtoD_v2", "cuMemcpyHtoD");
    api->cuMemGetInfo = (glm5_cu_mem_get_info_fn)glm5_find_cuda_symbol(api->library, "cuMemGetInfo_v2", "cuMemGetInfo");
    api->cuMemcpyHtoDAsync = (glm5_cu_memcpy_htod_async_fn)glm5_find_cuda_symbol(api->library, "cuMemcpyHtoDAsync_v2", "cuMemcpyHtoDAsync");
    api->cuStreamCreate = (glm5_cu_stream_create_fn)glm5_find_shared_symbol(api->library, "cuStreamCreate");
    api->cuStreamSynchronize = (glm5_cu_stream_synchronize_fn)glm5_find_shared_symbol(api->library, "cuStreamSynchronize");
    api->cuStreamDestroy = (glm5_cu_stream_destroy_fn)glm5_find_cuda_symbol(api->library, "cuStreamDestroy_v2", "cuStreamDestroy");
    // CUDA event API is optional and enables non-blocking stream completion checks.
    api->cuEventCreate = (glm5_cu_event_create_fn)glm5_find_shared_symbol(api->library, "cuEventCreate");
    api->cuEventRecord = (glm5_cu_event_record_fn)glm5_find_shared_symbol(api->library, "cuEventRecord");
    api->cuEventQuery = (glm5_cu_event_query_fn)glm5_find_shared_symbol(api->library, "cuEventQuery");
    api->cuEventDestroy = (glm5_cu_event_destroy_fn)glm5_find_cuda_symbol(api->library, "cuEventDestroy_v2", "cuEventDestroy");
    return api->cuInit && api->cuDeviceGet && api->cuCtxSetCurrent &&
        api->cuDevicePrimaryCtxRetain && api->cuDevicePrimaryCtxRelease &&
        api->cuMemAlloc && api->cuMemFree && api->cuMemcpyHtoD && api->cuMemGetInfo;
}
