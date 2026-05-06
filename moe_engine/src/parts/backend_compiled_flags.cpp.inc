static int compiled_directstorage_available() {
#if defined(moe_ENABLE_DIRECTSTORAGE) || defined(HAVE_DIRECTSTORAGE)
    return 1;
#else
    return 0;
#endif
}

static int compiled_gpudirect_available() {
#if defined(moe_ENABLE_GPUDIRECT_STORAGE) || defined(HAVE_GPUDIRECT_STORAGE)
    return 1;
#else
    // BUGFIX 973: Runtime GDS detection via dynamic library loading ★★★ FEATURE
    // Problem: GDS only available when compiled with HAVE_GPUDIRECT_STORAGE.
    // Users with cuFile installed but building without the flag get no GDS.
    // Solution: Try dlopen/LoadLibrary for cuFile library at runtime.
    // If found, GDS path can be used (cuFile API loaded via function pointers).
    // Impact: Enables NVMe→VRAM direct path without recompilation.
    static int cached = -1;
    if (cached >= 0) return cached;
#ifdef _WIN32
    HMODULE lib = LoadLibraryA("cufile.dll");
    cached = lib ? 1 : 0;
    if (lib) FreeLibrary(lib);
#elif defined(__linux__)
    void* lib = dlopen("libcufile.so", RTLD_LAZY | RTLD_NOLOAD);
    if (!lib) lib = dlopen("libcufile.so.0", RTLD_LAZY);
    cached = lib ? 1 : 0;
    if (lib) dlclose(lib);
#else
    cached = 0;
#endif
    return cached;
#endif
}

static int compiled_bam_nvme_available() {
#if defined(moe_ENABLE_BAM_NVME) || defined(HAVE_BAM_NVME)
    return 1;
#else
    return 0;
#endif
}

static int compiled_cuda_available() {
#ifdef moe_ENABLE_CUDA_BACKEND
    return 1;
#else
    return runtime_cuda_driver_available();
#endif
}

static int compiled_hip_available() {
#ifdef moe_ENABLE_HIP_BACKEND
    return 1;
#else
    return 0;
#endif
}

static int compiled_level_zero_available() {
#ifdef moe_ENABLE_LEVEL_ZERO_BACKEND
    return 1;
#else
    return 0;
#endif
}

static int compiled_sycl_available() {
#ifdef moe_ENABLE_SYCL_BACKEND
    return 1;
#else
    return 0;
#endif
}

static int compiled_metal_available() {
#if defined(__APPLE__) || defined(moe_ENABLE_METAL_BACKEND)
    return 1;
#else
    return 0;
#endif
}

static int compiled_vulkan_available() {
#ifdef moe_ENABLE_VULKAN_BACKEND
    return 1;
#else
    return 0;
#endif
}

static int compiled_directml_available() {
#ifdef moe_ENABLE_DIRECTML_BACKEND
    return 1;
#else
    return 0;
#endif
}

static int compiled_opencl_available() {
#ifdef moe_ENABLE_OPENCL_BACKEND
    return 1;
#else
    return 0;
#endif
}

static int compiled_webgpu_available() {
#ifdef moe_ENABLE_WEBGPU_BACKEND
    return 1;
#else
    return 0;
#endif
}
