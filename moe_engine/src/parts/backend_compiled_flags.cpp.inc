static int compiled_directstorage_available() {
#if defined(_WIN32) && (defined(moe_ENABLE_DIRECTSTORAGE) || defined(HAVE_DIRECTSTORAGE)) && defined(__has_include)
#if __has_include(<dstorage.h>) && __has_include(<d3d12.h>)
    return 1;
#else
    return 0;
#endif
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
    // BUGFIX NEW-BUG2: Static variable race condition ★★ IMPORTANT
    // Problem: static int cached = -1 → multiple threads race on check and update
    // → Multiple LoadLibrary calls, data race on cached write
    // Solution: Use atomic with CAS for thread-safe lazy initialization
    // Impact: Eliminates race condition, ensures single library check
    static std::atomic<int> cached{-1};
    int current = cached.load(std::memory_order_acquire);
    if (current >= 0) return current;

    int result = 0;
#ifdef _WIN32
    HMODULE lib = LoadLibraryA("cufile.dll");
    result = lib ? 1 : 0;
    if (lib) FreeLibrary(lib);
#elif defined(__linux__)
    void* lib = dlopen("libcufile.so", RTLD_LAZY | RTLD_NOLOAD);
    if (!lib) lib = dlopen("libcufile.so.0", RTLD_LAZY);
    result = lib ? 1 : 0;
    if (lib) dlclose(lib);
#else
    result = 0;
#endif

    // CAS to ensure only one thread updates
    int expected = -1;
    if (cached.compare_exchange_strong(expected, result, std::memory_order_release, std::memory_order_acquire)) {
        return result;
    }
    // Another thread won the race, use their result
    return cached.load(std::memory_order_acquire);
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
    return runtime_cuda_driver_available();
}

static int compiled_hip_available() {
    return runtime_hip_driver_available();
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
