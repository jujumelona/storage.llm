static int compiled_directstorage_available() {
#ifdef moe_ENABLE_DIRECTSTORAGE
    return 1;
#else
    return 0;
#endif
}

static int compiled_gpudirect_available() {
#ifdef moe_ENABLE_GPUDIRECT_STORAGE
    return 1;
#else
    return 0;
#endif
}

static int compiled_bam_nvme_available() {
#ifdef moe_ENABLE_BAM_NVME
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
