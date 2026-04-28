static void fill_backend_caps(
    glm5_backend_t preferred,
    glm5_platform_t platform,
    glm5_backend_caps_t* out_caps
) {
    if (!out_caps) {
        return;
    }
    std::memset(out_caps, 0, sizeof(*out_caps));
    if (platform == GLM5_PLATFORM_AUTO) {
        platform = default_platform();
    }
    const glm5_backend_t backend = choose_backend(preferred, platform);
    out_caps->backend = backend;
    out_caps->platform = platform;
    out_caps->device_id = 0;
    out_caps->supports_mmap = 1;

    if (backend == GLM5_BACKEND_CPU) {
        fill_cpu_backend_caps(out_caps);
        return;
    }

    fill_gpu_backend_base(out_caps);
    if (backend == GLM5_BACKEND_CUDA) {
        fill_cuda_backend_caps(out_caps);
    } else if (backend == GLM5_BACKEND_HIP) {
        fill_hip_backend_caps(out_caps);
    } else if (backend == GLM5_BACKEND_LEVEL_ZERO) {
        out_caps->supports_level_zero = compiled_level_zero_available();
        fill_shared_memory_backend_caps(out_caps);
    } else if (backend == GLM5_BACKEND_SYCL) {
        out_caps->supports_sycl = compiled_sycl_available();
        fill_shared_memory_backend_caps(out_caps);
    } else if (backend == GLM5_BACKEND_METAL) {
        fill_metal_backend_caps(out_caps);
    } else if (backend == GLM5_BACKEND_VULKAN) {
        fill_basic_gpu_caps(out_caps, compiled_vulkan_available(), &out_caps->supports_vulkan);
    } else if (backend == GLM5_BACKEND_DIRECTML) {
        fill_basic_gpu_caps(out_caps, compiled_directml_available(), &out_caps->supports_directml);
    } else if (backend == GLM5_BACKEND_OPENCL) {
        fill_basic_gpu_caps(out_caps, compiled_opencl_available(), &out_caps->supports_opencl);
    } else if (backend == GLM5_BACKEND_WEBGPU) {
        fill_webgpu_backend_caps(out_caps);
    } else {
        out_caps->supports_pinned_host = 1;
    }
}

