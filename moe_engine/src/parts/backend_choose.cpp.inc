static moe_platform_t default_platform() {
#ifdef _WIN32
    return moe_PLATFORM_WINDOWS_PC;
#elif defined(__APPLE__)
    return moe_PLATFORM_MACOS_APPLE;
#elif defined(__linux__)
    return moe_PLATFORM_LINUX_PC;
#else
    return moe_PLATFORM_CPU_ONLY;
#endif
}

static moe_backend_t choose_backend(moe_backend_t preferred, moe_platform_t platform) {
    if (preferred != moe_BACKEND_AUTO) {
        return preferred;
    }
    if (platform == moe_PLATFORM_AUTO) {
        platform = default_platform();
    }
    if (platform == moe_PLATFORM_MACOS_APPLE && compiled_metal_available()) {
        return moe_BACKEND_METAL;
    }
    if ((platform == moe_PLATFORM_WINDOWS_PC || platform == moe_PLATFORM_LINUX_PC) && compiled_cuda_available()) {
        return moe_BACKEND_CUDA;
    }
    if ((platform == moe_PLATFORM_WINDOWS_PC || platform == moe_PLATFORM_LINUX_PC) && compiled_hip_available()) {
        return moe_BACKEND_HIP;
    }
    if ((platform == moe_PLATFORM_WINDOWS_PC || platform == moe_PLATFORM_LINUX_PC) && compiled_level_zero_available()) {
        return moe_BACKEND_LEVEL_ZERO;
    }
    if ((platform == moe_PLATFORM_WINDOWS_PC || platform == moe_PLATFORM_LINUX_PC) && compiled_sycl_available()) {
        return moe_BACKEND_SYCL;
    }
    if (platform == moe_PLATFORM_WINDOWS_PC && compiled_directml_available()) {
        return moe_BACKEND_DIRECTML;
    }
    if (compiled_vulkan_available()) {
        return moe_BACKEND_VULKAN;
    }
    if (compiled_opencl_available()) {
        return moe_BACKEND_OPENCL;
    }
    if (compiled_webgpu_available()) {
        return moe_BACKEND_WEBGPU;
    }
    return moe_BACKEND_CPU;
}

