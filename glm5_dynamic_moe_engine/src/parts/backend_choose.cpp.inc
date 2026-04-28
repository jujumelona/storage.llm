static glm5_platform_t default_platform() {
#ifdef _WIN32
    return GLM5_PLATFORM_WINDOWS_PC;
#elif defined(__APPLE__)
    return GLM5_PLATFORM_MACOS_APPLE;
#elif defined(__linux__)
    return GLM5_PLATFORM_LINUX_PC;
#else
    return GLM5_PLATFORM_CPU_ONLY;
#endif
}

static glm5_backend_t choose_backend(glm5_backend_t preferred, glm5_platform_t platform) {
    if (preferred != GLM5_BACKEND_AUTO) {
        return preferred;
    }
    if (platform == GLM5_PLATFORM_AUTO) {
        platform = default_platform();
    }
    if (platform == GLM5_PLATFORM_MACOS_APPLE && compiled_metal_available()) {
        return GLM5_BACKEND_METAL;
    }
    if ((platform == GLM5_PLATFORM_WINDOWS_PC || platform == GLM5_PLATFORM_LINUX_PC) && compiled_cuda_available()) {
        return GLM5_BACKEND_CUDA;
    }
    if ((platform == GLM5_PLATFORM_WINDOWS_PC || platform == GLM5_PLATFORM_LINUX_PC) && compiled_hip_available()) {
        return GLM5_BACKEND_HIP;
    }
    if ((platform == GLM5_PLATFORM_WINDOWS_PC || platform == GLM5_PLATFORM_LINUX_PC) && compiled_level_zero_available()) {
        return GLM5_BACKEND_LEVEL_ZERO;
    }
    if ((platform == GLM5_PLATFORM_WINDOWS_PC || platform == GLM5_PLATFORM_LINUX_PC) && compiled_sycl_available()) {
        return GLM5_BACKEND_SYCL;
    }
    if (platform == GLM5_PLATFORM_WINDOWS_PC && compiled_directml_available()) {
        return GLM5_BACKEND_DIRECTML;
    }
    if (compiled_vulkan_available()) {
        return GLM5_BACKEND_VULKAN;
    }
    if (compiled_opencl_available()) {
        return GLM5_BACKEND_OPENCL;
    }
    if (compiled_webgpu_available()) {
        return GLM5_BACKEND_WEBGPU;
    }
    return GLM5_BACKEND_CPU;
}

