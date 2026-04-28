const char* glm5_backend_name(glm5_backend_t backend) {
    switch (backend) {
        case GLM5_BACKEND_AUTO:
            return "auto";
        case GLM5_BACKEND_CPU:
            return "cpu";
        case GLM5_BACKEND_CUDA:
            return "cuda";
        case GLM5_BACKEND_METAL:
            return "metal";
        case GLM5_BACKEND_VULKAN:
            return "vulkan";
        case GLM5_BACKEND_DIRECTML:
            return "directml";
        case GLM5_BACKEND_OPENCL:
            return "opencl";
        case GLM5_BACKEND_HIP:
            return "hip";
        case GLM5_BACKEND_LEVEL_ZERO:
            return "level_zero";
        case GLM5_BACKEND_SYCL:
            return "sycl";
        case GLM5_BACKEND_WEBGPU:
            return "webgpu";
        default:
            return "unknown";
    }
}

const char* glm5_platform_name(glm5_platform_t platform) {
    switch (platform) {
        case GLM5_PLATFORM_AUTO:
            return "auto";
        case GLM5_PLATFORM_WINDOWS_PC:
            return "windows_pc";
        case GLM5_PLATFORM_LINUX_PC:
            return "linux_pc";
        case GLM5_PLATFORM_MACOS_APPLE:
            return "macos_apple";
        case GLM5_PLATFORM_CPU_ONLY:
            return "cpu_only";
        default:
            return "unknown";
    }
}
