const char* moe_backend_name(moe_backend_t backend) {
    switch (backend) {
        case moe_BACKEND_AUTO:
            return "auto";
        case moe_BACKEND_CPU:
            return "cpu";
        case moe_BACKEND_CUDA:
            return "cuda";
        case moe_BACKEND_METAL:
            return "metal";
        case moe_BACKEND_VULKAN:
            return "vulkan";
        case moe_BACKEND_DIRECTML:
            return "directml";
        case moe_BACKEND_OPENCL:
            return "opencl";
        case moe_BACKEND_HIP:
            return "hip";
        case moe_BACKEND_LEVEL_ZERO:
            return "level_zero";
        case moe_BACKEND_SYCL:
            return "sycl";
        case moe_BACKEND_WEBGPU:
            return "webgpu";
        default:
            return "unknown";
    }
}

const char* moe_platform_name(moe_platform_t platform) {
    switch (platform) {
        case moe_PLATFORM_AUTO:
            return "auto";
        case moe_PLATFORM_WINDOWS_PC:
            return "windows_pc";
        case moe_PLATFORM_LINUX_PC:
            return "linux_pc";
        case moe_PLATFORM_MACOS_APPLE:
            return "macos_apple";
        case moe_PLATFORM_CPU_ONLY:
            return "cpu_only";
        default:
            return "unknown";
    }
}
