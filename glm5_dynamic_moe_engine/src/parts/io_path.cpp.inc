static glm5_io_path_t choose_active_io_path(const glm5_pc_engine* engine) {
    if (!engine) {
        return GLM5_IO_PATH_MMAP_PAGECACHE;
    }
    if (engine->backend_caps.backend == GLM5_BACKEND_METAL) {
        return GLM5_IO_PATH_MMAP_PAGECACHE;
    }
    if (engine->io_config.enable_bam_nvme && compiled_bam_nvme_available()) {
        return GLM5_IO_PATH_BAM_NVME;
    }
    if (engine->io_config.enable_gpudirect_storage && compiled_gpudirect_available()) {
        return GLM5_IO_PATH_GPUDIRECT_STORAGE;
    }
    if (engine->io_config.enable_directstorage && compiled_directstorage_available()) {
        return GLM5_IO_PATH_DIRECTSTORAGE;
    }
    if (engine->staging && engine->staging_pinned) {
        return GLM5_IO_PATH_PINNED_HOST;
    }
    return GLM5_IO_PATH_MMAP_PAGECACHE;
}

static int caps_gpu_backend_available(const glm5_backend_caps_t& caps) {
    switch (caps.backend) {
        case GLM5_BACKEND_CUDA:
            return caps.supports_cuda;
        case GLM5_BACKEND_METAL:
            return caps.supports_metal;
        case GLM5_BACKEND_VULKAN:
            return caps.supports_vulkan;
        case GLM5_BACKEND_DIRECTML:
            return caps.supports_directml;
        case GLM5_BACKEND_OPENCL:
            return caps.supports_opencl;
        case GLM5_BACKEND_HIP:
            return caps.supports_hip;
        case GLM5_BACKEND_LEVEL_ZERO:
            return caps.supports_level_zero;
        case GLM5_BACKEND_SYCL:
            return caps.supports_sycl;
        case GLM5_BACKEND_WEBGPU:
            return caps.supports_webgpu;
        default:
            return 0;
    }
}

