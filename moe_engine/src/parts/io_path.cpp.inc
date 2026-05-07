static moe_io_path_t choose_active_io_path(const moe_pc_engine* engine) {
    if (!engine) {
        return moe_IO_PATH_MMAP_PAGECACHE;
    }
    if (engine->backend_caps.backend == moe_BACKEND_CUDA &&
        engine->runtime_gpudirect_ready.load(std::memory_order_acquire) &&
        engine->juju_bundle_4k_aligned.load(std::memory_order_acquire)) {
        return moe_IO_PATH_GPUDIRECT_STORAGE;
    }
    if (engine->runtime_directstorage_gpu_ready.load(std::memory_order_acquire)) {
        return moe_IO_PATH_DIRECTSTORAGE_GPU;
    }
    if (engine->runtime_directstorage_staging_ready.load(std::memory_order_acquire)) {
        return moe_IO_PATH_DIRECTSTORAGE_STAGING;
    }
    if (engine->true_zero_copy_validated.load(std::memory_order_acquire)) {
        return moe_IO_PATH_ZERO_COPY_HOST;
    }
    if (engine->backend_caps.backend == moe_BACKEND_CUDA &&
        engine->backend_caps.supports_registered_io_buffers &&
        engine->io_config.enable_bam_nvme && compiled_bam_nvme_available()) {
        return moe_IO_PATH_BAM_NVME;
    }
    if (engine->staging && engine->staging_pinned) {
        return moe_IO_PATH_PINNED_HOST;
    }
    if (engine->backend_caps.supports_pinned_host) {
        return moe_IO_PATH_PINNED_HOST;
    }
    return moe_IO_PATH_MMAP_PAGECACHE;
}

static int engine_bypass_staging_pipeline(const moe_pc_engine* engine) {
    if (!engine) return 0;
    if (engine->backend_caps.backend == moe_BACKEND_METAL && engine->backend_caps.supports_unified_memory) {
        return !(engine->io_config.enable_pinned_host && engine->io_config.pinned_staging_bytes > 0);
    }
    if (engine->true_zero_copy_validated.load(std::memory_order_acquire)) return 1;
    return 0;
}

static int engine_has_device_upload_workers(const moe_pc_engine* engine) {
    if (!engine || !engine->config.prefer_gpu) {
        return 0;
    }
    // Metal unified memory has no discrete upload stage.
    if (engine->backend_caps.backend == moe_BACKEND_METAL &&
        engine->backend_caps.supports_unified_memory) {
        return 0;
    }
    if (engine->backend_caps.backend == moe_BACKEND_LEVEL_ZERO &&
        engine->backend_caps.supports_backend_async_api) {
        return 1;
    }
    if (engine->backend_caps.backend == moe_BACKEND_VULKAN &&
        engine->backend_caps.supports_backend_async_api) {
        return 1;
    }
    if (engine->backend_caps.backend == moe_BACKEND_DIRECTML &&
        engine->runtime_directstorage_gpu_ready.load(std::memory_order_acquire)) {
        return 1;
    }
    return (engine->backend_caps.backend == moe_BACKEND_CUDA ||
            engine->backend_caps.backend == moe_BACKEND_HIP) &&
           engine->backend_caps.supports_backend_async_api;
}

static int caps_gpu_backend_available(const moe_backend_caps_t& caps) {
    switch (caps.backend) {
        case moe_BACKEND_CUDA:
            return caps.supports_cuda;
        case moe_BACKEND_METAL:
            return caps.supports_metal;
        case moe_BACKEND_VULKAN:
            return caps.supports_vulkan;
        case moe_BACKEND_DIRECTML:
            return caps.supports_directml;
        case moe_BACKEND_OPENCL:
            return caps.supports_opencl;
        case moe_BACKEND_HIP:
            return caps.supports_hip;
        case moe_BACKEND_LEVEL_ZERO:
            return caps.supports_level_zero;
        case moe_BACKEND_SYCL:
            return caps.supports_sycl;
        case moe_BACKEND_WEBGPU:
            return caps.supports_webgpu;
        default:
            return 0;
    }
}
