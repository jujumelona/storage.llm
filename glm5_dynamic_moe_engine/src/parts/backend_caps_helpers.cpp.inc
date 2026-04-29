static uint32_t caps_hw_threads() {
    const unsigned int hw = std::thread::hardware_concurrency();
    return hw ? static_cast<uint32_t>(hw) : 1u;
}

static uint64_t normal_bundle_max_or(uint64_t fallback) {
    const glm5_storage_constants_t* storage = glm5_storage_constants();
    return storage && storage->normal_bundle_max_bytes ? storage->normal_bundle_max_bytes : fallback;
}

static uint64_t cache_for_experts(uint32_t expert_count) {
    return normal_bundle_max_or(32ull * 1024ull * 1024ull) * expert_count;
}

static uint64_t cache_with_common_for_experts(uint32_t expert_count) {
    return working_common_cache_bytes() + cache_for_experts(expert_count);
}

static uint64_t glm5_budget_from_free_device_memory(uint64_t free_bytes) {
    const uint64_t reserve = 512ull * 1024ull * 1024ull;
    if (free_bytes > reserve * 2ull) {
        return free_bytes - reserve;
    }
    return (free_bytes * 9ull) / 10ull;
}

static void fill_cpu_backend_caps(glm5_backend_caps_t* caps) {
    const uint32_t hw = caps_hw_threads();
    caps->supports_fp4 = 1;
    caps->supports_scale4 = 1;
    caps->compute_streams = std::min<uint32_t>(hw, 8u);
    caps->copy_streams = 0;
    caps->prefetch_streams = std::min<uint32_t>(std::max<uint32_t>(hw / 2u, 1u), 8u);
    caps->recommended_pinned_bytes = std::max<uint64_t>(256ull * 1024ull * 1024ull, cache_for_experts(8));
}

static void fill_gpu_backend_base(glm5_backend_caps_t* caps) {
    const uint32_t hw = caps_hw_threads();
    caps->compute_streams = hw >= 12u ? 2u : 1u;
    caps->copy_streams = hw >= 12u ? 2u : 1u;
    caps->prefetch_streams = hw >= 16u ? 3u : 2u;
    caps->supports_async_copy = 1;
    caps->supports_fp4 = compiled_nvfp4_device_kernels_available();
    caps->supports_scale4 = compiled_scale4_device_fusion_available();
    caps->recommended_vram_common_bytes = working_common_cache_bytes();
    caps->recommended_pinned_bytes = std::max<uint64_t>(512ull * 1024ull * 1024ull, cache_for_experts(12));
}

static void fill_cuda_backend_caps(glm5_backend_caps_t* caps) {
    caps->supports_cuda = compiled_cuda_available();
    caps->supports_pinned_host = 1;
    caps->supports_directstorage = compiled_directstorage_available();
    caps->supports_gpudirect_storage = compiled_gpudirect_available();
    caps->supports_gpu_direct_file_io =
        caps->supports_directstorage || caps->supports_gpudirect_storage || compiled_bam_nvme_available();
    caps->recommended_vram_cache_bytes = cache_with_common_for_experts(64);
    uint64_t total_bytes = 0;
    const uint64_t free_bytes = cuda_driver_detect_free_memory_bytes(caps->device_id, &total_bytes);
    if (free_bytes > 0) {
        caps->device_memory_bytes = free_bytes;
        caps->recommended_vram_cache_bytes = std::max<uint64_t>(
            caps->recommended_vram_cache_bytes,
            glm5_budget_from_free_device_memory(free_bytes)
        );
    } else if (total_bytes > 0) {
        caps->device_memory_bytes = total_bytes;
    }
}

static void fill_hip_backend_caps(glm5_backend_caps_t* caps) {
    caps->supports_hip = compiled_hip_available();
    caps->supports_pinned_host = 1;
    caps->supports_directstorage = compiled_directstorage_available();
    caps->supports_gpu_direct_file_io = caps->supports_directstorage;
    caps->recommended_vram_cache_bytes = cache_with_common_for_experts(48);
}

static void fill_shared_memory_backend_caps(glm5_backend_caps_t* caps) {
    caps->supports_pinned_host = 1;
    caps->supports_unified_memory = 1;
    caps->recommended_vram_cache_bytes = cache_with_common_for_experts(48);
}

static void fill_metal_backend_caps(glm5_backend_caps_t* caps) {
    caps->supports_metal = compiled_metal_available();
    caps->supports_unified_memory = 1;
    caps->supports_pinned_host = 0;
    caps->supports_gpu_direct_file_io = 0;
    caps->recommended_vram_cache_bytes = cache_with_common_for_experts(32);
    caps->recommended_pinned_bytes = 0;
}

static void fill_basic_gpu_caps(glm5_backend_caps_t* caps, int available, int* feature_flag) {
    *feature_flag = available;
    caps->supports_pinned_host = 1;
    caps->recommended_vram_cache_bytes = cache_with_common_for_experts(32);
}

static void fill_webgpu_backend_caps(glm5_backend_caps_t* caps) {
    caps->supports_webgpu = compiled_webgpu_available();
    caps->supports_pinned_host = 0;
    caps->recommended_vram_cache_bytes = cache_with_common_for_experts(16);
    caps->recommended_pinned_bytes = 0;
}

