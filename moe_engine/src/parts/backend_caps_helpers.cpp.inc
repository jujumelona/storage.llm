static uint32_t caps_hw_threads() {
    const unsigned int hw = std::thread::hardware_concurrency();
    return hw ? static_cast<uint32_t>(hw) : 1u;
}

static uint64_t normal_bundle_max_or(uint64_t fallback) {
    const moe_storage_constants_t* storage = moe_storage_constants();
    return storage && storage->normal_bundle_max_bytes ? storage->normal_bundle_max_bytes : fallback;
}

static uint64_t cache_for_experts(uint32_t expert_count) {
    return normal_bundle_max_or(32ull * 1024ull * 1024ull) * expert_count;
}

static uint64_t cache_with_common_for_experts(uint32_t expert_count) {
    return working_common_cache_bytes() + cache_for_experts(expert_count);
}

static uint64_t moe_budget_from_free_device_memory(uint64_t free_bytes) {
    return (free_bytes * 9ull) / 10ull;
}

static void fill_cpu_backend_caps(moe_backend_caps_t* caps) {
    const uint32_t hw = caps_hw_threads();
    caps->dma_bandwidth_gbps = 12.0;
    caps->tflops_hint = 0.10;
    caps->supports_fp4 = 1;
    caps->supports_scale4 = 1;
    caps->supports_weight_q2 = 1;
    caps->supports_weight_q3 = 1;
    caps->supports_weight_q4 = 1;
    caps->supports_weight_q8 = 1;
    caps->supports_weight_iq2 = 0;
    caps->supports_weight_mxfp4 = 0;
    caps->supports_unified_memory = 1;
    caps->supports_zero_copy_host = 1;
    caps->supports_zero_copy_mmap = 1;
    caps->supports_pinned_host = 0;
    caps->supports_backend_async_api = 0;
    caps->compute_streams = std::min<uint32_t>(hw, 8u);
    caps->copy_streams = 0;
    caps->prefetch_streams = std::min<uint32_t>(std::max<uint32_t>(hw / 2u, 1u), 8u);
    caps->recommended_pinned_bytes = 0;
}

static void fill_gpu_backend_base(moe_backend_caps_t* caps) {
    const uint32_t hw = caps_hw_threads();
    caps->dma_bandwidth_gbps = 25.0;
    caps->tflops_hint = 20.0;
    caps->compute_streams = hw >= 12u ? 2u : 1u;
    caps->copy_streams = hw >= 12u ? 2u : 1u;
    caps->prefetch_streams = hw >= 16u ? 3u : 2u;
    caps->supports_async_copy = 1;
    caps->supports_fp4 = compiled_nvfp4_device_kernels_available();
    caps->supports_scale4 = compiled_scale4_device_fusion_available();
    caps->supports_weight_q2 = 0;
    caps->supports_weight_q3 = 0;
    caps->supports_weight_q4 = caps->supports_fp4;
    caps->supports_weight_q8 = 0;
    caps->supports_weight_iq2 = 0;
    caps->supports_weight_mxfp4 = 0;
    caps->recommended_vram_common_bytes = working_common_cache_bytes();
    caps->recommended_pinned_bytes = std::max<uint64_t>(512ull * 1024ull * 1024ull, cache_for_experts(12));
}

static int cuda_driver_get_numa_node(uint32_t device_id);
static void fill_cuda_backend_caps(moe_backend_caps_t* caps) {
    caps->preferred_numa_node = cuda_driver_get_numa_node(caps->device_id);
    caps->supports_cuda = compiled_cuda_available();
    caps->supports_pinned_host = 1;
    caps->supports_fixed_read_staging = 1;
    caps->supports_registered_io_buffers = 1; // Enabled IORing/io_uring
    caps->supports_backend_async_api = 1;
    caps->supports_directstorage = compiled_directstorage_available();
    caps->supports_gpudirect_storage = compiled_gpudirect_available();
    caps->supports_gpu_direct_file_io =
        caps->supports_directstorage || caps->supports_gpudirect_storage || compiled_bam_nvme_available();
    caps->recommended_vram_cache_bytes = cache_with_common_for_experts(64);
    uint64_t total_bytes = 0;
    const uint64_t free_bytes = cuda_driver_detect_free_memory_bytes(caps->device_id, &total_bytes);

    // BUGFIX 130: Issue 33 - Validate VRAM query for driver bugs (WSL2/old drivers) ★★
    // Problem: cuDeviceTotalMem() on some old drivers (CUDA 11.x) or WSL2 returns
    // virtual address space size instead of physical VRAM.
    // device_memory_bytes set too high → over-admit experts → OOM.
    // Solution: Validate free vs total ratio. If free < 10% of total, likely driver bug.
    // Use recommended_vram_cache_bytes as fallback.
    if (free_bytes > 0) {
        // Sanity check: free should be at least 10% of total
        if (total_bytes > 0 && free_bytes < total_bytes / 10) {
            fprintf(stderr, "[storageLLM WARNING] VRAM query suspicious (free=%llu MB, total=%llu MB). "
                    "Likely driver bug (WSL2/old CUDA). Using fallback budget.\n",
                    (unsigned long long)(free_bytes / (1024*1024)),
                    (unsigned long long)(total_bytes / (1024*1024)));
            caps->device_memory_bytes = caps->recommended_vram_cache_bytes;
        } else {
            caps->device_memory_bytes = free_bytes;
            caps->recommended_vram_cache_bytes = std::max<uint64_t>(
                caps->recommended_vram_cache_bytes,
                moe_budget_from_free_device_memory(free_bytes)
            );
        }
    } else if (total_bytes > 0) {
        caps->device_memory_bytes = total_bytes;
    }
}

static int hip_driver_get_numa_node(uint32_t device_id);
static void fill_hip_backend_caps(moe_backend_caps_t* caps) {
    caps->preferred_numa_node = hip_driver_get_numa_node(caps->device_id);
    caps->supports_hip = compiled_hip_available();
    caps->supports_pinned_host = 1;
    caps->supports_fixed_read_staging = 1;
    caps->supports_backend_async_api = 1;
    caps->supports_directstorage = compiled_directstorage_available();
    caps->supports_gpu_direct_file_io = caps->supports_directstorage;
    caps->recommended_vram_cache_bytes = cache_with_common_for_experts(48);

    uint64_t total_bytes = 0;
    const uint64_t free_bytes = hip_driver_detect_free_memory_bytes(caps->device_id, &total_bytes);

    // BUGFIX 130: Same validation for HIP
    if (free_bytes > 0) {
        if (total_bytes > 0 && free_bytes < total_bytes / 10) {
            fprintf(stderr, "[storageLLM WARNING] HIP VRAM query suspicious (free=%llu MB, total=%llu MB). "
                    "Using fallback budget.\n",
                    (unsigned long long)(free_bytes / (1024*1024)),
                    (unsigned long long)(total_bytes / (1024*1024)));
            caps->device_memory_bytes = caps->recommended_vram_cache_bytes;
        } else {
            caps->device_memory_bytes = free_bytes;
            caps->recommended_vram_cache_bytes = std::max<uint64_t>(
                caps->recommended_vram_cache_bytes,
                moe_budget_from_free_device_memory(free_bytes)
            );
        }
    } else if (total_bytes > 0) {
        caps->device_memory_bytes = total_bytes;
    }

    // ============================================================================
    // AMD MI300X / MI300A Unified Memory Support
    // ============================================================================
    // Enable zero-copy unified memory pathways to bypass PCIe bottlenecks on MI300.
    // ============================================================================
    if (total_bytes >= 120ull * 1024ull * 1024ull * 1024ull) {
        caps->supports_unified_memory = 1;
        caps->supports_zero_copy_host = 1;
        caps->tflops_hint = 163.0;
        caps->dma_bandwidth_gbps = 800.0;
    }
}

static void fill_shared_memory_backend_caps(moe_backend_caps_t* caps) {
    caps->supports_pinned_host = 1;
    if (caps->device_id == 0) {
        caps->supports_unified_memory = 1;
        caps->supports_zero_copy_host = 1;
        caps->supports_zero_copy_mmap = 1;
    } else {
        caps->supports_unified_memory = 0;
        caps->supports_zero_copy_host = 0;
        caps->supports_zero_copy_mmap = 0;
    }
    caps->recommended_vram_cache_bytes = cache_with_common_for_experts(48);
}

// TODO: Issue 4 - Metal zero-copy optimization with MTLBuffer.
// Current: Metal uses supports_zero_copy_host=1 ??mmap bypass (BUGFIX 61 verified correct)
// Problem: Still copies mmap data to Metal buffer in some paths
// Apple Silicon Unified Memory: CPU and GPU share physical memory
// mmap weights can be read directly by GPU (no H2D copy needed)
// Solution: Use MTLBuffer with newBufferWithBytesNoCopy(mmap_ptr, size, MTLResourceStorageModeShared)
// - GPU reads mmap pointer directly, 0 bytes DMA
// - Eliminates all H2D copy overhead on Metal
// Implementation:
//   1. Add Metal API bindings (Objective-C++ wrapper)
//   2. In device_backend_async.cpp.inc, add Metal branch:
//      if (backend == moe_BACKEND_METAL) {
//          return metal_zero_copy_map(e, src, bytes);
//      }
//   3. metal_zero_copy_map creates MTLBuffer with newBufferWithBytesNoCopy
//   4. Return buffer handle as CUdeviceptr equivalent
// Trade-off: Metal API dependency, iOS/macOS-specific code, build system changes
// Status: Documented as TODO, current bypass path is correct (BUGFIX 61)
static void fill_metal_backend_caps(moe_backend_caps_t* caps) {
    caps->supports_metal = compiled_metal_available();
    caps->supports_unified_memory = 1;
    caps->supports_zero_copy_host = 1;
    caps->supports_zero_copy_mmap = 1;
    caps->supports_pinned_host = 0;
    caps->supports_backend_async_api = 1;
    caps->supports_gpu_direct_file_io = 0;
    caps->recommended_vram_cache_bytes = cache_with_common_for_experts(32);
    caps->tflops_hint = 14.0;
    caps->recommended_pinned_bytes = 0;
}

static void fill_basic_gpu_caps(moe_backend_caps_t* caps, int available, int* feature_flag) {
    *feature_flag = available;
    caps->supports_pinned_host = 1;
    caps->supports_fixed_read_staging = 1;
    caps->supports_backend_async_api = 0;
    caps->recommended_vram_cache_bytes = cache_with_common_for_experts(32);
    caps->tflops_hint = 14.0;
}

static void fill_webgpu_backend_caps(moe_backend_caps_t* caps) {
    caps->supports_webgpu = compiled_webgpu_available();
    caps->supports_pinned_host = 0;
    caps->supports_backend_async_api = 0;
    caps->recommended_vram_cache_bytes = cache_with_common_for_experts(16);
    caps->recommended_pinned_bytes = 0;
}
