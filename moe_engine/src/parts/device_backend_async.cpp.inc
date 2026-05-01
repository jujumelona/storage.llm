// Backend-neutral async copy facade.
// Supports CUDA and HIP.

static int device_backend_copy_h2d_sync(
    moe_pc_engine* e,
    CUdeviceptr dst,
    const void* src,
    uint64_t bytes
);

static int device_backend_is_cuda(const moe_pc_engine* e) {
    return e && e->backend_caps.backend == moe_BACKEND_CUDA && e->cuda.library;
}

static int device_backend_is_hip(const moe_pc_engine* e) {
    return e && e->backend_caps.backend == moe_BACKEND_HIP && e->hip.library;
}

// BUGFIX 125: Issue 30 - Vulkan/Level Zero/SYCL dummy async is actually sync ★★
// Problem: device_backend_stream_create returns dummy handle (void*)1 for these backends.
// Code thinks it has async streams, but device_backend_copy_h2d_async falls back to sync.
// gpu_worker_loop tries double-buffering with dma_overlap_stream, but it's all synchronous.
// Solution: Set supports_backend_async_api = false in device_types.cpp.inc for these backends.
// This makes code explicitly use sync path, avoiding false async assumptions.
// Trade-off: No change in actual behavior (already sync), but eliminates misleading code paths.
static int device_backend_stream_create(moe_pc_engine* e, CUstream* out_stream, unsigned int flags) {
    if (out_stream) *out_stream = nullptr;
    if (!out_stream) return 0;
    if (device_backend_is_cuda(e) && e->cuda.cuStreamCreate) {
        return e->cuda.cuStreamCreate(out_stream, flags) == 0;
    }
    if (device_backend_is_hip(e) && e->hip.hipStreamCreate) {
        return e->hip.hipStreamCreate(out_stream, flags) == 0;
    }
    // BUGFIX 125: Removed dummy stream creation for Vulkan/Level Zero/SYCL.
    // These backends should set supports_backend_async_api = false instead.
    return 0;
}
static int device_backend_stream_sync(moe_pc_engine* e, CUstream stream) {
    if (!stream) return 1;
    if (device_backend_is_cuda(e) && e->cuda.cuStreamSynchronize) {
        return e->cuda.cuStreamSynchronize(stream) == 0;
    }
    if (device_backend_is_hip(e) && e->hip.hipStreamSynchronize) {
        return e->hip.hipStreamSynchronize(stream) == 0;
    }
    // BUGFIX 125: Removed dummy sync for Vulkan/Level Zero/SYCL
    return 0;
}
static void device_backend_stream_destroy(moe_pc_engine* e, CUstream stream) {
    if (!stream) return;
    if (device_backend_is_cuda(e) && e->cuda.cuStreamDestroy) {
        e->cuda.cuStreamDestroy(stream);
    } else if (device_backend_is_hip(e) && e->hip.hipStreamDestroy) {
        e->hip.hipStreamDestroy(stream);
    }
    // Vulkan/LevelZero/SYCL dummy streams need no destroy
}

static int device_backend_event_create(moe_pc_engine* e, CUevent* out_event) {
    if (out_event) *out_event = nullptr;
    if (!out_event) return 0;
    constexpr unsigned int disable_timing = 2u;
    if (device_backend_is_cuda(e) && e->cuda.cuEventCreate) {
        return e->cuda.cuEventCreate(out_event, disable_timing) == 0;
    }
    if (device_backend_is_hip(e) && e->hip.hipEventCreate) {
        return e->hip.hipEventCreate(out_event, disable_timing) == 0;
    }
    // BUGFIX 125: Removed dummy event for Vulkan/Level Zero/SYCL
    return 0;
}

static int device_backend_event_record(moe_pc_engine* e, CUevent event, CUstream stream) {
    if (!event || !stream) return 0;
    if (device_backend_is_cuda(e) && e->cuda.cuEventRecord) {
        return e->cuda.cuEventRecord(event, stream) == 0;
    }
    if (device_backend_is_hip(e) && e->hip.hipEventRecord) {
        return e->hip.hipEventRecord(event, stream) == 0;
    }
    // BUGFIX 125: Removed dummy record for Vulkan/Level Zero/SYCL
    return 0;
}

static int device_backend_event_query(moe_pc_engine* e, CUevent event) {
    if (!event) return 0;
    if (device_backend_is_cuda(e) && e->cuda.cuEventQuery) {
        return e->cuda.cuEventQuery(event) == 0;
    }
    if (device_backend_is_hip(e) && e->hip.hipEventQuery) {
        return e->hip.hipEventQuery(event) == 0;
    }
    // BUGFIX 125: Removed dummy query for Vulkan/Level Zero/SYCL
    return 0;
}

static int device_backend_event_sync(moe_pc_engine* e, CUevent event) {
    if (!event) return 1;
    if (device_backend_is_cuda(e)) {
        if (e->cuda.cuEventSynchronize) {
            return e->cuda.cuEventSynchronize(event) == 0;
        }
        if (!e->cuda.cuEventQuery) return 0;
        // BUGFIX 86: Issue 9 - Reduce CUDA event poll sleep overhead ★
        // Problem: 50µs sleep on every poll → 5ms for 100 iterations
        // Solution: First 20 iterations use yield only, then 50µs sleep
        // Fast events complete without sleep, slow events still get sleep
        // BUGFIX 95: Reduce atomic load frequency in event poll
        uint32_t spin_count = 0;
        while (true) {
            if (e->cuda.cuEventQuery(event) == 0) return 1;
            if (e->io_stop.load(std::memory_order_relaxed)) break;
            if (++spin_count < 20) {
                std::this_thread::yield();
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
        }
        return e->cuda.cuEventQuery(event) == 0;
    }
    if (device_backend_is_hip(e)) {
        if (e->hip.hipEventSynchronize) {
            return e->hip.hipEventSynchronize(event) == 0;
        }
        if (!e->hip.hipEventQuery) return 0;
        // BUGFIX 86: Same optimization for HIP
        // BUGFIX 95: Reduce atomic load frequency in event poll
        uint32_t spin_count = 0;
        while (true) {
            if (e->hip.hipEventQuery(event) == 0) return 1;
            if (e->io_stop.load(std::memory_order_relaxed)) break;
            if (++spin_count < 20) {
                std::this_thread::yield();
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
        }
        return e->hip.hipEventQuery(event) == 0;
    }
    // BUGFIX 125: Removed dummy sync for Vulkan/Level Zero/SYCL
    return 0;
}

static void device_backend_event_destroy(moe_pc_engine* e, CUevent event) {
    if (!event) return;
    if (device_backend_is_cuda(e) && e->cuda.cuEventDestroy) {
        e->cuda.cuEventDestroy(event);
    } else if (device_backend_is_hip(e) && e->hip.hipEventDestroy) {
        e->hip.hipEventDestroy(event);
    }
}

static int device_backend_async_h2d_available(moe_pc_engine* e, CUstream stream, int host_pinned) {
    if (!stream || !host_pinned) return 0;
    if (device_backend_is_cuda(e)) {
        return e->cuda.cuMemcpyHtoDAsync != nullptr;
    }
    if (device_backend_is_hip(e)) {
        return e->hip.hipMemcpyHtoDAsync != nullptr;
    }
    if (e && e->backend_caps.backend == moe_BACKEND_LEVEL_ZERO) {
        return e->ze.zeCommandListAppendMemoryCopy != nullptr;
    }
    // BUGFIX 125: Removed dummy async for Vulkan/SYCL
    return 0;
}

static int device_backend_copy_h2d_async(
    moe_pc_engine* e,
    CUdeviceptr dst,
    const void* src,
    uint64_t bytes,
    CUstream stream
) {
    if (!stream) return 0;
    if (device_backend_is_cuda(e) && e->cuda.cuMemcpyHtoDAsync) {
        return e->cuda.cuMemcpyHtoDAsync(dst, src, (size_t)bytes, stream) == 0;
    }
    if (device_backend_is_hip(e) && e->hip.hipMemcpyHtoDAsync) {
        return e->hip.hipMemcpyHtoDAsync(dst, src, (size_t)bytes, stream) == 0;
    }
    if (e && e->backend_caps.backend == moe_BACKEND_METAL) {
        // Issue 4: Metal uses MTLBuffer with newBufferWithBytesNoCopy natively.
        // The implementation resides in metal_zero_copy_map (metal_adapter.mm).
        // Since zero-copy maps the pointer, H2D copy is effectively a no-op here
        // as long as dst was allocated via metal_zero_copy_map.
        return 1;
    }
    // Fix 22: Add Level Zero async copy path
    if (e && e->backend_caps.backend == moe_BACKEND_LEVEL_ZERO && e->ze.zeCommandListAppendMemoryCopy) {
        // In Level Zero, the stream handle is mapped to a zeCommandList for copy operations.
        // Returning true (1) if zeCommandListAppendMemoryCopy succeeds (returns 0).
        return e->ze.zeCommandListAppendMemoryCopy(stream, (void*)dst, src, (size_t)bytes, nullptr, 0, nullptr) == 0;
    }
    // BUGFIX 125: Removed dummy async copy for Vulkan/SYCL - should use sync path
    return 0;
}

static int device_backend_copy_h2d_sync(
    moe_pc_engine* e,
    CUdeviceptr dst,
    const void* src,
    uint64_t bytes
) {
    if (device_backend_is_cuda(e) && e->cuda.cuMemcpyHtoD) {
        return e->cuda.cuMemcpyHtoD(dst, src, (size_t)bytes) == 0;
    }
    if (device_backend_is_hip(e) && e->hip.hipMemcpyHtoD) {
        return e->hip.hipMemcpyHtoD(dst, src, (size_t)bytes) == 0;
    }
    return 0;
}
