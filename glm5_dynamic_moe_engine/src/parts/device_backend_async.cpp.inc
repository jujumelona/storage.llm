// Backend-neutral async copy facade.
// CUDA is the only implemented device allocator today. Other GPU backends must
// populate this facade before they can claim async upload support.

static int device_backend_is_cuda(const glm5_pc_engine* e) {
    return e && e->backend_caps.backend == GLM5_BACKEND_CUDA && e->cuda.library;
}

static int device_backend_stream_create(glm5_pc_engine* e, CUstream* out_stream, unsigned int flags) {
    if (out_stream) {
        *out_stream = nullptr;
    }
    if (!out_stream || !device_backend_is_cuda(e) || !e->cuda.cuStreamCreate) {
        return 0;
    }
    return e->cuda.cuStreamCreate(out_stream, flags) == 0;
}

static int device_backend_stream_sync(glm5_pc_engine* e, CUstream stream) {
    if (!stream) {
        return 1;
    }
    if (!device_backend_is_cuda(e) || !e->cuda.cuStreamSynchronize) {
        return 0;
    }
    return e->cuda.cuStreamSynchronize(stream) == 0;
}

static void device_backend_stream_destroy(glm5_pc_engine* e, CUstream stream) {
    if (stream && device_backend_is_cuda(e) && e->cuda.cuStreamDestroy) {
        e->cuda.cuStreamDestroy(stream);
    }
}

static int device_backend_async_h2d_available(glm5_pc_engine* e, CUstream stream, int host_pinned) {
    return device_backend_is_cuda(e) &&
        stream &&
        host_pinned &&
        e->cuda.cuMemcpyHtoDAsync;
}

static int device_backend_copy_h2d_async(
    glm5_pc_engine* e,
    CUdeviceptr dst,
    const void* src,
    uint64_t bytes,
    CUstream stream
) {
    if (!device_backend_is_cuda(e) || !e->cuda.cuMemcpyHtoDAsync || !stream) {
        return 0;
    }
    return e->cuda.cuMemcpyHtoDAsync(dst, src, (size_t)bytes, stream) == 0;
}

static int device_backend_copy_h2d_sync(
    glm5_pc_engine* e,
    CUdeviceptr dst,
    const void* src,
    uint64_t bytes
) {
    if (!device_backend_is_cuda(e) || !e->cuda.cuMemcpyHtoD) {
        return 0;
    }
    return e->cuda.cuMemcpyHtoD(dst, src, (size_t)bytes) == 0;
}
