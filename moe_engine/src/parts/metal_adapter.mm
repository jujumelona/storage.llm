#import <Metal/Metal.h>
#include "metal_adapter.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern "C" {

void* metal_zero_copy_map(void* device_handle, void* src, uint64_t bytes) {
    uint64_t prefix = 0;
    void* mapped = metal_zero_copy_map_aligned(device_handle, src, bytes, &prefix);
    if (prefix != 0) {
        if (mapped) {
            metal_zero_copy_unmap(mapped);
        }
        return nullptr;
    }
    return mapped;
}

void* metal_zero_copy_map_aligned(void* device_handle, void* src, uint64_t bytes, uint64_t* out_prefix) {
    if (out_prefix) *out_prefix = 0;
    id<MTLDevice> device = device_handle ? (__bridge id<MTLDevice>)device_handle : MTLCreateSystemDefaultDevice();
    if (!device || !src || bytes == 0) return nullptr;

    NSUInteger pageSize = [NSProcessInfo processInfo].pageSize;
    uintptr_t aligned_src = (uintptr_t)src & ~((uintptr_t)pageSize - 1u);
    uint64_t prefix = (uint64_t)((uintptr_t)src - aligned_src);
    uint64_t aligned_bytes = bytes + prefix;
    const uint64_t rem = aligned_bytes % (uint64_t)pageSize;
    if (rem) {
        aligned_bytes += (uint64_t)pageSize - rem;
    }
    if (out_prefix) *out_prefix = prefix;
    if (aligned_bytes < bytes) {
        return nullptr;
    }

    id<MTLBuffer> buffer = [device newBufferWithBytesNoCopy:(void*)aligned_src
                                                     length:(NSUInteger)aligned_bytes
                                                    options:MTLResourceStorageModeShared
                                                deallocator:nil];
    return buffer ? (void*)CFBridgingRetain(buffer) : nullptr;
}

void metal_zero_copy_unmap(void* buffer) {
    if (buffer) {
        id<MTLBuffer> mtlBuffer = (id<MTLBuffer>)CFBridgingRelease(buffer);
        mtlBuffer = nil;
    }
}

void* metal_buffer_alloc(void* device_handle, uint64_t bytes) {
    id<MTLDevice> device = device_handle ? (__bridge id<MTLDevice>)device_handle : MTLCreateSystemDefaultDevice();
    if (!device || bytes == 0 || bytes > (uint64_t)NSUIntegerMax) return nullptr;
    id<MTLBuffer> buffer = [device newBufferWithLength:(NSUInteger)bytes
                                               options:MTLResourceStorageModeShared];
    return buffer ? (void*)CFBridgingRetain(buffer) : nullptr;
}

static int metal_copy_h2d_sync_impl(void* dst_buffer, const void* src, uint64_t bytes) {
    if (!dst_buffer || !src || bytes == 0 || bytes > (uint64_t)NSUIntegerMax) return 0;
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)dst_buffer;
    if (!buffer || bytes > (uint64_t)[buffer length] || ![buffer contents]) return 0;
    memcpy([buffer contents], src, (size_t)bytes);
#if TARGET_OS_OSX
    [buffer didModifyRange:NSMakeRange(0, (NSUInteger)bytes)];
#endif
    return 1;
}

int metal_copy_h2d_async(void* dst_buffer, const void* src, uint64_t bytes, void* stream) {
    if (!dst_buffer || !src || !stream || bytes == 0 || bytes > (uint64_t)NSUIntegerMax) return 0;
    id<MTLBuffer> dst = (__bridge id<MTLBuffer>)dst_buffer;
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)stream;
    if (!dst || !queue || bytes > (uint64_t)[dst length]) return 0;
    id<MTLDevice> device = [dst device] ?: MTLCreateSystemDefaultDevice();
    if (!device) return 0;

    NSUInteger pageSize = [NSProcessInfo processInfo].pageSize;
    uintptr_t src_addr = (uintptr_t)src;
    uintptr_t aligned_src = src_addr & ~((uintptr_t)pageSize - 1u);
    uint64_t prefix = (uint64_t)(src_addr - aligned_src);

    id<MTLBuffer> srcBuffer = nil;
    NSUInteger srcOffset = 0;

    if (prefix == 0 && (bytes % pageSize == 0 || bytes + pageSize <= (uint64_t)NSUIntegerMax)) {
        uint64_t aligned_bytes = bytes;
        const uint64_t rem = aligned_bytes % (uint64_t)pageSize;
        if (rem) {
            aligned_bytes += (uint64_t)pageSize - rem;
        }
        srcBuffer = [device newBufferWithBytesNoCopy:(void*)src
                                              length:(NSUInteger)aligned_bytes
                                             options:MTLResourceStorageModeShared
                                         deallocator:nil];
        srcOffset = 0;
    }

    if (!srcBuffer) {
        srcBuffer = [device newBufferWithBytes:src
                                        length:(NSUInteger)bytes
                                       options:MTLResourceStorageModeShared];
        srcOffset = 0;
    }

    if (!srcBuffer) return 0;
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    if (!commandBuffer) return 0;
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    if (!blit) return 0;
    [blit copyFromBuffer:srcBuffer
            sourceOffset:srcOffset
                toBuffer:dst
       destinationOffset:0
                    size:(NSUInteger)bytes];
    [blit endEncoding];
    id<MTLBuffer> keepAliveSrc = srcBuffer;
    id<MTLBuffer> keepAliveDst = dst;
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
        (void)cb;
        (void)keepAliveSrc;
        (void)keepAliveDst;
    }];
    [commandBuffer commit];
    return 1;
}

int metal_copy_h2d_sync(void* dst_buffer, const void* src, uint64_t bytes) {
    return metal_copy_h2d_sync_impl(dst_buffer, src, bytes);
}

void* metal_stream_create(void* device_handle) {
    id<MTLDevice> device = device_handle ? (__bridge id<MTLDevice>)device_handle : MTLCreateSystemDefaultDevice();
    if (!device) return nullptr;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    return queue ? (void*)CFBridgingRetain(queue) : nullptr;
}

void metal_stream_destroy(void* stream) {
    if (stream) {
        id<MTLCommandQueue> queue = (id<MTLCommandQueue>)CFBridgingRelease(stream);
        queue = nil;
    }
}

struct MetalEvent {
    bool completed = false;
    dispatch_semaphore_t sem;
    MetalEvent() { sem = dispatch_semaphore_create(0); }
    ~MetalEvent() { }
};

void* metal_event_create() {
    return new MetalEvent();
}

void metal_event_destroy(void* event) {
    if (event) {
        delete (MetalEvent*)event;
    }
}

int metal_event_record(void* event, void* stream) {
    if (!event || !stream) return 0;
    MetalEvent* ev = (MetalEvent*)event;
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)stream;
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    if (!commandBuffer) return 0;

    ev->completed = false;
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
        (void)cb;
        ev->completed = true;
        dispatch_semaphore_signal(ev->sem);
    }];
    [commandBuffer commit];
    return 1;
}

int metal_event_query(void* event) {
    if (!event) return 1;
    MetalEvent* ev = (MetalEvent*)event;
    return ev->completed ? 1 : 0;
}

int metal_event_sync(void* event) {
    if (!event) return 1;
    MetalEvent* ev = (MetalEvent*)event;
    if (ev->completed) return 1;
    dispatch_semaphore_wait(ev->sem, DISPATCH_TIME_FOREVER);
    dispatch_semaphore_signal(ev->sem);
    return 1;
}

uint64_t metal_measure_h2d_bandwidth(void* device_handle) {
    id<MTLDevice> device = device_handle ? (__bridge id<MTLDevice>)device_handle : MTLCreateSystemDefaultDevice();
    if (!device) return 0;

    const size_t test_size = 100 * 1024 * 1024;
    void* host_buf = malloc(test_size);
    if (!host_buf) return 0;
    memset(host_buf, 0xAB, test_size);

    id<MTLBuffer> device_buf = [device newBufferWithLength:test_size
                                                   options:MTLResourceStorageModeShared];
    if (!device_buf) {
        free(host_buf);
        return 0;
    }

    for (int i = 0; i < 3; ++i) {
        memcpy([device_buf contents], host_buf, test_size);
#if TARGET_OS_OSX
        [device_buf didModifyRange:NSMakeRange(0, test_size)];
#endif
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < 10; ++i) {
        memcpy([device_buf contents], host_buf, test_size);
#if TARGET_OS_OSX
        [device_buf didModifyRange:NSMakeRange(0, test_size)];
#endif
    }
    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed_sec = (end.tv_sec - start.tv_sec) +
                        (end.tv_nsec - start.tv_nsec) / 1e9;

    uint64_t bytes_per_sec = 0;
    if (elapsed_sec > 0.0) {
        bytes_per_sec = (uint64_t)((test_size * 10.0 * 0.75) / elapsed_sec);
    }

    device_buf = nil;
    free(host_buf);
    return bytes_per_sec;
}

void* metal_get_default_device() {
    static id<MTLDevice> g_default_device = nil;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        g_default_device = MTLCreateSystemDefaultDevice();
        [g_default_device retain];
    });
    return (__bridge void*)g_default_device;
}

}
