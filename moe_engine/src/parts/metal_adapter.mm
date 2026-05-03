#import <Metal/Metal.h>
#include "metal_adapter.h"
#include <string.h>

extern "C" {

void* metal_zero_copy_map(void* device_handle, void* src, uint64_t bytes) {
    uint64_t prefix = 0;
    void* mapped = metal_zero_copy_map_aligned(device_handle, src, bytes, &prefix);
    NSUInteger pageSize = [NSProcessInfo processInfo].pageSize;
    if (prefix != 0 || (bytes % (uint64_t)pageSize) != 0) {
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

    // Create an MTLBuffer that wraps the existing mmap pointer without copying
    id<MTLBuffer> buffer = [device newBufferWithBytesNoCopy:(void*)aligned_src
                                                     length:(NSUInteger)aligned_bytes
                                                    options:MTLResourceStorageModeShared
                                                deallocator:nil];
    // Return the buffer as a void pointer (bridged to void* to pass through C-boundary)
    return (void*)CFBridgingRetain(buffer);
}

void metal_zero_copy_unmap(void* buffer) {
    if (buffer) {
        id<MTLBuffer> mtlBuffer = (id<MTLBuffer>)CFBridgingRelease(buffer);
        mtlBuffer = nil; // ARC will deallocate
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

    id<MTLBuffer> srcBuffer = [device newBufferWithBytes:src
                                                  length:(NSUInteger)bytes
                                                 options:MTLResourceStorageModeShared];
    if (!srcBuffer) return 0;
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    if (!commandBuffer) return 0;
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    if (!blit) return 0;
    [blit copyFromBuffer:srcBuffer
            sourceOffset:0
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

// Metal Async Callbacks
void* metal_stream_create(void* device_handle) {
    id<MTLDevice> device = device_handle ? (__bridge id<MTLDevice>)device_handle : MTLCreateSystemDefaultDevice();
    if (!device) return nullptr;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    return (void*)CFBridgingRetain(queue);
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

    // Create a dummy command buffer just to track completion
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    if (!commandBuffer) return 0;

    ev->completed = false;

    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
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
    // put token back since wait decrements it, allowing multiple syncs
    dispatch_semaphore_signal(ev->sem);
    return 1;
}

}

