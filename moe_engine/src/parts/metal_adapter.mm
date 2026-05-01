#import <Metal/Metal.h>
#include "metal_adapter.h"

extern "C" {

void* metal_zero_copy_map(void* device_handle, void* src, uint64_t bytes) {
    id<MTLDevice> device = device_handle ? (__bridge id<MTLDevice>)device_handle : MTLCreateSystemDefaultDevice();
    if (!device || !src || bytes == 0) return nullptr;
    
    // Fix 1: newBufferWithBytesNoCopy requires page-aligned pointer and length
    NSUInteger pageSize = [NSProcessInfo processInfo].pageSize;
    if ((uintptr_t)src % pageSize != 0 || bytes % pageSize != 0) return nullptr;
    
    // Create an MTLBuffer that wraps the existing mmap pointer without copying
    id<MTLBuffer> buffer = [device newBufferWithBytesNoCopy:src
                                                     length:bytes
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

}

