#import <Metal/Metal.h>
#include "metal_adapter.h"

extern "C" {

void* metal_zero_copy_map(void* src, uint64_t bytes) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device || !src || bytes == 0) return nullptr;
    
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
