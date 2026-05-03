#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void* metal_zero_copy_map(void* device_handle, void* src, uint64_t bytes);
void* metal_zero_copy_map_aligned(void* device_handle, void* src, uint64_t bytes, uint64_t* out_prefix);
void metal_zero_copy_unmap(void* buffer);
void* metal_buffer_alloc(void* device_handle, uint64_t bytes);
int metal_copy_h2d_async(void* dst_buffer, const void* src, uint64_t bytes, void* stream);
int metal_copy_h2d_sync(void* dst_buffer, const void* src, uint64_t bytes);

// Metal Async Stream & Event support
void* metal_stream_create(void* device_handle);
void metal_stream_destroy(void* stream);
void* metal_event_create();
void metal_event_destroy(void* event);
int metal_event_record(void* event, void* stream);
int metal_event_query(void* event);
int metal_event_sync(void* event);

#ifdef __cplusplus
}
#endif
