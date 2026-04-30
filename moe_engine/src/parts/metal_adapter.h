#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void* metal_zero_copy_map(void* device_handle, void* src, uint64_t bytes);
void metal_zero_copy_unmap(void* buffer);

#ifdef __cplusplus
}
#endif
