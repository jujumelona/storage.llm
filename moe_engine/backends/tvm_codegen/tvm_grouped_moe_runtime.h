#pragma once

#include "moe_engine/include/parts/moe_fast_backend_types.h.inc"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct storagellm_tvm_grouped_moe_runtime {
    void* module;
    void* packed_func;
    int32_t backend;
} storagellm_tvm_grouped_moe_runtime_t;

int storagellm_tvm_codegen_grouped_moe_indexed_device_f32(
    int32_t backend,
    const moe_grouped_expert_device_task_t* tasks,
    uint32_t task_count,
    void* stream_or_queue);

#ifdef __cplusplus
}
#endif
