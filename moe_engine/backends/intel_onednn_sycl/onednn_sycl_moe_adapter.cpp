#include "moe_pc_engine.h"

extern "C" int storagellm_onednn_sycl_grouped_moe_indexed_device_f32(
    int32_t backend,
    const moe_grouped_expert_device_task_t* tasks,
    uint32_t task_count,
    void* stream_or_queue
) {
    (void)backend;
    (void)tasks;
    (void)task_count;
    (void)stream_or_queue;
    // Stub: real oneDNN SYCL grouped MoE adapter must return 1.
    return 0;
}
