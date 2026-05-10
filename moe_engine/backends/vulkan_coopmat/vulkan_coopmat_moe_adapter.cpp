#include "moe_pc_engine.h"

extern "C" int storagellm_vulkan_coopmat_grouped_moe_indexed_device_f32(
    int32_t backend,
    const moe_grouped_expert_device_task_t* tasks,
    uint32_t task_count,
    void* stream_or_queue
) {
    (void)backend;
    (void)tasks;
    (void)task_count;
    (void)stream_or_queue;
    // Stub: real Vulkan cooperative-matrix MoE adapter must return 1.
    return 0;
}
