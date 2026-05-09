#include "cpu_parallel.h"
#include "cpu_tensor_view.h"
#include "moe_pc_engine.h"

#include <cstdint>

extern "C" int storagellm_onednn_cpu_grouped_moe_indexed_device_f32(
    int32_t backend,
    const moe_grouped_expert_device_task_t* tasks,
    uint32_t task_count,
    void* stream_or_queue
) {
    (void)stream_or_queue;
    if (backend != moe_BACKEND_CPU || !tasks || task_count == 0) {
        return 0;
    }
    for (uint32_t i = 0; i < task_count; ++i) {
        storagellm::cpu_moe::ValidatedTask validated{};
        if (!storagellm::cpu_moe::validate_task(tasks[i], validated)) {
            return 0;
        }
        storagellm::cpu_moe::run_task_parallel(validated);
    }
    return 1;
}
