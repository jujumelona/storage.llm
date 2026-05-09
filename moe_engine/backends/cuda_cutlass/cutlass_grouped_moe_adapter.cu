#include "moe_pc_engine.h"

#if defined(STORAGELLM_HAS_CUTLASS)
#include <cuda_runtime_api.h>
#include <cutlass/cutlass.h>
#endif

struct moe_model_grouped_expert_runtime_task;

extern "C" int storagellm_cutlass_grouped_moe_runtime_f32(
    int32_t backend,
    const moe_model_grouped_expert_runtime_task* tasks,
    uint32_t task_count
) {
#if !defined(STORAGELLM_HAS_CUTLASS)
    (void)backend;
    (void)tasks;
    (void)task_count;
    return 0;
#else
    if (backend != moe_BACKEND_CUDA || !tasks || task_count == 0) {
        return 0;
    }
    // TODO(real adapter): instantiate CUTLASS grouped GEMM for host-described
    // runtime tasks. This compatibility entry stays fail-closed until then.
    return 0;
#endif
}

extern "C" int storagellm_cutlass_grouped_moe_indexed_device_f32(
    int32_t backend,
    const moe_grouped_expert_device_task_t* tasks,
    uint32_t task_count,
    void* stream_or_queue
) {
#if !defined(STORAGELLM_HAS_CUTLASS)
    (void)backend;
    (void)tasks;
    (void)task_count;
    (void)stream_or_queue;
    return 0;
#else
    if (backend != moe_BACKEND_CUDA || !tasks || task_count == 0) {
        return 0;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_or_queue);
    (void)stream;
    for (uint32_t i = 0; i < task_count; ++i) {
        const auto& task = tasks[i];
        if (!task.gate_weight || !task.up_weight || !task.down_weight ||
            !task.d_input || !task.d_token_indices || !task.d_token_weights ||
            !task.d_accum || task.assignment_count == 0 ||
            task.input_stride < task.hidden_size ||
            task.accum_stride < task.hidden_size ||
            task.hidden_size == 0 || task.intermediate_size == 0) {
            return 0;
        }
    }
    // TODO(real adapter): build CUTLASS grouped problem list, execute gate/up
    // grouped GEMM, fused SiLU*up, down grouped GEMM, then weighted accumulate
    // directly into d_accum. Only a completed device-resident path returns 1.
    return 0;
#endif
}
