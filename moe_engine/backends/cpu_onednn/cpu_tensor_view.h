#pragma once

#include "moe_pc_engine.h"
#include <cstdint>

namespace storagellm::cpu_moe {

struct DeviceTensorBatchView {
    const void* rec;
    uint64_t key;
    uint64_t ptr;
    uint64_t bytes;
    uint32_t weight_format;
    uint32_t backend_kind;
    uint64_t backend_aux;
    uint32_t rows;
    uint32_t cols;
    uint64_t weight_row_bytes;
    uint64_t weight_bytes;
    uint64_t stream_bytes;
    uint32_t expert_gpu_layout_kind;
    uint64_t expert_gpu_layout_offset;
    uint64_t expert_gpu_layout_size;
    uint64_t expert_gpu_layout_row_bytes;
};

inline const float* weight_ptr_fp32(
    const void* view_ptr,
    uint32_t expected_rows,
    uint32_t expected_cols
) {
    const auto* v = reinterpret_cast<const DeviceTensorBatchView*>(view_ptr);
    if (!v || !v->ptr || v->backend_kind != 0u ||
        v->rows != expected_rows || v->cols != expected_cols) {
        return nullptr;
    }
    const uint64_t fp32_bytes =
        static_cast<uint64_t>(expected_rows) * expected_cols * sizeof(float);
    if (fp32_bytes == 0) {
        return nullptr;
    }
    if (v->weight_format == static_cast<uint32_t>(moe_WEIGHT_ENCODING_RAW_FP32) &&
        v->bytes >= fp32_bytes && v->weight_bytes >= fp32_bytes &&
        v->weight_row_bytes >= static_cast<uint64_t>(expected_cols) * sizeof(float)) {
        return reinterpret_cast<const float*>(static_cast<uintptr_t>(v->ptr));
    }
    if (v->expert_gpu_layout_kind == 3u &&
        v->expert_gpu_layout_size >= fp32_bytes &&
        v->expert_gpu_layout_row_bytes >= static_cast<uint64_t>(expected_cols) * sizeof(float) &&
        v->expert_gpu_layout_offset <= v->bytes &&
        fp32_bytes <= v->bytes - v->expert_gpu_layout_offset) {
        return reinterpret_cast<const float*>(
            static_cast<uintptr_t>(v->ptr + v->expert_gpu_layout_offset));
    }
    return nullptr;
}

struct ValidatedTask {
    const moe_grouped_expert_device_task_t* task = nullptr;
    const float* gate = nullptr;
    const float* up = nullptr;
    const float* down = nullptr;
};

inline bool validate_task(
    const moe_grouped_expert_device_task_t& task,
    ValidatedTask& out
) {
    if (!task.gate_weight || !task.up_weight || !task.down_weight ||
        !task.d_input || !task.d_token_indices || !task.d_token_weights ||
        !task.d_accum || task.assignment_count == 0 ||
        task.input_stride < task.hidden_size ||
        task.accum_stride < task.hidden_size ||
        task.hidden_size == 0 || task.intermediate_size == 0) {
        return false;
    }
    out.task = &task;
    out.gate = weight_ptr_fp32(task.gate_weight, task.intermediate_size, task.hidden_size);
    out.up = weight_ptr_fp32(task.up_weight, task.intermediate_size, task.hidden_size);
    out.down = weight_ptr_fp32(task.down_weight, task.hidden_size, task.intermediate_size);
    return out.gate && out.up && out.down;
}

} // namespace storagellm::cpu_moe
