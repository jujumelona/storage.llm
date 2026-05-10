#pragma once

#include "moe_pc_engine.h"
#include <cstdint>
#include <cstring>

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

inline float cpu_bf16_to_f32(uint16_t v) {
    uint32_t u = static_cast<uint32_t>(v) << 16;
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

inline float cpu_fp16_to_f32(uint16_t h) {
    const uint32_t sign = (static_cast<uint32_t>(h & 0x8000u)) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t out = 0;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            // Normalize subnormal half without unsigned exponent underflow.
            int e = -14;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --e;
            }
            mant &= 0x03ffu;
            out = sign | (static_cast<uint32_t>(e + 127) << 23) | (mant << 13);
        }
    } else if (exp == 0x1fu) {
        out = sign | 0x7f800000u | (mant << 13);
    } else {
        out = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
    }
    float f = 0.0f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

struct WeightMatrixView {
    const uint8_t* base = nullptr;
    uint32_t format = moe_WEIGHT_ENCODING_UNKNOWN;
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint64_t row_bytes = 0;

    bool valid() const { return base && rows != 0 && cols != 0 && row_bytes != 0; }
    bool is_fp32() const { return format == static_cast<uint32_t>(moe_WEIGHT_ENCODING_RAW_FP32); }
    bool is_bf16() const { return format == static_cast<uint32_t>(moe_WEIGHT_ENCODING_RAW_BF16); }
    bool is_fp16() const { return format == static_cast<uint32_t>(moe_WEIGHT_ENCODING_RAW_FP16); }
};

inline int weight_matrix_view_from_device_tensor(
    const void* view_ptr,
    uint32_t expected_rows,
    uint32_t expected_cols,
    WeightMatrixView& out
) {
    out = WeightMatrixView{};
    const auto* v = reinterpret_cast<const DeviceTensorBatchView*>(view_ptr);
    if (!v || !v->ptr || v->backend_kind != 0u ||
        v->rows != expected_rows || v->cols != expected_cols) {
        return 0;
    }

    const uint32_t fmt = v->weight_format;
    uint64_t elem_bytes = 0;
    if (fmt == static_cast<uint32_t>(moe_WEIGHT_ENCODING_RAW_FP32)) {
        elem_bytes = sizeof(float);
    } else if (fmt == static_cast<uint32_t>(moe_WEIGHT_ENCODING_RAW_BF16) ||
               fmt == static_cast<uint32_t>(moe_WEIGHT_ENCODING_RAW_FP16)) {
        elem_bytes = sizeof(uint16_t);
    } else {
        return 0;
    }

    const uint64_t min_row_bytes = static_cast<uint64_t>(expected_cols) * elem_bytes;
    const uint64_t min_bytes = static_cast<uint64_t>(expected_rows) * min_row_bytes;
    if (min_bytes == 0) return 0;

    // Normal CPU-resident raw tensor.
    if (v->bytes >= min_bytes && v->weight_bytes >= min_bytes &&
        v->weight_row_bytes >= min_row_bytes) {
        out.base = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(v->ptr));
        out.format = fmt;
        out.rows = expected_rows;
        out.cols = expected_cols;
        out.row_bytes = v->weight_row_bytes;
        return 1;
    }

    // Materialized layout inside an expert bundle.  Current bundle layout kind 3 is
    // used by the fast path for raw, row-major materialized weights.  Accept FP32
    // and 16-bit raw rows when row_bytes proves the layout is large enough.
    if (v->expert_gpu_layout_kind == 3u &&
        v->expert_gpu_layout_size >= min_bytes &&
        v->expert_gpu_layout_row_bytes >= min_row_bytes &&
        v->expert_gpu_layout_offset <= v->bytes &&
        min_bytes <= v->bytes - v->expert_gpu_layout_offset) {
        out.base = reinterpret_cast<const uint8_t*>(
            static_cast<uintptr_t>(v->ptr + v->expert_gpu_layout_offset));
        out.format = fmt;
        out.rows = expected_rows;
        out.cols = expected_cols;
        out.row_bytes = v->expert_gpu_layout_row_bytes;
        return 1;
    }
    return 0;
}

inline const float* weight_ptr_fp32(const WeightMatrixView& w, uint32_t row) {
    if (!w.valid() || !w.is_fp32() || row >= w.rows ||
        w.row_bytes < static_cast<uint64_t>(w.cols) * sizeof(float)) {
        return nullptr;
    }
    return reinterpret_cast<const float*>(w.base + static_cast<uint64_t>(row) * w.row_bytes);
}

inline float weight_value(const WeightMatrixView& w, uint32_t row, uint32_t col) {
    if (!w.valid() || row >= w.rows || col >= w.cols) return 0.0f;
    const uint8_t* row_base = w.base + static_cast<uint64_t>(row) * w.row_bytes;
    if (w.is_fp32()) {
        return reinterpret_cast<const float*>(row_base)[col];
    }
    const uint16_t raw = reinterpret_cast<const uint16_t*>(row_base)[col];
    if (w.is_bf16()) return cpu_bf16_to_f32(raw);
    if (w.is_fp16()) return cpu_fp16_to_f32(raw);
    return 0.0f;
}

struct ValidatedTask {
    const moe_grouped_expert_device_task_t* task = nullptr;
    WeightMatrixView gate{};
    WeightMatrixView up{};
    WeightMatrixView down{};
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
    return weight_matrix_view_from_device_tensor(task.gate_weight, task.intermediate_size, task.hidden_size, out.gate) &&
           weight_matrix_view_from_device_tensor(task.up_weight, task.intermediate_size, task.hidden_size, out.up) &&
           weight_matrix_view_from_device_tensor(task.down_weight, task.hidden_size, task.intermediate_size, out.down);
}

} // namespace storagellm::cpu_moe
