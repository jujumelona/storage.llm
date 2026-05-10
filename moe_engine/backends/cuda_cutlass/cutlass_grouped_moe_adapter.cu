#include "moe_pc_engine.h"

#include <cstdint>
#include <cmath>

#if defined(STORAGELLM_HAS_CUTLASS)
#include <cuda_runtime_api.h>
#include <cutlass/cutlass.h>
#endif

struct moe_model_grouped_expert_runtime_task;

#if defined(STORAGELLM_HAS_CUTLASS)
struct storagellm_cutlass_tensor_view {
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

static const float* storagellm_cutlass_weight_ptr_fp32(
    const void* view_ptr,
    uint32_t expected_rows,
    uint32_t expected_cols
) {
    const auto* v = reinterpret_cast<const storagellm_cutlass_tensor_view*>(view_ptr);
    if (!v || !v->ptr || v->rows != expected_rows || v->cols != expected_cols) return nullptr;
    const uint64_t row_bytes = static_cast<uint64_t>(expected_cols) * sizeof(float);
    const uint64_t fp32_bytes = static_cast<uint64_t>(expected_rows) * row_bytes;
    if (v->weight_format == static_cast<uint32_t>(moe_WEIGHT_ENCODING_RAW_FP32) &&
        v->bytes >= fp32_bytes && v->weight_bytes >= fp32_bytes && v->weight_row_bytes >= row_bytes) {
        return reinterpret_cast<const float*>(static_cast<uintptr_t>(v->ptr));
    }
    if (v->expert_gpu_layout_kind == 3u &&
        v->expert_gpu_layout_size >= fp32_bytes &&
        v->expert_gpu_layout_row_bytes >= row_bytes &&
        v->expert_gpu_layout_offset <= v->bytes &&
        fp32_bytes <= v->bytes - v->expert_gpu_layout_offset) {
        return reinterpret_cast<const float*>(static_cast<uintptr_t>(v->ptr + v->expert_gpu_layout_offset));
    }
    return nullptr;
}

__device__ float storagellm_cutlass_gelu_erf(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
}

__device__ float storagellm_cutlass_gelu_tanh(float x) {
    const float k = 0.7978845608028654f;
    const float inner = k * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__device__ float storagellm_cutlass_activation(uint32_t mode, float gate, float up) {
    if (!isfinite(gate) || !isfinite(up)) return 0.0f;
    float a = 0.0f;
    if (mode == 2u) a = storagellm_cutlass_gelu_tanh(gate);
    else if (mode == 1u) a = storagellm_cutlass_gelu_erf(gate);
    else a = gate > 40.0f ? gate : (gate < -40.0f ? 0.0f : gate / (1.0f + expf(-gate)));
    const float y = a * up;
    return isfinite(y) ? y : 0.0f;
}

__global__ void storagellm_cutlass_fused_moe_f32_kernel(
    const float* gate,
    const float* up,
    const float* down,
    const float* input,
    uint32_t input_stride,
    const uint32_t* token_indices,
    const float* token_weights,
    uint32_t assignment_offset,
    uint32_t assignment_count,
    uint32_t hidden,
    uint32_t intermediate,
    uint32_t activation_mode,
    float* accum,
    uint32_t accum_stride
) {
    const uint32_t local_row = blockIdx.x;
    if (local_row >= assignment_count) return;
    const uint32_t row = assignment_offset + local_row;
    const uint32_t token = token_indices[row];
    const float route = token_weights ? token_weights[row] : 1.0f;
    if (!isfinite(route)) return;
    const float* x = input + static_cast<uint64_t>(token) * input_stride;
    extern __shared__ float mid[];

    for (uint32_t r = threadIdx.x; r < intermediate; r += blockDim.x) {
        const float* gw = gate + static_cast<uint64_t>(r) * hidden;
        const float* uw = up + static_cast<uint64_t>(r) * hidden;
        float g = 0.0f;
        float u = 0.0f;
        for (uint32_t h = 0; h < hidden; ++h) {
            const float xv = x[h];
            g = fmaf(gw[h], xv, g);
            u = fmaf(uw[h], xv, u);
        }
        mid[r] = storagellm_cutlass_activation(activation_mode, g, u);
    }
    __syncthreads();

    for (uint32_t h = threadIdx.x; h < hidden; h += blockDim.x) {
        const float* dw = down + static_cast<uint64_t>(h) * intermediate;
        float y = 0.0f;
        for (uint32_t r = 0; r < intermediate; ++r) y = fmaf(dw[r], mid[r], y);
        atomicAdd(accum + static_cast<uint64_t>(token) * accum_stride + h, y * route);
    }
}

static int storagellm_cutlass_validate_task(const moe_grouped_expert_device_task_t& t) {
    return t.gate_weight && t.up_weight && t.down_weight && t.d_input &&
        t.d_token_indices && t.d_token_weights && t.d_accum && t.assignment_count != 0 &&
        t.input_stride >= t.hidden_size && t.accum_stride >= t.hidden_size &&
        t.hidden_size != 0 && t.intermediate_size != 0 &&
        storagellm_cutlass_weight_ptr_fp32(t.gate_weight, t.intermediate_size, t.hidden_size) &&
        storagellm_cutlass_weight_ptr_fp32(t.up_weight, t.intermediate_size, t.hidden_size) &&
        storagellm_cutlass_weight_ptr_fp32(t.down_weight, t.hidden_size, t.intermediate_size);
}
#endif

extern "C" int storagellm_cutlass_grouped_moe_runtime_f32(
    int32_t backend,
    const moe_model_grouped_expert_runtime_task* tasks,
    uint32_t task_count
) {
    (void)backend;
    (void)tasks;
    (void)task_count;
    // Host-described runtime tasks do not carry token-indexed device buffers.
    return 0;
}

extern "C" int storagellm_cutlass_grouped_moe_indexed_device_f32(
    int32_t backend,
    const moe_grouped_expert_device_task_t* tasks,
    uint32_t task_count,
    void* stream_or_queue
) {
#if !defined(STORAGELLM_HAS_CUTLASS)
    (void)backend; (void)tasks; (void)task_count; (void)stream_or_queue;
    return 0;
#else
    if (backend != moe_BACKEND_CUDA || !tasks || task_count == 0) return 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_or_queue);
    const uint32_t block = 256;
    const uint32_t max_intermediate = 65536u;
    for (uint32_t i = 0; i < task_count; ++i) {
        const auto& t = tasks[i];
        if (!storagellm_cutlass_validate_task(t) || t.intermediate_size > max_intermediate) return 0;
        const float* gate = storagellm_cutlass_weight_ptr_fp32(t.gate_weight, t.intermediate_size, t.hidden_size);
        const float* up = storagellm_cutlass_weight_ptr_fp32(t.up_weight, t.intermediate_size, t.hidden_size);
        const float* down = storagellm_cutlass_weight_ptr_fp32(t.down_weight, t.hidden_size, t.intermediate_size);
        const size_t shared_bytes = static_cast<size_t>(t.intermediate_size) * sizeof(float);
        storagellm_cutlass_fused_moe_f32_kernel<<<t.assignment_count, block, shared_bytes, stream>>>(
            gate, up, down,
            static_cast<const float*>(t.d_input),
            t.input_stride,
            t.d_token_indices,
            t.d_token_weights,
            t.assignment_offset,
            t.assignment_count,
            t.hidden_size,
            t.intermediate_size,
            t.activation_mode,
            static_cast<float*>(t.d_accum),
            t.accum_stride);
        if (cudaGetLastError() != cudaSuccess) return 0;
    }
    return 1;
#endif
}

extern "C" int storagellm_cutlass_grouped_moe_indexed_device_f32_v2(
    const moe_fast_backend_dispatch_request_t* request
) {
    if (!request || request->abi_version != STORAGELLM_FAST_BACKEND_DISPATCH_ABI_V2) return 0;
    void* q = request->legacy_stream_or_queue;
    if (request->context && request->context->context_kind == moe_FAST_BACKEND_CONTEXT_CUDA &&
        request->context->u.cuda.cuda_stream) {
        q = request->context->u.cuda.cuda_stream;
    }
    return storagellm_cutlass_grouped_moe_indexed_device_f32(
        request->backend, request->tasks, request->task_count, q);
}
