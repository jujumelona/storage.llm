#include "moe_pc_engine.h"

#include <cstdint>
#include <cstring>
#include <mutex>

#if defined(STORAGELLM_HAS_CLBLAST)
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif
#include <CL/cl.h>
#endif

#if defined(STORAGELLM_HAS_CLBLAST)
struct storagellm_cl_tensor_view {
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

static cl_mem storagellm_cl_weight_mem_fp32(const void* view_ptr, uint32_t rows, uint32_t cols, uint64_t* byte_offset) {
    if (byte_offset) *byte_offset = 0;
    const auto* v = reinterpret_cast<const storagellm_cl_tensor_view*>(view_ptr);
    if (!v || !v->ptr || v->rows != rows || v->cols != cols) return nullptr;
    const uint64_t row_bytes = static_cast<uint64_t>(cols) * sizeof(float);
    const uint64_t total_bytes = static_cast<uint64_t>(rows) * row_bytes;
    if (v->weight_format == static_cast<uint32_t>(moe_WEIGHT_ENCODING_RAW_FP32) &&
        v->bytes >= total_bytes && v->weight_bytes >= total_bytes && v->weight_row_bytes >= row_bytes) {
        return reinterpret_cast<cl_mem>(static_cast<uintptr_t>(v->ptr));
    }
    if (v->expert_gpu_layout_kind == 3u && v->expert_gpu_layout_size >= total_bytes &&
        v->expert_gpu_layout_row_bytes >= row_bytes && v->expert_gpu_layout_offset <= v->bytes &&
        total_bytes <= v->bytes - v->expert_gpu_layout_offset) {
        if (byte_offset) *byte_offset = v->expert_gpu_layout_offset;
        return reinterpret_cast<cl_mem>(static_cast<uintptr_t>(v->ptr));
    }
    return nullptr;
}

static const char* storagellm_opencl_fused_moe_source = R"CLC(
inline float storagellm_gelu_erf(float x) {
    return 0.5f * x * (1.0f + erf(x * 0.7071067811865476f));
}
inline float storagellm_gelu_tanh(float x) {
    const float k = 0.7978845608028654f;
    const float inner = k * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanh(inner));
}
inline void storagellm_atomic_add_f32(__global float* addr, float val) {
    volatile __global unsigned int* p = (volatile __global unsigned int*)addr;
    unsigned int old_bits = *p;
    for (;;) {
        const unsigned int assumed = old_bits;
        const float old_val = as_float(assumed);
        const unsigned int new_bits = as_uint(old_val + val);
        old_bits = atomic_cmpxchg(p, assumed, new_bits);
        if (old_bits == assumed) break;
    }
}
inline float storagellm_act(uint mode, float gate, float up) {
    if (!isfinite(gate) || !isfinite(up)) return 0.0f;
    float a = 0.0f;
    if (mode == 2u) a = storagellm_gelu_tanh(gate);
    else if (mode == 1u) a = storagellm_gelu_erf(gate);
    else a = gate > 40.0f ? gate : (gate < -40.0f ? 0.0f : gate / (1.0f + exp(-gate)));
    const float y = a * up;
    return isfinite(y) ? y : 0.0f;
}
__kernel void storagellm_opencl_fused_moe_f32(
    __global const float* gate,
    ulong gate_offset_f,
    __global const float* up,
    ulong up_offset_f,
    __global const float* down,
    ulong down_offset_f,
    __global const float* input,
    uint input_stride,
    __global const uint* token_indices,
    __global const float* token_weights,
    uint assignment_offset,
    uint assignment_count,
    uint hidden,
    uint intermediate,
    uint activation_mode,
    __global float* accum,
    uint accum_stride) {
    const uint local_row = get_global_id(0);
    const uint h = get_global_id(1);
    if (local_row >= assignment_count || h >= hidden) return;
    const uint row = assignment_offset + local_row;
    const uint token = token_indices[row];
    const float route = token_weights ? token_weights[row] : 1.0f;
    if (!isfinite(route)) return;
    __global const float* x = input + ((ulong)token) * input_stride;
    float y = 0.0f;
    for (uint r = 0; r < intermediate; ++r) {
        float g = 0.0f;
        float u = 0.0f;
        __global const float* gw = gate + gate_offset_f + ((ulong)r) * hidden;
        __global const float* uw = up + up_offset_f + ((ulong)r) * hidden;
        for (uint c = 0; c < hidden; ++c) {
            const float xv = x[c];
            g = fma(gw[c], xv, g);
            u = fma(uw[c], xv, u);
        }
        y = fma(down[down_offset_f + ((ulong)h) * intermediate + r], storagellm_act(activation_mode, g, u), y);
    }
    storagellm_atomic_add_f32(accum + ((ulong)token) * accum_stride + h, y * route);
}
)CLC";

struct storagellm_opencl_kernel_cache {
    cl_context context = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    std::mutex mutex;
    ~storagellm_opencl_kernel_cache() {
        if (kernel) clReleaseKernel(kernel);
        if (program) clReleaseProgram(program);
    }
};
static storagellm_opencl_kernel_cache g_cl_cache;

static int storagellm_opencl_get_kernel(cl_command_queue queue, cl_kernel* out) {
    if (!queue || !out) return 0;
    cl_int err = CL_SUCCESS;
    cl_context ctx = nullptr;
    err = clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(ctx), &ctx, nullptr);
    if (err != CL_SUCCESS || !ctx) return 0;
    std::lock_guard<std::mutex> lock(g_cl_cache.mutex);
    if (g_cl_cache.kernel && g_cl_cache.context == ctx) {
        *out = g_cl_cache.kernel;
        return 1;
    }
    if (g_cl_cache.kernel) { clReleaseKernel(g_cl_cache.kernel); g_cl_cache.kernel = nullptr; }
    if (g_cl_cache.program) { clReleaseProgram(g_cl_cache.program); g_cl_cache.program = nullptr; }
    const char* src = storagellm_opencl_fused_moe_source;
    const size_t len = std::strlen(src);
    g_cl_cache.program = clCreateProgramWithSource(ctx, 1, &src, &len, &err);
    if (err != CL_SUCCESS || !g_cl_cache.program) return 0;
    err = clBuildProgram(g_cl_cache.program, 0, nullptr, "-cl-std=CL1.2", nullptr, nullptr);
    if (err != CL_SUCCESS) return 0;
    g_cl_cache.kernel = clCreateKernel(g_cl_cache.program, "storagellm_opencl_fused_moe_f32", &err);
    if (err != CL_SUCCESS || !g_cl_cache.kernel) return 0;
    g_cl_cache.context = ctx;
    *out = g_cl_cache.kernel;
    return 1;
}

static int storagellm_opencl_run(const moe_grouped_expert_device_task_t* tasks, uint32_t task_count, cl_command_queue queue) {
    if (!tasks || task_count == 0 || !queue) return 0;
    cl_kernel kernel = nullptr;
    if (!storagellm_opencl_get_kernel(queue, &kernel)) return 0;
    for (uint32_t i = 0; i < task_count; ++i) {
        const auto& t = tasks[i];
        if (!t.gate_weight || !t.up_weight || !t.down_weight || !t.d_input || !t.d_token_indices ||
            !t.d_token_weights || !t.d_accum || t.assignment_count == 0 ||
            t.input_stride < t.hidden_size || t.accum_stride < t.hidden_size ||
            t.hidden_size == 0 || t.intermediate_size == 0) return 0;
        uint64_t gate_off_b = 0, up_off_b = 0, down_off_b = 0;
        cl_mem gate = storagellm_cl_weight_mem_fp32(t.gate_weight, t.intermediate_size, t.hidden_size, &gate_off_b);
        cl_mem up = storagellm_cl_weight_mem_fp32(t.up_weight, t.intermediate_size, t.hidden_size, &up_off_b);
        cl_mem down = storagellm_cl_weight_mem_fp32(t.down_weight, t.hidden_size, t.intermediate_size, &down_off_b);
        if (!gate || !up || !down) return 0;
        cl_mem input = reinterpret_cast<cl_mem>(const_cast<void*>(t.d_input));
        cl_mem idx = reinterpret_cast<cl_mem>(const_cast<uint32_t*>(t.d_token_indices));
        cl_mem weights = reinterpret_cast<cl_mem>(const_cast<float*>(t.d_token_weights));
        cl_mem accum = reinterpret_cast<cl_mem>(t.d_accum);
        const uint64_t gate_off_f = gate_off_b / sizeof(float);
        const uint64_t up_off_f = up_off_b / sizeof(float);
        const uint64_t down_off_f = down_off_b / sizeof(float);
        int a = 0;
        cl_int err = CL_SUCCESS;
        err |= clSetKernelArg(kernel, a++, sizeof(gate), &gate);
        err |= clSetKernelArg(kernel, a++, sizeof(gate_off_f), &gate_off_f);
        err |= clSetKernelArg(kernel, a++, sizeof(up), &up);
        err |= clSetKernelArg(kernel, a++, sizeof(up_off_f), &up_off_f);
        err |= clSetKernelArg(kernel, a++, sizeof(down), &down);
        err |= clSetKernelArg(kernel, a++, sizeof(down_off_f), &down_off_f);
        err |= clSetKernelArg(kernel, a++, sizeof(input), &input);
        err |= clSetKernelArg(kernel, a++, sizeof(t.input_stride), &t.input_stride);
        err |= clSetKernelArg(kernel, a++, sizeof(idx), &idx);
        err |= clSetKernelArg(kernel, a++, sizeof(weights), &weights);
        err |= clSetKernelArg(kernel, a++, sizeof(t.assignment_offset), &t.assignment_offset);
        err |= clSetKernelArg(kernel, a++, sizeof(t.assignment_count), &t.assignment_count);
        err |= clSetKernelArg(kernel, a++, sizeof(t.hidden_size), &t.hidden_size);
        err |= clSetKernelArg(kernel, a++, sizeof(t.intermediate_size), &t.intermediate_size);
        err |= clSetKernelArg(kernel, a++, sizeof(t.activation_mode), &t.activation_mode);
        err |= clSetKernelArg(kernel, a++, sizeof(accum), &accum);
        err |= clSetKernelArg(kernel, a++, sizeof(t.accum_stride), &t.accum_stride);
        if (err != CL_SUCCESS) return 0;
        size_t global[2] = { t.assignment_count, t.hidden_size };
        err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) return 0;
    }
    return 1;
}
#endif

extern "C" int storagellm_clblast_grouped_moe_indexed_device_f32(
    int32_t backend,
    const moe_grouped_expert_device_task_t* tasks,
    uint32_t task_count,
    void* stream_or_queue
) {
#if !defined(STORAGELLM_HAS_CLBLAST)
    (void)backend; (void)tasks; (void)task_count; (void)stream_or_queue;
    return 0;
#else
    if (backend != moe_BACKEND_OPENCL || !stream_or_queue) return 0;
    return storagellm_opencl_run(tasks, task_count, reinterpret_cast<cl_command_queue>(stream_or_queue));
#endif
}

extern "C" int storagellm_clblast_grouped_moe_indexed_device_f32_v2(
    const moe_fast_backend_dispatch_request_t* request
) {
    if (!request || request->abi_version != STORAGELLM_FAST_BACKEND_DISPATCH_ABI_V2) return 0;
    void* q = request->legacy_stream_or_queue;
    if (request->context && request->context->context_kind == moe_FAST_BACKEND_CONTEXT_OPENCL &&
        request->context->u.opencl.command_queue) {
        q = request->context->u.opencl.command_queue;
    }
    return storagellm_clblast_grouped_moe_indexed_device_f32(request->backend, request->tasks, request->task_count, q);
}
