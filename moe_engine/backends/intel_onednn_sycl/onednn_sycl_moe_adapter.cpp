#include "moe_pc_engine.h"

#include <cstdint>

#if defined(STORAGELLM_HAS_ONEDNN_SYCL)
#include <sycl/sycl.hpp>

struct storagellm_sycl_tensor_view {
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

static const float* storagellm_sycl_weight_ptr_fp32(
    const void* view_ptr,
    uint32_t rows,
    uint32_t cols
) {
    const auto* v = reinterpret_cast<const storagellm_sycl_tensor_view*>(view_ptr);
    if (!v || !v->ptr || v->rows != rows || v->cols != cols) return nullptr;
    const uint64_t row_bytes = static_cast<uint64_t>(cols) * sizeof(float);
    const uint64_t total_bytes = static_cast<uint64_t>(rows) * row_bytes;
    if (v->weight_format == static_cast<uint32_t>(moe_WEIGHT_ENCODING_RAW_FP32) &&
        v->bytes >= total_bytes && v->weight_bytes >= total_bytes && v->weight_row_bytes >= row_bytes) {
        return reinterpret_cast<const float*>(static_cast<uintptr_t>(v->ptr));
    }
    if (v->expert_gpu_layout_kind == 3u &&
        v->expert_gpu_layout_size >= total_bytes &&
        v->expert_gpu_layout_row_bytes >= row_bytes &&
        v->expert_gpu_layout_offset <= v->bytes &&
        total_bytes <= v->bytes - v->expert_gpu_layout_offset) {
        return reinterpret_cast<const float*>(static_cast<uintptr_t>(v->ptr + v->expert_gpu_layout_offset));
    }
    return nullptr;
}

static int storagellm_sycl_validate_task(const moe_grouped_expert_device_task_t& t) {
    return t.gate_weight && t.up_weight && t.down_weight && t.d_input && t.d_token_indices &&
        t.d_token_weights && t.d_accum && t.assignment_count != 0 &&
        t.input_stride >= t.hidden_size && t.accum_stride >= t.hidden_size &&
        t.hidden_size != 0 && t.intermediate_size != 0 &&
        storagellm_sycl_weight_ptr_fp32(t.gate_weight, t.intermediate_size, t.hidden_size) &&
        storagellm_sycl_weight_ptr_fp32(t.up_weight, t.intermediate_size, t.hidden_size) &&
        storagellm_sycl_weight_ptr_fp32(t.down_weight, t.hidden_size, t.intermediate_size);
}

static int storagellm_sycl_run(
    const moe_grouped_expert_device_task_t* tasks,
    uint32_t task_count,
    sycl::queue* queue
) {
    if (!tasks || task_count == 0 || !queue) return 0;
    try {
        for (uint32_t i = 0; i < task_count; ++i) {
            const auto& t = tasks[i];
            if (!storagellm_sycl_validate_task(t)) return 0;
            const float* gate = storagellm_sycl_weight_ptr_fp32(t.gate_weight, t.intermediate_size, t.hidden_size);
            const float* up = storagellm_sycl_weight_ptr_fp32(t.up_weight, t.intermediate_size, t.hidden_size);
            const float* down = storagellm_sycl_weight_ptr_fp32(t.down_weight, t.hidden_size, t.intermediate_size);
            const float* input = static_cast<const float*>(t.d_input);
            const uint32_t* token_indices = t.d_token_indices;
            const float* token_weights = t.d_token_weights;
            float* accum = static_cast<float*>(t.d_accum);
            const uint32_t input_stride = t.input_stride;
            const uint32_t assignment_offset = t.assignment_offset;
            const uint32_t assignment_count = t.assignment_count;
            const uint32_t hidden = t.hidden_size;
            const uint32_t intermediate = t.intermediate_size;
            const uint32_t activation_mode = t.activation_mode;
            const uint32_t accum_stride = t.accum_stride;
            queue->submit([&](sycl::handler& h) {
                h.parallel_for(sycl::range<2>(assignment_count, hidden), [=](sycl::id<2> id) {
                    const uint32_t local_row = static_cast<uint32_t>(id[0]);
                    const uint32_t hidx = static_cast<uint32_t>(id[1]);
                    const uint32_t row = assignment_offset + local_row;
                    const uint32_t token = token_indices[row];
                    const float route = token_weights ? token_weights[row] : 1.0f;
                    if (sycl::isnan(route) || sycl::isinf(route)) return;
                    float y = 0.0f;
                    for (uint32_t r = 0; r < intermediate; ++r) {
                        float g = 0.0f;
                        float u = 0.0f;
                        const uint64_t wbase = static_cast<uint64_t>(r) * hidden;
                        const uint64_t xbase = static_cast<uint64_t>(token) * input_stride;
                        for (uint32_t c = 0; c < hidden; ++c) {
                            const float xv = input[xbase + c];
                            g = sycl::fma(gate[wbase + c], xv, g);
                            u = sycl::fma(up[wbase + c], xv, u);
                        }
                        float a = 0.0f;
                        if (activation_mode == 2u) {
                            const float k = 0.7978845608028654f;
                            const float inner = k * (g + 0.044715f * g * g * g);
                            a = 0.5f * g * (1.0f + sycl::tanh(inner));
                        } else if (activation_mode == 1u) {
                            a = 0.5f * g * (1.0f + sycl::erf(g * 0.7071067811865476f));
                        } else {
                            a = g > 40.0f ? g : (g < -40.0f ? 0.0f : g / (1.0f + sycl::exp(-g)));
                        }
                        const float mid = (sycl::isnan(a) || sycl::isinf(a) || sycl::isnan(u) || sycl::isinf(u)) ? 0.0f : a * u;
                        y = sycl::fma(down[static_cast<uint64_t>(hidx) * intermediate + r], mid, y);
                    }
                    sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>
                        acc(accum[static_cast<uint64_t>(token) * accum_stride + hidx]);
                    acc.fetch_add(y * route);
                });
            });
        }
        return 1;
    } catch (...) {
        return 0;
    }
}
#endif

extern "C" int storagellm_onednn_sycl_grouped_moe_indexed_device_f32(
    int32_t backend,
    const moe_grouped_expert_device_task_t* tasks,
    uint32_t task_count,
    void* stream_or_queue
) {
#if !defined(STORAGELLM_HAS_ONEDNN_SYCL)
    (void)backend; (void)tasks; (void)task_count; (void)stream_or_queue;
    return 0;
#else
    if ((backend != moe_BACKEND_SYCL && backend != moe_BACKEND_LEVEL_ZERO) || !stream_or_queue) return 0;
    return storagellm_sycl_run(tasks, task_count, reinterpret_cast<sycl::queue*>(stream_or_queue));
#endif
}

extern "C" int storagellm_onednn_sycl_grouped_moe_indexed_device_f32_v2(
    const moe_fast_backend_dispatch_request_t* request
) {
#if !defined(STORAGELLM_HAS_ONEDNN_SYCL)
    (void)request;
    return 0;
#else
    if (!request || request->abi_version != STORAGELLM_FAST_BACKEND_DISPATCH_ABI_V2) return 0;
    void* q = request->legacy_stream_or_queue;
    if (request->context && request->context->context_kind == moe_FAST_BACKEND_CONTEXT_SYCL &&
        request->context->u.sycl.queue) {
        q = request->context->u.sycl.queue;
    }
    return storagellm_onednn_sycl_grouped_moe_indexed_device_f32(request->backend, request->tasks, request->task_count, q);
#endif
}
