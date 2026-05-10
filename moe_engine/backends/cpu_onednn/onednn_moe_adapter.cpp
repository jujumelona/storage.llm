#include "cpu_parallel.h"
#include "cpu_tensor_view.h"
#include "moe_pc_engine.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <new>
#include <algorithm>

#if defined(STORAGELLM_HAS_DNNL_PRIMITIVE)
#include <dnnl.hpp>
#endif

namespace {

static int storagellm_env_truthy_cpu(const char* name, int fallback) {
    const char* v = name ? std::getenv(name) : nullptr;
    if (!v || !*v) return fallback;
    return !(std::strcmp(v, "0") == 0 || std::strcmp(v, "false") == 0 ||
             std::strcmp(v, "FALSE") == 0 || std::strcmp(v, "off") == 0 ||
             std::strcmp(v, "OFF") == 0 || std::strcmp(v, "no") == 0 ||
             std::strcmp(v, "NO") == 0);
}

static uint32_t storagellm_env_u32_cpu(const char* name, uint32_t fallback, uint32_t lo, uint32_t hi) {
    const char* v = name ? std::getenv(name) : nullptr;
    if (!v || !*v) return fallback;
    char* end = nullptr;
    unsigned long parsed = std::strtoul(v, &end, 10);
    if (end == v) return fallback;
    if (parsed < lo) parsed = lo;
    if (parsed > hi) parsed = hi;
    return static_cast<uint32_t>(parsed);
}

#if defined(STORAGELLM_HAS_DNNL_PRIMITIVE)

static int storagellm_task_is_raw_fp32(const storagellm::cpu_moe::ValidatedTask& v) {
    return v.gate.is_fp32() && v.up.is_fp32() && v.down.is_fp32();
}

static const float* storagellm_fp32_row_base(const storagellm::cpu_moe::WeightMatrixView& w) {
    return storagellm::cpu_moe::weight_ptr_fp32(w, 0);
}

static int storagellm_run_task_dnnl(const storagellm::cpu_moe::ValidatedTask& v) {
    using namespace dnnl;
    const auto& task = *v.task;
    if (!storagellm_task_is_raw_fp32(v)) return 0;
    if (task.assignment_count == 0 || task.hidden_size == 0 || task.intermediate_size == 0) return 0;
    if (task.input_stride < task.hidden_size || task.accum_stride < task.hidden_size) return 0;

    const uint32_t N = task.assignment_count;
    const uint32_t H = task.hidden_size;
    const uint32_t I = task.intermediate_size;
    const uint32_t min_assign = storagellm_env_u32_cpu("STORAGELLM_CPU_DNNL_MIN_ASSIGNMENTS", 2u, 1u, 65536u);
    if (N < min_assign) return 0;

    const float* gate = storagellm_fp32_row_base(v.gate);
    const float* up = storagellm_fp32_row_base(v.up);
    const float* down = storagellm_fp32_row_base(v.down);
    if (!gate || !up || !down) return 0;

    try {
        static thread_local engine eng(engine::kind::cpu, 0);
        static thread_local stream strm(eng);

        std::vector<float> x(static_cast<size_t>(N) * H);
        std::vector<float> gate_out(static_cast<size_t>(N) * I);
        std::vector<float> up_out(static_cast<size_t>(N) * I);
        std::vector<float> mid(static_cast<size_t>(N) * I);
        std::vector<float> y(static_cast<size_t>(N) * H);

        const float* input = static_cast<const float*>(task.d_input);
        for (uint32_t a = 0; a < N; ++a) {
            const uint32_t global_a = task.assignment_offset + a;
            const uint32_t token = task.d_token_indices[global_a];
            const float* src = input + static_cast<uint64_t>(token) * task.input_stride;
            std::copy(src, src + H, x.data() + static_cast<size_t>(a) * H);
        }

        const memory::dims src_dims = {static_cast<memory::dim>(N), static_cast<memory::dim>(H)};
        const memory::dims ih_dims = {static_cast<memory::dim>(H), static_cast<memory::dim>(I)};
        const memory::dims ni_dims = {static_cast<memory::dim>(N), static_cast<memory::dim>(I)};
        const memory::dims dst_dims = {static_cast<memory::dim>(N), static_cast<memory::dim>(H)};

        const auto src_md = memory::desc(src_dims, memory::data_type::f32, {static_cast<memory::dim>(H), 1});
        // Gate/up are stored row-major as [I][H].  Present them to matmul as [H][I]
        // using non-default strides so oneDNN computes X[N,H] * W[H,I].
        const auto gate_md = memory::desc(ih_dims, memory::data_type::f32, {1, static_cast<memory::dim>(H)});
        const auto ni_md = memory::desc(ni_dims, memory::data_type::f32, {static_cast<memory::dim>(I), 1});
        const auto mid_md = memory::desc(ni_dims, memory::data_type::f32, {static_cast<memory::dim>(I), 1});
        // Down is stored row-major as [H][I].  Present as [I][H].
        const auto down_md = memory::desc(ih_dims, memory::data_type::f32, {1, static_cast<memory::dim>(I)});
        const auto dst_md = memory::desc(dst_dims, memory::data_type::f32, {static_cast<memory::dim>(H), 1});

        auto src_mem = memory(src_md, eng, x.data());
        auto gate_mem = memory(gate_md, eng, const_cast<float*>(gate));
        auto up_mem = memory(gate_md, eng, const_cast<float*>(up));
        auto gate_out_mem = memory(ni_md, eng, gate_out.data());
        auto up_out_mem = memory(ni_md, eng, up_out.data());
        auto mid_mem = memory(mid_md, eng, mid.data());
        auto down_mem = memory(down_md, eng, const_cast<float*>(down));
        auto dst_mem = memory(dst_md, eng, y.data());

        auto gu_pd = matmul::primitive_desc(eng, src_md, gate_md, ni_md);
        auto down_pd = matmul::primitive_desc(eng, mid_md, down_md, dst_md);
        auto gu_mm = matmul(gu_pd);
        auto down_mm = matmul(down_pd);
        gu_mm.execute(strm, {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_WEIGHTS, gate_mem}, {DNNL_ARG_DST, gate_out_mem}});
        gu_mm.execute(strm, {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_WEIGHTS, up_mem}, {DNNL_ARG_DST, up_out_mem}});
        strm.wait();

        for (uint64_t idx = 0, total = static_cast<uint64_t>(N) * I; idx < total; ++idx) {
            mid[idx] = storagellm::cpu_moe::activation(task.activation_mode, gate_out[idx], up_out[idx]);
        }

        down_mm.execute(strm, {{DNNL_ARG_SRC, mid_mem}, {DNNL_ARG_WEIGHTS, down_mem}, {DNNL_ARG_DST, dst_mem}});
        strm.wait();

        float* accum = static_cast<float*>(task.d_accum);
        for (uint32_t a = 0; a < N; ++a) {
            const uint32_t global_a = task.assignment_offset + a;
            const uint32_t token = task.d_token_indices[global_a];
            const float route_weight = task.d_token_weights ? task.d_token_weights[global_a] : 1.0f;
            if (!std::isfinite(route_weight)) continue;
            float* dst = accum + static_cast<uint64_t>(token) * task.accum_stride;
            const float* src = y.data() + static_cast<uint64_t>(a) * H;
            for (uint32_t h = 0; h < H; ++h) {
                dst[h] += src[h] * route_weight;
            }
        }
        return 1;
    } catch (const dnnl::error& e) {
        if (storagellm_env_truthy_cpu("STORAGELLM_CPU_DNNL_LOG_ERRORS", 0)) {
            std::fprintf(stderr, "[storageLLM CPU oneDNN] dnnl error status=%d message=%s\n", (int)e.status, e.message);
        }
        return 0;
    } catch (...) {
        return 0;
    }
}
#endif

} // namespace

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
#if defined(STORAGELLM_HAS_DNNL_PRIMITIVE)
    const int use_dnnl = storagellm_env_truthy_cpu("STORAGELLM_CPU_USE_DNNL", 1);
#endif
    for (uint32_t i = 0; i < task_count; ++i) {
        storagellm::cpu_moe::ValidatedTask validated{};
        if (!storagellm::cpu_moe::validate_task(tasks[i], validated)) {
            return 0;
        }
#if defined(STORAGELLM_HAS_DNNL_PRIMITIVE)
        if (use_dnnl && storagellm_run_task_dnnl(validated)) {
            continue;
        }
#endif
        storagellm::cpu_moe::run_task_parallel(validated);
    }
    return 1;
}


extern "C" int storagellm_onednn_cpu_grouped_moe_indexed_device_f32_v2(
    const moe_fast_backend_dispatch_request_t* request
) {
    if (!request || request->abi_version != STORAGELLM_FAST_BACKEND_DISPATCH_ABI_V2) {
        return 0;
    }
    return storagellm_onednn_cpu_grouped_moe_indexed_device_f32(
        request->backend, request->tasks, request->task_count, request->legacy_stream_or_queue);
}
