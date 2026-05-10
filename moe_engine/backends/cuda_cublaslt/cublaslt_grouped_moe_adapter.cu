#include "moe_pc_engine.h"

#include <climits>
#include <cstddef>
#include <mutex>
#include <vector>

#if defined(STORAGELLM_HAS_CUBLASLT)
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cublasLt.h>

struct storagellm_device_tensor_batch_view {
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

static cublasHandle_t g_storagellm_cublas = nullptr;
static cublasLtHandle_t g_storagellm_cublaslt = nullptr;
static std::once_flag g_storagellm_cublas_once;

struct storagellm_cublaslt_workspace {
    float* d_x = nullptr;
    float* d_gate = nullptr;
    float* d_up = nullptr;
    float* d_mid = nullptr;
    float* d_out = nullptr;
    size_t x_cap = 0;
    size_t mid_cap = 0;

    ~storagellm_cublaslt_workspace() {
        if (d_out) cudaFree(d_out);
        if (d_mid) cudaFree(d_mid);
        if (d_up) cudaFree(d_up);
        if (d_gate) cudaFree(d_gate);
        if (d_x) cudaFree(d_x);
    }

    static int ensure_buffer(float*& ptr, size_t& cap, size_t bytes) {
        if (bytes == 0) {
            return 0;
        }
        if (cap >= bytes && ptr) {
            return 1;
        }
        if (ptr) {
            cudaFree(ptr);
            ptr = nullptr;
            cap = 0;
        }
        if (cudaMalloc(reinterpret_cast<void**>(&ptr), bytes) != cudaSuccess) {
            return 0;
        }
        cap = bytes;
        return 1;
    }

    int ensure(size_t x_bytes, size_t mid_bytes) {
        return ensure_buffer(d_x, x_cap, x_bytes) &&
            ensure_buffer(d_out, x_cap_out(), x_bytes) &&
            ensure_buffer(d_gate, mid_cap_gate(), mid_bytes) &&
            ensure_buffer(d_up, mid_cap_up(), mid_bytes) &&
            ensure_buffer(d_mid, mid_cap, mid_bytes);
    }

private:
    size_t d_out_cap = 0;
    size_t d_gate_cap = 0;
    size_t d_up_cap = 0;
    size_t& x_cap_out() { return d_out_cap; }
    size_t& mid_cap_gate() { return d_gate_cap; }
    size_t& mid_cap_up() { return d_up_cap; }
};

static thread_local storagellm_cublaslt_workspace g_storagellm_cublaslt_workspace;


static int storagellm_ensure_cublas_grouped() {
    std::call_once(g_storagellm_cublas_once, [] {
        if (cublasCreate(&g_storagellm_cublas) != CUBLAS_STATUS_SUCCESS) {
            g_storagellm_cublas = nullptr;
            return;
        }
        if (cublasLtCreate(&g_storagellm_cublaslt) != CUBLAS_STATUS_SUCCESS) {
            g_storagellm_cublaslt = nullptr;
        }
    });
    return g_storagellm_cublas != nullptr;
}

static const float* storagellm_weight_ptr_fp32(
    const void* view_ptr,
    uint32_t expected_rows,
    uint32_t expected_cols
) {
    const auto* v =
        reinterpret_cast<const storagellm_device_tensor_batch_view*>(view_ptr);
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

static int storagellm_validate_cublas_task(
    const moe_grouped_expert_device_task_t& task
) {
    return task.gate_weight && task.up_weight && task.down_weight &&
        task.d_input && task.d_token_indices && task.d_token_weights &&
        task.assignment_count != 0 && task.d_accum &&
        task.input_stride >= task.hidden_size &&
        task.accum_stride >= task.hidden_size &&
        task.hidden_size != 0 && task.intermediate_size != 0 &&
        storagellm_weight_ptr_fp32(
            task.gate_weight, task.intermediate_size, task.hidden_size) &&
        storagellm_weight_ptr_fp32(
            task.up_weight, task.intermediate_size, task.hidden_size) &&
        storagellm_weight_ptr_fp32(
            task.down_weight, task.hidden_size, task.intermediate_size);
}

__global__ void storagellm_gather_rows_kernel(
    const float* input,
    uint32_t input_stride,
    const uint32_t* token_indices,
    uint32_t assignment_offset,
    uint32_t rows,
    uint32_t hidden,
    float* out
) {
    const uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t total = static_cast<uint64_t>(rows) * hidden;
    if (idx >= total) {
        return;
    }
    const uint64_t row = idx / hidden;
    const uint32_t col = static_cast<uint32_t>(idx - row * hidden);
    const uint32_t token = token_indices[assignment_offset + row];
    out[(static_cast<uint64_t>(assignment_offset) + row) * hidden + col] =
        input[static_cast<uint64_t>(token) * input_stride + col];
}

__device__ float storagellm_gelu_erf(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
}

__device__ float storagellm_gelu_tanh(float x) {
    const float k = 0.7978845608028654f;
    const float inner = k * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__device__ float storagellm_gated_activation(uint32_t mode, float gate, float up) {
    if (!isfinite(gate) || !isfinite(up)) {
        return 0.0f;
    }
    float activated = 0.0f;
    if (mode == 2u) {
        activated = storagellm_gelu_tanh(gate);
    } else if (mode == 1u) {
        activated = storagellm_gelu_erf(gate);
    } else {
        activated = gate > 40.0f ? gate : (gate < -40.0f ? 0.0f : gate / (1.0f + expf(-gate)));
    }
    const float result = activated * up;
    return isfinite(result) ? result : 0.0f;
}

__global__ void storagellm_silu_mul_rows_kernel(
    const float* gate,
    const float* up,
    uint32_t rows,
    uint32_t intermediate,
    uint32_t activation_mode,
    float* mid
) {
    const uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t total = static_cast<uint64_t>(rows) * intermediate;
    if (idx < total) {
        mid[idx] = storagellm_gated_activation(activation_mode, gate[idx], up[idx]);
    }
}

__global__ void storagellm_weighted_accum_rows_kernel(
    const float* rows,
    const uint32_t* token_indices,
    const float* weights,
    uint32_t assignment_offset,
    uint32_t row_count,
    uint32_t hidden,
    uint32_t accum_stride,
    float* accum
) {
    const uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t total = static_cast<uint64_t>(row_count) * hidden;
    if (idx >= total) {
        return;
    }
    const uint64_t row = idx / hidden;
    const uint32_t col = static_cast<uint32_t>(idx - row * hidden);
    const uint32_t global_row = assignment_offset + static_cast<uint32_t>(row);
    const uint32_t token = token_indices[global_row];
    const float v = rows[static_cast<uint64_t>(global_row) * hidden + col] *
        weights[global_row];
    atomicAdd(accum + static_cast<uint64_t>(token) * accum_stride + col, v);
}

static int storagellm_run_grouped_sgemm(
    const std::vector<cublasOperation_t>& trans_a,
    const std::vector<cublasOperation_t>& trans_b,
    const std::vector<int>& m,
    const std::vector<int>& n,
    const std::vector<int>& k,
    const std::vector<const float*>& a,
    const std::vector<int>& lda,
    const std::vector<const float*>& b,
    const std::vector<int>& ldb,
    const std::vector<float*>& c,
    const std::vector<int>& ldc,
    cudaStream_t stream
) {
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 12050
    const int group_count = static_cast<int>(a.size());
    if (group_count <= 0) {
        return 0;
    }
    const float** d_a = nullptr;
    const float** d_b = nullptr;
    float** d_c = nullptr;
    int ok = 0;
    std::vector<float> alpha(static_cast<size_t>(group_count), 1.0f);
    std::vector<float> beta(static_cast<size_t>(group_count), 0.0f);
    std::vector<int> group_size(static_cast<size_t>(group_count), 1);
    if (cudaMalloc(reinterpret_cast<void**>(&d_a), sizeof(float*) * a.size()) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&d_b), sizeof(float*) * b.size()) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&d_c), sizeof(float*) * c.size()) != cudaSuccess ||
        cudaMemcpyAsync(d_a, a.data(), sizeof(float*) * a.size(), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(d_b, b.data(), sizeof(float*) * b.size(), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(d_c, c.data(), sizeof(float*) * c.size(), cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        goto cleanup;
    }
    if (cublasSetStream(g_storagellm_cublas, stream) != CUBLAS_STATUS_SUCCESS) {
        goto cleanup;
    }
    ok = cublasSgemmGroupedBatched(
        g_storagellm_cublas,
        trans_a.data(),
        trans_b.data(),
        m.data(),
        n.data(),
        k.data(),
        alpha.data(),
        d_a,
        lda.data(),
        d_b,
        ldb.data(),
        beta.data(),
        d_c,
        ldc.data(),
        group_count,
        group_size.data()) == CUBLAS_STATUS_SUCCESS;

cleanup:
    if (d_c) cudaFree(d_c);
    if (d_b) cudaFree(d_b);
    if (d_a) cudaFree(d_a);
    return ok;
#else
    (void)trans_a;
    (void)trans_b;
    (void)m;
    (void)n;
    (void)k;
    (void)a;
    (void)lda;
    (void)b;
    (void)ldb;
    (void)c;
    (void)ldc;
    (void)stream;
    return 0;
#endif
}
#endif

extern "C" int storagellm_cublaslt_grouped_moe_indexed_device_f32(
    int32_t backend,
    const moe_grouped_expert_device_task_t* tasks,
    uint32_t task_count,
    void* stream_or_queue
) {
#if !defined(STORAGELLM_HAS_CUBLASLT)
    (void)backend;
    (void)tasks;
    (void)task_count;
    (void)stream_or_queue;
    return 0;
#else
    if (backend != moe_BACKEND_CUDA || !tasks || task_count == 0 ||
        !storagellm_ensure_cublas_grouped()) {
        return 0;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_or_queue);

    uint32_t hidden = tasks[0].hidden_size;
    uint32_t intermediate = tasks[0].intermediate_size;
    uint32_t activation_mode = tasks[0].activation_mode;
    uint32_t rows = 0;
    for (uint32_t i = 0; i < task_count; ++i) {
        if (!storagellm_validate_cublas_task(tasks[i]) ||
            tasks[i].hidden_size != hidden ||
            tasks[i].intermediate_size != intermediate ||
            tasks[i].activation_mode != activation_mode ||
            tasks[i].assignment_offset > UINT32_MAX - tasks[i].assignment_count) {
            return 0;
        }
        const uint32_t end = tasks[i].assignment_offset + tasks[i].assignment_count;
        rows = end > rows ? end : rows;
    }
    if (rows == 0 || hidden > static_cast<uint32_t>(INT_MAX) ||
        intermediate > static_cast<uint32_t>(INT_MAX)) {
        return 0;
    }

    const size_t x_bytes = static_cast<size_t>(rows) * hidden * sizeof(float);
    const size_t mid_bytes = static_cast<size_t>(rows) * intermediate * sizeof(float);
    int ok = 0;

    float* d_x = nullptr;
    float* d_gate = nullptr;
    float* d_up = nullptr;
    float* d_mid = nullptr;
    float* d_out = nullptr;
    auto& ws = g_storagellm_cublaslt_workspace;
    if (!ws.ensure(x_bytes, mid_bytes)) {
        goto cleanup;
    }
    d_x = ws.d_x;
    d_gate = ws.d_gate;
    d_up = ws.d_up;
    d_mid = ws.d_mid;
    d_out = ws.d_out;

    for (uint32_t i = 0; i < task_count; ++i) {
        const auto& t = tasks[i];
        const uint64_t gather_total =
            static_cast<uint64_t>(t.assignment_count) * hidden;
        const uint32_t block = 256;
        const uint32_t grid = static_cast<uint32_t>((gather_total + block - 1) / block);
        storagellm_gather_rows_kernel<<<grid, block, 0, stream>>>(
            static_cast<const float*>(t.d_input),
            t.input_stride,
            t.d_token_indices,
            t.assignment_offset,
            t.assignment_count,
            hidden,
            d_x);
    }
    if (cudaGetLastError() != cudaSuccess) {
        goto cleanup;
    }

    {
        std::vector<cublasOperation_t> trans_a(task_count, CUBLAS_OP_T);
        std::vector<cublasOperation_t> trans_b(task_count, CUBLAS_OP_N);
        std::vector<int> m(task_count, static_cast<int>(intermediate));
        std::vector<int> n(task_count);
        std::vector<int> k(task_count, static_cast<int>(hidden));
        std::vector<const float*> gate_w(task_count);
        std::vector<const float*> up_w(task_count);
        std::vector<const float*> x_ptr(task_count);
        std::vector<float*> gate_ptr(task_count);
        std::vector<float*> up_ptr(task_count);
        std::vector<int> lda(task_count, static_cast<int>(hidden));
        std::vector<int> ldb(task_count, static_cast<int>(hidden));
        std::vector<int> ldc(task_count, static_cast<int>(intermediate));
        for (uint32_t i = 0; i < task_count; ++i) {
            const auto& t = tasks[i];
            n[i] = static_cast<int>(t.assignment_count);
            gate_w[i] = storagellm_weight_ptr_fp32(t.gate_weight, intermediate, hidden);
            up_w[i] = storagellm_weight_ptr_fp32(t.up_weight, intermediate, hidden);
            x_ptr[i] = d_x + static_cast<size_t>(t.assignment_offset) * hidden;
            gate_ptr[i] = d_gate + static_cast<size_t>(t.assignment_offset) * intermediate;
            up_ptr[i] = d_up + static_cast<size_t>(t.assignment_offset) * intermediate;
        }
        if (!storagellm_run_grouped_sgemm(
                trans_a, trans_b, m, n, k, gate_w, lda, x_ptr, ldb, gate_ptr, ldc, stream) ||
            !storagellm_run_grouped_sgemm(
                trans_a, trans_b, m, n, k, up_w, lda, x_ptr, ldb, up_ptr, ldc, stream)) {
            goto cleanup;
        }
    }

    {
        const uint64_t act_total = static_cast<uint64_t>(rows) * intermediate;
        const uint32_t block = 256;
        const uint32_t grid = static_cast<uint32_t>((act_total + block - 1) / block);
        storagellm_silu_mul_rows_kernel<<<grid, block, 0, stream>>>(
            d_gate, d_up, rows, intermediate, activation_mode, d_mid);
    }
    if (cudaGetLastError() != cudaSuccess) {
        goto cleanup;
    }

    {
        std::vector<cublasOperation_t> trans_a(task_count, CUBLAS_OP_T);
        std::vector<cublasOperation_t> trans_b(task_count, CUBLAS_OP_N);
        std::vector<int> m(task_count, static_cast<int>(hidden));
        std::vector<int> n(task_count);
        std::vector<int> k(task_count, static_cast<int>(intermediate));
        std::vector<const float*> down_w(task_count);
        std::vector<const float*> mid_ptr(task_count);
        std::vector<float*> out_ptr(task_count);
        std::vector<int> lda(task_count, static_cast<int>(intermediate));
        std::vector<int> ldb(task_count, static_cast<int>(intermediate));
        std::vector<int> ldc(task_count, static_cast<int>(hidden));
        for (uint32_t i = 0; i < task_count; ++i) {
            const auto& t = tasks[i];
            n[i] = static_cast<int>(t.assignment_count);
            down_w[i] = storagellm_weight_ptr_fp32(t.down_weight, hidden, intermediate);
            mid_ptr[i] = d_mid + static_cast<size_t>(t.assignment_offset) * intermediate;
            out_ptr[i] = d_out + static_cast<size_t>(t.assignment_offset) * hidden;
        }
        if (!storagellm_run_grouped_sgemm(
                trans_a, trans_b, m, n, k, down_w, lda, mid_ptr, ldb, out_ptr, ldc, stream)) {
            goto cleanup;
        }
    }

    for (uint32_t i = 0; i < task_count; ++i) {
        const auto& t = tasks[i];
        const uint64_t accum_total =
            static_cast<uint64_t>(t.assignment_count) * hidden;
        const uint32_t block = 256;
        const uint32_t grid = static_cast<uint32_t>((accum_total + block - 1) / block);
        storagellm_weighted_accum_rows_kernel<<<grid, block, 0, stream>>>(
            d_out,
            t.d_token_indices,
            t.d_token_weights,
            t.assignment_offset,
            t.assignment_count,
            hidden,
            t.accum_stride,
            static_cast<float*>(t.d_accum));
    }
    if (cudaGetLastError() != cudaSuccess) {
        goto cleanup;
    }
    ok = 1;

cleanup:
    return ok;
#endif
}
