#include "benchmark.h"
#include "process.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#if defined(STORAGELLM_AUTOTUNE_HAS_CUBLASLT_NATIVE)
#include <cuda_runtime_api.h>
#endif

namespace storagellm::autotune {

struct Task {
    int32_t layer;
    int32_t expert;
    const void* gate_weight;
    const void* up_weight;
    const void* down_weight;
    const void* d_input;
    uint32_t input_stride;
    const uint32_t* d_token_indices;
    const float* d_token_weights;
    uint32_t assignment_offset;
    uint32_t assignment_count;
    void* d_accum;
    uint32_t accum_stride;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t activation_mode;
};

using Entry = int (*)(int32_t, const Task*, uint32_t, void*);

static int env_int_local(const char* name, int fallback) {
    const char* v = std::getenv(name);
    if (!v || !v[0]) return fallback;
    try { return std::stoi(v); } catch (...) { return fallback; }
}

template <typename Fn>
static int measure_adaptive_ms(Fn&& fn, double& latency_ms) {
    const int warmups = std::max(0, env_int_local("STORAGELLM_AUTOTUNE_WARMUPS", 3));
    const int min_iters = std::max(1, env_int_local("STORAGELLM_AUTOTUNE_MIN_ITERS", 8));
    const int max_iters = std::max(min_iters, env_int_local("STORAGELLM_AUTOTUNE_MAX_ITERS", 256));
    const double min_ms = std::max(1, env_int_local("STORAGELLM_AUTOTUNE_MIN_MS", 120));

    for (int i = 0; i < warmups; ++i) {
        if (!fn()) return 0;
    }

    int iters = 0;
    auto t0 = std::chrono::steady_clock::now();
    double elapsed_ms = 0.0;
    while (iters < max_iters) {
        if (!fn()) return 0;
        ++iters;
        if (iters >= min_iters) {
            auto now = std::chrono::steady_clock::now();
            elapsed_ms = std::chrono::duration<double, std::milli>(now - t0).count();
            if (elapsed_ms >= min_ms) break;
        }
    }
    if (iters <= 0 || elapsed_ms <= 0.0) return 0;
    latency_ms = elapsed_ms / static_cast<double>(iters);
    return 1;
}


#if defined(STORAGELLM_AUTOTUNE_HAS_CPU_NATIVE)
extern "C" int storagellm_onednn_cpu_grouped_moe_indexed_device_f32(
    int32_t backend,
    const Task* tasks,
    uint32_t task_count,
    void* stream_or_queue);
#endif

#if defined(STORAGELLM_AUTOTUNE_HAS_CUBLASLT_NATIVE)
extern "C" int storagellm_cublaslt_grouped_moe_indexed_device_f32(
    int32_t backend,
    const Task* tasks,
    uint32_t task_count,
    void* stream_or_queue);
#endif

static Entry load_entry(const std::string& path, void** handle, std::string& reason) {
#if defined(_WIN32)
    HMODULE h = LoadLibraryA(path.c_str());
    if (!h) { reason = "LoadLibrary failed"; return nullptr; }
    *handle = reinterpret_cast<void*>(h);
    auto fn = reinterpret_cast<Entry>(GetProcAddress(h, "storagellm_tvm_grouped_moe_entry"));
#else
    void* h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) { reason = dlerror(); return nullptr; }
    *handle = h;
    auto fn = reinterpret_cast<Entry>(dlsym(h, "storagellm_tvm_grouped_moe_entry"));
#endif
    if (!fn) reason = "entry symbol missing";
    return fn;
}

static void close_handle(void* h) {
    if (!h) return;
#if defined(_WIN32)
    FreeLibrary(reinterpret_cast<HMODULE>(h));
#else
    dlclose(h);
#endif
}

static void make_cpu_fixture(
    std::vector<float>& x,
    std::vector<float>& gate,
    std::vector<float>& up,
    std::vector<float>& down,
    std::vector<float>& accum,
    std::vector<float>& weights,
    std::vector<uint32_t>& idx,
    Task& task
) {
    const uint32_t H = 128;
    const uint32_t I = 256;
    const uint32_t A = 4;
    x.assign(H * A, 0.0f);
    gate.assign(I * H, 0.0f);
    up.assign(I * H, 0.0f);
    down.assign(H * I, 0.0f);
    accum.assign(H * A, 0.0f);
    weights.assign(A, 1.0f);
    idx.resize(A);
    for (uint32_t i = 0; i < A; ++i) idx[i] = i;
    for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin(double(i) * 0.01);
    for (size_t i = 0; i < gate.size(); ++i) { gate[i] = std::cos(double(i) * 0.003) * 0.01f; up[i] = std::sin(double(i) * 0.005) * 0.01f; }
    for (size_t i = 0; i < down.size(); ++i) down[i] = std::cos(double(i) * 0.007) * 0.01f;
    task = Task{};
    task.gate_weight = gate.data();
    task.up_weight = up.data();
    task.down_weight = down.data();
    task.d_input = x.data();
    task.input_stride = H;
    task.d_token_indices = idx.data();
    task.d_token_weights = weights.data();
    task.assignment_count = A;
    task.d_accum = accum.data();
    task.accum_stride = H;
    task.hidden_size = H;
    task.intermediate_size = I;
}

struct TensorView {
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

static TensorView make_view(float* p, uint32_t rows, uint32_t cols) {
    TensorView v{};
    v.ptr = reinterpret_cast<uint64_t>(p);
    v.bytes = static_cast<uint64_t>(rows) * cols * sizeof(float);
    v.weight_format = 2u;
    v.backend_kind = 0u;
    v.rows = rows;
    v.cols = cols;
    v.weight_row_bytes = static_cast<uint64_t>(cols) * sizeof(float);
    v.weight_bytes = v.bytes;
    v.stream_bytes = v.bytes;
    return v;
}

static void benchmark_cpu_native(Candidate& c) {
#if defined(STORAGELLM_AUTOTUNE_HAS_CPU_NATIVE)
    std::vector<float> x, gate, up, down, accum, weights;
    std::vector<uint32_t> idx;
    Task task{};
    make_cpu_fixture(x, gate, up, down, accum, weights, idx, task);
    TensorView gate_view = make_view(gate.data(), task.intermediate_size, task.hidden_size);
    TensorView up_view = make_view(up.data(), task.intermediate_size, task.hidden_size);
    TensorView down_view = make_view(down.data(), task.hidden_size, task.intermediate_size);
    task.gate_weight = &gate_view;
    task.up_weight = &up_view;
    task.down_weight = &down_view;
    const int backend_cpu = 1;
    if (!storagellm_onednn_cpu_grouped_moe_indexed_device_f32(backend_cpu, &task, 1, nullptr)) {
        c.reason = "CPU native adapter returned failure during warmup";
        return;
    }
    double latency = 0.0;
    const int measured_ok = measure_adaptive_ms([&]() {
        std::fill(accum.begin(), accum.end(), 0.0f);
        return storagellm_onednn_cpu_grouped_moe_indexed_device_f32(
            backend_cpu, &task, 1, nullptr) != 0;
    }, latency);
    if (!measured_ok) {
        c.reason = "CPU native adapter returned failure during adaptive measurement";
        return;
    }
    c.compiled = true;
    c.loadable = true;
    c.measured = true;
    c.latency_ms = latency;
    c.reason = "measured C++ CPU native F32 grouped-MoE adapter with warmup/adaptive iterations on this machine";
#else
    c.reason = "CPU native adapter was not linked into storagellm_host_autotune";
#endif
}


#if defined(STORAGELLM_AUTOTUNE_HAS_CUBLASLT_NATIVE)
template <typename T>
static int cuda_alloc_copy(T*& device_ptr, const std::vector<T>& host) {
    device_ptr = nullptr;
    if (host.empty()) return 0;
    const size_t bytes = host.size() * sizeof(T);
    if (cudaMalloc(reinterpret_cast<void**>(&device_ptr), bytes) != cudaSuccess) return 0;
    if (cudaMemcpy(device_ptr, host.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(device_ptr);
        device_ptr = nullptr;
        return 0;
    }
    return 1;
}

static void benchmark_cuda_cublaslt_native(Candidate& c) {
    const uint32_t H = 128;
    const uint32_t I = 256;
    const uint32_t A = 4;
    const uint32_t E = 2;
    std::vector<float> x(H * A), gate(E * I * H), up(E * I * H), down(E * H * I),
        accum(H * A, 0.0f), weights(E * A, 1.0f);
    std::vector<uint32_t> idx(E * A);
    for (uint32_t i = 0; i < A; ++i) {
        idx[i] = i;
        idx[A + i] = i;
    }
    for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin(double(i) * 0.01);
    for (size_t i = 0; i < gate.size(); ++i) {
        gate[i] = std::cos(double(i) * 0.003) * 0.01f;
        up[i] = std::sin(double(i) * 0.005) * 0.01f;
    }
    for (size_t i = 0; i < down.size(); ++i) down[i] = std::cos(double(i) * 0.007) * 0.01f;

    float *d_x = nullptr, *d_gate = nullptr, *d_up = nullptr, *d_down = nullptr, *d_accum = nullptr, *d_weights = nullptr;
    uint32_t* d_idx = nullptr;
    cudaStream_t stream = nullptr;
    int ok = 0;
    if (!cuda_alloc_copy(d_x, x) || !cuda_alloc_copy(d_gate, gate) ||
        !cuda_alloc_copy(d_up, up) || !cuda_alloc_copy(d_down, down) ||
        !cuda_alloc_copy(d_weights, weights) || !cuda_alloc_copy(d_idx, idx) ||
        cudaMalloc(reinterpret_cast<void**>(&d_accum), accum.size() * sizeof(float)) != cudaSuccess ||
        cudaStreamCreate(&stream) != cudaSuccess) {
        c.reason = "CUDA allocation/copy/stream setup failed for native cuBLASLt probe";
        goto cleanup;
    }
    if (cudaMemsetAsync(d_accum, 0, accum.size() * sizeof(float), stream) != cudaSuccess) {
        c.reason = "cudaMemsetAsync failed for native cuBLASLt probe";
        goto cleanup;
    }

    {
        TensorView gate_v[E];
        TensorView up_v[E];
        TensorView down_v[E];
        Task tasks[E]{};
        for (uint32_t e = 0; e < E; ++e) {
            gate_v[e] = make_view(d_gate + size_t(e) * I * H, I, H);
            up_v[e] = make_view(d_up + size_t(e) * I * H, I, H);
            down_v[e] = make_view(d_down + size_t(e) * H * I, H, I);
            tasks[e].layer = 0;
            tasks[e].expert = static_cast<int32_t>(e);
            tasks[e].gate_weight = &gate_v[e];
            tasks[e].up_weight = &up_v[e];
            tasks[e].down_weight = &down_v[e];
            tasks[e].d_input = d_x;
            tasks[e].input_stride = H;
            tasks[e].d_token_indices = d_idx;
            tasks[e].d_token_weights = d_weights;
            tasks[e].assignment_offset = e * A;
            tasks[e].assignment_count = A;
            tasks[e].d_accum = d_accum;
            tasks[e].accum_stride = H;
            tasks[e].hidden_size = H;
            tasks[e].intermediate_size = I;
            tasks[e].activation_mode = 0;
        }
        const int backend_cuda = 2;
        if (!storagellm_cublaslt_grouped_moe_indexed_device_f32(backend_cuda, tasks, E, stream) ||
            cudaStreamSynchronize(stream) != cudaSuccess) {
            c.reason = "native cuBLASLt adapter returned failure during warmup/sync";
            goto cleanup;
        }
        double latency = 0.0;
        const int measured_ok = measure_adaptive_ms([&]() {
            if (cudaMemsetAsync(d_accum, 0, accum.size() * sizeof(float), stream) != cudaSuccess) return false;
            if (!storagellm_cublaslt_grouped_moe_indexed_device_f32(backend_cuda, tasks, E, stream)) return false;
            return cudaStreamSynchronize(stream) == cudaSuccess;
        }, latency);
        if (!measured_ok) {
            c.reason = "native cuBLASLt adapter returned failure during adaptive measurement";
            goto cleanup;
        }
        c.compiled = true;
        c.loadable = true;
        c.measured = true;
        c.latency_ms = latency;
        c.reason = "measured native CUDA cuBLASLt grouped-MoE adapter on this GPU with warmup/adaptive iterations";
        ok = 1;
    }

cleanup:
    if (!ok && c.reason.empty()) c.reason = "native cuBLASLt probe failed";
    if (stream) cudaStreamDestroy(stream);
    if (d_accum) cudaFree(d_accum);
    if (d_idx) cudaFree(d_idx);
    if (d_weights) cudaFree(d_weights);
    if (d_down) cudaFree(d_down);
    if (d_up) cudaFree(d_up);
    if (d_gate) cudaFree(d_gate);
    if (d_x) cudaFree(d_x);
}
#else
static void benchmark_cuda_cublaslt_native(Candidate& c) {
    c.reason = "native cuBLASLt adapter was not linked into storagellm_host_autotune";
}
#endif

static void benchmark_tvm_cpu(Candidate& c) {
    if (!file_exists(c.library)) {
        c.reason = "library missing";
        return;
    }
    void* handle = nullptr;
    std::string reason;
    Entry fn = load_entry(c.library, &handle, reason);
    if (!fn) {
        c.loadable = false;
        c.reason = reason;
        close_handle(handle);
        return;
    }
    c.loadable = true;

    const uint32_t H = 128;
    const uint32_t I = 256;
    const uint32_t A = 4;
    std::vector<float> x(H * A), gate(I * H), up(I * H), down(H * I), accum(H * A), weights(A, 1.0f);
    std::vector<uint32_t> idx(A);
    for (uint32_t i = 0; i < A; ++i) idx[i] = i;
    for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin(double(i) * 0.01);
    for (size_t i = 0; i < gate.size(); ++i) { gate[i] = std::cos(double(i) * 0.003) * 0.01f; up[i] = std::sin(double(i) * 0.005) * 0.01f; }
    for (size_t i = 0; i < down.size(); ++i) down[i] = std::cos(double(i) * 0.007) * 0.01f;

    Task task{};
    task.gate_weight = gate.data();
    task.up_weight = up.data();
    task.down_weight = down.data();
    task.d_input = x.data();
    task.input_stride = H;
    task.d_token_indices = idx.data();
    task.d_token_weights = weights.data();
    task.assignment_count = A;
    task.d_accum = accum.data();
    task.accum_stride = H;
    task.hidden_size = H;
    task.intermediate_size = I;

    if (!fn(1, &task, 1, nullptr)) {
        c.reason = "entry returned failure during warmup";
        close_handle(handle);
        return;
    }

    double latency = 0.0;
    const int measured_ok = measure_adaptive_ms([&]() {
        std::fill(accum.begin(), accum.end(), 0.0f);
        return fn(1, &task, 1, nullptr) != 0;
    }, latency);
    if (!measured_ok) {
        c.reason = "entry returned failure during adaptive measurement";
        close_handle(handle);
        return;
    }
    c.measured = true;
    c.latency_ms = latency;
    c.reason = "measured by C++ host_autotune with warmup/adaptive iterations on this machine";
    close_handle(handle);
}

void benchmark_candidates(std::vector<Candidate>& candidates) {
    for (auto& c : candidates) {
        if (c.name == "cpu_native_f32") {
            benchmark_cpu_native(c);
        } else if (c.name == "cuda_cublaslt") {
            benchmark_cuda_cublaslt_native(c);
        } else if (c.kind == "tvm" && c.name == "tvm_cpu") {
            benchmark_tvm_cpu(c);
        } else if (c.kind == "native_unimplemented") {
            c.compiled = false;
            c.loadable = false;
            c.measured = false;
            c.reason = "not selected: native platform adapter still needs SDK-specific device-kernel implementation and cannot be counted as maximum-speed success";
        } else if (c.compiled) {
            c.reason = c.reason.empty() ? "compiled but not benchmarked by generic C++ probe" : c.reason;
        }
    }
}

} // namespace storagellm::autotune
