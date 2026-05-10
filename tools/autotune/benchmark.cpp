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

#if defined(STORAGELLM_AUTOTUNE_HAS_CUDA_RUNTIME)
#include <cuda_runtime_api.h>
#endif

#if defined(STORAGELLM_AUTOTUNE_HAS_HIP_RUNTIME)
#if defined(__has_include)
#if __has_include(<hip/hip_runtime.h>)
#include <hip/hip_runtime.h>
#define STORAGELLM_AUTOTUNE_HIP_USABLE 1
#endif
#endif
#endif

#if defined(STORAGELLM_AUTOTUNE_HAS_OPENCL_RUNTIME)
#if defined(__has_include)
#if __has_include(<CL/cl.h>)
#include <CL/cl.h>
#define STORAGELLM_AUTOTUNE_OPENCL_USABLE 1
#endif
#endif
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

#if defined(STORAGELLM_AUTOTUNE_HAS_CUTLASS_NATIVE)
extern "C" int storagellm_cutlass_grouped_moe_indexed_device_f32(
    int32_t backend,
    const Task* tasks,
    uint32_t task_count,
    void* stream_or_queue);
#endif

#if defined(STORAGELLM_AUTOTUNE_HAS_HIPBLASLT_NATIVE)
extern "C" int storagellm_hipblaslt_grouped_moe_indexed_device_f32(
    int32_t backend,
    const Task* tasks,
    uint32_t task_count,
    void* stream_or_queue);
#endif

#if defined(STORAGELLM_AUTOTUNE_HAS_CK_NATIVE)
extern "C" int storagellm_ck_grouped_moe_indexed_device_f32(
    int32_t backend,
    const Task* tasks,
    uint32_t task_count,
    void* stream_or_queue);
#endif

#if defined(STORAGELLM_AUTOTUNE_HAS_CLBLAST_NATIVE)
extern "C" int storagellm_clblast_grouped_moe_indexed_device_f32(
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

static float reference_activation(uint32_t mode, float gate, float up) {
    if (!std::isfinite(gate) || !std::isfinite(up)) return 0.0f;
    float activated = 0.0f;
    if (mode == 2u) {
        const float k = 0.7978845608028654f;
        const float inner = k * (gate + 0.044715f * gate * gate * gate);
        activated = 0.5f * gate * (1.0f + std::tanh(inner));
    } else if (mode == 1u) {
        activated = 0.5f * gate * (1.0f + std::erf(gate * 0.7071067811865476f));
    } else {
        activated = gate > 40.0f ? gate : (gate < -40.0f ? 0.0f : gate / (1.0f + std::exp(-gate)));
    }
    const float y = activated * up;
    return std::isfinite(y) ? y : 0.0f;
}

static void reference_grouped_moe_task(
    std::vector<float>& accum_ref,
    const std::vector<float>& x,
    const std::vector<float>& gate,
    const std::vector<float>& up,
    const std::vector<float>& down,
    const std::vector<float>& weights,
    const std::vector<uint32_t>& idx,
    uint32_t assignment_offset,
    uint32_t assignment_count,
    uint32_t H,
    uint32_t I,
    uint32_t activation_mode,
    size_t gate_offset,
    size_t up_offset,
    size_t down_offset
) {
    std::vector<float> mid(I, 0.0f);
    for (uint32_t local = 0; local < assignment_count; ++local) {
        const uint32_t global = assignment_offset + local;
        if (global >= idx.size()) continue;
        const uint32_t token = idx[global];
        if (static_cast<uint64_t>(token + 1u) * H > x.size() ||
            static_cast<uint64_t>(token + 1u) * H > accum_ref.size()) {
            continue;
        }
        const float route = global < weights.size() ? weights[global] : 1.0f;
        if (!std::isfinite(route)) continue;
        const float* input = x.data() + static_cast<size_t>(token) * H;
        for (uint32_t r = 0; r < I; ++r) {
            float g = 0.0f;
            float u = 0.0f;
            const float* gw = gate.data() + gate_offset + static_cast<size_t>(r) * H;
            const float* uw = up.data() + up_offset + static_cast<size_t>(r) * H;
            for (uint32_t h = 0; h < H; ++h) {
                const float xv = input[h];
                g += gw[h] * xv;
                u += uw[h] * xv;
            }
            mid[r] = reference_activation(activation_mode, g, u);
        }
        float* dst = accum_ref.data() + static_cast<size_t>(token) * H;
        for (uint32_t h = 0; h < H; ++h) {
            const float* dw = down.data() + down_offset + static_cast<size_t>(h) * I;
            float y = 0.0f;
            for (uint32_t r = 0; r < I; ++r) y += dw[r] * mid[r];
            dst[h] += y * route;
        }
    }
}

static bool validate_accum_against_reference(
    Candidate& c,
    const std::vector<float>& got,
    const std::vector<float>& ref,
    const char* label,
    double abs_tol = 2.5e-3,
    double rel_tol = 2.5e-3
) {
    if (got.size() != ref.size() || got.empty()) {
        c.reason = std::string(label ? label : "backend") + " correctness validation failed: output size mismatch";
        c.validation = "correctness-failed";
        return false;
    }
    double max_abs = 0.0;
    double max_rel = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double a = static_cast<double>(got[i]);
        const double b = static_cast<double>(ref[i]);
        if (!std::isfinite(a) || !std::isfinite(b)) {
            c.reason = std::string(label ? label : "backend") + " correctness validation failed: non-finite output";
            c.validation = "correctness-failed";
            return false;
        }
        const double abs_err = std::fabs(a - b);
        const double rel_err = abs_err / std::max(1.0, std::fabs(b));
        max_abs = std::max(max_abs, abs_err);
        max_rel = std::max(max_rel, rel_err);
    }
    c.correctness_max_abs = max_abs;
    c.correctness_max_rel = max_rel;
    if (max_abs > abs_tol && max_rel > rel_tol) {
        c.reason = std::string(label ? label : "backend") + " correctness validation failed: max_abs=" +
            std::to_string(max_abs) + " max_rel=" + std::to_string(max_rel);
        c.validation = "correctness-failed";
        return false;
    }
    c.verified = true;
    return true;
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
    std::fill(accum.begin(), accum.end(), 0.0f);
    if (!storagellm_onednn_cpu_grouped_moe_indexed_device_f32(backend_cpu, &task, 1, nullptr)) {
        c.reason = "CPU native adapter returned failure during warmup";
        return;
    }
    std::vector<float> ref(accum.size(), 0.0f);
    reference_grouped_moe_task(ref, x, gate, up, down, weights, idx, 0u, task.assignment_count,
        task.hidden_size, task.intermediate_size, task.activation_mode, 0u, 0u, 0u);
    if (!validate_accum_against_reference(c, accum, ref, "CPU native adapter")) {
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
    c.true_kernel = true;
    c.fused_moe = true;
    c.latency_ms = latency;
    c.validation = "linked+warmup+correctness+adaptive-measurement";
    c.reason = "measured and correctness-verified C++ CPU native F32 grouped-MoE adapter with warmup/adaptive iterations on this machine";
#else
    c.reason = "CPU native adapter was not linked into storagellm_host_autotune";
    c.validation = "not-linked";
#endif
}


#if defined(STORAGELLM_AUTOTUNE_HAS_CUDA_RUNTIME)
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

static void benchmark_cuda_indexed_native(Candidate& c, Entry fn, const char* label, bool fused) {
    if (!fn) {
        c.reason = std::string(label ? label : "CUDA backend") + " entry symbol was not linked";
        return;
    }
    int device_count = 0;
    const cudaError_t dev_rc = cudaGetDeviceCount(&device_count);
    if (dev_rc != cudaSuccess || device_count <= 0) {
        c.reason = std::string(label ? label : "CUDA backend") + " runtime device check failed: no CUDA device";
        c.validation = "runtime-device-missing";
        return;
    }
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
        c.reason = std::string(label ? label : "CUDA backend") + " allocation/copy/stream setup failed";
        goto cleanup;
    }
    if (cudaMemsetAsync(d_accum, 0, accum.size() * sizeof(float), stream) != cudaSuccess) {
        c.reason = std::string(label ? label : "CUDA backend") + " cudaMemsetAsync failed";
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
        if (!fn(backend_cuda, tasks, E, stream) || cudaStreamSynchronize(stream) != cudaSuccess) {
            c.reason = std::string(label ? label : "CUDA backend") + " returned failure during warmup/sync";
            c.validation = "warmup-failed";
            goto cleanup;
        }
        if (cudaMemcpy(accum.data(), d_accum, accum.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            c.reason = std::string(label ? label : "CUDA backend") + " failed to copy correctness output to host";
            c.validation = "correctness-copy-failed";
            goto cleanup;
        }
        {
            std::vector<float> ref(accum.size(), 0.0f);
            for (uint32_t e = 0; e < E; ++e) {
                reference_grouped_moe_task(ref, x, gate, up, down, weights, idx, e * A, A, H, I, 0u,
                    static_cast<size_t>(e) * I * H,
                    static_cast<size_t>(e) * I * H,
                    static_cast<size_t>(e) * H * I);
            }
            if (!validate_accum_against_reference(c, accum, ref, label ? label : "CUDA backend", 6.0e-3, 6.0e-3)) {
                goto cleanup;
            }
        }
        double latency = 0.0;
        const int measured_ok = measure_adaptive_ms([&]() {
            if (cudaMemsetAsync(d_accum, 0, accum.size() * sizeof(float), stream) != cudaSuccess) return false;
            if (!fn(backend_cuda, tasks, E, stream)) return false;
            return cudaStreamSynchronize(stream) == cudaSuccess;
        }, latency);
        if (!measured_ok) {
            c.reason = std::string(label ? label : "CUDA backend") + " returned failure during adaptive measurement";
            c.validation = "measurement-failed";
            goto cleanup;
        }
        c.compiled = true;
        c.loadable = true;
        c.measured = true;
        c.true_kernel = true;
        c.fused_moe = fused;
        c.latency_ms = latency;
        c.validation = "linked+runtime-device+warmup+correctness+adaptive-measurement";
        c.reason = std::string("measured and correctness-verified ") + (label ? label : "CUDA backend") + " on this CUDA device with warmup/adaptive iterations";
        ok = 1;
    }

cleanup:
    if (!ok && c.reason.empty()) c.reason = std::string(label ? label : "CUDA backend") + " probe failed";
    if (stream) cudaStreamDestroy(stream);
    if (d_accum) cudaFree(d_accum);
    if (d_idx) cudaFree(d_idx);
    if (d_weights) cudaFree(d_weights);
    if (d_down) cudaFree(d_down);
    if (d_up) cudaFree(d_up);
    if (d_gate) cudaFree(d_gate);
    if (d_x) cudaFree(d_x);
}
#endif

static void benchmark_cuda_cublaslt_native(Candidate& c) {
#if defined(STORAGELLM_AUTOTUNE_HAS_CUDA_RUNTIME) && defined(STORAGELLM_AUTOTUNE_HAS_CUBLASLT_NATIVE)
    benchmark_cuda_indexed_native(c, storagellm_cublaslt_grouped_moe_indexed_device_f32, "native CUDA cuBLASLt grouped-MoE adapter", false);
#else
    c.reason = "native cuBLASLt adapter was not linked into storagellm_host_autotune";
    c.validation = "not-linked";
#endif
}

static void benchmark_cuda_cutlass_native(Candidate& c) {
#if defined(STORAGELLM_AUTOTUNE_HAS_CUDA_RUNTIME) && defined(STORAGELLM_AUTOTUNE_HAS_CUTLASS_NATIVE)
    benchmark_cuda_indexed_native(c, storagellm_cutlass_grouped_moe_indexed_device_f32, "native CUDA CUTLASS/fused-kernel grouped-MoE adapter", true);
#else
    c.reason = "native CUTLASS/fused CUDA adapter was not linked into storagellm_host_autotune";
    c.validation = "not-linked";
#endif
}


#if defined(STORAGELLM_AUTOTUNE_HIP_USABLE)
template <typename T>
static int hip_alloc_copy(T*& device_ptr, const std::vector<T>& host) {
    device_ptr = nullptr;
    if (host.empty()) return 0;
    const size_t bytes = host.size() * sizeof(T);
    if (hipMalloc(reinterpret_cast<void**>(&device_ptr), bytes) != hipSuccess) return 0;
    if (hipMemcpy(device_ptr, host.data(), bytes, hipMemcpyHostToDevice) != hipSuccess) {
        hipFree(device_ptr);
        device_ptr = nullptr;
        return 0;
    }
    return 1;
}

static void benchmark_hip_indexed_native(Candidate& c, Entry fn, const char* label) {
    if (!fn) {
        c.reason = std::string(label ? label : "HIP backend") + " entry symbol was not linked";
        return;
    }
    int device_count = 0;
    const hipError_t dev_rc = hipGetDeviceCount(&device_count);
    if (dev_rc != hipSuccess || device_count <= 0) {
        c.reason = std::string(label ? label : "HIP backend") + " runtime device check failed: no HIP/ROCm device";
        c.validation = "runtime-device-missing";
        return;
    }
    if (hipSetDevice(0) != hipSuccess) {
        c.reason = std::string(label ? label : "HIP backend") + " could not select device 0";
        c.validation = "runtime-device-unusable";
        return;
    }

    const uint32_t H = 128;
    const uint32_t I = 256;
    const uint32_t A = 4;
    const uint32_t E = 2;
    std::vector<float> x(H * A), gate(E * I * H), up(E * I * H), down(E * H * I),
        accum(H * A, 0.0f), weights(E * A, 1.0f);
    std::vector<uint32_t> idx(E * A);
    for (uint32_t i = 0; i < A; ++i) { idx[i] = i; idx[A + i] = i; }
    for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin(double(i) * 0.01);
    for (size_t i = 0; i < gate.size(); ++i) {
        gate[i] = std::cos(double(i) * 0.003) * 0.01f;
        up[i] = std::sin(double(i) * 0.005) * 0.01f;
    }
    for (size_t i = 0; i < down.size(); ++i) down[i] = std::cos(double(i) * 0.007) * 0.01f;

    float *d_x = nullptr, *d_gate = nullptr, *d_up = nullptr, *d_down = nullptr, *d_accum = nullptr, *d_weights = nullptr;
    uint32_t* d_idx = nullptr;
    hipStream_t stream = nullptr;
    int ok = 0;
    if (!hip_alloc_copy(d_x, x) || !hip_alloc_copy(d_gate, gate) ||
        !hip_alloc_copy(d_up, up) || !hip_alloc_copy(d_down, down) ||
        !hip_alloc_copy(d_weights, weights) || !hip_alloc_copy(d_idx, idx) ||
        hipMalloc(reinterpret_cast<void**>(&d_accum), accum.size() * sizeof(float)) != hipSuccess ||
        hipStreamCreate(&stream) != hipSuccess) {
        c.reason = std::string(label ? label : "HIP backend") + " allocation/copy/stream setup failed";
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
        const int backend_hip = 7;
        if (hipMemsetAsync(d_accum, 0, accum.size() * sizeof(float), stream) != hipSuccess ||
            !fn(backend_hip, tasks, E, stream) || hipStreamSynchronize(stream) != hipSuccess) {
            c.reason = std::string(label ? label : "HIP backend") + " returned failure during warmup/sync";
            c.validation = "warmup-failed";
            goto cleanup;
        }
        if (hipMemcpy(accum.data(), d_accum, accum.size() * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess) {
            c.reason = std::string(label ? label : "HIP backend") + " failed to copy correctness output to host";
            c.validation = "correctness-copy-failed";
            goto cleanup;
        }
        {
            std::vector<float> ref(accum.size(), 0.0f);
            for (uint32_t e = 0; e < E; ++e) {
                reference_grouped_moe_task(ref, x, gate, up, down, weights, idx, e * A, A, H, I, 0u,
                    static_cast<size_t>(e) * I * H,
                    static_cast<size_t>(e) * I * H,
                    static_cast<size_t>(e) * H * I);
            }
            if (!validate_accum_against_reference(c, accum, ref, label ? label : "HIP backend", 6.0e-3, 6.0e-3)) {
                goto cleanup;
            }
        }
        double latency = 0.0;
        const int measured_ok = measure_adaptive_ms([&]() {
            if (hipMemsetAsync(d_accum, 0, accum.size() * sizeof(float), stream) != hipSuccess) return false;
            if (!fn(backend_hip, tasks, E, stream)) return false;
            return hipStreamSynchronize(stream) == hipSuccess;
        }, latency);
        if (!measured_ok) {
            c.reason = std::string(label ? label : "HIP backend") + " returned failure during adaptive measurement";
            c.validation = "measurement-failed";
            goto cleanup;
        }
        c.compiled = true;
        c.loadable = true;
        c.measured = true;
        c.true_kernel = true;
        c.fused_moe = true;
        c.latency_ms = latency;
        c.validation = "linked+runtime-device+warmup+correctness+adaptive-measurement";
        c.reason = std::string("measured and correctness-verified ") + (label ? label : "HIP backend") + " on this ROCm device with warmup/adaptive iterations";
        ok = 1;
    }

cleanup:
    if (!ok && c.reason.empty()) c.reason = std::string(label ? label : "HIP backend") + " probe failed";
    if (stream) hipStreamDestroy(stream);
    if (d_accum) hipFree(d_accum);
    if (d_idx) hipFree(d_idx);
    if (d_weights) hipFree(d_weights);
    if (d_down) hipFree(d_down);
    if (d_up) hipFree(d_up);
    if (d_gate) hipFree(d_gate);
    if (d_x) hipFree(d_x);
}
#endif

static void benchmark_rocm_hipblaslt_native(Candidate& c) {
#if defined(STORAGELLM_AUTOTUNE_HIP_USABLE) && defined(STORAGELLM_AUTOTUNE_HAS_HIPBLASLT_NATIVE)
    benchmark_hip_indexed_native(c, storagellm_hipblaslt_grouped_moe_indexed_device_f32, "native ROCm hipBLASLt/HIP fused grouped-MoE adapter");
#else
    c.reason = "native ROCm hipBLASLt adapter was not linked with a usable HIP runtime into storagellm_host_autotune";
    c.validation = "not-linked-or-runtime-header-missing";
#endif
}

static void benchmark_rocm_ck_native(Candidate& c) {
#if defined(STORAGELLM_AUTOTUNE_HIP_USABLE) && defined(STORAGELLM_AUTOTUNE_HAS_CK_NATIVE)
    benchmark_hip_indexed_native(c, storagellm_ck_grouped_moe_indexed_device_f32, "native ROCm CK/HIP fused grouped-MoE adapter");
#else
    c.reason = "native ROCm CK adapter was not linked with a usable HIP runtime into storagellm_host_autotune";
    c.validation = "not-linked-or-runtime-header-missing";
#endif
}

#if defined(STORAGELLM_AUTOTUNE_OPENCL_USABLE)
static int cl_choose_device(cl_platform_id* out_platform, cl_device_id* out_device) {
    if (!out_platform || !out_device) return 0;
    *out_platform = nullptr;
    *out_device = nullptr;
    cl_uint platform_count = 0;
    if (clGetPlatformIDs(0, nullptr, &platform_count) != CL_SUCCESS || platform_count == 0) return 0;
    std::vector<cl_platform_id> platforms(platform_count);
    if (clGetPlatformIDs(platform_count, platforms.data(), nullptr) != CL_SUCCESS) return 0;
    for (cl_platform_id p : platforms) {
        for (cl_device_type ty : {CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_ACCELERATOR, CL_DEVICE_TYPE_CPU}) {
            cl_uint n = 0;
            if (clGetDeviceIDs(p, ty, 0, nullptr, &n) != CL_SUCCESS || n == 0) continue;
            std::vector<cl_device_id> devs(n);
            if (clGetDeviceIDs(p, ty, n, devs.data(), nullptr) == CL_SUCCESS && !devs.empty()) {
                *out_platform = p;
                *out_device = devs[0];
                return 1;
            }
        }
    }
    return 0;
}

template <typename T>
static cl_mem cl_make_buffer(cl_context ctx, cl_command_queue q, cl_mem_flags flags, const std::vector<T>& host, int* ok) {
    if (!ok || !*ok || host.empty()) { if (ok) *ok = 0; return nullptr; }
    cl_int err = CL_SUCCESS;
    cl_mem m = clCreateBuffer(ctx, flags, host.size() * sizeof(T), nullptr, &err);
    if (err != CL_SUCCESS || !m) { *ok = 0; return nullptr; }
    err = clEnqueueWriteBuffer(q, m, CL_TRUE, 0, host.size() * sizeof(T), host.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) { clReleaseMemObject(m); *ok = 0; return nullptr; }
    return m;
}

static void benchmark_opencl_clblast_native(Candidate& c) {
#if defined(STORAGELLM_AUTOTUNE_HAS_CLBLAST_NATIVE)
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    if (!cl_choose_device(&platform, &device)) {
        c.reason = "OpenCL runtime device check failed: no OpenCL device";
        c.validation = "runtime-device-missing";
        return;
    }
    cl_int err = CL_SUCCESS;
    cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS || !ctx) {
        c.reason = "OpenCL context creation failed";
        c.validation = "runtime-device-unusable";
        return;
    }
#if defined(CL_VERSION_2_0)
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, device, nullptr, &err);
#else
    cl_command_queue q = clCreateCommandQueue(ctx, device, 0, &err);
#endif
    if (err != CL_SUCCESS || !q) {
        clReleaseContext(ctx);
        c.reason = "OpenCL command queue creation failed";
        c.validation = "runtime-device-unusable";
        return;
    }

    const uint32_t H = 128;
    const uint32_t I = 256;
    const uint32_t A = 4;
    const uint32_t E = 2;
    std::vector<float> x(H * A), gate(E * I * H), up(E * I * H), down(E * H * I),
        accum(H * A, 0.0f), weights(E * A, 1.0f);
    std::vector<uint32_t> idx(E * A);
    for (uint32_t i = 0; i < A; ++i) { idx[i] = i; idx[A + i] = i; }
    for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin(double(i) * 0.01);
    for (size_t i = 0; i < gate.size(); ++i) { gate[i] = std::cos(double(i) * 0.003) * 0.01f; up[i] = std::sin(double(i) * 0.005) * 0.01f; }
    for (size_t i = 0; i < down.size(); ++i) down[i] = std::cos(double(i) * 0.007) * 0.01f;

    int ok = 1;
    cl_mem d_x = cl_make_buffer(ctx, q, CL_MEM_READ_ONLY, x, &ok);
    cl_mem d_gate = cl_make_buffer(ctx, q, CL_MEM_READ_ONLY, gate, &ok);
    cl_mem d_up = cl_make_buffer(ctx, q, CL_MEM_READ_ONLY, up, &ok);
    cl_mem d_down = cl_make_buffer(ctx, q, CL_MEM_READ_ONLY, down, &ok);
    cl_mem d_weights = cl_make_buffer(ctx, q, CL_MEM_READ_ONLY, weights, &ok);
    cl_mem d_idx = cl_make_buffer(ctx, q, CL_MEM_READ_ONLY, idx, &ok);
    cl_mem d_accum = nullptr;
    if (ok) {
        d_accum = clCreateBuffer(ctx, CL_MEM_READ_WRITE, accum.size() * sizeof(float), nullptr, &err);
        if (err != CL_SUCCESS || !d_accum) ok = 0;
    }
    if (!ok || clFinish(q) != CL_SUCCESS) {
        c.reason = "OpenCL allocation/copy setup failed";
        c.validation = "fixture-setup-failed";
        goto cleanup;
    }

    {
        TensorView gate_v[E];
        TensorView up_v[E];
        TensorView down_v[E];
        Task tasks[E]{};
        for (uint32_t e = 0; e < E; ++e) {
            gate_v[e] = make_view(reinterpret_cast<float*>(d_gate), I, H);
            up_v[e] = make_view(reinterpret_cast<float*>(d_up), I, H);
            down_v[e] = make_view(reinterpret_cast<float*>(d_down), H, I);
            // OpenCL keeps all experts in one cl_mem per matrix.  Force the
            // adapter's expert-layout path so each task uses its byte offset;
            // raw-FP32 path would point every expert at offset 0.
            gate_v[e].weight_format = 0u;
            up_v[e].weight_format = 0u;
            down_v[e].weight_format = 0u;
            gate_v[e].bytes = uint64_t(E) * I * H * sizeof(float);
            up_v[e].bytes = uint64_t(E) * I * H * sizeof(float);
            down_v[e].bytes = uint64_t(E) * H * I * sizeof(float);
            gate_v[e].weight_bytes = gate_v[e].bytes;
            up_v[e].weight_bytes = up_v[e].bytes;
            down_v[e].weight_bytes = down_v[e].bytes;
            gate_v[e].expert_gpu_layout_kind = 3u;
            up_v[e].expert_gpu_layout_kind = 3u;
            down_v[e].expert_gpu_layout_kind = 3u;
            gate_v[e].expert_gpu_layout_offset = uint64_t(e) * I * H * sizeof(float);
            up_v[e].expert_gpu_layout_offset = uint64_t(e) * I * H * sizeof(float);
            down_v[e].expert_gpu_layout_offset = uint64_t(e) * H * I * sizeof(float);
            gate_v[e].expert_gpu_layout_size = uint64_t(I) * H * sizeof(float);
            up_v[e].expert_gpu_layout_size = uint64_t(I) * H * sizeof(float);
            down_v[e].expert_gpu_layout_size = uint64_t(H) * I * sizeof(float);
            gate_v[e].expert_gpu_layout_row_bytes = uint64_t(H) * sizeof(float);
            up_v[e].expert_gpu_layout_row_bytes = uint64_t(H) * sizeof(float);
            down_v[e].expert_gpu_layout_row_bytes = uint64_t(I) * sizeof(float);
            tasks[e].layer = 0;
            tasks[e].expert = static_cast<int32_t>(e);
            tasks[e].gate_weight = &gate_v[e];
            tasks[e].up_weight = &up_v[e];
            tasks[e].down_weight = &down_v[e];
            tasks[e].d_input = reinterpret_cast<const void*>(d_x);
            tasks[e].input_stride = H;
            tasks[e].d_token_indices = reinterpret_cast<const uint32_t*>(d_idx);
            tasks[e].d_token_weights = reinterpret_cast<const float*>(d_weights);
            tasks[e].assignment_offset = e * A;
            tasks[e].assignment_count = A;
            tasks[e].d_accum = reinterpret_cast<void*>(d_accum);
            tasks[e].accum_stride = H;
            tasks[e].hidden_size = H;
            tasks[e].intermediate_size = I;
            tasks[e].activation_mode = 0;
        }
        const int backend_opencl = 6;
        {
            const float zero = 0.0f;
            if (clEnqueueFillBuffer(q, d_accum, &zero, sizeof(zero), 0, accum.size() * sizeof(float), 0, nullptr, nullptr) != CL_SUCCESS ||
                !storagellm_clblast_grouped_moe_indexed_device_f32(backend_opencl, tasks, E, q) || clFinish(q) != CL_SUCCESS) {
                c.reason = "OpenCL/CLBlast adapter returned failure during warmup/sync";
                c.validation = "warmup-failed";
                goto cleanup;
            }
        }
        if (clEnqueueReadBuffer(q, d_accum, CL_TRUE, 0, accum.size() * sizeof(float), accum.data(), 0, nullptr, nullptr) != CL_SUCCESS) {
            c.reason = "OpenCL/CLBlast adapter failed to copy correctness output to host";
            c.validation = "correctness-copy-failed";
            goto cleanup;
        }
        {
            std::vector<float> ref(accum.size(), 0.0f);
            for (uint32_t e = 0; e < E; ++e) {
                reference_grouped_moe_task(ref, x, gate, up, down, weights, idx, e * A, A, H, I, 0u,
                    static_cast<size_t>(e) * I * H,
                    static_cast<size_t>(e) * I * H,
                    static_cast<size_t>(e) * H * I);
            }
            if (!validate_accum_against_reference(c, accum, ref, "OpenCL/CLBlast adapter", 8.0e-3, 8.0e-3)) {
                goto cleanup;
            }
        }
        double latency = 0.0;
        const int measured_ok = measure_adaptive_ms([&]() {
            const float zero = 0.0f;
            if (clEnqueueFillBuffer(q, d_accum, &zero, sizeof(zero), 0, accum.size() * sizeof(float), 0, nullptr, nullptr) != CL_SUCCESS) return false;
            if (!storagellm_clblast_grouped_moe_indexed_device_f32(backend_opencl, tasks, E, q)) return false;
            return clFinish(q) == CL_SUCCESS;
        }, latency);
        if (!measured_ok) {
            c.reason = "OpenCL/CLBlast adapter returned failure during adaptive measurement";
            c.validation = "measurement-failed";
            goto cleanup;
        }
        c.compiled = true;
        c.loadable = true;
        c.measured = true;
        c.true_kernel = true;
        c.fused_moe = true;
        c.latency_ms = latency;
        c.validation = "linked+runtime-device+warmup+correctness+adaptive-measurement";
        c.reason = "measured and correctness-verified native OpenCL/CLBlast fused grouped-MoE adapter with real OpenCL buffers/queue";
    }

cleanup:
    if (d_accum) clReleaseMemObject(d_accum);
    if (d_idx) clReleaseMemObject(d_idx);
    if (d_weights) clReleaseMemObject(d_weights);
    if (d_down) clReleaseMemObject(d_down);
    if (d_up) clReleaseMemObject(d_up);
    if (d_gate) clReleaseMemObject(d_gate);
    if (d_x) clReleaseMemObject(d_x);
    if (q) clReleaseCommandQueue(q);
    if (ctx) clReleaseContext(ctx);
#else
    c.reason = "native OpenCL/CLBlast adapter was not linked into storagellm_host_autotune";
    c.validation = "not-linked";
#endif
}
#else
static void benchmark_opencl_clblast_native(Candidate& c) {
    c.reason = "native OpenCL/CLBlast adapter was not linked with a usable OpenCL runtime into storagellm_host_autotune";
    c.validation = "not-linked-or-runtime-header-missing";
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
    c.compiled = true;

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

    std::fill(accum.begin(), accum.end(), 0.0f);
    if (!fn(1, &task, 1, nullptr)) {
        c.reason = "entry returned failure during warmup";
        close_handle(handle);
        return;
    }
    std::vector<float> ref(accum.size(), 0.0f);
    reference_grouped_moe_task(ref, x, gate, up, down, weights, idx, 0u, task.assignment_count,
        task.hidden_size, task.intermediate_size, task.activation_mode, 0u, 0u, 0u);
    if (!validate_accum_against_reference(c, accum, ref, "TVM CPU adapter")) {
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
    c.true_kernel = true;
    c.fused_moe = true;
    c.latency_ms = latency;
    c.validation = "library-load+symbol+warmup+correctness+adaptive-measurement";
    c.reason = "measured and correctness-verified by C++ host_autotune with warmup/adaptive iterations on this machine";
    close_handle(handle);
}

static bool candidate_needs_runtime_device(const Candidate& c) {
    return c.tvm_target != "cpu" && c.tvm_target != "llvm";
}

void benchmark_candidates(std::vector<Candidate>& candidates) {
    for (auto& c : candidates) {
        if (candidate_needs_runtime_device(c) && !c.runtime_device) {
            c.validation = "runtime-device-missing";
            c.reason = "not measured/selected: SDK or headers may be present, but no matching runtime device was detected";
            continue;
        }
        if (c.name == "cpu_native_f32") {
            benchmark_cpu_native(c);
        } else if (c.name == "cuda_cublaslt") {
            benchmark_cuda_cublaslt_native(c);
        } else if (c.name == "cuda_cutlass") {
            benchmark_cuda_cutlass_native(c);
        } else if (c.kind == "tvm" && c.name == "tvm_cpu") {
            benchmark_tvm_cpu(c);
        } else if (c.name == "rocm_hipblaslt") {
            benchmark_rocm_hipblaslt_native(c);
        } else if (c.name == "rocm_ck") {
            benchmark_rocm_ck_native(c);
        } else if (c.name == "opencl_clblast") {
            benchmark_opencl_clblast_native(c);
        } else if (c.kind == "tvm") {
            c.validation = "not-measured";
            c.reason = "only TVM CPU is emitted as an automatic candidate; device TVM libraries require a real device runtime fixture and are not advertised as fast candidates";
        } else if (c.compiled) {
            c.reason = c.reason.empty() ? "compiled but not benchmarked by generic C++ probe" : c.reason;
        }
    }
}

} // namespace storagellm::autotune
