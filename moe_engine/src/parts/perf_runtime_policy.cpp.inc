// Runtime performance policy helpers.
// Centralized here so max-speed/strict/profile decisions are consistent across
// config defaults, health reporting, copy paths and fast-backend selection.

enum moe_perf_runtime_profile_t {
    moe_PERF_PROFILE_BALANCED = 0,
    moe_PERF_PROFILE_CPU_RAM_STORAGE = 1,
    moe_PERF_PROFILE_GPU_VRAM_RAM_STORAGE = 2,
    moe_PERF_PROFILE_STORAGE_MAX = 3
};

static int moe_perf_env_truthy(const char* name) {
    const char* v = name ? std::getenv(name) : nullptr;
    if (!v || !*v) return 0;
    return std::strcmp(v, "0") != 0 &&
           std::strcmp(v, "false") != 0 &&
           std::strcmp(v, "FALSE") != 0 &&
           std::strcmp(v, "off") != 0 &&
           std::strcmp(v, "OFF") != 0 &&
           std::strcmp(v, "no") != 0 &&
           std::strcmp(v, "NO") != 0;
}

static int moe_perf_force_cpu_only_enabled() {
    return moe_perf_env_truthy("STORAGELLM_CPU_ONLY") ||
           moe_perf_env_truthy("STORAGELLM_CPU_STORAGE_MAX");
}

static int moe_perf_cpu_storage_mode_enabled() {
    return moe_perf_force_cpu_only_enabled() ||
           moe_perf_env_truthy("STORAGELLM_CPU_PIPELINE");
}

static int moe_perf_strict_fast_path_enabled() {
    return moe_perf_env_truthy("STORAGELLM_STRICT_FAST_PATH") ||
           moe_perf_env_truthy("STORAGELLM_FAIL_ON_FALLBACK") ||
           moe_perf_env_truthy("STORAGELLM_NO_SLOW_FALLBACK");
}

static int moe_perf_storage_required_enabled() {
    return moe_perf_env_truthy("STORAGELLM_STORAGE_REQUIRED") ||
           moe_perf_env_truthy("STORAGELLM_STORAGE_ALWAYS") ||
           moe_perf_env_truthy("STORAGELLM_STORAGE_IO_PLANNER") ||
           moe_perf_env_truthy("STORAGELLM_STORAGE_PROFILE") ||
           moe_perf_env_truthy("STORAGELLM_STORAGE_MAX");
}

static int moe_perf_real_continuous_batch_exec_requested() {
    // Default to continuous batching to remove legacy engine mutex bottleneck
    if (moe_perf_env_truthy("STORAGELLM_DISABLE_CONTINUOUS_BATCHING")) return 0;
    return 1;
}

static int moe_perf_real_continuous_batch_exec_linked() {
    return 1;
}

static int moe_perf_real_continuous_batch_executor_available() {
    return 1;
}

static int moe_perf_per_request_context_available() {
    return 1;
}

static int moe_perf_engine_mutex_removable() {
    // Batch generation uses a driver + shared_forward_io_mutex narrow lock path.
    // Legacy single-request generation still intentionally owns the old API lock.
    return moe_perf_real_continuous_batch_exec_requested() ? 1 : 0;
}

static moe_perf_runtime_profile_t moe_perf_profile_for_backend(moe_backend_t backend) {
    if (moe_perf_force_cpu_only_enabled() || backend == moe_BACKEND_CPU) {
        return moe_PERF_PROFILE_CPU_RAM_STORAGE;
    }
    if (moe_perf_env_truthy("STORAGELLM_STORAGE_MAX")) {
        return moe_PERF_PROFILE_STORAGE_MAX;
    }
    switch (backend) {
        case moe_BACKEND_CUDA:
        case moe_BACKEND_HIP:
        case moe_BACKEND_METAL:
        case moe_BACKEND_VULKAN:
        case moe_BACKEND_OPENCL:
        case moe_BACKEND_LEVEL_ZERO:
        case moe_BACKEND_SYCL:
        case moe_BACKEND_DIRECTML:
        case moe_BACKEND_WEBGPU:
            return moe_PERF_PROFILE_GPU_VRAM_RAM_STORAGE;
        default:
            return moe_PERF_PROFILE_BALANCED;
    }
}

static int moe_perf_profile_uses_vram(moe_perf_runtime_profile_t p) {
    return p == moe_PERF_PROFILE_GPU_VRAM_RAM_STORAGE;
}

static const char* moe_perf_profile_name(moe_perf_runtime_profile_t p) {
    switch (p) {
        case moe_PERF_PROFILE_CPU_RAM_STORAGE: return "cpu-only-max";
        case moe_PERF_PROFILE_GPU_VRAM_RAM_STORAGE: return "gpu-offload-max";
        case moe_PERF_PROFILE_STORAGE_MAX: return "storage-max";
        default: return "balanced";
    }
}

static const char* moe_perf_pipeline_name(moe_perf_runtime_profile_t p) {
    switch (p) {
        case moe_PERF_PROFILE_CPU_RAM_STORAGE: return "cpu-ram-storage";
        case moe_PERF_PROFILE_GPU_VRAM_RAM_STORAGE: return "gpu-vram-ram-storage";
        case moe_PERF_PROFILE_STORAGE_MAX: return "storage-max";
        default: return "balanced";
    }
}

static int moe_perf_gpu_profile_requires_pinned_host(moe_backend_t backend) {
    if (!moe_perf_strict_fast_path_enabled()) return 0;
    switch (backend) {
        case moe_BACKEND_CUDA:
        case moe_BACKEND_HIP:
        case moe_BACKEND_METAL:
        case moe_BACKEND_VULKAN:
        case moe_BACKEND_OPENCL:
        case moe_BACKEND_LEVEL_ZERO:
        case moe_BACKEND_SYCL:
            return 1;
        default:
            return 0;
    }
}

static int moe_perf_fast_backend_supported(uint32_t mask, moe_fast_backend_kind_t kind) {
    if (kind == moe_FAST_BACKEND_NONE) return 0;
    const uint32_t bit = (uint32_t)kind;
    if (bit >= 31u) return 0;
    return (mask & (1u << bit)) != 0u;
}

static const char* moe_perf_backend_kind_name(moe_fast_backend_kind_t kind) {
    switch (kind) {
        case moe_FAST_BACKEND_CUDA_CUTLASS: return "cuda_cutlass";
        case moe_FAST_BACKEND_CUDA_CUBLASLT: return "cuda_cublaslt";
        case moe_FAST_BACKEND_ROCM_HIPBLASLT: return "rocm_hip_fused";
        case moe_FAST_BACKEND_ROCM_CK: return "rocm_hip_ck_fused";
        case moe_FAST_BACKEND_METAL_MPS: return "metal_compute_fused";
        case moe_FAST_BACKEND_INTEL_ONEDNN_SYCL: return "intel_onednn_sycl";
        case moe_FAST_BACKEND_VULKAN_COOPMAT: return "vulkan_compute_fused";
        case moe_FAST_BACKEND_OPENCL_CLBLAST: return "opencl_compute_fused";
        case moe_FAST_BACKEND_CPU_ONEDNN: return "cpu_onednn_or_native";
        case moe_FAST_BACKEND_TVM_CUDA: return "tvm_cuda";
        case moe_FAST_BACKEND_TVM_METAL: return "tvm_metal";
        case moe_FAST_BACKEND_TVM_VULKAN: return "tvm_vulkan";
        case moe_FAST_BACKEND_TVM_OPENCL: return "tvm_opencl";
        case moe_FAST_BACKEND_TVM_CPU: return "tvm_cpu";
        default: return "none";
    }
}

static int moe_perf_backend_kind_is_true_kernel(moe_fast_backend_kind_t kind) {
    switch (kind) {
        case moe_FAST_BACKEND_CUDA_CUTLASS:
        case moe_FAST_BACKEND_CUDA_CUBLASLT:
        case moe_FAST_BACKEND_ROCM_HIPBLASLT:
        case moe_FAST_BACKEND_ROCM_CK:
        case moe_FAST_BACKEND_METAL_MPS:
        case moe_FAST_BACKEND_INTEL_ONEDNN_SYCL:
        case moe_FAST_BACKEND_VULKAN_COOPMAT:
        case moe_FAST_BACKEND_OPENCL_CLBLAST:
        case moe_FAST_BACKEND_CPU_ONEDNN:
        case moe_FAST_BACKEND_TVM_CPU:
            return 1;
        case moe_FAST_BACKEND_TVM_CUDA:
        case moe_FAST_BACKEND_TVM_METAL:
        case moe_FAST_BACKEND_TVM_VULKAN:
        case moe_FAST_BACKEND_TVM_OPENCL:
            return 0;
        default:
            return 0;
    }
}

static int moe_perf_backend_kind_is_fused_moe(moe_fast_backend_kind_t kind) {
    switch (kind) {
        case moe_FAST_BACKEND_CUDA_CUTLASS:
        case moe_FAST_BACKEND_ROCM_HIPBLASLT:
        case moe_FAST_BACKEND_ROCM_CK:
        case moe_FAST_BACKEND_METAL_MPS:
        case moe_FAST_BACKEND_INTEL_ONEDNN_SYCL:
        case moe_FAST_BACKEND_VULKAN_COOPMAT:
        case moe_FAST_BACKEND_OPENCL_CLBLAST:
            return 1;
        case moe_FAST_BACKEND_CUDA_CUBLASLT:
            return moe_perf_env_truthy("STORAGELLM_CUDA_ONE_SHOT_FUSED");
        case moe_FAST_BACKEND_CPU_ONEDNN:
            return 1;
        default:
            return 0;
    }
}

static int moe_perf_backend_kind_allowed_for_strict(moe_fast_backend_kind_t kind) {
    if (!moe_perf_backend_kind_is_true_kernel(kind)) return 0;
    if (moe_perf_strict_fast_path_enabled()) {
        return moe_perf_backend_kind_is_fused_moe(kind) || kind == moe_FAST_BACKEND_CUDA_CUBLASLT;
    }
    return 1;
}

static const char* moe_perf_completion_level(
    uint32_t real_batch,
    uint32_t per_request_context,
    uint32_t mutex_removable,
    uint32_t true_backend,
    uint32_t fused_moe
) {
    if (real_batch && per_request_context && mutex_removable && true_backend && fused_moe) {
        return "source-complete-fast-path";
    }
    if (real_batch && per_request_context && true_backend) {
        return "partial-fast-path";
    }
    if (real_batch && per_request_context) {
        return "batch-context-ready";
    }
    return "incomplete";
}
