// Storage runtime path policy.
//
// The storage engine has two valid high-speed shapes:
//   1) CPU RAM + storage: CPU kernels consume mmap/page-cache hot views.
//   2) GPU VRAM/RAM + storage: disk/page-cache/staging feeds device kernels.
// Keep this decision centralized and cached.  In particular,
// vram_budget_bytes == 0 is the GPU auto-budget sentinel before backend caps are
// probed; it must not be treated as CPU-only.

enum moe_storage_runtime_path_t : uint32_t {
    moe_STORAGE_RUNTIME_PATH_UNKNOWN = 0,
    moe_STORAGE_RUNTIME_PATH_CPU_RAM = 1,
    moe_STORAGE_RUNTIME_PATH_GPU_RAM = 2,
    moe_STORAGE_RUNTIME_PATH_GENERIC = 3
};

static int moe_storage_env_truthy_local(const char* name) {
    return moe_perf_env_truthy(name);
}

static int moe_storage_env_value_is(const char* name, const char* expected) {
    const char* v = name ? std::getenv(name) : nullptr;
    return v && expected && std::strcmp(v, expected) == 0;
}

static int moe_storage_env_backend_is_cpu() {
    return moe_storage_env_value_is("STORAGELLM_BACKEND", "cpu") ||
           moe_storage_env_value_is("STORAGELLM_BACKEND", "CPU");
}

static int moe_storage_env_platform_is_cpu() {
    return moe_storage_env_value_is("STORAGELLM_PLATFORM", "cpu") ||
           moe_storage_env_value_is("STORAGELLM_PLATFORM", "CPU") ||
           moe_storage_env_value_is("STORAGELLM_PLATFORM", "cpu_only") ||
           moe_storage_env_value_is("STORAGELLM_PLATFORM", "CPU_ONLY");
}

static int moe_storage_backend_is_device(moe_backend_t backend) {
    switch (backend) {
        case moe_BACKEND_CUDA:
        case moe_BACKEND_HIP:
        case moe_BACKEND_METAL:
        case moe_BACKEND_LEVEL_ZERO:
        case moe_BACKEND_SYCL:
        case moe_BACKEND_DIRECTML:
        case moe_BACKEND_VULKAN:
        case moe_BACKEND_OPENCL:
        case moe_BACKEND_WEBGPU:
            return 1;
        default:
            return 0;
    }
}

static int moe_storage_cpu_path_forced_by_env() {
    return moe_storage_env_truthy_local("STORAGELLM_CPU_ONLY") ||
           moe_storage_env_truthy_local("STORAGELLM_CPU_STORAGE_ENGINE") ||
           moe_storage_env_truthy_local("STORAGELLM_CPU_STORAGE_MAX") ||
           moe_storage_env_truthy_local("STORAGELLM_CPU_PIPELINE") ||
           moe_storage_env_backend_is_cpu() ||
           moe_storage_env_platform_is_cpu();
}

static uint32_t moe_storage_runtime_path_uncached(const moe_pc_engine* engine) {
    if (moe_storage_cpu_path_forced_by_env()) {
        return moe_STORAGE_RUNTIME_PATH_CPU_RAM;
    }
    if (!engine) {
        return moe_STORAGE_RUNTIME_PATH_UNKNOWN;
    }
    if (engine->config.platform == moe_PLATFORM_CPU_ONLY ||
        engine->config.preferred_backend == moe_BACKEND_CPU) {
        return moe_STORAGE_RUNTIME_PATH_CPU_RAM;
    }
    if (!engine->config.prefer_gpu &&
        !moe_storage_backend_is_device(engine->backend_caps.backend) &&
        engine->config.vram_budget_bytes == 0 &&
        engine->backend_caps.device_memory_bytes == 0) {
        return moe_STORAGE_RUNTIME_PATH_CPU_RAM;
    }
    if (moe_storage_backend_is_device(engine->backend_caps.backend)) {
        return moe_STORAGE_RUNTIME_PATH_GPU_RAM;
    }
    if (engine->config.prefer_gpu &&
        engine->config.platform != moe_PLATFORM_CPU_ONLY &&
        engine->config.preferred_backend != moe_BACKEND_CPU) {
        return moe_STORAGE_RUNTIME_PATH_GPU_RAM;
    }
    if (engine->config.vram_budget_bytes > 0 || engine->backend_caps.device_memory_bytes > 0) {
        return moe_STORAGE_RUNTIME_PATH_GPU_RAM;
    }
    return moe_STORAGE_RUNTIME_PATH_GENERIC;
}

static uint32_t moe_storage_runtime_path_refresh(moe_pc_engine* engine) {
    if (!engine) {
        return moe_STORAGE_RUNTIME_PATH_UNKNOWN;
    }
    const uint32_t path = moe_storage_runtime_path_uncached(engine);
    engine->storage_runtime_path.store(path, std::memory_order_relaxed);
    engine->cpu_only_runtime.store(path == moe_STORAGE_RUNTIME_PATH_CPU_RAM ? 1 : 0, std::memory_order_relaxed);
    engine->gpu_path_disabled_for_cpu_only.store(path == moe_STORAGE_RUNTIME_PATH_CPU_RAM ? 1 : 0, std::memory_order_relaxed);
    return path;
}

static uint32_t moe_storage_runtime_path_cached(const moe_pc_engine* engine) {
    if (!engine) {
        return moe_STORAGE_RUNTIME_PATH_UNKNOWN;
    }
    uint32_t path = engine->storage_runtime_path.load(std::memory_order_relaxed);
    if (path != moe_STORAGE_RUNTIME_PATH_UNKNOWN) {
        return path;
    }
    path = moe_storage_runtime_path_uncached(engine);
    const_cast<moe_pc_engine*>(engine)->storage_runtime_path.store(path, std::memory_order_relaxed);
    return path;
}

static int moe_storage_runtime_cpu_ram_path(const moe_pc_engine* engine) {
    return moe_storage_runtime_path_cached(engine) == moe_STORAGE_RUNTIME_PATH_CPU_RAM;
}

static int moe_storage_runtime_gpu_ram_path_requested(const moe_pc_engine* engine) {
    return moe_storage_runtime_path_cached(engine) == moe_STORAGE_RUNTIME_PATH_GPU_RAM;
}

static int moe_storage_runtime_gpu_ram_path_has_device_memory(const moe_pc_engine* engine) {
    if (moe_storage_runtime_path_cached(engine) != moe_STORAGE_RUNTIME_PATH_GPU_RAM) {
        return 0;
    }
    if (engine->config.vram_budget_bytes > 0 ||
        engine->backend_caps.device_memory_bytes > 0 ||
        engine->device_total_bytes.load(std::memory_order_relaxed) > 0) {
        return 1;
    }
    // A zero explicit budget in a GPU profile is the auto-budget sentinel.  Do not
    // collapse the path to CPU before backend caps/device memory probing runs.
    return engine->config.vram_budget_bytes == 0 ? 1 : 0;
}
