typedef enum {
    // Default/original runtime path.
    GLM5_KV_MODE_PLAIN = 0,
    // Feature flag only. Uses engine_core/kv/kv_qkv.*.
    GLM5_KV_MODE_QKV = 1
} glm5_kv_mode_t;

typedef enum {
    GLM5_TIER_VRAM = 0,
    GLM5_TIER_RAM = 1,
    GLM5_TIER_DB = 2,
    GLM5_TIER_STREAM = 3
} glm5_tier_t;

typedef enum {
    GLM5_PROJ_GATE = 0,
    GLM5_PROJ_UP = 1,
    GLM5_PROJ_DOWN = 2
} glm5_projection_t;

typedef enum {
    GLM5_IO_PATH_MMAP_PAGECACHE = 0,
    GLM5_IO_PATH_PINNED_HOST = 1,
    GLM5_IO_PATH_DIRECTSTORAGE = 2,
    GLM5_IO_PATH_GPUDIRECT_STORAGE = 3,
    GLM5_IO_PATH_BAM_NVME = 4
} glm5_io_path_t;

typedef enum {
    GLM5_BACKEND_AUTO = 0,
    GLM5_BACKEND_CPU = 1,
    GLM5_BACKEND_CUDA = 2,
    GLM5_BACKEND_METAL = 3,
    GLM5_BACKEND_VULKAN = 4,
    GLM5_BACKEND_DIRECTML = 5,
    GLM5_BACKEND_OPENCL = 6,
    GLM5_BACKEND_HIP = 7,
    GLM5_BACKEND_LEVEL_ZERO = 8,
    GLM5_BACKEND_SYCL = 9,
    GLM5_BACKEND_WEBGPU = 10
} glm5_backend_t;

typedef enum {
    GLM5_PLATFORM_AUTO = 0,
    GLM5_PLATFORM_WINDOWS_PC = 1,
    GLM5_PLATFORM_LINUX_PC = 2,
    GLM5_PLATFORM_MACOS_APPLE = 3,
    GLM5_PLATFORM_CPU_ONLY = 4
} glm5_platform_t;

typedef enum {
    GLM5_TRANSFER_STAGE_DISK_TO_RAM = 0,
    GLM5_TRANSFER_STAGE_RAM_TO_PINNED = 1,
    GLM5_TRANSFER_STAGE_PINNED_TO_GPU = 2,
    GLM5_TRANSFER_STAGE_DIRECT_TO_GPU = 3
} glm5_transfer_stage_t;

typedef enum {
    GLM5_PREFETCH_REASON_ROUTER_SELECTED = 0,
    GLM5_PREFETCH_REASON_LOOKAHEAD = 1,
    GLM5_PREFETCH_REASON_TOPOLOGY = 2
} glm5_prefetch_reason_t;

typedef enum {
    GLM5_PHASE_PREFILL = 0,
    GLM5_PHASE_DECODE = 1,
    GLM5_PHASE_BOTH = 2
} glm5_inference_phase_t;

typedef enum {
    GLM5_EXECUTION_PERFORMANCE = 0,
    GLM5_EXECUTION_BALANCED = 1,
    GLM5_EXECUTION_LOW_IMPACT = 2
} glm5_execution_policy_t;
