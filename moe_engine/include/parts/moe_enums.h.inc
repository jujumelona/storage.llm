typedef enum {
    // Legacy/debug runtime path. Offload-native models can forbid this.
    moe_KV_MODE_PLAIN = 0,
    // Default packed QKV runtime path.
    moe_KV_MODE_QKV = 1
} moe_kv_mode_t;

typedef enum {
    moe_TIER_VRAM = 0,
    moe_TIER_RAM = 1,
    moe_TIER_DB = 2,
    moe_TIER_STREAM = 3
} moe_tier_t;

typedef enum {
    moe_PROJ_GATE = 0,
    moe_PROJ_UP = 1,
    moe_PROJ_DOWN = 2
} moe_projection_t;

typedef enum {
    moe_IO_PATH_MMAP_PAGECACHE = 0,
    moe_IO_PATH_PINNED_HOST = 1,
    moe_IO_PATH_DIRECTSTORAGE = 2,
    moe_IO_PATH_GPUDIRECT_STORAGE = 3,
    moe_IO_PATH_BAM_NVME = 4,
    moe_IO_PATH_ZERO_COPY_HOST = 5
} moe_io_path_t;

typedef enum {
    moe_BACKEND_AUTO = 0,
    moe_BACKEND_CPU = 1,
    moe_BACKEND_CUDA = 2,
    moe_BACKEND_METAL = 3,
    moe_BACKEND_VULKAN = 4,
    moe_BACKEND_DIRECTML = 5,
    moe_BACKEND_OPENCL = 6,
    moe_BACKEND_HIP = 7,
    moe_BACKEND_LEVEL_ZERO = 8,
    moe_BACKEND_SYCL = 9,
    moe_BACKEND_WEBGPU = 10
} moe_backend_t;

typedef enum {
    moe_PLATFORM_AUTO = 0,
    moe_PLATFORM_WINDOWS_PC = 1,
    moe_PLATFORM_LINUX_PC = 2,
    moe_PLATFORM_MACOS_APPLE = 3,
    moe_PLATFORM_CPU_ONLY = 4
} moe_platform_t;

typedef enum {
    moe_TRANSFER_STAGE_DISK_TO_RAM = 0,
    moe_TRANSFER_STAGE_RAM_TO_PINNED = 1,
    moe_TRANSFER_STAGE_PINNED_TO_GPU = 2,
    moe_TRANSFER_STAGE_DIRECT_TO_GPU = 3
} moe_transfer_stage_t;

typedef enum {
    moe_PREFETCH_REASON_ROUTER_SELECTED = 0,
    moe_PREFETCH_REASON_LOOKAHEAD = 1,
    moe_PREFETCH_REASON_TOPOLOGY = 2
} moe_prefetch_reason_t;

typedef enum {
    moe_PHASE_PREFILL = 0,
    moe_PHASE_DECODE = 1,
    moe_PHASE_BOTH = 2
} moe_inference_phase_t;

typedef enum {
    moe_EXECUTION_PERFORMANCE = 0,
    moe_EXECUTION_BALANCED = 1,
    moe_EXECUTION_LOW_IMPACT = 2
} moe_execution_policy_t;
