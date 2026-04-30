const char* glm5_kv_mode_name(glm5_kv_mode_t mode) {
    switch (mode) {
        case GLM5_KV_MODE_PLAIN:
            return "plain";
        case GLM5_KV_MODE_QKV:
            return "qkv";
        default:
            return "unknown";
    }
}

const char* glm5_tier_name(glm5_tier_t tier) {
    switch (tier) {
        case GLM5_TIER_VRAM:
            return "vram";
        case GLM5_TIER_RAM:
            return "ram";
        case GLM5_TIER_DB:
            return "db";
        case GLM5_TIER_STREAM:
            return "stream";
        default:
            return "unknown";
    }
}

const char* glm5_projection_name(glm5_projection_t proj) {
    switch (proj) {
        case GLM5_PROJ_GATE:
            return "gate_proj";
        case GLM5_PROJ_UP:
            return "up_proj";
        case GLM5_PROJ_DOWN:
            return "down_proj";
        default:
            return "unknown";
    }
}

const char* glm5_io_path_name(glm5_io_path_t path) {
    switch (path) {
        case GLM5_IO_PATH_MMAP_PAGECACHE:
            return "mmap_pagecache";
        case GLM5_IO_PATH_PINNED_HOST:
            return "pinned_host";
        case GLM5_IO_PATH_DIRECTSTORAGE:
            return "directstorage";
        case GLM5_IO_PATH_GPUDIRECT_STORAGE:
            return "gpudirect_storage";
        case GLM5_IO_PATH_BAM_NVME:
            return "bam_nvme";
        case GLM5_IO_PATH_ZERO_COPY_HOST:
            return "zero_copy_host";
        default:
            return "unknown";
    }
}
