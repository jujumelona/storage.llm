const char* moe_kv_mode_name(moe_kv_mode_t mode) {
    switch (mode) {
        case moe_KV_MODE_PLAIN:
            return "plain";
        case moe_KV_MODE_QKV:
            return "qkv";
        default:
            return "unknown";
    }
}

const char* moe_tier_name(moe_tier_t tier) {
    switch (tier) {
        case moe_TIER_VRAM:
            return "vram";
        case moe_TIER_RAM:
            return "ram";
        case moe_TIER_DB:
            return "db";
        case moe_TIER_STREAM:
            return "stream";
        default:
            return "unknown";
    }
}

const char* moe_projection_name(moe_projection_t proj) {
    switch (proj) {
        case moe_PROJ_GATE:
            return "gate_proj";
        case moe_PROJ_UP:
            return "up_proj";
        case moe_PROJ_DOWN:
            return "down_proj";
        default:
            return "unknown";
    }
}

const char* moe_io_path_name(moe_io_path_t path) {
    switch (path) {
        case moe_IO_PATH_MMAP_PAGECACHE:
            return "mmap_pagecache";
        case moe_IO_PATH_PINNED_HOST:
            return "pinned_host";
        case moe_IO_PATH_DIRECTSTORAGE:
            return "directstorage";
        case moe_IO_PATH_GPUDIRECT_STORAGE:
            return "gpudirect_storage";
        case moe_IO_PATH_BAM_NVME:
            return "bam_nvme";
        case moe_IO_PATH_ZERO_COPY_HOST:
            return "zero_copy_host";
        default:
            return "unknown";
    }
}
