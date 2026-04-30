static uint64_t juju_clamp_u64(uint64_t value, uint64_t lo, uint64_t hi) {
    return std::max(lo, std::min(value, hi));
}

static double juju_positive_or(double value, double fallback) {
    return value > 0.0 ? value : fallback;
}

uint64_t moe_juju_align_up(uint64_t value, uint64_t alignment) {
    if (alignment == 0) {
        return value;
    }
    const uint64_t rem = value % alignment;
    return rem == 0 ? value : value + (alignment - rem);
}

int moe_juju_header_valid(const moe_juju_header_t* header) {
    if (!header) {
        return 0;
    }
    if (header->magic[0] != 'J' || header->magic[1] != 'U' ||
        header->magic[2] != 'J' || header->magic[3] != 'U') {
        return 0;
    }
    if (header->header_size != 0 && header->header_size != MOE_JUJU_HEADER_BYTES) {
        return 0;
    }
    if (header->version_major == 0) {
        return 0;
    }
    if (header->section_count > 65536u) {
        return 0;
    }
    if (header->file_size && header->section_table_offset >= header->file_size) {
        return 0;
    }
    if (header->section_table_size &&
        header->section_table_size < (uint64_t)header->section_count * sizeof(moe_juju_section_entry_t)) {
        return 0;
    }
    return 1;
}

int moe_juju_read_header(const char* path, moe_juju_header_t* out_header) {
    if (!path || !out_header) {
        return 0;
    }
    std::memset(out_header, 0, sizeof(*out_header));
    std::ifstream in;
    open_ifstream_path(in, path, std::ios::in | std::ios::binary);
    if (!in) {
        return 0;
    }
    in.read(reinterpret_cast<char*>(out_header), sizeof(*out_header));
    if ((uint64_t)in.gcount() != sizeof(*out_header)) {
        std::memset(out_header, 0, sizeof(*out_header));
        return 0;
    }
    if (!moe_juju_header_valid(out_header)) {
        std::memset(out_header, 0, sizeof(*out_header));
        return 0;
    }
    return 1;
}

int moe_juju_read_section_table(
    const char* path,
    const moe_juju_header_t* header,
    moe_juju_section_entry_t* out_entries,
    uint32_t max_entries,
    uint32_t* out_count
) {
    if (out_count) {
        *out_count = 0;
    }
    if (!path || !header || !moe_juju_header_valid(header)) {
        return 0;
    }
    if (out_count) {
        *out_count = header->section_count;
    }
    if (!out_entries || max_entries == 0) {
        return 1;
    }
    if (max_entries < header->section_count) {
        return 0;
    }
    const uint64_t bytes = (uint64_t)header->section_count * sizeof(moe_juju_section_entry_t);
    if (header->section_table_size && bytes > header->section_table_size) {
        return 0;
    }
    std::ifstream in;
    open_ifstream_path(in, path, std::ios::in | std::ios::binary);
    if (!in) {
        return 0;
    }
    in.seekg((std::streamoff)header->section_table_offset, std::ios::beg);
    if (!in) {
        return 0;
    }
    in.read(reinterpret_cast<char*>(out_entries), (std::streamsize)bytes);
    return (uint64_t)in.gcount() == bytes ? 1 : 0;
}

static double juju_compute_ms(
    const moe_juju_tier_capability_t* tier0,
    uint64_t expert_bytes,
    double expert_flops
) {
    if (tier0 && tier0->compute_tflops > 0.0 && expert_flops > 0.0) {
        return std::max(0.01, expert_flops / (tier0->compute_tflops * 1.0e12) * 1000.0);
    }
    const double bw = tier0 ? juju_positive_or(tier0->bandwidth_gbps, 1.0) : 1.0;
    return std::max(0.25, ((double)expert_bytes / 1.0e9) / bw * 1000.0);
}

static double juju_transfer_ms(const moe_juju_tier_capability_t* tier, uint64_t bytes) {
    if (!tier || tier->bandwidth_gbps <= 0.0 || bytes == 0) {
        return 0.0;
    }
    return ((double)bytes / 1.0e9) / tier->bandwidth_gbps * 1000.0;
}

static uint64_t juju_choose_block_size(uint64_t expert_bytes, const moe_juju_tier_capability_t* slow_tier) {
    const uint64_t kb = 1024ull;
    const uint64_t mb = 1024ull * kb;
    if (!slow_tier || !slow_tier->supports_async) {
        return 4ull * kb;
    }
    if (expert_bytes >= 128ull * mb) {
        return 2ull * mb;
    }
    if (expert_bytes >= 32ull * mb) {
        return 1ull * mb;
    }
    return 512ull * kb;
}

static uint64_t juju_choose_segment_bytes(
    uint64_t expert_bytes,
    uint64_t block_size,
    double compute_ms,
    double slow_transfer_ms
) {
    if (expert_bytes == 0) {
        return 0;
    }
    const uint64_t mb = 1024ull * 1024ull;
    if (expert_bytes <= 20ull * mb || slow_transfer_ms <= compute_ms) {
        return moe_juju_align_up(expert_bytes, 4096ull);
    }
    uint64_t segment = expert_bytes >= 128ull * mb ? 2ull * mb : 1ull * mb;
    segment = std::max(segment, block_size);
    segment = moe_juju_align_up(segment, 4096ull);
    return juju_clamp_u64(segment, block_size, expert_bytes);
}

int moe_juju_compute_pipeline_config(
    const moe_juju_tier_capability_t* tier0,
    const moe_juju_tier_capability_t* tier1,
    const moe_juju_tier_capability_t* tier2,
    uint64_t expert_bytes,
    double expert_flops,
    moe_juju_pipeline_config_t* out_config
) {
    if (!tier0 || !out_config || expert_bytes == 0) {
        return 0;
    }
    std::memset(out_config, 0, sizeof(*out_config));
    const double compute_ms = juju_compute_ms(tier0, expert_bytes, expert_flops);
    const double t1_to_t0_ms = juju_transfer_ms(tier1, expert_bytes);
    const double t2_to_t1_ms = juju_transfer_ms(tier2, expert_bytes);
    const uint64_t block_size = juju_choose_block_size(expert_bytes, tier2);
    const uint64_t segment_bytes = juju_choose_segment_bytes(expert_bytes, block_size, compute_ms, t2_to_t1_ms);
    const uint32_t prefetch_depth = (uint32_t)juju_clamp_u64(
        t2_to_t1_ms > 0.0 ? (uint64_t)std::ceil(t2_to_t1_ms / compute_ms) + 1ull : 1ull,
        1ull,
        16ull
    );
    const uint32_t io_threads = (uint32_t)juju_clamp_u64((uint64_t)prefetch_depth * 2ull, 1ull, 16ull);
    out_config->prefetch_depth = prefetch_depth;
    out_config->tier0_expert_count = tier0->capacity_bytes / expert_bytes;
    out_config->tier1_expert_count = tier1 ? tier1->capacity_bytes / expert_bytes : 0;
    out_config->io_threads = io_threads;
    out_config->block_size = block_size;
    out_config->segment_bytes = segment_bytes;
    out_config->segment_count = segment_bytes ? (uint32_t)((expert_bytes + segment_bytes - 1ull) / segment_bytes) : 0;
    out_config->compute_ms = compute_ms;
    out_config->tier1_to_tier0_ms = t1_to_t0_ms;
    out_config->tier2_to_tier1_ms = t2_to_t1_ms;
    return 1;
}

moe_juju_execution_path_t moe_juju_select_execution_path(
    int has_accelerator,
    uint64_t model_total_bytes,
    uint64_t expert_bytes,
    const moe_juju_tier_capability_t* tier0,
    const moe_juju_tier_capability_t* tier1,
    const moe_juju_tier_capability_t* tier2
) {
    const uint64_t t0 = tier0 ? tier0->capacity_bytes : 0;
    const uint64_t t1 = tier1 ? tier1->capacity_bytes : 0;
    if (model_total_bytes && t0 >= model_total_bytes) {
        return moe_JUJU_EXEC_FULL_COMPUTE_MEM;
    }
    if (has_accelerator && tier2 && tier2->supports_dma) {
        return moe_JUJU_EXEC_DIRECT_STORAGE_TO_COMPUTE;
    }
    if (has_accelerator && expert_bytes && t1 >= expert_bytes * 10ull) {
        return moe_JUJU_EXEC_THREE_TIER_PIPELINE;
    }
    if (has_accelerator && expert_bytes && t1 >= expert_bytes * 2ull) {
        return moe_JUJU_EXEC_ROTATING_BUFFER;
    }
    if (!has_accelerator && model_total_bytes && t0 >= model_total_bytes / 3ull) {
        return moe_JUJU_EXEC_CPU_WITH_CACHE;
    }
    return moe_JUJU_EXEC_STORAGE_STREAMING;
}

const char* moe_juju_execution_path_name(moe_juju_execution_path_t path) {
    switch (path) {
    case moe_JUJU_EXEC_FULL_COMPUTE_MEM:
        return "FULL_COMPUTE_MEM";
    case moe_JUJU_EXEC_DIRECT_STORAGE_TO_COMPUTE:
        return "DIRECT_STORAGE_TO_COMPUTE";
    case moe_JUJU_EXEC_THREE_TIER_PIPELINE:
        return "THREE_TIER_PIPELINE";
    case moe_JUJU_EXEC_ROTATING_BUFFER:
        return "ROTATING_BUFFER";
    case moe_JUJU_EXEC_CPU_WITH_CACHE:
        return "CPU_WITH_CACHE";
    case moe_JUJU_EXEC_STORAGE_STREAMING:
        return "STORAGE_STREAMING";
    default:
        return "UNKNOWN";
    }
}

const char* moe_juju_section_type_name(uint32_t type) {
    switch (type) {
    case moe_JUJU_SECTION_MODEL_META:
        return "MODEL_META";
    case moe_JUJU_SECTION_PREDICTOR:
        return "PREDICTOR";
    case moe_JUJU_SECTION_BUDDY_MAP:
        return "BUDDY_MAP";
    case moe_JUJU_SECTION_TIER_HINT:
        return "TIER_HINT";
    case moe_JUJU_SECTION_SHARED_WEIGHTS:
        return "SHARED_WEIGHTS";
    case moe_JUJU_SECTION_HOT_EXPERTS:
        return "HOT_EXPERTS";
    case moe_JUJU_SECTION_WARM_EXPERTS:
        return "WARM_EXPERTS";
    case moe_JUJU_SECTION_COLD_EXPERTS:
        return "COLD_EXPERTS";
    case moe_JUJU_SECTION_LAYER_ORDER_INDEX:
        return "LAYER_ORDER_INDEX";
    case moe_JUJU_SECTION_QKV_POLICY:
        return "QKV_POLICY";
    case moe_JUJU_SECTION_CUSTOM:
        return "CUSTOM";
    default:
        return "UNKNOWN";
    }
}
