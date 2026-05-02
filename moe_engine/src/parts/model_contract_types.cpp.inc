// Model contract type definitions and helper functions.
// Defines structures for GGUF/JUJU offload contracts and file metadata.

static uint32_t moe_weight_encoding_from_family(const std::string& family) {
    const std::string value = moe_lower_ascii_copy(family);
    if (value.find("iq2") != std::string::npos || value.find("ud-iq2") != std::string::npos) {
        if (value.find("iq2_xs") != std::string::npos) {
            return moe_WEIGHT_ENCODING_GGUF_IQ2_XS;
        }
        if (value.find("iq2_s") != std::string::npos) {
            return moe_WEIGHT_ENCODING_GGUF_IQ2_S;
        }
        return moe_WEIGHT_ENCODING_GGUF_IQ2;
    }
    if (value.find("iq1") != std::string::npos) {
        if (value.find("iq1_s") != std::string::npos) {
            return moe_WEIGHT_ENCODING_GGUF_IQ1_S;
        }
        if (value.find("iq1_m") != std::string::npos) {
            return moe_WEIGHT_ENCODING_GGUF_IQ1_M;
        }
        return moe_WEIGHT_ENCODING_GGUF_IQ1;
    }
    if (value.find("iq3") != std::string::npos) {
        if (value.find("iq3_s") != std::string::npos) {
            return moe_WEIGHT_ENCODING_GGUF_IQ3_S;
        }
        return moe_WEIGHT_ENCODING_GGUF_IQ3;
    }
    if (value.find("iq4") != std::string::npos) {
        if (value.find("iq4_nl") != std::string::npos) {
            return moe_WEIGHT_ENCODING_GGUF_IQ4_NL;
        }
        if (value.find("iq4_xs") != std::string::npos) {
            return moe_WEIGHT_ENCODING_GGUF_IQ4_XS;
        }
        return moe_WEIGHT_ENCODING_GGUF_IQ4;
    }
    if (value.find("mxfp4") != std::string::npos) {
        return moe_WEIGHT_ENCODING_MXFP4;
    }
    if (value.find("bf16") != std::string::npos || value.find("bfloat16") != std::string::npos) {
        return moe_WEIGHT_ENCODING_RAW_BF16;
    }
    if (value.find("q5_0") != std::string::npos) {
        return moe_WEIGHT_ENCODING_GGUF_Q5_0;
    }
    if (value.find("q5_1") != std::string::npos) {
        return moe_WEIGHT_ENCODING_GGUF_Q5_1;
    }
    if (value.find("q4_0") != std::string::npos) {
        return moe_WEIGHT_ENCODING_GGUF_Q4_0;
    }
    if (value.find("q4_1") != std::string::npos) {
        return moe_WEIGHT_ENCODING_GGUF_Q4_1;
    }
    if (value.find("q8_1") != std::string::npos) {
        return moe_WEIGHT_ENCODING_GGUF_Q8_1;
    }
    if (value.find("q8_0") != std::string::npos) {
        return moe_WEIGHT_ENCODING_GGUF_Q8_0;
    }
    if (value.find("nvfp4") != std::string::npos || value.find("fp4") != std::string::npos) {
        return moe_WEIGHT_ENCODING_NVFP4;
    }
    if (value.find("q8") != std::string::npos) {
        return moe_WEIGHT_ENCODING_LINEAR_Q8;
    }
    if (value.find("q4") != std::string::npos) {
        return moe_WEIGHT_ENCODING_LINEAR_Q4;
    }
    if (value.find("q3") != std::string::npos) {
        return moe_WEIGHT_ENCODING_LINEAR_Q3;
    }
    if (value.find("q2") != std::string::npos) {
        return moe_WEIGHT_ENCODING_LINEAR_Q2;
    }
    return moe_WEIGHT_ENCODING_UNKNOWN;
}

struct moe_offload_gguf_contract_t {
    uint32_t format_version = 0;
    bool backend_neutral = false;
    bool qkv_required = false;
    bool plain_kv_persistent_allowed = true;
    std::string weight_quant_family;
    std::string weight_kernel_family;
    std::string model_id;
    std::string model_name;
    std::string architecture;
    uint32_t weight_bits = 0;
    uint32_t weight_encoding = 0;
    uint32_t weight_block_size = 0;
    uint32_t modality_flags = moe_JUJU_MODALITY_TEXT;
    uint32_t optimization_catalog_count = 0;
    double reference_moe_compute_ms = 0.0;
    double reference_ram_to_gpu_ms = 0.0;
    double reference_nvme_to_ram_ms = 0.0;
    uint32_t qkv_k_bits = 0;
    uint32_t qkv_v_bits = 0;
    uint32_t qkv_group_size = 0;
    uint32_t qkv_page_size_tokens = 0;
    uint32_t qkv_sink_tokens = 0;
    uint64_t qkv_rotation_seed = 42;
    uint64_t qkv_qjl_seed = 43;
    int qkv_enable_qjl = -1;
    int qkv_enable_rotation = -1;
    uint32_t qkv_outlier_channels = 0;
    uint32_t qkv_outlier_bits = 0;
    bool has_juju_contracts = false;
};

struct moe_offload_gguf_file_t {
    std::string path;
    std::string split_parent_source_name;
    std::string split_artifact_source_name;
    uint64_t bytes = 0;
    uint64_t tensor_count = 0;
    uint64_t executable_tensor_count = 0;
    uint64_t max_executable_tensor_bytes = 0;
    uint64_t data_start = 0;
    uint32_t alignment = 32;
    uint32_t file_index = 0;
    uint32_t file_count = 0;
    uint32_t split_index = 0;
    uint32_t split_count = 0;
    bool split_enabled = false;
    bool is_juju = false;
    bool has_offload = false;
    bool has_metadata = false;
    std::string metadata_json;
    moe_offload_gguf_contract_t contract;
};

struct moe_offload_gguf_scan_t {
    std::vector<moe_offload_gguf_file_t> files;
    uint32_t expected_count = 0;
    uint32_t present_count = 0;
    uint32_t missing_count = 0;
    uint32_t first_missing = 0;
    uint64_t tensor_count = 0;
    uint64_t executable_tensor_count = 0;
    uint64_t max_executable_tensor_bytes = 0;
    uint64_t present_total_bytes = 0;
    bool juju_format = false;
    bool valid = false;
};

struct moe_gguf_tensor_dir_entry_t {
    std::string name;
    uint64_t shape[4]{};
    uint32_t dims = 0;
    uint32_t type = 0;
    uint8_t weight_encoding = moe_WEIGHT_ENCODING_UNKNOWN;
    uint64_t tensor_offset = 0;
    uint64_t file_offset = 0;
    uint64_t bytes = 0;
};

struct moe_gguf_tensor_dir_t {
    std::string path;
    uint64_t file_bytes = 0;
    uint64_t data_start = 0;
    uint32_t file_index = 0;
    uint32_t file_count = 0;
    uint32_t alignment = 32;
    std::vector<moe_gguf_tensor_dir_entry_t> tensors;
};

// Forward declaration for JUJU strategy application
static int moe_engine_apply_juju_strategy(
    moe_pc_engine* engine,
    const moe_offload_gguf_scan_t& scan,
    const moe_offload_gguf_contract_t& contract
);
