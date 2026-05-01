// Offload/JUJU model-root contract types and metadata parsers.
// Platform, GGUF, JSON, JUJU, file-reading, scan, and API helpers live in
// their dedicated parts and are included around this file by moe_pc_engine.cpp.

static uint32_t moe_weight_encoding_from_family(const std::string& family) {
    const std::string value = moe_lower_ascii_copy(family);
    if (value.find("iq2") != std::string::npos || value.find("ud-iq2") != std::string::npos) {
        return moe_WEIGHT_ENCODING_GGUF_IQ2;
    }
    if (value.find("iq3") != std::string::npos) {
        return moe_WEIGHT_ENCODING_GGUF_IQ3;
    }
    if (value.find("iq4") != std::string::npos) {
        return moe_WEIGHT_ENCODING_GGUF_IQ4;
    }
    if (value.find("mxfp4") != std::string::npos) {
        return moe_WEIGHT_ENCODING_MXFP4;
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

static int moe_engine_apply_juju_strategy(
    moe_pc_engine* engine,
    const moe_offload_gguf_scan_t& scan,
    const moe_offload_gguf_contract_t& contract
);

struct moe_gguf_tensor_dir_entry_t {
    std::string name;
    uint64_t shape[4]{};
    uint32_t dims = 0;
    uint32_t type = 0;
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

// NOTE: JUJU tensor structures and parsing functions moved to juju_parser.cpp.inc

static int moe_gguf_tensor_name_executable(const std::string& name) {
    if (name.empty()) {
        return 0;
    }
    const std::string value = moe_lower_ascii_copy(name);
    if (value == "token_embd.weight" ||
        value == "output.weight" ||
        value == "output_norm.weight" ||
        value.find("blk.") == 0 ||
        value.find(".blk.") != std::string::npos ||
        value.find("model.layers.") != std::string::npos) {
        return 1;
    }
    return 0;
}

// NOTE: JUJU parsing functions moved to juju_parser.cpp.inc

static int moe_read_gguf_tensor_infos(
    std::ifstream& input,
    uint64_t tensor_count,
    uint32_t alignment,
    uint64_t file_bytes,
    std::vector<moe_gguf_tensor_dir_entry_t>* tensors,
    uint64_t* data_start,
    uint64_t* executable_count,
    uint64_t* max_executable_bytes
) {
    if (!tensors || tensor_count > 10000000ull) {
        return 0;
    }
    tensors->clear();
    tensors->reserve((size_t)std::min<uint64_t>(tensor_count, 1000000ull));
    for (uint64_t i = 0; i < tensor_count; ++i) {
        moe_gguf_tensor_dir_entry_t entry{};
        if (!moe_gguf_read_string(input, &entry.name, 1024) ||
            !moe_gguf_read_u32(input, &entry.dims) ||
            entry.dims > 64) {
            return 0;
        }
        for (uint32_t d = 0; d < entry.dims; ++d) {
            uint64_t dim = 0;
            if (!moe_gguf_read_u64(input, &dim)) {
                return 0;
            }
            if (d < 4) {
                entry.shape[d] = dim;
            }
        }
        if (!moe_gguf_read_u32(input, &entry.type) ||
            !moe_gguf_read_u64(input, &entry.tensor_offset)) {
            return 0;
        }
        tensors->push_back(std::move(entry));
    }
    const uint64_t end = (uint64_t)input.tellg();
    const uint64_t base = moe_align_u64(end, alignment);
    if (data_start) {
        *data_start = base;
    }
    std::sort(tensors->begin(), tensors->end(), [](const moe_gguf_tensor_dir_entry_t& a, const moe_gguf_tensor_dir_entry_t& b) {
        return a.tensor_offset < b.tensor_offset;
    });
    uint64_t executable = 0;
    uint64_t max_exec = 0;
    const uint64_t payload_bytes = file_bytes > base ? file_bytes - base : 0;
    for (size_t i = 0; i < tensors->size(); ++i) {
        moe_gguf_tensor_dir_entry_t& entry = (*tensors)[i];
        const uint64_t next = (i + 1u < tensors->size()) ? (*tensors)[i + 1u].tensor_offset : payload_bytes;
        entry.bytes = next > entry.tensor_offset ? next - entry.tensor_offset : 0;
        entry.file_offset = base + entry.tensor_offset;
        if (moe_gguf_tensor_name_executable(entry.name)) {
            ++executable;
            if (entry.bytes > max_exec) {
                max_exec = entry.bytes;
            }
        }
    }
    if (executable_count) {
        *executable_count = executable;
    }
    if (max_executable_bytes) {
        *max_executable_bytes = max_exec;
    }
    return 1;
}

static int moe_read_gguf_tensor_directory(
    const std::string& path,
    moe_gguf_tensor_dir_t* out
) {
    if (!out || path.empty()) {
        return 0;
    }
    *out = moe_gguf_tensor_dir_t{};
    out->path = path;
    file_size_bytes(path, &out->file_bytes);
    std::ifstream input;
    open_ifstream_path(input, path, std::ios::in | std::ios::binary);
    if (!input) {
        return 0;
    }
    char magic[4]{};
    uint32_t version = 0;
    uint64_t tensor_count = 0;
    uint64_t kv_count = 0;
    if (!moe_gguf_read_exact(input, magic, sizeof(magic)) ||
        std::memcmp(magic, "GGUF", 4) != 0 ||
        !moe_gguf_read_u32(input, &version) ||
        !moe_gguf_read_u64(input, &tensor_count) ||
        !moe_gguf_read_u64(input, &kv_count) ||
        version == 0 ||
        kv_count > 1000000ull ||
        tensor_count > 10000000ull) {
        return 0;
    }
    for (uint64_t i = 0; i < kv_count; ++i) {
        std::string key;
        uint32_t type = 0;
        if (!moe_gguf_read_string(input, &key, 65536) ||
            !moe_gguf_read_u32(input, &type)) {
            return 0;
        }
        uint64_t u64_value = 0;
        int consumed = 0;
        if (key == "general.alignment" && moe_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            if (u64_value >= 8 && u64_value <= 65536) {
                out->alignment = (uint32_t)u64_value;
            }
        } else if (key == "offload.file_index" && moe_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->file_index = moe_u64_to_u32_clamped(u64_value, 0);
        } else if (key == "offload.file_count" && moe_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->file_count = moe_u64_to_u32_clamped(u64_value, 0);
        }
        if (!consumed && !moe_gguf_skip_value(input, type)) {
            return 0;
        }
    }
    uint64_t executable = 0;
    uint64_t max_executable = 0;
    return moe_read_gguf_tensor_infos(
        input,
        tensor_count,
        out->alignment,
        out->file_bytes,
        &out->tensors,
        &out->data_start,
        &executable,
        &max_executable);
}

static int moe_scan_gguf_tensor_infos(
    std::ifstream& input,
    uint64_t tensor_count,
    moe_offload_gguf_file_t* out
) {
    if (!out) {
        return 0;
    }
    std::vector<moe_gguf_tensor_dir_entry_t> tensors;
    uint64_t data_start = 0;
    uint64_t executable = 0;
    uint64_t max_executable = 0;
    if (!moe_read_gguf_tensor_infos(
            input,
            tensor_count,
            out->alignment,
            out->bytes,
            &tensors,
            &data_start,
            &executable,
            &max_executable)) {
        return 0;
    }
    out->tensor_count = tensor_count;
    out->data_start = data_start;
    out->executable_tensor_count = executable;
    out->max_executable_tensor_bytes = max_executable;
    return 1;
}

static void moe_parse_juju_split_metadata_json(
    moe_offload_gguf_file_t* meta,
    const std::string& json
) {
    if (!meta || json.empty()) {
        return;
    }
    uint64_t value = 0;
    std::string string_value;
    if (moe_json_get_u64_local(json, "split_count", &value) && value > 0) {
        meta->split_count = moe_u64_to_u32_clamped(value, 0);
        if (meta->file_count == 0) {
            meta->file_count = meta->split_count;
        }
    }
    if (moe_json_get_u64_local(json, "split_index", &value) && value > 0) {
        meta->split_index = moe_u64_to_u32_clamped(value, 0);
        if (meta->file_index == 0) {
            meta->file_index = meta->split_index;
        }
    }
    if (meta->split_parent_source_name.empty() &&
        moe_json_get_string_local(json, "parent_source_name", &string_value)) {
        meta->split_parent_source_name = string_value;
    }
    if (meta->split_parent_source_name.empty() &&
        moe_json_get_string_local(json, "source_name", &string_value)) {
        meta->split_parent_source_name = string_value;
    }
    if (meta->split_artifact_source_name.empty() &&
        moe_json_get_string_local(json, "artifact_source_name", &string_value)) {
        meta->split_artifact_source_name = string_value;
    }
    meta->split_enabled = meta->split_count > 1 && meta->split_index > 0;
}

static void moe_parse_offload_metadata_json(moe_offload_gguf_file_t* meta) {
    if (!meta || meta->metadata_json.empty()) {
        return;
    }
    meta->has_metadata = true;
    moe_parse_juju_split_metadata_json(meta, meta->metadata_json);
    uint64_t value = 0;
    double double_value = 0.0;
    int bool_value = 0;
    std::string string_value;
    if (meta->contract.format_version == 0 &&
        moe_json_get_u64_local(meta->metadata_json, "format_version", &value)) {
        meta->contract.format_version = moe_u64_to_u32_clamped(value, 0);
    }
    if (meta->file_count == 0 &&
        (moe_json_get_u64_local(meta->metadata_json, "file_count", &value) ||
         moe_json_get_u64_local(meta->metadata_json, "part_count", &value))) {
        meta->file_count = moe_u64_to_u32_clamped(value, 0);
    }
    if (meta->file_index == 0 &&
        (moe_json_get_u64_local(meta->metadata_json, "file_index", &value) ||
         moe_json_get_u64_local(meta->metadata_json, "part", &value))) {
        meta->file_index = moe_u64_to_u32_clamped(value, 0);
    }
    if (meta->contract.qkv_k_bits == 0 &&
        moe_json_get_u64_local(meta->metadata_json, "k_bits", &value)) {
        meta->contract.qkv_k_bits = moe_u64_to_u32_clamped(value, 0);
    }
    if (meta->contract.qkv_v_bits == 0 &&
        moe_json_get_u64_local(meta->metadata_json, "v_bits", &value)) {
        meta->contract.qkv_v_bits = moe_u64_to_u32_clamped(value, 0);
    }
    if (meta->contract.qkv_group_size == 0 &&
        moe_json_get_u64_local(meta->metadata_json, "group_size", &value)) {
        meta->contract.qkv_group_size = moe_u64_to_u32_clamped(value, 0);
    }
    if (meta->contract.qkv_page_size_tokens == 0 &&
        moe_json_get_u64_local(meta->metadata_json, "page_size_tokens", &value)) {
        meta->contract.qkv_page_size_tokens = moe_u64_to_u32_clamped(value, 0);
    }
    if (meta->contract.qkv_sink_tokens == 0 &&
        moe_json_get_u64_local(meta->metadata_json, "sink_tokens", &value)) {
        meta->contract.qkv_sink_tokens = moe_u64_to_u32_clamped(value, 0);
    }
    if (moe_json_get_u64_local(meta->metadata_json, "rotation_seed", &value)) {
        meta->contract.qkv_rotation_seed = value;
    }
    if (moe_json_get_u64_local(meta->metadata_json, "qjl_seed", &value)) {
        meta->contract.qkv_qjl_seed = value;
    }
    if (moe_json_get_bool_local(meta->metadata_json, "enable_qjl", &bool_value)) {
        meta->contract.qkv_enable_qjl = bool_value != 0 ? 1 : 0;
    }
    if (moe_json_get_bool_local(meta->metadata_json, "enable_rotation", &bool_value)) {
        meta->contract.qkv_enable_rotation = bool_value != 0 ? 1 : 0;
    }
    if (moe_json_get_bool_local(meta->metadata_json, "qkv_packed_cache_required", &bool_value)) {
        meta->contract.qkv_required = bool_value != 0;
    }
    if (moe_json_get_bool_local(meta->metadata_json, "persistent_plain_kv_cache_allowed", &bool_value)) {
        meta->contract.plain_kv_persistent_allowed = bool_value != 0;
    }
    if (meta->contract.weight_quant_family.empty() &&
        moe_json_get_string_local(meta->metadata_json, "source_weight_quant_family", &string_value)) {
        meta->contract.weight_quant_family = string_value;
    }
    if (meta->contract.model_id.empty() &&
        (moe_json_get_string_local(meta->metadata_json, "model_id", &string_value) ||
         moe_json_get_string_local(meta->metadata_json, "repo_id", &string_value) ||
         moe_json_get_string_local(meta->metadata_json, "target_repo_id", &string_value) ||
         moe_json_get_string_local(meta->metadata_json, "source_model_id", &string_value) ||
         moe_json_get_string_local(meta->metadata_json, "source_repo_id", &string_value) ||
         moe_json_get_string_local(meta->metadata_json, "base_model_id", &string_value))) {
        meta->contract.model_id = moe_repo_tail_or_value(string_value);
    }
    if (meta->contract.model_name.empty() &&
        (moe_json_get_string_local(meta->metadata_json, "model_name", &string_value) ||
         moe_json_get_string_local(meta->metadata_json, "name", &string_value))) {
        meta->contract.model_name = string_value;
    }
    if (meta->contract.architecture.empty() &&
        moe_json_get_string_local(meta->metadata_json, "architecture", &string_value)) {
        meta->contract.architecture = string_value;
    }
    if (meta->contract.weight_kernel_family.empty() &&
        moe_json_get_string_local(meta->metadata_json, "source_weight_kernel_family", &string_value)) {
        meta->contract.weight_kernel_family = string_value;
    }
    if (meta->contract.weight_bits == 0 &&
        moe_json_get_u64_local(meta->metadata_json, "source_weight_bits", &value)) {
        meta->contract.weight_bits = moe_u64_to_u32_clamped(value, 0);
    }
    if (meta->contract.weight_block_size == 0 &&
        moe_json_get_u64_local(meta->metadata_json, "source_weight_block_size", &value)) {
        meta->contract.weight_block_size = moe_u64_to_u32_clamped(value, 0);
    }
    if (meta->contract.optimization_catalog_count == 0 &&
        (moe_json_get_u64_local(meta->metadata_json, "actual_count", &value) ||
         moe_json_get_u64_local(meta->metadata_json, "format_maximization_target_count", &value))) {
        meta->contract.optimization_catalog_count = moe_u64_to_u32_clamped(value, 0);
    }
    if (moe_json_get_double_local(meta->metadata_json, "moe_compute_ms", &double_value) ||
        moe_json_get_double_local(meta->metadata_json, "reference_moe_compute_ms", &double_value)) {
        meta->contract.reference_moe_compute_ms = double_value;
    }
    if (moe_json_get_double_local(meta->metadata_json, "ram_to_gpu_ms", &double_value) ||
        moe_json_get_double_local(meta->metadata_json, "reference_ram_to_gpu_ms", &double_value)) {
        meta->contract.reference_ram_to_gpu_ms = double_value;
    }
    if (moe_json_get_double_local(meta->metadata_json, "nvme_to_ram_ms", &double_value) ||
        moe_json_get_double_local(meta->metadata_json, "reference_nvme_to_ram_ms", &double_value)) {
        meta->contract.reference_nvme_to_ram_ms = double_value;
    }
    if (meta->metadata_json.find("\"final_model_structure_contract\"") != std::string::npos ||
        meta->metadata_json.find("\"pipeline_budget_contract\"") != std::string::npos ||
        meta->metadata_json.find("\"execution_path_contract\"") != std::string::npos ||
        meta->metadata_json.find("\"expert_segmentation_contract\"") != std::string::npos ||
        meta->metadata_json.find("\"chunk_io_contract\"") != std::string::npos ||
        meta->metadata_json.find("\"universal_tier_contract\"") != std::string::npos ||
        meta->metadata_json.find("\"qkv_policy_contract\"") != std::string::npos) {
        meta->contract.has_juju_contracts = true;
    }
    if (meta->contract.weight_encoding == 0 && !meta->contract.weight_quant_family.empty()) {
        meta->contract.weight_encoding = moe_weight_encoding_from_family(meta->contract.weight_quant_family);
    }
}

// NOTE: Engine contract loading moved to model_engine_contract.cpp.inc

// NOTE: Public API functions moved to model_validation_api.cpp.inc
