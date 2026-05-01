// Model file reading functions
// Reads GGUF and JUJU offload files and extracts metadata

static int moe_read_offload_juju_file(
    const std::string& path,
    moe_offload_gguf_file_t* out
) {
    if (!out) {
        return 0;
    }
    *out = moe_offload_gguf_file_t{};
    out->path = path;
    out->is_juju = true;
    out->has_offload = true;
    out->has_metadata = true;
    moe_juju_header_t header{};
    std::vector<moe_juju_section_entry_t> sections;
    uint64_t file_bytes = 0;
    if (!moe_juju_read_table_local(path, &header, &sections, &file_bytes)) {
        return 0;
    }
    out->bytes = file_bytes;
    out->alignment = 4096;
    out->contract.format_version = header.version_major;
    out->contract.has_juju_contracts = true;
    out->contract.qkv_required = header.qkv_policy != 0;
    out->contract.plain_kv_persistent_allowed = !out->contract.qkv_required;
    out->contract.weight_bits = header.quant_type;
    if (header.model_name[0]) {
        size_t model_name_len = 0;
        while (model_name_len < sizeof(header.model_name) && header.model_name[model_name_len]) {
            ++model_name_len;
        }
        out->contract.model_name.assign(header.model_name, model_name_len);
    }

    std::string model_meta_json;
    if (moe_read_juju_json_section(path, sections, moe_JUJU_SECTION_MODEL_META, file_bytes, &model_meta_json)) {
        out->metadata_json = model_meta_json;
    }
    std::string qkv_json;
    if (moe_read_juju_json_section(path, sections, moe_JUJU_SECTION_QKV_POLICY, file_bytes, &qkv_json)) {
        out->contract.qkv_required = true;
        out->contract.plain_kv_persistent_allowed = false;
        if (!out->metadata_json.empty()) {
            out->metadata_json.push_back('\n');
        }
        out->metadata_json += qkv_json;
    }
    moe_parse_offload_metadata_json(out);
    out->contract.has_juju_contracts = true;
    if (out->contract.qkv_required) {
        out->contract.plain_kv_persistent_allowed = false;
    }
    if (out->contract.weight_bits == 0 && header.quant_type != 0) {
        out->contract.weight_bits = header.quant_type;
    }
    if (out->contract.weight_encoding == 0 && !out->contract.weight_quant_family.empty()) {
        out->contract.weight_encoding = moe_weight_encoding_from_family(out->contract.weight_quant_family);
    }

    std::string index_json;
    if (moe_read_juju_tensor_index_json(path, sections, file_bytes, &index_json)) {
        moe_parse_juju_split_metadata_json(out, index_json);
        uint64_t tensor_count = 0;
        uint64_t executable_count = 0;
        uint64_t max_executable = 0;
        if (moe_parse_juju_tensor_index_json(
                index_json,
                nullptr,
                &tensor_count,
                &executable_count,
                &max_executable)) {
            out->tensor_count = tensor_count;
            out->executable_tensor_count = executable_count;
            out->max_executable_tensor_bytes = max_executable;
        }
    }
    if (out->tensor_count == 0) {
        out->tensor_count = header.section_count;
    }
    return 1;
}

static int moe_read_offload_gguf_file(
    const std::string& path,
    moe_offload_gguf_file_t* out
) {
    if (!out) {
        return 0;
    }
    *out = moe_offload_gguf_file_t{};
    out->path = path;
    file_size_bytes(path, &out->bytes);
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
        kv_count > 1000000ull) {
        return 0;
    }
    out->tensor_count = tensor_count;
    for (uint64_t i = 0; i < kv_count; ++i) {
        std::string key;
        uint32_t type = 0;
        if (!moe_gguf_read_string(input, &key, 65536) ||
            !moe_gguf_read_u32(input, &type)) {
            return 0;
        }
        const bool is_offload = key.rfind("offload.", 0) == 0;
        if (is_offload) {
            out->has_offload = true;
        }
        uint64_t u64_value = 0;
        double f64_value = 0.0;
        std::string string_value;
        int consumed = 0;
        if (key == "general.alignment" && moe_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            if (u64_value >= 8 && u64_value <= 65536) {
                out->alignment = moe_u64_to_u32_clamped(u64_value, 32);
            }
        } else if ((key == "general.name" || key == "general.basename" || key == "general.file_type") &&
            type == moe_GGUF_TYPE_STRING) {
            consumed = 1;
            if (moe_gguf_read_string(input, &string_value, 65536) &&
                out->contract.model_name.empty()) {
                out->contract.model_name = string_value;
            }
        } else if (key == "general.architecture" && type == moe_GGUF_TYPE_STRING) {
            consumed = 1;
            if (moe_gguf_read_string(input, &string_value, 65536)) {
                out->contract.architecture = string_value;
            }
        } else if ((key == "offload.model_id" ||
                    key == "offload.repo_id" ||
                    key == "offload.source_model_id" ||
                    key == "offload.source_repo_id" ||
                    key == "offload.target_repo_id") &&
                   type == moe_GGUF_TYPE_STRING) {
            consumed = 1;
            if (moe_gguf_read_string(input, &string_value, 65536)) {
                out->contract.model_id = moe_repo_tail_or_value(string_value);
            }
        } else if (key == "offload.model_name" && type == moe_GGUF_TYPE_STRING) {
            consumed = 1;
            moe_gguf_read_string(input, &out->contract.model_name, 65536);
        } else if (key == "offload.metadata_v2" && type == moe_GGUF_TYPE_STRING) {
            consumed = 1;
            if (moe_gguf_read_string(input, &out->metadata_json, 128ull * 1024ull * 1024ull)) {
                out->has_metadata = !out->metadata_json.empty();
            }
        } else if (key == "offload.format_version" && moe_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->contract.format_version = moe_u64_to_u32_clamped(u64_value, 0);
        } else if (key == "offload.backend_neutral" && moe_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->contract.backend_neutral = u64_value != 0;
        } else if (key == "offload.file_index" && moe_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->file_index = moe_u64_to_u32_clamped(u64_value, 0);
        } else if (key == "offload.file_count" && moe_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->file_count = moe_u64_to_u32_clamped(u64_value, 0);
        } else if (key == "offload.weight_bits" && moe_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->contract.weight_bits = moe_u64_to_u32_clamped(u64_value, 0);
        } else if (key == "offload.weight_encoding" && moe_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->contract.weight_encoding = moe_u64_to_u32_clamped(u64_value, 0);
        } else if (key == "offload.weight_block_size" && moe_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->contract.weight_block_size = moe_u64_to_u32_clamped(u64_value, 0);
        } else if (key == "offload.optimization_catalog_count" && moe_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->contract.optimization_catalog_count = moe_u64_to_u32_clamped(u64_value, 0);
        } else if ((key == "offload.moe_compute_ms" ||
                    key == "offload.reference_moe_compute_ms") &&
                   moe_gguf_read_float_value(input, type, &f64_value)) {
            consumed = 1;
            out->contract.reference_moe_compute_ms = f64_value;
        } else if ((key == "offload.ram_to_gpu_ms" ||
                    key == "offload.reference_ram_to_gpu_ms") &&
                   moe_gguf_read_float_value(input, type, &f64_value)) {
            consumed = 1;
            out->contract.reference_ram_to_gpu_ms = f64_value;
        } else if ((key == "offload.nvme_to_ram_ms" ||
                    key == "offload.reference_nvme_to_ram_ms") &&
                   moe_gguf_read_float_value(input, type, &f64_value)) {
            consumed = 1;
            out->contract.reference_nvme_to_ram_ms = f64_value;
        } else if (key == "offload.weight_quant_family" && type == moe_GGUF_TYPE_STRING) {
            consumed = 1;
            moe_gguf_read_string(input, &out->contract.weight_quant_family, 65536);
        } else if (key == "offload.weight_kernel_family" && type == moe_GGUF_TYPE_STRING) {
            consumed = 1;
            moe_gguf_read_string(input, &out->contract.weight_kernel_family, 65536);
        } else if (key == "offload.qkv_k_bits" && moe_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->contract.qkv_k_bits = moe_u64_to_u32_clamped(u64_value, 0);
        } else if (key == "offload.qkv_v_bits" && moe_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->contract.qkv_v_bits = moe_u64_to_u32_clamped(u64_value, 0);
        } else if (key == "offload.qkv_group_size" && moe_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->contract.qkv_group_size = moe_u64_to_u32_clamped(u64_value, 0);
        } else if (key == "offload.qkv_page_size_tokens" && moe_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->contract.qkv_page_size_tokens = moe_u64_to_u32_clamped(u64_value, 0);
        } else if (key == "offload.qkv_sink_tokens" && moe_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->contract.qkv_sink_tokens = moe_u64_to_u32_clamped(u64_value, 0);
        } else if (key == "offload.qkv_rotation_seed" && moe_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->contract.qkv_rotation_seed = u64_value;
        } else if (key == "offload.qkv_qjl_seed" && moe_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->contract.qkv_qjl_seed = u64_value;
        } else if (key == "offload.qkv_enable_qjl" && moe_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->contract.qkv_enable_qjl = u64_value ? 1 : 0;
        } else if (key == "offload.qkv_enable_rotation" && moe_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->contract.qkv_enable_rotation = u64_value ? 1 : 0;
        } else if ((key == "offload.final_model_structure_contract" ||
                    key == "offload.pipeline_budget_contract" ||
                    key == "offload.execution_path_contract" ||
                    key == "offload.expert_segmentation_contract" ||
                    key == "offload.chunk_io_contract" ||
                    key == "offload.universal_tier_contract" ||
                    key == "offload.qkv_policy_contract") &&
                   type == moe_GGUF_TYPE_STRING) {
            consumed = 1;
            std::string ignored;
            moe_gguf_read_string(input, &ignored, 16ull * 1024ull * 1024ull);
            out->contract.has_juju_contracts = true;
        }
        if (!consumed && !moe_gguf_skip_value(input, type)) {
            return 0;
        }
    }
    moe_parse_offload_metadata_json(out);
    if (out->contract.qkv_k_bits || out->contract.qkv_v_bits) {
        out->contract.qkv_required = true;
    }
    if (out->contract.weight_encoding == 0 && !out->contract.weight_quant_family.empty()) {
        out->contract.weight_encoding = moe_weight_encoding_from_family(out->contract.weight_quant_family);
    }
    if (out->contract.qkv_required) {
        out->contract.plain_kv_persistent_allowed = false;
    }
    if (!moe_scan_gguf_tensor_infos(input, tensor_count, out)) {
        return 0;
    }
    return out->has_offload ? 1 : 0;
}
