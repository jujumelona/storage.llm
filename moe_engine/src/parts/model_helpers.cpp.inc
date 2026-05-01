// Model validation helper functions
// Model ID inference, check filling, QKV config conversion

static std::string moe_model_id_from_offload_scan(
    const std::string& root,
    const moe_offload_gguf_scan_t& scan
) {
    for (const moe_offload_gguf_file_t& file : scan.files) {
        const std::string slug = moe_slugify_model_id(moe_repo_tail_or_value(file.contract.model_id));
        if (!slug.empty()) {
            return slug;
        }
    }
    for (const moe_offload_gguf_file_t& file : scan.files) {
        const std::string slug = moe_slugify_model_id(file.contract.model_name);
        if (!slug.empty()) {
            return slug;
        }
    }
    for (const moe_offload_gguf_file_t& file : scan.files) {
        const std::string arch = moe_slugify_model_id(file.contract.architecture);
        if (!arch.empty()) {
            return arch + "-gguf-offload";
        }
    }
    const std::string root_slug = moe_slugify_model_id(moe_path_basename_local(root));
    return root_slug.empty() ? "storagellm-offload" : root_slug;
}

static void moe_fill_check_from_offload_gguf_scan(
    const moe_offload_gguf_scan_t& scan,
    moe_model_root_check_t* out_check
) {
    if (!out_check) {
        return;
    }
    std::memset(out_check, 0, sizeof(*out_check));
    out_check->expected_part_count = scan.expected_count;
    out_check->present_part_count = scan.present_count;
    out_check->missing_part_count = scan.missing_count;
    out_check->expected_total_bytes = scan.present_total_bytes;
    out_check->present_total_bytes = scan.present_total_bytes;
    if (scan.missing_count) {
        out_check->first_missing_part = scan.first_missing;
        std::ostringstream missing;
        missing << (scan.juju_format ? "JUJU shard index " : "offload GGUF shard index ")
                << scan.first_missing;
        copy_check_path(out_check->first_missing_path, sizeof(out_check->first_missing_path), missing.str());
    }
    out_check->valid = scan.valid ? 1 : 0;
}

static qkv_config_t moe_qkv_config_from_offload_contract(const moe_offload_gguf_contract_t& contract) {
    qkv_config_t cfg = qkv_config_default(128);
    if (contract.qkv_k_bits > 0) {
        cfg.k_bits = (int)std::min<uint32_t>(contract.qkv_k_bits, 4);
    }
    if (contract.qkv_v_bits > 0) {
        cfg.v_bits = (int)std::min<uint32_t>(contract.qkv_v_bits, 4);
        cfg.normal_bits = cfg.v_bits;
    }
    if (contract.qkv_group_size > 0) {
        cfg.group_size = contract.qkv_group_size;
    }
    if (contract.qkv_page_size_tokens > 0) {
        cfg.page_size_tokens = contract.qkv_page_size_tokens;
    }
    if (contract.qkv_sink_tokens > 0) {
        cfg.sink_tokens = contract.qkv_sink_tokens;
    }
    cfg.rotation_seed = contract.qkv_rotation_seed;
    cfg.qjl_seed = contract.qkv_qjl_seed;
    if (contract.qkv_enable_qjl >= 0) {
        cfg.enable_qjl = contract.qkv_enable_qjl != 0;
    }
    if (contract.qkv_enable_rotation >= 0) {
        cfg.enable_rotation = contract.qkv_enable_rotation != 0;
    }
    if (contract.qkv_outlier_channels > 0) {
        cfg.outlier_channels = (int)contract.qkv_outlier_channels;
    }
    if (contract.qkv_outlier_bits > 0) {
        cfg.outlier_bits = (int)std::min<uint32_t>(contract.qkv_outlier_bits, 4);
    }
    if (cfg.k_bits <= 1 || cfg.v_bits <= 1) {
        cfg.enable_qjl = false;
    }
    cfg.plain_kv_persistent_storage = contract.plain_kv_persistent_allowed;
    return cfg;
}
