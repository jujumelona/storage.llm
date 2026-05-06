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

static bool moe_qkv_contract_codebook_bits(int bits) {
    return bits >= 2 && bits <= 8;
}

static bool moe_qkv_contract_raw_bits(int bits) {
    return bits == 16 || bits == 32;
}

static qkv_config_t moe_qkv_config_from_offload_contract(const moe_offload_gguf_contract_t& contract) {
    qkv_config_t cfg = qkv_config_default(128);
    if (contract.qkv_k_bits > 0) {
        cfg.k_bits = (int)((contract.qkv_k_bits <= 8 || contract.qkv_k_bits == 16 || contract.qkv_k_bits == 32)
            ? contract.qkv_k_bits : 8);
    }
    if (contract.qkv_v_bits > 0) {
        cfg.v_bits = (int)((contract.qkv_v_bits <= 8 || contract.qkv_v_bits == 16 || contract.qkv_v_bits == 32)
            ? contract.qkv_v_bits : 8);
        cfg.normal_bits = cfg.v_bits;
    }
    if (contract.qkv_normal_bits > 0) {
        cfg.normal_bits = (int)((contract.qkv_normal_bits <= 8 || contract.qkv_normal_bits == 16 || contract.qkv_normal_bits == 32)
            ? contract.qkv_normal_bits : 8);
    }
    cfg.key_normal_bits = cfg.normal_bits;
    cfg.value_normal_bits = cfg.normal_bits;
    if (contract.qkv_key_normal_bits > 0) {
        cfg.key_normal_bits = (int)((contract.qkv_key_normal_bits <= 8 || contract.qkv_key_normal_bits == 16 || contract.qkv_key_normal_bits == 32)
            ? contract.qkv_key_normal_bits : 8);
    }
    if (contract.qkv_value_normal_bits > 0) {
        cfg.value_normal_bits = (int)((contract.qkv_value_normal_bits <= 8 || contract.qkv_value_normal_bits == 16 || contract.qkv_value_normal_bits == 32)
            ? contract.qkv_value_normal_bits : 8);
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
    } else if (contract.qkv_required) {
        cfg.enable_qjl = false;
    }
    if (contract.qkv_enable_rotation >= 0) {
        cfg.enable_rotation = contract.qkv_enable_rotation != 0;
    }
    if (contract.qkv_outlier_channels > 0) {
        cfg.outlier_channels = (int)contract.qkv_outlier_channels;
    }
    if (contract.qkv_outlier_bits > 0) {
        cfg.outlier_bits = (int)((contract.qkv_outlier_bits <= 8 || contract.qkv_outlier_bits == 16 || contract.qkv_outlier_bits == 32)
            ? contract.qkv_outlier_bits : 8);
    }
    cfg.key_outlier_bits = cfg.outlier_bits;
    cfg.value_outlier_bits = cfg.outlier_bits;
    if (contract.qkv_key_outlier_bits > 0) {
        cfg.key_outlier_bits = (int)((contract.qkv_key_outlier_bits <= 8 || contract.qkv_key_outlier_bits == 16 || contract.qkv_key_outlier_bits == 32)
            ? contract.qkv_key_outlier_bits : 8);
    }
    if (contract.qkv_value_outlier_bits > 0) {
        cfg.value_outlier_bits = (int)((contract.qkv_value_outlier_bits <= 8 || contract.qkv_value_outlier_bits == 16 || contract.qkv_value_outlier_bits == 32)
            ? contract.qkv_value_outlier_bits : 8);
    }
    const bool raw_qkv =
        moe_qkv_contract_raw_bits(cfg.k_bits) ||
        moe_qkv_contract_raw_bits(cfg.v_bits) ||
        moe_qkv_contract_raw_bits(cfg.normal_bits) ||
        moe_qkv_contract_raw_bits(cfg.outlier_bits) ||
        moe_qkv_contract_raw_bits(cfg.key_normal_bits) ||
        moe_qkv_contract_raw_bits(cfg.value_normal_bits) ||
        moe_qkv_contract_raw_bits(cfg.key_outlier_bits) ||
        moe_qkv_contract_raw_bits(cfg.value_outlier_bits);
    const bool codebook_qkv =
        moe_qkv_contract_codebook_bits(cfg.k_bits) &&
        moe_qkv_contract_codebook_bits(cfg.v_bits) &&
        moe_qkv_contract_codebook_bits(cfg.normal_bits) &&
        moe_qkv_contract_codebook_bits(cfg.outlier_bits) &&
        moe_qkv_contract_codebook_bits(cfg.key_normal_bits) &&
        moe_qkv_contract_codebook_bits(cfg.value_normal_bits) &&
        moe_qkv_contract_codebook_bits(cfg.key_outlier_bits) &&
        moe_qkv_contract_codebook_bits(cfg.value_outlier_bits);
    if (contract.qkv_required && codebook_qkv && cfg.outlier_channels > 0 && cfg.v_bits > 1 && cfg.normal_bits > cfg.v_bits) {
        std::fprintf(stderr,
            "[storagellm] qkv contract repair: normal_bits=%d -> %d for non-outlier value-channel layout\n",
            cfg.normal_bits,
            cfg.v_bits);
        cfg.normal_bits = cfg.v_bits;
        if (contract.qkv_value_normal_bits == 0) {
            cfg.value_normal_bits = cfg.v_bits;
        }
    }
    if (cfg.k_bits <= 1 || cfg.v_bits <= 1 ||
        cfg.k_bits > 8 || cfg.v_bits > 8 ||
        cfg.normal_bits > 8 || cfg.outlier_bits > 8 ||
        cfg.key_normal_bits > 8 || cfg.value_normal_bits > 8 ||
        cfg.key_outlier_bits > 8 || cfg.value_outlier_bits > 8 || raw_qkv) {
        cfg.enable_qjl = false;
    } else if (contract.qkv_required && codebook_qkv && !cfg.enable_qjl) {
        std::fprintf(stderr,
            "[storagellm] qkv contract repair: enable_qjl=0 -> 1 for required codebook QKV cache\n");
        cfg.enable_qjl = true;
    }
    cfg.plain_kv_persistent_storage = contract.plain_kv_persistent_allowed;
    return cfg;
}
