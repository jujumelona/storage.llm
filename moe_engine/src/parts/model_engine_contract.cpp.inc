// Engine contract loading
// Loads offload contract into engine state

static int moe_engine_load_offload_gguf_contract(
    moe_pc_engine* engine,
    const std::string& root,
    moe_model_root_check_t* out_check
) {
    moe_offload_gguf_scan_t scan{};
    if (!moe_scan_offload_model_root(root, &scan)) {
        return 0;
    }
    moe_fill_check_from_offload_gguf_scan(scan, out_check);
    moe_engine_clear_offload_gguf_contract(engine);
    if (engine) {
        engine->juju_split_missing_count.store(scan.juju_format ? scan.missing_count : 0, std::memory_order_relaxed);
    }
    if (!engine || !scan.valid || scan.files.empty()) {
        return 1;
    }
    const moe_offload_gguf_file_t& primary = scan.files.front();
    engine->offload_gguf_valid.store(1, std::memory_order_relaxed);
    engine->offload_gguf_file_count.store(scan.expected_count, std::memory_order_relaxed);
    engine->offload_gguf_tensor_count.store(scan.tensor_count, std::memory_order_relaxed);
    engine->offload_gguf_executable_tensor_count.store(scan.executable_tensor_count, std::memory_order_relaxed);
    engine->offload_gguf_first_file = primary.path;
    engine->offload_gguf_metadata_json = primary.metadata_json;
    engine->dynamic_architecture = primary.contract.architecture;
    engine->dynamic_model_id = primary.contract.model_id.empty() ?
        primary.contract.model_name : primary.contract.model_id;
    engine->offload_weight_quant_bits.store(primary.contract.weight_bits, std::memory_order_relaxed);
    engine->offload_weight_encoding.store(primary.contract.weight_encoding, std::memory_order_relaxed);
    engine->offload_weight_block_size.store(primary.contract.weight_block_size, std::memory_order_relaxed);
    engine->offload_optimization_catalog_count.store(primary.contract.optimization_catalog_count, std::memory_order_relaxed);
    engine->offload_weight_quant_family = primary.contract.weight_quant_family;
    engine->offload_weight_kernel_family = primary.contract.weight_kernel_family;
    engine->offload_qkv_cfg = moe_qkv_config_from_offload_contract(primary.contract);
    engine->offload_qkv_cfg_available.store(true, std::memory_order_release);
    const int qkv_required = primary.contract.qkv_required ||
        !primary.contract.plain_kv_persistent_allowed ||
        primary.contract.qkv_k_bits != 0 ||
        primary.contract.qkv_v_bits != 0;
    if (qkv_required) {
        engine->qkv_forced_by_format.store(1, std::memory_order_relaxed);
        engine->config.kv_mode = moe_KV_MODE_QKV;
        qkv_set_mode(true);
    }
    moe_engine_apply_juju_strategy(engine, scan, primary.contract);
    std::fprintf(stderr,
        "[storagellm] offload %s contract loaded: files=%u first=%s qkv=%s weight=%s/%ub catalog=%u\n",
        scan.juju_format ? "JUJU" : "GGUF",
        scan.expected_count,
        primary.path.c_str(),
        qkv_required ? "required" : "available",
        primary.contract.weight_quant_family.empty() ? "unknown" : primary.contract.weight_quant_family.c_str(),
        primary.contract.weight_bits,
        primary.contract.optimization_catalog_count);
    std::fflush(stderr);
    return 1;
}
