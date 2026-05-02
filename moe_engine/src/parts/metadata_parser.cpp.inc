// 메타데이터 파싱 유틸리티
// GGUF/JUJU 메타데이터 JSON 파싱

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
    if (meta->contract.weight_encoding == 0 &&
        (moe_json_get_u64_local(meta->metadata_json, "source_weight_encoding", &value) ||
         moe_json_get_u64_local(meta->metadata_json, "weight_encoding", &value))) {
        meta->contract.weight_encoding = moe_u64_to_u32_clamped(value, 0);
    }
    if (meta->contract.weight_block_size == 0 &&
        moe_json_get_u64_local(meta->metadata_json, "source_weight_block_size", &value)) {
        meta->contract.weight_block_size = moe_u64_to_u32_clamped(value, 0);
    }
    if (moe_json_get_u64_local(meta->metadata_json, "modality_flags", &value)) {
        meta->contract.modality_flags = moe_u64_to_u32_clamped(value, moe_JUJU_MODALITY_TEXT);
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
        meta->metadata_json.find("\"format_extension_contract\"") != std::string::npos ||
        meta->metadata_json.find("\"kernel_registry_contract\"") != std::string::npos ||
        meta->metadata_json.find("\"tokenizer_contract\"") != std::string::npos ||
        meta->metadata_json.find("\"adapter_registry_contract\"") != std::string::npos ||
        meta->metadata_json.find("\"validation_contract\"") != std::string::npos ||
        meta->metadata_json.find("\"research_offload_contract\"") != std::string::npos ||
        meta->metadata_json.find("\"multimodal_contract\"") != std::string::npos ||
        meta->metadata_json.find("\"qkv_policy_contract\"") != std::string::npos) {
        meta->contract.has_juju_contracts = true;
    }
    if (meta->contract.weight_encoding == 0 && !meta->contract.weight_quant_family.empty()) {
        meta->contract.weight_encoding = moe_weight_encoding_from_family(meta->contract.weight_quant_family);
    }
}
