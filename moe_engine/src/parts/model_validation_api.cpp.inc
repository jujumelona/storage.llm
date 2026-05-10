// Public API functions for model validation
// Main entry points called from other modules

int moe_storage_validate_model_root(
    const char* model_root,
    moe_model_root_check_t* out_check
) {
    if (!out_check) {
        return 0;
    }
    std::memset(out_check, 0, sizeof(*out_check));
    const std::string root = model_root ? model_root : "";
    moe_offload_gguf_scan_t gguf_scan{};
    if (!root.empty() && moe_scan_offload_model_root(root, &gguf_scan)) {
        moe_fill_check_from_offload_gguf_scan(gguf_scan, out_check);
        return 1;
    }
    return 1;
}

int moe_storage_infer_model_id(
    const char* model_root,
    char* out_model_id,
    size_t out_size
) {
    if (!out_model_id || out_size == 0) {
        return 0;
    }
    out_model_id[0] = '\0';
    const std::string root = model_root ? model_root : "";
    if (root.empty()) {
        return 0;
    }
    moe_offload_gguf_scan_t gguf_scan{};
    if (!moe_scan_offload_model_root(root, &gguf_scan)) {
        return 0;
    }
    const std::string id = moe_model_id_from_offload_scan(root, gguf_scan);
    if (id.empty()) {
        return 0;
    }
    moe_copy_string_to_buffer(id, out_model_id, out_size);
    return out_model_id[0] != '\0';
}
