static void copy_check_path(char* out, size_t out_size, const std::string& path) {
    if (!out || out_size == 0) {
        return;
    }
    out[0] = '\0';
    if (path.empty()) {
        return;
    }
    std::strncpy(out, path.c_str(), out_size - 1);
    out[out_size - 1] = '\0';
}

int glm5_storage_validate_model_root(
    const char* model_root,
    glm5_model_root_check_t* out_check
) {
    if (!out_check) {
        return 0;
    }
    std::memset(out_check, 0, sizeof(*out_check));
    const std::string root = model_root ? model_root : "";
    const uint32_t count = glm5_storage_part_count();
    out_check->expected_part_count = count;
    for (uint32_t i = 0; i < count; ++i) {
        const glm5_storage_part_spec_t* part = glm5_storage_part_at(i);
        if (!part || !part->path) {
            continue;
        }
        out_check->expected_total_bytes += part->bytes;
        const std::string exact = root.empty() ? std::string(part->path) : storagellm::path_join(root, part->path);
        const std::string resolved = resolve_model_file_path(root, part->path, "");
        uint64_t actual = 0;
        if (!file_size_bytes(resolved, &actual)) {
            ++out_check->missing_part_count;
            if (!out_check->first_missing_part) {
                out_check->first_missing_part = part->part;
                copy_check_path(out_check->first_missing_path, sizeof(out_check->first_missing_path), resolved);
            }
            continue;
        }
        ++out_check->present_part_count;
        out_check->present_total_bytes += actual;
        if (actual != part->bytes) {
            ++out_check->size_mismatch_count;
            if (!out_check->first_size_mismatch_part) {
                out_check->first_size_mismatch_part = part->part;
                out_check->first_expected_bytes = part->bytes;
                out_check->first_actual_bytes = actual;
                copy_check_path(
                    out_check->first_size_mismatch_path,
                    sizeof(out_check->first_size_mismatch_path),
                    resolved
                );
            }
        }
    }
    out_check->valid =
        out_check->missing_part_count == 0 &&
        out_check->size_mismatch_count == 0;
    return 1;
}
