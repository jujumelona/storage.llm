// Model root scanning functions
// Scans GGUF and JUJU files in model root directory

static int moe_read_model_root_text_sidecar_for_scan(
    const std::string& root,
    const char* file_name,
    uint64_t max_bytes,
    std::string* out
) {
    if (out) {
        out->clear();
    }
    if (!file_name || !out) {
        return 0;
    }
    const std::string path = storagellm::path_join(root, file_name);
    uint64_t bytes = 0;
    if (!file_size_bytes(path, &bytes) || bytes == 0 || bytes > max_bytes) {
        return 0;
    }
    std::ifstream input;
    open_ifstream_path(input, path, std::ios::in | std::ios::binary);
    if (!input) {
        return 0;
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    *out = buffer.str();
    return !out->empty();
}

static void moe_apply_runtime_contract_sidecar_to_scan(
    const std::string& root,
    moe_offload_gguf_scan_t* scan
) {
    if (!scan || scan->files.empty()) {
        return;
    }
    std::string contract_json;
    if (!moe_read_model_root_text_sidecar_for_scan(
            root, "storagellm_runtime_contract.json", 128ull * 1024ull * 1024ull, &contract_json)) {
        return;
    }
    for (moe_offload_gguf_file_t& file : scan->files) {
        if (!file.metadata_json.empty()) {
            file.metadata_json.push_back('\n');
        }
        file.metadata_json += contract_json;
        file.has_metadata = true;
        moe_parse_offload_metadata_json(&file);
        if (file.contract.qkv_required) {
            file.contract.plain_kv_persistent_allowed = false;
        }
    }
}

static int moe_scan_offload_gguf_root(
    const std::string& root,
    moe_offload_gguf_scan_t* out
) {
    if (!out) {
        return 0;
    }
    *out = moe_offload_gguf_scan_t{};
    std::vector<std::string> paths;
    moe_collect_gguf_files(root, 4, &paths);
    if (paths.empty()) {
        return 0;
    }
    std::sort(paths.begin(), paths.end());
    std::unordered_set<uint32_t> seen_indexes;
    bool saw_indexed = false;
    for (const std::string& path : paths) {
        moe_offload_gguf_file_t file{};
        if (!moe_read_offload_gguf_file(path, &file)) {
            continue;
        }
        out->present_total_bytes += file.bytes;
        if (file.file_count > out->expected_count) {
            out->expected_count = file.file_count;
        }
        if (file.file_index > 0) {
            saw_indexed = true;
            seen_indexes.insert(file.file_index);
        }
        out->files.push_back(std::move(file));
        out->tensor_count += out->files.back().tensor_count;
        out->executable_tensor_count += out->files.back().executable_tensor_count;
        if (out->files.back().max_executable_tensor_bytes > out->max_executable_tensor_bytes) {
            out->max_executable_tensor_bytes = out->files.back().max_executable_tensor_bytes;
        }
    }
    if (out->files.empty()) {
        return 0;
    }
    if (out->expected_count == 0) {
        out->expected_count = (uint32_t)out->files.size();
    }
    out->present_count = saw_indexed ? (uint32_t)seen_indexes.size() : (uint32_t)out->files.size();
    if (out->expected_count > out->present_count) {
        out->missing_count = out->expected_count - out->present_count;
    }
    if (saw_indexed && out->expected_count > 0) {
        for (uint32_t i = 1; i <= out->expected_count; ++i) {
            if (seen_indexes.find(i) == seen_indexes.end()) {
                out->first_missing = i;
                break;
            }
        }
    } else if (out->missing_count) {
        out->first_missing = out->present_count + 1;
    }
    out->valid = out->expected_count > 0 && out->present_count >= out->expected_count;
    return 1;
}

static int moe_scan_offload_juju_root(
    const std::string& root,
    moe_offload_gguf_scan_t* out
) {
    if (!out) {
        return 0;
    }
    *out = moe_offload_gguf_scan_t{};
    std::vector<std::string> paths;
    moe_collect_juju_files(root, 4, &paths);
    if (paths.empty()) {
        return 0;
    }
    std::sort(paths.begin(), paths.end());
    struct split_group_scan_t {
        uint32_t expected = 0;
        std::unordered_set<uint32_t> seen;
    };
    std::unordered_map<std::string, split_group_scan_t> split_groups;
    std::unordered_set<uint32_t> legacy_seen_indexes;
    uint32_t legacy_expected = 0;
    uint32_t legacy_unindexed_present = 0;
    bool saw_legacy_indexed = false;
    for (const std::string& path : paths) {
        moe_offload_gguf_file_t file{};
        if (!moe_read_offload_juju_file(path, &file)) {
            continue;
        }
        out->juju_format = true;
        out->present_total_bytes += file.bytes;
        if (file.split_enabled) {
            std::string group_key = file.split_parent_source_name;
            if (group_key.empty()) {
                group_key = file.contract.model_id.empty() ? file.path : file.contract.model_id;
            }
            split_group_scan_t& group = split_groups[group_key];
            group.expected = std::max(group.expected, file.split_count);
            group.seen.insert(file.split_index);
        } else {
            if (file.file_count > legacy_expected) {
                legacy_expected = file.file_count;
            }
            if (file.file_index > 0) {
                saw_legacy_indexed = true;
                legacy_seen_indexes.insert(file.file_index);
            } else {
                ++legacy_unindexed_present;
            }
        }
        out->files.push_back(std::move(file));
        out->tensor_count += out->files.back().tensor_count;
        out->executable_tensor_count += out->files.back().executable_tensor_count;
        if (out->files.back().max_executable_tensor_bytes > out->max_executable_tensor_bytes) {
            out->max_executable_tensor_bytes = out->files.back().max_executable_tensor_bytes;
        }
    }
    if (out->files.empty()) {
        return 0;
    }
    uint32_t expected_total = 0;
    uint32_t present_total = 0;
    uint32_t missing_total = 0;
    uint32_t first_missing = 0;
    for (const auto& item : split_groups) {
        const split_group_scan_t& group = item.second;
        const uint32_t expected = group.expected ? group.expected : (uint32_t)group.seen.size();
        const uint32_t present = (uint32_t)group.seen.size();
        expected_total += expected;
        present_total += present;
        if (expected > present) {
            missing_total += expected - present;
            if (first_missing == 0) {
                for (uint32_t i = 1; i <= expected; ++i) {
                    if (group.seen.find(i) == group.seen.end()) {
                        first_missing = i;
                        break;
                    }
                }
            }
        }
    }
    const uint32_t legacy_indexed_present = saw_legacy_indexed ? (uint32_t)legacy_seen_indexes.size() : 0;
    if (legacy_expected == 0) {
        legacy_expected = legacy_indexed_present;
    }
    const uint32_t legacy_expected_total = legacy_expected + legacy_unindexed_present;
    const uint32_t legacy_present_total = legacy_indexed_present + legacy_unindexed_present;
    expected_total += legacy_expected_total;
    present_total += legacy_present_total;
    if (legacy_expected_total > legacy_present_total) {
        missing_total += legacy_expected_total - legacy_present_total;
        // BUGFIX 271: Only set first_missing if not already set by split_groups ★
        // Problem: first_missing set by split_groups, then overwritten by legacy check
        // Solution: Only set if still 0 (not set by split_groups)
        if (first_missing == 0) {
            for (uint32_t i = 1; i <= legacy_expected; ++i) {
                if (legacy_seen_indexes.find(i) == legacy_seen_indexes.end()) {
                    first_missing = i;
                    break;
                }
            }
        }
    }
    if (expected_total == 0) {
        expected_total = (uint32_t)out->files.size();
        present_total = (uint32_t)out->files.size();
    }
    out->expected_count = expected_total;
    out->present_count = present_total;
    out->missing_count = missing_total;
    out->first_missing = first_missing;
    out->valid = out->expected_count > 0 && out->missing_count == 0;
    return 1;
}

static int moe_scan_offload_model_root(
    const std::string& root,
    moe_offload_gguf_scan_t* out
) {
    if (moe_scan_offload_juju_root(root, out) || moe_scan_offload_gguf_root(root, out)) {
        moe_apply_runtime_contract_sidecar_to_scan(root, out);
        return 1;
    }
    return 0;
}
