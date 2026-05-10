// Generic file collection by extension
// Eliminates duplication between gguf and juju file collection

static void moe_collect_files_by_extension(
    const std::string& path,
    int depth_left,
    const char* extension,
    const char* exclude_suffix,
    std::vector<std::string>* out
) {
    if (!out || path.empty() || !extension || out->size() >= 512) {
        return;
    }
    
    // Check if path is a file
    uint64_t file_bytes = 0;
    if (file_size_bytes(path, &file_bytes)) {
        if (moe_ends_with_ci(path, extension)) {
            if (!exclude_suffix || !moe_ends_with_ci(path, exclude_suffix)) {
                out->push_back(path);
            }
        }
        return;
    }
    
    // Check if path is a directory
    if (depth_left <= 0 || !moe_path_is_directory_local(path)) {
        return;
    }
    
    // Iterate directory using platform abstraction
    moe_dir_iterator_t* it = moe_open_directory(path);
    if (!it) {
        return;
    }
    
    std::string name;
    bool is_dir = false;
    while (moe_next_entry(it, &name, &is_dir)) {
        if (moe_skip_scan_dir_name(name)) {
            continue;
        }
        
        const std::string child = storagellm::path_join(path, name);
        
        if (is_dir) {
            moe_collect_files_by_extension(child, depth_left - 1, extension, exclude_suffix, out);
        } else if (moe_ends_with_ci(name, extension)) {
            if (!exclude_suffix || !moe_ends_with_ci(name, exclude_suffix)) {
                out->push_back(child);
            }
        }
        
        if (out->size() >= 512) {
            break;
        }
    }
    
    moe_close_directory(it);
}

// Wrapper for GGUF files
static void moe_collect_gguf_files(
    const std::string& path,
    int depth_left,
    std::vector<std::string>* out
) {
    moe_collect_files_by_extension(path, depth_left, ".gguf", nullptr, out);
}

// Wrapper for JUJU files
static void moe_collect_juju_files(
    const std::string& path,
    int depth_left,
    std::vector<std::string>* out
) {
    moe_collect_files_by_extension(path, depth_left, ".juju", ".juju.idx", out);
}
