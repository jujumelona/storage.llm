// Platform-specific directory iteration abstraction
// Provides cross-platform directory scanning without duplication

#ifdef _WIN32
struct moe_dir_iterator_t {
    HANDLE handle;
    WIN32_FIND_DATAW data;
    bool first;
    bool valid;
};

static moe_dir_iterator_t* moe_open_directory(const std::string& path) {
    const std::string pattern = storagellm::path_join(path, "*");
    const std::wstring wide_pattern = storagellm::wide_path_from_utf8(pattern);
    if (wide_pattern.empty()) {
        return nullptr;
    }
    
    moe_dir_iterator_t* it = new moe_dir_iterator_t{};
    it->handle = FindFirstFileW(wide_pattern.c_str(), &it->data);
    it->first = true;
    it->valid = (it->handle != INVALID_HANDLE_VALUE);
    
    if (!it->valid) {
        delete it;
        return nullptr;
    }
    return it;
}

static bool moe_next_entry(moe_dir_iterator_t* it, std::string* name, bool* is_dir) {
    if (!it || !it->valid) {
        return false;
    }
    
    if (it->first) {
        it->first = false;
    } else {
        if (!FindNextFileW(it->handle, &it->data)) {
            it->valid = false;
            return false;
        }
    }
    
    if (name) {
        *name = moe_wide_to_utf8(it->data.cFileName);
    }
    if (is_dir) {
        *is_dir = (it->data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
    }
    return true;
}

static void moe_close_directory(moe_dir_iterator_t* it) {
    if (it) {
        if (it->handle != INVALID_HANDLE_VALUE) {
            FindClose(it->handle);
        }
        delete it;
    }
}

#else  // Linux/Unix

struct moe_dir_iterator_t {
    DIR* dir;
};

static moe_dir_iterator_t* moe_open_directory(const std::string& path) {
    DIR* dir = opendir(path.c_str());
    if (!dir) {
        return nullptr;
    }
    
    moe_dir_iterator_t* it = new moe_dir_iterator_t{};
    it->dir = dir;
    return it;
}

static bool moe_next_entry(moe_dir_iterator_t* it, std::string* name, bool* is_dir) {
    if (!it || !it->dir) {
        return false;
    }
    
    dirent* entry = readdir(it->dir);
    if (!entry) {
        return false;
    }
    
    if (name) {
        *name = entry->d_name ? entry->d_name : "";
    }
    if (is_dir) {
        // Note: d_type may not be supported on all filesystems
        // Fallback to stat if needed, but for now assume it works
        *is_dir = (entry->d_type == DT_DIR);
    }
    return true;
}

static void moe_close_directory(moe_dir_iterator_t* it) {
    if (it) {
        if (it->dir) {
            closedir(it->dir);
        }
        delete it;
    }
}

#endif
