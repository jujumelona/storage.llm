// File operations and path utilities

static void open_ifstream_path(
    std::ifstream& input,
    const std::string& path,
    std::ios::openmode mode = std::ios::in
) {
#ifdef _WIN32
    const std::wstring wide = storagellm::wide_path_from_utf8(path);
    if (!wide.empty()) {
        input.open(wide.c_str(), mode);
    }
#else
    input.open(path, mode);
#endif
}

static void open_ofstream_path(
    std::ofstream& output,
    const std::string& path,
    std::ios::openmode mode = std::ios::out
) {
#ifdef _WIN32
    const std::wstring wide = storagellm::wide_path_from_utf8(path);
    if (!wide.empty()) {
        output.open(wide.c_str(), mode);
    }
#else
    output.open(path, mode);
#endif
}

static int file_size_bytes(const std::string& path, uint64_t* out_bytes) {
    if (out_bytes) {
        *out_bytes = 0;
    }
    if (path.empty()) {
        return 0;
    }
#ifdef _WIN32
    const std::wstring wide = storagellm::wide_path_from_utf8(path);
    if (wide.empty()) {
        return 0;
    }
    WIN32_FILE_ATTRIBUTE_DATA data{};
    if (!GetFileAttributesExW(wide.c_str(), GetFileExInfoStandard, &data)) {
        return 0;
    }
    if (data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
        return 0;
    }
    ULARGE_INTEGER value{};
    value.HighPart = data.nFileSizeHigh;
    value.LowPart = data.nFileSizeLow;
    if (out_bytes) {
        *out_bytes = value.QuadPart;
    }
    return 1;
#else
    struct stat st{};
    if (stat(path.c_str(), &st) != 0 || S_ISDIR(st.st_mode)) {
        return 0;
    }
    if (out_bytes) {
        *out_bytes = (uint64_t)st.st_size;
    }
    return 1;
#endif
}

static std::string resolve_model_file_path(
    const std::string& root,
    const std::string& file,
    const std::string& fallback
) {
    if (!root.empty() && !file.empty()) {
        const std::string rooted = storagellm::path_join(root, file);
        uint64_t ignored = 0;
        if (file_size_bytes(rooted, &ignored)) {
            return rooted;
        }
        const size_t slash = file.find_last_of("/\\");
        if (slash != std::string::npos && slash + 1 < file.size()) {
            const std::string flat = storagellm::path_join(root, file.substr(slash + 1));
            if (file_size_bytes(flat, &ignored)) {
                return flat;
            }
        }
        return rooted;
    }
    if (!fallback.empty()) {
        return fallback;
    }
    return file;
}
