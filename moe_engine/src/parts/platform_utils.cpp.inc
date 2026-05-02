// Platform-specific utility functions
// UTF-8 conversion, path utilities, string utilities

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

static void moe_platform_init_limits() {
#ifndef _WIN32
    struct rlimit rl;
    if (getrlimit(RLIMIT_NOFILE, &rl) == 0) {
        if (rl.rlim_max == RLIM_INFINITY || rl.rlim_max > 65535) {
            rl.rlim_cur = rl.rlim_max == RLIM_INFINITY ? 65535 : rl.rlim_max;
        } else {
            rl.rlim_cur = rl.rlim_max;
        }
        setrlimit(RLIMIT_NOFILE, &rl);
    }
#endif
}

#ifdef _WIN32
static std::string moe_wide_to_utf8(const wchar_t* value) {
    if (!value || !value[0]) {
        return std::string();
    }
    const int bytes = WideCharToMultiByte(CP_UTF8, 0, value, -1, nullptr, 0, nullptr, nullptr);
    if (bytes <= 0) {
        return std::string();
    }
    std::string out((size_t)bytes, '\0');
    if (!WideCharToMultiByte(CP_UTF8, 0, value, -1, &out[0], bytes, nullptr, nullptr)) {
        return std::string();
    }
    if (!out.empty() && out.back() == '\0') {
        out.pop_back();
    }
    return out;
}
#else
#include <dirent.h>
#endif

static int moe_path_is_directory_local(const std::string& path) {
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
    return (data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) ? 1 : 0;
#else
    struct stat st{};
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
#endif
}

static bool moe_ascii_ieq(char a, char b) {
    if (a >= 'A' && a <= 'Z') {
        a = (char)(a - 'A' + 'a');
    }
    if (b >= 'A' && b <= 'Z') {
        b = (char)(b - 'A' + 'a');
    }
    return a == b;
}

static bool moe_ends_with_ci(const std::string& value, const char* suffix) {
    if (!suffix) {
        return false;
    }
    const size_t n = value.size();
    const size_t m = std::strlen(suffix);
    if (m > n) {
        return false;
    }
    for (size_t i = 0; i < m; ++i) {
        if (!moe_ascii_ieq(value[n - m + i], suffix[i])) {
            return false;
        }
    }
    return true;
}

static bool moe_skip_scan_dir_name(const std::string& name) {
    return name == "." || name == ".." || name == ".git" || name == "build" ||
        name == "colab" || name == ".moe_model_lib_cache";
}

static std::string moe_lower_ascii_copy(std::string value) {
    for (char& c : value) {
        if (c >= 'A' && c <= 'Z') {
            c = (char)(c - 'A' + 'a');
        }
    }
    return value;
}

static std::string moe_path_basename_local(std::string path) {
    while (!path.empty() && (path.back() == '/' || path.back() == '\\')) {
        path.pop_back();
    }
    const size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

static std::string moe_repo_tail_or_value(const std::string& value) {
    const size_t slash = value.find_last_of('/');
    return slash == std::string::npos ? value : value.substr(slash + 1);
}

static void moe_copy_string_to_buffer(const std::string& value, char* out, size_t out_size) {
    if (!out || out_size == 0) {
        return;
    }
    out[0] = '\0';
    if (value.empty()) {
        return;
    }
    std::strncpy(out, value.c_str(), out_size - 1);
    out[out_size - 1] = '\0';
}

static std::string moe_slugify_model_id(const std::string& value) {
    std::string out;
    out.reserve(value.size());
    bool last_dash = false;
    for (char c : value) {
        unsigned char uc = static_cast<unsigned char>(c);
        if (uc >= 'A' && uc <= 'Z') {
            c = static_cast<char>(uc - 'A' + 'a');
            uc = static_cast<unsigned char>(c);
        }
        const bool keep = (uc >= 'a' && uc <= 'z') ||
            (uc >= '0' && uc <= '9') || c == '.' || c == '_';
        if (keep) {
            out.push_back(c);
            last_dash = false;
        } else if (c == '-' || c == ' ' || c == '/' || c == '\\' || c == ':') {
            if (!out.empty() && !last_dash) {
                out.push_back('-');
                last_dash = true;
            }
        }
    }
    while (!out.empty() && out.back() == '-') {
        out.pop_back();
    }
    return out;
}

static uint32_t moe_u64_to_u32_clamped(uint64_t value, uint32_t fallback) {
    if (value == 0 || value > UINT32_MAX) {
        return fallback;
    }
    return (uint32_t)value;
}

static uint64_t moe_align_u64(uint64_t value, uint32_t alignment) {
    const uint64_t a = alignment ? (uint64_t)alignment : 32ull;
    const uint64_t r = value % a;
    return r ? value + (a - r) : value;
}
