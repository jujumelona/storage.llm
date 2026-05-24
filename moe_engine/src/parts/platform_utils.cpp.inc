// Platform-specific utility functions
// UTF-8 conversion, path utilities, string utilities

#if defined(__linux__)
#include <sched.h>
#endif

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

static uint32_t moe_parse_env_u32(const char* name) {
    const char* value = name ? std::getenv(name) : nullptr;
    if (!value || !value[0]) {
        return 0;
    }
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (end == value || parsed == 0 || parsed > 4096ul) {
        return 0;
    }
    return (uint32_t)parsed;
}

static uint32_t moe_linux_cpu_affinity_count() {
#if defined(__linux__) && defined(CPU_SETSIZE)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        uint32_t count = 0;
        for (int i = 0; i < CPU_SETSIZE; ++i) {
            if (CPU_ISSET(i, &set)) {
                ++count;
            }
        }
        return count;
    }
#endif
    return 0;
}

static uint32_t moe_linux_cpu_quota_count() {
#if defined(__linux__)
    const char* paths[] = {
        "/sys/fs/cgroup/cpu.max",
        "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
    };
    {
        std::ifstream in(paths[0]);
        std::string quota;
        std::string period;
        if (in >> quota >> period && quota != "max") {
            char* end_q = nullptr;
            char* end_p = nullptr;
            const unsigned long long q = std::strtoull(quota.c_str(), &end_q, 10);
            const unsigned long long p = std::strtoull(period.c_str(), &end_p, 10);
            if (end_q && *end_q == '\0' && end_p && *end_p == '\0' && q > 0 && p > 0) {
                return (uint32_t)std::max<unsigned long long>(1ull, (q + p - 1ull) / p);
            }
        }
    }
    {
        std::ifstream quota_in(paths[1]);
        std::ifstream period_in("/sys/fs/cgroup/cpu/cpu.cfs_period_us");
        long long q = 0;
        long long p = 0;
        if ((quota_in >> q) && (period_in >> p) && q > 0 && p > 0) {
            return (uint32_t)std::max<long long>(1, (q + p - 1) / p);
        }
    }
#endif
    return 0;
}

static uint32_t moe_cpu_thread_count_hint() {
    const uint32_t override_threads = moe_parse_env_u32("STORAGELLM_CPU_THREADS");
    if (override_threads > 0) {
        return std::min<uint32_t>(override_threads, 256u);
    }
    uint32_t threads = UINT32_MAX;
    const unsigned int cpp_threads = std::thread::hardware_concurrency();
    if (cpp_threads > 0) {
        threads = std::min<uint32_t>(threads, (uint32_t)cpp_threads);
    }
#ifdef _WIN32
    const DWORD active = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    if (active > 0) {
        threads = std::min<uint32_t>(threads, (uint32_t)active);
    }
#else
#ifdef _SC_NPROCESSORS_ONLN
    const long online = sysconf(_SC_NPROCESSORS_ONLN);
    if (online > 0) {
        threads = std::min<uint32_t>(threads, (uint32_t)online);
    }
#endif
    const uint32_t affinity = moe_linux_cpu_affinity_count();
    if (affinity > 0) {
        threads = std::min<uint32_t>(threads, affinity);
    }
    const uint32_t quota = moe_linux_cpu_quota_count();
    if (quota > 0) {
        threads = std::min<uint32_t>(threads, quota);
    }
#endif
    if (threads == 0 || threads == UINT32_MAX) {
        threads = 1;
    }
    return std::min<uint32_t>(threads, 256u);
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
