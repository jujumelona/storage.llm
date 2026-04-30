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

static void moe_collect_gguf_files(
    const std::string& path,
    int depth_left,
    std::vector<std::string>* out
) {
    if (!out || path.empty() || out->size() >= 512) {
        return;
    }
    uint64_t file_bytes = 0;
    if (file_size_bytes(path, &file_bytes)) {
        if (moe_ends_with_ci(path, ".gguf")) {
            out->push_back(path);
        }
        return;
    }
    if (depth_left <= 0 || !moe_path_is_directory_local(path)) {
        return;
    }
#ifdef _WIN32
    const std::string pattern = storagellm::path_join(path, "*");
    const std::wstring wide_pattern = storagellm::wide_path_from_utf8(pattern);
    if (wide_pattern.empty()) {
        return;
    }
    WIN32_FIND_DATAW data{};
    HANDLE find = FindFirstFileW(wide_pattern.c_str(), &data);
    if (find == INVALID_HANDLE_VALUE) {
        return;
    }
    do {
        const std::string name = moe_wide_to_utf8(data.cFileName);
        if (moe_skip_scan_dir_name(name)) {
            continue;
        }
        const std::string child = storagellm::path_join(path, name);
        if (data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            moe_collect_gguf_files(child, depth_left - 1, out);
        } else if (moe_ends_with_ci(name, ".gguf")) {
            out->push_back(child);
        }
        if (out->size() >= 512) {
            break;
        }
    } while (FindNextFileW(find, &data));
    FindClose(find);
#else
    DIR* dir = opendir(path.c_str());
    if (!dir) {
        return;
    }
    while (dirent* entry = readdir(dir)) {
        const std::string name = entry->d_name ? entry->d_name : "";
        if (moe_skip_scan_dir_name(name)) {
            continue;
        }
        const std::string child = storagellm::path_join(path, name);
        if (moe_path_is_directory_local(child)) {
            moe_collect_gguf_files(child, depth_left - 1, out);
        } else if (moe_ends_with_ci(name, ".gguf")) {
            out->push_back(child);
        }
        if (out->size() >= 512) {
            break;
        }
    }
    closedir(dir);
#endif
}

enum {
    moe_GGUF_TYPE_UINT8 = 0,
    moe_GGUF_TYPE_INT8 = 1,
    moe_GGUF_TYPE_UINT16 = 2,
    moe_GGUF_TYPE_INT16 = 3,
    moe_GGUF_TYPE_UINT32 = 4,
    moe_GGUF_TYPE_INT32 = 5,
    moe_GGUF_TYPE_FLOAT32 = 6,
    moe_GGUF_TYPE_BOOL = 7,
    moe_GGUF_TYPE_STRING = 8,
    moe_GGUF_TYPE_ARRAY = 9,
    moe_GGUF_TYPE_UINT64 = 10,
    moe_GGUF_TYPE_INT64 = 11,
    moe_GGUF_TYPE_FLOAT64 = 12
};

static int moe_gguf_read_exact(std::ifstream& input, void* data, size_t bytes) {
    if (bytes == 0) {
        return 1;
    }
    input.read(static_cast<char*>(data), static_cast<std::streamsize>(bytes));
    return (size_t)input.gcount() == bytes;
}

static int moe_gguf_read_u8(std::ifstream& input, uint8_t* out) {
    return out && moe_gguf_read_exact(input, out, sizeof(*out));
}

static int moe_gguf_read_u16(std::ifstream& input, uint16_t* out) {
    return out && moe_gguf_read_exact(input, out, sizeof(*out));
}

static int moe_gguf_read_u32(std::ifstream& input, uint32_t* out) {
    return out && moe_gguf_read_exact(input, out, sizeof(*out));
}

static int moe_gguf_read_u64(std::ifstream& input, uint64_t* out) {
    return out && moe_gguf_read_exact(input, out, sizeof(*out));
}

static int moe_gguf_skip_bytes(std::ifstream& input, uint64_t bytes) {
    while (bytes > 0) {
        const uint64_t chunk = std::min<uint64_t>(bytes, 0x3fffffffull);
        input.seekg((std::streamoff)chunk, std::ios::cur);
        if (!input.good()) {
            return 0;
        }
        bytes -= chunk;
    }
    return 1;
}

static int moe_gguf_fixed_type_size(uint32_t type, uint64_t* out) {
    if (!out) {
        return 0;
    }
    switch (type) {
        case moe_GGUF_TYPE_UINT8:
        case moe_GGUF_TYPE_INT8:
        case moe_GGUF_TYPE_BOOL:
            *out = 1;
            return 1;
        case moe_GGUF_TYPE_UINT16:
        case moe_GGUF_TYPE_INT16:
            *out = 2;
            return 1;
        case moe_GGUF_TYPE_UINT32:
        case moe_GGUF_TYPE_INT32:
        case moe_GGUF_TYPE_FLOAT32:
            *out = 4;
            return 1;
        case moe_GGUF_TYPE_UINT64:
        case moe_GGUF_TYPE_INT64:
        case moe_GGUF_TYPE_FLOAT64:
            *out = 8;
            return 1;
        default:
            return 0;
    }
}

static int moe_gguf_skip_string(std::ifstream& input) {
    uint64_t len = 0;
    return moe_gguf_read_u64(input, &len) && moe_gguf_skip_bytes(input, len);
}

static int moe_gguf_read_string(std::ifstream& input, std::string* out, uint64_t max_bytes) {
    uint64_t len = 0;
    if (!out || !moe_gguf_read_u64(input, &len)) {
        return 0;
    }
    if (len > max_bytes || len > (uint64_t)std::numeric_limits<size_t>::max()) {
        moe_gguf_skip_bytes(input, len);
        return 0;
    }
    out->assign((size_t)len, '\0');
    return len == 0 || moe_gguf_read_exact(input, &(*out)[0], (size_t)len);
}

static int moe_gguf_skip_value(std::ifstream& input, uint32_t type) {
    if (type == moe_GGUF_TYPE_STRING) {
        return moe_gguf_skip_string(input);
    }
    if (type == moe_GGUF_TYPE_ARRAY) {
        uint32_t item_type = 0;
        uint64_t item_count = 0;
        if (!moe_gguf_read_u32(input, &item_type) || !moe_gguf_read_u64(input, &item_count)) {
            return 0;
        }
        if (item_type == moe_GGUF_TYPE_STRING) {
            for (uint64_t i = 0; i < item_count; ++i) {
                if (!moe_gguf_skip_string(input)) {
                    return 0;
                }
            }
            return 1;
        }
        uint64_t item_size = 0;
        if (!moe_gguf_fixed_type_size(item_type, &item_size) ||
            item_count > UINT64_MAX / item_size) {
            return 0;
        }
        return moe_gguf_skip_bytes(input, item_count * item_size);
    }
    uint64_t size = 0;
    return moe_gguf_fixed_type_size(type, &size) && moe_gguf_skip_bytes(input, size);
}

static int moe_gguf_read_uint_value(std::ifstream& input, uint32_t type, uint64_t* out) {
    if (!out) {
        return 0;
    }
    switch (type) {
        case moe_GGUF_TYPE_UINT8: {
            uint8_t v = 0;
            if (!moe_gguf_read_u8(input, &v)) return 0;
            *out = v;
            return 1;
        }
        case moe_GGUF_TYPE_BOOL: {
            uint8_t v = 0;
            if (!moe_gguf_read_u8(input, &v)) return 0;
            *out = v ? 1u : 0u;
            return 1;
        }
        case moe_GGUF_TYPE_UINT16: {
            uint16_t v = 0;
            if (!moe_gguf_read_u16(input, &v)) return 0;
            *out = v;
            return 1;
        }
        case moe_GGUF_TYPE_UINT32: {
            uint32_t v = 0;
            if (!moe_gguf_read_u32(input, &v)) return 0;
            *out = v;
            return 1;
        }
        case moe_GGUF_TYPE_UINT64:
            return moe_gguf_read_u64(input, out);
        default:
            return 0;
    }
}

static int moe_gguf_read_float_value(std::ifstream& input, uint32_t type, double* out) {
    if (!out) {
        return 0;
    }
    switch (type) {
        case moe_GGUF_TYPE_FLOAT32: {
            float v = 0.0f;
            if (!moe_gguf_read_exact(input, &v, sizeof(v))) return 0;
            *out = (double)v;
            return 1;
        }
        case moe_GGUF_TYPE_FLOAT64: {
            double v = 0.0;
            if (!moe_gguf_read_exact(input, &v, sizeof(v))) return 0;
            *out = v;
            return 1;
        }
        case moe_GGUF_TYPE_UINT8:
        case moe_GGUF_TYPE_BOOL:
        case moe_GGUF_TYPE_UINT16:
        case moe_GGUF_TYPE_UINT32:
        case moe_GGUF_TYPE_UINT64: {
            uint64_t v = 0;
            if (!moe_gguf_read_uint_value(input, type, &v)) return 0;
            *out = (double)v;
            return 1;
        }
        default:
            return 0;
    }
}

static int moe_json_find_scalar(const std::string& text, const char* key, size_t* value_pos) {
    if (!key || !value_pos) {
        return 0;
    }
    const std::string needle = std::string("\"") + key + "\"";
    size_t pos = 0;
    while ((pos = text.find(needle, pos)) != std::string::npos) {
        const size_t colon = text.find_first_not_of(" \t\r\n", pos + needle.size());
        if (colon != std::string::npos && colon < text.size() && text[colon] == ':') {
            *value_pos = text.find_first_not_of(" \t\r\n", colon + 1);
            return *value_pos != std::string::npos;
        }
        ++pos;
    }
    return 0;
}

static int moe_json_get_u64_local(const std::string& text, const char* key, uint64_t* out) {
    size_t value = 0;
    if (!out || !moe_json_find_scalar(text, key, &value) || value >= text.size()) {
        return 0;
    }
    char* end = nullptr;
    const char* begin = text.c_str() + value;
    const uint64_t parsed = std::strtoull(begin, &end, 10);
    if (!end || end == begin) {
        return 0;
    }
    *out = parsed;
    return 1;
}

static int moe_json_get_double_local(const std::string& text, const char* key, double* out) {
    size_t value = 0;
    if (!out || !moe_json_find_scalar(text, key, &value) || value >= text.size()) {
        return 0;
    }
    char* end = nullptr;
    const char* begin = text.c_str() + value;
    const double parsed = std::strtod(begin, &end);
    if (!end || end == begin || !std::isfinite(parsed)) {
        return 0;
    }
    *out = parsed;
    return 1;
}

static int moe_json_get_bool_local(const std::string& text, const char* key, int* out) {
    size_t value = 0;
    if (!out || !moe_json_find_scalar(text, key, &value) || value >= text.size()) {
        return 0;
    }
    if (text.compare(value, 4, "true") == 0) {
        *out = 1;
        return 1;
    }
    if (text.compare(value, 5, "false") == 0) {
        *out = 0;
        return 1;
    }
    return 0;
}

static int moe_json_get_string_local(const std::string& text, const char* key, std::string* out) {
    size_t value = 0;
    if (!out || !moe_json_find_scalar(text, key, &value) || value >= text.size() || text[value] != '"') {
        return 0;
    }
    ++value;
    std::string parsed;
    while (value < text.size()) {
        const char c = text[value++];
        if (c == '"') {
            *out = parsed;
            return 1;
        }
        if (c == '\\' && value < text.size()) {
            parsed.push_back(text[value++]);
        } else {
            parsed.push_back(c);
        }
    }
    return 0;
}

static uint32_t moe_u64_to_u32_clamped(uint64_t value, uint32_t fallback) {
    if (value == 0 || value > UINT32_MAX) {
        return fallback;
    }
    return (uint32_t)value;
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

static uint32_t moe_weight_encoding_from_family(const std::string& family) {
    const std::string value = moe_lower_ascii_copy(family);
    if (value.find("iq2") != std::string::npos || value.find("ud-iq2") != std::string::npos) {
        return moe_WEIGHT_ENCODING_GGUF_IQ2;
    }
    if (value.find("iq3") != std::string::npos) {
        return moe_WEIGHT_ENCODING_GGUF_IQ3;
    }
    if (value.find("iq4") != std::string::npos) {
        return moe_WEIGHT_ENCODING_GGUF_IQ4;
    }
    if (value.find("mxfp4") != std::string::npos) {
        return moe_WEIGHT_ENCODING_MXFP4;
    }
    if (value.find("nvfp4") != std::string::npos || value.find("fp4") != std::string::npos) {
        return moe_WEIGHT_ENCODING_NVFP4;
    }
    if (value.find("q8") != std::string::npos) {
        return moe_WEIGHT_ENCODING_LINEAR_Q8;
    }
    if (value.find("q4") != std::string::npos) {
        return moe_WEIGHT_ENCODING_LINEAR_Q4;
    }
    if (value.find("q3") != std::string::npos) {
        return moe_WEIGHT_ENCODING_LINEAR_Q3;
    }
    if (value.find("q2") != std::string::npos) {
        return moe_WEIGHT_ENCODING_LINEAR_Q2;
    }
    return moe_WEIGHT_ENCODING_UNKNOWN;
}

struct moe_offload_gguf_contract_t {
    uint32_t format_version = 0;
    bool backend_neutral = false;
    bool qkv_required = false;
    bool plain_kv_persistent_allowed = true;
    std::string weight_quant_family;
    std::string weight_kernel_family;
    std::string model_id;
    std::string model_name;
    std::string architecture;
    uint32_t weight_bits = 0;
    uint32_t weight_encoding = 0;
    uint32_t weight_block_size = 0;
    uint32_t optimization_catalog_count = 0;
    double reference_moe_compute_ms = 0.0;
    double reference_ram_to_gpu_ms = 0.0;
    double reference_nvme_to_ram_ms = 0.0;
    uint32_t qkv_k_bits = 0;
    uint32_t qkv_v_bits = 0;
    uint32_t qkv_group_size = 0;
    uint32_t qkv_page_size_tokens = 0;
    uint32_t qkv_sink_tokens = 0;
    uint64_t qkv_rotation_seed = 42;
    uint64_t qkv_qjl_seed = 43;
    int qkv_enable_qjl = -1;
    int qkv_enable_rotation = -1;
    uint32_t qkv_outlier_channels = 0;
    uint32_t qkv_outlier_bits = 0;
    bool has_juju_contracts = false;
};

struct moe_offload_gguf_file_t {
    std::string path;
    uint64_t bytes = 0;
    uint64_t tensor_count = 0;
    uint64_t executable_tensor_count = 0;
    uint64_t data_start = 0;
    uint32_t alignment = 32;
    uint32_t file_index = 0;
    uint32_t file_count = 0;
    bool has_offload = false;
    bool has_metadata = false;
    std::string metadata_json;
    moe_offload_gguf_contract_t contract;
};

struct moe_offload_gguf_scan_t {
    std::vector<moe_offload_gguf_file_t> files;
    uint32_t expected_count = 0;
    uint32_t present_count = 0;
    uint32_t missing_count = 0;
    uint32_t first_missing = 0;
    uint64_t tensor_count = 0;
    uint64_t executable_tensor_count = 0;
    uint64_t present_total_bytes = 0;
    bool valid = false;
};

static int moe_engine_apply_juju_strategy(
    moe_pc_engine* engine,
    const moe_offload_gguf_scan_t& scan,
    const moe_offload_gguf_contract_t& contract
);

struct moe_gguf_tensor_dir_entry_t {
    std::string name;
    uint64_t shape[4]{};
    uint32_t dims = 0;
    uint32_t type = 0;
    uint64_t tensor_offset = 0;
    uint64_t file_offset = 0;
    uint64_t bytes = 0;
};

struct moe_gguf_tensor_dir_t {
    std::string path;
    uint64_t file_bytes = 0;
    uint64_t data_start = 0;
    uint32_t file_index = 0;
    uint32_t file_count = 0;
    uint32_t alignment = 32;
    std::vector<moe_gguf_tensor_dir_entry_t> tensors;
};

static uint64_t moe_align_u64(uint64_t value, uint32_t alignment) {
    const uint64_t a = alignment ? (uint64_t)alignment : 32ull;
    const uint64_t r = value % a;
    return r ? value + (a - r) : value;
}

static int moe_gguf_tensor_name_executable(const std::string& name) {
    if (name.empty()) {
        return 0;
    }
    const std::string value = moe_lower_ascii_copy(name);
    if (value == "token_embd.weight" ||
        value == "output.weight" ||
        value == "output_norm.weight" ||
        value.find("blk.") == 0 ||
        value.find(".blk.") != std::string::npos ||
        value.find("model.layers.") != std::string::npos) {
        return 1;
    }
    return 0;
}

static int moe_read_gguf_tensor_infos(
    std::ifstream& input,
    uint64_t tensor_count,
    uint32_t alignment,
    uint64_t file_bytes,
    std::vector<moe_gguf_tensor_dir_entry_t>* tensors,
    uint64_t* data_start,
    uint64_t* executable_count
) {
    if (!tensors || tensor_count > 10000000ull) {
        return 0;
    }
    tensors->clear();
    tensors->reserve((size_t)std::min<uint64_t>(tensor_count, 1000000ull));
    for (uint64_t i = 0; i < tensor_count; ++i) {
        moe_gguf_tensor_dir_entry_t entry{};
        if (!moe_gguf_read_string(input, &entry.name, 1024) ||
            !moe_gguf_read_u32(input, &entry.dims) ||
            entry.dims > 64) {
            return 0;
        }
        for (uint32_t d = 0; d < entry.dims; ++d) {
            uint64_t dim = 0;
            if (!moe_gguf_read_u64(input, &dim)) {
                return 0;
            }
            if (d < 4) {
                entry.shape[d] = dim;
            }
        }
        if (!moe_gguf_read_u32(input, &entry.type) ||
            !moe_gguf_read_u64(input, &entry.tensor_offset)) {
            return 0;
        }
        tensors->push_back(std::move(entry));
    }
    const uint64_t end = (uint64_t)input.tellg();
    const uint64_t base = moe_align_u64(end, alignment);
    if (data_start) {
        *data_start = base;
    }
    std::sort(tensors->begin(), tensors->end(), [](const moe_gguf_tensor_dir_entry_t& a, const moe_gguf_tensor_dir_entry_t& b) {
        return a.tensor_offset < b.tensor_offset;
    });
    uint64_t executable = 0;
    const uint64_t payload_bytes = file_bytes > base ? file_bytes - base : 0;
    for (size_t i = 0; i < tensors->size(); ++i) {
        moe_gguf_tensor_dir_entry_t& entry = (*tensors)[i];
        const uint64_t next = (i + 1u < tensors->size()) ? (*tensors)[i + 1u].tensor_offset : payload_bytes;
        entry.bytes = next > entry.tensor_offset ? next - entry.tensor_offset : 0;
        entry.file_offset = base + entry.tensor_offset;
        if (moe_gguf_tensor_name_executable(entry.name)) {
            ++executable;
        }
    }
    if (executable_count) {
        *executable_count = executable;
    }
    return 1;
}

static int moe_read_gguf_tensor_directory(
    const std::string& path,
    moe_gguf_tensor_dir_t* out
) {
    if (!out || path.empty()) {
        return 0;
    }
    *out = moe_gguf_tensor_dir_t{};
    out->path = path;
    file_size_bytes(path, &out->file_bytes);
    std::ifstream input;
    open_ifstream_path(input, path, std::ios::in | std::ios::binary);
    if (!input) {
        return 0;
    }
    char magic[4]{};
    uint32_t version = 0;
    uint64_t tensor_count = 0;
    uint64_t kv_count = 0;
    if (!moe_gguf_read_exact(input, magic, sizeof(magic)) ||
        std::memcmp(magic, "GGUF", 4) != 0 ||
        !moe_gguf_read_u32(input, &version) ||
        !moe_gguf_read_u64(input, &tensor_count) ||
        !moe_gguf_read_u64(input, &kv_count) ||
        version == 0 ||
        kv_count > 1000000ull ||
        tensor_count > 10000000ull) {
        return 0;
    }
    for (uint64_t i = 0; i < kv_count; ++i) {
        std::string key;
        uint32_t type = 0;
        if (!moe_gguf_read_string(input, &key, 65536) ||
            !moe_gguf_read_u32(input, &type)) {
            return 0;
        }
        uint64_t u64_value = 0;
        int consumed = 0;
        if (key == "general.alignment" && moe_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            if (u64_value >= 8 && u64_value <= 65536) {
                out->alignment = (uint32_t)u64_value;
            }
        } else if (key == "offload.file_index" && moe_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->file_index = moe_u64_to_u32_clamped(u64_value, 0);
        } else if (key == "offload.file_count" && moe_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->file_count = moe_u64_to_u32_clamped(u64_value, 0);
        }
        if (!consumed && !moe_gguf_skip_value(input, type)) {
            return 0;
        }
    }
    uint64_t executable = 0;
    return moe_read_gguf_tensor_infos(
        input,
        tensor_count,
        out->alignment,
        out->file_bytes,
        &out->tensors,
        &out->data_start,
        &executable);
}

static int moe_scan_gguf_tensor_infos(
    std::ifstream& input,
    uint64_t tensor_count,
    moe_offload_gguf_file_t* out
) {
    if (!out) {
        return 0;
    }
    std::vector<moe_gguf_tensor_dir_entry_t> tensors;
    uint64_t data_start = 0;
    uint64_t executable = 0;
    if (!moe_read_gguf_tensor_infos(
            input,
            tensor_count,
            out->alignment,
            out->bytes,
            &tensors,
            &data_start,
            &executable)) {
        return 0;
    }
    out->tensor_count = tensor_count;
    out->data_start = data_start;
    out->executable_tensor_count = executable;
    return 1;
}

static void moe_parse_offload_metadata_json(moe_offload_gguf_file_t* meta) {
    if (!meta || meta->metadata_json.empty()) {
        return;
    }
    meta->has_metadata = true;
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
    if (meta->contract.weight_block_size == 0 &&
        moe_json_get_u64_local(meta->metadata_json, "source_weight_block_size", &value)) {
        meta->contract.weight_block_size = moe_u64_to_u32_clamped(value, 0);
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
        meta->metadata_json.find("\"qkv_policy_contract\"") != std::string::npos) {
        meta->contract.has_juju_contracts = true;
    }
    if (meta->contract.weight_encoding == 0 && !meta->contract.weight_quant_family.empty()) {
        meta->contract.weight_encoding = moe_weight_encoding_from_family(meta->contract.weight_quant_family);
    }
}

static int moe_read_offload_gguf_file(
    const std::string& path,
    moe_offload_gguf_file_t* out
) {
    if (!out) {
        return 0;
    }
    *out = moe_offload_gguf_file_t{};
    out->path = path;
    file_size_bytes(path, &out->bytes);
    std::ifstream input;
    open_ifstream_path(input, path, std::ios::in | std::ios::binary);
    if (!input) {
        return 0;
    }
    char magic[4]{};
    uint32_t version = 0;
    uint64_t tensor_count = 0;
    uint64_t kv_count = 0;
    if (!moe_gguf_read_exact(input, magic, sizeof(magic)) ||
        std::memcmp(magic, "GGUF", 4) != 0 ||
        !moe_gguf_read_u32(input, &version) ||
        !moe_gguf_read_u64(input, &tensor_count) ||
        !moe_gguf_read_u64(input, &kv_count) ||
        version == 0 ||
        kv_count > 1000000ull) {
        return 0;
    }
    out->tensor_count = tensor_count;
    for (uint64_t i = 0; i < kv_count; ++i) {
        std::string key;
        uint32_t type = 0;
        if (!moe_gguf_read_string(input, &key, 65536) ||
            !moe_gguf_read_u32(input, &type)) {
            return 0;
        }
        const bool is_offload = key.rfind("offload.", 0) == 0;
        if (is_offload) {
            out->has_offload = true;
        }
        uint64_t u64_value = 0;
        double f64_value = 0.0;
        std::string string_value;
        int consumed = 0;
        if (key == "general.alignment" && moe_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            if (u64_value >= 8 && u64_value <= 65536) {
                out->alignment = moe_u64_to_u32_clamped(u64_value, 32);
            }
        } else if ((key == "general.name" || key == "general.basename" || key == "general.file_type") &&
            type == moe_GGUF_TYPE_STRING) {
            consumed = 1;
            if (moe_gguf_read_string(input, &string_value, 65536) &&
                out->contract.model_name.empty()) {
                out->contract.model_name = string_value;
            }
        } else if (key == "general.architecture" && type == moe_GGUF_TYPE_STRING) {
            consumed = 1;
            if (moe_gguf_read_string(input, &string_value, 65536)) {
                out->contract.architecture = string_value;
            }
        } else if ((key == "offload.model_id" ||
                    key == "offload.repo_id" ||
                    key == "offload.source_model_id" ||
                    key == "offload.source_repo_id" ||
                    key == "offload.target_repo_id") &&
                   type == moe_GGUF_TYPE_STRING) {
            consumed = 1;
            if (moe_gguf_read_string(input, &string_value, 65536)) {
                out->contract.model_id = moe_repo_tail_or_value(string_value);
            }
        } else if (key == "offload.model_name" && type == moe_GGUF_TYPE_STRING) {
            consumed = 1;
            moe_gguf_read_string(input, &out->contract.model_name, 65536);
        } else if (key == "offload.metadata_v2" && type == moe_GGUF_TYPE_STRING) {
            consumed = 1;
            if (moe_gguf_read_string(input, &out->metadata_json, 128ull * 1024ull * 1024ull)) {
                out->has_metadata = !out->metadata_json.empty();
            }
        } else if (key == "offload.format_version" && moe_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->contract.format_version = moe_u64_to_u32_clamped(u64_value, 0);
        } else if (key == "offload.backend_neutral" && moe_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->contract.backend_neutral = u64_value != 0;
        } else if (key == "offload.file_index" && moe_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->file_index = moe_u64_to_u32_clamped(u64_value, 0);
        } else if (key == "offload.file_count" && moe_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->file_count = moe_u64_to_u32_clamped(u64_value, 0);
        } else if (key == "offload.weight_bits" && moe_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->contract.weight_bits = moe_u64_to_u32_clamped(u64_value, 0);
        } else if (key == "offload.weight_encoding" && moe_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->contract.weight_encoding = moe_u64_to_u32_clamped(u64_value, 0);
        } else if (key == "offload.weight_block_size" && moe_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->contract.weight_block_size = moe_u64_to_u32_clamped(u64_value, 0);
        } else if (key == "offload.optimization_catalog_count" && moe_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->contract.optimization_catalog_count = moe_u64_to_u32_clamped(u64_value, 0);
        } else if ((key == "offload.moe_compute_ms" ||
                    key == "offload.reference_moe_compute_ms") &&
                   moe_gguf_read_float_value(input, type, &f64_value)) {
            consumed = 1;
            out->contract.reference_moe_compute_ms = f64_value;
        } else if ((key == "offload.ram_to_gpu_ms" ||
                    key == "offload.reference_ram_to_gpu_ms") &&
                   moe_gguf_read_float_value(input, type, &f64_value)) {
            consumed = 1;
            out->contract.reference_ram_to_gpu_ms = f64_value;
        } else if ((key == "offload.nvme_to_ram_ms" ||
                    key == "offload.reference_nvme_to_ram_ms") &&
                   moe_gguf_read_float_value(input, type, &f64_value)) {
            consumed = 1;
            out->contract.reference_nvme_to_ram_ms = f64_value;
        } else if (key == "offload.weight_quant_family" && type == moe_GGUF_TYPE_STRING) {
            consumed = 1;
            moe_gguf_read_string(input, &out->contract.weight_quant_family, 65536);
        } else if (key == "offload.weight_kernel_family" && type == moe_GGUF_TYPE_STRING) {
            consumed = 1;
            moe_gguf_read_string(input, &out->contract.weight_kernel_family, 65536);
        } else if (key == "offload.qkv_k_bits" && moe_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->contract.qkv_k_bits = moe_u64_to_u32_clamped(u64_value, 0);
        } else if (key == "offload.qkv_v_bits" && moe_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->contract.qkv_v_bits = moe_u64_to_u32_clamped(u64_value, 0);
        } else if (key == "offload.qkv_group_size" && moe_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->contract.qkv_group_size = moe_u64_to_u32_clamped(u64_value, 0);
        } else if (key == "offload.qkv_page_size_tokens" && moe_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->contract.qkv_page_size_tokens = moe_u64_to_u32_clamped(u64_value, 0);
        } else if (key == "offload.qkv_sink_tokens" && moe_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->contract.qkv_sink_tokens = moe_u64_to_u32_clamped(u64_value, 0);
        } else if (key == "offload.qkv_rotation_seed" && moe_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->contract.qkv_rotation_seed = u64_value;
        } else if (key == "offload.qkv_qjl_seed" && moe_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->contract.qkv_qjl_seed = u64_value;
        } else if (key == "offload.qkv_enable_qjl" && moe_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->contract.qkv_enable_qjl = u64_value ? 1 : 0;
        } else if (key == "offload.qkv_enable_rotation" && moe_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->contract.qkv_enable_rotation = u64_value ? 1 : 0;
        } else if ((key == "offload.final_model_structure_contract" ||
                    key == "offload.pipeline_budget_contract" ||
                    key == "offload.execution_path_contract" ||
                    key == "offload.expert_segmentation_contract" ||
                    key == "offload.chunk_io_contract" ||
                    key == "offload.universal_tier_contract" ||
                    key == "offload.qkv_policy_contract") &&
                   type == moe_GGUF_TYPE_STRING) {
            consumed = 1;
            std::string ignored;
            moe_gguf_read_string(input, &ignored, 16ull * 1024ull * 1024ull);
            out->contract.has_juju_contracts = true;
        }
        if (!consumed && !moe_gguf_skip_value(input, type)) {
            return 0;
        }
    }
    moe_parse_offload_metadata_json(out);
    if (out->contract.qkv_k_bits || out->contract.qkv_v_bits) {
        out->contract.qkv_required = true;
    }
    if (out->contract.weight_encoding == 0 && !out->contract.weight_quant_family.empty()) {
        out->contract.weight_encoding = moe_weight_encoding_from_family(out->contract.weight_quant_family);
    }
    if (out->contract.qkv_required) {
        out->contract.plain_kv_persistent_allowed = false;
    }
    if (!moe_scan_gguf_tensor_infos(input, tensor_count, out)) {
        return 0;
    }
    return out->has_offload ? 1 : 0;
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

static std::string moe_model_id_from_offload_scan(
    const std::string& root,
    const moe_offload_gguf_scan_t& scan
) {
    for (const moe_offload_gguf_file_t& file : scan.files) {
        const std::string slug = moe_slugify_model_id(moe_repo_tail_or_value(file.contract.model_id));
        if (!slug.empty()) {
            return slug;
        }
    }
    for (const moe_offload_gguf_file_t& file : scan.files) {
        const std::string slug = moe_slugify_model_id(file.contract.model_name);
        if (!slug.empty()) {
            return slug;
        }
    }
    for (const moe_offload_gguf_file_t& file : scan.files) {
        const std::string arch = moe_slugify_model_id(file.contract.architecture);
        if (!arch.empty()) {
            return arch + "-gguf-offload";
        }
    }
    const std::string root_slug = moe_slugify_model_id(moe_path_basename_local(root));
    return root_slug.empty() ? "storagellm-offload" : root_slug;
}

static void moe_fill_check_from_offload_gguf_scan(
    const moe_offload_gguf_scan_t& scan,
    moe_model_root_check_t* out_check
) {
    if (!out_check) {
        return;
    }
    std::memset(out_check, 0, sizeof(*out_check));
    out_check->expected_part_count = scan.expected_count;
    out_check->present_part_count = scan.present_count;
    out_check->missing_part_count = scan.missing_count;
    out_check->expected_total_bytes = scan.present_total_bytes;
    out_check->present_total_bytes = scan.present_total_bytes;
    if (scan.missing_count) {
        out_check->first_missing_part = scan.first_missing;
        std::ostringstream missing;
        missing << "offload GGUF shard index " << scan.first_missing;
        copy_check_path(out_check->first_missing_path, sizeof(out_check->first_missing_path), missing.str());
    }
    out_check->valid = scan.valid ? 1 : 0;
}

static qkv_config_t moe_qkv_config_from_offload_contract(const moe_offload_gguf_contract_t& contract) {
    qkv_config_t cfg = qkv_config_default(128);
    if (contract.qkv_k_bits > 0) {
        cfg.k_bits = (int)std::min<uint32_t>(contract.qkv_k_bits, 4);
    }
    if (contract.qkv_v_bits > 0) {
        cfg.v_bits = (int)std::min<uint32_t>(contract.qkv_v_bits, 4);
        cfg.normal_bits = cfg.v_bits;
    }
    if (contract.qkv_group_size > 0) {
        cfg.group_size = contract.qkv_group_size;
    }
    if (contract.qkv_page_size_tokens > 0) {
        cfg.page_size_tokens = contract.qkv_page_size_tokens;
    }
    if (contract.qkv_sink_tokens > 0) {
        cfg.sink_tokens = contract.qkv_sink_tokens;
    }
    cfg.rotation_seed = contract.qkv_rotation_seed;
    cfg.qjl_seed = contract.qkv_qjl_seed;
    if (contract.qkv_enable_qjl >= 0) {
        cfg.enable_qjl = contract.qkv_enable_qjl != 0;
    }
    if (contract.qkv_enable_rotation >= 0) {
        cfg.enable_rotation = contract.qkv_enable_rotation != 0;
    }
    if (contract.qkv_outlier_channels > 0) {
        cfg.outlier_channels = (int)contract.qkv_outlier_channels;
    }
    if (contract.qkv_outlier_bits > 0) {
        cfg.outlier_bits = (int)std::min<uint32_t>(contract.qkv_outlier_bits, 4);
    }
    if (cfg.k_bits <= 1 || cfg.v_bits <= 1) {
        cfg.enable_qjl = false;
    }
    cfg.plain_kv_persistent_storage = contract.plain_kv_persistent_allowed;
    return cfg;
}

static void moe_engine_reset_qkv_cache(moe_pc_engine* engine) {
    if (!engine) {
        return;
    }
    std::lock_guard<std::mutex> lock(engine->qkv_mutex);
    for (size_t i = 0; i < engine->qkv_head_state_slots.size(); ++i) {
        if (i < engine->qkv_head_state_valid.size() && engine->qkv_head_state_valid[i]) {
            qkv_free(&engine->qkv_head_state_slots[i]);
        }
    }
    engine->qkv_head_state_slots.clear();
    engine->qkv_head_state_valid.clear();
    engine->qkv_head_state_locks.clear();
    engine->qkv_cfg_initialized.store(false, std::memory_order_release);
    engine->qkv_last_ctx = 0;
}

static void moe_engine_clear_offload_gguf_contract(moe_pc_engine* engine) {
    if (!engine) {
        return;
    }
    engine->offload_gguf_valid.store(0, std::memory_order_relaxed);
    engine->offload_gguf_file_count.store(0, std::memory_order_relaxed);
    engine->offload_gguf_tensor_count.store(0, std::memory_order_relaxed);
    engine->offload_gguf_executable_tensor_count.store(0, std::memory_order_relaxed);
    engine->qkv_forced_by_format.store(0, std::memory_order_relaxed);
    engine->offload_qkv_cfg_available.store(false, std::memory_order_release);
    engine->offload_qkv_cfg = qkv_config_default(128);
    engine->offload_weight_quant_bits.store(0, std::memory_order_relaxed);
    engine->offload_weight_encoding.store(0, std::memory_order_relaxed);
    engine->offload_weight_block_size.store(0, std::memory_order_relaxed);
    engine->offload_optimization_catalog_count.store(0, std::memory_order_relaxed);
    engine->juju_strategy_valid.store(0, std::memory_order_relaxed);
    engine->juju_execution_path = moe_JUJU_EXEC_STORAGE_STREAMING;
    std::memset(&engine->juju_pipeline_config, 0, sizeof(engine->juju_pipeline_config));
    engine->offload_gguf_first_file.clear();
    engine->offload_gguf_metadata_json.clear();
    engine->offload_weight_quant_family.clear();
    engine->offload_weight_kernel_family.clear();
}

static int moe_engine_load_offload_gguf_contract(
    moe_pc_engine* engine,
    const std::string& root,
    moe_model_root_check_t* out_check
) {
    moe_offload_gguf_scan_t scan{};
    if (!moe_scan_offload_gguf_root(root, &scan)) {
        return 0;
    }
    moe_fill_check_from_offload_gguf_scan(scan, out_check);
    moe_engine_clear_offload_gguf_contract(engine);
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
        "[storagellm] offload GGUF contract loaded: files=%u first=%s qkv=%s weight=%s/%ub catalog=%u\n",
        scan.expected_count,
        primary.path.c_str(),
        qkv_required ? "required" : "available",
        primary.contract.weight_quant_family.empty() ? "unknown" : primary.contract.weight_quant_family.c_str(),
        primary.contract.weight_bits,
        primary.contract.optimization_catalog_count);
    std::fflush(stderr);
    return 1;
}

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
    if (!root.empty() && moe_scan_offload_gguf_root(root, &gguf_scan)) {
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
    if (!moe_scan_offload_gguf_root(root, &gguf_scan)) {
        return 0;
    }
    const std::string id = moe_model_id_from_offload_scan(root, gguf_scan);
    if (id.empty()) {
        return 0;
    }
    moe_copy_string_to_buffer(id, out_model_id, out_size);
    return out_model_id[0] != '\0';
}
