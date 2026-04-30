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
static std::string glm5_wide_to_utf8(const wchar_t* value) {
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

static int glm5_path_is_directory_local(const std::string& path) {
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

static bool glm5_ascii_ieq(char a, char b) {
    if (a >= 'A' && a <= 'Z') {
        a = (char)(a - 'A' + 'a');
    }
    if (b >= 'A' && b <= 'Z') {
        b = (char)(b - 'A' + 'a');
    }
    return a == b;
}

static bool glm5_ends_with_ci(const std::string& value, const char* suffix) {
    if (!suffix) {
        return false;
    }
    const size_t n = value.size();
    const size_t m = std::strlen(suffix);
    if (m > n) {
        return false;
    }
    for (size_t i = 0; i < m; ++i) {
        if (!glm5_ascii_ieq(value[n - m + i], suffix[i])) {
            return false;
        }
    }
    return true;
}

static bool glm5_skip_scan_dir_name(const std::string& name) {
    return name == "." || name == ".." || name == ".git" || name == "build" ||
        name == "colab" || name == ".glm5_model_lib_cache";
}

static void glm5_collect_gguf_files(
    const std::string& path,
    int depth_left,
    std::vector<std::string>* out
) {
    if (!out || path.empty() || out->size() >= 512) {
        return;
    }
    uint64_t file_bytes = 0;
    if (file_size_bytes(path, &file_bytes)) {
        if (glm5_ends_with_ci(path, ".gguf")) {
            out->push_back(path);
        }
        return;
    }
    if (depth_left <= 0 || !glm5_path_is_directory_local(path)) {
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
        const std::string name = glm5_wide_to_utf8(data.cFileName);
        if (glm5_skip_scan_dir_name(name)) {
            continue;
        }
        const std::string child = storagellm::path_join(path, name);
        if (data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            glm5_collect_gguf_files(child, depth_left - 1, out);
        } else if (glm5_ends_with_ci(name, ".gguf")) {
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
        if (glm5_skip_scan_dir_name(name)) {
            continue;
        }
        const std::string child = storagellm::path_join(path, name);
        if (glm5_path_is_directory_local(child)) {
            glm5_collect_gguf_files(child, depth_left - 1, out);
        } else if (glm5_ends_with_ci(name, ".gguf")) {
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
    GLM5_GGUF_TYPE_UINT8 = 0,
    GLM5_GGUF_TYPE_INT8 = 1,
    GLM5_GGUF_TYPE_UINT16 = 2,
    GLM5_GGUF_TYPE_INT16 = 3,
    GLM5_GGUF_TYPE_UINT32 = 4,
    GLM5_GGUF_TYPE_INT32 = 5,
    GLM5_GGUF_TYPE_FLOAT32 = 6,
    GLM5_GGUF_TYPE_BOOL = 7,
    GLM5_GGUF_TYPE_STRING = 8,
    GLM5_GGUF_TYPE_ARRAY = 9,
    GLM5_GGUF_TYPE_UINT64 = 10,
    GLM5_GGUF_TYPE_INT64 = 11,
    GLM5_GGUF_TYPE_FLOAT64 = 12
};

static int glm5_gguf_read_exact(std::ifstream& input, void* data, size_t bytes) {
    if (bytes == 0) {
        return 1;
    }
    input.read(static_cast<char*>(data), static_cast<std::streamsize>(bytes));
    return (size_t)input.gcount() == bytes;
}

static int glm5_gguf_read_u8(std::ifstream& input, uint8_t* out) {
    return out && glm5_gguf_read_exact(input, out, sizeof(*out));
}

static int glm5_gguf_read_u16(std::ifstream& input, uint16_t* out) {
    return out && glm5_gguf_read_exact(input, out, sizeof(*out));
}

static int glm5_gguf_read_u32(std::ifstream& input, uint32_t* out) {
    return out && glm5_gguf_read_exact(input, out, sizeof(*out));
}

static int glm5_gguf_read_u64(std::ifstream& input, uint64_t* out) {
    return out && glm5_gguf_read_exact(input, out, sizeof(*out));
}

static int glm5_gguf_skip_bytes(std::ifstream& input, uint64_t bytes) {
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

static int glm5_gguf_fixed_type_size(uint32_t type, uint64_t* out) {
    if (!out) {
        return 0;
    }
    switch (type) {
        case GLM5_GGUF_TYPE_UINT8:
        case GLM5_GGUF_TYPE_INT8:
        case GLM5_GGUF_TYPE_BOOL:
            *out = 1;
            return 1;
        case GLM5_GGUF_TYPE_UINT16:
        case GLM5_GGUF_TYPE_INT16:
            *out = 2;
            return 1;
        case GLM5_GGUF_TYPE_UINT32:
        case GLM5_GGUF_TYPE_INT32:
        case GLM5_GGUF_TYPE_FLOAT32:
            *out = 4;
            return 1;
        case GLM5_GGUF_TYPE_UINT64:
        case GLM5_GGUF_TYPE_INT64:
        case GLM5_GGUF_TYPE_FLOAT64:
            *out = 8;
            return 1;
        default:
            return 0;
    }
}

static int glm5_gguf_skip_string(std::ifstream& input) {
    uint64_t len = 0;
    return glm5_gguf_read_u64(input, &len) && glm5_gguf_skip_bytes(input, len);
}

static int glm5_gguf_read_string(std::ifstream& input, std::string* out, uint64_t max_bytes) {
    uint64_t len = 0;
    if (!out || !glm5_gguf_read_u64(input, &len)) {
        return 0;
    }
    if (len > max_bytes || len > (uint64_t)std::numeric_limits<size_t>::max()) {
        glm5_gguf_skip_bytes(input, len);
        return 0;
    }
    out->assign((size_t)len, '\0');
    return len == 0 || glm5_gguf_read_exact(input, &(*out)[0], (size_t)len);
}

static int glm5_gguf_skip_value(std::ifstream& input, uint32_t type) {
    if (type == GLM5_GGUF_TYPE_STRING) {
        return glm5_gguf_skip_string(input);
    }
    if (type == GLM5_GGUF_TYPE_ARRAY) {
        uint32_t item_type = 0;
        uint64_t item_count = 0;
        if (!glm5_gguf_read_u32(input, &item_type) || !glm5_gguf_read_u64(input, &item_count)) {
            return 0;
        }
        if (item_type == GLM5_GGUF_TYPE_STRING) {
            for (uint64_t i = 0; i < item_count; ++i) {
                if (!glm5_gguf_skip_string(input)) {
                    return 0;
                }
            }
            return 1;
        }
        uint64_t item_size = 0;
        if (!glm5_gguf_fixed_type_size(item_type, &item_size) ||
            item_count > UINT64_MAX / item_size) {
            return 0;
        }
        return glm5_gguf_skip_bytes(input, item_count * item_size);
    }
    uint64_t size = 0;
    return glm5_gguf_fixed_type_size(type, &size) && glm5_gguf_skip_bytes(input, size);
}

static int glm5_gguf_read_uint_value(std::ifstream& input, uint32_t type, uint64_t* out) {
    if (!out) {
        return 0;
    }
    switch (type) {
        case GLM5_GGUF_TYPE_UINT8: {
            uint8_t v = 0;
            if (!glm5_gguf_read_u8(input, &v)) return 0;
            *out = v;
            return 1;
        }
        case GLM5_GGUF_TYPE_BOOL: {
            uint8_t v = 0;
            if (!glm5_gguf_read_u8(input, &v)) return 0;
            *out = v ? 1u : 0u;
            return 1;
        }
        case GLM5_GGUF_TYPE_UINT16: {
            uint16_t v = 0;
            if (!glm5_gguf_read_u16(input, &v)) return 0;
            *out = v;
            return 1;
        }
        case GLM5_GGUF_TYPE_UINT32: {
            uint32_t v = 0;
            if (!glm5_gguf_read_u32(input, &v)) return 0;
            *out = v;
            return 1;
        }
        case GLM5_GGUF_TYPE_UINT64:
            return glm5_gguf_read_u64(input, out);
        default:
            return 0;
    }
}

static int glm5_json_find_scalar(const std::string& text, const char* key, size_t* value_pos) {
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

static int glm5_json_get_u64_local(const std::string& text, const char* key, uint64_t* out) {
    size_t value = 0;
    if (!out || !glm5_json_find_scalar(text, key, &value) || value >= text.size()) {
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

static int glm5_json_get_bool_local(const std::string& text, const char* key, int* out) {
    size_t value = 0;
    if (!out || !glm5_json_find_scalar(text, key, &value) || value >= text.size()) {
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

static int glm5_json_get_string_local(const std::string& text, const char* key, std::string* out) {
    size_t value = 0;
    if (!out || !glm5_json_find_scalar(text, key, &value) || value >= text.size() || text[value] != '"') {
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

static uint32_t glm5_u64_to_u32_clamped(uint64_t value, uint32_t fallback) {
    if (value == 0 || value > UINT32_MAX) {
        return fallback;
    }
    return (uint32_t)value;
}

static std::string glm5_lower_ascii_copy(std::string value) {
    for (char& c : value) {
        if (c >= 'A' && c <= 'Z') {
            c = (char)(c - 'A' + 'a');
        }
    }
    return value;
}

static uint32_t glm5_weight_encoding_from_family(const std::string& family) {
    const std::string value = glm5_lower_ascii_copy(family);
    if (value.find("iq2") != std::string::npos || value.find("ud-iq2") != std::string::npos) {
        return GLM5_WEIGHT_ENCODING_GGUF_IQ2;
    }
    if (value.find("iq3") != std::string::npos) {
        return GLM5_WEIGHT_ENCODING_GGUF_IQ3;
    }
    if (value.find("iq4") != std::string::npos) {
        return GLM5_WEIGHT_ENCODING_GGUF_IQ4;
    }
    if (value.find("mxfp4") != std::string::npos) {
        return GLM5_WEIGHT_ENCODING_MXFP4;
    }
    if (value.find("nvfp4") != std::string::npos || value.find("fp4") != std::string::npos) {
        return GLM5_WEIGHT_ENCODING_NVFP4;
    }
    if (value.find("q8") != std::string::npos) {
        return GLM5_WEIGHT_ENCODING_LINEAR_Q8;
    }
    if (value.find("q4") != std::string::npos) {
        return GLM5_WEIGHT_ENCODING_LINEAR_Q4;
    }
    if (value.find("q3") != std::string::npos) {
        return GLM5_WEIGHT_ENCODING_LINEAR_Q3;
    }
    if (value.find("q2") != std::string::npos) {
        return GLM5_WEIGHT_ENCODING_LINEAR_Q2;
    }
    return GLM5_WEIGHT_ENCODING_UNKNOWN;
}

struct glm5_offload_gguf_contract_t {
    uint32_t format_version = 0;
    bool backend_neutral = false;
    bool qkv_required = false;
    bool plain_kv_persistent_allowed = true;
    std::string weight_quant_family;
    std::string weight_kernel_family;
    uint32_t weight_bits = 0;
    uint32_t weight_encoding = 0;
    uint32_t weight_block_size = 0;
    uint32_t qkv_k_bits = 0;
    uint32_t qkv_v_bits = 0;
    uint32_t qkv_group_size = 0;
    uint32_t qkv_page_size_tokens = 0;
    uint32_t qkv_sink_tokens = 0;
    uint32_t qkv_outlier_channels = 0;
    uint32_t qkv_outlier_bits = 0;
};

struct glm5_offload_gguf_file_t {
    std::string path;
    uint64_t bytes = 0;
    uint64_t tensor_count = 0;
    uint32_t file_index = 0;
    uint32_t file_count = 0;
    bool has_offload = false;
    bool has_metadata = false;
    std::string metadata_json;
    glm5_offload_gguf_contract_t contract;
};

struct glm5_offload_gguf_scan_t {
    std::vector<glm5_offload_gguf_file_t> files;
    uint32_t expected_count = 0;
    uint32_t present_count = 0;
    uint32_t missing_count = 0;
    uint32_t first_missing = 0;
    uint64_t tensor_count = 0;
    uint64_t present_total_bytes = 0;
    bool valid = false;
};

static void glm5_parse_offload_metadata_json(glm5_offload_gguf_file_t* meta) {
    if (!meta || meta->metadata_json.empty()) {
        return;
    }
    meta->has_metadata = true;
    uint64_t value = 0;
    int bool_value = 0;
    std::string string_value;
    if (meta->contract.format_version == 0 &&
        glm5_json_get_u64_local(meta->metadata_json, "format_version", &value)) {
        meta->contract.format_version = glm5_u64_to_u32_clamped(value, 0);
    }
    if (meta->file_count == 0 &&
        (glm5_json_get_u64_local(meta->metadata_json, "file_count", &value) ||
         glm5_json_get_u64_local(meta->metadata_json, "part_count", &value))) {
        meta->file_count = glm5_u64_to_u32_clamped(value, 0);
    }
    if (meta->file_index == 0 &&
        (glm5_json_get_u64_local(meta->metadata_json, "file_index", &value) ||
         glm5_json_get_u64_local(meta->metadata_json, "part", &value))) {
        meta->file_index = glm5_u64_to_u32_clamped(value, 0);
    }
    if (meta->contract.qkv_k_bits == 0 &&
        glm5_json_get_u64_local(meta->metadata_json, "k_bits", &value)) {
        meta->contract.qkv_k_bits = glm5_u64_to_u32_clamped(value, 0);
    }
    if (meta->contract.qkv_v_bits == 0 &&
        glm5_json_get_u64_local(meta->metadata_json, "v_bits", &value)) {
        meta->contract.qkv_v_bits = glm5_u64_to_u32_clamped(value, 0);
    }
    if (meta->contract.qkv_group_size == 0 &&
        glm5_json_get_u64_local(meta->metadata_json, "group_size", &value)) {
        meta->contract.qkv_group_size = glm5_u64_to_u32_clamped(value, 0);
    }
    if (meta->contract.qkv_page_size_tokens == 0 &&
        glm5_json_get_u64_local(meta->metadata_json, "page_size_tokens", &value)) {
        meta->contract.qkv_page_size_tokens = glm5_u64_to_u32_clamped(value, 0);
    }
    if (meta->contract.qkv_sink_tokens == 0 &&
        glm5_json_get_u64_local(meta->metadata_json, "sink_tokens", &value)) {
        meta->contract.qkv_sink_tokens = glm5_u64_to_u32_clamped(value, 0);
    }
    if (glm5_json_get_bool_local(meta->metadata_json, "qkv_packed_cache_required", &bool_value)) {
        meta->contract.qkv_required = bool_value != 0;
    }
    if (glm5_json_get_bool_local(meta->metadata_json, "persistent_plain_kv_cache_allowed", &bool_value)) {
        meta->contract.plain_kv_persistent_allowed = bool_value != 0;
    }
    if (meta->contract.weight_quant_family.empty() &&
        glm5_json_get_string_local(meta->metadata_json, "source_weight_quant_family", &string_value)) {
        meta->contract.weight_quant_family = string_value;
    }
    if (meta->contract.weight_kernel_family.empty() &&
        glm5_json_get_string_local(meta->metadata_json, "source_weight_kernel_family", &string_value)) {
        meta->contract.weight_kernel_family = string_value;
    }
    if (meta->contract.weight_bits == 0 &&
        glm5_json_get_u64_local(meta->metadata_json, "source_weight_bits", &value)) {
        meta->contract.weight_bits = glm5_u64_to_u32_clamped(value, 0);
    }
    if (meta->contract.weight_block_size == 0 &&
        glm5_json_get_u64_local(meta->metadata_json, "source_weight_block_size", &value)) {
        meta->contract.weight_block_size = glm5_u64_to_u32_clamped(value, 0);
    }
    if (meta->contract.weight_encoding == 0 && !meta->contract.weight_quant_family.empty()) {
        meta->contract.weight_encoding = glm5_weight_encoding_from_family(meta->contract.weight_quant_family);
    }
}

static int glm5_read_offload_gguf_file(
    const std::string& path,
    glm5_offload_gguf_file_t* out
) {
    if (!out) {
        return 0;
    }
    *out = glm5_offload_gguf_file_t{};
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
    if (!glm5_gguf_read_exact(input, magic, sizeof(magic)) ||
        std::memcmp(magic, "GGUF", 4) != 0 ||
        !glm5_gguf_read_u32(input, &version) ||
        !glm5_gguf_read_u64(input, &tensor_count) ||
        !glm5_gguf_read_u64(input, &kv_count) ||
        version == 0 ||
        kv_count > 1000000ull) {
        return 0;
    }
    out->tensor_count = tensor_count;
    for (uint64_t i = 0; i < kv_count; ++i) {
        std::string key;
        uint32_t type = 0;
        if (!glm5_gguf_read_string(input, &key, 65536) ||
            !glm5_gguf_read_u32(input, &type)) {
            return 0;
        }
        const bool is_offload = key.rfind("offload.", 0) == 0;
        if (is_offload) {
            out->has_offload = true;
        }
        uint64_t u64_value = 0;
        std::string string_value;
        int consumed = 0;
        if (key == "offload.metadata_v2" && type == GLM5_GGUF_TYPE_STRING) {
            consumed = 1;
            if (glm5_gguf_read_string(input, &out->metadata_json, 128ull * 1024ull * 1024ull)) {
                out->has_metadata = !out->metadata_json.empty();
            }
        } else if (key == "offload.format_version" && glm5_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->contract.format_version = glm5_u64_to_u32_clamped(u64_value, 0);
        } else if (key == "offload.backend_neutral" && glm5_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->contract.backend_neutral = u64_value != 0;
        } else if (key == "offload.file_index" && glm5_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->file_index = glm5_u64_to_u32_clamped(u64_value, 0);
        } else if (key == "offload.file_count" && glm5_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->file_count = glm5_u64_to_u32_clamped(u64_value, 0);
        } else if (key == "offload.weight_bits" && glm5_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->contract.weight_bits = glm5_u64_to_u32_clamped(u64_value, 0);
        } else if (key == "offload.weight_encoding" && glm5_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->contract.weight_encoding = glm5_u64_to_u32_clamped(u64_value, 0);
        } else if (key == "offload.weight_block_size" && glm5_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->contract.weight_block_size = glm5_u64_to_u32_clamped(u64_value, 0);
        } else if (key == "offload.weight_quant_family" && type == GLM5_GGUF_TYPE_STRING) {
            consumed = 1;
            glm5_gguf_read_string(input, &out->contract.weight_quant_family, 65536);
        } else if (key == "offload.weight_kernel_family" && type == GLM5_GGUF_TYPE_STRING) {
            consumed = 1;
            glm5_gguf_read_string(input, &out->contract.weight_kernel_family, 65536);
        } else if (key == "offload.qkv_k_bits" && glm5_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->contract.qkv_k_bits = glm5_u64_to_u32_clamped(u64_value, 0);
        } else if (key == "offload.qkv_v_bits" && glm5_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->contract.qkv_v_bits = glm5_u64_to_u32_clamped(u64_value, 0);
        } else if (key == "offload.qkv_group_size" && glm5_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->contract.qkv_group_size = glm5_u64_to_u32_clamped(u64_value, 0);
        } else if (key == "offload.qkv_page_size_tokens" && glm5_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->contract.qkv_page_size_tokens = glm5_u64_to_u32_clamped(u64_value, 0);
        } else if (key == "offload.qkv_sink_tokens" && glm5_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->contract.qkv_sink_tokens = glm5_u64_to_u32_clamped(u64_value, 0);
        }
        if (!consumed && !glm5_gguf_skip_value(input, type)) {
            return 0;
        }
    }
    glm5_parse_offload_metadata_json(out);
    if (out->contract.qkv_k_bits || out->contract.qkv_v_bits) {
        out->contract.qkv_required = true;
    }
    if (out->contract.weight_encoding == 0 && !out->contract.weight_quant_family.empty()) {
        out->contract.weight_encoding = glm5_weight_encoding_from_family(out->contract.weight_quant_family);
    }
    if (out->contract.qkv_required) {
        out->contract.plain_kv_persistent_allowed = false;
    }
    return out->has_offload ? 1 : 0;
}

static int glm5_scan_offload_gguf_root(
    const std::string& root,
    glm5_offload_gguf_scan_t* out
) {
    if (!out) {
        return 0;
    }
    *out = glm5_offload_gguf_scan_t{};
    std::vector<std::string> paths;
    glm5_collect_gguf_files(root, 4, &paths);
    if (paths.empty()) {
        return 0;
    }
    std::sort(paths.begin(), paths.end());
    std::unordered_set<uint32_t> seen_indexes;
    bool saw_indexed = false;
    for (const std::string& path : paths) {
        glm5_offload_gguf_file_t file{};
        if (!glm5_read_offload_gguf_file(path, &file)) {
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

static void glm5_fill_check_from_offload_gguf_scan(
    const glm5_offload_gguf_scan_t& scan,
    glm5_model_root_check_t* out_check
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

static qkv_config_t glm5_qkv_config_from_offload_contract(const glm5_offload_gguf_contract_t& contract) {
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

static void glm5_engine_reset_qkv_cache(glm5_pc_engine* engine) {
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

static void glm5_engine_clear_offload_gguf_contract(glm5_pc_engine* engine) {
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
    engine->offload_gguf_first_file.clear();
    engine->offload_gguf_metadata_json.clear();
    engine->offload_weight_quant_family.clear();
    engine->offload_weight_kernel_family.clear();
}

static int glm5_engine_load_offload_gguf_contract(
    glm5_pc_engine* engine,
    const std::string& root,
    glm5_model_root_check_t* out_check
) {
    glm5_offload_gguf_scan_t scan{};
    if (!glm5_scan_offload_gguf_root(root, &scan)) {
        return 0;
    }
    glm5_fill_check_from_offload_gguf_scan(scan, out_check);
    glm5_engine_clear_offload_gguf_contract(engine);
    if (!engine || !scan.valid || scan.files.empty()) {
        return 1;
    }
    const glm5_offload_gguf_file_t& primary = scan.files.front();
    engine->offload_gguf_valid.store(1, std::memory_order_relaxed);
    engine->offload_gguf_file_count.store(scan.expected_count, std::memory_order_relaxed);
    engine->offload_gguf_tensor_count.store(scan.tensor_count, std::memory_order_relaxed);
    engine->offload_gguf_executable_tensor_count.store(0, std::memory_order_relaxed);
    engine->offload_gguf_first_file = primary.path;
    engine->offload_gguf_metadata_json = primary.metadata_json;
    engine->offload_weight_quant_bits.store(primary.contract.weight_bits, std::memory_order_relaxed);
    engine->offload_weight_encoding.store(primary.contract.weight_encoding, std::memory_order_relaxed);
    engine->offload_weight_block_size.store(primary.contract.weight_block_size, std::memory_order_relaxed);
    engine->offload_weight_quant_family = primary.contract.weight_quant_family;
    engine->offload_weight_kernel_family = primary.contract.weight_kernel_family;
    engine->offload_qkv_cfg = glm5_qkv_config_from_offload_contract(primary.contract);
    engine->offload_qkv_cfg_available.store(true, std::memory_order_release);
    const int qkv_required = primary.contract.qkv_required ||
        !primary.contract.plain_kv_persistent_allowed ||
        primary.contract.qkv_k_bits != 0 ||
        primary.contract.qkv_v_bits != 0;
    if (qkv_required) {
        engine->qkv_forced_by_format.store(1, std::memory_order_relaxed);
        engine->config.kv_mode = GLM5_KV_MODE_QKV;
        qkv_set_mode(true);
    }
    std::fprintf(stderr,
        "[storagellm] offload GGUF contract loaded: files=%u first=%s qkv=%s weight=%s/%ub\n",
        scan.expected_count,
        primary.path.c_str(),
        qkv_required ? "required" : "available",
        primary.contract.weight_quant_family.empty() ? "unknown" : primary.contract.weight_quant_family.c_str(),
        primary.contract.weight_bits);
    std::fflush(stderr);
    return 1;
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
    glm5_offload_gguf_scan_t gguf_scan{};
    if (!root.empty() && glm5_scan_offload_gguf_root(root, &gguf_scan)) {
        glm5_fill_check_from_offload_gguf_scan(gguf_scan, out_check);
        return 1;
    }
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
