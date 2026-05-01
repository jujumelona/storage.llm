// JSON parsing utilities
// Lightweight JSON parser for extracting values from metadata

static int moe_json_find_scalar(const std::string& text, const char* key, size_t* value_pos) {
    if (!key || !value_pos) {
        return 0;
    }
    // BUGFIX 90: Issue 14 - Optimize JSON key search with string_view ★
    // Problem: std::string("\"") + key + "\"" creates temporary string on every call
    // Called frequently during model validation (hundreds of keys)
    // Solution: Use string_view or pre-compute needle length
    // Avoid temporary string allocation
    const size_t key_len = std::strlen(key);
    const size_t needle_len = key_len + 2;  // "key"

    size_t pos = 0;
    while (pos + needle_len <= text.size()) {
        // Manual search: look for "key"
        if (text[pos] == '"' &&
            text.compare(pos + 1, key_len, key) == 0 &&
            text[pos + 1 + key_len] == '"') {
            const size_t colon = text.find_first_not_of(" \t\r\n", pos + needle_len);
            if (colon != std::string::npos && colon < text.size() && text[colon] == ':') {
                *value_pos = text.find_first_not_of(" \t\r\n", colon + 1);
                return *value_pos != std::string::npos;
            }
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

static size_t moe_json_match_bracket_local(
    const std::string& text,
    size_t begin,
    char open_ch,
    char close_ch
) {
    if (begin >= text.size() || text[begin] != open_ch) {
        return std::string::npos;
    }
    uint32_t depth = 0;
    bool in_string = false;
    bool escape = false;
    for (size_t i = begin; i < text.size(); ++i) {
        const char c = text[i];
        if (in_string) {
            if (escape) {
                escape = false;
            } else if (c == '\\') {
                escape = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }
        if (c == '"') {
            in_string = true;
            continue;
        }
        if (c == open_ch) {
            ++depth;
        } else if (c == close_ch) {
            if (depth == 0) {
                return std::string::npos;
            }
            --depth;
            if (depth == 0) {
                return i + 1;
            }
        }
    }
    return std::string::npos;
}

static int moe_json_find_array_local(
    const std::string& text,
    const char* key,
    size_t* begin,
    size_t* end
) {
    size_t value = 0;
    if (!begin || !end || !moe_json_find_scalar(text, key, &value) ||
        value >= text.size() || text[value] != '[') {
        return 0;
    }
    const size_t close = moe_json_match_bracket_local(text, value, '[', ']');
    if (close == std::string::npos) {
        return 0;
    }
    *begin = value;
    *end = close;
    return 1;
}

static int moe_json_next_object_in_array(
    const std::string& text,
    size_t* cursor,
    size_t end,
    size_t* obj_begin,
    size_t* obj_end
) {
    if (!cursor || !obj_begin || !obj_end || *cursor >= end) {
        return 0;
    }
    size_t pos = *cursor;
    while (pos < end && text[pos] != '{') {
        ++pos;
    }
    if (pos >= end) {
        *cursor = end;
        return 0;
    }
    const size_t close = moe_json_match_bracket_local(text, pos, '{', '}');
    if (close == std::string::npos || close > end) {
        *cursor = end;
        return 0;
    }
    *obj_begin = pos;
    *obj_end = close;
    *cursor = close;
    return 1;
}

static int moe_json_get_u64_slice(
    const std::string& text,
    size_t begin,
    size_t end,
    const char* key,
    uint64_t* out
) {
    if (begin >= end || end > text.size()) {
        return 0;
    }
    return moe_json_get_u64_local(text.substr(begin, end - begin), key, out);
}

static int moe_json_get_string_slice(
    const std::string& text,
    size_t begin,
    size_t end,
    const char* key,
    std::string* out
) {
    if (begin >= end || end > text.size()) {
        return 0;
    }
    return moe_json_get_string_local(text.substr(begin, end - begin), key, out);
}

static int moe_json_get_u64_array_slice(
    const std::string& text,
    size_t begin,
    size_t end,
    const char* key,
    uint64_t* values,
    uint32_t max_values,
    uint32_t* out_count
) {
    if (out_count) {
        *out_count = 0;
    }
    if (!values || max_values == 0 || begin >= end || end > text.size()) {
        return 0;
    }
    const std::string slice = text.substr(begin, end - begin);
    size_t arr_begin = 0;
    size_t arr_end = 0;
    if (!moe_json_find_array_local(slice, key, &arr_begin, &arr_end)) {
        return 0;
    }
    uint32_t count = 0;
    size_t pos = arr_begin + 1;
    while (pos < arr_end && count < max_values) {
        pos = slice.find_first_not_of(" \t\r\n,", pos);
        if (pos == std::string::npos || pos >= arr_end || slice[pos] == ']') {
            break;
        }
        char* parsed_end = nullptr;
        const char* start = slice.c_str() + pos;
        const uint64_t value = std::strtoull(start, &parsed_end, 10);
        if (!parsed_end || parsed_end == start) {
            break;
        }
        values[count++] = value;
        pos = (size_t)(parsed_end - slice.c_str());
    }
    if (out_count) {
        *out_count = count;
    }
    return count > 0 ? 1 : 0;
}

