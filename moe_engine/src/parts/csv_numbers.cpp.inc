static std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> out;
    out.reserve(64);
    std::string cur;
    bool in_quote = false;
    for (size_t i = 0; i < line.size(); ++i) {
        const char ch = line[i];
        if (ch == '"') {
            if (in_quote && i + 1 < line.size() && line[i + 1] == '"') {
                cur.push_back('"');
                ++i;
            } else {
                in_quote = !in_quote;
            }
        } else if (ch == ',' && !in_quote) {
            out.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(ch);
        }
    }
    out.push_back(cur);
    if (!out.empty() && !out.back().empty() && out.back().back() == '\r') {
        out.back().pop_back();
    }
    return out;
}

static uint64_t parse_u64(const std::string& value) {
    if (value.empty()) {
        return 0;
    }
    // BUGFIX 207: Add error handling for strtoull ★
    // Problem: strtoull without error checking cannot distinguish "0" from parse error
    // If value="abc", strtoull returns 0 (same as value="0")
    // Solution: Check end pointer to verify parsing succeeded
    char* end = nullptr;
    errno = 0;  // Reset errno before call
    const uint64_t result = std::strtoull(value.c_str(), &end, 10);

    // Check for parsing errors:
    // 1. end == value.c_str(): no digits parsed
    // 2. errno == ERANGE: overflow (result == ULLONG_MAX)
    // 3. *end != '\0': trailing non-digit characters (partial parse)
    if (end == value.c_str() || errno == ERANGE) {
        return 0;  // Parse error or overflow
    }
    // Allow trailing whitespace but not other characters
    while (end && *end != '\0') {
        if (*end != ' ' && *end != '\t' && *end != '\r' && *end != '\n') {
            return 0;  // Invalid trailing characters
        }
        ++end;
    }
    return result;
}

static uint32_t parse_u32(const std::string& value) {
    return static_cast<uint32_t>(parse_u64(value));
}

static int csv_col_bounds_fast(
    const std::string& line,
    size_t target,
    size_t* out_begin,
    size_t* out_len
) {
    if (!out_begin || !out_len || line.find('"') != std::string::npos) {
        return 0;
    }
    size_t col = 0;
    size_t begin = 0;
    for (size_t i = 0; i <= line.size(); ++i) {
        if (i == line.size() || line[i] == ',') {
            if (col == target) {
                size_t end = i;
                if (end > begin && line[end - 1] == '\r') {
                    --end;
                }
                *out_begin = begin;
                *out_len = end - begin;
                return 1;
            }
            begin = i + 1;
            ++col;
        }
    }
    return 0;
}

static uint64_t parse_csv_u64_col_fast(const std::string& line, size_t target) {
    size_t begin = 0;
    size_t len = 0;
    if (!csv_col_bounds_fast(line, target, &begin, &len) || len == 0) {
        return 0;
    }
    uint64_t value = 0;
    const size_t end = begin + len;
    for (size_t i = begin; i < end; ++i) {
        const char ch = line[i];
        if (ch < '0' || ch > '9') {
            break;
        }
        value = value * 10ull + (uint64_t)(ch - '0');
    }
    return value;
}

static uint32_t parse_csv_u32_col_fast(const std::string& line, size_t target) {
    return static_cast<uint32_t>(parse_csv_u64_col_fast(line, target));
}

static std::string csv_string_col_fast(const std::string& line, size_t target) {
    size_t begin = 0;
    size_t len = 0;
    if (!csv_col_bounds_fast(line, target, &begin, &len)) {
        const std::vector<std::string> row = split_csv_line(line);
        return target < row.size() ? row[target] : std::string();
    }
    return line.substr(begin, len);
}

