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
    return out;
}

static uint64_t parse_u64(const std::string& value) {
    if (value.empty()) {
        return 0;
    }
    return static_cast<uint64_t>(std::strtoull(value.c_str(), nullptr, 10));
}

static uint32_t parse_u32(const std::string& value) {
    return static_cast<uint32_t>(parse_u64(value));
}

