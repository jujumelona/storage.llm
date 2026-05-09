#pragma once

#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace storagellm {

// Safe JSON parser that properly handles string contexts and escape sequences
// Addresses Bug B1: JSON parser structure bug

inline bool skip_json_string(const std::string& s, size_t& i) {
    if (i >= s.size() || s[i] != '"') return false;
    ++i;
    bool esc = false;
    while (i < s.size()) {
        const char ch = s[i++];
        if (esc) {
            esc = false;
            continue;
        }
        if (ch == '\\') {
            esc = true;
            continue;
        }
        if (ch == '"') return true;
    }
    return false;
}

inline bool find_top_level_key_colon(const std::string& s, const std::string& key, size_t& colon) {
    bool in_string = false, esc = false;
    int brace_depth = 0;

    for (size_t i = 0; i < s.size(); ++i) {
        const char ch = s[i];

        // Handle string context
        if (in_string) {
            if (esc) {
                esc = false;
            } else if (ch == '\\') {
                esc = true;
            } else if (ch == '"') {
                in_string = false;
            }
            continue;
        }

        // Check for key match at top level
        if (ch == '"') {
            size_t begin = i + 1;
            size_t temp_i = i;
            if (!skip_json_string(s, temp_i)) return false;

            // Only match keys at brace_depth == 1 (top level of object)
            if (brace_depth == 1 && s.compare(begin, (temp_i - 1) - begin, key) == 0) {
                size_t j = temp_i;
                while (j < s.size() && std::isspace(static_cast<unsigned char>(s[j]))) ++j;
                if (j < s.size() && s[j] == ':') {
                    colon = j;
                    return true;
                }
            }
            i = temp_i - 1;
            continue;
        }

        if (ch == '{') ++brace_depth;
        else if (ch == '}') --brace_depth;
    }
    return false;
}

inline bool json_is_value_delimiter(char ch) {
    return ch == '\0' || ch == ',' || ch == '}' || ch == ']';
}

inline bool json_parse_int_token(const char*& p, const char* end, int* out) {
    if (!p || !end || !out || p >= end) return false;

    const char* start = p;
    if (*p == '-') {
        ++p;
        if (p >= end || !std::isdigit(static_cast<unsigned char>(*p))) {
            p = start;
            return false;
        }
    } else if (!std::isdigit(static_cast<unsigned char>(*p))) {
        return false;
    }

    while (p < end && std::isdigit(static_cast<unsigned char>(*p))) {
        ++p;
    }

    const std::string token(start, static_cast<size_t>(p - start));
    errno = 0;
    char* parsed_end = nullptr;
    const long value = std::strtol(token.c_str(), &parsed_end, 10);
    if (errno == ERANGE || parsed_end == token.c_str() || *parsed_end != '\0') {
        return false;
    }
    if (value < static_cast<long>(std::numeric_limits<int>::min()) ||
        value > static_cast<long>(std::numeric_limits<int>::max())) {
        return false;
    }

    *out = static_cast<int>(value);
    return true;
}

inline bool json_read_int(const std::string& body, const char* key, int* out) {
    if (!key || !out) return false;

    size_t colon = 0;
    if (!find_top_level_key_colon(body, key, colon)) return false;

    const char* begin = body.c_str();
    const char* end = begin + body.size();
    const char* p = begin + colon + 1;
    while (p < end && std::isspace(static_cast<unsigned char>(*p))) ++p;

    // Check for non-numeric value (e.g., string, object, array)
    if (p >= end || *p == '"' || *p == '{' || *p == '[') return false;

    int value = 0;
    if (!json_parse_int_token(p, end, &value)) return false;

    while (p < end && std::isspace(static_cast<unsigned char>(*p))) ++p;
    if (p < end && !json_is_value_delimiter(*p)) return false;

    *out = value;
    return true;
}

inline std::vector<int> json_read_int_array(const std::string& body, const char* key) {
    std::vector<int> values;
    auto fail = [&values]() {
        values.clear();
        return values;
    };

    if (!key) return values;

    size_t colon = 0;
    if (!find_top_level_key_colon(body, key, colon)) return values;

    const char* begin = body.c_str();
    const char* end = begin + body.size();
    const char* p = begin + colon + 1;
    while (p < end && std::isspace(static_cast<unsigned char>(*p))) ++p;

    // Must start with '['
    if (p >= end || *p != '[') return values;
    ++p;

    // Find matching ']' while respecting strings and rejecting nested arrays/objects.
    int bracket_depth = 1;
    bool in_string = false, esc = false;
    const char* array_start = p;
    const char* array_end = nullptr;

    while (p < end && bracket_depth > 0) {
        if (in_string) {
            if (esc) {
                esc = false;
            } else if (*p == '\\') {
                esc = true;
            } else if (*p == '"') {
                in_string = false;
            }
        } else {
            if (*p == '"') {
                in_string = true;
            } else if (*p == '[') {
                return fail();
            } else if (*p == ']') {
                --bracket_depth;
                if (bracket_depth == 0) {
                    array_end = p;
                }
            } else if (*p == '{' || *p == '}') {
                return fail();
            }
        }
        ++p;
    }

    if (!array_end) return fail();

    p = array_start;
    bool expect_value = true;
    while (p < array_end) {
        while (p < array_end && std::isspace(static_cast<unsigned char>(*p))) ++p;
        if (p >= array_end) {
            return expect_value && !values.empty() ? fail() : values;
        }

        if (expect_value) {
            int value = 0;
            if (!json_parse_int_token(p, array_end, &value)) return fail();
            values.push_back(value);
            expect_value = false;
            continue;
        }

        if (*p != ',') return fail();
        ++p;
        expect_value = true;
    }

    if (expect_value && !values.empty()) return fail();
    return values;
}

}  // namespace storagellm
