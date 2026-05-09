#pragma once

#include <cctype>
#include <cstdlib>
#include <cstring>
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

inline bool json_read_int(const std::string& body, const char* key, int* out) {
    if (!key || !out) return false;

    size_t colon = 0;
    if (!find_top_level_key_colon(body, key, colon)) return false;

    const char* p = body.c_str() + colon + 1;
    while (*p && std::isspace(static_cast<unsigned char>(*p))) ++p;

    // Check for non-numeric value (e.g., string, object, array)
    if (*p == '"' || *p == '{' || *p == '[') return false;

    char* end = nullptr;
    long v = std::strtol(p, &end, 10);
    if (end == p) return false;

    *out = static_cast<int>(v);
    return true;
}

inline std::vector<int> json_read_int_array(const std::string& body, const char* key) {
    std::vector<int> values;
    if (!key) return values;

    size_t colon = 0;
    if (!find_top_level_key_colon(body, key, colon)) return values;

    const char* p = body.c_str() + colon + 1;
    while (*p && std::isspace(static_cast<unsigned char>(*p))) ++p;

    // Must start with '['
    if (*p != '[') return values;
    ++p;

    // Find matching ']' while respecting nesting
    int bracket_depth = 1;
    bool in_string = false, esc = false;
    const char* array_start = p;
    const char* array_end = nullptr;

    while (*p && bracket_depth > 0) {
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
                ++bracket_depth;
            } else if (*p == ']') {
                --bracket_depth;
                if (bracket_depth == 0) {
                    array_end = p;
                }
            } else if (*p == '{' || *p == '}') {
                // Nested objects not allowed in int array
                return values;
            }
        }
        ++p;
    }

    if (!array_end) return values;

    // Parse integers from array
    p = array_start;
    while (p < array_end) {
        while (p < array_end && std::isspace(static_cast<unsigned char>(*p))) ++p;
        if (p >= array_end) break;

        // Check for nested array
        if (*p == '[') {
            values.clear();
            return values;  // Reject nested arrays
        }

        if (*p == ',' || *p == ']') {
            ++p;
            continue;
        }

        char* next = nullptr;
        long value = std::strtol(p, &next, 10);

        if (next != p) {
            values.push_back(static_cast<int>(value));
            p = next;
        } else {
            // Invalid value in array
            values.clear();
            return values;
        }
    }

    return values;
}

}  // namespace storagellm
