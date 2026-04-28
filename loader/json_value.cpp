#include "json_scan.h"

#include <cstdlib>

namespace storagellm {

static size_t find_value(const JsonSlice& slice, const char* key) {
    if (!slice.text || !key) {
        return std::string::npos;
    }
    const std::string needle = std::string("\"") + key + "\"";
    const size_t key_pos = slice.text->find(needle, slice.begin);
    if (key_pos == std::string::npos || key_pos >= slice.end) {
        return std::string::npos;
    }
    const size_t colon = slice.text->find(':', key_pos + needle.size());
    return colon == std::string::npos || colon >= slice.end ? std::string::npos : colon + 1;
}

bool json_get_u64(const JsonSlice& slice, const char* key, uint64_t* out) {
    const size_t value = find_value(slice, key);
    if (value == std::string::npos || !out) {
        return false;
    }
    char* end = nullptr;
    *out = std::strtoull(slice.text->c_str() + value, &end, 10);
    return end && end != slice.text->c_str() + value;
}

bool json_get_string(const JsonSlice& slice, const char* key, std::string* out) {
    const size_t value = find_value(slice, key);
    if (value == std::string::npos || !out) {
        return false;
    }
    const size_t begin = slice.text->find('"', value);
    // Check bounds
    if (begin == std::string::npos || begin >= slice.end) {
        return false;
    }
    // Bug 5: Handle escaped quotes inside string
    size_t end = begin + 1;
    while (end < slice.end) {
        if ((*slice.text)[end] == '\\') {
            end += 2;  // Skip escape sequence
            continue;
        }
        if ((*slice.text)[end] == '"') {
            break;
        }
        ++end;
    }
    if (end >= slice.end) {
        return false;
    }
    *out = slice.text->substr(begin + 1, end - begin - 1);
    return true;
}

}  // namespace storagellm
