#include "json_scan.h"

#include <cerrno>
#include <cstdlib>

namespace storagellm {

static size_t find_value(const JsonSlice& slice, const char* key) {
    if (!slice.text || !key) {
        return std::string::npos;
    }
    const std::string needle = std::string("\"") + key + "\"";
    size_t pos = slice.begin;
    for (;;) {
        const size_t key_pos = slice.text->find(needle, pos);
        if (key_pos == std::string::npos || key_pos >= slice.end) {
            return std::string::npos;
        }
        // Verify the key is followed by ':'; otherwise this was text inside a string value.
        const size_t colon = slice.text->find_first_not_of(" \t\r\n", key_pos + needle.size());
        if (colon != std::string::npos && colon < slice.end && (*slice.text)[colon] == ':') {
            return colon + 1;
        }
        pos = key_pos + 1;
    }
}

bool json_get_u64(const JsonSlice& slice, const char* key, uint64_t* out) {
    const size_t value = find_value(slice, key);
    if (value == std::string::npos || !out) {
        return false;
    }
    if (value >= slice.end) return false;
    // BUGFIX 462: errno 체크 추가
    errno = 0;
    char* end = nullptr;
    const char* begin = slice.text->c_str() + value;
    const char* limit = slice.text->c_str() + slice.end;
    *out = std::strtoull(begin, &end, 10);
    // Bug 1: Verify strtoull didn't read past slice boundary
    // BUGFIX 463: errno == ERANGE 체크 추가
    if (!end || end == begin || end > limit || errno == ERANGE) return false;
    return true;
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
            // BUGFIX 464: end + 2 범위 체크
            if (end + 1 >= slice.end) break;
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
    // BUGFIX 465: substr 범위 체크
    if (begin + 1 > end || end > slice.text->size()) {
        return false;
    }
    *out = slice.text->substr(begin + 1, end - begin - 1);
    return true;
}

}  // namespace storagellm
