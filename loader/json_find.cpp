#include "json_scan.h"

#include <cstring>

namespace storagellm {

static bool find_object_after(const std::string& text, size_t pos, JsonSlice* out) {
    if (!out) {
        return false;
    }
    const size_t colon = text.find(':', pos);
    const size_t begin = text.find('{', colon == std::string::npos ? pos : colon);
    if (begin == std::string::npos) {
        return false;
    }
    const size_t end = json_match_object(text, begin);
    if (end == std::string::npos) {
        return false;
    }
    *out = JsonSlice{&text, begin, end};
    return true;
}

bool json_find_key_object(const std::string& text, const std::string& key, JsonSlice* out) {
    if (!out) {
        return false;
    }
    const size_t pos = text.find("\"" + key + "\"");
    return pos != std::string::npos && find_object_after(text, pos, out);
}

bool json_find_member_object(const JsonSlice& slice, const char* key, JsonSlice* out) {
    if (!slice.text || !key || !out) {
        return false;
    }
    const std::string needle = std::string("\"") + key + "\"";

    // BUGFIX 542: Check needle length doesn't overflow ★
    // Problem: key can be very long causing string allocation failure
    // Solution: Check key length before creating needle
    const size_t key_len = std::strlen(key);
    if (key_len > 10000) {
        return false;
    }

    size_t pos = slice.begin;
    while ((pos = slice.text->find(needle, pos)) != std::string::npos && pos < slice.end) {
        const size_t after_key = pos + needle.size();
        if (after_key > slice.end) {
            return false;
        }
        // Bug 5: Verify this is a real JSON key (followed by ':'), not text in a string value
        const size_t colon = slice.text->find_first_not_of(" \t\r\n", after_key);
        if (colon == std::string::npos || colon >= slice.end || (*slice.text)[colon] != ':') {
            ++pos;
            continue;
        }
        return find_object_after(*slice.text, pos, out) && out->end <= slice.end;
    }
    return false;
}

}  // namespace storagellm
