#pragma once

#include <cstdlib>
#include <string>
#include <vector>

namespace storagellm {

inline bool json_read_int(const std::string& body, const char* key, int* out) {
    if (!key || !out) return false;
    const size_t key_len = std::strlen(key);
    size_t pos = 0;
    while ((pos = body.find('"', pos)) != std::string::npos) {
        if (pos + 1 + key_len < body.size() &&
            body.compare(pos + 1, key_len, key) == 0 &&
            body[pos + 1 + key_len] == '"') {
            const size_t colon = body.find(':', pos + key_len + 2);
            if (colon != std::string::npos) {
                char* end = nullptr;
                *out = static_cast<int>(std::strtol(body.c_str() + colon + 1, &end, 10));
                return end && end != body.c_str() + colon + 1;
            }
        }
        pos++;
    }
    return false;
}

inline std::vector<int> json_read_int_array(const std::string& body, const char* key) {
    std::vector<int> values;
    if (!key) return values;
    const size_t key_len = std::strlen(key);
    size_t pos = 0;
    while ((pos = body.find('"', pos)) != std::string::npos) {
        if (pos + 1 + key_len < body.size() &&
            body.compare(pos + 1, key_len, key) == 0 &&
            body[pos + 1 + key_len] == '"') {
            break;
        }
        pos++;
    }
    if (pos == std::string::npos) return values;

    const size_t lbr = body.find('[', pos + key_len + 2);
    const size_t rbr = lbr == std::string::npos ? lbr : body.find(']', lbr + 1);
    if (lbr == std::string::npos || rbr == std::string::npos) return values;
    const char* p = body.c_str() + lbr + 1;
    const char* e = body.c_str() + rbr;
    while (p < e) {
        char* next = nullptr;
        const long value = std::strtol(p, &next, 10);
        if (next != p) values.push_back(static_cast<int>(value));
        if (!next || next == p) break;
        p = next && next > p ? next + 1 : p + 1;
    }
    return values;
}

}  // namespace storagellm
