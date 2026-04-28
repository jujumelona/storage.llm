#pragma once

#include <cstdlib>
#include <string>
#include <vector>

namespace storagellm {

inline bool json_read_int(const std::string& body, const char* key, int* out) {
    const std::string needle = std::string("\"") + key + "\"";
    const size_t pos = body.find(needle);
    if (pos == std::string::npos || !out) return false;
    const size_t colon = body.find(':', pos + needle.size());
    if (colon == std::string::npos) return false;
    char* end = nullptr;
    *out = static_cast<int>(std::strtol(body.c_str() + colon + 1, &end, 10));
    return end && end != body.c_str() + colon + 1;
}

inline std::vector<int> json_read_int_array(const std::string& body, const char* key) {
    std::vector<int> values;
    const std::string needle = std::string("\"") + key + "\"";
    const size_t pos = body.find(needle);
    const size_t lbr = pos == std::string::npos ? pos : body.find('[', pos + needle.size());
    const size_t rbr = lbr == std::string::npos ? lbr : body.find(']', lbr + 1);
    if (lbr == std::string::npos || rbr == std::string::npos) return values;
    const char* p = body.c_str() + lbr + 1;
    const char* e = body.c_str() + rbr;
    while (p < e) {
        char* next = nullptr;
        const long value = std::strtol(p, &next, 10);
        if (next != p) values.push_back(static_cast<int>(value));
        p = next && next > p ? next + 1 : p + 1;
    }
    return values;
}

}  // namespace storagellm
