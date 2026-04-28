#pragma once

#include <cctype>
#include <cstdint>
#include <cstdlib>

namespace storagellm {

inline uint64_t parse_byte_size(const char* raw) {
    if (!raw || !raw[0]) {
        return 0;
    }
    char* end = nullptr;
    const double value = std::strtod(raw, &end);
    if (value <= 0.0) {
        return 0;
    }
    uint64_t scale = 1;
    if (end && *end) {
        const char unit = static_cast<char>(std::tolower(static_cast<unsigned char>(*end)));
        if (unit == 'k') scale = 1024ull;
        if (unit == 'm') scale = 1024ull * 1024ull;
        if (unit == 'g') scale = 1024ull * 1024ull * 1024ull;
        if (unit == 't') scale = 1024ull * 1024ull * 1024ull * 1024ull;
    }
    return static_cast<uint64_t>(value * static_cast<double>(scale));
}

}  // namespace storagellm
