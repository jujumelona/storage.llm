#include "json_scan.h"

namespace storagellm {

size_t json_match_object(const std::string& text, size_t object_begin) {
    if (object_begin >= text.size() || text[object_begin] != '{') {
        return std::string::npos;
    }
    int depth = 0;
    bool quote = false;
    bool escape = false;
    for (size_t i = object_begin; i < text.size(); ++i) {
        const char c = text[i];
        if (escape) {
            escape = false;
            continue;
        }
        if (quote && c == '\\') {
            escape = true;
            continue;
        }
        if (c == '"') {
            quote = !quote;
        } else if (!quote && c == '{') {
            ++depth;
        } else if (!quote && c == '}' && --depth == 0) {
            return i + 1;
        }
    }
    return std::string::npos;
}

}  // namespace storagellm
