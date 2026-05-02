#include "path_join.h"

namespace storagellm {

static bool path_is_absolute(const std::string& path) {
    if (path.empty()) {
        return false;
    }
    if (path[0] == '/' || path[0] == '\\') {
        return true;
    }
#ifdef _WIN32
    return path.size() >= 2 && path[1] == ':';
#else
    return false;
#endif
}

std::string path_join(const std::string& root, const std::string& rel) {
    if (root.empty() || path_is_absolute(rel)) {
        return rel;
    }

    // BUGFIX 543: Check combined path length to prevent overflow ★
    // Problem: root + rel can exceed filesystem limits causing errors
    // Solution: Check combined length before concatenation
    if (root.size() + rel.size() + 1 > 4096) {
        return rel;  // Return rel as fallback
    }

    const char last = root[root.size() - 1];
    if (last == '/' || last == '\\') {
        return root + rel;
    }
#ifdef _WIN32
    return root + "\\" + rel;
#else
    return root + "/" + rel;
#endif
}

}  // namespace storagellm
