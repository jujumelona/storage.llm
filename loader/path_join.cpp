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
