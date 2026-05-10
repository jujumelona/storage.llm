#include "file_read.h"
#include "wide_path.h"

#include <fstream>
#include <sstream>
#include <string>

namespace storagellm {

bool read_text_file(const char* path, std::string* out) {
    if (!path || !out) {
        return false;
    }
    std::ifstream file;
#ifdef _WIN32
    const std::wstring wide = wide_path_from_utf8(path);
    if (!wide.empty()) {
        file.open(wide.c_str(), std::ios::binary);
    }
#else
    file.open(path, std::ios::binary);
#endif
    if (!file) {
        return false;
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    if (file.fail() && !file.eof()) {
        return false;  // Read error occurred
    }
    *out = buffer.str();
    return true;
}

}  // namespace storagellm
