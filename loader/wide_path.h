#pragma once

#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace storagellm {

#ifdef _WIN32
inline std::wstring wide_path_from_utf8(const char* path) {
    if (!path || !path[0]) {
        return std::wstring();
    }
    int count = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, nullptr, 0);
    bool use_acp = false;
    if (count <= 0) {
        count = MultiByteToWideChar(CP_ACP, 0, path, -1, nullptr, 0);
        use_acp = true;
    }
    if (count <= 0) {
        return std::wstring();
    }
    std::wstring out((size_t)count, L'\0');
    if (!MultiByteToWideChar(
            use_acp ? CP_ACP : CP_UTF8,
            use_acp ? 0 : MB_ERR_INVALID_CHARS,
            path,
            -1,
            &out[0],
            count)) {
        return std::wstring();
    }
    if (!out.empty() && out.back() == L'\0') {
        out.pop_back();
    }
    return out;
}

inline std::wstring wide_path_from_utf8(const std::string& path) {
    return wide_path_from_utf8(path.c_str());
}
#endif

}  // namespace storagellm
