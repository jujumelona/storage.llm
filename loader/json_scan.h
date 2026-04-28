#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace storagellm {

struct JsonSlice {
    const std::string* text = nullptr;
    size_t begin = 0;
    size_t end = 0;
};

bool json_find_key_object(const std::string& text, const std::string& key, JsonSlice* out);
bool json_find_member_object(const JsonSlice& slice, const char* key, JsonSlice* out);
bool json_get_u64(const JsonSlice& slice, const char* key, uint64_t* out);
bool json_get_string(const JsonSlice& slice, const char* key, std::string* out);
size_t json_match_object(const std::string& text, size_t object_begin);

}  // namespace storagellm
