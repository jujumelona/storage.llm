#pragma once

#include <string>

namespace storagellm {

bool read_text_file(const char* path, std::string* out);

}  // namespace storagellm
