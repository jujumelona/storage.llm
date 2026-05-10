#pragma once
#include <string>
#include <vector>

namespace storagellm::autotune {
struct ProcessResult {
    int code = -1;
    std::string output;
};

std::string shell_quote(const std::string& s);
ProcessResult run_capture(const std::string& command);
int run_passthrough(const std::string& command);
std::string find_executable(const std::vector<std::string>& names);
bool file_exists(const std::string& path);
bool dir_exists(const std::string& path);
std::string shared_library_suffix();
std::string executable_suffix();
}
