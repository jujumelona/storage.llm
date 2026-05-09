#include "process.h"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#if !defined(_WIN32)
#include <sys/wait.h>
#endif

#if defined(_WIN32)
#define popen _popen
#define pclose _pclose
#endif

namespace fs = std::filesystem;

namespace storagellm::autotune {

std::string shell_quote(const std::string& s) {
#if defined(_WIN32)
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else out += c;
    }
    out += "\"";
    return out;
#else
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    out += "'";
    return out;
#endif
}

ProcessResult run_capture(const std::string& command) {
    ProcessResult r;
#if defined(_WIN32)
    std::string cmd = command + " 2>&1";
#else
    std::string cmd = command + " 2>&1";
#endif
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        r.output = "popen failed";
        return r;
    }
    std::array<char, 4096> buf{};
    while (fgets(buf.data(), static_cast<int>(buf.size()), pipe)) {
        r.output += buf.data();
    }
    int rc = pclose(pipe);
#if defined(_WIN32)
    r.code = rc;
#else
    if (rc == -1) r.code = -1;
    else r.code = WEXITSTATUS(rc);
#endif
    return r;
}

int run_passthrough(const std::string& command) {
    return std::system(command.c_str());
}

std::string find_executable(const std::vector<std::string>& names) {
    for (const auto& name : names) {
#if defined(_WIN32)
        auto r = run_capture("where " + shell_quote(name));
#else
        auto r = run_capture("command -v " + shell_quote(name));
#endif
        if (r.code == 0 && !r.output.empty()) {
            std::istringstream is(r.output);
            std::string line;
            if (std::getline(is, line)) {
                while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
                if (!line.empty()) return line;
            }
        }
    }
    return {};
}

bool file_exists(const std::string& path) {
    std::error_code ec;
    return fs::is_regular_file(fs::path(path), ec);
}

bool dir_exists(const std::string& path) {
    std::error_code ec;
    return fs::is_directory(fs::path(path), ec);
}

std::string shared_library_suffix() {
#if defined(_WIN32)
    return ".dll";
#elif defined(__APPLE__)
    return ".dylib";
#else
    return ".so";
#endif
}

std::string executable_suffix() {
#if defined(_WIN32)
    return ".exe";
#else
    return "";
#endif
}

} // namespace storagellm::autotune
