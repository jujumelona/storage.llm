#pragma once

#include <cstdint>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <cstdio>
#include <cstring>
#include <sys/sysinfo.h>
#include <unistd.h>
#endif

namespace storagellm {

inline uint64_t available_ram_bytes() {
#ifdef _WIN32
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    return GlobalMemoryStatusEx(&status) ? static_cast<uint64_t>(status.ullAvailPhys) : 0;
#elif defined(__linux__)
    std::FILE* meminfo = std::fopen("/proc/meminfo", "r");
    if (meminfo) {
        char key[64];
        unsigned long long value_kib = 0;
        char unit[32];
        while (std::fscanf(meminfo, "%63s %llu %31s\n", key, &value_kib, unit) == 3) {
            if (std::strcmp(key, "MemAvailable:") == 0) {
                std::fclose(meminfo);
                return static_cast<uint64_t>(value_kib) * 1024ull;
            }
        }
        std::fclose(meminfo);
    }
    struct sysinfo info {};
    if (sysinfo(&info) == 0) {
        return static_cast<uint64_t>(info.freeram) * static_cast<uint64_t>(info.mem_unit);
    }
    return 0;
#else
    return 0;
#endif
}

inline uint64_t default_ram_budget_bytes(uint32_t percent) {
    const uint64_t available = available_ram_bytes();
    if (available == 0) {
        return 8ull * 1024ull * 1024ull * 1024ull;
    }
    return (available * (percent > 90u ? 90u : percent)) / 100ull;
}

inline uint64_t current_process_rss_bytes() {
#if defined(__linux__)
    std::FILE* statm = std::fopen("/proc/self/statm", "r");
    if (!statm) {
        return 0;
    }
    unsigned long long total_pages = 0;
    unsigned long long resident_pages = 0;
    const int parsed = std::fscanf(statm, "%llu %llu", &total_pages, &resident_pages);
    std::fclose(statm);
    if (parsed != 2) {
        return 0;
    }
    const long page_size = sysconf(_SC_PAGESIZE);
    return page_size > 0 ? resident_pages * static_cast<uint64_t>(page_size) : 0;
#else
    return 0;
#endif
}

}  // namespace storagellm
