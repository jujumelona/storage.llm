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
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <sys/sysctl.h>
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
#elif defined(__APPLE__)
    // BUGFIX 48: Issue 19 - macOS available RAM implementation ?�★
    // Problem: macOS path returned 0, causing moe_generation_should_preallocate_kv_cache
    // to always return false. KV cache never pre-allocates on macOS.
    // Solution: Use host_statistics64 to get vm_statistics64_data_t. Do not count
    // inactive pages here; unified-memory GPU pressure can make them unavailable
    // quickly enough to over-admit RAM/Metal work under heavy I/O.
    mach_port_t host_port = mach_host_self();
    vm_size_t page_size = 0;
    host_page_size(host_port, &page_size);

    vm_statistics64_data_t vm_stat;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    kern_return_t kr = host_statistics64(
        host_port,
        HOST_VM_INFO64,
        reinterpret_cast<host_info64_t>(&vm_stat),
        &count
    );

    if (kr != KERN_SUCCESS) {
        return 0;
    }

    const uint64_t available_pages = static_cast<uint64_t>(vm_stat.free_count);
    return available_pages * static_cast<uint64_t>(page_size);
#else
    return 0;
#endif
}

inline uint64_t total_ram_bytes() {
#ifdef _WIN32
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    return GlobalMemoryStatusEx(&status) ? static_cast<uint64_t>(status.ullTotalPhys) : 0;
#elif defined(__linux__)
    struct sysinfo info {};
    if (sysinfo(&info) == 0) {
        return static_cast<uint64_t>(info.totalram) * static_cast<uint64_t>(info.mem_unit);
    }
    return 0;
#elif defined(__APPLE__)
    int mib[2] = {CTL_HW, HW_MEMSIZE};
    uint64_t physical_memory = 0;
    size_t length = sizeof(physical_memory);
    sysctl(mib, 2, &physical_memory, &length, NULL, 0);
    return physical_memory;
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

