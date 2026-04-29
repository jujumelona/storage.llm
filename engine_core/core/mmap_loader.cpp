// ============================================================
// Memory-mapped file loader
// ============================================================
// Supports whole-file and partial range mappings.

#include "mmap_loader.h"
#include "../../loader/wide_path.h"
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>

#ifndef _WIN32
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#ifdef _WIN32
static HANDLE mmap_open_read_file(const char* path) {
    const std::wstring wide_path = storagellm::wide_path_from_utf8(path);
    if (wide_path.empty()) {
        return INVALID_HANDLE_VALUE;
    }
    return CreateFileW(
        wide_path.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifndef _WIN32
static void mmap_discard_pages(mmap_context_t* ctx, void* addr, size_t size) {
    if (!ctx || !addr || addr == MAP_FAILED || size == 0) {
        return;
    }
    madvise(addr, size, MADV_DONTNEED);
#ifdef POSIX_FADV_DONTNEED
    if (ctx->fd >= 0) {
        posix_fadvise(ctx->fd, 0, 0, POSIX_FADV_DONTNEED);
    }
#endif
}
#else
static void mmap_discard_pages(mmap_context_t*, void*, size_t) {}
#endif

// ============================================================
// Map or unmap into an existing context
// ============================================================

int mmap_file(mmap_context_t* ctx, const char* path) {
    if (!ctx || !path) return 0;

    memset(ctx, 0, sizeof(mmap_context_t));

#ifdef _WIN32
    ctx->fd = -1;
    ctx->hFile = mmap_open_read_file(path);

    if (ctx->hFile == INVALID_HANDLE_VALUE) {
        return 0;
    }

    LARGE_INTEGER fileSize;
    if (!GetFileSizeEx(ctx->hFile, &fileSize)) {
        CloseHandle(ctx->hFile);
        ctx->hFile = NULL;
        return 0;
    }
    ctx->size = (size_t)fileSize.QuadPart;

    ctx->hMap = CreateFileMappingW(
        ctx->hFile,
        NULL,
        PAGE_READONLY,
        0,
        0,
        NULL
    );

    if (!ctx->hMap) {
        CloseHandle(ctx->hFile);
        ctx->hFile = NULL;
        return 0;
    }

    ctx->addr = MapViewOfFile(
        ctx->hMap,
        FILE_MAP_READ,
        0,
        0,
        0
    );

    if (!ctx->addr) {
        CloseHandle(ctx->hMap);
        CloseHandle(ctx->hFile);
        ctx->hMap = NULL;
        ctx->hFile = NULL;
        return 0;
    }

    return 1;
#else
    ctx->fd = open(path, O_RDONLY);
    if (ctx->fd < 0) {
        return 0;
    }

    struct stat st;
    if (fstat(ctx->fd, &st) < 0) {
        close(ctx->fd);
        ctx->fd = -1;
        return 0;
    }
    ctx->size = (size_t)st.st_size;

    // Bug 3: Reject zero-size files (mmap with length=0 returns EINVAL)
    if (ctx->size == 0) {
        close(ctx->fd);
        ctx->fd = -1;
        return 0;
    }

    ctx->addr = mmap(
        NULL,
        ctx->size,
        PROT_READ,
        MAP_SHARED,
        ctx->fd,
        0
    );

    if (ctx->addr == MAP_FAILED) {
        close(ctx->fd);
        ctx->fd = -1;
        ctx->addr = NULL;
        return 0;
    }

    madvise(ctx->addr, ctx->size, MADV_RANDOM);
    return 1;
#endif
}

void munmap_file(mmap_context_t* ctx) {
    if (!ctx) return;

    void* unmap_addr = ctx->mapped_size > 0 ? (char*)ctx->addr - ctx->mapped_offset : ctx->addr;
    size_t unmap_size = ctx->mapped_size > 0 ? ctx->mapped_size : ctx->size;

#ifdef _WIN32
    if (unmap_addr) {
        UnmapViewOfFile(unmap_addr);
        ctx->addr = NULL;
    }
    if (ctx->hMap) {
        CloseHandle(ctx->hMap);
        ctx->hMap = NULL;
    }
    if (ctx->hFile && ctx->hFile != INVALID_HANDLE_VALUE) {
        CloseHandle(ctx->hFile);
        ctx->hFile = NULL;
    }
#else
    if (unmap_addr && unmap_addr != MAP_FAILED) {
        mmap_discard_pages(ctx, unmap_addr, unmap_size);
        munmap(unmap_addr, unmap_size);
        ctx->addr = NULL;
    }
    if (ctx->fd >= 0) {
        close(ctx->fd);
        ctx->fd = -1;
    }
#endif
    ctx->size = 0;
    // Fix: Clear mapped_size and mapped_offset to prevent double-unmap UB
    ctx->mapped_size = 0;
    ctx->mapped_offset = 0;
}

// ============================================================
// Windows implementation
// ============================================================

#ifdef _WIN32

mmap_context_t* mmap_load(const char* path) {
    mmap_context_t* ctx = (mmap_context_t*)malloc(sizeof(mmap_context_t));
    if (!ctx) return NULL;

    memset(ctx, 0, sizeof(mmap_context_t));
    ctx->fd = -1;

    // Align the mapping start to the allocation granularity.
    ctx->hFile = mmap_open_read_file(path);

    if (ctx->hFile == INVALID_HANDLE_VALUE) {
        free(ctx);
        return NULL;
    }

    // Open and map the aligned byte range.
    LARGE_INTEGER fileSize;
    if (!GetFileSizeEx(ctx->hFile, &fileSize)) {
        CloseHandle(ctx->hFile);
        free(ctx);
        return NULL;
    }
    ctx->size = (size_t)fileSize.QuadPart;

    // Expose the requested offset inside the aligned mapping.
    ctx->hMap = CreateFileMappingW(
        ctx->hFile,
        NULL,
        PAGE_READONLY,
        0,
        0,
        NULL
    );

    if (!ctx->hMap) {
        CloseHandle(ctx->hFile);
        free(ctx);
        return NULL;
    }

    // Cleanup
    ctx->addr = MapViewOfFile(
        ctx->hMap,
        FILE_MAP_READ,
        0,
        0,
        0
    );

    if (!ctx->addr) {
        CloseHandle(ctx->hMap);
        CloseHandle(ctx->hFile);
        free(ctx);
        return NULL;
    }

    return ctx;
}

void mmap_unload(mmap_context_t* ctx) {
    if (!ctx) return;
    void* unmap_addr = ctx->mapped_size > 0 ? (char*)ctx->addr - ctx->mapped_offset : ctx->addr;
    if (unmap_addr) {
        UnmapViewOfFile(unmap_addr);
    }
    if (ctx->hMap) {
        CloseHandle(ctx->hMap);
        ctx->hMap = NULL;  // Fix: prevent double-close
    }
    if (ctx->hFile && ctx->hFile != INVALID_HANDLE_VALUE) {
        CloseHandle(ctx->hFile);
        ctx->hFile = NULL;  // Fix: prevent double-close
    }
    free(ctx);
}

// ============================================================
// POSIX implementation (Linux, macOS)
// ============================================================

#else

mmap_context_t* mmap_load(const char* path) {
    mmap_context_t* ctx = (mmap_context_t*)malloc(sizeof(mmap_context_t));
    if (!ctx) return NULL;

    memset(ctx, 0, sizeof(mmap_context_t));

    // Align the mapping start to the page size.
    ctx->fd = open(path, O_RDONLY);
    if (ctx->fd < 0) {
        free(ctx);
        return NULL;
    }

    // Map the aligned byte range.
    struct stat st;
    if (fstat(ctx->fd, &st) < 0) {
        close(ctx->fd);
        free(ctx);
        return NULL;
    }
    ctx->size = (size_t)st.st_size;

    // Bug 1: Reject zero-size files (mmap with length=0 returns EINVAL on Linux, undefined on macOS)
    if (ctx->size == 0) {
        close(ctx->fd);
        free(ctx);
        return NULL;
    }

    // mmap
    ctx->addr = mmap(
        NULL,
        ctx->size,
        PROT_READ,
        MAP_SHARED,
        ctx->fd,
        0
    );

    if (ctx->addr == MAP_FAILED) {
        close(ctx->fd);
        free(ctx);
        return NULL;
    }

    madvise(ctx->addr, ctx->size, MADV_RANDOM);

    return ctx;
}

void mmap_unload(mmap_context_t* ctx) {
    if (!ctx) return;
    void* unmap_addr = ctx->mapped_size > 0 ? (char*)ctx->addr - ctx->mapped_offset : ctx->addr;
    size_t unmap_size = ctx->mapped_size > 0 ? ctx->mapped_size : ctx->size;
    if (unmap_addr && unmap_addr != MAP_FAILED) {
        mmap_discard_pages(ctx, unmap_addr, unmap_size);
        munmap(unmap_addr, unmap_size);
    }
    if (ctx->fd >= 0) {
        close(ctx->fd);
    }
    free(ctx);
}

#endif

// ============================================================
// Whole-file convenience wrappers
// ============================================================

void* mmap_get_ptr(const mmap_context_t* ctx, size_t offset) {
    if (!ctx || !ctx->addr || offset >= ctx->size) {
        return NULL;
    }
    return (char*)ctx->addr + offset;
}

size_t mmap_get_size(const mmap_context_t* ctx) {
    return ctx ? ctx->size : 0;
}

// Partial range mapping
mmap_context_t* mmap_load_partial(const char* path, size_t offset, size_t length) {
    mmap_context_t* ctx = (mmap_context_t*)malloc(sizeof(mmap_context_t));
    if (!ctx) return NULL;
    if (!path || length == 0) { free(ctx); return NULL; }
    memset(ctx, 0, sizeof(mmap_context_t));

#ifdef _WIN32
    ctx->fd = -1;
    ctx->hFile = mmap_open_read_file(path);
    if (ctx->hFile == INVALID_HANDLE_VALUE) { free(ctx); return NULL; }

    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    DWORD allocGranularity = sysInfo.dwAllocationGranularity;

    size_t aligned_offset = (offset / allocGranularity) * allocGranularity;
    size_t offset_diff = offset - aligned_offset;
    size_t map_length = length + offset_diff;

    ctx->hMap = CreateFileMappingW(ctx->hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!ctx->hMap) { CloseHandle(ctx->hFile); free(ctx); return NULL; }

    DWORD offsetHigh = (DWORD)(aligned_offset >> 32);
    DWORD offsetLow = (DWORD)(aligned_offset & 0xFFFFFFFF);

    void* addr = MapViewOfFile(ctx->hMap, FILE_MAP_READ, offsetHigh, offsetLow, map_length);
    if (!addr) { CloseHandle(ctx->hMap); CloseHandle(ctx->hFile); free(ctx); return NULL; }

    ctx->mapped_offset = offset_diff;
    ctx->mapped_size = map_length;
    ctx->addr = (char*)addr + offset_diff;
    ctx->size = length;
    return ctx;
#else
    ctx->fd = open(path, O_RDONLY);
    if (ctx->fd < 0) { free(ctx); return NULL; }

    // Fix: Validate file size before mapping to prevent SIGBUS
    struct stat st;
    if (fstat(ctx->fd, &st) < 0) { close(ctx->fd); free(ctx); return NULL; }

    long page_size = sysconf(_SC_PAGESIZE);
    // Fix: Validate sysconf return value to prevent UB
    if (page_size <= 0) page_size = 4096;
    size_t aligned_offset = (offset / (size_t)page_size) * (size_t)page_size;
    size_t offset_diff = offset - aligned_offset;

    // Fix: Check for integer overflow in length + offset_diff
    if (length > SIZE_MAX - offset_diff) { close(ctx->fd); free(ctx); return NULL; }
    size_t map_length = length + offset_diff;

    // Fix: Verify map_length doesn't exceed file size
    if ((size_t)st.st_size < aligned_offset || map_length > (size_t)st.st_size - aligned_offset) {
        close(ctx->fd); free(ctx); return NULL;
    }

    // Fix: Use MAP_SHARED for consistency with mmap_load()
    void* addr = mmap(NULL, map_length, PROT_READ, MAP_SHARED, ctx->fd, aligned_offset);
    if (addr == MAP_FAILED) { close(ctx->fd); free(ctx); return NULL; }

    // Bug Fix: MADV_SEQUENTIAL → MADV_WILLNEED for partial expert mappings.
    // Expert access is random (MoE router), not sequential. SEQUENTIAL causes
    // kernel to prefetch beyond requested range, wasting RAM bandwidth and page cache.
    // WILLNEED only prefetches the specified range.
    madvise(addr, map_length, MADV_WILLNEED);

    ctx->mapped_offset = offset_diff;
    ctx->mapped_size = map_length;
    ctx->addr = (char*)addr + offset_diff;
    ctx->size = length;
    return ctx;
#endif
}

// Memory locking helpers
int mmap_lock(mmap_context_t* ctx) {
    if (!ctx || !ctx->addr) return -1;

#ifdef _WIN32
    void* base = ctx->mapped_size > 0 ? (char*)ctx->addr - ctx->mapped_offset : ctx->addr;
    size_t sz = ctx->mapped_size > 0 ? ctx->mapped_size : ctx->size;
    return VirtualLock(base, sz) ? 0 : -1;
#else
    // Fix: Use actual mapping base for partial mappings (page-aligned address required)
    void* base = ctx->mapped_size > 0 ? (char*)ctx->addr - ctx->mapped_offset : ctx->addr;
    size_t sz = ctx->mapped_size > 0 ? ctx->mapped_size : ctx->size;
    return mlock(base, sz);
#endif
}

int mmap_unlock(mmap_context_t* ctx) {
    if (!ctx || !ctx->addr) return -1;

#ifdef _WIN32
    void* base = ctx->mapped_size > 0 ? (char*)ctx->addr - ctx->mapped_offset : ctx->addr;
    size_t sz = ctx->mapped_size > 0 ? ctx->mapped_size : ctx->size;
    return VirtualUnlock(base, sz) ? 0 : -1;
#else
    // Fix: Use actual mapping base for partial mappings (page-aligned address required)
    void* base = ctx->mapped_size > 0 ? (char*)ctx->addr - ctx->mapped_offset : ctx->addr;
    size_t sz = ctx->mapped_size > 0 ? ctx->mapped_size : ctx->size;
    return munlock(base, sz);
#endif
}

// Prefetch helpers
void mmap_prefetch(mmap_context_t* ctx, size_t offset, size_t length) {
    if (!ctx || !ctx->addr) return;
    if (offset >= ctx->size || length == 0) return;

#ifndef _WIN32
    // POSIX: MADV_WILLNEED
    // Critical Fix 2: Prevent integer overflow in offset + length
    // If offset + length > SIZE_MAX, wraps to small value causing massive madvise call
    if (length > ctx->size - offset) {
        length = ctx->size - offset;
    }
    char* addr = (char*)ctx->addr + offset;
    size_t end = offset + length; // Now safe from overflow
    if (end > ctx->size) end = ctx->size;

    madvise(addr, end - offset, MADV_WILLNEED);
#else
    // Critical Fix 2: Prevent integer overflow (Windows path)
    if (length > ctx->size - offset) {
        length = ctx->size - offset;
    }
    char* addr = (char*)ctx->addr + offset;
    size_t end = offset + length;
    if (end > ctx->size) end = ctx->size;
    size_t prefetch_len = end - offset;

    typedef BOOL (WINAPI *PrefetchVirtualMemoryFn)(
        HANDLE,
        ULONG_PTR,
        PWIN32_MEMORY_RANGE_ENTRY,
        ULONG
    );
    static PrefetchVirtualMemoryFn pfn = reinterpret_cast<PrefetchVirtualMemoryFn>(
        GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "PrefetchVirtualMemory"));
    if (pfn) {
        WIN32_MEMORY_RANGE_ENTRY range;
        range.VirtualAddress = addr;
        range.NumberOfBytes = prefetch_len;
        if (pfn(GetCurrentProcess(), 1, &range, 0)) {
            return;
        }
    }
    // Fallback for older Windows versions.
    volatile char dummy = addr[0];
    dummy ^= addr[prefetch_len - 1];
    (void)dummy;
#endif
}

#ifdef __cplusplus
}
#endif

