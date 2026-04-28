// ============================================================
// Memory-mapped file loader
// ============================================================

#pragma once

#include <stddef.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/types.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================
// mmap context
// ============================================================

typedef struct mmap_context_t {
    void* addr;          // Mapped address
    size_t size;         // File size
    int fd;              // File descriptor (-1 on Windows)
#ifdef _WIN32
    HANDLE hFile;
    HANDLE hMap;
#endif
    size_t mapped_offset; // Partial mapping offset
    size_t mapped_size;   // Partial mapping size
} mmap_context_t;

// Compatibility alias
typedef mmap_context_t mmap_ctx_t;

// ============================================================
// mmap API
// ============================================================

// Load and map an entire file.
mmap_context_t* mmap_load(const char* path);
void mmap_unload(mmap_context_t* ctx);

// Map or unmap into an existing context.
int mmap_file(mmap_context_t* ctx, const char* path);
void munmap_file(mmap_context_t* ctx);

// Accessors
void* mmap_get_ptr(const mmap_context_t* ctx, size_t offset);
size_t mmap_get_size(const mmap_context_t* ctx);

// Load a byte range from a file.
mmap_context_t* mmap_load_partial(const char* path, size_t offset, size_t length);

// Lock or unlock mapped pages when supported.
int mmap_lock(mmap_context_t* ctx);
int mmap_unlock(mmap_context_t* ctx);

// Hint that a mapped range will be used soon.
void mmap_prefetch(mmap_context_t* ctx, size_t offset, size_t length);

#ifdef __cplusplus
}
#endif

