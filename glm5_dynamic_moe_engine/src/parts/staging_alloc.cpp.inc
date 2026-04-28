static uint8_t* allocate_staging(uint64_t bytes, bool* pinned) {
    if (pinned) {
        *pinned = false;
    }
    if (bytes == 0) {
        return nullptr;
    }

#ifdef _WIN32
    void* ptr = VirtualAlloc(nullptr, (SIZE_T)bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!ptr) {
        return nullptr;
    }
    if (pinned) {
        *pinned = VirtualLock(ptr, (SIZE_T)bytes) != 0;
        if (!*pinned) {
            const SIZE_T min_ws = (SIZE_T)(bytes + 16ull * 1024ull * 1024ull);
            const SIZE_T max_ws = (SIZE_T)(bytes + 64ull * 1024ull * 1024ull);
            SetProcessWorkingSetSize(GetCurrentProcess(), min_ws, max_ws);
            *pinned = VirtualLock(ptr, (SIZE_T)bytes) != 0;
        }
    }
    return static_cast<uint8_t*>(ptr);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 4096, static_cast<size_t>(bytes)) != 0) {
        return nullptr;
    }
    if (pinned) {
        *pinned = mlock(ptr, static_cast<size_t>(bytes)) == 0;
    }
    return static_cast<uint8_t*>(ptr);
#endif
}

static void free_staging(uint8_t* ptr, uint64_t bytes, bool pinned) {
    if (!ptr) {
        return;
    }
#ifdef _WIN32
    if (pinned) {
        VirtualUnlock(ptr, (SIZE_T)bytes);
    }
    VirtualFree(ptr, 0, MEM_RELEASE);
#else
    if (pinned) {
        munlock(ptr, static_cast<size_t>(bytes));
    }
    free(ptr);
#endif
}

static uint8_t* allocate_best_staging(uint64_t requested_bytes, uint64_t* actual_bytes, bool* pinned) {
    if (actual_bytes) {
        *actual_bytes = 0;
    }
    if (pinned) {
        *pinned = false;
    }
    uint64_t bytes = requested_bytes;
    const uint64_t min_pinned = 1ull * 1024ull * 1024ull;
    while (bytes >= min_pinned) {
        bool got_pinned = false;
        uint8_t* ptr = allocate_staging(bytes, &got_pinned);
        if (ptr && got_pinned) {
            if (actual_bytes) {
                *actual_bytes = bytes;
            }
            if (pinned) {
                *pinned = true;
            }
            return ptr;
        }
        // Fix 60: Non-pinned success → free and try smaller pinned allocation
        // Non-pinned staging defeats async DMA (cuMemcpyHtoDAsync becomes sync)
        if (ptr) {
            free_staging(ptr, bytes, false);
        }
        bytes /= 2;
    }

    // Final fallback: original requested size, unpinned (honest sync path)
    uint8_t* ptr = allocate_staging(requested_bytes, pinned);
    if (ptr && actual_bytes) {
        *actual_bytes = requested_bytes;
    }
    return ptr;
}

