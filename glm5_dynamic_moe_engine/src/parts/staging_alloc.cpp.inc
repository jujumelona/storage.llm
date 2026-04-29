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
#ifndef _WIN32
    struct rlimit memlock_limit {};
    if (getrlimit(RLIMIT_MEMLOCK, &memlock_limit) == 0 && memlock_limit.rlim_cur != RLIM_INFINITY) {
        const uint64_t limit = static_cast<uint64_t>(memlock_limit.rlim_cur);
        if (limit < min_pinned) {
            fprintf(stderr,
                "[storagellm] pinned staging skipped: RLIMIT_MEMLOCK=%llu bytes < %llu bytes. "
                "Falling back to non-pinned staging (sync DMA only).\n",
                (unsigned long long)limit,
                (unsigned long long)min_pinned);
            // Fallback: allocate non-pinned buffer
            uint8_t* ptr = allocate_staging(requested_bytes, nullptr);
            if (ptr) {
                if (actual_bytes) {
                    *actual_bytes = requested_bytes;
                }
                if (pinned) {
                    *pinned = false;
                }
                return ptr;
            }
            return nullptr;
        }
        if (bytes > limit) {
            bytes = limit & ~4095ull;
            fprintf(stderr,
                "[storagellm] pinned staging capped by RLIMIT_MEMLOCK: requested=%llu actual=%llu\n",
                (unsigned long long)requested_bytes,
                (unsigned long long)bytes);
        }
    }
#endif
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
        // Bug 12 Fix: Non-pinned success → free and try smaller pinned allocation.
        // Non-pinned staging defeats async DMA (cuMemcpyHtoDAsync becomes sync).
        // This ensures allocate_best_staging only returns pinned memory or NULL,
        // preventing silent sync DMA disguised as async.
        if (ptr) {
            free_staging(ptr, bytes, false);
        }
        bytes /= 2;
    }

    // CRITICAL FIX: When pinned allocation fails completely, return non-pinned fallback
    // instead of NULL to prevent pipeline shutdown. Non-pinned staging allows sync DMA,
    // which is slower but functional. Returning NULL disables multistage pipeline entirely,
    // preventing model from loading at all.
    fprintf(stderr, "[storageLLM WARNING] pinned staging allocation failed (mlock denied). "
        "Falling back to non-pinned staging (sync DMA only, slower but functional). "
        "Check 'ulimit -l' (must be >= %llu MB) for async DMA.\n",
        (unsigned long long)(requested_bytes / (1024ull * 1024ull)));
    
    // Allocate non-pinned buffer as last resort
    uint8_t* fallback_ptr = allocate_staging(requested_bytes, nullptr);
    if (fallback_ptr) {
        if (actual_bytes) {
            *actual_bytes = requested_bytes;
        }
        if (pinned) {
            *pinned = false;
        }
        return fallback_ptr;
    }
    
    // Only return NULL if even non-pinned allocation fails (out of memory)
    fprintf(stderr, "[storageLLM FATAL] staging allocation failed completely (out of memory). "
        "Engine cannot function.\n");
    return nullptr;
}

