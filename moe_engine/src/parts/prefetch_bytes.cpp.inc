static uint64_t tensor_stream_bytes(const moe_tensor_record& rec) {
    return rec.info.weight_byte_length + rec.info.scale_byte_length + rec.info.scale2_byte_length;
}

static uint64_t expert_triplet_bytes(const moe_pc_engine* engine, int layer, int expert) {
    const uint64_t cached = expert_bytes_get(engine, layer, expert);
    if (cached) {
        return cached;
    }
    uint64_t bytes = 0;
    const moe_projection_t projs[3] = {moe_PROJ_GATE, moe_PROJ_UP, moe_PROJ_DOWN};
    for (moe_projection_t proj : projs) {
        const moe_tensor_record* rec = find_tensor_record(engine, layer, expert, proj);
        if (rec) {
            bytes += tensor_stream_bytes(*rec);
        }
    }
    return bytes;
}

// TODO: Issue 8 - Windows PrefetchVirtualMemory ineffective.
// Problem: PrefetchVirtualMemory on some Windows versions is no-op
// GetCurrentProcess() handle doesn't work reliably for prefetch
// Page faults still occur at DMA time, blocking IO workers
// Solution options:
// 1. FILE_FLAG_NO_BUFFERING + separate IO thread for async read
// 2. CreateFile + ReadFileEx with overlapped I/O
// 3. Use OpenProcess(PROCESS_ALL_ACCESS, FALSE, GetCurrentProcessId())
// Status: Documented as TODO, requires Windows-specific testing
static void touch_mapped_range(const moe_pc_engine* engine, const uint8_t* src, uint64_t bytes) {
#ifdef _WIN32
    if (engine && engine->io_config.enable_directstorage) {
        return; // DirectStorage가 prefetch를 처리하므로 불필요
    }
#endif
    if (!src || bytes == 0) {
        return;
    }
    // PIPELINE FIX 1: Remove synchronous page scan loop.
    // Old code: 5,000 page accesses for 20MB tensor ??io_worker blocks on page faults
    // New code: OS async hint only ??io_worker continues immediately
    // Trade-off: Some page faults may occur at memcpy time, but MADV_WILLNEED
    // usually completes before next task starts. Non-blocking is better.
    // BUGFIX 78: Issue 8 - Windows PrefetchVirtualMemory ineffective.
    // Problem: GetCurrentProcess() pseudo-handle doesn't always work with PrefetchVirtualMemory
    // Solution: Use OpenProcess to get a real handle to the current process
    // BUGFIX 82: Issue 4 - Skip Windows prefetch when DirectStorage enabled.
    // DirectStorage handles prefetch internally, PrefetchVirtualMemory is redundant
#ifdef _WIN32
    // Note: DirectStorage check would go here if engine pointer was available
    // For now, we keep the improved OpenProcess approach
    WIN32_MEMORY_RANGE_ENTRY entry;
    entry.VirtualAddress = const_cast<uint8_t*>(src);
    entry.NumberOfBytes = (SIZE_T)bytes;
    HANDLE hProcess = OpenProcess(PROCESS_ALL_ACCESS, FALSE, GetCurrentProcessId());
    BOOL success = FALSE;
    if (hProcess) {
        success = PrefetchVirtualMemory(hProcess, 1, &entry, 0);
        CloseHandle(hProcess);
    } else {
        success = PrefetchVirtualMemory(GetCurrentProcess(), 1, &entry, 0);
    }
    if (!success && engine) {
        // Cast away const to update atomic metrics
        const_cast<moe_pc_engine*>(engine)->io_atomic.bytes_touch_fallback.fetch_add(bytes, std::memory_order_relaxed);
    }
#elif defined(__linux__)
    madvise(const_cast<uint8_t*>(src), (size_t)bytes, MADV_WILLNEED);
#endif
    // Removed: volatile uint8_t sink loop that caused blocking page faults
}

static uint32_t acquire_staging_slot_internal(moe_pc_engine* engine, uint32_t start_idx, uint32_t count, uint32_t worker_index, uint32_t turn, uint64_t required_bytes);

static uint32_t acquire_staging_slot(
    moe_pc_engine* engine,
    uint32_t worker_index,
    uint32_t turn,
    uint64_t required_bytes,
    bool for_pinned = true,
    bool for_direct = false
) {
    if (!engine || engine->staging_slots.empty()) {
        return UINT32_MAX;
    }
    const uint32_t total_count = static_cast<uint32_t>(engine->staging_slots.size());
    const uint32_t pinned_slot_count = engine->staging_pinned_slot_count.load(std::memory_order_acquire);
    const uint32_t gpu_slot_count = engine->staging_gpu_slot_count.load(std::memory_order_acquire);
    const uint32_t direct_slot_count = engine->staging_direct_slot_count.load(std::memory_order_acquire);
    uint32_t pinned_count = std::min(pinned_slot_count, total_count);
    uint32_t gpu_count = std::min(gpu_slot_count, total_count - pinned_count);
    uint32_t direct_count = std::min(direct_slot_count, total_count - pinned_count - gpu_count);
    if (pinned_count + gpu_count + direct_count < total_count) {
        const uint32_t spare = total_count - pinned_count - gpu_count - direct_count;
        if (direct_count > 0 || for_direct) {
            direct_count += spare;
        } else if (gpu_count > 0 || !for_pinned) {
            gpu_count += spare;
        } else {
            pinned_count += spare;
        }
    }
    const uint32_t gpu_start = pinned_count;
    const uint32_t direct_start = pinned_count + gpu_count;
    uint32_t pref_start = 0;
    uint32_t pref_count = 0;
    if (for_direct) {
        pref_start = direct_start;
        pref_count = direct_count;
    } else if (for_pinned) {
        pref_start = 0;
        pref_count = pinned_count;
    } else {
        pref_start = gpu_start;
        pref_count = gpu_count;
    }
    if (pref_count == 0) {
        return UINT32_MAX;
    }
    return acquire_staging_slot_internal(engine, pref_start, pref_count, worker_index, turn, required_bytes);
}

static uint32_t acquire_staging_slot_internal(moe_pc_engine* engine, uint32_t start_idx, uint32_t count, uint32_t worker_index, uint32_t turn, uint64_t required_bytes) {
    const uint32_t preferred = count > 1 ? (worker_index * 2u + turn) % count : 0u;
    for (;;) {
        for (uint32_t probe = 0; probe < count; ++probe) {
            const uint32_t idx = start_idx + ((preferred + probe) % count);
            bool expected = false;
            if (engine->staging_slots[idx].in_use.compare_exchange_strong(
                    expected,
                    true,
                    std::memory_order_acquire,
                    std::memory_order_relaxed)) {
                if (required_bytes > 0 && engine->staging_slots[idx].size < required_bytes) {
                    engine->staging_slots[idx].in_use.store(false, std::memory_order_release);
                    continue;
                }
                engine->staging_free_slot_count.fetch_sub(1, std::memory_order_relaxed);
                return idx;
            }
        }
        if (engine->io_stop.load()) {
            return UINT32_MAX;
        }

        // BUGFIX 68: Issue 5 - Wait on correct CV based on slot type ??
        // Old: All workers wait on staging_slot_cv ??Thundering Herd
        // New: Wait on type-specific CV ??only relevant workers wake
        std::condition_variable* target_cv = &engine->staging_slot_cv;
        const uint32_t pinned_count = engine->staging_pinned_slot_count.load(std::memory_order_acquire);
        const uint32_t gpu_start = pinned_count;
        const uint32_t gpu_count = engine->staging_gpu_slot_count.load(std::memory_order_acquire);
        const uint32_t direct_start = pinned_count + gpu_count;

        if (start_idx < pinned_count) {
            target_cv = &engine->staging_pinned_cv;
        } else if (start_idx >= pinned_count && start_idx < direct_start) {
            target_cv = &engine->staging_gpu_cv;
        } else {
            target_cv = &engine->staging_direct_cv;
        }

        std::unique_lock<std::mutex> lock(engine->staging_slot_mutex);
        // Fix 4: Use predicate wait to eliminate polling delay.
        // Wake immediately when a slot is freed, timeout after 5ms max.
        const bool got_slot = target_cv->wait_for(
            lock, std::chrono::milliseconds(5),
            [&]() { return engine->staging_free_slot_count.load(std::memory_order_acquire) > 0; }
        );
        if (!got_slot) {
            return UINT32_MAX;
        }
    }
}

static void release_staging_slot(moe_pc_engine* engine, uint32_t slot_index) {
    if (engine && slot_index < engine->staging_slots.size()) {
        engine->staging_slots[slot_index].in_use.store(false, std::memory_order_release);
        engine->staging_free_slot_count.fetch_add(1, std::memory_order_release);

        // BUGFIX 68: Issue 5 - Notify only relevant CV based on slot type ??
        // Old: notify_one on staging_slot_cv ??random worker wakes (may be wrong type)
        // New: Notify type-specific CV ??correct worker type wakes
        // Impact: Eliminates unnecessary wakeups, reduces context switches
        const uint32_t pinned_count = engine->staging_pinned_slot_count.load(std::memory_order_acquire);
        const uint32_t gpu_count = engine->staging_gpu_slot_count.load(std::memory_order_acquire);
        const uint32_t gpu_start = pinned_count;
        const uint32_t direct_start = pinned_count + gpu_count;

        if (slot_index < pinned_count) {
            engine->staging_pinned_cv.notify_one();
        } else if (slot_index >= pinned_count && slot_index < direct_start) {
            engine->staging_gpu_cv.notify_one();
        } else {
            engine->staging_direct_cv.notify_one();
        }

        // Also notify general CV for backward compatibility
        engine->staging_slot_cv.notify_one();
    }
}

static uint64_t copy_mapped_range_to_staging(
    moe_pc_engine* engine,
    const uint8_t* src,
    uint64_t bytes,
    uint32_t slot_index
) {
    if (!engine || !src || bytes == 0 || slot_index >= engine->staging_slots.size()) {
        // Problem 5 Fix: Separate transient contention from structural deficit
        // staging_slot_contention = can't acquire slot (transient, retry will succeed)
        // staging_wraps is kept for backward compatibility but now tracks contention
        if (engine && slot_index == UINT32_MAX) {
            engine->io_atomic.staging_slot_contention.fetch_add(1);
            engine->io_atomic.staging_wraps.fetch_add(1);
        }
        return 0;
    }
    moe_staging_slot& slot = engine->staging_slots[slot_index];
    if (!slot.buf || slot.size == 0 || bytes > slot.size) {
        // Problem 5 Fix: This is also transient contention (slot too small for this tensor)
        engine->io_atomic.staging_slot_contention.fetch_add(1);
        engine->io_atomic.staging_wraps.fetch_add(1);
        return 0;
    }

    if (bytes >= 1024ull * 1024ull) {
#if defined(__AVX2__)
        if (reinterpret_cast<uintptr_t>(slot.buf) % 32 == 0) {
            uint8_t* d = static_cast<uint8_t*>(slot.buf);
            const uint8_t* s = static_cast<const uint8_t*>(src);
            size_t i = 0;
            for (; i + 127 < static_cast<size_t>(bytes); i += 128) {
                __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + i));
                __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + i + 32));
                __m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + i + 64));
                __m256i v3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + i + 96));
                _mm256_stream_si256(reinterpret_cast<__m256i*>(d + i), v0);
                _mm256_stream_si256(reinterpret_cast<__m256i*>(d + i + 32), v1);
                _mm256_stream_si256(reinterpret_cast<__m256i*>(d + i + 64), v2);
                _mm256_stream_si256(reinterpret_cast<__m256i*>(d + i + 96), v3);
            }
            _mm_sfence();
            if (i < static_cast<size_t>(bytes)) {
                std::memcpy(d + i, s + i, static_cast<size_t>(bytes) - i);
            }
        } else {
            std::memcpy(slot.buf, src, static_cast<size_t>(bytes));
        }
#else
        std::memcpy(slot.buf, src, static_cast<size_t>(bytes));
#endif
    } else {
        std::memcpy(slot.buf, src, static_cast<size_t>(bytes));
    }
    return bytes;
}

static uint64_t copy_mapped_range_to_staging_linear(
    moe_pc_engine* engine,
    const uint8_t* src,
    uint64_t bytes,
    uint32_t slot_index,
    uint64_t* staging_offset
) {
    if (!engine || !src || bytes == 0 || slot_index >= engine->staging_slots.size() || !staging_offset) {
        // Problem 5 Fix: Separate transient contention from structural deficit
        if (engine && slot_index == UINT32_MAX) {
            engine->io_atomic.staging_slot_contention.fetch_add(1);
            engine->io_atomic.staging_wraps.fetch_add(1);
        }
        return 0;
    }
    moe_staging_slot& slot = engine->staging_slots[slot_index];
    if (!slot.buf || *staging_offset > slot.size || bytes > slot.size - *staging_offset) {
        // Problem 5 Fix: This is also transient contention
        engine->io_atomic.staging_slot_contention.fetch_add(1);
        engine->io_atomic.staging_wraps.fetch_add(1);
        return 0;
    }
    if (bytes >= 1024ull * 1024ull) {
#if defined(__AVX2__)
        if (reinterpret_cast<uintptr_t>(slot.buf + *staging_offset) % 32 == 0) {
            uint8_t* d = static_cast<uint8_t*>(slot.buf + *staging_offset);
            const uint8_t* s = static_cast<const uint8_t*>(src);
            size_t i = 0;
            for (; i + 127 < static_cast<size_t>(bytes); i += 128) {
                __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + i));
                __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + i + 32));
                __m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + i + 64));
                __m256i v3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + i + 96));
                _mm256_stream_si256(reinterpret_cast<__m256i*>(d + i), v0);
                _mm256_stream_si256(reinterpret_cast<__m256i*>(d + i + 32), v1);
                _mm256_stream_si256(reinterpret_cast<__m256i*>(d + i + 64), v2);
                _mm256_stream_si256(reinterpret_cast<__m256i*>(d + i + 96), v3);
            }
            _mm_sfence();
            if (i < static_cast<size_t>(bytes)) {
                std::memcpy(d + i, s + i, static_cast<size_t>(bytes) - i);
            }
        } else {
            std::memcpy(slot.buf + *staging_offset, src, static_cast<size_t>(bytes));
        }
#else
        std::memcpy(slot.buf + *staging_offset, src, static_cast<size_t>(bytes));
#endif
    } else {
        std::memcpy(slot.buf + *staging_offset, src, static_cast<size_t>(bytes));
    }
    *staging_offset += bytes;
    return bytes;
}
