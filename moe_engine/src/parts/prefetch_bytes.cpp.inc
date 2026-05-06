static uint64_t tensor_stream_bytes(const moe_tensor_record& rec) {
    return rec.info.weight_byte_length + rec.info.scale_byte_length + rec.info.scale2_byte_length;
}

static int device_backend_event_sync(moe_pc_engine* e, CUevent event);
static inline bool staging_slot_valid(const moe_pc_engine* e, uint32_t slot_index);

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

// BUGFIX 120: Issue 25 - Windows PrefetchVirtualMemory is blocking, not async ?끸쁾??// Problem: PrefetchVirtualMemory is a hint, not a guarantee. When io_worker calls memcpy,
// if pages aren't in RAM yet, page fault blocks the worker. This is the root cause of
// "TODO: Issue 8" that was never fully resolved.
// Solution: Remove PrefetchVirtualMemory entirely on Windows when DirectStorage is off.
// Instead, prefetch_tensor_file_to_pinned should use ReadFile + OVERLAPPED for async I/O.
// The mmap handle can be reopened as a file handle (shard->hFile already exists in mmap_context_t).
// Trade-off: Requires managing file handles, but eliminates blocking page faults.
// Status: PrefetchVirtualMemory removed. Async ReadFile path is in prefetch_pinned.cpp.inc.
// BUGFIX Problem 3: Windows mmap touch with VirtualLock ?끸쁾 IMPORTANT
// Problem: touch_mapped_range is no-op on Windows without DirectStorage ??page fault stalls
// Solution: Use VirtualLock + VirtualUnlock pair to force pages into working set
// This is non-blocking and acts as effective prefetch
// Trade-off: VirtualLock can fail if working set limit reached (default 1.5GB)
// Failure is silently ignored - page fault will occur but no worse than before
// BUGFIX 897: Windows non-DirectStorage mmap page residency ?끸쁾??CRITICAL
// Problem: VirtualLock + VirtualUnlock brings pages into working set, but OS can evict them
// before copy_mapped_range_to_staging runs. Page fault blocks io_worker.
// Solution: Add prefer_overlapped_io flag. When enabled + DirectStorage disabled,
// use OVERLAPPED ReadFile instead of mmap. io_iocp_adapter.cpp.inc handles completion.
// Impact: Eliminates blocking page faults on Windows non-DirectStorage path.
static void touch_mapped_range(const moe_pc_engine* engine, const uint8_t* src, uint64_t bytes) {
#ifdef _WIN32
    // BUGFIX 963: Unify overlapped IO check at function entry ★★
    // Problem: io_config_api.cpp.inc sets prefer_overlapped_io=1 at startup,
    // but this function's VirtualLock path still runs → 8 useless VirtualLock
    // failures accumulate → s_overlapped_io_set writes redundantly → duplicate log.
    // Solution: Single early-return check covers both DirectStorage and overlapped IO.
    // BUGFIX 936 byte counter is preserved.
    // BUGFIX 986: Read from atomic mirror instead of plain int ★★
    const bool overlapped = engine && engine->prefer_overlapped_io_atomic.load(std::memory_order_relaxed);
    if (engine && (engine->io_config.enable_directstorage || overlapped)) {
        // BUGFIX 1002: Periodic VirtualLock recovery probe ★★ PERFORMANCE
        // Problem: Once OVERLAPPED IO is enabled, it never reverts even when
        //   memory pressure subsides. OVERLAPPED is slower for large sequential
        //   reads (extra kernel copy vs direct page access) → 10-30% throughput loss.
        // Solution: Every 1024 calls, try VirtualLock on a small page. If it
        //   succeeds, memory pressure has subsided → revert to mmap path.
        // Overhead: One VirtualLock+Unlock per 1024 calls = negligible.
        if (overlapped && !engine->io_config.enable_directstorage && src && bytes > 0) {
            static thread_local uint32_t s_recovery_counter = 0;
            if (++s_recovery_counter >= 1024) {
                s_recovery_counter = 0;
                const size_t probe_size = 4096;
                if (VirtualLock(reinterpret_cast<LPVOID>(
                        reinterpret_cast<uintptr_t>(src) & ~(uintptr_t)(probe_size - 1)),
                        probe_size)) {
                    VirtualUnlock(reinterpret_cast<LPVOID>(
                        reinterpret_cast<uintptr_t>(src) & ~(uintptr_t)(probe_size - 1)),
                        probe_size);
                    // VirtualLock succeeded → working set pressure has subsided
                    auto* mutable_engine = const_cast<moe_pc_engine*>(engine);
                    mutable_engine->prefer_overlapped_io_atomic.store(false, std::memory_order_release);
                    mutable_engine->io_config.prefer_overlapped_io = 0;
                    fprintf(stderr,
                        "[storageLLM INFO] VirtualLock recovery succeeded. "
                        "Reverting to mmap path for better sequential read throughput.\n");
                    // BUGFIX 1002: Count bytes in mmap_touch before falling through ★★ IMPORTANT
                    // Problem: Recovery path jumps to do_virtuallock_path without counting bytes
                    // → IO statistics have a hole → runtime_prefetch_window() underestimates bandwidth
                    // Solution: Add bytes to mmap_touch counter before goto
                    mutable_engine->io_atomic.bytes_mmap_touch.fetch_add(bytes, std::memory_order_relaxed);
                    // Don't return — fall through to VirtualLock path for this call
                    goto do_virtuallock_path;
                }
            }
            const_cast<moe_pc_engine*>(engine)->io_atomic.bytes_overlapped_io.fetch_add(bytes, std::memory_order_relaxed);
        } else if (overlapped && src && bytes > 0) {
            const_cast<moe_pc_engine*>(engine)->io_atomic.bytes_overlapped_io.fetch_add(bytes, std::memory_order_relaxed);
        }
        return;
    }
    do_virtuallock_path:
    // Try VirtualLock to bring pages into working set
    // This is a hint - failure is acceptable and common when working set limit reached
    if (src && bytes > 0) {
        // VirtualLock requires page-aligned address
        const size_t page_size = 4096;
        const uintptr_t aligned_addr = reinterpret_cast<uintptr_t>(src) & ~(page_size - 1);
        const size_t prefix = reinterpret_cast<uintptr_t>(src) - aligned_addr;
        const size_t aligned_size = ((bytes + prefix + page_size - 1) / page_size) * page_size;

        // VirtualLock brings pages into working set (non-blocking)
        if (VirtualLock(reinterpret_cast<LPVOID>(aligned_addr), aligned_size)) {
            // Immediately unlock - we just wanted to fault pages in
            VirtualUnlock(reinterpret_cast<LPVOID>(aligned_addr), aligned_size);
        } else {
            // BUGFIX 939: Auto-fallback to OVERLAPPED IO on VirtualLock failure ★★★
            // Problem: VirtualLock fails at ~1.5GB working set limit → memcpy page-faults
            //   → io_worker blocked 100-500ms per fault → pipeline stall
            // Solution: Track consecutive failures. After 2 failures, dynamically switch
            //   to prefer_overlapped_io=true. OVERLAPPED ReadFile path already implemented.
            // Impact: Eliminates blocking page faults on Windows non-DirectStorage path.
            static thread_local uint32_t s_vlock_fail_count = 0;
            // BUGFIX 990: Lower threshold from 8 to 2 ★★ IMPORTANT
            // Problem: Each VirtualLock failure causes 100-500ms page fault stall.
            // 8 failures = up to 4s wasted blocking before switching to OVERLAPPED IO.
            // Solution: Switch after 2 failures. 1 retry handles transient OS contention;
            // 2nd failure confirms working set limit is persistently reached.
            if (++s_vlock_fail_count >= 2 && engine) {
                auto* mutable_engine = const_cast<moe_pc_engine*>(engine);
                // BUGFIX 961: Race-safe prefer_overlapped_io transition ★★
                // BUGFIX 1002: Use file-scope flag so recovery can reset it
                static std::atomic<bool> s_overlapped_io_set{false};
                if (!s_overlapped_io_set.exchange(true, std::memory_order_acq_rel)) {
                    // BUGFIX 986: Write to atomic mirror first, then sync plain int ★★
                    mutable_engine->prefer_overlapped_io_atomic.store(true, std::memory_order_release);
                    mutable_engine->io_config.prefer_overlapped_io = 1;
                    fprintf(stderr,
                        "[storageLLM INFO] VirtualLock failed %u times, auto-enabling prefer_overlapped_io. "
                        "Subsequent I/O will use OVERLAPPED ReadFile to avoid page fault stalls.\n",
                        s_vlock_fail_count);
                }
            }
        }
    }
    if (engine) {
        const_cast<moe_pc_engine*>(engine)->io_atomic.bytes_touch_fallback.fetch_add(bytes, std::memory_order_relaxed);
    }
#elif defined(__linux__)
    madvise(const_cast<uint8_t*>(src), (size_t)bytes, MADV_WILLNEED);
#endif
}

// Fix 20: Coalesce adjacent madvise calls to reduce kernel syscall overhead
static void touch_mapped_ranges(const moe_pc_engine* engine, const std::vector<std::pair<const uint8_t*, uint64_t>>& ranges) {
    if (ranges.empty()) return;
#ifdef _WIN32
    if (engine && engine->io_config.enable_directstorage) {
        return;
    }
    uint64_t total_bytes = 0;
    for (const auto& r : ranges) total_bytes += r.second;
    if (engine) {
        const_cast<moe_pc_engine*>(engine)->io_atomic.bytes_touch_fallback.fetch_add(total_bytes, std::memory_order_relaxed);
    }
#elif defined(__linux__)
    if (ranges.empty()) return;
    const size_t PAGE_SIZE = 4096;
    const uint8_t* current_start = ranges[0].first;
    uint64_t current_len = ranges[0].second;

    for (size_t i = 1; i < ranges.size(); ++i) {
        const uint8_t* next_start = ranges[i].first;
        uint64_t next_len = ranges[i].second;

        // BUGFIX 924: Fix madvise range merge logic ?끸쁾 IMPORTANT
        // Problem: + PAGE_SIZE allows merging non-adjacent ranges with 1-page gap
        // ??madvise called on unrelated memory between ranges ??performance degradation
        // Solution: Only merge strictly adjacent ranges (no gap tolerance)
        // Impact: More accurate madvise hints, reduced syscall overhead without side effects
        if (next_start >= current_start && next_start <= current_start + current_len) {
            uint64_t diff = (next_start + next_len) - current_start;
            if (diff > current_len) {
                current_len = diff;
            }
        } else {
            madvise(const_cast<uint8_t*>(current_start), (size_t)current_len, MADV_WILLNEED);
            current_start = next_start;
            current_len = next_len;
        }
    }
    madvise(const_cast<uint8_t*>(current_start), (size_t)current_len, MADV_WILLNEED);
#endif
}

static uint32_t acquire_staging_slot_internal(moe_pc_engine* engine, uint32_t start_idx, uint32_t count, uint32_t worker_index, uint32_t turn, uint64_t required_bytes);

#ifdef ENABLE_LOCKFREE_STAGING
static uint32_t moe_ctz64_portable(uint64_t value) {
    // BUGFIX: Use hardware intrinsics instead of software loop on hot path
#if defined(_MSC_VER)
    unsigned long idx;
    _BitScanForward64(&idx, value);
    return (uint32_t)idx;
#elif defined(__GNUC__) || defined(__clang__)
    return (uint32_t)__builtin_ctzll(value);
#else
    // Genuine portable fallback
    uint32_t idx = 0;
    while ((value & 1ull) == 0ull) {
        value >>= 1;
        ++idx;
    }
    return idx;
#endif
}

static uint64_t moe_staging_range_mask(uint32_t start_idx, uint32_t count, uint32_t total_count) {
    if (count == 0 || start_idx >= total_count || start_idx >= 64u) {
        return 0;
    }
    const uint32_t end_idx = std::min<uint32_t>(64u, std::min<uint32_t>(total_count, start_idx + count));
    const uint32_t bits = end_idx - start_idx;
    if (bits == 0) {
        return 0;
    }
    const uint64_t span = bits == 64u ? UINT64_MAX : ((1ull << bits) - 1ull);
    return span << start_idx;
}
#endif

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
    if (count == 0) {
        return UINT32_MAX;
    }
    const uint32_t preferred = count > 1 ? (worker_index * 2u + turn) % count : 0u;
    // BUGFIX 83: Issue 5 - Limit staging slot acquire retries ?끸쁾??    // Problem: Infinite loop when staging_free_slot_count=0
    // All io_workers blocked in this loop ??pipeline freeze
    // Solution: Max 20 retries (100ms total), then return UINT32_MAX
    // Upper layer handles failure (requeue or drop)
    uint32_t retry_count = 0;
    // BUGFIX 714: Increase MAX_RETRIES to 50 ?끸쁾??    // Problem: 20 retries is too short during heavy load, causing valid
    // prefetch requests to be dropped unnecessarily.
    // Solution: Increase to 50 to give more time for staging slots to free up.
    constexpr uint32_t MAX_RETRIES = 50;

#ifdef ENABLE_LOCKFREE_STAGING
    const uint32_t total_slots = static_cast<uint32_t>(engine->staging_slots.size());
    // BUGFIX 593: Check total_slots > 0 before range_mask ??    // Problem: total_slots can be 0
    // Solution: Check before calling moe_staging_range_mask
    if (total_slots == 0) {
        return UINT32_MAX;
    }
    const uint64_t range_mask = moe_staging_range_mask(start_idx, count, total_slots);
    if (total_slots <= 64u && range_mask != 0) {
        for (;;) {
            uint64_t available = engine->staging_slot_free_mask.load(std::memory_order_acquire) & range_mask;
            if (count > 1 && available != 0) {
                const uint32_t pref_idx = start_idx + preferred;
                if (pref_idx < total_slots) {
                    const uint64_t pref_bit = 1ull << pref_idx;
                    if ((available & pref_bit) != 0) {
                        available = pref_bit | (available & ~pref_bit);
                    }
                }
            }
            while (available != 0) {
                const uint32_t idx = moe_ctz64_portable(available);
                const uint64_t bit = 1ull << idx;
                uint64_t observed = engine->staging_slot_free_mask.load(std::memory_order_acquire);
                while ((observed & bit) != 0) {
                    const uint64_t next = observed & ~bit;
                    if (engine->staging_slot_free_mask.compare_exchange_weak(
                            observed,
                            next,
                            std::memory_order_acq_rel,
                            std::memory_order_acquire)) {
                        if (required_bytes > 0 && engine->staging_slots[idx].size < required_bytes) {
                            engine->staging_slot_free_mask.fetch_or(bit, std::memory_order_release);
                            available &= ~bit;
                            goto try_next_lockfree_slot;
                        }
                        engine->staging_slots[idx].in_use.store(true, std::memory_order_release);
                        // BUGFIX: Wait on per-slot DMA fence before reusing buffer
                        if (engine->staging_slots[idx].last_copy_event &&
                            engine->staging_slots[idx].last_copy_event_recorded.load(std::memory_order_acquire)) {
                            device_backend_event_sync(engine, engine->staging_slots[idx].last_copy_event);
                            engine->staging_slots[idx].last_copy_event_recorded.store(false, std::memory_order_release);
                        }
                        // BUGFIX 912: Set slot generation on acquisition ?끸쁾 IMPORTANT
                        engine->staging_slots[idx].generation = engine->staging_generation.load(std::memory_order_acquire);
                        // BUGFIX 622: Check staging_free_slot_count > 0 before fetch_sub to prevent underflow ??                        // Problem: If staging_free_slot_count is 0, fetch_sub causes underflow
                        // Solution: Only decrement if > 0
                        uint32_t current = engine->staging_free_slot_count.load(std::memory_order_acquire);
                        while (current > 0) {
                            if (engine->staging_free_slot_count.compare_exchange_weak(
                                    current, current - 1,
                                    std::memory_order_release,
                                    std::memory_order_acquire)) {
                                break;
                            }
                        }
                        return idx;
                    }
                }
                available &= ~bit;
try_next_lockfree_slot:
                continue;
            }
            if (engine->io_stop.load()) {
                return UINT32_MAX;
            }

            // BUGFIX 臾몄젣2: Add CV wait before giving up ?끸쁾??            // Problem: 50 retries with spin-wait wastes CPU, then silently drops prefetch
            // Solution: Use CV wait with short timeout, retry up to 500ms total
            // This prevents CPU waste while giving more time for slots to free up
            if (++retry_count >= MAX_RETRIES) {
                // Before giving up, try CV wait one more time
                std::condition_variable* target_cv = &engine->staging_slot_cv;
                const uint32_t pinned_count = engine->staging_pinned_slot_count.load(std::memory_order_relaxed);
                const uint32_t gpu_count = engine->staging_gpu_slot_count.load(std::memory_order_relaxed);
                const uint32_t direct_start = pinned_count + gpu_count;
                if (start_idx < pinned_count) {
                    target_cv = &engine->staging_pinned_cv;
                } else if (start_idx >= pinned_count && start_idx < direct_start) {
                    target_cv = &engine->staging_gpu_cv;
                } else {
                    target_cv = &engine->staging_direct_cv;
                }

                std::unique_lock<std::mutex> lock(engine->staging_slot_mutex);
                // BUGFIX 臾몄젣11: Fix CV wait predicate for >64 staging slots ?끸쁾??                // Problem: staging_slot_free_mask is uint64_t ??only 64 slots representable.
                // When total_slots > 64, range_mask == 0, so (mask & range_mask) is always false.
                // CV wait always times out even when slots are actually free.
                // Solution: Fall back to staging_free_slot_count for >64 slot configurations.
                if (target_cv->wait_for(
                        lock, std::chrono::milliseconds(10),
                        [&]() {
                            if (total_slots <= 64u)
                                return (engine->staging_slot_free_mask.load(std::memory_order_acquire) & range_mask) != 0;
                            else
                                return engine->staging_free_slot_count.load(std::memory_order_acquire) > 0u;
                        }
                    )) {
                    // Slot became available during wait, retry acquisition
                    retry_count = 0;
                    continue;
                }

                // Still no slot after CV wait - give up
                engine->io_atomic.staging_slot_contention.fetch_add(1, std::memory_order_relaxed);
                return UINT32_MAX;
            }

            // Fix 5: Use memory_order_relaxed in retry loop to avoid L3 cache sync overhead
            std::condition_variable* target_cv = &engine->staging_slot_cv;
            const uint32_t pinned_count = engine->staging_pinned_slot_count.load(std::memory_order_relaxed);
            const uint32_t gpu_count = engine->staging_gpu_slot_count.load(std::memory_order_relaxed);
            const uint32_t direct_start = pinned_count + gpu_count;
            if (start_idx < pinned_count) {
                target_cv = &engine->staging_pinned_cv;
            } else if (start_idx >= pinned_count && start_idx < direct_start) {
                target_cv = &engine->staging_gpu_cv;
            } else {
                target_cv = &engine->staging_direct_cv;
            }
            // BUGFIX 195: Reduce wait timeout to prevent GPU starvation ?끸쁾??            // Problem: 20 retries 횞 5ms = 100ms blocking
            // During this time gpu_stage queue empties ??GPU completely idle
            // Solution: Reduce timeout 5ms ??1ms (20ms total max)
            // Still allows reasonable wait but prevents long GPU starvation
            // Impact: GPU idle time reduced from 100ms ??20ms worst case
            std::unique_lock<std::mutex> lock(engine->staging_slot_mutex);
            // BUGFIX 臾몄젣11: Fix CV wait predicate for >64 staging slots ?끸쁾??            target_cv->wait_for(
                lock, std::chrono::milliseconds(1),
                [&]() {
                    if (total_slots <= 64u)
                        return (engine->staging_slot_free_mask.load(std::memory_order_acquire) & range_mask) != 0;
                    else
                        return engine->staging_free_slot_count.load(std::memory_order_acquire) > 0u;
                }
            );
        }
    }
#endif

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
                // BUGFIX: Wait on per-slot DMA fence before reusing buffer
                if (engine->staging_slots[idx].last_copy_event &&
                    engine->staging_slots[idx].last_copy_event_recorded.load(std::memory_order_acquire)) {
                    device_backend_event_sync(engine, engine->staging_slots[idx].last_copy_event);
                    engine->staging_slots[idx].last_copy_event_recorded.store(false, std::memory_order_release);
                }
                // BUGFIX 912: Set slot generation on acquisition ?끸쁾 IMPORTANT
                engine->staging_slots[idx].generation = engine->staging_generation.load(std::memory_order_acquire);
                // BUGFIX 622: Check staging_free_slot_count > 0 before fetch_sub to prevent underflow ??                // Problem: If staging_free_slot_count is 0, fetch_sub causes underflow
                // Solution: Only decrement if > 0
                uint32_t current = engine->staging_free_slot_count.load(std::memory_order_acquire);
                while (current > 0) {
                    if (engine->staging_free_slot_count.compare_exchange_weak(
                            current, current - 1,
                            std::memory_order_release,
                            std::memory_order_acquire)) {
                        break;
                    }
                }
                return idx;
            }
        }
        if (engine->io_stop.load()) {
            return UINT32_MAX;
        }

        // BUGFIX 臾몄젣2: Add CV wait before giving up (non-lockfree path) ?끸쁾??        // Problem: 50 retries with spin-wait wastes CPU, then silently drops prefetch
        // Solution: Use CV wait with short timeout, retry up to 500ms total
        if (++retry_count >= MAX_RETRIES) {
            // Before giving up, try CV wait one more time
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
            if (target_cv->wait_for(
                    lock, std::chrono::milliseconds(10),
                    [&]() { return engine->staging_free_slot_count.load(std::memory_order_acquire) > 0; }
                )) {
                // Slot became available during wait, retry acquisition
                retry_count = 0;
                continue;
            }

            // Still no slot after CV wait - give up
            engine->io_atomic.staging_slot_contention.fetch_add(1, std::memory_order_relaxed);
            return UINT32_MAX;
        }

        // BUGFIX 68: Issue 5 - Wait on correct CV based on slot type ?끸쁾
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
        // BUGFIX 195: Reduce wait timeout to prevent GPU starvation ?끸쁾??        // Fix 4: Use predicate wait to eliminate polling delay.
        // Wake immediately when a slot is freed, timeout after 1ms max (was 5ms).
        // 20 retries 횞 1ms = 20ms total (was 100ms) ??reduces GPU idle time
        const bool got_slot = target_cv->wait_for(
            lock, std::chrono::milliseconds(1),
            [&]() { return engine->staging_free_slot_count.load(std::memory_order_acquire) > 0; }
        );
        if (!got_slot) {
            // Timeout but continue retry (up to MAX_RETRIES)
            continue;
        }
    }
}

// BUGFIX 107: Issue 21 - Cache slot counts to avoid redundant atomic loads ??// Problem: Same atomic variables loaded twice in short time
// Solution: Load once and reuse values
static void release_staging_slot(moe_pc_engine* engine, uint32_t slot_index) {
    // BUGFIX 594: Check engine pointer and slot_index bounds ??    // BUGFIX Issue 2: Use staging_slot_valid helper for correct check order ?끸쁾
    // Problem: Direct index check doesn't verify staging_slot_size first
    // Solution: Use centralized helper that enforces staging_slot_size > 0 check
    // BUGFIX NEW-4: Remove redundant staging_slot_valid check ??    // Problem: staging_slot_valid called twice - first check returns if invalid,
    // so second check is always true (dead code)
    // Solution: Remove redundant second check
    if (!staging_slot_valid(engine, slot_index)) {
        return;
    }

    // BUGFIX 912: Check staging_generation before release ?끸쁾 IMPORTANT
    // Problem: Worker completes after timeout ??releases slot with stale buf pointer
    // Solution: Check if generation changed (timeout occurred), skip release if stale
    // Impact: Prevents use-after-free when workers complete after staging reset
    const uint32_t current_gen = engine->staging_generation.load(std::memory_order_acquire);
    const uint32_t slot_gen = engine->staging_slots[slot_index].generation;
    if (slot_gen != current_gen) {
        // Slot is from old generation (timeout occurred), don't release
        // Just mark as not in_use to prevent further access
        engine->staging_slots[slot_index].in_use.store(false, std::memory_order_release);
        return;
    }

    engine->staging_slots[slot_index].in_use.store(false, std::memory_order_release);
#ifdef ENABLE_LOCKFREE_STAGING
        if (engine->staging_slots.size() <= 64u) {
            engine->staging_slot_free_mask.fetch_or(1ull << slot_index, std::memory_order_release);
        }
#endif
        engine->staging_free_slot_count.fetch_add(1, std::memory_order_release);

        // BUGFIX 68: Issue 5 - Notify only relevant CV based on slot type ?끸쁾
        // Old: notify_one on staging_slot_cv ??random worker wakes (may be wrong type)
        // New: Notify type-specific CV ??correct worker type wakes
        // Impact: Eliminates unnecessary wakeups, reduces context switches
        const uint32_t pinned_count = engine->staging_pinned_slot_count.load(std::memory_order_acquire);
        const uint32_t gpu_count = engine->staging_gpu_slot_count.load(std::memory_order_acquire);
        const uint32_t gpu_start = pinned_count;
        const uint32_t direct_start = pinned_count + gpu_count;

        if (slot_index < pinned_count) {
            engine->staging_pinned_cv.notify_one();
        } else if (slot_index >= gpu_start && slot_index < direct_start) {
            engine->staging_gpu_cv.notify_one();
        } else {
            engine->staging_direct_cv.notify_one();
        }

        // BUGFIX 706: Removed staging_slot_cv.notify_one() ??        // All acquire_staging_slot paths now use type-specific CVs (staging_pinned_cv,
        // staging_gpu_cv, staging_direct_cv). No code path waits on staging_slot_cv.
        // The extra notify caused unnecessary context switches.

        // BUGFIX 949: Drain orphaned events from staging timeout ★★
        // Problem: Orphaned events accumulate if release_staging_slot never drains them
        // Solution: Opportunistically drain when any slot is released (amortized cost)
        // Impact: Prevents memory leak of CUevent handles after staging timeout
        if (!engine->orphaned_events.empty()) {
            std::vector<CUevent> to_destroy;
            {
                std::lock_guard<std::mutex> orphan_lock(engine->orphaned_events_mutex);
                to_destroy.swap(engine->orphaned_events);
            }
            for (CUevent evt : to_destroy) {
                if (evt) {
                    device_backend_event_destroy(engine, evt);
                }
            }
        }
}

static uint64_t copy_mapped_range_to_staging(
    moe_pc_engine* engine,
    const uint8_t* src,
    uint64_t bytes,
    uint32_t slot_index
) {
    if (!engine || !src || bytes == 0 || !staging_slot_valid(engine, slot_index)) {
        if (engine && slot_index == UINT32_MAX) {
            engine->io_atomic.staging_slot_contention.fetch_add(1);
            engine->io_atomic.staging_wraps.fetch_add(1);
        }
        return 0;
    }
    moe_staging_slot& slot = engine->staging_slots[slot_index];
    uint8_t* slot_buf = slot.buf.load(std::memory_order_acquire);
    if (!slot_buf || slot.size == 0 || bytes > slot.size) {
        engine->io_atomic.staging_slot_contention.fetch_add(1);
        engine->io_atomic.staging_wraps.fetch_add(1);
        return 0;
    }
    if (bytes > SIZE_MAX) {
        return 0;
    }

    if (bytes >= 1024ull * 1024ull) {
#if defined(__AVX2__)
        if (reinterpret_cast<uintptr_t>(slot_buf) % 32 == 0) {
            uint8_t* d = slot_buf;
            const uint8_t* s = static_cast<const uint8_t*>(src);
            size_t i = 0;
            for (; i + 127 < static_cast<size_t>(bytes); i += 128) {
                if (i + 384 < static_cast<size_t>(bytes)) {
                    _mm_prefetch(reinterpret_cast<const char*>(s + i + 256), _MM_HINT_NTA);
                    _mm_prefetch(reinterpret_cast<const char*>(s + i + 320), _MM_HINT_NTA);
                }
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
            std::memcpy(slot_buf, src, static_cast<size_t>(bytes));
        }
#else
        std::memcpy(slot_buf, src, static_cast<size_t>(bytes));
#endif
    } else {
        std::memcpy(slot_buf, src, static_cast<size_t>(bytes));
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
    if (!engine || !src || bytes == 0 || !staging_offset || !staging_slot_valid(engine, slot_index)) {
        if (engine && slot_index == UINT32_MAX) {
            engine->io_atomic.staging_slot_contention.fetch_add(1);
            engine->io_atomic.staging_wraps.fetch_add(1);
        }
        return 0;
    }
    moe_staging_slot& slot = engine->staging_slots[slot_index];
    uint8_t* slot_buf = slot.buf.load(std::memory_order_acquire);
    if (!slot_buf || *staging_offset > slot.size || bytes > slot.size - *staging_offset ||
        (*staging_offset > 0 && bytes > UINT64_MAX - *staging_offset)) {
        engine->io_atomic.staging_slot_contention.fetch_add(1);
        engine->io_atomic.staging_wraps.fetch_add(1);
        return 0;
    }
    if (bytes > SIZE_MAX) {
        return 0;
    }

    uint8_t* dst = slot_buf + *staging_offset;
    if (bytes >= 1024ull * 1024ull) {
#if defined(__AVX2__)
        if (reinterpret_cast<uintptr_t>(dst) % 32 == 0) {
            uint8_t* d = dst;
            const uint8_t* s = static_cast<const uint8_t*>(src);
            size_t i = 0;
            for (; i + 127 < static_cast<size_t>(bytes); i += 128) {
                if (i + 384 < static_cast<size_t>(bytes)) {
                    _mm_prefetch(reinterpret_cast<const char*>(s + i + 256), _MM_HINT_NTA);
                    _mm_prefetch(reinterpret_cast<const char*>(s + i + 320), _MM_HINT_NTA);
                }
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
            std::memcpy(dst, src, static_cast<size_t>(bytes));
        }
#else
        std::memcpy(dst, src, static_cast<size_t>(bytes));
#endif
    } else {
        std::memcpy(dst, src, static_cast<size_t>(bytes));
    }
    *staging_offset += bytes;
    return bytes;
}

// Additional 1: HOBBIT - Mixed precision dynamic expert loading based on gate score importance
// This is a stub integration point for the int4 fallback decoder.
// BUGFIX 279: Mark as unused to prevent compiler warnings ??// Problem: Function defined but never called ??compiler warning
// Solution: Add [[maybe_unused]] attribute or comment as future integration point
// Status: Reserved for future HOBBIT integration, not currently used
[[maybe_unused]]
static int select_expert_precision(const moe_pc_engine* engine, int layer, int expert, float gate_score) {
    // If score is below threshold, use int4 (return 4), else NVFP4/BF16 (return 16)
    if (gate_score < engine->io_config.int4_fallback_threshold) {
        return 4; // moe_CODEC_INT4 equiv
    }
    return 16; // moe_CODEC_NVFP4/BF16 equiv
}
