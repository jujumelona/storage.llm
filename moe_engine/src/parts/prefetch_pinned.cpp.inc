static int read_file_range_to_buffer(
    std::ifstream& input,
    uint64_t file_offset,
    uint8_t* dst,
    uint64_t bytes
) {
    if (!input || !dst || bytes == 0 ||
        file_offset > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
        return 0;
    }
    input.clear();
    input.seekg(static_cast<std::streamoff>(file_offset), std::ios::beg);
    if (!input) {
        return 0;
    }
    uint64_t copied = 0;
    while (copied < bytes) {
        const uint64_t remaining = bytes - copied;
        const std::streamsize chunk = static_cast<std::streamsize>(
            std::min<uint64_t>(remaining, 64ull * 1024ull * 1024ull));
        input.read(reinterpret_cast<char*>(dst + copied), chunk);
        // BUGFIX 258: Check for EOF and partial reads ★★
        // Problem: gcount() != chunk treated as error, but EOF with partial read is valid
        // Solution: Check EOF separately, allow partial read at end of file
        const std::streamsize actual = input.gcount();
        if (actual == 0) {
            // No data read - check if EOF or error
            if (input.eof()) {
                // EOF reached but still have bytes to read - file too short
                return 0;
            }
            // Read error
            return 0;
        }
        if (actual != chunk) {
            // Partial read - check if EOF
            if (!input.eof()) {
                // Partial read without EOF - error
                return 0;
            }
            // EOF with partial read - check if we got all requested bytes
            if (copied + static_cast<uint64_t>(actual) != bytes) {
                // File too short
                return 0;
            }
        }
        copied += static_cast<uint64_t>(actual);
    }
    return 1;
}

// BUGFIX 51: Issue 22 - Cache ifstream per worker thread ?�★
// Problem: prefetch_tensor_file_to_pinned opens/closes ifstream for every tensor.
// 8 experts × 3 projections = 24 file opens per MoE layer. Same shard file reopened repeatedly.
// Each open requires seekg(), and non-mmap path uses different OS cache than mmap path.
// Solution: Use thread_local ifstream cache keyed by path. Reuse open streams when path matches.
// Fall back to mmap path when available (more efficient, shares OS page cache).
static int prefetch_tensor_file_to_pinned(
    moe_pc_engine* engine,
    const moe_tensor_record& rec,
    uint32_t slot_index,
    uint64_t* out_staging_bytes
) {
    if (out_staging_bytes) {
        *out_staging_bytes = 0;
    }
    if (!engine ||
        slot_index >= engine->staging_slots.size() ||
        !engine->staging_slots[slot_index].buf) {
        return 0;
    }
    const uint64_t total_bytes = tensor_stream_bytes(rec);
    if (total_bytes == 0) {
        return 0;
    }
    if (total_bytes > engine->staging_slots[slot_index].size) {
        // BUGFIX 문제3: Dynamic slot reallocation for oversized tensors ★★★
        // Problem: Oversized tensors silently fail forever, causing blocking sync loads
        // Solution: Attempt dynamic reallocation if tensor exceeds slot size
        engine->io_atomic.staging_slot_contention.fetch_add(1, std::memory_order_relaxed);

        static std::atomic<uint64_t> s_last_warn_time_ms{0};
        static std::atomic<uint32_t> s_realloc_attempt_count{0};
        const uint64_t now_ms = moe_profile_now_us() / 1000;
        const uint64_t last_warn = s_last_warn_time_ms.load(std::memory_order_relaxed);

        // Check if we should attempt reallocation
        const uint32_t realloc_attempts = s_realloc_attempt_count.load(std::memory_order_relaxed);
        const bool should_try_realloc = (realloc_attempts < 3) &&
                                        (total_bytes <= engine->staging_size / 2);

        if (should_try_realloc) {
            // Attempt to trigger slot reallocation
            s_realloc_attempt_count.fetch_add(1, std::memory_order_relaxed);

            // Update max_tensor_stream_bytes if this tensor is larger
            uint64_t current_max = engine->max_tensor_stream_bytes;
            while (total_bytes > current_max) {
                if (engine->max_tensor_stream_bytes.compare_exchange_weak(
                        current_max, total_bytes,
                        std::memory_order_relaxed, std::memory_order_relaxed)) {
                    break;
                }
            }

            fprintf(stderr,
                "[storageLLM WARNING] Tensor %llu bytes exceeds slot size %llu\n"
                "  Attempting dynamic slot reallocation (attempt %u/3)...\n"
                "  This will pause prefetch for ~5 seconds.\n",
                (unsigned long long)total_bytes,
                (unsigned long long)engine->staging_slots[slot_index].size,
                realloc_attempts + 1);

            // Trigger slot reconfiguration
            configure_staging_slots(engine);

            // Check if reallocation succeeded
            if (slot_index < engine->staging_slots.size() &&
                total_bytes <= engine->staging_slots[slot_index].size) {
                fprintf(stderr,
                    "[storageLLM INFO] Slot reallocation succeeded. New slot size: %llu bytes\n",
                    (unsigned long long)engine->staging_slots[slot_index].size);
                // Retry the prefetch with new slot size - fall through
            } else {
                fprintf(stderr,
                    "[storageLLM ERROR] Slot reallocation failed. Tensor still too large.\n"
                    "  Increase staging_size to at least %llu bytes.\n",
                    (unsigned long long)(total_bytes * 4));
                return 0;
            }
        } else {
            // Already tried reallocation 3 times or tensor too large
            if (now_ms - last_warn > 1000) {
                if (s_last_warn_time_ms.compare_exchange_weak(last_warn, now_ms, std::memory_order_relaxed)) {
                    fprintf(stderr,
                        "[storageLLM ERROR] Tensor %llu bytes exceeds slot size %llu\n"
                        "  Dynamic reallocation %s\n"
                        "  This tensor will use blocking sync load (severe performance impact).\n"
                        "  Increase staging_size to at least %llu bytes and restart.\n",
                        (unsigned long long)total_bytes,
                        (unsigned long long)engine->staging_slots[slot_index].size,
                        should_try_realloc ? "in progress" : "failed after 3 attempts",
                        (unsigned long long)(total_bytes * 4));
                }
            }
            return 0;
        }
    }
    mmap_context_t* shard = get_or_map_tensor_shard(engine, rec);
    if (!shard) return 0;
    uint64_t file_size = static_cast<uint64_t>(mmap_get_size(shard));

    const uint64_t ranges[][2] = {
        {rec.info.weight_byte_offset, rec.info.weight_byte_length},
        {rec.info.scale_byte_offset, rec.info.scale_byte_length},
        {rec.info.scale2_byte_offset, rec.info.scale2_byte_length},
    };

    uint64_t staging_offset = 0;

    // Check if we can use IORing/io_uring registered buffer I/O
    if (engine->ring_adapter.initialized) {
        int submits = 0;
        for (const auto& range : ranges) {
            const uint64_t offset = range[0];
            const uint64_t length = range[1];
            if (length == 0) continue;
            // BUGFIX 318: Check offset + length doesn't overflow (IORing path) ★
            // Problem: Same overflow issue as other paths
            // Solution: Use same check pattern
            if (offset > file_size || length > file_size - offset ||
                staging_offset > total_bytes || length > total_bytes - staging_offset) {
                return 0;
            }
#ifdef _WIN32
            void* file_handle = shard->hFile;
#else
            void* file_handle = nullptr;
#endif
            if (engine->ring_adapter.prep_read_fixed(shard->fd, file_handle,
                engine->staging_slots[slot_index].buf + staging_offset, length, offset, slot_index, (void*)1, staging_offset)) {
                submits++;
            }
            staging_offset += length;
        }

        if (submits > 0) {
            engine->ring_adapter.submit();
            int completions = 0;
            // BUGFIX 121: Issue 26 - io_uring completion wait is spin-yield loop ★★
            // Problem: yield() busy-wait consumes CPU while io_worker waits for I/O.
            // Solution: Add wait_completion() to io_ring_adapter that uses io_uring_wait_cqe (Linux)
            // or blocking wait (Windows). io_worker has no other work during this wait, so blocking is fine.
            // Trade-off: None - blocking wait is correct behavior for synchronous I/O completion.
            while (completions < submits) {
                void* data = engine->ring_adapter.peek_completion();
                if (data) {
                    completions++;
                } else {
                    // Yield once, then block wait if still no completion
                    std::this_thread::yield();
                    if (completions < submits) {
                        data = engine->ring_adapter.wait_completion();
                        if (data) completions++;
                    }
                }
            }
        }
    } else if (shard && mmap_get_ptr(shard, 0)) {
        // Fallback to mmap direct memcpy to avoid dual caching
        for (const auto& range : ranges) {
            const uint64_t offset = range[0];
            const uint64_t length = range[1];
            if (length == 0) continue;
            // BUGFIX 321: Check offset + length doesn't overflow (mmap path) ★
            // Problem: Same overflow issue in mmap path
            // Solution: Use same check pattern
            if (offset > file_size || length > file_size - offset ||
                staging_offset > total_bytes || length > total_bytes - staging_offset) {
                return 0;
            }
            memcpy(engine->staging_slots[slot_index].buf + staging_offset, static_cast<const uint8_t*>(mmap_get_ptr(shard, static_cast<size_t>(offset))), static_cast<size_t>(length));
            staging_offset += length;
        }
    } else {
#if defined(_WIN32)
        // BUGFIX: IOCP-based async I/O for Windows without IORing (pre-Win11 22H2)
        // Avoids blocking io_worker threads with synchronous ifstream reads
        static thread_local moe_iocp_adapter_t tls_iocp;
        const std::string path = rec.resolved_path_index < engine->tensor_paths.size()
            ? engine->tensor_paths[rec.resolved_path_index] : std::string();

        bool iocp_ok = false;
        if (!path.empty()) {
            // Lazily init per-thread IOCP
            if (!tls_iocp.initialized) {
                tls_iocp.init(32);
            }
            if (tls_iocp.initialized) {
                HANDLE hFile = iocp_open_file_async(path.c_str());
                if (hFile != INVALID_HANDLE_VALUE) {
                    tls_iocp.associate_file(hFile);
                    int submits = 0;
                    for (const auto& range : ranges) {
                        const uint64_t offset = range[0];
                        const uint64_t length = range[1];
                        if (length == 0) continue;
                        if (offset > file_size || length > file_size - offset ||
                            staging_offset > total_bytes || length > total_bytes - staging_offset) {
                            CloseHandle(hFile);
                            return 0;
                        }
                        if (tls_iocp.submit_read(hFile, engine->staging_slots[slot_index].buf + staging_offset,
                                offset, static_cast<uint32_t>(length), (void*)1)) {
                            submits++;
                        }
                        staging_offset += length;
                    }
                    int completions = 0;
                    while (completions < submits) {
                        void* data = tls_iocp.wait_completion(50);
                        if (data) completions++;
                    }
                    CloseHandle(hFile);
                    iocp_ok = (completions == submits);
                }
            }
        }
        if (!iocp_ok) {
#endif
        // Fallback to thread-local ifstream cache
        const std::string ifstream_path = rec.resolved_path_index < engine->tensor_paths.size()
            ? engine->tensor_paths[rec.resolved_path_index] : std::string();

        struct cached_stream_t {
            moe_pc_engine* engine;
            std::string path;
            std::ifstream stream;
        };
        static thread_local cached_stream_t tls_stream_cache;

        std::ifstream* input = nullptr;
        if (tls_stream_cache.engine == engine && tls_stream_cache.path == ifstream_path && tls_stream_cache.stream.is_open()) {
            input = &tls_stream_cache.stream;
        } else {
            if (tls_stream_cache.stream.is_open()) tls_stream_cache.stream.close();
            open_ifstream_path(tls_stream_cache.stream, ifstream_path, std::ios::in | std::ios::binary);
            if (!tls_stream_cache.stream) return 0;
            tls_stream_cache.engine = engine;
            tls_stream_cache.path = ifstream_path;
            input = &tls_stream_cache.stream;
        }

        for (const auto& range : ranges) {
            const uint64_t offset = range[0];
            const uint64_t length = range[1];
            if (length == 0) continue;
            // BUGFIX 319: Check offset + length doesn't overflow ★
            // Problem: Same overflow issue in ifstream fallback path
            // Solution: Use same check pattern
            if (offset > file_size || length > file_size - offset ||
                staging_offset > total_bytes || length > total_bytes - staging_offset) {
                return 0;
            }
            if (!read_file_range_to_buffer(*input, offset, engine->staging_slots[slot_index].buf + staging_offset, length)) {
                return 0;
            }
            staging_offset += length;
        }
#if defined(_WIN32)
        }
#endif
    }

    // BUGFIX 320: Check staging_offset == total_bytes before returning success ★
    // Problem: If loop exits early due to error, staging_offset != total_bytes
    // But function returns 1 (success) anyway
    // Solution: Check staging_offset == total_bytes before returning
    if (staging_offset != total_bytes) {
        return 0;
    }
    if (out_staging_bytes) {
        *out_staging_bytes = staging_offset;
    }
    // BUGFIX 257: Add memory_order_relaxed for statistics counters ★
    engine->io_atomic.bytes_prefetched.fetch_add(staging_offset, std::memory_order_relaxed);
    engine->io_atomic.bytes_disk_to_ram.fetch_add(staging_offset, std::memory_order_relaxed);
    engine->io_atomic.disk_stage_done.fetch_add(1, std::memory_order_relaxed);
    engine->io_atomic.pinned_stage_done.fetch_add(1, std::memory_order_relaxed);
    return 1;
}

static int prefetch_tensor_ram_to_pinned(
    moe_pc_engine* engine,
    const moe_tensor_record& rec,
    uint32_t slot_index,
    uint64_t* out_staging_bytes,
    moe_tier_t target_tier = moe_TIER_VRAM
) {
    if (out_staging_bytes) {
        *out_staging_bytes = 0;
    }
    mmap_context_t* shard = get_or_map_tensor_shard(engine, rec);
    if (!engine || !shard || !mmap_get_ptr(shard, 0)) {
        return 0;
    }
    const uint64_t file_size = static_cast<uint64_t>(mmap_get_size(shard));
    const uint64_t ranges[][2] = {
        {rec.info.weight_byte_offset, rec.info.weight_byte_length},
        {rec.info.scale_byte_offset, rec.info.scale_byte_length},
        {rec.info.scale2_byte_offset, rec.info.scale2_byte_length},
    };

    uint64_t copied = 0;
    uint64_t staging_offset = 0;
    const uint64_t total_bytes = tensor_stream_bytes(rec);
    const bool can_stage =
        engine->staging &&
        slot_index < engine->staging_slots.size() &&
        engine->staging_slots[slot_index].buf &&
        total_bytes <= engine->staging_slots[slot_index].size;
    if (!can_stage) {
        // Problem 5 Fix: can_stage==false is transient contention, not structural deficit
        engine->io_atomic.staging_slot_contention.fetch_add(1);
        engine->io_atomic.staging_wraps.fetch_add(1);
        return 0;
    }
    for (const auto& range : ranges) {
        const uint64_t offset = range[0];
        const uint64_t length = range[1];
        if (length == 0) {
            continue;
        }
        if (offset > file_size || length > file_size - offset) {
            return 0;
        }
        const uint8_t* src = static_cast<const uint8_t*>(mmap_get_ptr(shard, static_cast<size_t>(offset)));
        if (!src) {
            return 0;
        }
        const uint64_t before = staging_offset;
        const uint64_t n = copy_mapped_range_to_staging_linear(engine, src, length, slot_index, &staging_offset);
        if (n != length || staging_offset != before + length) {
            return 0;
        }
        // BUGFIX 291: Check staging_offset doesn't exceed total_bytes ★
        // Problem: staging_offset can overflow total_bytes if ranges are malformed
        // Solution: Check staging_offset <= total_bytes after each range
        if (staging_offset > total_bytes) {
            return 0;
        }
        copied += n;

        // Pipeline Fix 3: Use MADV_COLD to accelerate page cache eviction without race
        // MADV_DONTNEED causes page faults if concurrent readers access same mmap range.
        // MADV_COLD (Linux 5.4+) lowers LRU priority without invalidating pages,
        // allowing safe concurrent access while accelerating memory reclaim.
        //
        // Pipeline Fix 3b: Apply MADV_COLD for non-VRAM targets unconditionally.
        // For VRAM targets, only apply when RAM pressure > 85% to prevent OOM from
        // page cache bloat. When VRAM experts are later evicted back to RAM tier,
        // residency_eviction.cpp.inc applies MADV_WILLNEED to re-warm the pages.
#ifdef __linux__
        #ifndef MADV_COLD
        #define MADV_COLD 20
        #endif
        // Apply MADV_COLD for non-VRAM targets (DB fallback path)
        if (target_tier != moe_TIER_VRAM) {
            madvise(const_cast<uint8_t*>(src), static_cast<size_t>(length), MADV_COLD);
        } else {
            // BUGFIX 703: Conditional MADV_COLD for VRAM targets under RAM pressure ★★
            // Problem: VRAM-bound mmap data stays warm in page cache indefinitely.
            //   With 32GB RAM and 300GB model, page cache fills up → OOM risk.
            // Solution: When available RAM < 15% of total, demote VRAM pages too.
            //   MADV_WILLNEED in residency_eviction handles the reverse on VRAM eviction.
            const uint64_t avail = storagellm::available_ram_bytes();
            const uint64_t total = storagellm::total_ram_bytes();
            // BUGFIX 712: Increased MADV_COLD threshold to 30% ★★★
            // Problem: 15% threshold caused massive I/O spikes when RAM was almost exhausted.
            // Solution: Evict earlier at 30% to smooth out page cache eviction.
            if (total > 0 && avail < total * 30ull / 100ull) {
                madvise(const_cast<uint8_t*>(src), static_cast<size_t>(length), MADV_COLD);
            }
        }
        // MADV_DONTNEED only on shutdown when no concurrent readers exist
        if (engine->io_stop.load(std::memory_order_relaxed)) {
            madvise(const_cast<uint8_t*>(src), static_cast<size_t>(length), MADV_DONTNEED);
        }
#endif
    }

    if (out_staging_bytes && can_stage && staging_offset == total_bytes && copied == total_bytes) {
        *out_staging_bytes = staging_offset;
    }
    // BUGFIX 259: Add memory_order_relaxed for statistics counters ★
    // Fix 38: Only count actual pinned staging copies, not touch-fallback bytes
    engine->io_atomic.bytes_ram_to_pinned.fetch_add(copied, std::memory_order_relaxed);
    engine->io_atomic.pinned_stage_done.fetch_add(1, std::memory_order_relaxed);
    return 1;
}


