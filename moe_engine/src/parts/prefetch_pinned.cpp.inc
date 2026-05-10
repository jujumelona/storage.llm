static int enqueue_directstorage_staging_pending(
    moe_pc_engine* engine,
    const moe_dstorage_pending_copy& pending,
    uint32_t staging_slot,
    const moe_prefetch_task& next_task,
    uint64_t expected_bytes
);
static int read_file_range_to_buffer(
    moe_pc_engine* engine,
    std::ifstream& input,
    uint64_t file_offset,
    uint8_t* dst,
    uint64_t bytes
) {
    if (!input || !dst || bytes == 0 ||
        file_offset > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
        return 0;
    }

    // BUGFIX Problem D: Adaptive storage bandwidth measurement ???    // Measure bandwidth from large reads (>10MB) to avoid init delay
    const uint64_t measure_threshold = 10ull * 1024ull * 1024ull;
    const bool should_measure = engine && bytes >= measure_threshold;

    const uint64_t start_us = should_measure ? moe_profile_now_us() : 0;

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
        // BUGFIX 258: Check for EOF and partial reads ?
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

    // BUGFIX 951: Adaptive EMA (Exponential Moving Average) for storage bandwidth ???PERFORMANCE
    // Problem: Bandwidth was measured once and fixed, ignoring SSD thermal throttling,
    //   cache exhaustion, or OS page cache hits over time.
    // Solution: Use EMA (alpha=0.2) to update measured bandwidth continuously
    //   on large reads, adapting to current system conditions.
    // Impact: Juju strategy router uses accurate real-time I/O latency estimations
    // BUGFIX Bug #12: Use dedicated bw_measurement_mutex instead of device_mutex ??MINOR
    // Problem: device_mutex is shared with GPU operations ??IO/GPU worker contention
    // Solution: Use separate bw_measurement_mutex for EMA updates only
    // Impact: Eliminates mutex contention, improves throughput
    if (should_measure && engine) {
        const uint64_t elapsed_us = moe_profile_now_us() - start_us;
        if (elapsed_us > 0) {
            const uint64_t current_bw = (bytes * 1000000ull) / elapsed_us;
            static std::mutex s_storage_bw_update_mutex;
            std::lock_guard<std::mutex> bw_lock(s_storage_bw_update_mutex);
            const uint64_t old_bw = engine->backend_caps.measured_storage_bw_bytes_per_sec;
            const uint64_t new_bw =
                (old_bw == 0) ? current_bw : (old_bw * 8ull + current_bw * 2ull) / 10ull;
            engine->backend_caps.measured_storage_bw_bytes_per_sec = new_bw;
            if (old_bw == 0) {
                const double gb_per_sec = (double)current_bw / 1e9;
                fprintf(stderr,
                    "[storageLLM INFO] Initial storage bandwidth measured: %.2f GB/s from %llu MB read\n",
                    gb_per_sec, (unsigned long long)(bytes / (1024*1024)));
            }
        }
    }

    return 1;
}

// BUGFIX 51: Issue 22 - Cache ifstream per worker thread ?
// Problem: prefetch_tensor_file_to_pinned opens/closes ifstream for every tensor.
// 8 experts  3 projections = 24 file opens per MoE layer. Same shard file reopened repeatedly.
// Each open requires seekg(), and non-mmap path uses different OS cache than mmap path.
// Solution: Use thread_local ifstream cache keyed by path. Reuse open streams when path matches.
// Fall back to mmap path when available (more efficient, shares OS page cache).
static int prefetch_tensor_file_to_pinned(
    moe_pc_engine* engine,
    const moe_tensor_record& rec,
    uint32_t* io_slot_index,
    uint64_t* out_staging_bytes
) {
    if (out_staging_bytes) {
        *out_staging_bytes = 0;
    }
    uint32_t slot_index = io_slot_index ? *io_slot_index : UINT32_MAX;

    // BUGFIX Issue 5: Generalized unified memory fast path ???    // Problem: Metal, AMD APU, Intel integrated GPU all have unified memory
    // Old: Metal-only fast path hardcoded to moe_BACKEND_METAL
    // Solution: Check supports_unified_memory flag instead of backend type
    // This enables zero-copy for AMD APU (780M), Intel Iris Xe, and Metal
    // BUGFIX Issue 3: Add explicit zero-copy signal to caller ???    // Problem: Caller doesn't know if unified memory fast path was taken
    // Solution: Set io_slot_index to UINT32_MAX to signal "no slot used"
    if (engine && engine->backend_caps.supports_unified_memory &&
        engine->backend_caps.unified_mmap_gpu_accessible) {
        // Unified memory path - map file data directly without staging
        // Works for Metal (Apple Silicon), AMD APU (780M), Intel integrated (Iris Xe)
        mmap_context_t* shard = get_or_map_tensor_shard(engine, rec);
        if (!shard || !mmap_get_ptr(shard, 0)) {
            return 0;
        }
        const uint64_t total_bytes = tensor_stream_bytes(rec);
        if (total_bytes == 0) {
            return 0;
        }
        // For unified memory backends, we don't use staging slots at all
        // The mmap'd data is directly accessible by GPU (zero-copy)
        // Signal to caller that no slot was used by setting io_slot_index to UINT32_MAX
        if (io_slot_index) {
            *io_slot_index = UINT32_MAX;
        }
        if (out_staging_bytes) {
            *out_staging_bytes = total_bytes;
        }
        engine->io_atomic.bytes_prefetched.fetch_add(total_bytes, std::memory_order_relaxed);
        engine->io_atomic.disk_stage_done.fetch_add(1, std::memory_order_relaxed);
        engine->io_atomic.pinned_stage_done.fetch_add(1, std::memory_order_relaxed);
        return 1;
    }

    // BUGFIX Issue 3: Use staging_slot_valid() helper for correct check order ???    // Problem: Manual checks can get order wrong (size before staging_slot_size)
    // Solution: Use centralized helper that enforces correct order
    if (!engine || !staging_slot_valid(engine, slot_index)) {
        if (engine && engine->staging_slot_size == 0 && slot_index < engine->staging_slots.size()) {
            fprintf(stderr,
                "[storageLLM ERROR] prefetch_tensor_file_to_pinned: staging_slot_size=0, "
                "slots are invalid after reset timeout. Rejecting prefetch.\n");
        }
        return 0;
    }
    const uint64_t total_bytes = tensor_stream_bytes(rec);
    if (total_bytes == 0) {
        return 0;
    }
    if (total_bytes > engine->staging_slots[slot_index].size) {
        // BUGFIX 3: Dynamic slot reallocation for oversized tensors ???        // Problem: Oversized tensors silently fail forever, causing blocking sync loads
        // Solution: Attempt dynamic reallocation if tensor exceeds slot size
        engine->io_atomic.oversized_tensor_fallbacks.fetch_add(1, std::memory_order_relaxed);

        static std::atomic<uint64_t> s_last_warn_time_ms{0};
        const uint64_t now_ms = moe_profile_now_us() / 1000;
        const uint64_t last_warn = s_last_warn_time_ms.load(std::memory_order_relaxed);

        // Check if we should attempt reallocation
        // BUGFIX Bug #7: Fix sys_free calculation to use actual available RAM ???CRITICAL
        // Problem: tier_budget() returns total RAM budget, not remaining free RAM.
        // Old code: sys_free = ram_budget (total budget, e.g., 16GB)
        // When RAM is 90% full (14.4GB used), sys_free still shows 16GB ??OOM on realloc
        // Solution: Subtract tier_used() from tier_budget() to get actual free RAM
        // sys_free = ram_budget - ram_used (e.g., 16GB - 14.4GB = 1.6GB)
        // This prevents reallocation attempts when RAM is actually full
        // New buffer allocated BEFORE old one freed ??temporarily uses 2 staging memory
        const uint64_t ram_budget = tier_budget(engine, moe_TIER_RAM);
        const uint64_t ram_used = tier_used(engine, moe_TIER_RAM);
        const uint64_t sys_free = (ram_budget > ram_used) ? (ram_budget - ram_used) :
            (engine->backend_caps.system_memory_bytes > 0
                ? (engine->backend_caps.system_memory_bytes * 85ull / 100ull)  // conservative fallback
                : UINT64_MAX);
        const bool enough_headroom = sys_free > total_bytes + 256ull * 1024ull * 1024ull;
        const bool should_try_realloc = enough_headroom && total_bytes <= engine->staging_size / 2;

        if (should_try_realloc) {
            // BUGFIX Issue 2-3: Fix race condition in dynamic slot reallocation ???            // Problem: Old sequence "release ??configure ??acquire" allows other workers
            // to access memory being freed during configure_staging_slots, causing use-after-free.
            // Solution: New sequence "acquire new ??if success, release old" prevents race.
            // Trade-off: Temporarily uses one extra slot worth of memory during reallocation.

            uint64_t observed_max = engine->max_tensor_stream_bytes.load(std::memory_order_relaxed);
            while (total_bytes > observed_max &&
                   !engine->max_tensor_stream_bytes.compare_exchange_weak(
                       observed_max,
                       total_bytes,
                       std::memory_order_relaxed,
                       std::memory_order_relaxed)) {
            }

            fprintf(stderr,
                "[storageLLM WARNING] Tensor %llu bytes exceeds slot size %llu\n"
                "  Attempting dynamic slot reallocation...\n"
                "  This will pause prefetch for ~5 seconds.\n",
                (unsigned long long)total_bytes,
                (unsigned long long)engine->staging_slots[slot_index].size);

            // Step 1: Try to allocate a new larger buffer BEFORE releasing the old one
            // This prevents other workers from accessing memory being freed
            bool got_pinned = engine->staging_pinned;
            bool got_numa = engine->staging_numa_allocated;
            bool got_device_host = engine->staging_device_host_allocated;
            uint8_t* new_buf = allocate_staging_for_engine(
                engine,
                engine->backend_caps.preferred_numa_node,
                total_bytes,
                &got_pinned,
                &got_numa,
                &got_device_host);

            if (!new_buf) {
                // Allocation failed - keep using old slot (graceful degradation)
                fprintf(stderr,
                    "[storageLLM ERROR] Buffer reallocation failed. Could not allocate %llu bytes.\n"
                    "  Keeping old slot. Increase staging_size to at least %llu bytes.\n",
                    (unsigned long long)total_bytes,
                    (unsigned long long)(total_bytes * 4));
                return 0;
            }

            // BUGFIX 979: Fix use-after-free in staging slot reallocation ???CRITICAL
            // Problem: slot.buf freed outside staging_slot_mutex. Another io_worker
            // can pass staging_slot_valid() and read the freed pointer concurrently.
            // Solution: Snapshot old values and swap to new buffer inside the lock.
            // Free old buffer AFTER lock release ??no thread can see old_buf anymore.
            moe_staging_slot& slot = engine->staging_slots[slot_index];
            uint8_t* old_buf = nullptr;
            uint64_t old_size = 0;
            bool old_pinned = false, old_numa = false, old_device_host = false;
            {
                std::lock_guard<std::mutex> lock(engine->staging_slot_mutex);
                old_buf           = slot.buf;
                old_size          = slot.size;
                old_pinned        = slot.pinned;
                old_numa          = slot.numa_allocated;
                old_device_host   = slot.device_host_allocated;
                slot.buf                  = new_buf;
                slot.size                 = total_bytes;
                slot.pinned               = got_pinned;
                slot.numa_allocated       = got_numa;
                slot.device_host_allocated = got_device_host;
            }
            // Lock released ??old_buf is now invisible to all other threads
            if (old_buf) {
                free_staging_for_engine(engine, old_buf, old_size, old_pinned,
                                        old_numa, old_device_host);
            }

            // Check if reallocation succeeded
            if (staging_slot_valid(engine, slot_index) &&
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
                uint64_t expected_last_warn = last_warn;
                if (s_last_warn_time_ms.compare_exchange_weak(expected_last_warn, now_ms, std::memory_order_relaxed)) {
                    fprintf(stderr,
                        "[storageLLM ERROR] Tensor %llu bytes exceeds slot size %llu\n"
                        "  Dynamic reallocation %s\n"
                        "  This tensor will use blocking sync load (severe performance impact).\n"
                        "  Increase staging_size to at least %llu bytes and restart.\n",
                        (unsigned long long)total_bytes,
                        (unsigned long long)engine->staging_slots[slot_index].size,
                        should_try_realloc ? "in progress" : "skipped because tensor exceeds half of staging_size",
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

    // BUGFIX Problem 3: Validate io_generation to prevent path transition buffer corruption ???CRITICAL
    // Problem: IO worker captures task with OVERLAPPED enabled, engine transitions to mmap,
    // worker allocates slot under old generation and submits OVERLAPPED read using stale assumption.
    // Solution: Validate slot's io_generation matches engine's current generation before submitting I/O.
    uint32_t current_io_gen = engine->prefer_overlapped_io_generation.load(std::memory_order_relaxed);
    if (slot_index != UINT32_MAX && engine->staging_slots[slot_index].io_generation != current_io_gen) {
        return 0; // Generation mismatch, abort prefetch to avoid corruption
    }
    void* io_gen_ptr = (void*)(uintptr_t)current_io_gen;

    // Check if we can use IORing/io_uring registered buffer I/O.  The engine owns
    // one registered-buffer ring, so the whole SQE prepare/submit/wait sequence
    // is serialized to prevent one worker from consuming another worker's CQEs.
    if (staging_offset != total_bytes && engine->ring_adapter.initialized) {
        int submits = 0;
        std::lock_guard<std::mutex> ring_lock(engine->ring_adapter.submit_mutex);
        for (const auto& range : ranges) {
            const uint64_t offset = range[0];
            const uint64_t length = range[1];
            if (length == 0) continue;
            // BUGFIX 318: Check offset + length doesn't overflow (IORing path) ??            // Problem: Same overflow issue as other paths
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
                engine->staging_slots[slot_index].buf + staging_offset, length, offset, slot_index, io_gen_ptr, staging_offset)) {
                submits++;
            }
            staging_offset += length;
        }

        if (submits > 0) {
            engine->ring_adapter.submit();
            int completions = 0;
            // BUGFIX 121: Issue 26 - io_uring completion wait is spin-yield loop ?
            // Problem: yield() busy-wait consumes CPU while io_worker waits for I/O.
            // Solution: Add wait_completion() to io_ring_adapter that uses io_uring_wait_cqe (Linux)
            // or blocking wait (Windows). io_worker has no other work during this wait, so blocking is fine.
            // Trade-off: None - blocking wait is correct behavior for synchronous I/O completion.
            while (completions < submits) {
                void* data = engine->ring_adapter.wait_completion();
                if (data) {
                    if ((uint32_t)(uintptr_t)data == engine->prefer_overlapped_io_generation.load(std::memory_order_relaxed)) {
                        completions++;
                    }
                } else break;
            }
        }
    } else if (staging_offset != total_bytes && shard && mmap_get_ptr(shard, 0)) {
        // Fallback to mmap direct memcpy to avoid dual caching
        for (const auto& range : ranges) {
            const uint64_t offset = range[0];
            const uint64_t length = range[1];
            if (length == 0) continue;
            // BUGFIX 321: Check offset + length doesn't overflow (mmap path) ??            // Problem: Same overflow issue in mmap path
            // Solution: Use same check pattern
            if (offset > file_size || length > file_size - offset ||
                staging_offset > total_bytes || length > total_bytes - staging_offset) {
                return 0;
            }
            memcpy(engine->staging_slots[slot_index].buf + staging_offset, static_cast<const uint8_t*>(mmap_get_ptr(shard, static_cast<size_t>(offset))), static_cast<size_t>(length));
            staging_offset += length;
        }
    } else if (staging_offset != total_bytes) {
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
                                offset, static_cast<uint32_t>(length), io_gen_ptr)) {
                            submits++;
                        }
                        staging_offset += length;
                    }
                    int completions = 0;
                    while (completions < submits) {
                        void* data = tls_iocp.wait_completion(50);
                        if (data) {
                            if ((uint32_t)(uintptr_t)data == engine->prefer_overlapped_io_generation.load(std::memory_order_relaxed)) {
                                completions++;
                            } else {
                                // If stale generation, wait loops but will eventually time out if no more valid completions
                            }
                        }
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
            uint64_t generation;
            std::ifstream stream;
        };
        static thread_local cached_stream_t tls_stream_cache;

        std::ifstream* input = nullptr;
        if (tls_stream_cache.engine == engine &&
            tls_stream_cache.path == ifstream_path &&
            tls_stream_cache.generation == engine->staging_generation.load(std::memory_order_relaxed) &&
            tls_stream_cache.stream.is_open()) {
            input = &tls_stream_cache.stream;
        } else {
            if (tls_stream_cache.stream.is_open()) tls_stream_cache.stream.close();
            open_ifstream_path(tls_stream_cache.stream, ifstream_path, std::ios::in | std::ios::binary);
            if (!tls_stream_cache.stream) return 0;
            tls_stream_cache.engine = engine;
            tls_stream_cache.path = ifstream_path;
            tls_stream_cache.generation = engine->staging_generation.load(std::memory_order_relaxed);
            input = &tls_stream_cache.stream;
        }

        for (const auto& range : ranges) {
            const uint64_t offset = range[0];
            const uint64_t length = range[1];
            if (length == 0) continue;
            // BUGFIX 319: Check offset + length doesn't overflow ??            // Problem: Same overflow issue in ifstream fallback path
            // Solution: Use same check pattern
            if (offset > file_size || length > file_size - offset ||
                staging_offset > total_bytes || length > total_bytes - staging_offset) {
                return 0;
            }
            if (!read_file_range_to_buffer(engine, *input, offset, engine->staging_slots[slot_index].buf + staging_offset, length)) {
                return 0;
            }
            staging_offset += length;
        }
#if defined(_WIN32)
        }
#endif
    }

    // BUGFIX 320: Check staging_offset == total_bytes before returning success ??    // Problem: If loop exits early due to error, staging_offset != total_bytes
    // But function returns 1 (success) anyway
    // Solution: Check staging_offset == total_bytes before returning
    if (staging_offset != total_bytes) {
        return 0;
    }
    if (out_staging_bytes) {
        *out_staging_bytes = staging_offset;
    }
    // BUGFIX 257: Add memory_order_relaxed for statistics counters ??    engine->io_atomic.bytes_prefetched.fetch_add(staging_offset, std::memory_order_relaxed);
    engine->io_atomic.bytes_disk_to_ram.fetch_add(staging_offset, std::memory_order_relaxed);
    engine->io_atomic.disk_stage_done.fetch_add(1, std::memory_order_relaxed);
    engine->io_atomic.pinned_stage_done.fetch_add(1, std::memory_order_relaxed);
    return 1;
}

static int prefetch_tensor_file_to_pinned_directstorage_async(
    moe_pc_engine* engine,
    const moe_tensor_record& rec,
    uint32_t slot_index,
    const moe_prefetch_task& next_task,
    uint64_t expected_bytes
) {
#if defined(_WIN32)
    if (!engine ||
        slot_index == UINT32_MAX ||
        expected_bytes == 0 ||
        !staging_slot_valid(engine, slot_index) ||
        !engine->io_config.enable_directstorage ||
        !engine->backend_caps.supports_directstorage ||
        !compiled_directstorage_available()) {
        return 0;
    }
    mmap_context_t* shard = get_or_map_tensor_shard(engine, rec);
    if (!shard) {
        return 0;
    }
    const uint64_t file_size = static_cast<uint64_t>(mmap_get_size(shard));
    const uint64_t ranges[][2] = {
        {rec.info.weight_byte_offset, rec.info.weight_byte_length},
        {rec.info.scale_byte_offset, rec.info.scale_byte_length},
        {rec.info.scale2_byte_offset, rec.info.scale2_byte_length},
    };
    const std::string ds_path = rec.resolved_path_index < engine->tensor_paths.size()
        ? engine->tensor_paths[rec.resolved_path_index] : std::string();
    const std::wstring wide_path = storagellm::wide_path_from_utf8(ds_path);
    moe_dstorage_pending_copy pending{};
    if (wide_path.empty() ||
        !directstorage_submit_ranges_to_staging(
            engine,
            wide_path.c_str(),
            ranges,
            sizeof(ranges) / sizeof(ranges[0]),
            engine->staging_slots[slot_index].buf.load(std::memory_order_acquire),
            expected_bytes,
            file_size,
            &pending)) {
        return 0;
    }
    if (!enqueue_directstorage_staging_pending(
            engine,
            pending,
            slot_index,
            next_task,
            expected_bytes)) {
        directstorage_release_pending_copy_resources(&pending);
        return 0;
    }
    return 1;
#else
    (void)engine;
    (void)rec;
    (void)slot_index;
    (void)next_task;
    (void)expected_bytes;
    return 0;
#endif
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
        staging_slot_valid(engine, slot_index) &&
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
        // BUGFIX 291: Check staging_offset doesn't exceed total_bytes ??        // Problem: staging_offset can overflow total_bytes if ranges are malformed
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
            // BUGFIX 703: Conditional MADV_COLD for VRAM targets under RAM pressure ?
            // Problem: VRAM-bound mmap data stays warm in page cache indefinitely.
            //   With 32GB RAM and 300GB model, page cache fills up ??OOM risk.
            // Solution: When available RAM < 15% of total, demote VRAM pages too.
            //   MADV_WILLNEED in residency_eviction handles the reverse on VRAM eviction.
            const uint64_t avail = storagellm::available_ram_bytes();
            const uint64_t total = storagellm::total_ram_bytes();
            // BUGFIX 712: Increased MADV_COLD threshold to 30% ???            // Problem: 15% threshold caused massive I/O spikes when RAM was almost exhausted.
            // Solution: Evict earlier at 30% to smooth out page cache eviction.
            const uint32_t threshold_pct = engine->io_config.madv_cold_threshold_pct;
            if (total > 0 && avail < total * (uint64_t)threshold_pct / 100ull) {
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
    // BUGFIX 259: Add memory_order_relaxed for statistics counters ??    // Fix 38: Only count actual pinned staging copies, not touch-fallback bytes
    engine->io_atomic.bytes_ram_to_pinned.fetch_add(copied, std::memory_order_relaxed);
    engine->io_atomic.pinned_stage_done.fetch_add(1, std::memory_order_relaxed);
    return 1;
}

static int prefetch_tensor_ram_to_pinned_at(
    moe_pc_engine* engine,
    const moe_tensor_record& rec,
    uint32_t slot_index,
    uint64_t* inout_staging_offset,
    uint64_t* out_tensor_bytes,
    moe_tier_t target_tier = moe_TIER_VRAM
) {
    if (out_tensor_bytes) {
        *out_tensor_bytes = 0;
    }
    if (!engine || !inout_staging_offset) {
        return 0;
    }
    mmap_context_t* shard = get_or_map_tensor_shard(engine, rec);
    if (!shard || !mmap_get_ptr(shard, 0)) {
        return 0;
    }
    const uint64_t total_bytes = tensor_stream_bytes(rec);
    if (total_bytes == 0 ||
        !staging_slot_valid(engine, slot_index) ||
        !engine->staging_slots[slot_index].buf ||
        *inout_staging_offset > engine->staging_slots[slot_index].size ||
        total_bytes > engine->staging_slots[slot_index].size - *inout_staging_offset) {
        engine->io_atomic.staging_slot_contention.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }

    const uint64_t file_size = static_cast<uint64_t>(mmap_get_size(shard));
    const uint64_t ranges[][2] = {
        {rec.info.weight_byte_offset, rec.info.weight_byte_length},
        {rec.info.scale_byte_offset, rec.info.scale_byte_length},
        {rec.info.scale2_byte_offset, rec.info.scale2_byte_length},
    };

    uint64_t staging_offset = *inout_staging_offset;
    uint64_t copied = 0;
    for (const auto& range : ranges) {
        const uint64_t offset = range[0];
        const uint64_t length = range[1];
        if (length == 0) {
            continue;
        }
        if (offset > file_size || length > file_size - offset) {
            return 0;
        }
        const uint8_t* src = static_cast<const uint8_t*>(
            mmap_get_ptr(shard, static_cast<size_t>(offset)));
        if (!src) {
            return 0;
        }
        const uint64_t before = staging_offset;
        const uint64_t n = copy_mapped_range_to_staging_linear(
            engine, src, length, slot_index, &staging_offset);
        if (n != length || staging_offset != before + length) {
            return 0;
        }
        copied += n;
#ifdef __linux__
        if (target_tier != moe_TIER_VRAM) {
            #ifndef MADV_COLD
            #define MADV_COLD 20
            #endif
            madvise(const_cast<uint8_t*>(src), static_cast<size_t>(length), MADV_COLD);
        }
#else
        (void)target_tier;
#endif
    }
    if (copied != total_bytes) {
        return 0;
    }
    *inout_staging_offset = staging_offset;
    if (out_tensor_bytes) {
        *out_tensor_bytes = copied;
    }
    engine->io_atomic.bytes_ram_to_pinned.fetch_add(copied, std::memory_order_relaxed);
    return 1;
}
