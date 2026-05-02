static int prefetch_tensor_disk_to_ram(moe_pc_engine* engine, const moe_tensor_record& rec) {
    mmap_context_t* shard = get_or_map_tensor_shard(engine, rec);
    if (!engine || !shard || !mmap_get_ptr(shard, 0)) {
        return 0;
    }

    const uint64_t file_size = static_cast<uint64_t>(mmap_get_size(shard));
    // Bug Fix: file_size (entire shard, GB) vs pinned_staging_bytes (hundreds MB) ??always false.
    // Use tensor_stream_bytes (actual transfer size) instead.
    const uint64_t transfer_bytes = tensor_stream_bytes(rec);
    if (engine->staging_pinned && transfer_bytes > 0 && transfer_bytes <= engine->io_config.pinned_staging_bytes) {
        lock_mapped_shard_once(engine, shard);
    }
    const uint64_t ranges[][2] = {
        {rec.info.weight_byte_offset, rec.info.weight_byte_length},
        {rec.info.scale_byte_offset, rec.info.scale_byte_length},
        {rec.info.scale2_byte_offset, rec.info.scale2_byte_length},
    };

    uint64_t prefetched = 0;
    bool hint_issued = false;  // Fix 35: track whether any mmap_prefetch was actually issued
    for (const auto& range : ranges) {
        const uint64_t offset = range[0];
        const uint64_t length = range[1];
        if (length == 0) {
            continue;
        }
        // BUGFIX 316: Check offset + length doesn't overflow before comparison ★
        // Problem: offset + length can overflow uint64_t if ranges are malformed
        // Causes offset > file_size check to pass incorrectly
        // Solution: Check length <= file_size - offset instead
        if (offset > file_size || length > file_size - offset) {
            return 0;
        }
        if (engine->io_config.enable_mmap_prefetch) {
            mmap_prefetch(shard, static_cast<size_t>(offset), static_cast<size_t>(length));
            hint_issued = true;
        }
    }
    // Bug #5: Removed fixed 100µs sleep → touch_mapped_range guarantees pages regardless.
    // NVMe responds in <10µs (wasted wait), HDD needs 4-10ms (insufficient wait).
    // Synchronously touch pages to guarantee data residency in RAM
    for (const auto& range : ranges) {
        const uint64_t offset = range[0];
        const uint64_t length = range[1];
        if (length > 0) {
            const uint8_t* src = static_cast<const uint8_t*>(mmap_get_ptr(shard, static_cast<size_t>(offset)));
            // BUGFIX 317: Check src validity before using ★
            // Problem: mmap_get_ptr can return null if offset is invalid
            // Dereferencing null src in touch_mapped_range causes crash
            // Solution: Check src before use
            if (!src) {
                return 0;
            }
            touch_mapped_range(engine, src, length);
            if (!hint_issued) {
                // BUGFIX 155: Add memory_order_relaxed for statistics counter
                engine->io_atomic.bytes_touch_fallback.fetch_add(length, std::memory_order_relaxed);
            }
            prefetched += length;
        }
    }

    // BUGFIX 155: Add memory_order_relaxed for statistics counters
    engine->io_atomic.bytes_prefetched.fetch_add(prefetched, std::memory_order_relaxed);
    if (hint_issued) {
        engine->io_atomic.bytes_disk_hint_issued.fetch_add(prefetched, std::memory_order_relaxed);
    }
    engine->io_atomic.disk_stage_done.fetch_add(1, std::memory_order_relaxed);
    return 1;
}

// Optimization 5: Linux io_uring registered buffer.
// The active path uses moe_io_ring_adapter. Keep this legacy path opt-in so
// liburing-less Lightning images still build.
#if defined(__linux__) && defined(MOE_HAVE_LIBURING) && MOE_HAVE_LIBURING && defined(MOE_ENABLE_LEGACY_URING_STATE)
static bool init_io_uring(moe_pc_engine* engine) {
    if (!engine) return false;
    auto& state = engine->uring_state;
    if (state.initialized) return true;

    // IORING_SETUP_SQPOLL for kernel-side polling (zero syscall)
    if (io_uring_queue_init(128, &state.ring, IORING_SETUP_SQPOLL) < 0) {
        return false;
    }

    // BUGFIX 531: Check staging_slots.size() before resize to prevent overflow ★
    // Problem: staging_slots.size() can be very large (> 100000) causing memory allocation failure
    // Solution: Validate size is reasonable before resize
    if (engine->staging_slots.size() > 100000) {
        io_uring_queue_exit(&state.ring);
        return false;
    }

    state.registered_iovecs.resize(engine->staging_slots.size());
    for (size_t i = 0; i < engine->staging_slots.size(); ++i) {
        state.registered_iovecs[i].iov_base = engine->staging_slots[i].buf; // Note: using .buf instead of ptr
        state.registered_iovecs[i].iov_len  = engine->staging_slots[i].size;
    }

    int ret = io_uring_register_buffers(&state.ring, state.registered_iovecs.data(), (unsigned)state.registered_iovecs.size());
    if (ret < 0) {
        io_uring_queue_exit(&state.ring);
        return false;
    }
    state.initialized = true;
    return true;
}

static void disk_read_to_staging_uring(moe_pc_engine* engine, int fd, uint64_t offset, uint32_t slot_idx, uint64_t bytes) {
    // BUGFIX 532: Check engine and slot_idx validity before use ★
    // Problem: No validation of engine or slot_idx before array access
    // Solution: Check engine and slot_idx < staging_slots.size()
    if (!engine || slot_idx >= engine->staging_slots.size()) {
        return;
    }

    struct io_uring_sqe* sqe = io_uring_get_sqe(&engine->uring_state.ring);
    if (!sqe) return;
    // IORING_OP_READ_FIXED: read directly into registered buffer, bypassing kernel copy
    io_uring_prep_read_fixed(sqe, fd, engine->staging_slots[slot_idx].buf, bytes, offset, (int)slot_idx);
    sqe->user_data = slot_idx;
    io_uring_submit(&engine->uring_state.ring);
}
#endif
