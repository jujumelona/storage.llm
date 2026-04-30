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
        if (input.gcount() != chunk) {
            return 0;
        }
        copied += static_cast<uint64_t>(chunk);
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
    if (total_bytes == 0 || total_bytes > engine->staging_slots[slot_index].size) {
        return 0;
    }
    const std::string path = rec.resolved_path_index < engine->tensor_paths.size()
        ? engine->tensor_paths[rec.resolved_path_index] : std::string();
    uint64_t file_size = 0;
    if (path.empty() || !file_size_bytes(path, &file_size)) {
        return 0;
    }

    // Thread-local ifstream cache: reuse open streams for same path
    struct cached_stream_t {
        moe_pc_engine* engine;
        std::string path;
        std::ifstream stream;
    };
    static thread_local cached_stream_t tls_stream_cache;

    std::ifstream* input = nullptr;
    if (tls_stream_cache.engine == engine && tls_stream_cache.path == path && tls_stream_cache.stream.is_open()) {
        // Reuse cached stream
        input = &tls_stream_cache.stream;
    } else {
        // Close old stream and open new one
        if (tls_stream_cache.stream.is_open()) {
            tls_stream_cache.stream.close();
        }
        open_ifstream_path(tls_stream_cache.stream, path, std::ios::in | std::ios::binary);
        if (!tls_stream_cache.stream) {
            return 0;
        }
        tls_stream_cache.engine = engine;
        tls_stream_cache.path = path;
        input = &tls_stream_cache.stream;
    }

    const uint64_t ranges[][2] = {
        {rec.info.weight_byte_offset, rec.info.weight_byte_length},
        {rec.info.scale_byte_offset, rec.info.scale_byte_length},
        {rec.info.scale2_byte_offset, rec.info.scale2_byte_length},
    };
    uint64_t staging_offset = 0;
    for (const auto& range : ranges) {
        const uint64_t offset = range[0];
        const uint64_t length = range[1];
        if (length == 0) {
            continue;
        }
        if (offset > file_size || length > file_size - offset ||
            staging_offset > total_bytes || length > total_bytes - staging_offset) {
            return 0;
        }
        if (!read_file_range_to_buffer(
                *input,
                offset,
                engine->staging_slots[slot_index].buf + staging_offset,
                length)) {
            return 0;
        }
        staging_offset += length;
    }
    if (staging_offset != total_bytes) {
        return 0;
    }
    if (out_staging_bytes) {
        *out_staging_bytes = staging_offset;
    }
    engine->io_atomic.bytes_prefetched.fetch_add(staging_offset);
    engine->io_atomic.bytes_disk_to_ram.fetch_add(staging_offset);
    engine->io_atomic.disk_stage_done.fetch_add(1);
    engine->io_atomic.pinned_stage_done.fetch_add(1);
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
        copied += n;

        // Pipeline Fix 3: Use MADV_COLD to accelerate page cache eviction without race
        // MADV_DONTNEED causes page faults if concurrent readers access same mmap range.
        // MADV_COLD (Linux 5.4+) lowers LRU priority without invalidating pages,
        // allowing safe concurrent access while accelerating memory reclaim.
        //
        // Pipeline Fix 3b: Only apply MADV_COLD when target is not VRAM
        // If expert will be in VRAM and later evicted to RAM tier, we want mmap pages
        // to remain in page cache for fast re-access. Only evict pages for DB-bound data.
#ifdef __linux__
        #ifndef MADV_COLD
        #define MADV_COLD 20
        #endif
        // Apply MADV_COLD only for non-VRAM targets (DB fallback path)
        if (target_tier != moe_TIER_VRAM) {
            madvise(const_cast<uint8_t*>(src), static_cast<size_t>(length), MADV_COLD);
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
    // Fix 38: Only count actual pinned staging copies, not touch-fallback bytes
    engine->io_atomic.bytes_ram_to_pinned.fetch_add(copied);
    engine->io_atomic.pinned_stage_done.fetch_add(1);
    return 1;
}
