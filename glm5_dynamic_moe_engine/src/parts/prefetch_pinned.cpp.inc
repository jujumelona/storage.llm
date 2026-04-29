static int prefetch_tensor_ram_to_pinned(
    glm5_pc_engine* engine,
    const glm5_tensor_record& rec,
    uint32_t slot_index,
    uint64_t* out_staging_bytes,
    glm5_tier_t target_tier = GLM5_TIER_VRAM
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

    // Problem 2 Fix: Sync fallback when staging is NULL (mlock failed)
    // Without staging, use direct sync H2D copy instead of failing silently
    if (!can_stage && target_tier == GLM5_TIER_VRAM) {
        // Sync fallback: allocate temp pinned buffer and copy directly to GPU
        void* temp_pinned = nullptr;
        if (engine->cuda.cuMemAllocHost && engine->cuda.cuMemcpyHtoD && engine->cuda.cuMemFreeHost) {
            if (engine->cuda.cuMemAllocHost(&temp_pinned, (size_t)total_bytes) == 0 && temp_pinned) {
                uint64_t temp_offset = 0;
                int copy_ok = 1;
                for (const auto& range : ranges) {
                    const uint64_t offset = range[0];
                    const uint64_t length = range[1];
                    if (length == 0) continue;
                    if (offset > file_size || length > file_size - offset) {
                        copy_ok = 0;
                        break;
                    }
                    const uint8_t* src = static_cast<const uint8_t*>(mmap_get_ptr(shard, static_cast<size_t>(offset)));
                    if (!src) {
                        copy_ok = 0;
                        break;
                    }
                    std::memcpy(static_cast<uint8_t*>(temp_pinned) + temp_offset, src, (size_t)length);
                    temp_offset += length;
                }
                if (copy_ok && temp_offset == total_bytes) {
                    // Now copy to GPU synchronously
                    const uint64_t key = device_key_tensor(rec);
                    CUdeviceptr gpu_ptr = 0;
                    if (device_alloc_tensor(engine, rec, &gpu_ptr) && gpu_ptr) {
                        device_set_current(engine);
                        if (engine->cuda.cuMemcpyHtoD(gpu_ptr, temp_pinned, (size_t)total_bytes) == 0) {
                            engine->io_atomic.bytes_sync_h2d_fallback.fetch_add(total_bytes);
                            engine->cuda.cuMemFreeHost(temp_pinned);
                            if (out_staging_bytes) {
                                *out_staging_bytes = total_bytes;
                            }
                            return 1;
                        }
                    }
                }
                engine->cuda.cuMemFreeHost(temp_pinned);
            }
        }
        // Sync fallback failed
        engine->io_atomic.staging_slot_contention.fetch_add(1);
        engine->io_atomic.staging_wraps.fetch_add(1);
        return 0;
    }

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
        if (target_tier != GLM5_TIER_VRAM) {
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
