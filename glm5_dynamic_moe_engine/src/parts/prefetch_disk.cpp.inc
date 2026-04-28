static int prefetch_tensor_disk_to_ram(glm5_pc_engine* engine, const glm5_tensor_record& rec) {
    if (engine && (int)rec.info.expert >= 0) {
        const size_t eslot = expert_dense_slot((int)rec.info.layer, (int)rec.info.expert);
        std::lock_guard<std::mutex> lock(engine->map_mutex);
        if (eslot < engine->expert_states.size() && engine->expert_states[eslot].active &&
            (engine->expert_states[eslot].tier == GLM5_TIER_RAM ||
             engine->expert_states[eslot].tier == GLM5_TIER_VRAM)) {
            return 1;
        }
    }
    mmap_context_t* shard = get_or_map_tensor_shard(engine, rec);
    if (!engine || !shard || !mmap_get_ptr(shard, 0)) {
        return 0;
    }

    const uint64_t file_size = static_cast<uint64_t>(mmap_get_size(shard));
    if (engine->staging_pinned && file_size > 0 && file_size <= engine->io_config.pinned_staging_bytes) {
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
        if (offset > file_size || length > file_size - offset) {
            return 0;
        }
        if (engine->io_config.enable_mmap_prefetch) {
            mmap_prefetch(shard, static_cast<size_t>(offset), static_cast<size_t>(length));
            hint_issued = true;
        }
    }
    // Bug #5: Removed fixed 100µs sleep — touch_mapped_range guarantees pages regardless.
    // NVMe responds in <10µs (wasted wait), HDD needs 4-10ms (insufficient wait).
    // Synchronously touch pages to guarantee data residency in RAM
    for (const auto& range : ranges) {
        const uint64_t offset = range[0];
        const uint64_t length = range[1];
        if (length > 0) {
            const uint8_t* src = static_cast<const uint8_t*>(mmap_get_ptr(shard, static_cast<size_t>(offset)));
            if (!src) {
                return 0;
            }
            if (!hint_issued) {
                touch_mapped_range(src, length);
                engine->io_atomic.bytes_touch_fallback.fetch_add(length);
            }
            prefetched += length;
        }
    }

    engine->io_atomic.bytes_prefetched.fetch_add(prefetched);
    engine->io_atomic.bytes_disk_to_ram.fetch_add(prefetched);
    engine->io_atomic.disk_stage_done.fetch_add(1);
    return 1;
}
