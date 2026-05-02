static uint64_t device_key_raw_span(const moe_storage_raw_span_spec_t& s, uint8_t pool) {
    // Bug 11 Fix: Use deterministic structured key instead of 40-bit hash.
    // Hash collision probability ~0.003% with 7,900 spans causes silent data corruption.
    // New key: part_id (16b) | layer (16b) | offset>>12 (28b) = unique per span.
    const uint64_t layer = (uint64_t)((uint32_t)s.layer & 0xffffu);
    const uint64_t part = (uint64_t)((uint32_t)s.part & 0xffffu);
    return (4ull << 60)
         | ((uint64_t)(pool & 0x0fu) << 56)
         | (part << 40)
         | (layer << 24)
         | ((s.offset >> 12) & 0x00ffffffull);
}

static thread_local uint32_t s_raw_span_acquire_turn = 0;

static int device_copy_raw_span(
    moe_pc_engine* e,
    const moe_storage_raw_span_spec_t& s,
    uint8_t pool,
    int allow_sync_fallback = 1
) {
    if (!e || s.bytes == 0 || !device_allocator_init(e) || !device_set_current(e)) {
        return 0;
    }
    // BUGFIX 595: Check s.bytes upper bound ★
    // Problem: s.bytes can be corrupted (very large value)
    // Solution: Validate before processing
    if (s.bytes > 10ull * 1024 * 1024 * 1024) { // 10GB sanity check for raw span
        return 0;
    }
    const moe_storage_part_spec_t* part = moe_storage_part_by_id(s.part);
    if (!part || !part->path || e->model_root.empty()) {
        return 0;
    }
    mmap_context_t* mapped = get_or_map_shard_partial_by_id(
        e,
        s.part,
        resolve_model_file_path(e->model_root, part->path ? part->path : "", ""),
        s.offset,
        s.bytes
    );
    if (!mapped || !mmap_get_ptr(mapped, 0)) {
        return 0;
    }
    // BUGFIX 263: Check for negative mmap_get_size return value ★★
    // Problem: mmap_get_size returns ssize_t (can be negative on error)
    // Casting to uint64_t converts negative to very large positive
    // Solution: Check for negative before casting
    const ssize_t file_size_signed = mmap_get_size(mapped);
    if (file_size_signed < 0) {
        return 0;
    }
    const uint64_t file_size = (uint64_t)file_size_signed;
    if (s.bytes > file_size) {
        return 0;
    }

    const uint64_t key = device_key_raw_span(s, pool);
    const uint8_t* src = static_cast<const uint8_t*>(mmap_get_ptr(mapped, 0));
    if (!src) {
        return 0;
    }
    // BUGFIX 596: Check staging_slot_size > 0 before comparison ★
    // Problem: staging_slot_size can be 0
    // Solution: Check before comparison
    CUstream stream = device_tensor_stream_current(e);
    // Bug #10: QKV attention weights (raw spans) use pinned slot pool (for_pinned = true)
    // This prevents attention DMA from blocking MoE weight DMA in the GPU direct pool.
    const uint32_t slot_idx = (e->staging_slots.size() > 0 && e->staging_slot_size > 0 && s.bytes <= e->staging_slot_size)
        ? acquire_staging_slot(e, 0u, s_raw_span_acquire_turn++, s.bytes, true) : UINT32_MAX;

    if (slot_idx != UINT32_MAX) {
        moe_staging_slot& slot = e->staging_slots[slot_idx];
        if (!slot.buf || s.bytes > slot.size) {
            release_staging_slot(e, slot_idx);
            return 0;
        }
        const uint64_t copied = copy_mapped_range_to_staging(e, src, s.bytes, slot_idx);
        if (copied != s.bytes) {
            release_staging_slot(e, slot_idx);
            return 0;
        }

        const int async_copy = device_backend_async_h2d_available(
            e, stream, slot.pinned ? 1 : 0);
        CUdeviceptr dst = 0;
        if (!device_prepare_tensor_allocation(e, key, s.bytes, pool, async_copy, &dst)) {
            release_staging_slot(e, slot_idx);
            return 0;
        }
        const int ok = async_copy ?
            device_backend_copy_h2d_async(e, dst, slot.buf, s.bytes, stream) :
            device_backend_copy_h2d_sync(e, dst, slot.buf, s.bytes);
        if (ok && async_copy) {
            device_defer_staging_copy(e, slot_idx, key, stream);
        } else {
            release_staging_slot(e, slot_idx);
        }
        if (!ok) {
            if (async_copy) {
                device_cancel_pending_copy(e, key);
            }
            device_free_key(e, key);
            return 0;
        }
        device_touch_allocation(e, key);
        return 1;
    }

    if (!allow_sync_fallback) {
        return moe_DEVICE_COPY_BACKPRESSURE;
    }
    e->io_atomic.bytes_sync_h2d_fallback.fetch_add(s.bytes);

    CUdeviceptr dst = 0;
    if (!device_prepare_tensor_allocation(e, key, s.bytes, pool, 0, &dst)) {
        return 0;
    }
    if (!device_backend_copy_h2d_sync(e, dst, src, s.bytes)) {
        device_free_key(e, key);
        return 0;
    }
    device_touch_allocation(e, key);
    return 1;
}
