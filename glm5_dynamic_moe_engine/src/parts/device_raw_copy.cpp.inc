static uint64_t device_key_raw_span(const glm5_storage_raw_span_spec_t& s, uint8_t pool) {
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
    glm5_pc_engine* e,
    const glm5_storage_raw_span_spec_t& s,
    uint8_t pool,
    int allow_sync_fallback = 1
) {
    if (!e || s.bytes == 0 || !device_allocator_init(e) || !device_set_current(e)) {
        return 0;
    }
    const glm5_storage_part_spec_t* part = glm5_storage_part_by_id(s.part);
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
    const uint64_t file_size = (uint64_t)mmap_get_size(mapped);
    if (s.bytes > file_size) {
        return 0;
    }

    const uint64_t key = device_key_raw_span(s, pool);
    const uint8_t* src = static_cast<const uint8_t*>(mmap_get_ptr(mapped, 0));
    if (!src) {
        return 0;
    }
    CUstream stream = device_tensor_stream_current(e);
    // Bug #10: QKV attention weights (raw spans) use pinned slot pool (for_pinned = true)
    // This prevents attention DMA from blocking MoE weight DMA in the GPU direct pool.
    const uint32_t slot_idx = (e->staging_slots.size() > 0 && s.bytes <= e->staging_slot_size)
        ? acquire_staging_slot(e, 0u, s_raw_span_acquire_turn++, true) : UINT32_MAX;

    if (slot_idx != UINT32_MAX) {
        glm5_staging_slot& slot = e->staging_slots[slot_idx];
        if (!slot.buf || s.bytes > slot.size) {
            release_staging_slot(e, slot_idx);
            return 0;
        }
        std::memcpy(slot.buf, src, static_cast<size_t>(s.bytes));

        const int async_copy = stream && e->cuda.cuMemcpyHtoDAsync && slot.pinned;
        CUdeviceptr dst = 0;
        if (!device_prepare_tensor_allocation(e, key, s.bytes, pool, async_copy, &dst)) {
            release_staging_slot(e, slot_idx);
            return 0;
        }
        const int ok = async_copy ?
            e->cuda.cuMemcpyHtoDAsync(dst, slot.buf, (size_t)s.bytes, stream) == 0 :
            (e->cuda.cuMemcpyHtoD && e->cuda.cuMemcpyHtoD(dst, slot.buf, (size_t)s.bytes) == 0);
        if (ok && async_copy) {
            s_tl_tensor_pending_keys[e].push_back(key);
            s_tl_deferred_staging_slots[e].push_back(slot_idx);
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

    if (!allow_sync_fallback && e->staging_slots.size() > 0 && s.bytes <= e->staging_slot_size) {
        return GLM5_DEVICE_COPY_BACKPRESSURE;
    }
    e->io_atomic.bytes_sync_h2d_fallback.fetch_add(s.bytes);

    CUdeviceptr dst = 0;
    if (!device_prepare_tensor_allocation(e, key, s.bytes, pool, 0, &dst)) {
        return 0;
    }
    if (!e->cuda.cuMemcpyHtoD || e->cuda.cuMemcpyHtoD(dst, src, (size_t)s.bytes) != 0) {
        device_free_key(e, key);
        return 0;
    }
    device_touch_allocation(e, key);
    return 1;
}
