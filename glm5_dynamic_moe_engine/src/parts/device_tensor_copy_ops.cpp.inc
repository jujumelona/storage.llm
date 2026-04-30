// Device tensor copy operations

static uint64_t device_key_tensor(const glm5_tensor_record& r) {
    return (3ull << 60) |
        (((uint64_t)r.info.layer & 0xFFFull) << 48) |
        (((uint64_t)r.info.expert & 0xFFFull) << 36) |
        (((uint64_t)r.info.proj & 0xFull) << 32) |
        (((uint64_t)r.info.part & 0xFFull) << 24) |
        ((uint64_t)r.info.shard & 0xFFFFFFull);
}

// Fix 49: _locked version — caller already holds device_mutex
static void device_free_expert_locked(
    glm5_pc_engine* e,
    int layer,
    int expert,
    std::vector<std::pair<CUdeviceptr, uint64_t>>* deferred_free_ptrs
) {
    if (!e || e->device_allocator_ready.load() <= 0) {
        return;
    }
    const glm5_projection_t projs[3] = {GLM5_PROJ_GATE, GLM5_PROJ_UP, GLM5_PROJ_DOWN};
    for (glm5_projection_t proj : projs) {
        const glm5_tensor_record* r = find_tensor_record(e, layer, expert, proj);
        if (r) {
            device_free_locked(e, device_key_tensor(*r), deferred_free_ptrs);
        }
    }
}

// Fix 49: Public version — acquires device_mutex then delegates
static void device_free_expert(glm5_pc_engine* e, int layer, int expert) {
    if (!e || e->device_allocator_ready.load() <= 0) {
        return;
    }
    std::vector<std::pair<CUdeviceptr, uint64_t>> deferred_free_ptrs;
    {
        std::lock_guard<std::mutex> lock(e->device_mutex);
        device_free_expert_locked(e, layer, expert, &deferred_free_ptrs);
    }
    device_free_deferred_ptrs(e, deferred_free_ptrs);
}

static void device_free_experts_batched(
    glm5_pc_engine* e,
    const std::vector<std::pair<int, int>>& experts,
    size_t batch_size = 8
) {
    if (!e || e->device_allocator_ready.load() <= 0 || experts.empty()) {
        return;
    }
    if (batch_size == 0) {
        batch_size = 1;
    }
    for (size_t i = 0; i < experts.size(); i += batch_size) {
        std::vector<std::pair<CUdeviceptr, uint64_t>> deferred_free_ptrs;
        {
            std::lock_guard<std::mutex> lock(e->device_mutex);
            const size_t end = std::min(i + batch_size, experts.size());
            for (size_t j = i; j < end; ++j) {
                device_free_expert_locked(e, experts[j].first, experts[j].second, &deferred_free_ptrs);
            }
        }
        device_free_deferred_ptrs(e, deferred_free_ptrs);
    }
}

static int device_copy_tensor_from_host(
    glm5_pc_engine* e,
    const glm5_tensor_record& r,
    uint8_t pool,
    const uint8_t* host,
    uint64_t host_bytes
) {
    const uint64_t bytes = tensor_stream_bytes(r);
    if (!e || !host || host_bytes < bytes || bytes == 0 ||
        !device_allocator_init(e) || !device_set_current(e)) {
        return 0;
    }
    CUstream stream = device_tensor_stream_current(e);
    const int async_copy = device_backend_async_h2d_available(e, stream, 1);
    const uint64_t key = device_key_tensor(r);
    CUdeviceptr dst = 0;
    if (!device_prepare_tensor_allocation(e, key, bytes, pool, async_copy, &dst)) {
        return 0;
    }

    const int ok = async_copy ?
        device_backend_copy_h2d_async(e, dst, host, bytes, stream) :
        device_backend_copy_h2d_sync(e, dst, host, bytes);
    if (!ok) {
        e->io_atomic.device_copy_failures.fetch_add(1, std::memory_order_relaxed);
        if (async_copy) {
            device_backend_stream_sync(e, stream);
        }
        if (async_copy) {
            device_cancel_pending_copy(e, key);
        }
        device_free_key(e, key);
        return 0;
    }
    if (async_copy) {
        s_tl_tensor_pending_keys[e].push_back(key);
    }
    device_touch_allocation(e, key);
    return 1;
}

static int device_copy_tensor(
    glm5_pc_engine* e,
    const glm5_tensor_record& r,
    uint8_t pool,
    int allow_sync_fallback = 1
) {
    if (!e || tensor_stream_bytes(r) == 0 || !device_allocator_init(e) || !device_set_current(e)) {
        return 0;
    }
    mmap_context_t* shard = get_or_map_tensor_shard(e, r);
    if (!shard || !mmap_get_ptr(shard, 0)) {
        return 0;
    }
    const uint64_t key = device_key_tensor(r);
    const uint64_t bytes = tensor_stream_bytes(r);
    const uint64_t ranges[3][2] = {
        {r.info.weight_byte_offset, r.info.weight_byte_length},
        {r.info.scale_byte_offset, r.info.scale_byte_length},
        {r.info.scale2_byte_offset, r.info.scale2_byte_length}
    };
    const uint64_t file_size = static_cast<uint64_t>(mmap_get_size(shard));
    for (const auto& range : ranges) {
        if (range[1] && (range[0] > file_size || range[1] > file_size - range[0])) {
            return 0;
        }
    }


    CUstream stream = device_tensor_stream_current(e);
    CUdeviceptr dst = 0;

    // Fix 55: Try staging fast-path — consolidate 3 ranges into pinned buffer
    // for single true-async DMA instead of 3 false-async non-pinned copies
    // Bug #9: GPU direct uses second half of slot pool (for_pinned = false)
    // PIPELINE FIX 17: Use thread-local turn counter to reduce slot contention ★★
    // Old: worker_index=0, turn=0 fixed → all workers compete for slot 0
    // New: thread-local turn counter → workers spread across slot pool
    static thread_local uint32_t tl_device_copy_turn = 0;
    const uint32_t slot_idx = (e->staging_slots.size() > 0 && bytes <= e->staging_slot_size)
        ? acquire_staging_slot(e, 0u, tl_device_copy_turn++, false) : UINT32_MAX;

    if (slot_idx != UINT32_MAX) {
        // Staging fast-path: memcpy ranges → pinned staging → single H2D async
        glm5_staging_slot& slot = e->staging_slots[slot_idx];
        if (!slot.buf || bytes > slot.size) {
            release_staging_slot(e, slot_idx);
            return 0;
        }
        uint8_t* staging_buf = slot.buf;
        // PIPELINE FIX 16: Use AVX2 non-temporal copy instead of plain memcpy ★★★
        // Old: plain memcpy → L1/L2 cache pollution, ~1.5-2x slower
        // New: copy_mapped_range_to_staging_linear → AVX2 streaming stores
        // Matches io_worker performance for mmap→pinned copies
        uint64_t offset = 0;
        int copy_ok = 1;
        for (const auto& range : ranges) {
            if (!range[1]) continue;
            const uint8_t* src = static_cast<const uint8_t*>(mmap_get_ptr(shard, static_cast<size_t>(range[0])));
            if (!src) {
                copy_ok = 0;
                break;
            }
            const uint64_t copied = copy_mapped_range_to_staging_linear(e, src, range[1], slot_idx, &offset);
            if (copied != range[1]) {
                copy_ok = 0;
                break;
            }
        }
        if (!copy_ok) {
            release_staging_slot(e, slot_idx);
            return 0;
        }
        const int async_copy = device_backend_async_h2d_available(
            e, stream, e->staging_slots[slot_idx].pinned ? 1 : 0);
        if (!device_prepare_tensor_allocation(e, key, bytes, pool, async_copy, &dst)) {
            release_staging_slot(e, slot_idx);
            return 0;
        }
        int ok = 0;
        if (async_copy) {
            ok = device_backend_copy_h2d_async(e, dst, staging_buf, bytes, stream);
            // Fix 63: Do NOT sync here — defer slot release to device_stream_sync_current
            // This enables true async DMA pipelining across batch
        } else {
            ok = device_backend_copy_h2d_sync(e, dst, staging_buf, bytes);
        }
        if (ok && async_copy) {
            // Fix 63: Defer slot release — GPU still reading from staging buffer
            s_tl_deferred_staging_slots[e].push_back(slot_idx);
        } else {
            release_staging_slot(e, slot_idx);
        }
        if (!ok) {
            e->io_atomic.device_copy_failures.fetch_add(1, std::memory_order_relaxed);
            if (async_copy) device_cancel_pending_copy(e, key);
            device_free_key(e, key);
            return 0;
        }
        if (async_copy) {
            s_tl_tensor_pending_keys[e].push_back(key);
        }
        device_touch_allocation(e, key);
        return 1;
    }

    // Fix 55: No staging — use explicit synchronous cuMemcpyHtoD (no false async)
    if (!allow_sync_fallback && e->staging_slots.size() > 0 && bytes <= e->staging_slot_size) {
        return GLM5_DEVICE_COPY_BACKPRESSURE;
    }
    if (e) {
        e->io_atomic.bytes_sync_h2d_fallback.fetch_add(bytes);
    }
    if (!device_prepare_tensor_allocation(e, key, bytes, pool, 0 /*sync*/, &dst)) {
        return 0;
    }
    CUdeviceptr cur = dst;
    for (const auto& range : ranges) {
        if (!range[1]) continue;
        const void* src = mmap_get_ptr(shard, static_cast<size_t>(range[0]));
        if (!src) {
            device_free_key(e, key);
            return 0;
        }
        if (!device_backend_copy_h2d_sync(e, cur, src, range[1])) {
            e->io_atomic.device_copy_failures.fetch_add(1, std::memory_order_relaxed);
            device_free_key(e, key);
            return 0;
        }
        cur += range[1];
    }
    device_touch_allocation(e, key);
    return 1;
}
