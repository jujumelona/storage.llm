// Device tensor copy operations

static uint64_t device_key_tensor(const moe_tensor_record& r) {
    return (3ull << 60) |
        (((uint64_t)r.info.layer & 0xFFFull) << 48) |
        (((uint64_t)r.info.expert & 0xFFFull) << 36) |
        (((uint64_t)r.info.proj & 0xFull) << 32) |
        (((uint64_t)r.info.part & 0xFFull) << 24) |
        ((uint64_t)r.info.shard & 0xFFFFFFull);
}

// Fix 49: _locked version ??caller already holds device_mutex
static void device_free_expert_locked(
    moe_pc_engine* e,
    int layer,
    int expert,
    std::vector<std::pair<CUdeviceptr, uint64_t>>* deferred_free_ptrs
) {
    if (!e || e->device_allocator_ready.load() <= 0) {
        return;
    }
    const moe_projection_t projs[3] = {moe_PROJ_GATE, moe_PROJ_UP, moe_PROJ_DOWN};
    for (moe_projection_t proj : projs) {
        const moe_tensor_record* r = find_tensor_record(e, layer, expert, proj);
        if (r) {
            device_free_locked(e, device_key_tensor(*r), deferred_free_ptrs);
        }
    }
}

// Fix 49: Public version ??acquires device_mutex then delegates
static void device_free_expert(moe_pc_engine* e, int layer, int expert) {
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
    moe_pc_engine* e,
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
    moe_pc_engine* e,
    const moe_tensor_record& r,
    uint8_t pool,
    const uint8_t* host,
    uint64_t host_bytes,
    uint32_t staging_slot = UINT32_MAX
) {
    const uint64_t bytes = tensor_stream_bytes(r);
    if (!e || !host || host_bytes < bytes || bytes == 0 ||
        !device_allocator_init(e) || !device_set_current(e)) {
        return 0;
    }
    CUstream stream = device_tensor_stream_current(e);
    bool is_host_pinned = true; if (staging_slot != UINT32_MAX && staging_slot < e->staging_slots.size()) { is_host_pinned = e->staging_slots[staging_slot].pinned; } if (!is_host_pinned) { e->io_atomic.nonpinned_fallback_count.fetch_add(1, std::memory_order_relaxed); } const int async_copy = is_host_pinned && device_backend_async_h2d_available(e, stream, 1);
    const uint64_t key = device_key_tensor(r);
    CUdeviceptr dst = 0;
    if (!device_prepare_tensor_allocation(e, key, bytes, pool, async_copy, &dst, host)) {
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

    // BUGFIX 193: Release staging slot immediately after async copy starts ★★★
    // Problem: Slot released in callback after GPU copy completes
    // → io_worker ↔ pinned_worker ↔ gpu_worker all spin-wait for slots
    // → Pipeline stalls, throughput drops
    // Solution: Release slot right after cuMemcpyAsync succeeds
    // GPU DMA already started, staging buffer no longer needed by CPU
    // Event-based completion tracking ensures correctness
    // Impact: Slot available ~100-500µs earlier, eliminates spin-wait
    if (async_copy) {
        if (staging_slot != UINT32_MAX) {
            // Defer only key completion, not slot release
            device_defer_staging_copy(e, UINT32_MAX, key, stream);
            // Release slot immediately - DMA already in flight
            release_staging_slot(e, staging_slot);
        } else {
            s_tl_tensor_pending_keys[e].push_back(key);
        }
    } else if (staging_slot != UINT32_MAX) {
        release_staging_slot(e, staging_slot);
    }
    device_touch_allocation(e, key);
    return 1;
}

static int device_copy_tensor(
    moe_pc_engine* e,
    const moe_tensor_record& r,
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

    // Fix 55: Try staging fast-path ??consolidate 3 ranges into pinned buffer
    // for single true-async DMA instead of 3 false-async non-pinned copies
    // Bug #9: GPU direct uses second half of slot pool (for_pinned = false)
    // PIPELINE FIX 17: Use thread-local turn counter to reduce slot contention ?�★
    // Old: worker_index=0, turn=0 fixed ??all workers compete for slot 0
    // New: thread-local turn counter ??workers spread across slot pool
    static thread_local uint32_t tl_device_copy_turn = 0;
    const uint32_t slot_idx = (e->staging_slots.size() > 0 && bytes <= e->staging_slot_size)
        ? acquire_staging_slot(e, 0u, tl_device_copy_turn++, bytes, false) : UINT32_MAX;

    if (slot_idx != UINT32_MAX) {
        // Staging fast-path: memcpy ranges ??pinned staging ??single H2D async
        moe_staging_slot& slot = e->staging_slots[slot_idx];
        if (!slot.buf || bytes > slot.size) {
            release_staging_slot(e, slot_idx);
            return 0;
        }
        uint8_t* staging_buf = slot.buf;
        // PIPELINE FIX 16: Use AVX2 non-temporal copy instead of plain memcpy ?�★??
        // Old: plain memcpy ??L1/L2 cache pollution, ~1.5-2x slower
        // New: copy_mapped_range_to_staging_linear ??AVX2 streaming stores
        // Matches io_worker performance for mmap?�pinned copies
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
        if (!device_prepare_tensor_allocation(e, key, bytes, pool, async_copy, &dst, staging_buf)) {
            release_staging_slot(e, slot_idx);
            return 0;
        }
        int ok = 0;
        if (async_copy) {
            ok = device_backend_copy_h2d_async(e, dst, staging_buf, bytes, stream);
            // BUGFIX 193: Release slot immediately after async copy starts ★★★
            // Fix 63: Do NOT sync here → defer only key completion
            // Slot released right after DMA starts, not after completion
        } else {
            ok = device_backend_copy_h2d_sync(e, dst, staging_buf, bytes);
        }
        if (ok && async_copy) {
            // BUGFIX 193: Defer only key, release slot immediately
            device_defer_staging_copy(e, UINT32_MAX, key, stream);
            release_staging_slot(e, slot_idx);
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
            // Staged async copies are completed by their per-copy event.
        }
        device_touch_allocation(e, key);
        return 1;
    }

    // Fix 55: No staging ??use explicit synchronous cuMemcpyHtoD (no false async)
    if (!allow_sync_fallback && e->staging_slots.size() > 0 && bytes <= e->staging_slot_size) {
        return moe_DEVICE_COPY_BACKPRESSURE;
    }
    if (e) {
        e->io_atomic.bytes_sync_h2d_fallback.fetch_add(bytes);
    }
    const void* host_ptr = shard ? mmap_get_ptr(shard, static_cast<size_t>(ranges[0][0])) : nullptr;
    if (!device_prepare_tensor_allocation(e, key, bytes, pool, 0 /*sync*/, &dst, host_ptr)) {
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
