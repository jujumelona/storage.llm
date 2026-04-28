static uint64_t device_key_tensor(const glm5_tensor_record& r) {
    return (3ull << 60) | ((uint64_t)r.info.layer << 48) | ((uint64_t)r.info.expert << 36) |
        ((uint64_t)r.info.proj << 32) | ((uint64_t)r.info.part << 24) | r.info.shard;
}

// Fix 49: _locked version — caller already holds device_mutex
static void device_free_expert_locked(
    glm5_pc_engine* e,
    int layer,
    int expert,
    std::vector<CUdeviceptr>* deferred_free_ptrs
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
    std::vector<CUdeviceptr> deferred_free_ptrs;
    {
        std::lock_guard<std::mutex> lock(e->device_mutex);
        device_free_expert_locked(e, layer, expert, &deferred_free_ptrs);
    }
    device_free_deferred_ptrs(e, deferred_free_ptrs);
}

#include <unordered_map>
static thread_local std::unordered_map<glm5_pc_engine*, CUstream> s_tl_tensor_streams;
static thread_local std::unordered_map<glm5_pc_engine*, std::vector<uint64_t>> s_tl_tensor_pending_keys;
// Fix 63: Deferred staging slot release — freed after batch stream sync
static thread_local std::unordered_map<glm5_pc_engine*, std::vector<uint32_t>> s_tl_deferred_staging_slots;
static constexpr int GLM5_DEVICE_COPY_BACKPRESSURE = -2;

static CUstream device_tensor_stream_current(glm5_pc_engine* e) {
    if (!e || !e->cuda.cuStreamCreate) {
        return nullptr;
    }
    if (!device_set_current(e)) {
        return nullptr;
    }
    auto it = s_tl_tensor_streams.find(e);
    if (it != s_tl_tensor_streams.end()) {
        return it->second;
    }
    CUstream stream = nullptr;
    // Fix 4: Use CU_STREAM_NON_BLOCKING (=1) to prevent implicit sync with default stream.
    // This allows multiple HtoD copy streams to run truly in parallel.
    if (e->cuda.cuStreamCreate(&stream, 1u /* CU_STREAM_NON_BLOCKING */) != 0) {
        return nullptr;
    }
    s_tl_tensor_streams[e] = stream;
    return stream;
}

static int device_mark_copy_pending_locked(glm5_pc_engine* e, uint64_t key) {
    if (!e) {
        return 0;
    }
    auto it = e->device_allocations.find(key);
    if (it == e->device_allocations.end() || it->second.pending_free) {
        return 0;
    }
    ++it->second.inflight_copies;
    return 1;
}

static void device_complete_copy_locked(
    glm5_pc_engine* e,
    uint64_t key,
    std::vector<CUdeviceptr>* deferred_free_ptrs = nullptr
) {
    if (!e) {
        return;
    }
    auto it = e->device_allocations.find(key);
    if (it == e->device_allocations.end()) {
        return;
    }
    if (it->second.inflight_copies > 0) {
        --it->second.inflight_copies;
    }
    if (it->second.inflight_copies == 0 && it->second.pending_free) {
        device_release_allocation_locked(e, it, deferred_free_ptrs);
    }
}

static void device_complete_pending_copies(glm5_pc_engine* e) {
    if (!e) return;
    auto it = s_tl_tensor_pending_keys.find(e);
    if (it == s_tl_tensor_pending_keys.end() || it->second.empty()) return;
    std::vector<CUdeviceptr> deferred_free_ptrs;
    {
        std::lock_guard<std::mutex> lock(e->device_mutex);
        for (uint64_t key : it->second) {
            device_complete_copy_locked(e, key, &deferred_free_ptrs);
        }
    }
    it->second.clear();
    device_free_deferred_ptrs(e, deferred_free_ptrs);
}

static void device_force_clear_pending_copies(glm5_pc_engine* e) {
    if (!e) return;
    std::vector<CUdeviceptr> deferred_free_ptrs;
    {
        std::lock_guard<std::mutex> lock(e->device_mutex);
        for (auto it = e->device_allocations.begin(); it != e->device_allocations.end(); ) {
            it->second.inflight_copies = 0;
            if (it->second.pending_free) {
                auto victim = it++;
                device_release_allocation_locked(e, victim, &deferred_free_ptrs);
            } else {
                ++it;
            }
        }
    }
    device_free_deferred_ptrs(e, deferred_free_ptrs);
}

static void device_cancel_pending_copy(glm5_pc_engine* e, uint64_t key) {
    if (!e) {
        return;
    }
    {
        std::vector<CUdeviceptr> deferred_free_ptrs;
        {
            std::lock_guard<std::mutex> lock(e->device_mutex);
            device_complete_copy_locked(e, key, &deferred_free_ptrs);
        }
        device_free_deferred_ptrs(e, deferred_free_ptrs);
    }
}

static void device_free_key(glm5_pc_engine* e, uint64_t key) {
    if (!e) {
        return;
    }
    std::vector<CUdeviceptr> deferred_free_ptrs;
    {
        std::lock_guard<std::mutex> lock(e->device_mutex);
        device_free_locked(e, key, &deferred_free_ptrs);
    }
    device_free_deferred_ptrs(e, deferred_free_ptrs);
}

static int device_prepare_tensor_allocation(
    glm5_pc_engine* e,
    uint64_t key,
    uint64_t bytes,
    uint8_t pool,
    int async_copy,
    CUdeviceptr* out_dst
) {
    if (!e || !out_dst) {
        return 0;
    }
    *out_dst = 0;
    std::vector<CUdeviceptr> deferred_free_ptrs;
    std::unique_lock<std::mutex> lock(e->device_mutex);
    glm5_device_allocation* a = nullptr;
    auto it = e->device_allocations.find(key);
    if (it != e->device_allocations.end() && it->second.pending_free) {
        return 0;
    }
    if (it == e->device_allocations.end() || it->second.bytes < bytes) {
        if (it != e->device_allocations.end()) {
            if (it->second.inflight_copies != 0) {
                it->second.pending_free = true;
                return 0;
            }
            device_free_locked(e, key, &deferred_free_ptrs);
        }
        device_make_room_locked(e, pool, bytes, deferred_free_ptrs);
        if (!deferred_free_ptrs.empty()) {
            lock.unlock();
            device_free_deferred_ptrs(e, deferred_free_ptrs);
            // Bug #5: Bounded retry loop instead of recursion (max 8 retries).
            // Prevents stack overflow under extreme VRAM pressure.
            for (int _retry = 0; _retry < 8; ++_retry) {
                deferred_free_ptrs.clear();
                lock.lock();
                it = e->device_allocations.find(key);
                if (it != e->device_allocations.end() && it->second.pending_free) return 0;
                if (it == e->device_allocations.end() || it->second.bytes < bytes) {
                    if (it != e->device_allocations.end()) {
                        if (it->second.inflight_copies != 0) { it->second.pending_free = true; return 0; }
                        device_free_locked(e, key, &deferred_free_ptrs);
                    }
                    device_make_room_locked(e, pool, bytes, deferred_free_ptrs);
                    if (!deferred_free_ptrs.empty()) {
                        lock.unlock();
                        device_free_deferred_ptrs(e, deferred_free_ptrs);
                        continue;
                    }
                }
                break;
            }
            if (!lock.owns_lock()) lock.lock();
            it = e->device_allocations.find(key);
        }
        if (device_budget_bytes(e) && e->device_allocated_bytes.load() + bytes > device_budget_bytes(e)) {
            return 0;
        }
        glm5_device_allocation next;
        next.key = key;
        next.bytes = bytes;
        next.pool = pool;
        if (e->cuda.cuMemAlloc(&next.ptr, (size_t)bytes) != 0) {
            return 0;
        }
        e->device_allocated_bytes.fetch_add(bytes);
        e->device_allocation_count.fetch_add(1);
        if (auto* c = device_pool_counter(e, pool)) {
            c->fetch_add(bytes);
        }
        // Fix 64: cuMemGetInfo is slow driver call — do NOT call under device_mutex
        a = &e->device_allocations.emplace(key, next).first->second;
        if (pool == GLM5_DEVICE_POOL_ACTIVATION) {
            e->activation_keys.insert(key);
        }
    } else {
        a = &it->second;
    }
    if (async_copy && !device_mark_copy_pending_locked(e, key)) {
        device_free_locked(e, key, &deferred_free_ptrs);
        lock.unlock();
        device_free_deferred_ptrs(e, deferred_free_ptrs);
        return 0;
    }
    *out_dst = a->ptr;
    return 1;
}

static void device_touch_allocation(glm5_pc_engine* e, uint64_t key) {
    if (!e) {
        return;
    }
    std::lock_guard<std::mutex> lock(e->device_mutex);
    auto it = e->device_allocations.find(key);
    if (it != e->device_allocations.end()) {
        it->second.last_touch_epoch = e->expert_touch_epoch.fetch_add(1) + 1;
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
    const int async_copy = stream && e->cuda.cuMemcpyHtoDAsync;
    const uint64_t key = device_key_tensor(r);
    CUdeviceptr dst = 0;
    if (!device_prepare_tensor_allocation(e, key, bytes, pool, async_copy, &dst)) {
        return 0;
    }

    const int ok = async_copy ?
        e->cuda.cuMemcpyHtoDAsync(dst, host, (size_t)bytes, stream) == 0 :
        (e->cuda.cuMemcpyHtoD && e->cuda.cuMemcpyHtoD(dst, host, (size_t)bytes) == 0);
    if (!ok) {
        if (async_copy && e->cuda.cuStreamSynchronize) {
            e->cuda.cuStreamSynchronize(stream);
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
    const uint32_t slot_idx = (e->staging_slots.size() > 0 && bytes <= e->staging_slot_size)
        ? acquire_staging_slot(e, 0u, 0u, false) : UINT32_MAX;

    if (slot_idx != UINT32_MAX) {
        // Staging fast-path: memcpy ranges → pinned staging → single H2D async
        glm5_staging_slot& slot = e->staging_slots[slot_idx];
        if (!slot.buf || bytes > slot.size) {
            release_staging_slot(e, slot_idx);
            return 0;
        }
        uint8_t* staging_buf = slot.buf;
        uint64_t offset = 0;
        for (const auto& range : ranges) {
            if (!range[1]) continue;
            const uint8_t* src = static_cast<const uint8_t*>(mmap_get_ptr(shard, static_cast<size_t>(range[0])));
            if (!src) {
                release_staging_slot(e, slot_idx);
                return 0;
            }
            memcpy(staging_buf + offset, src, (size_t)range[1]);
            offset += range[1];
        }
        const int async_copy = stream && e->cuda.cuMemcpyHtoDAsync;
        if (!device_prepare_tensor_allocation(e, key, bytes, pool, async_copy, &dst)) {
            release_staging_slot(e, slot_idx);
            return 0;
        }
        int ok = 0;
        if (async_copy) {
            ok = e->cuda.cuMemcpyHtoDAsync(dst, staging_buf, (size_t)bytes, stream) == 0;
            // Fix 63: Do NOT sync here — defer slot release to device_stream_sync_current
            // This enables true async DMA pipelining across batch
        } else {
            ok = e->cuda.cuMemcpyHtoD && e->cuda.cuMemcpyHtoD(dst, staging_buf, (size_t)bytes) == 0;
        }
        if (ok && async_copy) {
            // Fix 63: Defer slot release — GPU still reading from staging buffer
            s_tl_deferred_staging_slots[e].push_back(slot_idx);
        } else {
            release_staging_slot(e, slot_idx);
        }
        if (!ok) {
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
        if (!e->cuda.cuMemcpyHtoD || e->cuda.cuMemcpyHtoD(cur, src, (size_t)range[1]) != 0) {
            device_free_key(e, key);
            return 0;
        }
        cur += range[1];
    }
    device_touch_allocation(e, key);
    return 1;
}

static void device_stream_sync_current(glm5_pc_engine* e) {
    if (e && e->cuda.cuStreamSynchronize) {
        auto it = s_tl_tensor_streams.find(e);
        if (it != s_tl_tensor_streams.end() && it->second) {
            e->cuda.cuStreamSynchronize(it->second);
        }
    }
    device_complete_pending_copies(e);
    // Fix 63: Release deferred staging slots after stream sync (GPU done reading)
    auto sit = s_tl_deferred_staging_slots.find(e);
    if (sit != s_tl_deferred_staging_slots.end() && !sit->second.empty()) {
        for (uint32_t slot : sit->second) {
            release_staging_slot(e, slot);
        }
        sit->second.clear();
        // Bug #2: Wake all waiters after batch deferred slot release
        e->staging_slot_cv.notify_all();
    }
}

static void device_stream_destroy_current(glm5_pc_engine* e) {
    device_stream_sync_current(e);
    if (e && e->cuda.cuStreamDestroy) {
        auto it = s_tl_tensor_streams.find(e);
        if (it != s_tl_tensor_streams.end() && it->second) {
            e->cuda.cuStreamDestroy(it->second);
            s_tl_tensor_streams.erase(it);
        }
    }
    s_tl_tensor_pending_keys.erase(e);
}

