// Device tensor allocation preparation

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
    device_refresh_mem_info(e);
    std::vector<std::pair<CUdeviceptr, uint64_t>> deferred_free_ptrs;
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
        device_make_room_locked(e, pool, bytes, lock, deferred_free_ptrs);
        if (!deferred_free_ptrs.empty()) {
            lock.unlock();
            device_free_deferred_ptrs(e, deferred_free_ptrs);
            device_refresh_mem_info(e);
            lock.lock();
            it = e->device_allocations.find(key);
            if (it != e->device_allocations.end() && it->second.pending_free) {
                return 0;
            }
            if (it != e->device_allocations.end() && it->second.bytes >= bytes) {
                a = &it->second;
            } else if (it != e->device_allocations.end()) {
                if (it->second.inflight_copies != 0) {
                    it->second.pending_free = true;
                    return 0;
                }
                device_free_locked(e, key, &deferred_free_ptrs);
                if (!deferred_free_ptrs.empty()) {
                    lock.unlock();
                    device_free_deferred_ptrs(e, deferred_free_ptrs);
                    device_refresh_mem_info(e);
                    lock.lock();
                    if (e->device_allocations.find(key) != e->device_allocations.end()) {
                        e->io_atomic.staging_wraps.fetch_add(1);
                        return 0;
                    }
                }
            }
        }
        if (!a) {
            it = e->device_allocations.find(key);
            if (it != e->device_allocations.end() && it->second.pending_free) {
                return 0;
            }
            if (it != e->device_allocations.end() && it->second.bytes >= bytes) {
                a = &it->second;
            } else if (it != e->device_allocations.end()) {
                if (it->second.inflight_copies != 0) {
                    it->second.pending_free = true;
                    return 0;
                }
                device_free_locked(e, key, &deferred_free_ptrs);
                if (!deferred_free_ptrs.empty()) {
                    lock.unlock();
                    device_free_deferred_ptrs(e, deferred_free_ptrs);
                    device_refresh_mem_info(e);
                    lock.lock();
                    if (e->device_allocations.find(key) != e->device_allocations.end()) {
                        e->io_atomic.staging_wraps.fetch_add(1);
                        return 0;
                    }
                }
            }
        }
        if (!a && device_budget_bytes(e) && e->device_allocated_bytes.load() + bytes > device_budget_bytes(e)) {
            e->io_atomic.staging_wraps.fetch_add(1);
            return 0;
        }
        if (!a) {
            glm5_device_allocation next;
            next.key = key;
            next.bytes = bytes;
            next.pool = pool;
            if (e->cuda.cuMemAlloc(&next.ptr, (size_t)bytes) != 0) {
                lock.unlock();
                device_refresh_mem_info(e);
                return 0;
            }
            e->device_allocated_bytes.fetch_add(bytes);
            device_account_free_bytes_sub(e, bytes);
            e->device_allocation_count.fetch_add(1);
            if (auto* c = device_pool_counter(e, pool)) {
                c->fetch_add(bytes);
            }
        // Fix 64: cuMemGetInfo is slow driver call — do NOT call under device_mutex
            a = &e->device_allocations.emplace(key, next).first->second;
            if (pool == GLM5_DEVICE_POOL_ACTIVATION) {
                e->activation_keys.insert(key);
            }
        }
    } else {
        a = &it->second;
    }
    if (!a) {
        return 0;
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
