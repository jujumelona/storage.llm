// Device tensor allocation preparation

static int device_prepare_tensor_allocation(
    moe_pc_engine* e,
    uint64_t key,
    uint64_t bytes,
    uint8_t pool,
    int async_copy,
    CUdeviceptr* out_dst,
    const void* host_ptr = nullptr
) {
    if (!e || !out_dst) {
        return 0;
    }
    *out_dst = 0;
    device_refresh_mem_info(e);
    std::vector<std::pair<CUdeviceptr, uint64_t>> deferred_free_ptrs;
    std::unique_lock<std::mutex> lock(e->device_mutex);
    moe_device_allocation* a = nullptr;
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
            // BUGFIX 116: Clear deferred_free_ptrs after first free to prevent double-free
            deferred_free_ptrs.clear();
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
        // BUGFIX 243: Remove VRAM double-counting ★★★
        // Problem: prefetch_admitted_tier already reserved VRAM via pending_vram_reservation CAS loop.
        // Calling device_try_reserve_pending_vram here AGAIN causes double-counting:
        // Same expert reserves VRAM twice → budget depletes 2x faster → false "VRAM full" errors.
        // Solution: Remove device_try_reserve_pending_vram call entirely. Admission layer already reserved.
        // Trade-off: Relies on admission layer to always reserve before calling this function.
        // This is guaranteed by prefetch_plan_execute.cpp.inc Pass 1 admission logic.
        // Note: BUGFIX 198 already flagged this as "Dead code - DO NOT CALL".
        // Original code:
        //   if (!a && !device_try_reserve_pending_vram(e, bytes)) {
        //       e->io_atomic.device_alloc_failures.fetch_add(1, std::memory_order_relaxed);
        //       e->io_atomic.staging_wraps.fetch_add(1);
        //       return 0;
        //   }
        // Removed to prevent double-counting.
        if (!a) {
            moe_device_allocation next;
            next.key = key;
            next.bytes = bytes;
            next.pool = pool;
            int alloc_ok = 0;
#if defined(__APPLE__) && defined(__MACH__)
            if (e->backend_caps.backend == moe_BACKEND_METAL && host_ptr) {
                extern "C" void* metal_zero_copy_map(void* device_handle, void* src, uint64_t bytes);
                next.ptr = (CUdeviceptr)metal_zero_copy_map(e, const_cast<void*>(host_ptr), bytes);
                alloc_ok = (next.ptr != 0) ? 1 : 0;
            } else
#endif
            if (e->backend_caps.backend == moe_BACKEND_HIP && e->hip.hipMalloc) {
                alloc_ok = (e->hip.hipMalloc(&next.ptr, (size_t)bytes) == 0) ? 1 : 0;
            } else if (e->backend_caps.backend == moe_BACKEND_CUDA && e->cuda.cuMemAlloc) {
                alloc_ok = (e->cuda.cuMemAlloc(&next.ptr, (size_t)bytes) == 0) ? 1 : 0;
            }
            if (!alloc_ok) {
                // NOTE: pending_vram_reservation은 prefetch_plan_execute의 Pass 2에서
                // 이미 해제됨. 여기서 별도 해제 불필요 (BUGFIX 243 참고)
                e->io_atomic.device_alloc_failures.fetch_add(1, std::memory_order_relaxed);
                lock.unlock();
                device_refresh_mem_info(e, 1);
                return 0;
            }
            e->device_allocated_bytes.fetch_add(bytes);
            device_account_free_bytes_sub(e, bytes);
            e->device_allocation_count.fetch_add(1);
            if (auto* c = device_pool_counter(e, pool)) {
                c->fetch_add(bytes);
            }
        // Fix 64: cuMemGetInfo is slow driver call →do NOT call under device_mutex
            const uint64_t bud = device_budget_bytes(e);
            // BUGFIX 99: Precompute percentage threshold
            const uint64_t bud_threshold_85 = (bud * 85ull) / 100ull;
            if (bud > 0 && e->device_allocated_bytes.load(std::memory_order_relaxed) > bud_threshold_85) {
                // BUGFIX 90: Fix Lost Wakeup by setting vram_eviction_needed before notify
                e->vram_eviction_needed.store(true, std::memory_order_relaxed);
                e->vram_eviction_cv.notify_one();
            }
            a = &e->device_allocations.emplace(key, next).first->second;
            if (pool == moe_DEVICE_POOL_ACTIVATION) {
                e->activation_keys.insert(key);
                e->device_activation_count.fetch_add(1, std::memory_order_relaxed);
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
