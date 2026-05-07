// Device tensor allocation preparation

static int device_prepare_tensor_allocation(
    moe_pc_engine* e,
    uint64_t key,
    uint64_t bytes,
    uint8_t pool,
    int async_copy,
    CUdeviceptr* out_dst,
    const void* host_ptr = nullptr,
    const vram_reservation_token& reservation = vram_reservation_token{}
) {
    // Validate staging state before any slot access.
    // Problem: after reset_staging_slots timeout, staging_slot_size is 0 but vector entries may remain.
    // Code that checks slot_index < staging_slots.size() passes, but slot.buf may be nullptr
    // Solution: Check staging_slot_size > 0 before using any slot
    // BUGFIX 760: Check bytes for SIZE_MAX overflow.
    if (!e || !out_dst || bytes > SIZE_MAX) {
        return 0;
    }
    // BUGFIX 761: Check bytes for zero ??紐꾧께
    if (bytes == 0) {
        return 0;
    }
    // BUGFIX 847: Check key for zero ??紐꾧께
    if (key == 0) {
        return 0;
    }
    if (pool == moe_DEVICE_POOL_EXPERT &&
        (!reservation.valid || reservation.pool != pool || reservation.bytes < bytes)) {
        return 0;
    }
    (void)async_copy;
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
        if (!a) {
            moe_device_allocation next;
            next.key = key;
            next.bytes = bytes;
            next.pool = pool;
            next.ptr = 0;
            next.backend_kind = moe_DEVICE_ALLOC_NATIVE_PTR;
            next.backend_aux = 0;
            auto try_device_alloc = [&]() -> int {
                next.ptr = 0;
#if defined(__APPLE__) && defined(__MACH__)
                if (e->backend_caps.backend == moe_BACKEND_METAL) {
                    extern "C" void* metal_buffer_alloc(void* device_handle, uint64_t bytes);
                    next.ptr = (CUdeviceptr)metal_buffer_alloc(nullptr, bytes);
                    return (next.ptr != 0) ? 1 : 0;
                } else
#endif
                if (e->backend_caps.backend == moe_BACKEND_HIP && e->hip.hipMalloc) {
                    return (e->hip.hipMalloc(&next.ptr, (size_t)bytes) == 0) ? 1 : 0;
                } else if (e->backend_caps.backend == moe_BACKEND_CUDA && e->cuda.cuMemAlloc) {
                    return (e->cuda.cuMemAlloc(&next.ptr, (size_t)bytes) == 0) ? 1 : 0;
                } else if (e->backend_caps.backend == moe_BACKEND_LEVEL_ZERO &&
                           e->ze.context && e->ze.device && e->ze.zeMemAllocDevice) {
                    struct ze_device_mem_alloc_desc_t_local {
                        uint32_t stype;
                        const void* pNext;
                        uint32_t flags;
                        uint32_t ordinal;
                    } desc{0x00010002u, nullptr, 0u, 0u};
                    void* ptr = nullptr;
                    if (e->ze.zeMemAllocDevice(
                            e->ze.context,
                            &desc,
                            static_cast<size_t>(bytes),
                            4096u,
                            e->ze.device,
                            &ptr) == 0 && ptr) {
                        next.ptr = reinterpret_cast<CUdeviceptr>(ptr);
                        return 1;
                    }
                } else if (e->backend_caps.backend == moe_BACKEND_VULKAN &&
                           moe_vulkan_transfer_runtime_ready(e)) {
                    void* buffer = nullptr;
                    void* memory = nullptr;
                    if (moe_vulkan_create_device_buffer(e, bytes, &buffer, &memory) &&
                        buffer && memory) {
                        next.ptr = reinterpret_cast<CUdeviceptr>(buffer);
                        next.backend_kind = moe_DEVICE_ALLOC_VULKAN_BUFFER;
                        next.backend_aux = reinterpret_cast<uint64_t>(memory);
                        return 1;
                    }
                } else if (e->backend_caps.backend == moe_BACKEND_DIRECTML &&
                           e->runtime_directstorage_gpu_ready.load(std::memory_order_acquire)) {
                    void* resource = nullptr;
                    if (directstorage_create_d3d12_buffer_resource(e, bytes, &resource) && resource) {
                        next.ptr = reinterpret_cast<CUdeviceptr>(resource);
                        next.backend_kind = moe_DEVICE_ALLOC_D3D12_RESOURCE;
                        next.backend_aux = 0;
                        return 1;
                    }
                }
                return 0;
            };
            int alloc_ok = try_device_alloc();
            if (!alloc_ok) {
                e->io_atomic.device_alloc_failures.fetch_add(1, std::memory_order_relaxed);
                e->vram_eviction_needed.store(true, std::memory_order_relaxed);
                e->vram_eviction_cv.notify_one();
                lock.unlock();
                device_refresh_mem_info(e, 1);
                device_free_pool(e, moe_DEVICE_POOL_STREAM);
                device_free_pool(e, moe_DEVICE_POOL_ACTIVATION);
                device_refresh_mem_info(e, 1);
                lock.lock();
                device_make_room_locked(e, pool, bytes, lock, deferred_free_ptrs);
                if (!deferred_free_ptrs.empty()) {
                    lock.unlock();
                    device_free_deferred_ptrs(e, deferred_free_ptrs);
                    deferred_free_ptrs.clear();
                    device_refresh_mem_info(e, 1);
                    lock.lock();
                }
                alloc_ok = try_device_alloc();
            }
            if (!alloc_ok) {
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
            a = &e->device_allocations.emplace(key, next).first->second;
            if (pool == moe_DEVICE_POOL_ACTIVATION) {
                e->activation_keys.insert(key);
                e->device_activation_count.fetch_add(1, std::memory_order_relaxed);
            }
        }
    } else {
        a = &it->second;
    }
    auto final_it = e->device_allocations.find(key);
    if (!a ||
        final_it == e->device_allocations.end() ||
        final_it->second.pending_free ||
        final_it->second.bytes < bytes ||
        final_it->second.ptr == 0) {
        return 0;
    }
    if (!device_mark_copy_pending_locked(e, key)) {
        device_free_locked(e, key, &deferred_free_ptrs);
        lock.unlock();
        device_free_deferred_ptrs(e, deferred_free_ptrs);
        return 0;
    }
    const CUdeviceptr ptr_snapshot = final_it->second.ptr;
    lock.unlock();

    const uint64_t raw_budget = device_raw_budget_bytes(e);
    const uint64_t used =
        e->vram_resident_bytes.load(std::memory_order_relaxed) +
        e->vram_inflight_bytes.load(std::memory_order_relaxed) +
        e->vram_kv_bytes.load(std::memory_order_relaxed) +
        e->vram_common_bytes.load(std::memory_order_relaxed);
    const uint64_t threshold_85 = (raw_budget / 100ull) * 85ull;
    if (raw_budget > 0 && threshold_85 > 0 && used >= threshold_85) {
        e->vram_eviction_needed.store(true, std::memory_order_relaxed);
        e->vram_eviction_cv.notify_one();
    }

    *out_dst = ptr_snapshot;
    return ptr_snapshot != 0;
}
