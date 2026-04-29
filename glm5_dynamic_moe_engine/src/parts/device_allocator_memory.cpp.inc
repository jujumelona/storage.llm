// Device memory management and accounting

static uint64_t device_budget_bytes(const glm5_pc_engine* e) {
    if (!e) return 0;
    
    // Problem 3 Fix: Account for external VRAM usage in budget calculation
    // External usage = VRAM used by other processes, CUDA JIT cache, NCCL, etc.
    // Without this, device_budget_bytes() overestimates available VRAM → cuMemAlloc fails
    const uint64_t external_usage = e->device_external_usage_estimate.load(std::memory_order_relaxed);
    const uint64_t total_bytes = e->device_total_bytes.load(std::memory_order_relaxed);
    
    if (e->config.vram_budget_bytes) {
        // User-specified budget: clamp to actual available VRAM minus external usage
        const uint64_t available = total_bytes > external_usage ? total_bytes - external_usage : 0;
        return std::min(e->config.vram_budget_bytes, available);
    }
    
    const uint64_t allocated = e->device_allocated_bytes.load(std::memory_order_relaxed);
    const uint64_t free_b = e->device_free_bytes.load(std::memory_order_relaxed);
    if (allocated || free_b) {
        const uint64_t engine_total = allocated > UINT64_MAX - free_b ? UINT64_MAX : allocated + free_b;
        // Subtract external usage from total available
        const uint64_t available = total_bytes > external_usage ? total_bytes - external_usage : engine_total;
        return std::min(engine_total, available);
    }
    
    const uint64_t fallback = e->backend_caps.device_memory_bytes ? 
        e->backend_caps.device_memory_bytes : e->backend_caps.recommended_vram_cache_bytes;
    const uint64_t available = total_bytes > external_usage ? total_bytes - external_usage : fallback;
    return std::min(fallback, available);
}

static void device_refresh_mem_info(glm5_pc_engine* e) {
    if (!e) return;
    // Problem 3 Fix: Track external VRAM usage to prevent cuMemAlloc failures
    // Diagnostics/failure-path refresh only. Hot paths maintain device_free_bytes directly.
    const auto now = std::chrono::steady_clock::now();
    int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    int64_t last = e->last_mem_query_ms.load(std::memory_order_relaxed);
    if (now_ms - last < 1000) {
        return;
    }
    if (!e->last_mem_query_ms.compare_exchange_strong(last, now_ms)) {
        return;
    }
    size_t free_b = 0, total_b = 0;
    if (e && e->cuda.cuMemGetInfo && device_set_current(e) && e->cuda.cuMemGetInfo(&free_b, &total_b) == 0) {
        // Bug 1: Do NOT overwrite device_free_bytes — it's maintained by alloc/free counters
        // Overwriting causes race with concurrent alloc/free, leading to budget miscalculation
        // e->device_free_bytes.store((uint64_t)free_b);  // REMOVED
        
        // Problem 3 Fix: Estimate external VRAM usage (other processes, CUDA JIT, NCCL, etc.)
        // external_usage = (device_total_bytes - cuMemGetInfo.free) - device_allocated_bytes
        // This captures VRAM used outside this engine's tracking.
        const uint64_t allocated_by_engine = e->device_allocated_bytes.load(std::memory_order_relaxed);
        const uint64_t total_used = total_b > free_b ? total_b - free_b : 0;
        const uint64_t external_usage = total_used > allocated_by_engine ? total_used - allocated_by_engine : 0;
        e->device_external_usage_estimate.store(external_usage, std::memory_order_relaxed);
        e->device_total_bytes.store((uint64_t)total_b);
    }
}

static void device_account_free_bytes_sub(glm5_pc_engine* e, uint64_t bytes) {
    if (!e || bytes == 0) {
        return;
    }
    uint64_t current = e->device_free_bytes.load(std::memory_order_relaxed);
    for (;;) {
        const uint64_t next = current > bytes ? current - bytes : 0;
        if (e->device_free_bytes.compare_exchange_weak(
                current,
                next,
                std::memory_order_relaxed,
                std::memory_order_relaxed)) {
            return;
        }
    }
}

static void device_free_deferred_ptrs(glm5_pc_engine* e, std::vector<std::pair<CUdeviceptr, uint64_t>>& ptrs) {
    if (!e || ptrs.empty() || !e->cuda.cuMemFree) {
        ptrs.clear();
        return;
    }
    device_set_current(e);
    for (auto& pair : ptrs) {
        if (pair.first) {
            e->cuda.cuMemFree(pair.first);
            // Bug 2: device_free_bytes already updated in device_release_allocation_locked
            // Do NOT double-add here
            // e->device_free_bytes.fetch_add(pair.second, std::memory_order_relaxed);  // REMOVED
        }
    }
    ptrs.clear();
}

static void device_release_allocation_locked(
    glm5_pc_engine* e,
    std::unordered_map<uint64_t, glm5_device_allocation>::iterator it,
    std::vector<std::pair<CUdeviceptr, uint64_t>>* deferred_free_ptrs = nullptr
) {
    if (!e || it == e->device_allocations.end()) return;
    const glm5_device_allocation a = it->second;
    if (a.ptr) {
        if (deferred_free_ptrs) {
            deferred_free_ptrs->push_back({a.ptr, a.bytes});
            // Bug 2: Update device_free_bytes immediately to maintain accurate budget
            // Without this, device_budget_bytes() underestimates during deferred free window
            e->device_free_bytes.fetch_add(a.bytes, std::memory_order_relaxed);
        } else {
            device_set_current(e);
            if (e->cuda.cuMemFree) {
                e->cuda.cuMemFree(a.ptr);
                e->device_free_bytes.fetch_add(a.bytes, std::memory_order_relaxed);
            }
        }
    }
    e->device_allocated_bytes.fetch_sub(a.bytes);
    e->device_allocation_count.fetch_sub(1);
    if (auto* c = device_pool_counter(e, a.pool)) c->fetch_sub(a.bytes);
    if (a.pool == GLM5_DEVICE_POOL_ACTIVATION) {
        e->activation_keys.erase(it->first);
    }
    e->device_allocations.erase(it);
    // Fix 50: Removed device_refresh_mem_info from under device_mutex
    // cuMemGetInfo is a slow driver call that blocks all alloc/free while held.
    // Callers should invoke device_refresh_mem_info() after releasing mutex.
}

static void device_free_locked(
    glm5_pc_engine* e,
    uint64_t key,
    std::vector<std::pair<CUdeviceptr, uint64_t>>* deferred_free_ptrs = nullptr
) {
    auto it = e->device_allocations.find(key);
    if (it == e->device_allocations.end()) return;
    if (it->second.inflight_copies != 0) {
        it->second.pending_free = true;
        return;
    }
    device_release_allocation_locked(e, it, deferred_free_ptrs);
}

static void device_free_pool(glm5_pc_engine* e, uint8_t pool) {
    if (!e || e->device_allocator_ready.load() <= 0) return;
    std::vector<std::pair<CUdeviceptr, uint64_t>> deferred_free_ptrs;
    std::vector<uint64_t> keys;
    {
        std::lock_guard<std::mutex> lock(e->device_mutex);
        for (const auto& it : e->device_allocations) if (it.second.pool == pool) keys.push_back(it.first);
        for (uint64_t key : keys) device_free_locked(e, key, &deferred_free_ptrs);
        if (pool == GLM5_DEVICE_POOL_ACTIVATION) {
            decltype(e->activation_evict_heap) empty;
            e->activation_evict_heap.swap(empty);
        }
    }
    device_free_deferred_ptrs(e, deferred_free_ptrs);
}
