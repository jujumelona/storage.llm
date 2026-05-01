// Device memory management and accounting

// BUGFIX 76: Issue 11 - Document KV cache vs VRAM budget separation ??
// Clarification: KV cache is RAM-only, NOT included in VRAM budget
// - KV cache: CPU RAM (moe_kv_cache in generation_kv_cache.cpp.inc)
// - VRAM budget: GPU memory for expert weights only (device_allocations map)
// - QKV state (engine_core/kv/kv_qkv.cpp): Also CPU RAM, ~0.69 GiB
// - device_budget_bytes() calculates VRAM for experts, does NOT include KV/QKV
// - Separate accounting: tier_used(moe_TIER_RAM) for KV, device_allocated_bytes for VRAM
// - No interaction: KV growth does not reduce expert cache space
// Helper to encapsulate VRAM budget accounting.
// Do not modify engine->device_free_bytes directly.
static void device_account_bytes_freed(moe_pc_engine* e, uint64_t bytes) {
    if (!e) return;
    if (e->backend_caps.backend == moe_BACKEND_METAL) return; // budget computed dynamically
    e->device_free_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

static uint64_t device_budget_bytes(const moe_pc_engine* e) {
    if (!e) return 0;
    if (e->config.vram_budget_bytes) return e->config.vram_budget_bytes;
    if (e->backend_caps.supports_unified_memory &&
        e->backend_caps.backend == moe_BACKEND_METAL) {
        const uint64_t sys = e->backend_caps.system_memory_bytes > 0
            ? e->backend_caps.system_memory_bytes : 16ull * 1024 * 1024 * 1024;
        const uint64_t budget = (sys * 60ull) / 100ull;
        const uint64_t ram_used = e->ram_used_bytes.load(std::memory_order_relaxed);
        return budget > ram_used ? budget - ram_used : 0;
    }
    const uint64_t allocated = e->device_allocated_bytes.load(std::memory_order_relaxed);
    const uint64_t free_b = e->device_free_bytes.load(std::memory_order_relaxed);
    if (allocated || free_b) {
        // BUGFIX 115: Handle transient underflows from wait-free saturating subtract
        int64_t s_free = (int64_t)free_b;
        if (s_free < 0) s_free = 0;
        return allocated > UINT64_MAX - (uint64_t)s_free ? UINT64_MAX : allocated + (uint64_t)s_free;
    }
    return e->backend_caps.device_memory_bytes ? e->backend_caps.device_memory_bytes : e->backend_caps.recommended_vram_cache_bytes;
}

static int device_try_reserve_pending_vram(moe_pc_engine* e, uint64_t bytes) {
    if (!e || bytes == 0) {
        return 0;
    }
    const uint64_t budget = device_budget_bytes(e);
    if (budget == 0) {
        return 1;
    }
    uint64_t pending = e->pending_vram_reservation.load(std::memory_order_relaxed);
    for (;;) {
        const uint64_t allocated = e->device_allocated_bytes.load(std::memory_order_relaxed);
        if (allocated > budget || bytes > budget - allocated) {
            return 0;
        }
        const uint64_t remaining = budget - allocated;
        if (pending > remaining || bytes > remaining - pending) {
            return 0;
        }
        if (e->pending_vram_reservation.compare_exchange_weak(
                pending,
                pending + bytes,
                std::memory_order_relaxed,
                std::memory_order_relaxed)) {
            return 1;
        }
    }
}

static void device_release_pending_vram_reservation(moe_pc_engine* e, uint64_t bytes) {
    if (!e || bytes == 0) {
        return;
    }
    const uint64_t old_val = e->pending_vram_reservation.fetch_sub(bytes, std::memory_order_relaxed);
    if (old_val < bytes) {
        e->pending_vram_reservation.fetch_add(bytes - old_val, std::memory_order_relaxed);
    }
}

static void device_refresh_mem_info(moe_pc_engine* e) {
    if (!e) return;
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
    int mem_queried = 0;
    if (e && e->cuda.cuMemGetInfo && device_set_current(e) && e->cuda.cuMemGetInfo(&free_b, &total_b) == 0) {
        mem_queried = 1;
    } else if (e && e->hip.hipMemGetInfo && device_set_current(e) && e->hip.hipMemGetInfo(&free_b, &total_b) == 0) {
        mem_queried = 1;
    } else if (e && e->backend_caps.backend == moe_BACKEND_VULKAN && e->vk.vkGetPhysicalDeviceMemoryProperties2 && e->vk.physicalDevice) {
        // BUGFIX 115: Vulkan memory query (when fully implemented)
        // VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{};
        // budget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
        // VkPhysicalDeviceMemoryProperties2 props{};
        // props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
        // props.pNext = &budget;
        // e->vk.vkGetPhysicalDeviceMemoryProperties2(e->vk.physicalDevice, &props);
        // Sum budget.heapBudget[i] for device-local heaps
        // free_b = calculated_free; total_b = calculated_total; mem_queried = 1;
    } else if (e && e->backend_caps.backend == moe_BACKEND_LEVEL_ZERO && e->ze.zeDeviceGetMemoryProperties && e->ze.device) {
        // BUGFIX 115: Level Zero memory query (when fully implemented)
        // ze_device_memory_properties_t mem_props{};
        // mem_props.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
        // uint32_t count = 1;
        // e->ze.zeDeviceGetMemoryProperties(e->ze.device, &count, &mem_props);
        // free_b = mem_props.totalSize; total_b = mem_props.totalSize; mem_queried = 1;
    }
    if (mem_queried) {
        // Delta sync for device_free_bytes to recover from accounting drift
        const uint64_t tracked_free = e->device_free_bytes.load(std::memory_order_relaxed);
        const int64_t delta = (int64_t)free_b - (int64_t)tracked_free;
        if (delta > 0) {
            e->device_free_bytes.fetch_add((uint64_t)delta, std::memory_order_relaxed);
        } else if (delta < 0) {
            const uint64_t neg_delta = (uint64_t)(-delta);
            const uint64_t old = e->device_free_bytes.fetch_sub(neg_delta, std::memory_order_relaxed);
            if (old < neg_delta) {
                e->device_free_bytes.fetch_add(neg_delta - old, std::memory_order_relaxed);
            }
        }
        e->device_total_bytes.store((uint64_t)total_b);
    }
}

// BUGFIX 54: Issue 29 - Replace CAS spin with fetch_sub ?�★
// Problem: device_account_free_bytes_sub uses CAS loop for saturating subtract.
// Called on every allocation ??high-frequency contention. Same issue as total_queue_depth.
// Solution: Use fetch_sub with post-correction. device_refresh_mem_info runs every 1s
// and corrects drift, so temporary underflow is acceptable. Eliminates CAS contention.
static void device_account_free_bytes_sub(moe_pc_engine* e, uint64_t bytes) {
    if (!e || bytes == 0) {
        return;
    }
    if (e->backend_caps.backend == moe_BACKEND_METAL) return; // budget computed dynamically
    
    // Fast path: atomic subtract without CAS loop
    const uint64_t old_val = e->device_free_bytes.fetch_sub(bytes, std::memory_order_relaxed);

    // BUGFIX 115: Fix wait-free saturating subtract data race ??
    // Problem: If old_val < bytes, store(0) overwrites any concurrent fetch_add,
    // permanently leaking VRAM budget (memory leak in accounting).
    // The previous author claimed device_refresh_mem_info corrects this, but they
    // explicitly removed the cuMemGetInfo sync in Bug 1! So the corruption is permanent.
    // Solution: Instead of store(0), add back the exact amount of over-subtraction.
    // This preserves concurrent additions while correctly saturating at 0.
    if (old_val < bytes) {
        e->device_free_bytes.fetch_add(bytes - old_val, std::memory_order_relaxed);
    }
}

static void device_free_deferred_ptrs(moe_pc_engine* e, std::vector<std::pair<CUdeviceptr, uint64_t>>& ptrs) {
    if (!e || ptrs.empty() ) {
        ptrs.clear();
        return;
    }
    device_set_current(e);
    for (auto& pair : ptrs) {
        if (pair.first) {
#if defined(__APPLE__) && defined(__MACH__)
            if (e->backend_caps.backend == moe_BACKEND_METAL) {
                extern "C" void metal_zero_copy_unmap(void* buffer);
                metal_zero_copy_unmap((void*)pair.first);
            } else
#endif
            if (e->backend_caps.backend == moe_BACKEND_HIP && e->hip.hipFree) {
                e->hip.hipFree(pair.first);
            } else if (e->backend_caps.backend == moe_BACKEND_CUDA && e->cuda.cuMemFree) {
                e->cuda.cuMemFree(pair.first);
            }
            // Bug 2: device_free_bytes already updated in device_release_allocation_locked
            // Do NOT double-add here
            // e->device_free_bytes.fetch_add(pair.second, std::memory_order_relaxed);  // REMOVED
        }
    }
    ptrs.clear();
}

static void device_release_allocation_locked(
    moe_pc_engine* e,
    std::unordered_map<uint64_t, moe_device_allocation>::iterator it,
    std::vector<std::pair<CUdeviceptr, uint64_t>>* deferred_free_ptrs = nullptr
) {
    if (!e || it == e->device_allocations.end()) return;
    const moe_device_allocation a = it->second;
    if (a.ptr) {
        if (deferred_free_ptrs) {
            deferred_free_ptrs->push_back({a.ptr, a.bytes});
            // Bug 2: Update device_free_bytes immediately to maintain accurate budget
            // Without this, device_budget_bytes() underestimates during deferred free window
            device_account_bytes_freed(e, a.bytes);
        } else {
            device_set_current(e);
            if (1) {
#if defined(__APPLE__) && defined(__MACH__)
                if (e->backend_caps.backend == moe_BACKEND_METAL) {
                    extern "C" void metal_zero_copy_unmap(void* buffer);
                    metal_zero_copy_unmap((void*)a.ptr);
                } else
#endif
                if (e->backend_caps.backend == moe_BACKEND_HIP && e->hip.hipFree) {
                    e->hip.hipFree(a.ptr);
                } else if (e->backend_caps.backend == moe_BACKEND_CUDA && e->cuda.cuMemFree) {
                    e->cuda.cuMemFree(a.ptr);
                }
                device_account_bytes_freed(e, a.bytes);
            }
        }
    }
    e->device_allocated_bytes.fetch_sub(a.bytes);
    e->device_allocation_count.fetch_sub(1);
    if (auto* c = device_pool_counter(e, a.pool)) c->fetch_sub(a.bytes);
    if (a.pool == moe_DEVICE_POOL_ACTIVATION) {
        e->activation_keys.erase(it->first);
        e->device_activation_count.fetch_sub(1, std::memory_order_relaxed);
    }
    e->device_allocations.erase(it);
    // Fix 50: Removed device_refresh_mem_info from under device_mutex
    // cuMemGetInfo is a slow driver call that blocks all alloc/free while held.
    // Callers should invoke device_refresh_mem_info() after releasing mutex.
}

static void device_free_locked(
    moe_pc_engine* e,
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

static void device_free_pool(moe_pc_engine* e, uint8_t pool) {
    if (!e || e->device_allocator_ready.load() <= 0) return;
    std::vector<std::pair<CUdeviceptr, uint64_t>> deferred_free_ptrs;
    std::vector<uint64_t> keys;
    {
        std::lock_guard<std::mutex> lock(e->device_mutex);
        for (const auto& it : e->device_allocations) if (it.second.pool == pool) keys.push_back(it.first);
        for (uint64_t key : keys) device_free_locked(e, key, &deferred_free_ptrs);
        if (pool == moe_DEVICE_POOL_ACTIVATION) {
            decltype(e->activation_evict_heap) empty;
            e->activation_evict_heap.swap(empty);
        }
    }
    device_free_deferred_ptrs(e, deferred_free_ptrs);
}



