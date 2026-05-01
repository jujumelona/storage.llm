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

// BUGFIX 124: Issue 29 - Metal unified memory budget double-counted ★★
// Problem: Metal VRAM budget subtracts ram_used_bytes, but RAM budget also tracks ram_used_bytes.
// Same 4GB data consumes 4GB from VRAM budget + 4GB from RAM budget = 8GB total.
// Solution: Metal should use unified budget only. Set RAM budget to 0 in device_allocator_init.
// All expert data tracked in "VRAM" budget (actually unified memory budget).
// Trade-off: Relies on OS unified memory pressure management instead of explicit RAM eviction.
// Metal handles this natively, so no functional loss.

// BUGFIX 199: Metal VRAM budget configurable ★
// Problem: 60% hardcoded → M3 Max 128GB wastes 25GB (60%=76.8GB vs 80%=102.4GB)
// Solution: Use io_config.metal_memory_budget_pct (default 60, safe range 60-85)
// Trade-off: >85% risks macOS memory pressure → swapping or app termination
static uint64_t device_budget_bytes(const moe_pc_engine* e) {
    if (!e) return 0;
    if (e->config.vram_budget_bytes) return e->config.vram_budget_bytes;
    if (e->backend_caps.supports_unified_memory &&
        e->backend_caps.backend == moe_BACKEND_METAL) {
        // BUGFIX 124: Metal unified memory - do NOT subtract ram_used_bytes here.
        // RAM budget is set to 0 in device_allocator_init, so all tracking is in VRAM budget.
        const uint64_t sys = e->backend_caps.system_memory_bytes > 0
            ? e->backend_caps.system_memory_bytes : 16ull * 1024 * 1024 * 1024;
        // BUGFIX 199: Use configurable percentage (default 60)
        // BUGFIX 232: Ensure pct is never 0 to prevent zero budget
        const uint64_t pct = (e->io_config.metal_memory_budget_pct > 0 && e->io_config.metal_memory_budget_pct <= 100)
            ? e->io_config.metal_memory_budget_pct : 60ull;
        const uint64_t budget = (sys * pct) / 100ull;
        if (budget == 0) {
            return (sys * 60ull) / 100ull; // Fallback to 60% if calculation results in 0
        }
        return budget;
    }
    const uint64_t allocated = e->device_allocated_bytes.load(std::memory_order_relaxed);
    const uint64_t free_b = e->device_free_bytes.load(std::memory_order_relaxed);
    if (allocated || free_b) {
        // BUGFIX 122: With CAS loop in device_try_reserve_pending_vram, underflow is eliminated.
        // Keep clamp for safety, but it should never trigger now.
        int64_t s_free = (int64_t)free_b;
        if (s_free < 0) s_free = 0;
        return allocated > UINT64_MAX - (uint64_t)s_free ? UINT64_MAX : allocated + (uint64_t)s_free;
    }
    return e->backend_caps.device_memory_bytes ? e->backend_caps.device_memory_bytes : e->backend_caps.recommended_vram_cache_bytes;
}

// BUGFIX 122: Issue 27 - pending VRAM reservation TOCTOU race ★★★
// Problem: Two threads can both pass check at step 2, then both fetch_add at step 3.
// If both exceed budget at step 4, both fetch_sub, causing transient underflow.
// device_budget_bytes() clamps underflow to 0, but this is a workaround, not a fix.
// Solution: Replace check-then-add with CAS loop for atomic reservation.
// Trade-off: CAS contention possible, but VRAM allocation is infrequent (expert triplet granularity).
// Max 8 GPU workers, so contention is low. Eliminates underflow bug permanently.

// BUGFIX 198: Dead code - DO NOT CALL ★
// This function is ONLY called from device_tensor_copy_alloc.cpp.inc line 89.
// pending_vram_reservation is ALSO managed by prefetch_admission.cpp.inc CAS loop.
// Two independent reservation systems sharing the same atomic → double-counting risk.
// If this function is called elsewhere, VRAM budget will be over-reserved.
// Status: Currently used in one location. If removed from there, mark [[deprecated]].
static int device_try_reserve_pending_vram(moe_pc_engine* e, uint64_t bytes) {
    if (!e || bytes == 0) {
        return 0;
    }
    const uint64_t budget = device_budget_bytes(e);
    if (budget == 0) {
        return 1;
    }
    // CAS loop: atomically reserve only if budget allows
    uint64_t cur = e->pending_vram_reservation.load(std::memory_order_relaxed);
    do {
        const uint64_t alloc_now = e->device_allocated_bytes.load(std::memory_order_acquire);
        if (alloc_now + cur + bytes > budget) {
            return 0; // Would exceed budget
        }
    } while (!e->pending_vram_reservation.compare_exchange_weak(
        cur, cur + bytes,
        std::memory_order_acq_rel, std::memory_order_relaxed));
    return 1;
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
        // BUGFIX 115: Vulkan memory query
        struct VkPhysicalDeviceMemoryBudgetPropertiesEXT_local {
            uint32_t sType;
            void* pNext;
            uint64_t heapUsage[16];
            uint64_t heapBudget[16];
        } budget{};
        budget.sType = 1000237000; // VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT
        struct VkMemoryHeap_local {
            uint64_t size;
            uint32_t flags;
        };
        struct VkPhysicalDeviceMemoryProperties_local {
            uint32_t memoryTypeCount;
            struct { uint32_t propertyFlags; uint32_t heapIndex; } memoryTypes[32];
            uint32_t memoryHeapCount;
            VkMemoryHeap_local memoryHeaps[16];
        };
        struct VkPhysicalDeviceMemoryProperties2_local {
            uint32_t sType;
            void* pNext;
            VkPhysicalDeviceMemoryProperties_local memoryProperties;
        } props{};
        props.sType = 1000059006; // VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2
        props.pNext = &budget;
        e->vk.vkGetPhysicalDeviceMemoryProperties2(e->vk.physicalDevice, &props);
        uint64_t calc_free = 0;
        uint64_t calc_total = 0;
        for (uint32_t i = 0; i < props.memoryProperties.memoryHeapCount; ++i) {
            if (props.memoryProperties.memoryHeaps[i].flags & 1) {
                calc_total += props.memoryProperties.memoryHeaps[i].size;
                if (budget.heapBudget[i] > 0) {
                    calc_free += (budget.heapBudget[i] > budget.heapUsage[i]) ? (budget.heapBudget[i] - budget.heapUsage[i]) : 0;
                } else {
                    // Fallback to 70% if VK_EXT_memory_budget is missing
                    calc_free += (props.memoryProperties.memoryHeaps[i].size * 7) / 10;
                }
            }
        }
        if (calc_total > 0) {
            free_b = calc_free;
            total_b = calc_total;
            mem_queried = 1;
        }
    } else if (e && e->backend_caps.backend == moe_BACKEND_LEVEL_ZERO && e->ze.zeDeviceGetMemoryProperties && e->ze.device) {
        // BUGFIX 115: Level Zero memory query
        struct ze_device_memory_properties_t_local {
            uint32_t stype;
            void* pNext;
            uint32_t flags;
            uint32_t maxClockRate;
            uint32_t maxBusWidth;
            uint64_t totalSize;
            char name[256];
        };
        struct ze_device_memory_ext_properties_t_local {
            uint32_t stype;
            void* pNext;
            uint32_t type;
            uint64_t physicalSize;
            uint64_t allocatedSize;
        } ext_props{};
        ext_props.stype = 0x0001000B; // ZE_STRUCTURE_TYPE_DEVICE_MEMORY_EXT_PROPERTIES
        ze_device_memory_properties_t_local mem_props{};
        mem_props.stype = 0x1; // ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES
        mem_props.pNext = &ext_props;
        uint32_t count = 1;
        if (e->ze.zeDeviceGetMemoryProperties(e->ze.device, &count, &mem_props) == 0) {
            total_b = mem_props.totalSize;
            if (ext_props.allocatedSize > 0) {
                free_b = (mem_props.totalSize > ext_props.allocatedSize) ? (mem_props.totalSize - ext_props.allocatedSize) : mem_props.totalSize;
            } else {
                // Fallback to 70% if allocatedSize not populated
                free_b = (mem_props.totalSize * 7) / 10;
            }
            mem_queried = 1;
        }
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



