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

// BUGFIX 124: Issue 29 - Metal unified memory budget double-counted ★★
// Problem: Metal VRAM budget subtracts ram_used_bytes, but RAM budget also tracks ram_used_bytes.
// Same 4GB data consumes 4GB from VRAM budget + 4GB from RAM budget = 8GB total.
// Solution: Metal should use unified budget only. Set RAM budget to 0 in device_allocator_init.
// All expert data tracked in "VRAM" budget (actually unified memory budget).
// Trade-off: Relies on OS unified memory pressure management instead of explicit RAM eviction.
// Metal handles this natively, so no functional loss.

// BUGFIX 199: Metal VRAM budget configurable ★
// Problem: 60% hardcoded → M3 Max 128GB wastes 25GB (60%=76.8GB vs 80%=102.4GB)
// Solution: Use io_config.metal_memory_budget_pct (0=auto, safe range 50-90)
// Trade-off: >85% risks macOS memory pressure → swapping or app termination
static void warn_device_budget_exhausted(
    const moe_pc_engine* e,
    uint64_t raw_budget,
    uint64_t reserved_total,
    uint64_t pending,
    uint64_t kv_reserved,
    uint64_t common_reserved
) {
    static std::atomic<int64_t> s_last_warn_ms{0};
    const auto now = std::chrono::steady_clock::now();
    const int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    int64_t last = s_last_warn_ms.load(std::memory_order_relaxed);
    if (now_ms - last < 1000) {
        return;
    }
    if (!s_last_warn_ms.compare_exchange_strong(last, now_ms, std::memory_order_relaxed)) {
        return;
    }
    fprintf(stderr,
        "[storageLLM WARNING] device_budget_bytes exhausted backend=%d raw=%llu reserved=%llu pending=%llu kv=%llu common=%llu allocated=%llu free=%llu configured=%llu\n",
        e ? (int)e->backend_caps.backend : -1,
        (unsigned long long)raw_budget,
        (unsigned long long)reserved_total,
        (unsigned long long)pending,
        (unsigned long long)kv_reserved,
        (unsigned long long)common_reserved,
        (unsigned long long)(e ? e->device_allocated_bytes.load(std::memory_order_relaxed) : 0),
        (unsigned long long)(e ? e->device_free_bytes.load(std::memory_order_relaxed) : 0),
        (unsigned long long)(e ? e->config.vram_budget_bytes : 0));
}

static uint64_t device_budget_bytes(const moe_pc_engine* e) {
    if (!e) return 0;

    // BUGFIX: Subtract pending_vram_reservation from total budget to prevent over-allocation
    // In-flight tensor copies consume budget even before device_free_bytes is updated.
    const uint64_t pending = e->pending_vram_reservation.load(std::memory_order_relaxed);
    // BUGFIX: VRAM 예산에서 KV + common 예약분을 먼저 빼야 함 (device_try_reserve_pending_vram 대신 여기서 일괄 적용)
    const uint64_t kv_reserved = e->vram_kv_budget_bytes.load(std::memory_order_relaxed);
    const uint64_t common_reserved = e->common_vram_reserved_bytes.load(std::memory_order_relaxed);
    uint64_t reserved_total = pending;
    if (reserved_total > UINT64_MAX - kv_reserved) {
        reserved_total = UINT64_MAX;
    } else {
        reserved_total += kv_reserved;
    }
    if (reserved_total > UINT64_MAX - common_reserved) {
        reserved_total = UINT64_MAX;
    } else {
        reserved_total += common_reserved;
    }

    if (e->config.vram_budget_bytes) {
        if (e->config.vram_budget_bytes <= reserved_total) {
            warn_device_budget_exhausted(e, e->config.vram_budget_bytes, reserved_total, pending, kv_reserved, common_reserved);
            return 0;
        }
        return e->config.vram_budget_bytes - reserved_total;
    }
    if (e->backend_caps.supports_unified_memory &&
        e->backend_caps.backend == moe_BACKEND_METAL) {
        // BUGFIX 124: Metal unified memory - do NOT subtract ram_used_bytes here.
        // RAM budget is set to 0 in device_allocator_init, so all tracking is in VRAM budget.
        const uint64_t sys = e->backend_caps.system_memory_bytes > 0
            ? e->backend_caps.system_memory_bytes : 16ull * 1024 * 1024 * 1024;
        // BUGFIX 199: Use configurable percentage (0=auto from device_allocator_init)
        // BUGFIX 232: Ensure pct is never 0 to prevent zero budget
        // BUGFIX 274: Simplify zero budget check - pct validation already ensures non-zero
        uint64_t pct = e->io_config.metal_memory_budget_pct;
        if (pct == 0) {
            pct = 75ull;
        }
        if (pct < 50 || pct > 90) {
            if (pct != 0) {
                fprintf(stderr,
                    "[storageLLM WARNING] metal_memory_budget_pct=%llu out of safe range [50,90], clamping\n",
                    (unsigned long long)pct);
            }
            pct = 75ull;
        }
        const uint64_t raw_budget = (sys * pct) / 100ull;
        if (raw_budget <= reserved_total) {
            warn_device_budget_exhausted(e, raw_budget, reserved_total, pending, kv_reserved, common_reserved);
            return 0;
        }
        return raw_budget - reserved_total;
    }
    const uint64_t allocated = e->device_allocated_bytes.load(std::memory_order_relaxed);
    const uint64_t free_b = e->device_free_bytes.load(std::memory_order_relaxed);
    if (allocated || free_b) {
        // BUGFIX 122: With CAS loop in device_try_reserve_pending_vram, underflow is eliminated.
        // Keep clamp for safety, but it should never trigger now.
        int64_t s_free = (int64_t)free_b;
        if (s_free < 0) s_free = 0;
        const uint64_t raw_budget = allocated > UINT64_MAX - (uint64_t)s_free ? UINT64_MAX : allocated + (uint64_t)s_free;
        if (raw_budget <= reserved_total) {
            warn_device_budget_exhausted(e, raw_budget, reserved_total, pending, kv_reserved, common_reserved);
            return 0;
        }
        return raw_budget - reserved_total;
    }
    const uint64_t raw_budget = e->backend_caps.device_memory_bytes ? e->backend_caps.device_memory_bytes : e->backend_caps.recommended_vram_cache_bytes;
    if (raw_budget <= reserved_total) {
        warn_device_budget_exhausted(e, raw_budget, reserved_total, pending, kv_reserved, common_reserved);
        return 0;
    }
    return raw_budget - reserved_total;
}

static uint64_t device_memory_headroom_pct(uint8_t configured) {
    if (configured == 0 || configured > 90) {
        return 10ull;
    }
    return configured;
}

// BUGFIX 122: Issue 27 - pending VRAM reservation TOCTOU race ★★★
// Problem: Two threads can both pass check at step 2, then both fetch_add at step 3.
// If both exceed budget at step 4, both fetch_sub, causing transient underflow.
// device_budget_bytes() clamps underflow to 0, but this is a workaround, not a fix.
// Solution: Replace check-then-add with CAS loop for atomic reservation.
// Trade-off: CAS contention possible, but VRAM allocation is infrequent (expert triplet granularity).
// Max 8 GPU workers, so contention is low. Eliminates underflow bug permanently.

// BUGFIX 문제6: Removed deprecated reservation functions ★★
// Problem: device_try_reserve_pending_vram() and device_release_pending_vram_reservation()
// marked [[deprecated]] but still linked. They share pending_vram_reservation atomic with
// prefetch_admission.cpp.inc CAS loop, risking double-counting if accidentally called.
// Solution: Completely removed both functions. Only call site (device_tensor_copy_alloc.cpp.inc:91)
// was already commented out. prefetch_admission.cpp.inc CAS loop is the sole reservation mechanism.

static void device_refresh_mem_info(moe_pc_engine* e, int retry_path) {
    if (!e) return;
    const auto now = std::chrono::steady_clock::now();
    int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    int64_t last = e->last_mem_query_ms.load(std::memory_order_relaxed);
    const bool external_budget_backend =
        e->backend_caps.backend == moe_BACKEND_VULKAN ||
        e->backend_caps.backend == moe_BACKEND_LEVEL_ZERO ||
        e->backend_caps.backend == moe_BACKEND_SYCL;
    const int64_t throttle_ms = retry_path || external_budget_backend ? 1000 : 5000;
    if (now_ms - last < throttle_ms) {
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
        // BUGFIX 472: memoryHeapCount 범위 체크 (최대 16개)
        uint32_t heap_count = props.memoryProperties.memoryHeapCount;
        if (heap_count > 16) heap_count = 16;
        for (uint32_t i = 0; i < heap_count; ++i) {
            if (props.memoryProperties.memoryHeaps[i].flags & 1) {
                calc_total += props.memoryProperties.memoryHeaps[i].size;
                if (budget.heapBudget[i] > 0) {
                    uint64_t avail = (budget.heapBudget[i] > budget.heapUsage[i]) ? (budget.heapBudget[i] - budget.heapUsage[i]) : 0;
                    const uint64_t keep_free_pct = 100ull - device_memory_headroom_pct(e->io_config.vulkan_memory_headroom_pct);
                    calc_free += (avail * keep_free_pct) / 100ull;
                } else {
                    const uint64_t keep_free_pct = 100ull - device_memory_headroom_pct(e->io_config.vulkan_memory_headroom_pct);
                    calc_free += (props.memoryProperties.memoryHeaps[i].size * keep_free_pct) / 100ull;
                }
            }
        }
        if (calc_total > 0) {
            free_b = calc_free;
            total_b = calc_total;
            mem_queried = 1;
        }
    } else if (e &&
               (e->backend_caps.backend == moe_BACKEND_LEVEL_ZERO ||
                e->backend_caps.backend == moe_BACKEND_SYCL) &&
               e->ze.zeDeviceGetMemoryProperties && e->ze.device) {
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
        mem_props.stype = 0x7; // ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES
        mem_props.pNext = &ext_props;
        uint32_t count = 1;
        if (e->ze.zeDeviceGetMemoryProperties(e->ze.device, &count, &mem_props) == 0) {
            total_b = mem_props.totalSize;
            if (ext_props.allocatedSize > 0) {
                uint64_t avail = (mem_props.totalSize > ext_props.allocatedSize) ? (mem_props.totalSize - ext_props.allocatedSize) : mem_props.totalSize;
                const uint64_t keep_free_pct = 100ull - device_memory_headroom_pct(e->io_config.level_zero_memory_headroom_pct);
                free_b = (avail * keep_free_pct) / 100ull;
            } else {
                const uint64_t keep_free_pct = 100ull - device_memory_headroom_pct(e->io_config.level_zero_memory_headroom_pct);
                free_b = (mem_props.totalSize * keep_free_pct) / 100ull;
            }
            mem_queried = 1;
        }
    }
    if (mem_queried) {
        // BUGFIX 239: Subtract pending_vram_reservation from driver-reported free memory ★★★
        // Problem: Driver reports memory that is "not yet allocated" as free, but our
        // pending_vram_reservation already committed this memory to in-flight prefetch tasks.
        // Without this correction, device_refresh_mem_info treats pending reservations as
        // "accounting drift" and adds them back to device_free_bytes every 1 second.
        // This causes VRAM budget to "magically" grow beyond physical limits.
        // Solution: Subtract pending_vram_reservation from driver-reported free_b before
        // calculating delta. This ensures pending reservations are not double-counted.
        const uint64_t pending = e->pending_vram_reservation.load(std::memory_order_relaxed);
        if (free_b > pending) {
            free_b -= pending;
        } else {
            free_b = 0;
        }

        // Delta sync for device_free_bytes to recover from accounting drift
        const uint64_t tracked_free = e->device_free_bytes.load(std::memory_order_relaxed);
        const int64_t delta = (int64_t)free_b - (int64_t)tracked_free;
        const uint64_t abs_delta = delta < 0 ? (uint64_t)(-delta) : (uint64_t)delta;
        const uint64_t sync_threshold = std::max<uint64_t>(
            64ull * 1024ull * 1024ull,
            ((uint64_t)total_b) / 100ull);
        if (!retry_path && abs_delta < sync_threshold) {
            e->device_total_bytes.store((uint64_t)total_b);
            return;
        }
        if (delta > 0) {
            e->device_free_bytes.fetch_add((uint64_t)delta, std::memory_order_relaxed);
        } else if (delta < 0) {
            const uint64_t neg_delta = (uint64_t)(-delta);
            const uint64_t old = e->device_free_bytes.fetch_sub(neg_delta, std::memory_order_relaxed);
            if (old < neg_delta) {
                e->device_free_bytes.fetch_add(neg_delta - old, std::memory_order_relaxed);
            }
        }
        if (total_b > 0) {
            e->backend_caps.device_memory_bytes = (uint64_t)total_b;
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
    // BUGFIX 521: Check ptrs.size() upper bound to prevent excessive iteration ★
    // Problem: ptrs can contain arbitrarily many entries if eviction is aggressive
    // Loop can run billions of iterations
    // Solution: Add reasonable upper bound check (e.g., 100000)
    if (ptrs.size() > 100000) {
        fprintf(stderr, "[storageLLM ERROR] device_free_deferred_ptrs: excessive ptr count %zu\n",
            ptrs.size());
        ptrs.clear();
        return;
    }
    device_set_current(e);
    for (auto& pair : ptrs) {
        // BUGFIX 592: Check pair.second (bytes) is reasonable ★
        // Problem: pair.second can be corrupted (very large value)
        // Solution: Validate bytes before processing
        if (pair.second > 100ull * 1024 * 1024 * 1024) { // 100GB sanity check
            fprintf(stderr, "[storageLLM ERROR] device_free_deferred_ptrs: suspicious bytes %llu\n",
                (unsigned long long)pair.second);
            continue;
        }
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
    // BUGFIX 620: Check device_allocated_bytes > 0 before fetch_sub to prevent underflow ★
    // Problem: If device_allocated_bytes is 0, fetch_sub causes underflow
    // Solution: Only decrement if > 0
    uint64_t current_allocated = e->device_allocated_bytes.load(std::memory_order_acquire);
    while (current_allocated > 0 && current_allocated >= a.bytes) {
        if (e->device_allocated_bytes.compare_exchange_weak(
                current_allocated, current_allocated - a.bytes,
                std::memory_order_release,
                std::memory_order_acquire)) {
            break;
        }
    }
    // BUGFIX 620: Check device_allocation_count > 0 before fetch_sub to prevent underflow ★
    uint64_t current_count = e->device_allocation_count.load(std::memory_order_acquire);
    while (current_count > 0) {
        if (e->device_allocation_count.compare_exchange_weak(
                current_count, current_count - 1,
                std::memory_order_release,
                std::memory_order_acquire)) {
            break;
        }
    }
    if (auto* c = device_pool_counter(e, a.pool)) {
        uint64_t pool_current = c->load(std::memory_order_acquire);
        while (pool_current > 0 && pool_current >= a.bytes) {
            if (c->compare_exchange_weak(
                    pool_current, pool_current - a.bytes,
                    std::memory_order_release,
                    std::memory_order_acquire)) {
                break;
            }
        }
    }
    if (a.pool == moe_DEVICE_POOL_ACTIVATION) {
        e->activation_keys.erase(it->first);
        // BUGFIX 621: Check device_activation_count > 0 before fetch_sub to prevent underflow ★
        uint64_t current_activation = e->device_activation_count.load(std::memory_order_acquire);
        while (current_activation > 0) {
            if (e->device_activation_count.compare_exchange_weak(
                    current_activation, current_activation - 1,
                    std::memory_order_release,
                    std::memory_order_acquire)) {
                break;
            }
        }
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



