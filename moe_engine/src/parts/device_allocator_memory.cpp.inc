// Device memory management and accounting

static uint64_t device_budget_bytes(const moe_pc_engine* e);

// BUGFIX 76: Issue 11 - Document KV cache vs VRAM budget separation ??筌뤾쑨猿?
// Clarification: There are two distinct KV cache concepts:
// 1. KV cache pool (CPU RAM, moe_kv_cache): tracked by tier_used(moe_TIER_RAM)
// 2. GPU-side KV cache VRAM allocation: tracked by device_kv_cache_bytes
// - device_budget_bytes() DOES subtract device_kv_cache_bytes from VRAM budget.
// - GPU-side KV growth DOES reduce expert cache space dynamically.

// Helper to encapsulate VRAM budget accounting.
// Do not modify engine->device_free_bytes directly.
static void device_atomic_sub_saturating(std::atomic<uint64_t>& counter, uint64_t bytes) {
    if (bytes == 0) {
        return;
    }
    uint64_t cur = counter.load(std::memory_order_acquire);
    while (cur > 0) {
        const uint64_t next = cur > bytes ? cur - bytes : 0;
        if (counter.compare_exchange_weak(
                cur,
                next,
                std::memory_order_release,
                std::memory_order_acquire)) {
            return;
        }
    }
}

static void device_cufile_unregister_buffer(moe_pc_engine* e, void* ptr) {
    if (!e || !ptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(e->cufile_cache_mutex);
    const uint64_t key = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr));
    auto it = e->cufile_buffer_registration_cache.find(key);
    if (it == e->cufile_buffer_registration_cache.end()) {
        return;
    }
    if (e->cuda.cuFileBufDeregister) {
        e->cuda.cuFileBufDeregister(ptr);
    }
    e->cufile_buffer_registration_cache.erase(it);
}

static void device_cufile_cleanup_caches(moe_pc_engine* e) {
    if (!e) {
        return;
    }
    std::lock_guard<std::mutex> lock(e->cufile_cache_mutex);
    if (e->cuda.cuFileBufDeregister) {
        for (const auto& item : e->cufile_buffer_registration_cache) {
            void* ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(item.first));
            e->cuda.cuFileBufDeregister(ptr);
        }
    }
    e->cufile_buffer_registration_cache.clear();
    if (e->cuda.cuFileHandleDeregister) {
        for (const auto& item : e->cufile_file_handle_cache) {
            if (item.second) {
                e->cuda.cuFileHandleDeregister(item.second);
            }
        }
    }
    e->cufile_file_handle_cache.clear();
}

static std::atomic<uint64_t>* device_vram_resident_counter_for_pool(moe_pc_engine* e, uint8_t pool) {
    if (!e) {
        return nullptr;
    }
    if (pool == moe_DEVICE_POOL_EXPERT) {
        return &e->vram_resident_bytes;
    }
    if (pool == moe_DEVICE_POOL_FIXED ||
        pool == moe_DEVICE_POOL_ACTIVATION ||
        pool == moe_DEVICE_POOL_STREAM) {
        return &e->vram_common_bytes;
    }
    if (pool == moe_DEVICE_POOL_KV_CACHE) {
        return &e->vram_kv_bytes;
    }
    return nullptr;
}

static void device_vram_add_resident_bytes(moe_pc_engine* e, uint8_t pool, uint64_t bytes) {
    if (auto* counter = device_vram_resident_counter_for_pool(e, pool)) {
        counter->fetch_add(bytes, std::memory_order_release);
    }
}

static void device_vram_sub_resident_bytes(moe_pc_engine* e, uint8_t pool, uint64_t bytes) {
    if (auto* counter = device_vram_resident_counter_for_pool(e, pool)) {
        device_atomic_sub_saturating(*counter, bytes);
    }
}

static void device_vram_mark_allocation_resident(moe_pc_engine* e, uint64_t key) {
    if (!e || key == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(e->device_mutex);
    auto it = e->device_allocations.find(key);
    if (it != e->device_allocations.end() && !it->second.resident_accounted) {
        it->second.resident_accounted = true;
        device_vram_add_resident_bytes(e, it->second.pool, it->second.bytes);
    }
}

static void device_vram_release_reservation_token(moe_pc_engine* e, const vram_reservation_token& token) {
    if (!e || !token.valid || token.bytes == 0) {
        return;
    }
    device_atomic_sub_saturating(e->vram_inflight_bytes, token.bytes);
}

static void device_vram_complete_reservation_token(
    moe_pc_engine* e,
    const vram_reservation_token& token,
    uint64_t key
) {
    if (!e || !token.valid || token.bytes == 0) {
        return;
    }
    device_vram_release_reservation_token(e, token);
    device_vram_mark_allocation_resident(e, key);
}

static int device_vram_try_reserve_token(
    moe_pc_engine* e,
    uint64_t bytes,
    uint8_t pool,
    vram_reservation_token* out_token
) {
    if (out_token) {
        *out_token = vram_reservation_token{};
    }
    if (!e || bytes == 0) {
        return 0;
    }
    uint64_t cur = e->vram_inflight_bytes.load(std::memory_order_relaxed);
    uint32_t spin = 0;
    for (;;) {
        const uint64_t budget = device_budget_bytes(e);
        if (budget == 0 || bytes > budget) {
            return 0;
        }
        if (cur > UINT64_MAX - bytes) {
            return 0;
        }
        if (e->vram_inflight_bytes.compare_exchange_weak(
                cur,
                cur + bytes,
                std::memory_order_acq_rel,
                std::memory_order_acquire)) {
            if (out_token) {
                out_token->bytes = bytes;
                out_token->pool = pool;
                out_token->valid = true;
            }
            return 1;
        }
        if (++spin > 8) {
            std::this_thread::yield();
            spin = 0;
        }
    }
}

static void device_account_bytes_freed(moe_pc_engine* e, uint64_t bytes) {
    if (!e) return;
    if (e->backend_caps.backend == moe_BACKEND_METAL) return; // budget computed dynamically
    e->device_free_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

// BUGFIX 124: Issue 29 - Metal unified memory budget double-counted ??筌뤾쑨猿?
// Problem: Metal VRAM budget subtracts ram_used_bytes, but RAM budget also tracks ram_used_bytes.
// Same 4GB data consumes 4GB from VRAM budget + 4GB from RAM budget = 8GB total.
// Solution: Metal should use unified budget only. Set RAM budget to 0 in device_allocator_init.
// All expert data tracked in "VRAM" budget (actually unified memory budget).
// Trade-off: Relies on OS unified memory pressure management instead of explicit RAM eviction.
// Metal handles this natively, so no functional loss.

// BUGFIX 199: Metal VRAM budget configurable ??// Problem: 60% hardcoded ??M3 Max 128GB wastes 25GB (60%=76.8GB vs 80%=102.4GB)
// Solution: Use io_config.metal_memory_budget_pct (0=auto, safe range 50-90)
// Trade-off: >85% risks macOS memory pressure ??swapping or app termination
static void warn_device_budget_exhausted(const moe_pc_engine* e, uint64_t raw_budget, uint64_t reserved_total, uint64_t inflight_reserved, uint64_t kv_reserved, uint64_t common_reserved) {
    if (!e || e->backend_caps.backend == moe_BACKEND_CPU) {
        return;
    }
    if (raw_budget == 0 && e->config.vram_budget_bytes == 0 &&
        e->device_allocated_bytes.load(std::memory_order_relaxed) == 0 &&
        e->device_free_bytes.load(std::memory_order_relaxed) == 0) {
        return;
    }
    // Use per-engine timer state so one engine cannot suppress another engine's warnings.
    // One engine's warning suppresses other engine's warnings in multi-engine process
    // Solution: Use e->last_budget_warn_ms instead of static variable
    const auto now = std::chrono::steady_clock::now();
    const int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    auto* mutable_e = const_cast<moe_pc_engine*>(e);
    int64_t last = mutable_e->last_budget_warn_ms.load(std::memory_order_relaxed);
    if (now_ms - last < 1000) {
        return;
    }
    if (!mutable_e->last_budget_warn_ms.compare_exchange_strong(last, now_ms, std::memory_order_relaxed)) {
        return;
    }
    fprintf(stderr,
        "[storageLLM WARNING] device_budget_bytes exhausted backend=%d raw=%llu reserved=%llu pending=%llu kv=%llu common=%llu allocated=%llu free=%llu configured=%llu\n",
        e ? (int)e->backend_caps.backend : -1,
        (unsigned long long)raw_budget,
        (unsigned long long)reserved_total,
        (unsigned long long)inflight_reserved,
        (unsigned long long)kv_reserved,
        (unsigned long long)common_reserved,
        (unsigned long long)(e ? e->device_allocated_bytes.load(std::memory_order_relaxed) : 0),
        (unsigned long long)(e ? e->device_free_bytes.load(std::memory_order_relaxed) : 0),
        (unsigned long long)(e ? e->config.vram_budget_bytes : 0));
}

static uint64_t device_budget_bytes(const moe_pc_engine* e) {
    if (!e) return 0;
    auto add_sat = [](uint64_t acc, uint64_t value) -> uint64_t {
        return value > UINT64_MAX - acc ? UINT64_MAX : acc + value;
    };
    uint64_t raw_budget = 0;
    if (e->backend_caps.backend == moe_BACKEND_METAL &&
        e->backend_caps.supports_unified_memory) {
        const uint64_t sys = e->backend_caps.system_memory_bytes > 0
            ? e->backend_caps.system_memory_bytes : 16ull * 1024ull * 1024ull * 1024ull;
        uint64_t pct = e->io_config.metal_memory_budget_pct;
        if (pct == 0 || pct < 50 || pct > 90) {
            pct = 75ull;
        }
        raw_budget = (sys * pct) / 100ull;
    } else if (e->config.vram_budget_bytes) {
        raw_budget = e->config.vram_budget_bytes;
    } else {
        const uint64_t allocated = e->device_allocated_bytes.load(std::memory_order_relaxed);
        const uint64_t free_b = e->device_free_bytes.load(std::memory_order_relaxed);
        raw_budget = (allocated || free_b)
            ? (allocated > UINT64_MAX - free_b ? UINT64_MAX : allocated + free_b)
            : (e->backend_caps.device_memory_bytes
                ? e->backend_caps.device_memory_bytes
                : e->backend_caps.recommended_vram_cache_bytes);
    }
    if (raw_budget == 0) {
        return 0;
    }
    const uint64_t resident = e->vram_resident_bytes.load(std::memory_order_relaxed);
    const uint64_t inflight = e->vram_inflight_bytes.load(std::memory_order_relaxed);
    const uint64_t kv_actual = e->kv_actually_in_vram.load(std::memory_order_acquire)
        ? e->device_kv_cache_bytes.load(std::memory_order_relaxed)
        : 0;
    const uint64_t kv_reserved = std::max(
        e->vram_kv_bytes.load(std::memory_order_relaxed),
        kv_actual);
    uint64_t common_reserved = e->vram_common_bytes.load(std::memory_order_relaxed);
    if (!e->common_vram_upload_complete.load(std::memory_order_acquire)) {
        common_reserved = std::max(
            common_reserved,
            e->common_vram_reserved_bytes.load(std::memory_order_relaxed));
    }
    uint64_t safety_margin = raw_budget >= 512ull * 1024ull * 1024ull
        ? 64ull * 1024ull * 1024ull
        : raw_budget / 16ull;
    if (safety_margin > 256ull * 1024ull * 1024ull) {
        safety_margin = 256ull * 1024ull * 1024ull;
    }
    uint64_t reserved_total = 0;
    reserved_total = add_sat(reserved_total, resident);
    reserved_total = add_sat(reserved_total, inflight);
    reserved_total = add_sat(reserved_total, kv_reserved);
    reserved_total = add_sat(reserved_total, common_reserved);
    reserved_total = add_sat(reserved_total, safety_margin);
    if (raw_budget <= reserved_total) {
        warn_device_budget_exhausted(e, raw_budget, reserved_total, inflight, kv_reserved, common_reserved);
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

static void device_refresh_mem_info(moe_pc_engine* e, int retry_path) {
    if (!e) return;
    const auto now = std::chrono::steady_clock::now();
    int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    int64_t last = e->last_mem_query_ms.load(std::memory_order_relaxed);
    const bool external_budget_backend =
        e->backend_caps.backend == moe_BACKEND_VULKAN ||
        e->backend_caps.backend == moe_BACKEND_LEVEL_ZERO ||
        e->backend_caps.backend == moe_BACKEND_SYCL;
    // BUGFIX 891: Unify throttle to 1s for all paths ??筌뤾쑨猿??CRITICAL
    // Problem: 5s throttle for non-external backends causes stale VRAM estimates.
    // With top-k=8 experts loading/unloading per token, 5s gap allows hundreds of MB drift.
    // Solution: Use 1s throttle for all paths. CAS guard prevents concurrent driver calls.
    // Impact: More accurate VRAM tracking, minimal overhead (~1 driver call/sec max).
    const int64_t throttle_ms = 1000;
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
        // BUGFIX 472: memoryHeapCount ?類????癲ル슪???띿물?(癲ル슔?됭짆? 16??
        uint32_t heap_count = props.memoryProperties.memoryHeapCount;
        if (heap_count > 16) heap_count = 16;
        for (uint32_t i = 0; i < heap_count; ++i) {
            if (props.memoryProperties.memoryHeaps[i].flags & 1) {
                calc_total += props.memoryProperties.memoryHeaps[i].size;
                if (budget.heapBudget[i] > 0) {
                    uint64_t avail = (budget.heapBudget[i] > budget.heapUsage[i]) ? (budget.heapBudget[i] - budget.heapUsage[i]) : 0;
                    // BUGFIX 895: Remove headroom application here ??筌뤾쑨猿??CRITICAL
                    // Problem: Headroom applied here, then again in prefetch_admission.cpp.inc
                    // Double headroom: (avail * 90%) * 90% = avail * 81% instead of 90%
                    // Solution: Use raw avail, let prefetch_admission apply single headroom
                    // Impact: Correct VRAM budget, prevents over-conservative allocation
                    calc_free += avail;
                } else {
                    // Fallback when budget not available: use heap size directly
                    calc_free += props.memoryProperties.memoryHeaps[i].size;
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
            // BUGFIX 926: Remove headroom application here ??筌뤾쑨猿??CRITICAL
            // Problem: Headroom applied here, then again in prefetch_admission.cpp.inc
            // Double headroom: (avail * 80%) * 80% = avail * 64% instead of 80%
            // Solution: Use raw avail, let prefetch_admission apply single headroom
            // Impact: Correct VRAM budget for Intel GPUs, prevents over-conservative allocation
            if (ext_props.allocatedSize > 0) {
                uint64_t avail = (mem_props.totalSize > ext_props.allocatedSize)
                    ? (mem_props.totalSize - ext_props.allocatedSize)
                    : mem_props.totalSize;
                free_b = avail;
            } else {
                free_b = mem_props.totalSize;
            }
            mem_queried = 1;
        }
    }
    if (mem_queried) {
        // BUGFIX 239: Subtract vram_inflight_bytes from driver-reported free memory ??筌뤾쑨猿??        // Problem: Driver reports memory that is "not yet allocated" as free, but our
        // vram_inflight_bytes already committed this memory to in-flight prefetch tasks.
        // Without this correction, device_refresh_mem_info treats pending reservations as
        // "accounting drift" and adds them back to device_free_bytes every 1 second.
        // This causes VRAM budget to "magically" grow beyond physical limits.
        // calculating delta. This ensures pending reservations are not double-counted.
        const uint64_t pending = e->vram_inflight_bytes.load(std::memory_order_relaxed);
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

    // BUGFIX 998: Metal H2D bandwidth EMA update ??筌뤾쑨猿?IMPORTANT
    // Problem: Metal H2D bandwidth is measured once at init. If OS background
    //   pressure is high at init, a low value is locked in permanently.
    //   juju_runtime_strategy overlap budget is based on this value ??underestimate.
    // Solution: On each 1s refresh cycle, re-measure via metal_measure_h2d_bandwidth
    //   and apply EMA (alpha=0.2) to smooth transient noise.
    // Guard: Only runs if enable_bw_benchmark AND backend is Metal AND a non-zero
    //   initial measurement exists (skip if init benchmark was disabled).
    // BUGFIX NEW-25: Skip Metal H2D bandwidth measurement during active inference ??筌뤾쑨猿??    // Problem: metal_measure_h2d_bandwidth() submits a blit command to the Metal command
    // queue. During inference, forward pass commands occupy the queue ??measurement blit
    // waits behind them ??measured latency includes inference delay ??EMA corrupted.
    // Solution: Check total_queue_depth > 0 as proxy for active inference. If tokens are
    // queued, skip this cycle. EMA retains last valid value (juju_runtime_strategy is safe).
    // Impact: Clean bandwidth samples, accurate overlap budget, no measurement stalls.
    if (e->backend_caps.backend == moe_BACKEND_METAL &&
        e->io_config.enable_bw_benchmark &&
        e->backend_caps.measured_h2d_bw_bytes_per_sec > 0 &&
        e->total_queue_depth.load(std::memory_order_relaxed) == 0) {
        const uint64_t new_sample = metal_measure_h2d_bandwidth(metal_get_default_device());
        if (new_sample > 0) {
            const uint64_t old_bw = e->backend_caps.measured_h2d_bw_bytes_per_sec;
            // EMA alpha=0.2: new_ema = old * 0.8 + sample * 0.2
            const uint64_t ema_bw = (old_bw * 4ull + new_sample) / 5ull;
            e->backend_caps.measured_h2d_bw_bytes_per_sec = ema_bw;
        }
    }
}

// BUGFIX 54: Issue 29 - Replace CAS spin with fetch_sub ???ル∥???
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
    // Bound ptrs.size() to prevent excessive iteration if eviction is aggressive.
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
                // Check pair.second; corrupted byte counts can be extremely large.
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
                // BUGFIX 928: Check hipFree return value ??筌뤾쑨猿?IMPORTANT
                // Problem: hipFree can fail if context lost or pointer already freed
                // ??Memory leak tracked in device_allocated_bytes but not actually freed
                // Solution: Log error on failure (best-effort, continue loop)
                // Impact: Enables debugging of VRAM accounting mismatches
                const int hip_err = e->hip.hipFree(pair.first);
                if (hip_err != 0) {
                    fprintf(stderr, "[storageLLM ERROR] hipFree failed: ptr=%p bytes=%llu err=%d\n",
                        (void*)pair.first, (unsigned long long)pair.second, hip_err);
                }
            } else if (e->backend_caps.backend == moe_BACKEND_CUDA && e->cuda.cuMemFree) {
                // BUGFIX 928: Check cuMemFree return value ??筌뤾쑨猿?IMPORTANT
                // Problem: cuMemFree can fail if context lost or pointer already freed
                // ??Memory leak tracked in device_allocated_bytes but not actually freed
                // Solution: Log error on failure (best-effort, continue loop)
                // Impact: Enables debugging of VRAM accounting mismatches
                const int cuda_err = e->cuda.cuMemFree(pair.first);
                if (cuda_err != 0) {
                    fprintf(stderr, "[storageLLM ERROR] cuMemFree failed: ptr=%p bytes=%llu err=%d\n",
                        (void*)pair.first, (unsigned long long)pair.second, cuda_err);
                }
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
        device_cufile_unregister_buffer(e, reinterpret_cast<void*>(static_cast<uintptr_t>(a.ptr)));
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
    if (a.resident_accounted) {
        device_vram_sub_resident_bytes(e, a.pool, a.bytes);
    }
    // BUGFIX 620: Check device_allocated_bytes > 0 before fetch_sub to prevent underflow ??    // Problem: If device_allocated_bytes is 0, fetch_sub causes underflow
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
