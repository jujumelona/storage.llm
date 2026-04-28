static void device_free_expert_locked(
    glm5_pc_engine* e,
    int layer,
    int expert,
    std::vector<CUdeviceptr>* deferred_free_ptrs = nullptr
);

static std::atomic<uint64_t>* device_pool_counter(glm5_pc_engine* e, uint8_t pool) {
    if (!e) return nullptr;
    if (pool == GLM5_DEVICE_POOL_FIXED) return &e->device_fixed_bytes;
    if (pool == GLM5_DEVICE_POOL_ACTIVATION) return &e->device_activation_bytes;
    if (pool == GLM5_DEVICE_POOL_EXPERT) return &e->device_expert_bytes;
    return &e->device_stream_bytes;
}
static int device_allocator_init(glm5_pc_engine* e) {
    if (!e || e->device_allocator_ready.load() != 0) return e && e->device_allocator_ready.load() > 0;
    if (!e->config.prefer_gpu || e->backend_caps.backend != GLM5_BACKEND_CUDA) return 0;
    std::lock_guard<std::mutex> lock(e->device_mutex);
    if (e->device_allocator_ready.load() != 0) return e->device_allocator_ready.load() > 0;
    if (!cuda_driver_load(&e->cuda) || e->cuda.cuInit(0) != 0 ||
        e->cuda.cuDeviceGet(&e->cuda.device, (int)e->backend_caps.device_id) != 0 ||
        e->cuda.cuDevicePrimaryCtxRetain(&e->cuda.context, e->cuda.device) != 0 ||
        e->cuda.cuCtxSetCurrent(e->cuda.context) != 0) {
        glm5_close_shared_library(e->cuda.library);
        e->cuda = glm5_cuda_driver_api{};
        e->device_allocator_ready.store(-1);
        return 0;
    }
    size_t free_b = 0, total_b = 0;
    if (e->cuda.cuMemGetInfo(&free_b, &total_b) == 0) {
        e->device_free_bytes.store((uint64_t)free_b);
        e->device_total_bytes.store((uint64_t)total_b);
        if (e->backend_caps.device_memory_bytes == 0) e->backend_caps.device_memory_bytes = (uint64_t)total_b;
    }
    e->device_allocator_ready.store(1);
    return 1;
}
static uint64_t device_budget_bytes(const glm5_pc_engine* e) {
    if (!e) return 0;
    if (e->config.vram_budget_bytes) return e->config.vram_budget_bytes;
    return e->backend_caps.device_memory_bytes ? e->backend_caps.device_memory_bytes : e->backend_caps.recommended_vram_cache_bytes;
}
static int device_set_current(glm5_pc_engine* e) {
    static thread_local CUcontext tls_ctx = nullptr;
    if (!e || !e->cuda.cuCtxSetCurrent) return 0;
    if (tls_ctx == e->cuda.context) return 1;
    if (e->cuda.cuCtxSetCurrent(e->cuda.context) != 0) return 0;
    tls_ctx = e->cuda.context;
    return 1;
}
static void device_refresh_mem_info_locked(glm5_pc_engine* e) {
    if (!e) return;
    // Fix 41: Per-engine throttle instead of global static (multi-engine safe)
    const auto now = std::chrono::steady_clock::now();
    int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    int64_t last = e->last_mem_query_ms.load(std::memory_order_relaxed);
    if (now_ms - last < 200) {
        return;
    }
    if (!e->last_mem_query_ms.compare_exchange_strong(last, now_ms)) {
        return;
    }
    size_t free_b = 0, total_b = 0;
    if (e && e->cuda.cuMemGetInfo && device_set_current(e) && e->cuda.cuMemGetInfo(&free_b, &total_b) == 0) {
        e->device_free_bytes.store((uint64_t)free_b);
        e->device_total_bytes.store((uint64_t)total_b);
    }
}
static void device_free_deferred_ptrs(glm5_pc_engine* e, std::vector<CUdeviceptr>& ptrs) {
    if (!e || ptrs.empty() || !e->cuda.cuMemFree) {
        ptrs.clear();
        return;
    }
    device_set_current(e);
    for (CUdeviceptr ptr : ptrs) {
        if (ptr) {
            e->cuda.cuMemFree(ptr);
        }
    }
    ptrs.clear();
}

static void device_release_allocation_locked(
    glm5_pc_engine* e,
    std::unordered_map<uint64_t, glm5_device_allocation>::iterator it,
    std::vector<CUdeviceptr>* deferred_free_ptrs = nullptr
) {
    if (!e || it == e->device_allocations.end()) return;
    const glm5_device_allocation a = it->second;
    if (a.ptr) {
        if (deferred_free_ptrs) {
            deferred_free_ptrs->push_back(a.ptr);
        } else {
            device_set_current(e);
            if (e->cuda.cuMemFree) e->cuda.cuMemFree(a.ptr);
        }
    }
    e->device_allocated_bytes.fetch_sub(a.bytes);
    e->device_allocation_count.fetch_sub(1);
    if (auto* c = device_pool_counter(e, a.pool)) c->fetch_sub(a.bytes);
    if (a.pool == GLM5_DEVICE_POOL_ACTIVATION) {
        e->activation_keys.erase(it->first);
    }
    e->device_allocations.erase(it);
    // Fix 50: Removed device_refresh_mem_info_locked from under device_mutex
    // cuMemGetInfo is a slow driver call that blocks all alloc/free while held.
    // Callers should invoke device_refresh_mem_info_locked() after releasing mutex.
}
static void device_free_locked(
    glm5_pc_engine* e,
    uint64_t key,
    std::vector<CUdeviceptr>* deferred_free_ptrs = nullptr
) {
    auto it = e->device_allocations.find(key);
    if (it == e->device_allocations.end()) return;
    if (it->second.inflight_copies != 0) {
        it->second.pending_free = true;
        return;
    }
    device_release_allocation_locked(e, it, deferred_free_ptrs);
}
// Fix 4: LRU eviction via evict_heaps + fallback O(N) scan
// Bug #1 (Critical): Collect victims for deferred free OUTSIDE device_mutex
static void device_make_room_locked(
    glm5_pc_engine* e, uint8_t pool, uint64_t bytes,
    std::vector<CUdeviceptr>& deferred_free_ptrs
) {
    if (!e) return;
    const uint64_t budget = device_budget_bytes(e);
    if (budget == 0) return;
    while (e->device_allocated_bytes.load() + bytes > budget) {
        if (pool != GLM5_DEVICE_POOL_ACTIVATION) {
            uint64_t act_key = 0;
            uint64_t act_min_epoch = UINT64_MAX;
            // Bug #13: Still O(N) but activation_keys is small (~80 entries for 80 layers).
            // The real fix is to avoid calling this in a tight loop — see evict heap below.
            // For safety, limit scan to first 128 entries to bound device_mutex hold time.
            int scan_count = 0;
            for (uint64_t key : e->activation_keys) {
                if (++scan_count > 128) break;
                auto alloc_it = e->device_allocations.find(key);
                if (alloc_it != e->device_allocations.end() && alloc_it->second.inflight_copies == 0) {
                    if (alloc_it->second.last_touch_epoch < act_min_epoch) {
                        act_min_epoch = alloc_it->second.last_touch_epoch;
                        act_key = key;
                    }
                }
            }
            if (act_min_epoch != UINT64_MAX) {
                // Bug #1: Defer cuMemFree — remove from map but don't call driver yet
                auto it = e->device_allocations.find(act_key);
                if (it != e->device_allocations.end()) {
                    device_release_allocation_locked(e, it, &deferred_free_ptrs);
                }
                continue;
            }
        }
        const int heap_idx = tier_heap_index(GLM5_TIER_VRAM);
        bool found_via_heap = false;
        if (heap_idx >= 0) {
            uint32_t stale_skip = 0;
            const uint32_t stale_limit = e->expert_states.size() > 0
                ? (uint32_t)std::min<size_t>(e->expert_states.size(), expert_dense_slot_width()) : 128u;
            glm5_evict_candidate victim{};
            bool victim_valid = false;
            {
                std::lock_guard<std::mutex> heap_lock(e->heap_mutex);
                while (!e->evict_heaps[heap_idx].empty() && stale_skip < stale_limit) {
                    victim = e->evict_heaps[heap_idx].top();
                    e->evict_heaps[heap_idx].pop();
                    if (victim.slot >= e->expert_states.size()) { ++stale_skip; continue; }
                    const glm5_block_state& st = e->expert_states[victim.slot];
                    if (!st.active || st.tier != GLM5_TIER_VRAM || st.bytes != victim.bytes ||
                        st.last_touch_epoch != victim.last_touch_epoch) { ++stale_skip; continue; }
                    victim_valid = true;
                    break;
                }
            }
            if (victim_valid) {
                int layer = 0;
                int expert = 0;
                expert_dense_slot_decode(victim.slot, &layer, &expert);
                {
                    std::lock_guard<std::mutex> map_lock(e->map_mutex);
                    glm5_block_state& state = e->expert_states[victim.slot];
                    if (state.active && state.tier == GLM5_TIER_VRAM) {
                        subtract_tier_usage(e, GLM5_TIER_VRAM, state.bytes);
                        e->vram_expert_count.fetch_sub(1);
                        state.tier = tier_used(e, GLM5_TIER_RAM) + state.bytes <= tier_budget(e, GLM5_TIER_RAM) ? GLM5_TIER_RAM : GLM5_TIER_DB;
                        if (state.tier == GLM5_TIER_RAM) e->ram_expert_count.fetch_add(1);
                        else if (state.tier == GLM5_TIER_DB) e->db_expert_count.fetch_add(1);
                        add_tier_usage(e, state.tier, state.bytes);
                        replace_evict_candidate(e, victim.slot, state);
                    }
                }
                device_free_expert_locked(e, layer, expert, &deferred_free_ptrs);
                found_via_heap = true;
            }
        }
        if (found_via_heap) continue;
        // Fix 67: Compact heap before O(N) fallback to reduce stale buildup
        if (heap_idx >= 0) {
            std::lock_guard<std::mutex> heap_lock(e->heap_mutex);
            compact_evict_heap_locked(e, heap_idx);
        }
        // Bug #1 Final Fix: O(N) fallback REMOVED.
        // Heap was empty or all stale after compact. Return immediately
        // instead of scanning 512 device_allocations entries under device_mutex.
        return;
    }
}

static void device_free_pool(glm5_pc_engine* e, uint8_t pool) {
    if (!e || e->device_allocator_ready.load() <= 0) return;
    std::vector<CUdeviceptr> deferred_free_ptrs;
    std::vector<uint64_t> keys;
    {
        std::lock_guard<std::mutex> lock(e->device_mutex);
        for (const auto& it : e->device_allocations) if (it.second.pool == pool) keys.push_back(it.first);
        for (uint64_t key : keys) device_free_locked(e, key, &deferred_free_ptrs);
    }
    device_free_deferred_ptrs(e, deferred_free_ptrs);
}

static void device_allocator_shutdown(glm5_pc_engine* e) {
    if (!e) return;
    std::vector<CUdeviceptr> deferred_free_ptrs;
    std::vector<uint64_t> keys;
    {
        std::lock_guard<std::mutex> lock(e->device_mutex);
        for (const auto& it : e->device_allocations) keys.push_back(it.first);
        for (uint64_t key : keys) device_free_locked(e, key, &deferred_free_ptrs);
    }
    device_free_deferred_ptrs(e, deferred_free_ptrs);
    if (e->cuda.cuDevicePrimaryCtxRelease && e->cuda.context) e->cuda.cuDevicePrimaryCtxRelease(e->cuda.device);
    glm5_close_shared_library(e->cuda.library);
    e->cuda = glm5_cuda_driver_api{};
    e->device_allocator_ready.store(0);
}
