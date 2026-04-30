// Device memory eviction and room-making logic

static void device_free_expert_locked(
    glm5_pc_engine* e,
    int layer,
    int expert,
    std::vector<std::pair<CUdeviceptr, uint64_t>>* deferred_free_ptrs = nullptr
);

static void device_push_activation_candidate_locked(glm5_pc_engine* e, uint64_t key, const glm5_device_allocation& a) {
    if (!e || a.pool != GLM5_DEVICE_POOL_ACTIVATION || a.pending_free || a.bytes == 0) {
        return;
    }
    glm5_device_evict_candidate candidate{};
    candidate.key = key;
    candidate.bytes = a.bytes;
    candidate.last_touch_epoch = a.last_touch_epoch;
    e->activation_evict_heap.push(candidate);
}

static int device_free_expert_if_pending_evict_locked(
    glm5_pc_engine* e,
    int layer,
    int expert,
    std::vector<std::pair<CUdeviceptr, uint64_t>>* deferred_free_ptrs
) {
    if (!e || layer < 0 || expert < 0) {
        return 0;
    }
    const size_t slot = expert_dense_slot(layer, expert);
    if (slot == tensor_slot_invalid() || slot >= e->expert_states.size()) {
        return 0;
    }
    {
        std::lock_guard<std::mutex> map_lock(e->map_mutex);
        glm5_block_state& state = e->expert_states[slot];
        if (!state.pending_evict) {
            return 0;
        }
        state.pending_evict = false;
    }
    device_free_expert_locked(e, layer, expert, deferred_free_ptrs);
    return 1;
}

static uint32_t device_collect_activation_victims_locked(
    glm5_pc_engine* e,
    uint64_t bytes,
    std::vector<uint64_t>& keys
) {
    if (!e) {
        return 0;
    }
    const uint64_t budget = device_budget_bytes(e);
    if (budget == 0) {
        return 0;
    }
    uint64_t projected_free = 0;
    uint32_t stale = 0;
    // Pipeline Fix 4: Dynamic stale limit based on heap size
    // Fixed 128 limit causes eviction failure when heap has many stale entries.
    // Allow scanning up to half the heap size to find valid victims.
    const uint32_t stale_limit = std::max(128u, (uint32_t)e->activation_evict_heap.size() / 2u);
    while (e->device_allocated_bytes.load(std::memory_order_relaxed) > projected_free &&
           e->device_allocated_bytes.load(std::memory_order_relaxed) - projected_free + bytes > budget &&
           !e->activation_evict_heap.empty()) {
        const glm5_device_evict_candidate victim = e->activation_evict_heap.top();
        e->activation_evict_heap.pop();
        auto it = e->device_allocations.find(victim.key);
        if (it == e->device_allocations.end() || it->second.pool != GLM5_DEVICE_POOL_ACTIVATION ||
            it->second.pending_free || it->second.inflight_copies != 0 ||
            it->second.bytes != victim.bytes || it->second.last_touch_epoch != victim.last_touch_epoch) {
            if (++stale >= stale_limit) {
                break;
            }
            continue;
        }
        it->second.pending_free = true;
        keys.push_back(victim.key);
        projected_free += it->second.bytes;
        stale = 0;
    }
    return static_cast<uint32_t>(keys.size());
}

static void device_make_room_locked(
    glm5_pc_engine* e, uint8_t pool, uint64_t bytes,
    std::unique_lock<std::mutex>& device_lock,
    std::vector<std::pair<CUdeviceptr, uint64_t>>& deferred_free_ptrs
) {
    if (!e) return;
    const uint64_t budget = device_budget_bytes(e);
    if (budget == 0) return;
    const uint64_t room_bytes = bytes > UINT64_MAX - bytes / 10ull ? UINT64_MAX : bytes + bytes / 10ull;
    
    // Problem 1 Fix: Track actual freed bytes to prevent loop exit without freeing memory
    // device_free_locked sets pending_free=true when inflight_copies!=0, but doesn't
    // actually reduce device_allocated_bytes. Loop must verify real progress.
    uint64_t freed_this_iteration = 0;

    // PIPELINE FIX 3: Add retry counter for inflight DMA drain ★★★
    // Old: Single 5ms wait then break → allocation fails if DMA not complete
    // New: Retry up to 4 times (20ms total) → PCIe 4.0 x16 transfers 20MB in ~0.6ms
    // Trade-off: Max 20ms delay in worst case, but prevents prefetch task drops
    static constexpr int DMA_DRAIN_MAX_RETRIES = 4;
    int dma_drain_retries = 0;

    while (e->device_allocated_bytes.load(std::memory_order_relaxed) + bytes > budget) {
        bool expected = false;
        if (!e->vram_eviction_in_progress.compare_exchange_strong(
                expected, true, std::memory_order_acq_rel, std::memory_order_acquire)) {
            // Bug 17 Fix: 5ms → 500µs to reduce hot-path latency during frequent expert swaps
            std::this_thread::yield();
            e->vram_eviction_cv.wait_for(
                device_lock,
                std::chrono::microseconds(500),
                [&]() {
                    return !e->vram_eviction_in_progress.load(std::memory_order_acquire);
                });
            continue;
        }
        struct EvictGuard {
            std::atomic<bool>* flag;
            std::condition_variable* cv;
            ~EvictGuard() {
                flag->store(false, std::memory_order_release);
                cv->notify_all();
            }
        } guard{&e->vram_eviction_in_progress, &e->vram_eviction_cv};

        const uint64_t allocated_before = e->device_allocated_bytes.load(std::memory_order_relaxed);
        
        std::vector<std::pair<int, int>> expert_victims;
        // Bug Fix: ACTIVATION pool can also evict experts when activation eviction is insufficient
        // Without this, ACTIVATION allocation fails when activation_evict_heap is empty,
        // even though expert triplets could be evicted to make room
        std::vector<uint64_t> activation_victims;
        device_collect_activation_victims_locked(e, room_bytes, activation_victims);
        for (uint64_t key : activation_victims) {
            device_free_locked(e, key, &deferred_free_ptrs);
        }
        // If activation eviction didn't free enough, evict experts
        if (e->device_allocated_bytes.load(std::memory_order_relaxed) + bytes > budget) {
            device_lock.unlock();
            demote_cold_until(e, GLM5_TIER_VRAM, room_bytes, expert_victims);
            if (expert_victims.empty() &&
                e->device_allocated_bytes.load(std::memory_order_relaxed) + bytes > budget) {
                demote_one_vram_for_device_room(e, expert_victims);
            }
            device_lock.lock();
            for (auto& victim : expert_victims) {
                device_free_expert_if_pending_evict_locked(e, victim.first, victim.second, &deferred_free_ptrs);
            }
        }
        
        const uint64_t allocated_after = e->device_allocated_bytes.load(std::memory_order_relaxed);
        freed_this_iteration = allocated_before > allocated_after ? allocated_before - allocated_after : 0;
        
        // Problem 1 Fix: Break if no actual memory was freed (all victims had inflight_copies)
        // Without this, loop spins when all eviction candidates are pending DMA completion
        if (expert_victims.empty() && activation_victims.empty()) {
            break;
        }
        if (freed_this_iteration == 0) {
            // PIPELINE FIX 3: Retry up to 4 times before giving up
            if (dma_drain_retries++ >= DMA_DRAIN_MAX_RETRIES) {
                break;  // Give up after 4 retries (20ms total)
            }
            // BUG 2 FIX: Wait briefly for inflight DMA to complete before giving up ★★
            // All victims have inflight_copies → device_allocated_bytes didn't decrease.
            // If we break immediately, VRAM budget still exceeded → allocation fails.
            // Wait 5ms for DMA completion, then re-check budget condition.
            e->vram_eviction_in_progress.store(false, std::memory_order_release);
            e->vram_eviction_cv.notify_all();
            e->vram_eviction_cv.wait_for(device_lock, std::chrono::milliseconds(5));
            // After wait, while condition re-evaluates. If DMA completed, budget satisfied → exit.
            // Otherwise, continue to retry (up to DMA_DRAIN_MAX_RETRIES times).
        } else {
            // Reset retry counter on successful eviction
            dma_drain_retries = 0;
        }
    }
}
