// Device memory eviction and room-making logic

static void device_free_expert_locked(
    moe_pc_engine* e,
    int layer,
    int expert,
    std::vector<std::pair<CUdeviceptr, uint64_t>>* deferred_free_ptrs = nullptr
);

static void device_push_activation_candidate_locked(moe_pc_engine* e, uint64_t key, const moe_device_allocation& a) {
    if (!e || a.pool != moe_DEVICE_POOL_ACTIVATION || a.pending_free || a.bytes == 0) {
        return;
    }
    moe_device_evict_candidate candidate{};
    candidate.key = key;
    candidate.bytes = a.bytes;
    candidate.last_touch_epoch = a.last_touch_epoch;
    e->activation_evict_heap.push(candidate);
}

// BUGFIX 69: Issue 6 - Periodic heap rebuild to remove stale entries.
// BUGFIX 73: Issue 9 - Combine active_count and heap rebuild in single map scan.
// Problem: device_collect_activation_victims_locked scans map twice:
// 1. Count active allocations (for stale ratio check)
// 2. Rebuild heap (scan map again to re-push entries)
// With thousands of allocations, double scan adds ms-level latency under device_lock
// Solution: Rebuild function returns active_count, single map traversal
static uint32_t device_rebuild_activation_heap_locked(moe_pc_engine* e) {
    if (!e) {
        return 0;
    }
    // Clear heap
    while (!e->activation_evict_heap.empty()) {
        e->activation_evict_heap.pop();
    }
    // Single map scan: rebuild heap AND count active allocations
    uint32_t active_count = 0;
    for (auto& [key, alloc] : e->device_allocations) {
        if (alloc.pool == moe_DEVICE_POOL_ACTIVATION && !alloc.pending_free && alloc.bytes > 0) {
            device_push_activation_candidate_locked(e, key, alloc);
            ++active_count;
        }
    }
    return active_count;
}

static uint32_t device_rebuild_activation_heap_unlocked(
    moe_pc_engine* e,
    std::unique_lock<std::mutex>&
) {
    return device_rebuild_activation_heap_locked(e);
}

static int device_free_expert_if_pending_evict_locked(
    moe_pc_engine* e,
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
        moe_block_state& state = e->expert_states[slot];
        if (!state.pending_evict) {
            return 0;
        }
        state.pending_evict = false;
    }
    device_free_expert_locked(e, layer, expert, deferred_free_ptrs);
    return 1;
}

static uint32_t device_collect_activation_victims_locked(
    std::unique_lock<std::mutex>& device_lock,
    moe_pc_engine* e,
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
    // BUGFIX 69: Issue 6 - Trigger heap rebuild when stale ratio high.
    // BUGFIX 73: Issue 9 - Single map scan for count + rebuild.
    // Old: Scan map to count active ??scan map again in rebuild (double traversal)
    // New: Rebuild returns active_count, single map scan
    const uint32_t stale_limit = std::max(128u, (uint32_t)e->activation_evict_heap.size() / 2u);

    // Check if rebuild needed (heap size > 2x active allocations = stale ratio > 50%)
    // Rebuild function does single map scan: count + rebuild simultaneously
    // BUGFIX 88: Issue 8 - Eliminate O(n) scan with atomic active_count
    uint32_t active_count = (uint32_t)e->device_activation_count.load(std::memory_order_relaxed);
    const bool should_rebuild = e->activation_evict_heap.size() > 32u;  // Only check if heap is large
    if (should_rebuild) {
        if (e->activation_evict_heap.size() > active_count * 2u) {
            active_count = device_rebuild_activation_heap_unlocked(e, device_lock);
        }
        const bool was_stale = e->activation_evict_heap.size() > active_count * 2u;
        if (was_stale) {
            stale = 0;  // Reset stale counter after rebuild
        }
    }

    // BUGFIX 20: Fix TOCTOU - snapshot device_allocated_bytes once.
    // Old: Two loads in while condition ??underflow if freed between loads
    // New: Single snapshot ??prevents uint64 wraparound, accurate eviction
    const uint64_t allocated_snap = e->device_allocated_bytes.load(std::memory_order_relaxed);
    while (allocated_snap > projected_free &&
           allocated_snap - projected_free + bytes > budget &&
           !e->activation_evict_heap.empty()) {
        const moe_device_evict_candidate victim = e->activation_evict_heap.top();
        e->activation_evict_heap.pop();
        auto it = e->device_allocations.find(victim.key);
        if (it == e->device_allocations.end() || it->second.pool != moe_DEVICE_POOL_ACTIVATION ||
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
    moe_pc_engine* e, uint8_t pool, uint64_t bytes,
    std::unique_lock<std::mutex>& device_lock,
    std::vector<std::pair<CUdeviceptr, uint64_t>>& deferred_free_ptrs
) {
    // BUGFIX 56: Issue 1 - Release device_lock during DMA drain wait.
    // Problem: device_lock held for up to 20ms (4 retries × 5ms) during DMA drain.
    // All GPU upload workers blocked on device_lock ??entire pipeline frozen.
    // At 10ms/token, this wastes 2 tokens worth of time.
    // Solution: Collect eviction victims with lock held, release lock during wait,
    // reacquire to check progress. Use optimistic reservation to prevent over-allocation.
    if (!e) return;
    const uint64_t budget = device_budget_bytes(e);
    if (budget == 0) return;
    const uint64_t room_bytes = bytes > UINT64_MAX - bytes / 10ull ? UINT64_MAX : bytes + bytes / 10ull;

    static constexpr int DMA_DRAIN_MAX_RETRIES = 4;
    int dma_drain_retries = 0;

    while (e->device_allocated_bytes.load(std::memory_order_relaxed) + bytes > budget) {
        bool expected = false;
        if (!e->vram_eviction_in_progress.compare_exchange_strong(
                expected, true, std::memory_order_acq_rel, std::memory_order_acquire)) {
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
        std::vector<uint64_t> activation_victims;
        device_collect_activation_victims_locked(device_lock, e, room_bytes, activation_victims);
        for (uint64_t key : activation_victims) {
            device_free_locked(e, key, &deferred_free_ptrs);
        }

        // If activation eviction didn't free enough, evict experts
        if (e->device_allocated_bytes.load(std::memory_order_relaxed) + bytes > budget) {
            // BUGFIX 56: Release device_lock before calling demote_cold_until.
            // demote_cold_until can take several ms (heap compaction + map_mutex operations)
            // Holding device_lock during this blocks all GPU workers
            device_lock.unlock();
            demote_cold_until(e, moe_TIER_VRAM, room_bytes, expert_victims);
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
        const uint64_t freed_this_iteration = allocated_before > allocated_after ? allocated_before - allocated_after : 0;

        if (expert_victims.empty() && activation_victims.empty()) {
            break;
        }
        if (freed_this_iteration == 0) {
            if (dma_drain_retries++ >= DMA_DRAIN_MAX_RETRIES) {
                break;
            }
            // BUGFIX 56: Release device_lock during DMA drain wait.
            // Old: Held device_lock for 5ms wait ??all GPU workers blocked
            // New: Release lock, wait, reacquire ??GPU workers can proceed
            e->vram_eviction_in_progress.store(false, std::memory_order_release);
            e->vram_eviction_cv.notify_all();

            device_lock.unlock();
            uint32_t wait_ms = 1u << (dma_drain_retries - 1); // 1, 2, 4, 8
            wait_ms = std::min(wait_ms, 4u);
            std::this_thread::sleep_for(std::chrono::milliseconds(wait_ms));
            device_lock.lock();

            // After reacquiring lock, while condition re-evaluates
        } else {
            dma_drain_retries = 0;
        }
    }
}
