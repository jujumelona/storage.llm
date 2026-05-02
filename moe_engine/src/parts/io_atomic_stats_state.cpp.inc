// Bug #18: Separate hot counters by worker type to eliminate false sharing
// Each worker type gets its own cache line (alignas(64))

struct alignas(64) moe_io_atomic_stats {
    // --- Cache line 1: general counters (written by orchestrator) ---
    std::atomic<uint64_t> queued_requests{0};
    std::atomic<uint64_t> completed_requests{0};
    std::atomic<uint64_t> failed_requests{0};
    std::atomic<uint64_t> dropped_requests{0};
    std::atomic<uint64_t> urgent_dropped_requests{0};  // BUGFIX 201: Separate urgent drop counter
    std::atomic<uint64_t> bytes_prefetched{0};

    // --- Cache line 2: disk worker counters ---
    alignas(64) std::atomic<uint64_t> bytes_disk_to_ram{0};
    std::atomic<uint64_t> bytes_disk_hint_issued{0};
    std::atomic<uint64_t> disk_stage_done{0};
    std::atomic<uint64_t> bytes_touch_fallback{0};  // Fix 38: fallback touch bytes

    // --- Cache line 3: pinned worker counters ---
    alignas(64) std::atomic<uint64_t> bytes_ram_to_pinned{0};
    std::atomic<uint64_t> pinned_stage_done{0};
    std::atomic<uint64_t> staging_wraps{0};

    // --- Cache line 4: GPU worker counters ---
    alignas(64) std::atomic<uint64_t> bytes_pinned_to_gpu{0};
    std::atomic<uint64_t> gpu_stage_done{0};
    std::atomic<uint64_t> bytes_sync_h2d_fallback{0};

    // BUGFIX 38: Separate hot error counters to prevent false sharing ?�★??    // Old: device_alloc_failures and device_copy_failures in same cacheline as gpu_stage_done
    // New: Each hot counter gets its own cacheline
    // Rationale: Multiple io_workers update device_copy_failures simultaneously
    //            ??cacheline ping-pong with gpu_stage_done updates
    // --- Cache line 5: device allocation failures (hot, written by all workers) ---
    alignas(64) std::atomic<uint64_t> device_alloc_failures{0};

    // --- Cache line 6: device copy failures (hot, written by all workers) ---
    alignas(64) std::atomic<uint64_t> device_copy_failures{0};
    std::atomic<uint64_t> nonpinned_fallback_count{0};
    std::atomic<uint64_t> eviction_depth_exceeded{0}; // BUGFIX: Track deep recursion eviction failures

    // --- Cache line 7: direct upload + topology ---
    alignas(64) std::atomic<uint64_t> bytes_direct_to_gpu{0};
    std::atomic<uint64_t> direct_stage_done{0};
    std::atomic<uint64_t> topology_predictions{0};
    std::atomic<uint64_t> topology_updates{0};

    // BUGFIX 38: Separate staging slot contention counter ?�★??    // Old: staging_slot_contention shared cacheline with status flags
    // New: Own cacheline (updated by all pinned_workers on every acquire attempt)
    // --- Cache line 8: staging slot contention (extremely hot) ---
    alignas(64) std::atomic<uint64_t> staging_slot_contention{0};
    std::atomic<uint64_t> staging_slot_waits{0};

    // --- Cache line 9: status flags (read-mostly) ---
    alignas(64) std::atomic<uint32_t> active_path{0};
    std::atomic<uint32_t> pinned_host_active{0};
    std::atomic<uint32_t> directstorage_active{0};
    std::atomic<uint32_t> gpudirect_storage_active{0};
    std::atomic<uint32_t> bam_nvme_active{0};
    std::atomic<uint32_t> staging_slot_deficit{0};
};
