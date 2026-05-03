#include <cassert>
// BUGFIX 58/62: Issue 4A - Dynamic batch_max based on backend_caps ?�★
// Problem: moe_gpu_stage_batch_max = 4 hardcoded
// PCIe 5.0 x16: 4 too low (underutilizes bandwidth)
// PCIe 3.0 x4: 4 too high (causes queue backup)
// Solution: Use backend_caps.copy_streams as upper bound
// Allows dynamic adaptation to hardware capabilities
static uint32_t moe_gpu_stage_batch_max(const moe_pc_engine* engine) {
    if (!engine || engine->backend_caps.copy_streams <= 0) {
        return 4;
    }
    const uint32_t streams = engine->backend_caps.copy_streams;
    const uint32_t derived = streams > UINT32_MAX / 4u ? UINT32_MAX : streams * 4u;
    return std::max(4u, std::min(derived, 32u));
}

static void reset_staging_slots(moe_pc_engine* engine) {
    if (!engine) {
        return;
    }
    // BUGFIX 168: Prevent concurrent access during clear()
    // staging_slots.clear() invalidates iterators and pointers while workers may be accessing slots.
    // Solution: Set free_slot_count to 0 first to block new acquisitions, then wait for in-use slots.
    std::lock_guard<std::mutex> lock(engine->staging_slot_mutex);

    // Step 1: Block new slot acquisitions
    engine->staging_free_slot_count.store(0, std::memory_order_release);

    // Step 2: Wait for all slots to be released (in_use == false)
    // Timeout after 5 seconds to prevent deadlock if workers are stuck
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    bool all_released = false;
    while (!all_released && std::chrono::steady_clock::now() < deadline) {
        all_released = true;
        for (const auto& slot : engine->staging_slots) {
            if (slot.in_use.load(std::memory_order_acquire)) {
                all_released = false;
                break;
            }
        }
        if (!all_released) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    // BUGFIX 252: Do NOT proceed with clear() if slots still in use ★★★
    // Problem: Timeout warning printed but clear() still called → use-after-free
    // Workers still accessing slots when memory is freed → crash
    // Solution: Return early on timeout, leave slots intact
    // Trade-off: Staging slots leak, but prevents crash
    if (!all_released) {
        fprintf(stderr, "[storageLLM ERROR] reset_staging_slots: Some slots still in use after 5s timeout. "
                "Cannot safely reset - aborting reset to prevent use-after-free. "
                "Staging slots will leak until engine shutdown.\n");
        // 최소한 ring_adapter를 drain 시도 (free는 하지 않음)
        engine->ring_adapter.drain_without_free();
        // Restore free_slot_count to allow continued operation
        engine->staging_free_slot_count.store(static_cast<uint32_t>(engine->staging_slots.size()),
                                              std::memory_order_release);
        return;
    }

    engine->staging_generation.fetch_add(1, std::memory_order_relaxed);
    engine->ring_adapter.cleanup();
    // Slots are non-owning slices into engine->staging. The caller owns and
    // frees the single backing staging buffer when capacity/config changes.
    for (auto& slot : engine->staging_slots) {
        slot.buf = nullptr;
        slot.size = 0;
    }
    engine->staging_slots.clear();
    engine->staging_slot_size = 0;
#ifdef ENABLE_LOCKFREE_STAGING
    engine->staging_slot_free_mask.store(0, std::memory_order_release);
#endif
    engine->staging_pinned_slot_count.store(0, std::memory_order_release);
    engine->staging_gpu_slot_count.store(0, std::memory_order_release);
    engine->staging_direct_slot_count.store(0, std::memory_order_release);
    // BUGFIX 204: Add memory_order_relaxed for statistics counter ★
    engine->io_atomic.staging_slot_deficit.store(0, std::memory_order_relaxed);
    // BUGFIX 68: Issue 5 - Notify all 3 CVs on reset ??
    engine->staging_pinned_cv.notify_all();
    engine->staging_gpu_cv.notify_all();
    engine->staging_direct_cv.notify_all();
    engine->staging_slot_cv.notify_all();
}

static uint64_t moe_staging_min_slot_bytes(const moe_pc_engine* engine) {
    // Size slots to actual tensor data + small padding.
    // Old 64MB floor wasted ~44MB per slot for NVFP4 (~20MB triplets).
    const moe_storage_constants_t* storage = moe_storage_constants();
    const uint64_t bundle = storage ? storage->normal_bundle_max_bytes : 0ull;
    const uint64_t base = engine ? std::max<uint64_t>(engine->max_tensor_stream_bytes, bundle) : bundle;
    const uint64_t padding = 4ull * 1024ull * 1024ull;
    uint64_t sz = base > 0 ? base + padding : padding;
    return (sz + 31ull) & ~31ull;
}

static uint32_t moe_staging_direct_slot_target(const moe_pc_engine* engine, uint32_t disk_workers) {
    if (!engine ||
        !engine_has_device_upload_workers(engine) ||
        !engine->backend_caps.supports_fixed_read_staging ||
        !engine->io_config.enable_multistage_pipeline ||
        !engine->staging ||
        !engine->staging_pinned ||
        engine->io_config.direct_upload_min_bytes == 0) {
        return 0;
    }
    return std::max(1u, disk_workers);
}

static void moe_staging_slot_targets(
    const moe_pc_engine* engine,
    uint32_t* pinned_out,
    uint32_t* gpu_out,
    uint32_t* direct_out
) {
    uint32_t pinned_workers = runtime_pinned_workers(engine);
    uint32_t disk_workers = runtime_disk_workers(engine);
    uint32_t gpu_workers = runtime_gpu_workers(engine);
    if (disk_workers == 0 && engine) {
        disk_workers = engine->io_config.disk_worker_threads ?
            engine->io_config.disk_worker_threads : engine->io_config.io_worker_threads;
    }
    if (pinned_workers == 0 && engine && !engine_bypass_staging_pipeline(engine)) {
        pinned_workers = 1;
    }
    if (gpu_workers == 0 && engine_has_device_upload_workers(engine)) {
        gpu_workers = 1;
    }
    const uint32_t gpu_batch_eff = moe_gpu_stage_batch_max(engine);
    uint32_t pinned_target = pinned_workers > 0 ? std::max(1u, pinned_workers * 2u) : 0u;
    uint32_t gpu_target = gpu_workers > 0 ? std::max(1u, gpu_workers * gpu_batch_eff) : 0u;
    uint32_t direct_target = moe_staging_direct_slot_target(engine, disk_workers);

    if (engine && engine->backend_caps.backend == moe_BACKEND_METAL) {
        pinned_target = 0;
        gpu_target = 0;
        direct_target = disk_workers > 0 ? std::max(1u, disk_workers) : 1u;
    }

    if (pinned_out) *pinned_out = pinned_target;
    if (gpu_out) *gpu_out = gpu_target;
    if (direct_out) *direct_out = direct_target;
}

// BUGFIX 75: Issue 10 - Use largest remainder method for slot distribution ?�★
// Problem: Integer division rounding causes direct_slots to become 0
// Example: total=7, ratio 2:3:1 ??pinned=2, gpu=3, direct=1, sum=6
// while loop adds 1 to gpu ??direct=0 ??DirectStorage disabled
// Solution: Largest remainder method ensures fair distribution
// 1. Allocate integer parts first
// 2. Distribute remainders to slots with largest fractional parts
static void moe_scale_staging_slot_counts(
    uint32_t total_count,
    uint32_t pinned_target,
    uint32_t gpu_target,
    uint32_t direct_target,
    uint32_t* pinned_out,
    uint32_t* gpu_out,
    uint32_t* direct_out
) {
    // BUGFIX 544: Check output pointers before use ★
    // Problem: pinned_out, gpu_out, direct_out can be null
    // Function writes to them without checking, causing null pointer dereference
    // Solution: Return early if any output pointer is null
    if (!pinned_out || !gpu_out || !direct_out) {
        return;
    }

    uint32_t pinned = 0;
    uint32_t gpu = 0;
    uint32_t direct = 0;
    const uint64_t desired_total =
        (uint64_t)pinned_target + (uint64_t)gpu_target + (uint64_t)direct_target;

    // BUGFIX 282: Add explicit check for desired_total > 0 before division ★
    // Problem: While outer if checks desired_total > 0, the division happens in else branch
    // If desired_total somehow becomes 0 between check and division (race condition or compiler optimization)
    // division by zero occurs
    // Solution: Add defensive check right before division operations
    if (total_count > 0 && desired_total > 0) {
        if (desired_total <= total_count) {
            pinned = pinned_target;
            gpu = gpu_target;
            direct = direct_target;
        } else {
            // Largest remainder method for fair distribution
            // Step 1: Allocate integer parts
            // BUGFIX 282: Recheck desired_total > 0 for defense in depth ★
            if (desired_total == 0) {
                // Should never happen, but prevents UB
                if (pinned_out) *pinned_out = 0;
                if (gpu_out) *gpu_out = 0;
                if (direct_out) *direct_out = 0;
                return;
            }
            const double pinned_exact = ((double)total_count * pinned_target) / (double)desired_total;
            const double gpu_exact = ((double)total_count * gpu_target) / (double)desired_total;
            const double direct_exact = ((double)total_count * direct_target) / (double)desired_total;

            pinned = (uint32_t)pinned_exact;
            gpu = (uint32_t)gpu_exact;
            direct = (uint32_t)direct_exact;

            // Step 2: Distribute remainder based on fractional parts
            uint32_t allocated = pinned + gpu + direct;
            if (allocated < total_count) {
                uint32_t remainder = total_count - allocated;
                struct entry { double frac; uint32_t* target; uint32_t tgt; };
                entry entries[] = {
                    {pinned_exact - pinned, &pinned, pinned_target},
                    {gpu_exact - gpu, &gpu, gpu_target},
                    {direct_exact - direct, &direct, direct_target}
                };
                std::sort(entries, entries+3, [](auto& a, auto& b){ return a.frac > b.frac; });
                for (uint32_t i = 0; i < remainder && i < 3; ++i) {
                    if (entries[i].tgt > 0) ++(*entries[i].target);
                }
            }

            // Ensure non-zero targets get at least 1 slot
            auto ensure_one = [&](uint32_t target, uint32_t& value) {
                const uint32_t sum = pinned + gpu + direct;
                if (target > 0 && value == 0 && sum < total_count) {
                    value = 1;
                }
            };
            ensure_one(gpu_target, gpu);
            ensure_one(direct_target, direct);
            ensure_one(pinned_target, pinned);
        }

        // Final adjustment if sum exceeds total (shouldn't happen with LRM, but safety check)
        auto sum_counts = [&]() { return pinned + gpu + direct; };
        while (sum_counts() > total_count) {
            if (gpu > 1 && gpu >= pinned && gpu >= direct) {
                --gpu;
            } else if (direct > 1 && direct >= pinned) {
                --direct;
            } else if (pinned > 1) {
                --pinned;
            } else if (gpu > 0) {
                --gpu;
            } else if (direct > 0) {
                --direct;
            } else {
                --pinned;
            }
        }
    }

    // BUGFIX: Guard against direct slots dropping to 0 due to rounding
    if (direct_target > 0 && direct == 0 && total_count >= 3) {
        direct = 1;
        if (gpu > 0) {
            gpu -= 1;
        } else if (pinned > 0) {
            pinned -= 1;
        }
    }

    #ifndef NDEBUG
    assert(pinned + gpu + direct <= total_count);
#endif
    if (pinned_out) *pinned_out = pinned;
    if (gpu_out) *gpu_out = gpu;
    if (direct_out) *direct_out = direct;
}

static uint32_t moe_staging_min_slot_count(const moe_pc_engine* engine) {
    if (!engine || !engine->io_config.enable_pinned_host ||
        !engine->backend_caps.supports_pinned_host ||
        engine_bypass_staging_pipeline(engine)) {
        return 0;
    }
    uint32_t pinned_target = 0;
    uint32_t gpu_target = 0;
    uint32_t direct_target = 0;
    moe_staging_slot_targets(engine, &pinned_target, &gpu_target, &direct_target);
    const uint32_t total = pinned_target + gpu_target + direct_target;

    // BUGFIX 194: Add worker-based minimum to prevent structural slot shortage ★★★
    // Problem: slot_count based only on target ratios, can be less than worker count
    // → Workers spin-wait for slots, pipeline stalls
    // Solution: Ensure min slots >= (disk + pinned + gpu) workers × 2
    // × 2 factor allows one slot in-flight, one being filled
    // Impact: Eliminates structural slot shortage, improves throughput
    const uint32_t worker_based_min =
        runtime_disk_workers(engine) +
        runtime_pinned_workers(engine) +
        runtime_gpu_workers(engine);
    const uint32_t worker_min_with_margin = worker_based_min > 0 ? worker_based_min * 2u : 0u;

    const uint32_t original_min = total > 0 ? std::max(4u, total) : 0u;
    return std::max(original_min, worker_min_with_margin);
}

static uint32_t moe_staging_target_slot_count(const moe_pc_engine* engine) {
    uint32_t pinned_target = 0;
    uint32_t gpu_target = 0;
    uint32_t direct_target = 0;
    moe_staging_slot_targets(engine, &pinned_target, &gpu_target, &direct_target);
    return pinned_target + gpu_target + direct_target;
}

static uint32_t moe_staging_max_slot_target(const moe_pc_engine* engine) {
    if (!engine || !engine->io_config.enable_pinned_host ||
        !engine->backend_caps.supports_pinned_host ||
        engine_bypass_staging_pipeline(engine)) {
        return 0;
    }
    return std::max(
        moe_staging_target_slot_count(engine),
        moe_staging_min_slot_count(engine));
}

static uint64_t moe_recommended_staging_bytes(const moe_pc_engine* engine) {
    const uint32_t min_slots = std::max(
        moe_staging_min_slot_count(engine),
        moe_staging_max_slot_target(engine));
    if (!engine || min_slots == 0) {
        return 0;
    }
    const uint32_t target_slots = min_slots + std::max(1u, min_slots / 4u);
    return (uint64_t)target_slots * moe_staging_min_slot_bytes(engine);
}

static int moe_ensure_staging_capacity(moe_pc_engine* engine, uint64_t required_bytes) {
    if (!engine || required_bytes == 0 || engine->staging_size >= required_bytes) {
        return 1;
    }
    if (!engine->io_config.enable_pinned_host) {
        return 0;
    }

    bool new_pinned = false;
    bool new_numa = false;
    uint64_t actual_bytes = 0;
    uint8_t* new_staging = allocate_best_staging(
        engine,
        required_bytes,
        &actual_bytes,
        &new_pinned,
        &new_numa);
    if (!new_staging) {
        fprintf(stderr,
            "[storageLLM WARNING] unable to grow staging from %llu to %llu bytes; keeping existing staging\n",
            (unsigned long long)engine->staging_size,
            (unsigned long long)required_bytes);
        return 0;
    }

    free_staging(engine->staging, engine->staging_size, engine->staging_pinned, engine->staging_numa_allocated);
    engine->staging = new_staging;
    engine->staging_size = actual_bytes;
    engine->staging_pinned = new_pinned;
    engine->staging_numa_allocated = new_numa;
    engine->io_config.pinned_staging_bytes = actual_bytes;
    if (!new_pinned) {
        fprintf(stderr,
            "[storageLLM WARNING] staging grew to %llu bytes as non-pinned memory; DMA uses sync fallback\n",
            (unsigned long long)actual_bytes);
    } else {
        fprintf(stderr,
            "[storageLLM INFO] staging grew to %llu bytes for worker slot budget\n",
            (unsigned long long)actual_bytes);
    }
    return actual_bytes >= required_bytes ? 1 : 0;
}

static void update_staging_slot_deficit(moe_pc_engine* engine) {
    if (!engine) {
        return;
    }
    const uint32_t min_slots = moe_staging_min_slot_count(engine);
    uint32_t deficit = 0;
    if (min_slots > 0) {
        const uint32_t actual = static_cast<uint32_t>(engine->staging_slots.size());
        if (actual < min_slots) {
            deficit = min_slots - actual;
        }
    }
    engine->io_atomic.staging_slot_deficit.store(deficit, std::memory_order_relaxed);  // BUGFIX 204: Add memory_order
    // Problem 5 Fix: Don't increment staging_wraps here ??that's for transient contention
    // staging_slot_deficit = structural shortage (config issue, tracked here)
    // staging_slot_contention = transient slot exhaustion (tracked in prefetch_bytes.cpp.inc)
    // Mixing them in staging_wraps made diagnostics impossible.
}

static void update_staging_slot_split_for_workers(moe_pc_engine* engine) {
    if (!engine) {
        return;
    }
    if (engine_bypass_staging_pipeline(engine) || !engine->backend_caps.supports_pinned_host) {
        engine->staging_pinned_slot_count.store(0, std::memory_order_release);
        engine->staging_gpu_slot_count.store(0, std::memory_order_release);
        engine->staging_direct_slot_count.store(0, std::memory_order_release);
        update_staging_slot_deficit(engine);
        return;
    }
    uint32_t pinned_slot_target = 0;
    uint32_t gpu_slot_target = 0;
    uint32_t direct_slot_target = 0;
    moe_staging_slot_targets(engine, &pinned_slot_target, &gpu_slot_target, &direct_slot_target);
    const uint32_t total_count = static_cast<uint32_t>(engine->staging_slots.size());
    uint32_t pinned_slots = 0;
    uint32_t gpu_slots = 0;
    uint32_t direct_slots = 0;
    moe_scale_staging_slot_counts(
        total_count,
        pinned_slot_target,
        gpu_slot_target,
        direct_slot_target,
        &pinned_slots,
        &gpu_slots,
        &direct_slots);
    engine->staging_pinned_slot_count.store(pinned_slots, std::memory_order_release);
    engine->staging_gpu_slot_count.store(gpu_slots, std::memory_order_release);
    engine->staging_direct_slot_count.store(direct_slots, std::memory_order_release);
    update_staging_slot_deficit(engine);
}

static void configure_staging_slots(moe_pc_engine* engine) {
    reset_staging_slots(engine);
    if (!engine || !engine->staging || engine->staging_size == 0 ||
        engine_bypass_staging_pipeline(engine) ||
        !engine->backend_caps.supports_pinned_host) {
        update_staging_slot_deficit(engine);
        return;
    }

    // BUGFIX 문제1: Override pinned flag for Metal unified memory ★★★
    // Problem: Metal uses unified memory (CPU/GPU share same memory)
    // mlock is unnecessary and can silently fail under memory pressure
    // Solution: Force staging_pinned=false for Metal backend
    // Impact: No regression - Metal path uses metal_zero_copy_map() which doesn't need pinned memory
    if (engine->backend_caps.backend == moe_BACKEND_METAL && engine->staging_pinned) {
        fprintf(stderr, "[storageLLM INFO] Metal unified memory detected - disabling mlock for staging buffers\n");
        engine->staging_pinned = false;
    }

    // PIPELINE FIX 5: Validate initialization order ★
    // max_tensor_stream_bytes must be set BEFORE configure_staging_slots is called.
    // It's set during tensor index parsing (safetensors manifest load).
    // If 0, slot size defaults to 4MB ??NVFP4 expert triplets (~20MB) fail to copy.
    if (engine->max_tensor_stream_bytes == 0) {
        fprintf(stderr, "[storageLLM CRITICAL] configure_staging_slots called before "
            "max_tensor_stream_bytes was set. Tensors must be indexed first. "
            "Slot size will be 4MB, causing copy failures for large tensors.\n");
        update_staging_slot_deficit(engine);
        return;
    }

    const uint64_t recommended_staging = moe_recommended_staging_bytes(engine);
    if (recommended_staging > engine->staging_size) {
        moe_ensure_staging_capacity(engine, recommended_staging);
    }
    if (!engine->staging || engine->staging_size == 0) {
        update_staging_slot_deficit(engine);
        return;
    }

    uint32_t pinned_slot_target = 0;
    uint32_t gpu_slot_target = 0;
    uint32_t direct_slot_target = 0;
    moe_staging_slot_targets(engine, &pinned_slot_target, &gpu_slot_target, &direct_slot_target);
    uint32_t slot_count = pinned_slot_target + gpu_slot_target + direct_slot_target;
    const uint32_t max_slot_target = moe_staging_max_slot_target(engine);
    if (slot_count == 0) {
        slot_count = 1;
    }
    const uint64_t min_slot = moe_staging_min_slot_bytes(engine);
    // BUGFIX 145: Check min_slot > 0 before division to prevent division by zero
    // BUGFIX 583: Also check staging_size > 0 ★
    // Problem: staging_size can be 0
    // Solution: Check both before division
    if (min_slot > 0 && engine->staging_size > 0 && engine->staging_size >= min_slot && max_slot_target > slot_count) {
        const uint32_t byte_limited_slots = static_cast<uint32_t>(engine->staging_size / min_slot);
        if (byte_limited_slots > slot_count) {
            slot_count = std::min(max_slot_target, byte_limited_slots);
        }
    }
    const uint32_t max_by_bytes = min_slot > 0 ? static_cast<uint32_t>(engine->staging_size / min_slot) : slot_count;
    // BUGFIX 584: Check slot_count > 0 before comparison ★
    // Problem: slot_count can be 0
    // Solution: Check before comparison
    if (slot_count > 0 && slot_count > max_by_bytes) {
        slot_count = max_by_bytes > 0 ? max_by_bytes : 1u;
    }
    if (slot_count == 0) {
        update_staging_slot_deficit(engine);
        return;
    }

    uint64_t slot_size = engine->staging_size / slot_count;
    // BUGFIX 145: Check min_slot > 0 before division
    // BUGFIX 585: Also check slot_size > 0 and staging_size > 0 ★
    // Problem: slot_size can be 0 if staging_size < slot_count
    // Solution: Check all conditions
    if (min_slot > 0 && slot_size > 0 && slot_size < min_slot && engine->staging_size > 0 && engine->staging_size >= min_slot) {
        slot_count = static_cast<uint32_t>(engine->staging_size / min_slot);
        if (slot_count == 0) slot_count = 1;
        slot_size = engine->staging_size / slot_count;
    }
    // Fix 66: gpu_workers need slots too (deferred release until stream sync)
    const uint32_t min_slots = moe_staging_min_slot_count(engine);

    // BUGFIX 문제1: Enforce worker-based minimum with memory budget adjustment ★★★
    // Problem: Memory budget can force slot_count below worker count → structural shortage
    // Old: slot_count = staging_size / slot_size, can be < worker_count
    // New: Ensure slot_count >= max(min_slots, worker_count * 1.5)
    // If memory insufficient, reduce slot_size (strip padding) or warn user
    const uint32_t worker_count = runtime_disk_workers(engine) +
                                   runtime_pinned_workers(engine) +
                                   runtime_gpu_workers(engine);
    const uint32_t worker_based_min = worker_count > 0 ?
        std::max(worker_count, (worker_count * 3u) / 2u) : 0u;
    const uint32_t enforced_min_slots = std::max(min_slots, worker_based_min);

    // BUGFIX 586: Check min_slots > 0 and staging_size / min_slots doesn't overflow ★
    // Problem: staging_size / min_slots can overflow or be invalid
    // Solution: Check min_slots > 0 before division
    if (enforced_min_slots > 0 && slot_count < enforced_min_slots) {
        // Try to meet enforced minimum by reducing slot size
        const uint64_t reduced_slot_size = engine->staging_size / enforced_min_slots;

        if (reduced_slot_size >= engine->max_tensor_stream_bytes) {
            // Can meet minimum by stripping padding
            slot_count = enforced_min_slots;
            slot_size = reduced_slot_size;
            fprintf(stderr, "[storageLLM INFO] Adjusted slot_count to %u (worker-based minimum) "
                    "by reducing slot_size to %llu bytes\n",
                    slot_count, (unsigned long long)slot_size);
        } else if (engine->staging_size / enforced_min_slots >= min_slot) {
            // Can meet minimum with original min_slot calculation
            slot_count = enforced_min_slots;
            slot_size = engine->staging_size / slot_count;
            fprintf(stderr, "[storageLLM INFO] Adjusted slot_count to %u (worker-based minimum)\n",
                    slot_count);
        } else {
            // Memory budget insufficient - need to increase staging_size
            const uint64_t required_staging = enforced_min_slots * moe_staging_min_slot_bytes(engine);
            fprintf(stderr,
                "[storageLLM ERROR] Staging memory budget insufficient for worker count!\n"
                "  Workers: %u (disk=%u, pinned=%u, gpu=%u)\n"
                "  Required minimum slots: %u\n"
                "  Current staging_size: %llu bytes\n"
                "  Required staging_size: %llu bytes\n"
                "  Increase staging_size or reduce worker threads to prevent pipeline stalls.\n",
                worker_count,
                runtime_disk_workers(engine),
                runtime_pinned_workers(engine),
                runtime_gpu_workers(engine),
                enforced_min_slots,
                (unsigned long long)engine->staging_size,
                (unsigned long long)required_staging);

            // Use what we can, but warn about structural shortage
            if (slot_count < min_slots && engine->staging_size / min_slots >= engine->max_tensor_stream_bytes) {
                slot_count = min_slots;
                slot_size = engine->staging_size / slot_count;
            }
        }
    }
    if (slot_size == 0) {
        update_staging_slot_deficit(engine);
        return;
    }
    engine->staging_slot_size = slot_size;
    engine->staging_slots.reserve(slot_count);
#if defined(__APPLE__) && defined(__MACH__)
extern "C" void* metal_zero_copy_map(void* device_handle, void* src, uint64_t bytes);
#endif

    for (uint32_t i = 0; i < slot_count; ++i) {
        moe_staging_slot slot;
        slot.buf = engine->staging + slot_size * i;
        // BUGFIX 587: Check slot_size * i doesn't overflow ★
        // Problem: slot_size * i can overflow
        // Solution: Check before calculation
        const uint64_t offset = (uint64_t)slot_size * i;
        if (offset > engine->staging_size) {
            fprintf(stderr, "[storageLLM ERROR] Staging slot offset overflow at slot %u\n", i);
            break;
        }
        slot.buf = engine->staging + offset;
        slot.size = i + 1 == slot_count ? engine->staging_size - offset : slot_size;
        slot.pinned = engine->staging_pinned;
        slot.numa_allocated = engine->staging_numa_allocated;
#if defined(__APPLE__) && defined(__MACH__)
        if (engine->backend_caps.backend == moe_BACKEND_METAL) {
            slot.metal_buffer = metal_zero_copy_map(nullptr, slot.buf, slot.size);
        } else {
            slot.metal_buffer = nullptr;
        }
#else
        slot.metal_buffer = nullptr;
#endif
        slot.in_use.store(false, std::memory_order_relaxed);  // BUGFIX 204: Add memory_order
        engine->staging_slots.emplace_back(std::move(slot));
    }
#ifdef ENABLE_LOCKFREE_STAGING
    if (slot_count <= 64u) {
        const uint64_t free_mask = slot_count == 64u ? UINT64_MAX : ((1ull << slot_count) - 1ull);
        engine->staging_slot_free_mask.store(free_mask, std::memory_order_release);
    } else {
        engine->staging_slot_free_mask.store(0, std::memory_order_release);
    }
#endif

    // CRITICAL FIX: When staging is non-pinned, log warning about sync DMA performance impact
    if (!engine->staging_pinned && slot_count > 0) {
        fprintf(stderr, "[storageLLM WARNING] All %u staging slots are NON-PINNED (mlock failed). "
            "DMA will be SYNCHRONOUS, blocking IO workers. Performance will be severely degraded. "
            "Increase 'ulimit -l' to at least %llu bytes for async DMA.\n",
            slot_count,
            (unsigned long long)engine->staging_size);
    }
    engine->staging_free_slot_count.store(slot_count, std::memory_order_release);

    // Register staging slots with IORing/io_uring
    if (engine->backend_caps.supports_registered_io_buffers && slot_count > 0 && engine->staging_pinned) {
        const uint32_t queue_depth = std::max(64u, slot_count * 2);
        if (engine->ring_adapter.init(queue_depth)) {
            std::vector<void*> bufs(slot_count);
            std::vector<size_t> lens(slot_count);
            for (uint32_t i = 0; i < slot_count; ++i) {
                bufs[i] = engine->staging_slots[i].buf;
                lens[i] = engine->staging_slots[i].size;
            }
            if (engine->ring_adapter.register_buffers(bufs.data(), lens.data(), slot_count)) {
                fprintf(stderr, "[storageLLM INFO] Successfully registered %u staging buffers with io_uring/IORing.\n", slot_count);
            } else {
                // BUGFIX 119: Cleanup ring_adapter on register_buffers failure
                // If init() succeeded but register_buffers() failed, ring_adapter is in
                // initialized state but buffers are not registered. This causes
                // "buffer not registered" errors when io_uring path is attempted.
                // Solution: Call cleanup() to disable io_uring path entirely.
                fprintf(stderr, "[storageLLM WARNING] Failed to register staging buffers with io_uring/IORing. Disabling io_uring path.\n");
                engine->ring_adapter.cleanup();
            }
        }
    }

    update_staging_slot_split_for_workers(engine);
    const uint32_t warning_slot_target = pinned_slot_target + gpu_slot_target + direct_slot_target;
    if (slot_count < warning_slot_target) {
        fprintf(stderr, "[WARNING] Insufficient staging slots: %u < %u. Expect pipeline stalls.\n",
            slot_count, warning_slot_target);
    }
    if (slot_size < min_slot) {
        fprintf(stderr, "[WARNING] Staging slot size too small: %llu < %llu. Large tensor staging may fail.\n",
            (unsigned long long)slot_size,
            (unsigned long long)min_slot);
    }
}
