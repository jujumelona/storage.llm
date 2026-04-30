// BUGFIX 58/62: Issue 4A - Dynamic batch_max based on backend_caps ?�★
// Problem: moe_gpu_stage_batch_max = 4 hardcoded
// PCIe 5.0 x16: 4 too low (underutilizes bandwidth)
// PCIe 3.0 x4: 4 too high (causes queue backup)
// Solution: Use backend_caps.copy_streams as upper bound
// Allows dynamic adaptation to hardware capabilities
static uint32_t moe_gpu_stage_batch_max(const moe_pc_engine* engine) {
    if (!engine || engine->backend_caps.copy_streams <= 0) {
        return 4;  // Fallback for CPU-only or uninitialized
    }
    // Cap at min(copy_streams, 8) to prevent excessive queue depth
    // Most PCIe controllers handle 4-8 concurrent DMA efficiently
    return std::min(8u, (uint32_t)engine->backend_caps.copy_streams);
}

static void reset_staging_slots(moe_pc_engine* engine) {
    if (!engine) {
        return;
    }
    std::lock_guard<std::mutex> lock(engine->staging_slot_mutex);
    for (auto& slot : engine->staging_slots) {
        slot.buf = nullptr;
        slot.size = 0;
    }
    engine->staging_slots.clear();
    engine->staging_slot_size = 0;
    engine->staging_pinned_slot_count.store(0, std::memory_order_release);
    engine->staging_gpu_slot_count.store(0, std::memory_order_release);
    engine->staging_direct_slot_count.store(0, std::memory_order_release);
    engine->io_atomic.staging_slot_deficit.store(0);
    // BUGFIX 68: Issue 5 - Notify all 3 CVs on reset ??
    engine->staging_pinned_cv.notify_all();
    engine->staging_gpu_cv.notify_all();
    engine->staging_direct_cv.notify_all();
    engine->staging_slot_cv.notify_all();
}

static uint64_t moe_staging_min_slot_bytes(const moe_pc_engine* engine) {
    // Size slots to actual tensor data + small padding.
    // Old 64MB floor wasted ~44MB per slot for NVFP4 (~20MB triplets).
    const uint64_t base = engine ? engine->max_tensor_stream_bytes : 0ull;
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
    const uint32_t pinned_target = pinned_workers > 0 ? std::max(1u, pinned_workers * 2u) : 0u;
    const uint32_t gpu_target = gpu_workers > 0 ? std::max(1u, gpu_workers * gpu_batch_eff) : 0u;
    const uint32_t direct_target = moe_staging_direct_slot_target(engine, disk_workers);
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
    uint32_t pinned = 0;
    uint32_t gpu = 0;
    uint32_t direct = 0;
    const uint64_t desired_total =
        (uint64_t)pinned_target + (uint64_t)gpu_target + (uint64_t)direct_target;
    if (total_count > 0 && desired_total > 0) {
        if (desired_total <= total_count) {
            pinned = pinned_target;
            gpu = gpu_target;
            direct = direct_target;
        } else {
            // Largest remainder method for fair distribution
            // Step 1: Allocate integer parts
            const double pinned_exact = ((double)total_count * pinned_target) / (double)desired_total;
            const double gpu_exact = ((double)total_count * gpu_target) / (double)desired_total;
            const double direct_exact = ((double)total_count * direct_target) / (double)desired_total;

            pinned = (uint32_t)pinned_exact;
            gpu = (uint32_t)gpu_exact;
            direct = (uint32_t)direct_exact;

            // Step 2: Distribute remainder based on fractional parts
            uint32_t allocated = pinned + gpu + direct;
            if (allocated < total_count) {
                const double pinned_frac = pinned_exact - pinned;
                const double gpu_frac = gpu_exact - gpu;
                const double direct_frac = direct_exact - direct;

                uint32_t remainder = total_count - allocated;
                while (remainder > 0) {
                    // Give to slot with largest fractional part
                    if (direct_frac >= pinned_frac && direct_frac >= gpu_frac && direct_target > 0) {
                        ++direct;
                        --remainder;
                        if (remainder == 0) break;
                    }
                    if (gpu_frac >= pinned_frac && gpu_target > 0) {
                        ++gpu;
                        --remainder;
                        if (remainder == 0) break;
                    }
                    if (pinned_target > 0) {
                        ++pinned;
                        --remainder;
                        if (remainder == 0) break;
                    }
                    // Fallback if all targets are 0 (shouldn't happen)
                    if (pinned_target == 0 && gpu_target == 0 && direct_target == 0) {
                        ++gpu;
                        --remainder;
                    }
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
    return total > 0 ? std::max(4u, total) : 0u;
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
    return (uint64_t)min_slots * moe_staging_min_slot_bytes(engine);
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
    engine->io_atomic.staging_slot_deficit.store(deficit);
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
    // PIPELINE FIX 5: Validate initialization order ?�★
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
    if (engine->staging_size >= min_slot && max_slot_target > slot_count) {
        const uint32_t byte_limited_slots = static_cast<uint32_t>(engine->staging_size / min_slot);
        if (byte_limited_slots > slot_count) {
            slot_count = std::min(max_slot_target, byte_limited_slots);
        }
    }
    const uint32_t max_by_bytes = min_slot > 0 ? static_cast<uint32_t>(engine->staging_size / min_slot) : slot_count;
    if (slot_count > max_by_bytes) {
        slot_count = max_by_bytes > 0 ? max_by_bytes : 1u;
    }
    if (slot_count == 0) {
        update_staging_slot_deficit(engine);
        return;
    }

    uint64_t slot_size = engine->staging_size / slot_count;
    if (slot_size < min_slot && engine->staging_size >= min_slot) {
        slot_count = static_cast<uint32_t>(engine->staging_size / min_slot);
        if (slot_count == 0) slot_count = 1;
        slot_size = engine->staging_size / slot_count;
    }
    // Fix 66: gpu_workers need slots too (deferred release until stream sync)
    const uint32_t min_slots = moe_staging_min_slot_count(engine);
    if (slot_count < min_slots && engine->staging_size / min_slots >= min_slot) {
        slot_count = min_slots;
        slot_size = engine->staging_size / slot_count;
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
        slot.size = i + 1 == slot_count ? engine->staging_size - slot_size * i : slot_size;
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
        slot.in_use.store(false);
        engine->staging_slots.emplace_back(std::move(slot));
    }

    // CRITICAL FIX: When staging is non-pinned, log warning about sync DMA performance impact
    if (!engine->staging_pinned && slot_count > 0) {
        fprintf(stderr, "[storageLLM WARNING] All %u staging slots are NON-PINNED (mlock failed). "
            "DMA will be SYNCHRONOUS, blocking IO workers. Performance will be severely degraded. "
            "Increase 'ulimit -l' to at least %llu bytes for async DMA.\n",
            slot_count,
            (unsigned long long)engine->staging_size);
    }
    engine->staging_free_slot_count.store(slot_count, std::memory_order_release);
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
