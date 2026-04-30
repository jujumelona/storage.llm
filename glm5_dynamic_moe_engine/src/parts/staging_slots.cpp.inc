static constexpr uint32_t glm5_gpu_stage_batch_max = 4;

static void reset_staging_slots(glm5_pc_engine* engine) {
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
    engine->io_atomic.staging_slot_deficit.store(0);
    engine->staging_slot_cv.notify_all();
}

static uint64_t glm5_staging_min_slot_bytes(const glm5_pc_engine* engine) {
    // Size slots to actual tensor data + small padding.
    // Old 64MB floor wasted ~44MB per slot for NVFP4 (~20MB triplets).
    const uint64_t base = engine ? engine->max_tensor_stream_bytes : 0ull;
    const uint64_t padding = 4ull * 1024ull * 1024ull;
    uint64_t sz = base > 0 ? base + padding : padding;
    return (sz + 31ull) & ~31ull;
}

static uint32_t glm5_staging_min_slot_count(const glm5_pc_engine* engine) {
    if (!engine || !engine->io_config.enable_pinned_host ||
        !engine->backend_caps.supports_pinned_host ||
        engine_bypass_staging_pipeline(engine)) {
        return 0;
    }
    uint32_t pinned_workers = runtime_pinned_workers(engine);
    if (pinned_workers == 0) pinned_workers = 1;
    uint32_t disk_workers = runtime_disk_workers(engine);
    if (disk_workers == 0) disk_workers = engine->io_config.disk_worker_threads ? engine->io_config.disk_worker_threads : 1u;
    // Bug Fix: GPU worker slots missing. Each gpu_worker holds up to 8 slots
    // during device_stream_sync_current(). Without this, staging starvation occurs.
    uint32_t gpu_workers = runtime_gpu_workers(engine);
    if (gpu_workers == 0) gpu_workers = 1;
    // Keep the staging GPU batch size aligned with gpu_worker_loop.
    const uint32_t gpu_batch_eff = glm5_gpu_stage_batch_max;
    return std::max(4u, (disk_workers + pinned_workers) * 2u + gpu_workers * gpu_batch_eff);
}

static uint32_t glm5_staging_target_slot_count_for_workers(uint32_t pinned_workers, uint32_t gpu_workers) {
    if (pinned_workers == 0) {
        pinned_workers = 1;
    }
    if (gpu_workers == 0) {
        gpu_workers = 1;
    }
    // Keep the staging GPU batch size aligned with gpu_worker_loop.
    const uint32_t gpu_batch_eff = glm5_gpu_stage_batch_max;
    const uint32_t pinned_slot_target = std::max(1u, pinned_workers * 2u);
    const uint32_t gpu_slot_target = std::max(1u, gpu_workers * gpu_batch_eff);
    return pinned_slot_target + gpu_slot_target;
}

static uint32_t glm5_staging_max_slot_target(const glm5_pc_engine* engine) {
    if (!engine || !engine->io_config.enable_pinned_host ||
        !engine->backend_caps.supports_pinned_host ||
        engine_bypass_staging_pipeline(engine)) {
        return 0;
    }
    const uint32_t pinned_workers = std::max(runtime_pinned_workers(engine), pinned_worker_cap(engine));
    const uint32_t gpu_workers = std::max(runtime_gpu_workers(engine), gpu_worker_cap(engine));
    return std::max(
        glm5_staging_target_slot_count_for_workers(pinned_workers, gpu_workers),
        glm5_staging_min_slot_count(engine));
}

static uint64_t glm5_recommended_staging_bytes(const glm5_pc_engine* engine) {
    const uint32_t min_slots = std::max(
        glm5_staging_min_slot_count(engine),
        glm5_staging_max_slot_target(engine));
    if (!engine || min_slots == 0) {
        return 0;
    }
    return (uint64_t)min_slots * glm5_staging_min_slot_bytes(engine);
}

static void update_staging_slot_deficit(glm5_pc_engine* engine) {
    if (!engine) {
        return;
    }
    const uint32_t min_slots = glm5_staging_min_slot_count(engine);
    uint32_t deficit = 0;
    if (min_slots > 0) {
        const uint32_t actual = static_cast<uint32_t>(engine->staging_slots.size());
        if (actual < min_slots) {
            deficit = min_slots - actual;
        }
    }
    engine->io_atomic.staging_slot_deficit.store(deficit);
    // Problem 5 Fix: Don't increment staging_wraps here — that's for transient contention
    // staging_slot_deficit = structural shortage (config issue, tracked here)
    // staging_slot_contention = transient slot exhaustion (tracked in prefetch_bytes.cpp.inc)
    // Mixing them in staging_wraps made diagnostics impossible.
}

static void update_staging_slot_split_for_workers(glm5_pc_engine* engine) {
    if (!engine) {
        return;
    }
    if (engine_bypass_staging_pipeline(engine) || !engine->backend_caps.supports_pinned_host) {
        engine->staging_pinned_slot_count.store(0, std::memory_order_release);
        update_staging_slot_deficit(engine);
        return;
    }
    uint32_t pinned_workers = runtime_pinned_workers(engine);
    uint32_t gpu_workers = runtime_gpu_workers(engine);
    if (pinned_workers == 0) {
        pinned_workers = 1;
    }
    if (gpu_workers == 0) {
        gpu_workers = 1;
    }
    // Keep the staging GPU batch size aligned with gpu_worker_loop.
    const uint32_t gpu_batch_eff = glm5_gpu_stage_batch_max;
    const uint32_t pinned_slot_target = std::max(1u, pinned_workers * 2u);
    const uint32_t gpu_slot_target = std::max(1u, gpu_workers * gpu_batch_eff);
    const uint32_t total_count = static_cast<uint32_t>(engine->staging_slots.size());
    uint32_t pinned_slots = 0;
    if (total_count > 0) {
        const uint64_t desired_total = (uint64_t)pinned_slot_target + (uint64_t)gpu_slot_target;
        pinned_slots = desired_total > 0
            ? static_cast<uint32_t>(((uint64_t)total_count * (uint64_t)pinned_slot_target) / desired_total)
            : total_count / 2u;
        pinned_slots = std::min(pinned_slots, total_count);
        if (pinned_slots == 0 && total_count > 1) {
            pinned_slots = 1;
        }
        if (pinned_slots >= total_count && total_count > 1) {
            pinned_slots = total_count - 1u;
        }
    }
    engine->staging_pinned_slot_count.store(pinned_slots, std::memory_order_release);
    update_staging_slot_deficit(engine);
}

static void configure_staging_slots(glm5_pc_engine* engine) {
    reset_staging_slots(engine);
    if (!engine || !engine->staging || engine->staging_size == 0 ||
        engine_bypass_staging_pipeline(engine) ||
        !engine->backend_caps.supports_pinned_host) {
        update_staging_slot_deficit(engine);
        return;
    }
    // PIPELINE FIX 5: Validate initialization order ★★
    // max_tensor_stream_bytes must be set BEFORE configure_staging_slots is called.
    // It's set during tensor index parsing (safetensors manifest load).
    // If 0, slot size defaults to 4MB → NVFP4 expert triplets (~20MB) fail to copy.
    if (engine->max_tensor_stream_bytes == 0) {
        fprintf(stderr, "[storageLLM CRITICAL] configure_staging_slots called before "
            "max_tensor_stream_bytes was set. Tensors must be indexed first. "
            "Slot size will be 4MB, causing copy failures for large tensors.\n");
        update_staging_slot_deficit(engine);
        return;
    }

    uint32_t disk_workers = runtime_disk_workers(engine);
    uint32_t pinned_workers = runtime_pinned_workers(engine);
    if (disk_workers == 0) {
        disk_workers = engine->io_config.io_worker_threads ? engine->io_config.io_worker_threads : 1u;
    }
    if (pinned_workers == 0) {
        pinned_workers = 1;
    }

    // Include gpu_workers because they hold slots until stream sync.
    uint32_t gpu_workers = runtime_gpu_workers(engine);
    if (gpu_workers == 0) {
        gpu_workers = 1;
    }
    // Keep the staging GPU batch size aligned with gpu_worker_loop.
    const uint32_t gpu_batch_eff = glm5_gpu_stage_batch_max;
    uint32_t pinned_slot_target = std::max(1u, pinned_workers * 2u);
    uint32_t gpu_slot_target = std::max(1u, gpu_workers * gpu_batch_eff);
    uint32_t slot_count = pinned_slot_target + gpu_slot_target;
    const uint32_t max_slot_target = glm5_staging_max_slot_target(engine);
    if (slot_count == 0) {
        slot_count = 1;
    }
    const uint64_t min_slot = glm5_staging_min_slot_bytes(engine);
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
    const uint32_t min_slots = glm5_staging_min_slot_count(engine);
    if (slot_count < min_slots && engine->staging_size / min_slots >= min_slot) {
        slot_count = min_slots;
        slot_size = engine->staging_size / slot_count;
    }
    if (slot_size == 0) {
        update_staging_slot_deficit(engine);
        return;
    }
    if (slot_count < pinned_slot_target + gpu_slot_target) {
        const uint64_t desired_total = (uint64_t)pinned_slot_target + (uint64_t)gpu_slot_target;
        pinned_slot_target = static_cast<uint32_t>(
            std::max<uint64_t>(1ull, ((uint64_t)slot_count * (uint64_t)pinned_slot_target) / desired_total));
        if (pinned_slot_target >= slot_count && slot_count > 1) {
            pinned_slot_target = slot_count - 1;
        }
    }
    engine->staging_slot_size = slot_size;
    engine->staging_pinned_slot_count.store(std::min(pinned_slot_target, slot_count), std::memory_order_release);
    engine->staging_slots.reserve(slot_count);
    for (uint32_t i = 0; i < slot_count; ++i) {
        glm5_staging_slot slot;
        slot.buf = engine->staging + slot_size * i;
        slot.size = i + 1 == slot_count ? engine->staging_size - slot_size * i : slot_size;
        slot.pinned = engine->staging_pinned;
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
    // Keep the warning threshold aligned with gpu_worker_loop.
    const uint32_t gpu_batch_eff = glm5_gpu_stage_batch_max;
    const uint32_t warning_slot_target = (disk_workers + pinned_workers) * 2u + gpu_workers * gpu_batch_eff;
    if (slot_count < warning_slot_target) {
        fprintf(stderr, "[WARNING] Insufficient staging slots: %u < %u. Expect pipeline stalls.\n",
            slot_count, warning_slot_target);
    }
}
