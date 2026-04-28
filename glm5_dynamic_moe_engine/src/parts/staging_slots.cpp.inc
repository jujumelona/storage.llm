static void reset_staging_slots(glm5_pc_engine* engine) {
    if (!engine) {
        return;
    }
    std::lock_guard<std::mutex> lock(engine->staging_slot_mutex);
    engine->staging_slots.clear();
    engine->staging_slot_size = 0;
    engine->staging_pinned_slot_count = 0;
    engine->io_atomic.staging_slot_deficit.store(0);
    engine->staging_slot_cv.notify_all();
}

static uint64_t glm5_staging_min_slot_bytes(const glm5_pc_engine* engine) {
    // Size slots to actual tensor data + small padding.
    // Old 64MB floor wasted ~44MB per slot for NVFP4 (~20MB triplets).
    const uint64_t base = engine ? engine->max_tensor_stream_bytes : 0ull;
    const uint64_t padding = 4ull * 1024ull * 1024ull;
    return base > 0 ? base + padding : padding;
}

static uint32_t glm5_staging_min_slot_count(const glm5_pc_engine* engine) {
    if (!engine || !engine->io_config.enable_pinned_host) {
        return 0;
    }
    uint32_t pinned_workers = engine->io_config.pinned_worker_threads;
    if (pinned_workers == 0) {
        pinned_workers = 1;
    }
    const uint32_t gpu_workers = engine->io_config.gpu_worker_threads;
    constexpr uint32_t gpu_batch_max = 8;
    const uint32_t pinned_slot_target = std::max(1u, pinned_workers * 2u);
    const uint32_t gpu_slot_target = std::max(1u, gpu_workers * gpu_batch_max);
    return std::max(pinned_slot_target + gpu_slot_target, 4u);
}

static uint64_t glm5_recommended_staging_bytes(const glm5_pc_engine* engine) {
    const uint32_t min_slots = glm5_staging_min_slot_count(engine);
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
    if (deficit) {
        engine->io_atomic.staging_wraps.fetch_add(1);
    }
}

static void configure_staging_slots(glm5_pc_engine* engine) {
    reset_staging_slots(engine);
    if (!engine || !engine->staging || engine->staging_size == 0) {
        update_staging_slot_deficit(engine);
        return;
    }

    uint32_t disk_workers = engine->io_config.disk_worker_threads;
    uint32_t pinned_workers = engine->io_config.pinned_worker_threads;
    if (disk_workers == 0) {
        disk_workers = engine->io_config.io_worker_threads ? engine->io_config.io_worker_threads : 1u;
    }
    if (pinned_workers == 0) {
        pinned_workers = 1;
    }

    // Fix 66: Include gpu_workers — they hold slots until stream sync (Fix 63)
    const uint32_t gpu_workers = engine->io_config.gpu_worker_threads;
    constexpr uint32_t gpu_batch_max = 8;
    uint32_t pinned_slot_target = std::max(1u, pinned_workers * 2u);
    uint32_t gpu_slot_target = std::max(1u, gpu_workers * gpu_batch_max);
    uint32_t slot_count = pinned_slot_target + gpu_slot_target;
    if (slot_count == 0) {
        slot_count = 1;
    }
    if (engine->staging_size < slot_count) {
        slot_count = static_cast<uint32_t>(engine->staging_size);
    }
    if (slot_count == 0) {
        update_staging_slot_deficit(engine);
        return;
    }

    uint64_t slot_size = engine->staging_size / slot_count;
    const uint64_t min_slot = glm5_staging_min_slot_bytes(engine);
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
    engine->staging_pinned_slot_count = pinned_slot_target;
    engine->staging_slots.reserve(slot_count);
    for (uint32_t i = 0; i < slot_count; ++i) {
        glm5_staging_slot slot;
        slot.buf = engine->staging + slot_size * i;
        slot.size = i + 1 == slot_count ? engine->staging_size - slot_size * i : slot_size;
        slot.in_use.store(false);
        engine->staging_slots.emplace_back(std::move(slot));
    }
    update_staging_slot_deficit(engine);
}
