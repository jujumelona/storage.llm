static uint32_t engine_hw_threads() {
    return moe_cpu_thread_count_hint();
}

static uint32_t disk_worker_cap() {
    const uint32_t hw = engine_hw_threads();
    return hw < 16u ? hw : 16u;
}

static uint32_t pinned_worker_cap(const moe_pc_engine* engine) {
    if (!engine || engine_bypass_staging_pipeline(engine) ||
        !engine->io_config.enable_pinned_host ||
        !engine->backend_caps.supports_pinned_host) {
        return 0;
    }
    // BUGFIX 332: Check copy_streams > 0 before multiplication ★
    // Problem: copy_streams can be 0, causing cap to be 0
    // Solution: Use max(1, copy_streams) to ensure cap >= 1
    uint32_t cap = engine && engine->backend_caps.copy_streams ?
        engine->backend_caps.copy_streams * 2u : 2u;
    if (cap == 0) {
        cap = 1;
    }
    return cap > 16u ? 16u : cap;
}

static uint32_t gpu_worker_cap(const moe_pc_engine* engine) {
    if (!engine || engine_bypass_staging_pipeline(engine) ||
        !engine_has_device_upload_workers(engine)) {
        return 0;
    }
    // BUGFIX 333: Ensure minimum of 1 worker even if copy_streams is 0 ★
    // Problem: copy_streams can be 0, returning 0 workers
    // Solution: Return max(1, copy_streams)
    return engine->backend_caps.copy_streams ? engine->backend_caps.copy_streams : 1u;
}

static uint32_t clamp_to_cap(uint32_t value, uint32_t cap) {
    if (value == 0) {
        value = 1;
    }
    return value > cap ? cap : value;
}
