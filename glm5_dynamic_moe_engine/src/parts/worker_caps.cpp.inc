static uint32_t engine_hw_threads() {
    const unsigned int hw = std::thread::hardware_concurrency();
    return hw ? static_cast<uint32_t>(hw) : 1u;
}

static uint32_t disk_worker_cap() {
    const uint32_t hw = engine_hw_threads();
    return hw < 16u ? hw : 16u;
}

static uint32_t pinned_worker_cap(const glm5_pc_engine* engine) {
    uint32_t cap = engine && engine->backend_caps.copy_streams ?
        engine->backend_caps.copy_streams * 2u : 2u;
    if (cap == 0) {
        cap = 1;
    }
    return cap > 8u ? 8u : cap;
}

static uint32_t gpu_worker_cap(const glm5_pc_engine* engine) {
    return engine && engine->backend_caps.copy_streams ? engine->backend_caps.copy_streams : 1u;
}

static uint32_t clamp_to_cap(uint32_t value, uint32_t cap) {
    if (value == 0) {
        value = 1;
    }
    return value > cap ? cap : value;
}
