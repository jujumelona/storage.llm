static void cpu_row_worker_loop(glm5_pc_engine* engine, uint32_t worker_index) {
    uint64_t seen = 0;
    for (;;) {
        uint32_t rows = 0;
        uint32_t block = 1;
        {
            std::unique_lock<std::mutex> lock(engine->cpu_row_mutex);
            engine->cpu_row_cv.wait(lock, [&] {
                return engine->cpu_row_stop ||
                    worker_index >= engine->cpu_row_target_workers.load(std::memory_order_relaxed) ||
                    engine->cpu_row_generation != seen;
            });
            if (engine->cpu_row_stop ||
                worker_index >= engine->cpu_row_target_workers.load(std::memory_order_relaxed)) {
                return;
            }
            seen = engine->cpu_row_generation;
            rows = engine->cpu_row_count;
            block = engine->cpu_block_rows;
        }
        auto fn = engine->cpu_row_fn;
        void* ctx = engine->cpu_row_ctx;
        while (engine->cpu_rows_ok.load(std::memory_order_acquire)) {
            const uint32_t start = engine->cpu_next_row.fetch_add(block);
            if (start >= rows) {
                break;
            }
            uint32_t end = start + block;
            if (end > rows) {
                end = rows;
            }
            for (uint32_t row = start; row < end && engine->cpu_rows_ok.load(std::memory_order_acquire); ++row) {
                if (!fn(ctx, row)) {
                    engine->cpu_rows_ok.store(0, std::memory_order_release);
                    break;
                }
            }
        }
        if (engine->cpu_active_workers.fetch_sub(1) == 1) {
            std::lock_guard<std::mutex> lock(engine->cpu_row_mutex);
            engine->cpu_row_done_generation = seen;
            engine->cpu_row_done_cv.notify_one();
        }
    }
}

static void ensure_cpu_row_workers(glm5_pc_engine* engine, uint32_t /*ignored*/) {
    if (!engine) return;
    uint32_t target = engine->cpu_row_target_workers.load(std::memory_order_relaxed);
    if (target == 0) {
        const unsigned int hw_raw = std::thread::hardware_concurrency();
        const uint32_t hw = std::max(2u, hw_raw ? (uint32_t)hw_raw : 1u);
        const uint32_t io_threads = runtime_disk_workers(engine) +
             runtime_pinned_workers(engine) +
             runtime_gpu_workers(engine);
        target = hw > io_threads + 1u ? hw - io_threads - 1u : 1u;
        engine->cpu_row_target_workers.store(target, std::memory_order_relaxed);
    }

    if (engine->cpu_row_workers.size() == target) {
        return;
    }

    if (engine->cpu_row_workers.size() > target) {
        // Fix 6: Don't join on hot path — move excess threads to deferred_joins.
        // Workers self-terminate when they see worker_index >= target.
        engine->cpu_row_cv.notify_all();
        for (size_t i = target; i < engine->cpu_row_workers.size(); ++i) {
            engine->deferred_joins.emplace_back(std::move(engine->cpu_row_workers[i]));
        }
        engine->cpu_row_workers.resize(target);
    }

    while (engine->cpu_row_workers.size() < target) {
        const uint32_t index = static_cast<uint32_t>(engine->cpu_row_workers.size());
        engine->cpu_row_workers.emplace_back(cpu_row_worker_loop, engine, index);
    }
}

static void stop_cpu_row_workers(glm5_pc_engine* engine) {
    if (!engine) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(engine->cpu_row_mutex);
        engine->cpu_row_stop = true;
        engine->cpu_row_target_workers.store(0, std::memory_order_relaxed);
    }
    engine->cpu_row_cv.notify_all();
    for (std::thread& worker : engine->cpu_row_workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    engine->cpu_row_workers.clear();
    // Fix 6: Join deferred zombie threads from hot-path resize
    for (std::thread& t : engine->deferred_joins) {
        if (t.joinable()) {
            t.join();
        }
    }
    engine->deferred_joins.clear();
}
