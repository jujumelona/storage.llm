static uint32_t row_hw_threads() {
    const unsigned int hw = std::thread::hardware_concurrency();
    return hw ? static_cast<uint32_t>(hw) : 1u;
}

static uint32_t auto_row_block(uint32_t rows, uint32_t workers) {
    if (workers == 0) {
        workers = 1;
    }
    const uint32_t block = rows / (workers * 4u);
    return block ? block : 1u;
}

template <typename RowFn>
static int glm5_parallel_rows(glm5_pc_engine* engine, uint32_t row_count, uint32_t block_rows, RowFn fn) {
    if (row_count == 0) {
        return 1;
    }
    uint32_t workers = row_hw_threads();
    if (block_rows == 0) {
        block_rows = auto_row_block(row_count, workers);
    }
    const uint32_t max_blocks = (row_count + block_rows - 1u) / block_rows;
    if (workers > max_blocks) {
        workers = max_blocks;
    }
    if (!engine || workers <= 1) {
        for (uint32_t row = 0; row < row_count; ++row) {
            if (!fn(row)) {
                return 0;
            }
        }
        return 1;
    }

    // Bug #7 / Fix 5: Atomic counter for concurrent expert parallelism.
    // First entrant owns the worker pool; concurrent callers (including nested
    // calls from expert-level parallel_rows -> run_expert_triplet -> parallel_rows)
    // gracefully fall back to single-threaded execution. This is correct by design:
    // expert-level parallelism handles inter-expert concurrency, while row-level
    // parallelism within each expert is handled by the pool-owning thread.
    const int prev_active = engine->cpu_row_active.fetch_add(1, std::memory_order_acq_rel);
    if (prev_active > 0) {
        // Another thread owns the pool - run single-threaded for this expert
        for (uint32_t row = 0; row < row_count; ++row) {
            if (!fn(row)) {
                engine->cpu_row_active.fetch_sub(1, std::memory_order_release);
                return 0;
            }
        }
        engine->cpu_row_active.fetch_sub(1, std::memory_order_release);
        return 1;
    }

    std::unique_lock<std::mutex> lock(engine->cpu_row_mutex);
    engine->cpu_row_done_cv.wait(lock, [&] { return engine->cpu_row_done_generation == engine->cpu_row_generation; });
    lock.unlock();
    ensure_cpu_row_workers(engine, 0);
    lock.lock();
    engine->cpu_row_ctx = &fn;
    engine->cpu_row_fn = [](void* ctx, uint32_t row) { return (*static_cast<RowFn*>(ctx))(row); };
    engine->cpu_row_count = row_count;
    engine->cpu_block_rows = block_rows;
    engine->cpu_next_row.store(0);
    engine->cpu_rows_ok.store(1, std::memory_order_release);
    engine->cpu_active_workers.store(static_cast<uint32_t>(engine->cpu_row_workers.size()));
    const uint64_t generation = engine->cpu_row_generation + 1;
    engine->cpu_row_generation = generation;
    lock.unlock();
    engine->cpu_row_cv.notify_all();

    // Fix 27: Use acquire ordering to see worker failures promptly
    while (engine->cpu_rows_ok.load(std::memory_order_acquire)) {
        const uint32_t start = engine->cpu_next_row.fetch_add(block_rows);
        if (start >= row_count) {
            break;
        }
        uint32_t end = start + block_rows;
        if (end > row_count) {
            end = row_count;
        }
        for (uint32_t row = start; row < end && engine->cpu_rows_ok.load(std::memory_order_acquire); ++row) {
            if (!fn(row)) {
                engine->cpu_rows_ok.store(0, std::memory_order_release);
                break;
            }
        }
    }

    lock.lock();
    engine->cpu_row_done_cv.wait(lock, [&] { return engine->cpu_row_done_generation == generation; });
    engine->cpu_row_fn = nullptr;
    const int ok = engine->cpu_rows_ok.load(std::memory_order_acquire);
    engine->cpu_row_active.fetch_sub(1, std::memory_order_release);  // Bug #7: Decrement counter
    engine->cpu_row_done_cv.notify_all();
    lock.unlock();    return ok;
}
