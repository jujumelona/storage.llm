static uint32_t row_hw_threads() {
    static const uint32_t hw = []() -> uint32_t {
        const unsigned int n = std::thread::hardware_concurrency();
        return n ? static_cast<uint32_t>(n) : 1u;
    }();
    return hw;
}

static uint32_t auto_row_block(uint32_t rows, uint32_t workers) {
    if (workers == 0) {
        workers = 1;
    }
    const uint32_t block = rows / (workers * 4u);
    return block ? block : 1u;
}

template <typename RowFn>
static int moe_parallel_rows(moe_pc_engine* engine, uint32_t row_count, uint32_t block_rows, RowFn fn) {
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

    // Persistent row workers can be retired asynchronously by the runtime
    // orchestrator. A stale joinable worker counted in cpu_active_workers can
    // leave the caller blocked on cpu_row_done_cv forever. Use a per-dispatch
    // fork/join batch here: it keeps row-level CPU parallelism, has no retained
    // worker liveness contract, and nested calls still fall back above.
    std::atomic<uint32_t> next_row{0};
    std::atomic<int> ok{1};
    auto run_rows = [&]() {
        while (ok.load(std::memory_order_acquire)) {
            const uint32_t start = next_row.fetch_add(block_rows, std::memory_order_relaxed);
            if (start >= row_count) {
                break;
            }
            uint32_t end = start + block_rows;
            if (end > row_count) {
                end = row_count;
            }
            for (uint32_t row = start; row < end && ok.load(std::memory_order_acquire); ++row) {
                if (!fn(row)) {
                    ok.store(0, std::memory_order_release);
                    break;
                }
            }
        }
    };

    std::vector<std::thread> local_workers;
    local_workers.reserve(workers > 1 ? workers - 1u : 0u);
    try {
        for (uint32_t i = 1; i < workers; ++i) {
            local_workers.emplace_back(run_rows);
        }
        run_rows();
    } catch (...) {
        ok.store(0, std::memory_order_release);
    }
    for (std::thread& worker : local_workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    engine->cpu_row_active.fetch_sub(1, std::memory_order_release);  // Bug #7: Decrement counter
    return ok.load(std::memory_order_acquire);
}
