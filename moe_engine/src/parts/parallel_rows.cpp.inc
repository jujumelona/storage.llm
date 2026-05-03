#if defined(__APPLE__) && defined(__MACH__)
#include <pthread/qos.h>
#include <sys/sysctl.h>
#endif

static uint32_t row_thread_override() {
    const uint32_t rows = moe_parse_env_u32("STORAGELLM_CPU_ROW_THREADS");
    return rows > 0 ? std::min<uint32_t>(rows, 256u) : 0u;
}

#if defined(__APPLE__) && defined(__MACH__)
static uint32_t macos_performance_core_count() {
    uint32_t value = 0;
    size_t size = sizeof(value);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &value, &size, nullptr, 0) == 0 && value > 0) {
        return value;
    }
    return 0;
}
#endif

static uint32_t row_hw_threads() {
    static const uint32_t hw = []() -> uint32_t {
#if defined(__APPLE__) && defined(__MACH__)
        const uint32_t override_threads = row_thread_override();
        if (override_threads > 0) {
            return override_threads;
        }
        const uint32_t pcores = macos_performance_core_count();
        if (pcores > 0) {
            return pcores;
        }
#endif
        return moe_cpu_thread_count_hint();
    }();
    return hw;
}

static inline void moe_cpu_enable_fast_fp_mode() {
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
#if defined(_MM_DENORMALS_ZERO_ON)
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif
#endif
}

static inline void moe_cpu_apply_row_worker_qos() {
#if defined(__APPLE__) && defined(__MACH__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
}

static uint32_t moe_cpu_row_min_block() {
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    if (__builtin_cpu_supports("avx512f")) return 64u;
    if (__builtin_cpu_supports("avx2")) return 32u;
#endif
#if defined(__AVX512F__)
    return 64u;
#elif defined(__AVX2__)
    return 32u;
#else
    return 16u;
#endif
}

// BUGFIX 710: Minimum block size to prevent excessive atomic fetch_add contention ★★
// Problem: 256 rows / (16 workers * 4) = 4 rows per block → 64 fetch_add ops per dispatch.
//   False sharing + cache ping-pong on next_row atomic degrades throughput.
// Solution: min_block = 16 ensures at most (rows / 16) atomic ops regardless of worker count.
//   For 256 rows → 16 ops instead of 64. 16 rows ≈ 4 cache lines of work per steal.
static uint32_t auto_row_block(uint32_t rows, uint32_t workers) {
    if (workers == 0) {
        workers = 1;
    }
    const uint32_t block = rows / (workers * 4u);
    const uint32_t min_block = moe_cpu_row_min_block();
    return std::max(block ? block : 1u, min_block);
}

// ============================================================================
// Persistent CPU Row Thread Pool (BUGFIX 704) ★★★
// ============================================================================
// Problem: moe_parallel_rows created workers-1 std::threads per call.
//   MoE inference calls this dozens of times per token (gate/up/down × N_experts).
//   Thread creation overhead (50-100µs/thread) adds several ms per token.
// Solution: Static persistent pool with condition_variable dispatch.
//   Threads sleep on CV, wake on dispatch, execute work, signal completion.
//   Nested calls (cpu_row_active > 0) still fall back to single-threaded.
// ============================================================================

struct moe_cpu_row_pool_t {
    struct task_t {
        std::function<void()> fn;
    };

    std::vector<std::thread> threads;
    std::mutex mtx;
    std::condition_variable dispatch_cv;
    std::condition_variable done_cv;

    std::function<void()> work_fn;   // shared work function for all workers
    std::atomic<uint32_t> pending{0};
    std::atomic<uint32_t> running{0};
    std::atomic<bool> shutdown{false};
    uint64_t generation = 0;
    uint32_t active_count = 0;
    bool initialized{false};

    void init(uint32_t count) {
        if (count == 0) return;
        std::lock_guard<std::mutex> lk(mtx);
        initialized = true;
        const uint32_t start = static_cast<uint32_t>(threads.size());
        if (start >= count) return;
        threads.reserve(count);
        for (uint32_t i = start; i < count; ++i) {
            const uint64_t initial_generation = generation;
            threads.emplace_back([this, i, initial_generation]() {
                moe_cpu_enable_fast_fp_mode();
                moe_cpu_apply_row_worker_qos();
                uint64_t seen_generation = initial_generation;
                for (;;) {
                    std::function<void()> fn;
                    {
                        std::unique_lock<std::mutex> lk(mtx);
                        dispatch_cv.wait(lk, [this, &seen_generation]() {
                            return shutdown.load(std::memory_order_relaxed) ||
                                   generation != seen_generation;
                        });
                        if (shutdown.load(std::memory_order_relaxed)) {
                            return;
                        }
                        seen_generation = generation;
                        if (i >= active_count) {
                            continue;
                        }
                        fn = work_fn;
                        if (!fn) {
                            continue;
                        }
                        running.fetch_add(1, std::memory_order_acq_rel);
                    }
                    if (fn) fn();
                    running.fetch_sub(1, std::memory_order_acq_rel);
                    if (pending.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                        done_cv.notify_all();
                    }
                }
            });
        }
    }

    void dispatch(std::function<void()> fn, uint32_t count) {
        {
            std::lock_guard<std::mutex> lk(mtx);
            work_fn = std::move(fn);
            active_count = std::min<uint32_t>(count, static_cast<uint32_t>(threads.size()));
            ++generation;
            running.store(0, std::memory_order_release);
            pending.store(active_count, std::memory_order_release);
        }
        dispatch_cv.notify_all();
    }

    void wait() {
        std::unique_lock<std::mutex> lk(mtx);
        for (;;) {
            if (pending.load(std::memory_order_acquire) == 0) {
                return;
            }
            if (done_cv.wait_for(lk, std::chrono::seconds(2), [this]() {
                    return pending.load(std::memory_order_acquire) == 0;
                })) {
                return;
            }
            const uint32_t stuck_pending = pending.load(std::memory_order_acquire);
            const uint32_t stuck_running = running.load(std::memory_order_acquire);
            if (stuck_pending > 0 && stuck_running == 0) {
                std::fprintf(
                    stderr,
                    "[storagellm cpu_row_pool] recovered stale dispatch generation=%llu pending=%u active=%u\n",
                    (unsigned long long)generation,
                    stuck_pending,
                    active_count);
                active_count = 0;
                work_fn = nullptr;
                pending.store(0, std::memory_order_release);
                done_cv.notify_all();
                return;
            }
        }
    }

    ~moe_cpu_row_pool_t() {
        shutdown.store(true, std::memory_order_release);
        dispatch_cv.notify_all();
        for (auto& t : threads) {
            if (t.joinable()) t.join();
        }
    }
};

static moe_cpu_row_pool_t& moe_get_cpu_row_pool() {
    static moe_cpu_row_pool_t pool;
    return pool;
}

template <typename RowFn>
static int moe_parallel_rows(moe_pc_engine* engine, uint32_t row_count, uint32_t block_rows, RowFn fn) {
    if (row_count == 0) {
        return 1;
    }
    moe_cpu_enable_fast_fp_mode();
    const uint64_t trace_start_us = moe_profile_now_us();
    const char* trace_label = moe_profile_parallel_label();
    uint32_t workers = row_hw_threads();
    if (block_rows == 0) {
        block_rows = auto_row_block(row_count, workers);
    }
    const uint32_t max_blocks = (row_count + block_rows - 1u) / block_rows;
    if (workers > max_blocks) {
        workers = max_blocks;
    }
    if (!engine || workers <= 1) {
        int ok = 1;
        for (uint32_t row = 0; row < row_count; ++row) {
            if (!fn(row)) {
                ok = 0;
                break;
            }
        }
        moe_profile_log(
            "parallel_rows",
            "label=%s rows=%u block=%u workers=%u mode=serial ok=%d ms=%.3f",
            trace_label, row_count, block_rows, workers, ok, moe_profile_elapsed_ms(trace_start_us));
        return ok;
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
        int ok = 1;
        for (uint32_t row = 0; row < row_count; ++row) {
            if (!fn(row)) {
                ok = 0;
                break;
            }
        }
        // BUGFIX 614: Check cpu_row_active > 0 before fetch_sub to prevent underflow ★
        int current = engine->cpu_row_active.load(std::memory_order_acquire);
        while (current > 0) {
            if (engine->cpu_row_active.compare_exchange_weak(
                    current, current - 1,
                    std::memory_order_release,
                    std::memory_order_acquire)) {
                break;
            }
        }
        moe_profile_log(
            "parallel_rows",
            "label=%s rows=%u block=%u workers=%u mode=nested_serial active_before=%d ok=%d ms=%.3f",
            trace_label, row_count, block_rows, workers, prev_active, ok, moe_profile_elapsed_ms(trace_start_us));
        return ok;
    }

    // BUGFIX 704: Use persistent thread pool instead of per-call fork/join ★★★
    // Old: std::thread creation + join per call → 50-100µs/thread overhead
    // New: Static pool with CV dispatch → ~1µs wake latency
    std::atomic<uint32_t> next_row{0};
    std::atomic<int> ok{1};
    auto run_rows = [&]() {
        moe_cpu_enable_fast_fp_mode();
        moe_cpu_apply_row_worker_qos();
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

    auto& pool = moe_get_cpu_row_pool();
    pool.init(workers > 1 ? workers - 1 : 0);

    const uint32_t pool_workers = std::min(workers - 1u,
        static_cast<uint32_t>(pool.threads.size()));

    if (pool_workers > 0 && pool.initialized) {
        // Dispatch to persistent pool
        pool.dispatch(run_rows, pool_workers);
        run_rows();  // Main thread participates
        pool.wait();
    } else {
        // Fallback to fork/join if pool not available
        std::vector<std::thread> local_workers;
        local_workers.reserve(workers > 1 ? workers - 1u : 0u);
        try {
            for (uint32_t i = 1; i < workers; ++i) {
                local_workers.emplace_back(run_rows);
            }
            run_rows();
        } catch (...) {
            ok.store(0, std::memory_order_release);
            for (std::thread& worker : local_workers) {
                if (worker.joinable()) worker.join();
            }
            engine->cpu_row_active.fetch_sub(1, std::memory_order_release);
            return 0;
        }
        for (std::thread& worker : local_workers) {
            if (worker.joinable()) worker.join();
        }
    }

    // BUGFIX 614: Check cpu_row_active > 0 before fetch_sub to prevent underflow ★
    int current = engine->cpu_row_active.load(std::memory_order_acquire);
    while (current > 0) {
        if (engine->cpu_row_active.compare_exchange_weak(
                current, current - 1,
                std::memory_order_release,
                std::memory_order_acquire)) {
            break;
        }
    }
    const int final_ok = ok.load(std::memory_order_acquire);
    moe_profile_log(
        "parallel_rows",
        "label=%s rows=%u block=%u workers=%u pool_workers=%u mode=pool ok=%d ms=%.3f",
        trace_label, row_count, block_rows, workers, pool_workers, final_ok,
        moe_profile_elapsed_ms(trace_start_us));
    return final_ok;
}
