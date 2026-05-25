#pragma once

#include "cpu_assignment_kernel.h"
#include <algorithm>
#include <cstdlib>
#include <atomic>
#include <climits>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#if defined(STORAGELLM_HAS_OPENMP)
#include <omp.h>
#endif

namespace storagellm::cpu_moe {

inline uint32_t env_u32(const char* name, uint32_t fallback, uint32_t lo, uint32_t hi) {
    const char* v = name ? std::getenv(name) : nullptr;
    if (!v || !*v) return fallback;
    char* end = nullptr;
    unsigned long parsed = std::strtoul(v, &end, 10);
    if (end == v) return fallback;
    if (parsed < lo) parsed = lo;
    if (parsed > hi) parsed = hi;
    return static_cast<uint32_t>(parsed);
}

inline bool env_truthy(const char* name) {
    const char* v = name ? std::getenv(name) : nullptr;
    if (!v || !*v) return false;
    return !(std::strcmp(v, "0") == 0 || std::strcmp(v, "false") == 0 ||
             std::strcmp(v, "FALSE") == 0 || std::strcmp(v, "off") == 0 ||
             std::strcmp(v, "OFF") == 0);
}

inline uint32_t effective_cpu_threads() {
    const uint32_t forced = env_u32("STORAGELLM_CPU_THREADS", 0u, 0u, 256u);
    if (forced > 0) {
        return forced;
    }
    uint32_t threads = UINT32_MAX;
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    threads = std::min<uint32_t>(threads, static_cast<uint32_t>(hw));
#if defined(__linux__)
#if defined(_SC_NPROCESSORS_ONLN)
    const long online = sysconf(_SC_NPROCESSORS_ONLN);
    if (online > 0) {
        threads = std::min<uint32_t>(threads, static_cast<uint32_t>(online));
    }
#endif
#if defined(CPU_SETSIZE)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        uint32_t affinity = 0;
        for (int i = 0; i < CPU_SETSIZE; ++i) {
            if (CPU_ISSET(i, &set)) {
                ++affinity;
            }
        }
        if (affinity > 0) {
            threads = std::min<uint32_t>(threads, affinity);
        }
    }
#endif
    std::ifstream cpu_max("/sys/fs/cgroup/cpu.max");
    std::string quota;
    std::string period;
    if (cpu_max >> quota >> period && quota != "max") {
        char* end_q = nullptr;
        char* end_p = nullptr;
        const unsigned long long q = std::strtoull(quota.c_str(), &end_q, 10);
        const unsigned long long p = std::strtoull(period.c_str(), &end_p, 10);
        if (end_q && *end_q == '\0' && end_p && *end_p == '\0' && q > 0 && p > 0) {
            threads = std::min<uint32_t>(
                threads,
                static_cast<uint32_t>(std::max<unsigned long long>(1ull, (q + p - 1ull) / p)));
        }
    }
#elif defined(_WIN32)
    const DWORD active = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    if (active > 0) {
        threads = std::min<uint32_t>(threads, static_cast<uint32_t>(active));
    }
#endif
    if (threads == 0 || threads == UINT32_MAX) {
        threads = 1;
    }
    return std::min<uint32_t>(threads, 256u);
}

inline void maybe_pin_current_worker(uint32_t worker_index) {
    if (!env_truthy("STORAGELLM_CPU_AFFINITY") &&
        !env_truthy("STORAGELLM_CPU_PIN_WORKERS")) {
        return;
    }
    const unsigned hw = std::max<uint32_t>(1u, effective_cpu_threads());
    const uint32_t core = hw ? (worker_index % hw) : 0u;
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    (void)pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#elif defined(_WIN32)
    if (core < 64u) {
        const DWORD_PTR mask = (DWORD_PTR(1) << core);
        (void)SetThreadAffinityMask(GetCurrentThread(), mask);
    }
#else
    (void)core;
#endif
}

inline uint32_t choose_thread_count(uint32_t assignments, uint64_t work_per_assignment) {
    const unsigned hw = std::max<uint32_t>(1u, effective_cpu_threads());
    const uint32_t forced = env_u32("STORAGELLM_CPU_MOE_THREADS", 0u, 0u, hw);
    if (forced > 0) {
        return std::max<uint32_t>(1, std::min<uint32_t>(assignments ? assignments : 1u, forced));
    }
    const bool max_mode = env_truthy("STORAGELLM_CPU_STORAGE_MAX") ||
                          env_truthy("STORAGELLM_CPU_PIPELINE") ||
                          env_truthy("STORAGELLM_CPU_ONLY_MAX") ||
                          env_truthy("STORAGELLM_MAX_PARALLELISM");
    const uint64_t min_work = max_mode ? 32768u : 131072u;
    if (!max_mode && (assignments < 4 || work_per_assignment < 32768u)) {
        return 1;
    }
    const uint64_t total_work = static_cast<uint64_t>(assignments) * work_per_assignment;
    const uint32_t by_work = static_cast<uint32_t>(std::max<uint64_t>(1, total_work / min_work));
    const uint32_t by_assignment = std::max<uint32_t>(1, assignments);
    return std::max<uint32_t>(1, std::min<uint32_t>({by_assignment, hw, by_work}));
}


#if !defined(STORAGELLM_HAS_OPENMP)
class PersistentAssignmentPool {
public:
    void ensure(uint32_t worker_count) {
        if (worker_count == 0) return;
        std::lock_guard<std::mutex> lk(mtx_);
        const uint32_t start = static_cast<uint32_t>(threads_.size());
        if (start >= worker_count) return;
        threads_.reserve(worker_count);
        for (uint32_t i = start; i < worker_count; ++i) {
            threads_.emplace_back([this, i]() {
                maybe_pin_current_worker(i);
                worker_loop();
            });
        }
    }

    void run(uint32_t rows, uint32_t worker_count, const std::function<void(uint32_t)>& row_fn) {
        if (rows == 0 || worker_count == 0) return;
        ensure(worker_count);
        const uint32_t active = std::min<uint32_t>(worker_count, static_cast<uint32_t>(threads_.size()));
        if (active == 0) {
            for (uint32_t r = 0; r < rows; ++r) row_fn(r);
            return;
        }
        {
            std::lock_guard<std::mutex> lk(mtx_);
            row_fn_ = row_fn;
            row_count_ = rows;
            next_row_.store(0, std::memory_order_release);
            pending_.store(active, std::memory_order_release);
            active_workers_ = active;
            ++generation_;
        }
        cv_.notify_all();
        // Main thread participates; worker threads cover the rest.
        steal_rows(row_fn, rows);
        wait_done();
    }

    ~PersistentAssignmentPool() {
        shutdown_.store(true, std::memory_order_release);
        cv_.notify_all();
        for (auto& t : threads_) {
            if (t.joinable()) t.join();
        }
    }

private:
    void steal_rows(const std::function<void(uint32_t)>& fn, uint32_t rows) {
        for (;;) {
            const uint32_t r = next_row_.fetch_add(1, std::memory_order_relaxed);
            if (r >= rows) break;
            fn(r);
        }
    }

    void worker_loop() {
        uint64_t seen = 0;
        for (;;) {
            std::function<void(uint32_t)> fn;
            uint32_t rows = 0;
            {
                std::unique_lock<std::mutex> lk(mtx_);
                cv_.wait(lk, [&]() {
                    return shutdown_.load(std::memory_order_relaxed) || generation_ != seen;
                });
                if (shutdown_.load(std::memory_order_relaxed)) return;
                seen = generation_;
                fn = row_fn_;
                rows = row_count_;
            }
            if (fn) steal_rows(fn, rows);
            if (pending_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                done_cv_.notify_all();
            }
        }
    }

    void wait_done() {
        std::unique_lock<std::mutex> lk(mtx_);
        done_cv_.wait(lk, [&]() {
            return pending_.load(std::memory_order_acquire) == 0;
        });
    }

    std::vector<std::thread> threads_;
    std::mutex mtx_;
    std::condition_variable cv_;
    std::condition_variable done_cv_;
    std::function<void(uint32_t)> row_fn_;
    std::atomic<uint32_t> next_row_{0};
    std::atomic<uint32_t> pending_{0};
    std::atomic<bool> shutdown_{false};
    uint64_t generation_{0};
    uint32_t row_count_{0};
    uint32_t active_workers_{0};
};

inline PersistentAssignmentPool& assignment_pool() {
    static PersistentAssignmentPool pool;
    return pool;
}
#endif


inline bool task_has_duplicate_output_tokens(const ValidatedTask& ctx) {
    const auto& task = *ctx.task;
    const uint32_t count = task.assignment_count;
    if (count < 2 || !task.d_token_indices) return false;
    // Top-k MoE usually has multiple assignments per token.  If we parallelize
    // assignments directly, those rows race on accum[token] += ... .  Detect the
    // common duplicate-token case and route it through private row outputs plus a
    // deterministic reduction.
    std::unordered_set<uint32_t> seen;
    seen.reserve(count * 2u);
    for (uint32_t r = 0; r < count; ++r) {
        const uint32_t token = task.d_token_indices[task.assignment_offset + r];
        if (!seen.insert(token).second) return true;
    }
    return false;
}

inline void reduce_private_assignment_outputs(const ValidatedTask& ctx, const std::vector<float>& outputs, uint32_t threads) {
    const auto& task = *ctx.task;
    const uint32_t count = task.assignment_count;
    const uint32_t H = task.hidden_size;
    if (outputs.size() < static_cast<size_t>(count) * H) return;

    // Duplicate-token accumulation is correctness-sensitive.  Parallelizing over
    // assignment rows would race on accum[token], so group rows by token and then
    // reduce independent token groups in parallel.  Within a token group, rows are
    // reduced in ascending assignment order for deterministic FP32 accumulation.
    std::unordered_map<uint32_t, std::vector<uint32_t>> by_token;
    by_token.reserve(count * 2u);
    for (uint32_t row = 0; row < count; ++row) {
        const uint32_t global_row = task.assignment_offset + row;
        const uint32_t token = task.d_token_indices[global_row];
        by_token[token].push_back(row);
    }

    std::vector<std::pair<uint32_t, std::vector<uint32_t>>> groups;
    groups.reserve(by_token.size());
    for (auto& kv : by_token) {
        std::sort(kv.second.begin(), kv.second.end());
        groups.emplace_back(kv.first, std::move(kv.second));
    }

    auto* accum_base = static_cast<float*>(task.d_accum);
    auto reduce_group = [&](uint32_t group_index) {
        const auto& group = groups[group_index];
        const uint32_t token = group.first;
        float* accum = accum_base + static_cast<uint64_t>(token) * task.accum_stride;
        for (uint32_t row : group.second) {
            const float* src = outputs.data() + static_cast<uint64_t>(row) * H;
            for (uint32_t h = 0; h < H; ++h) {
                accum[h] += src[h];
            }
        }
    };

#if defined(STORAGELLM_HAS_OPENMP)
    #pragma omp parallel for schedule(static) num_threads(static_cast<int>(std::max<uint32_t>(1u, threads)))
    for (int g = 0; g < static_cast<int>(groups.size()); ++g) {
        maybe_pin_current_worker(static_cast<uint32_t>(omp_get_thread_num()));
        reduce_group(static_cast<uint32_t>(g));
    }
#else
    if (groups.size() <= 1u || threads <= 1u) {
        for (uint32_t g = 0; g < groups.size(); ++g) reduce_group(g);
    } else {
        assignment_pool().run(static_cast<uint32_t>(groups.size()), threads > 1u ? threads - 1u : 0u, reduce_group);
    }
#endif
}

inline void run_task_parallel_duplicate_safe(const ValidatedTask& ctx, uint32_t threads) {
    const auto& task = *ctx.task;
    const uint32_t count = task.assignment_count;
    const uint32_t H = task.hidden_size;
    if (count == 0 || H == 0) return;
    const uint64_t output_floats = static_cast<uint64_t>(count) * H;
    const uint64_t output_bytes = output_floats * sizeof(float);
    const uint64_t max_private_bytes = static_cast<uint64_t>(env_u32(
        "STORAGELLM_CPU_DUP_ACCUM_PRIVATE_MB", 256u, 1u, 65536u)) << 20;

    // If private reduction memory would be too large, fall back to serial direct
    // accumulation. Correctness first: a direct parallel += on duplicate tokens is
    // not allowed because it loses expert contributions nondeterministically.
    if (output_bytes > max_private_bytes) {
        std::vector<float> mid(task.intermediate_size);
        run_assignment_range(ctx, 0, count, mid);
        return;
    }

    std::vector<float> private_outputs(static_cast<size_t>(output_floats));

#if defined(STORAGELLM_HAS_OPENMP)
    #pragma omp parallel for schedule(static) num_threads(static_cast<int>(threads))
    for (int row = 0; row < static_cast<int>(count); ++row) {
        maybe_pin_current_worker(static_cast<uint32_t>(omp_get_thread_num()));
        std::vector<float> mid(task.intermediate_size);
        compute_assignment_output_f32(
            ctx,
            static_cast<uint32_t>(row),
            mid,
            private_outputs.data() + static_cast<uint64_t>(row) * H);
    }
#else
    auto run_one = [&](uint32_t row) {
        thread_local std::vector<float> mid;
        if (mid.size() < task.intermediate_size) {
            mid.resize(task.intermediate_size);
        }
        compute_assignment_output_f32(
            ctx,
            row,
            mid,
            private_outputs.data() + static_cast<uint64_t>(row) * H);
    };
    assignment_pool().run(count, threads > 1u ? threads - 1u : 0u, run_one);
#endif
    reduce_private_assignment_outputs(ctx, private_outputs, threads);
}

inline void run_task_parallel(const ValidatedTask& ctx) {
    const auto& task = *ctx.task;
    const uint32_t count = task.assignment_count;
    const uint64_t work_per_assignment =
        static_cast<uint64_t>(task.intermediate_size) * task.hidden_size * 2u +
        static_cast<uint64_t>(task.hidden_size) * task.intermediate_size;
    const uint32_t threads = choose_thread_count(count, work_per_assignment);

    if (threads <= 1) {
        std::vector<float> mid(task.intermediate_size);
        run_assignment_range(ctx, 0, count, mid);
        return;
    }

    if (task_has_duplicate_output_tokens(ctx)) {
        run_task_parallel_duplicate_safe(ctx, threads);
        return;
    }

#if defined(STORAGELLM_HAS_OPENMP)
    #pragma omp parallel num_threads(static_cast<int>(threads))
    {
        maybe_pin_current_worker(static_cast<uint32_t>(omp_get_thread_num()));
        std::vector<float> mid(task.intermediate_size);
        #pragma omp for schedule(static)
        for (int row = 0; row < static_cast<int>(count); ++row) {
            run_assignment_range(ctx, static_cast<uint32_t>(row), static_cast<uint32_t>(row + 1), mid);
        }
    }
#else
    // Persistent pool: avoid creating threads for every MoE gate/up/down call.
    // Each worker owns its temporary mid buffer for the duration of the task.
    assignment_pool().run(count, threads > 1u ? threads - 1u : 0u, [&](uint32_t row) {
        thread_local std::vector<float> mid;
        if (mid.size() < task.intermediate_size) {
            mid.resize(task.intermediate_size);
        }
        run_assignment_range(ctx, row, row + 1u, mid);
    });
#endif
}

} // namespace storagellm::cpu_moe
