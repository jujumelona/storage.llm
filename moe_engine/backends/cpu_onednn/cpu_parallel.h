#pragma once

#include "cpu_assignment_kernel.h"
#include <algorithm>
#include <thread>
#include <vector>

#if defined(STORAGELLM_HAS_OPENMP)
#include <omp.h>
#endif

namespace storagellm::cpu_moe {

inline uint32_t choose_thread_count(uint32_t assignments, uint64_t work_per_assignment) {
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    if (assignments < 4 || work_per_assignment < 32768u) {
        return 1;
    }
    const uint64_t total_work = static_cast<uint64_t>(assignments) * work_per_assignment;
    const uint32_t by_work = static_cast<uint32_t>(std::max<uint64_t>(1, total_work / 131072u));
    return std::max<uint32_t>(1, std::min<uint32_t>({assignments, hw, by_work}));
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

#if defined(STORAGELLM_HAS_OPENMP)
    #pragma omp parallel num_threads(static_cast<int>(threads))
    {
        std::vector<float> mid(task.intermediate_size);
        #pragma omp for schedule(static)
        for (int row = 0; row < static_cast<int>(count); ++row) {
            run_assignment_range(ctx, static_cast<uint32_t>(row), static_cast<uint32_t>(row + 1), mid);
        }
    }
#else
    std::vector<std::thread> pool;
    pool.reserve(threads);
    const uint32_t chunk = (count + threads - 1u) / threads;
    for (uint32_t t = 0; t < threads; ++t) {
        const uint32_t begin = std::min<uint32_t>(count, t * chunk);
        const uint32_t end = std::min<uint32_t>(count, begin + chunk);
        if (begin >= end) continue;
        pool.emplace_back([&, begin, end]() {
            std::vector<float> mid(task.intermediate_size);
            run_assignment_range(ctx, begin, end, mid);
        });
    }
    for (auto& th : pool) th.join();
#endif
}

} // namespace storagellm::cpu_moe
