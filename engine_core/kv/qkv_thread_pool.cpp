#include "qkv_thread_pool.h"

#include <climits>

QkvThreadPool::QkvThreadPool(int num_threads) {
    // BUGFIX 419: num_threads 유효성 체크
    if (num_threads <= 0 || num_threads > 1024) {
        num_threads = 1;
    }
    for (int i = 0; i < num_threads; ++i) {
        workers.emplace_back([this]() {
            while (true) {
                int my_task = -1;
                int local_total = 0;
                std::function<void(int)> my_fn;
                {
                    std::unique_lock<std::mutex> lock(mtx);
                    cv.wait(lock, [this]() {
                        return stop ||
                            current_task.load(std::memory_order_relaxed) <
                                total_tasks.load(std::memory_order_acquire);
                    });
                    local_total = total_tasks.load(std::memory_order_acquire);
                    if (stop &&
                        current_task.load(std::memory_order_relaxed) >= local_total) {
                        return;
                    }
                    my_fn = task;
                }
                for (;;) {
                    my_task = current_task.fetch_add(1, std::memory_order_relaxed);
                    if (my_task >= local_total) {
                        break;
                    }
                    my_fn(my_task);
                    const int done = completed_tasks.fetch_add(1, std::memory_order_release) + 1;
                    if (done == local_total) {
                        done_cv.notify_one();
                    }
                }
            }
        });
    }
}

QkvThreadPool::~QkvThreadPool() {
    {
        std::unique_lock<std::mutex> lock(mtx);
        stop = true;
    }
    cv.notify_all();
    for (auto& th : workers) {
        th.join();
    }
}

void QkvThreadPool::run(int num_tasks, std::function<void(int)> fn) {
    // BUGFIX 420: num_tasks 유효성 체크
    if (num_tasks <= 0 || num_tasks > INT_MAX / 2) return;
    // BUGFIX 909: Lower serialization threshold and use atomic for completed_tasks ★★ PERFORMANCE
    // Problem 1: Threshold 1024 too high → 512-token context (65K ops) runs single-threaded
    // Problem 2: completed_tasks under mutex → lock contention on every task completion
    // Solution 1: Lower threshold to 256 (parallelizes 256-1023 range)
    // Solution 2: Use atomic for completed_tasks (already atomic in header, just use it)
    // Impact: Better parallelization for medium workloads, reduced lock contention
    // Trade-off: n < 64 should stay serial to avoid thread wakeup overhead
    if (num_tasks < 256) {
        // Serial execution for small workloads (< 256 tasks)
        for (int i = 0; i < num_tasks; ++i) {
            fn(i);
        }
        return;
    }
    {
        std::unique_lock<std::mutex> lock(mtx);
        // Bug 2 Fix: Check stop flag to prevent deadlock if pool is shutting down.
        // If destructor already set stop=true and workers exited, notify_all has
        // no listeners and done_cv.wait would hang forever.
        if (stop) return;
        task = fn;
        total_tasks.store(num_tasks, std::memory_order_release);
        current_task.store(0, std::memory_order_relaxed);
        completed_tasks.store(0, std::memory_order_release);
    }
    cv.notify_all();
    {
        std::unique_lock<std::mutex> lock(mtx);
        // Bug 2 Fix: Wait ONLY for task completion, not stop flag.
        // Workers complete current tasks before exiting, so completed_tasks will
        // reach total_tasks even during shutdown. Checking stop here causes
        // premature return with incomplete attention scores.
        done_cv.wait(lock, [this]() {
            return completed_tasks.load(std::memory_order_acquire) >=
                total_tasks.load(std::memory_order_acquire);
        });
    }
}
