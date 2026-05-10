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
                std::function<void(int)> my_fn;
                int local_total = 0;
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
                // BUGFIX 940: Lock-free task loop ★★★ PERFORMANCE
                // Workers self-assign tasks via atomic fetch_add.
                // No mutex held during task execution → zero lock contention.
                for (;;) {
                    const int my_task = current_task.fetch_add(1, std::memory_order_relaxed);
                    if (my_task >= local_total) {
                        break;
                    }
                    my_fn(my_task);
                    const int done = completed_tasks.fetch_add(1, std::memory_order_acq_rel) + 1;
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
    // BUGFIX 909/940: Dynamic serialization threshold ★★ PERFORMANCE
    // Problem: Fixed threshold 1024 → medium workloads (256-1023) run single-threaded
    // Solution: Lower to 64 for small SIMD-width workloads, stay serial only when
    //   thread wakeup overhead (~5μs per worker) exceeds task compute time.
    // Trade-off: n < 64 stays serial to avoid wakeup cost > compute cost.
    const int serial_threshold = 64;
    if (num_tasks < serial_threshold) {
        for (int i = 0; i < num_tasks; ++i) {
            fn(i);
        }
        return;
    }
    {
        std::unique_lock<std::mutex> lock(mtx);
        // Bug 2 Fix: Check stop flag to prevent deadlock if pool is shutting down.
        if (stop) return;
        task = fn;
        total_tasks.store(num_tasks, std::memory_order_release);
        current_task.store(0, std::memory_order_relaxed);
        completed_tasks.store(0, std::memory_order_release);
    }
    cv.notify_all();
    {
        std::unique_lock<std::mutex> lock(mtx);
        done_cv.wait(lock, [this]() {
            return completed_tasks.load(std::memory_order_acquire) >=
                total_tasks.load(std::memory_order_acquire);
        });
    }
}
