#include "qkv_thread_pool.h"

QkvThreadPool::QkvThreadPool(int num_threads) {
    for (int i = 0; i < num_threads; ++i) {
        workers.emplace_back([this]() {
            while (true) {
                int my_task = -1;
                int local_total = 0;
                std::function<void(int)> my_fn;
                {
                    std::unique_lock<std::mutex> lock(mtx);
                    cv.wait(lock, [this]() { return stop || current_task < total_tasks; });
                    if (stop && current_task >= total_tasks) return;
                    my_task = current_task++;
                    // Bug 3: Copy total_tasks under mutex to avoid data race
                    local_total = total_tasks;
                    my_fn = task;
                }
                // Bug 3: Use local_total instead of total_tasks (no mutex here)
                if (my_task < local_total) {
                    my_fn(my_task);
                    {
                        std::unique_lock<std::mutex> lock(mtx);
                        completed_tasks++;
                        if (completed_tasks == total_tasks) {
                            done_cv.notify_one();
                        }
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
    if (num_tasks == 0) return;
    {
        std::unique_lock<std::mutex> lock(mtx);
        // Bug 2 Fix: Check stop flag to prevent deadlock if pool is shutting down.
        // If destructor already set stop=true and workers exited, notify_all has
        // no listeners and done_cv.wait would hang forever.
        if (stop) return;
        task = fn;
        total_tasks = num_tasks;
        current_task = 0;
        completed_tasks = 0;
    }
    cv.notify_all();
    {
        std::unique_lock<std::mutex> lock(mtx);
        // Bug 2 Fix: Wait ONLY for task completion, not stop flag.
        // Workers complete current tasks before exiting, so completed_tasks will
        // reach total_tasks even during shutdown. Checking stop here causes
        // premature return with incomplete attention scores.
        done_cv.wait(lock, [this]() { return completed_tasks == total_tasks; });
    }
}
