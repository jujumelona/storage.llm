#pragma once

#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>

// QKV Thread Pool - Parallel execution for K scoring and V accumulation
class QkvThreadPool {
public:
    std::vector<std::thread> workers;
    std::condition_variable cv;
    std::condition_variable done_cv;
    std::mutex mtx;
    std::mutex run_mtx;
    std::function<void(int)> task;
    std::atomic<int> total_tasks{0};
    std::atomic<int> current_task{0};
    std::atomic<int> completed_tasks{0};
    bool stop = false;

    explicit QkvThreadPool(int num_threads);
    ~QkvThreadPool();

    void run(int num_tasks, std::function<void(int)> fn);
};
