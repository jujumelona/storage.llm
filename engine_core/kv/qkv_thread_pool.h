#pragma once

#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <functional>

// QKV Thread Pool - Parallel execution for K scoring and V accumulation
class QkvThreadPool {
public:
    std::vector<std::thread> workers;
    std::condition_variable cv;
    std::condition_variable done_cv;
    std::mutex mtx;
    std::function<void(int)> task;
    int total_tasks = 0;
    int current_task = 0;
    int completed_tasks = 0;
    bool stop = false;

    explicit QkvThreadPool(int num_threads);
    ~QkvThreadPool();

    void run(int num_tasks, std::function<void(int)> fn);
};
