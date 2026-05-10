// ============================================================================
// Continuous Batching Scheduler (always-built, paged-KV independent)
// ============================================================================
// The old scheduler was hidden behind MOE_ENABLE_PAGED_KV, so the public batch
// API could look present while the executable path was not compiled.  This
// scheduler is deliberately independent from paged-KV: it admits/runs requests
// with per-request scratch owned by the real executor in generation_api_batch.
// Paged-KV can be plugged back in later as an allocator, not as a compile gate.

#ifndef STORAGELLM_GENERATION_BATCH_SCHEDULER_INC
#define STORAGELLM_GENERATION_BATCH_SCHEDULER_INC

#include <vector>
#include <deque>
#include <mutex>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <utility>
#include <unordered_map>
#include <condition_variable>
#include <chrono>

static uint32_t moe_batch_sched_env_u32(const char* name, uint32_t fallback, uint32_t lo, uint32_t hi) {
    const char* v = name ? std::getenv(name) : nullptr;
    if (!v || !*v) return fallback;
    char* end = nullptr;
    unsigned long parsed = std::strtoul(v, &end, 10);
    if (end == v) return fallback;
    if (parsed < lo) parsed = lo;
    if (parsed > hi) parsed = hi;
    return static_cast<uint32_t>(parsed);
}

static bool moe_batch_sched_env_truthy(const char* name) {
    const char* v = name ? std::getenv(name) : nullptr;
    if (!v || !*v) return false;
    return !(std::strcmp(v, "0") == 0 || std::strcmp(v, "false") == 0 ||
             std::strcmp(v, "FALSE") == 0 || std::strcmp(v, "off") == 0 ||
             std::strcmp(v, "OFF") == 0 || std::strcmp(v, "no") == 0 ||
             std::strcmp(v, "NO") == 0);
}

static void moe_batch_sched_atomic_max(std::atomic<uint64_t>& target, uint64_t value) {
    uint64_t cur = target.load(std::memory_order_relaxed);
    while (cur < value && !target.compare_exchange_weak(
            cur, value, std::memory_order_relaxed, std::memory_order_relaxed)) {}
}

enum moe_request_phase_t {
    moe_REQ_WAITING,
    moe_REQ_PREFILL,
    moe_REQ_DECODE,
    moe_REQ_FINISHED,
    moe_REQ_FAILED
};

struct moe_batch_request_t {
    uint64_t request_id = 0;
    std::vector<int32_t> prompt_tokens;
    std::vector<int32_t> generated_tokens;
    uint32_t processed_tokens = 0;
    uint32_t produced_tokens = 0;
    uint32_t max_new_tokens = 0;
    uint32_t eos_token_id = 0;
    uint32_t stop_on_eos = 1;
    int32_t current_token = 0;
    int failed = 0;
    moe_request_phase_t phase = moe_REQ_WAITING;
    uint32_t skip_count = 0;

    // Per-session adaptive router EMA is kept here and copied to/from the
    // request's decode scratch by the executor.  This keeps multi-domain
    // batches from sharing hysteresis state through the engine object.
    uint32_t smoothed_router_entropy_milli = 0;
    uint32_t smoothed_adaptive_prefetch_depth = 2;
};

struct moe_batch_completed_request_t {
    std::vector<int32_t> generated_tokens;
    uint32_t prompt_tokens = 0;
    uint32_t produced_tokens = 0;
    int failed = 0;
};

struct moe_continuous_batch_scheduler_t {
    std::deque<moe_batch_request_t*> waiting_queue;
    std::vector<moe_batch_request_t*> running_queue;
    std::unordered_map<uint64_t, moe_batch_completed_request_t> completed_results;
    std::deque<uint64_t> completed_result_order;
    std::mutex mtx;
    std::condition_variable cv;
    uint32_t max_running_requests = 16;
    uint32_t max_tokens_per_iteration = 512;
    uint32_t max_completed_results = 4096;
    size_t round_robin_cursor = 0;
    std::atomic<uint64_t> submitted_requests{0};
    std::atomic<uint64_t> admitted_requests{0};
    std::atomic<uint64_t> completed_requests{0};
    std::atomic<uint64_t> failed_requests{0};
    std::atomic<uint64_t> cancelled_requests{0};
    std::atomic<uint64_t> schedule_iterations{0};
    std::atomic<uint64_t> prefill_scheduled{0};
    std::atomic<uint64_t> decode_scheduled{0};
    std::atomic<uint64_t> max_waiting_observed{0};
    std::atomic<uint64_t> max_running_observed{0};
    std::atomic<uint64_t> max_completed_observed{0};

    moe_continuous_batch_scheduler_t() {
        const bool max_parallel = moe_batch_sched_env_truthy("STORAGELLM_MAX_PARALLELISM") ||
                                  moe_batch_sched_env_truthy("STORAGELLM_CONTINUOUS_BATCHING") ||
                                  moe_batch_sched_env_truthy("STORAGELLM_CONTINUOUS_BATCH_MAX") ||
                                  moe_batch_sched_env_truthy("STORAGELLM_ENABLE_REAL_CONTINUOUS_BATCH_EXEC");
        const uint32_t default_running = max_parallel ? 64u : 16u;
        const uint32_t default_tokens = max_parallel ? 2048u : 512u;
        max_running_requests = moe_batch_sched_env_u32(
            "STORAGELLM_CONTINUOUS_BATCH_MAX_REQUESTS", default_running, 1u, 1024u);
        max_tokens_per_iteration = moe_batch_sched_env_u32(
            "STORAGELLM_CONTINUOUS_BATCH_MAX_TOKENS", default_tokens, 1u, 65536u);
        max_completed_results = moe_batch_sched_env_u32(
            "STORAGELLM_CONTINUOUS_BATCH_MAX_COMPLETED", 4096u, 16u, 1048576u);
    }

    ~moe_continuous_batch_scheduler_t() {
        std::lock_guard<std::mutex> lock(mtx);
        for (auto* req : waiting_queue) delete req;
        waiting_queue.clear();
        for (auto* req : running_queue) delete req;
        running_queue.clear();
    }

    void add_request(moe_batch_request_t* req) {
        if (!req) return;
        std::lock_guard<std::mutex> lock(mtx);
        waiting_queue.push_back(req);
        submitted_requests.fetch_add(1, std::memory_order_relaxed);
        moe_batch_sched_atomic_max(max_waiting_observed, (uint64_t)waiting_queue.size());
        cv.notify_all();
    }

    void cleanup_finished_locked() {
        for (auto*& req : running_queue) {
            if (!req) continue;
            if (req->phase == moe_REQ_FINISHED || req->phase == moe_REQ_FAILED) {
                moe_batch_completed_request_t completed;
                completed.generated_tokens = std::move(req->generated_tokens);
                completed.prompt_tokens = (uint32_t)req->prompt_tokens.size();
                completed.produced_tokens = req->produced_tokens;
                completed.failed = (req->phase == moe_REQ_FAILED || req->failed) ? 1 : 0;
                const uint64_t completed_id = req->request_id;
                const bool replacing_existing = completed_results.find(completed_id) != completed_results.end();
                completed_results[completed_id] = std::move(completed);
                if (!replacing_existing) {
                    completed_result_order.push_back(completed_id);
                }
                // Evict oldest completed results deterministically instead of erasing
                // an arbitrary unordered_map bucket.  Request threads usually fetch
                // immediately, but under load a random eviction can drop a newer
                // result while preserving an older one.  Keep FIFO order and tolerate
                // stale ids left by fetch/remove.
                while (completed_results.size() > max_completed_results && !completed_result_order.empty()) {
                    const uint64_t evict_id = completed_result_order.front();
                    completed_result_order.pop_front();
                    auto evict_it = completed_results.find(evict_id);
                    if (evict_it != completed_results.end()) {
                        completed_results.erase(evict_it);
                    }
                }
                moe_batch_sched_atomic_max(max_completed_observed, (uint64_t)completed_results.size());
                if (req->phase == moe_REQ_FINISHED) {
                    completed_requests.fetch_add(1, std::memory_order_relaxed);
                } else {
                    failed_requests.fetch_add(1, std::memory_order_relaxed);
                }
                delete req;
                req = nullptr;
            }
        }
        running_queue.erase(std::remove(running_queue.begin(), running_queue.end(), nullptr), running_queue.end());
        cv.notify_all();
    }

    void cleanup_finished() {
        std::lock_guard<std::mutex> lock(mtx);
        cleanup_finished_locked();
    }

    // Admit waiting requests and return a stable snapshot of currently running
    // requests.  The executor owns per-request scratch outside this scheduler;
    // this object only handles admission/fairness/accounting.
    void schedule_next_iteration(std::vector<moe_batch_request_t*>& out_prefill,
                                 std::vector<moe_batch_request_t*>& out_decode) {
        std::lock_guard<std::mutex> lock(mtx);
        schedule_iterations.fetch_add(1, std::memory_order_relaxed);
        cleanup_finished_locked();

        while (!waiting_queue.empty() && running_queue.size() < max_running_requests) {
            moe_batch_request_t* req = waiting_queue.front();
            waiting_queue.pop_front();
            if (!req) continue;
            // A one-token prompt has no prefix tokens to prefill; it should enter
            // decode immediately with current_token initialized from the prompt by
            // the executor state.  Otherwise the scheduler can keep a zero-prefill
            // request in PREFILL forever because there is no token to select.
            req->phase = (req->prompt_tokens.size() <= 1u) ? moe_REQ_DECODE : moe_REQ_PREFILL;
            req->skip_count = 0;
            running_queue.push_back(req);
            admitted_requests.fetch_add(1, std::memory_order_relaxed);
        }

        moe_batch_sched_atomic_max(max_waiting_observed, (uint64_t)waiting_queue.size());
        moe_batch_sched_atomic_max(max_running_observed, (uint64_t)running_queue.size());

        uint32_t tokens_budget = max_tokens_per_iteration;
        const size_t n = running_queue.size();
        if (n == 0) {
            round_robin_cursor = 0;
        } else {
            if (round_robin_cursor >= n) round_robin_cursor = 0;
            // Fair multi-round scheduling: the previous implementation visited each
            // running request only once, so max_tokens_per_iteration was effectively
            // capped at running_queue.size().  Real continuous batching should spend
            // the whole token budget across active requests while still rotating the
            // starting point to avoid starvation.  We enqueue one token per request per
            // round, repeat until the budget is spent, and cap each request by its
            // actual remaining prefill/decode tokens.
            std::vector<uint32_t> scheduled_for_request(n, 0u);
            size_t idx = round_robin_cursor;
            size_t idle_visits = 0;
            while (tokens_budget > 0 && idle_visits < n) {
                moe_batch_request_t* req = running_queue[idx];
                bool selected = false;
                if (req) {
                    if (req->phase == moe_REQ_PREFILL) {
                        const uint32_t prompt = static_cast<uint32_t>(req->prompt_tokens.size());
                        const uint32_t prefill_total = prompt > 0u ? prompt - 1u : 0u;
                        const uint32_t already = req->processed_tokens + scheduled_for_request[idx];
                        if (already < prefill_total) {
                            out_prefill.push_back(req);
                            ++scheduled_for_request[idx];
                            --tokens_budget;
                            selected = true;
                        }
                    } else if (req->phase == moe_REQ_DECODE) {
                        const uint32_t already = req->produced_tokens + scheduled_for_request[idx];
                        if (already < req->max_new_tokens) {
                            out_decode.push_back(req);
                            ++scheduled_for_request[idx];
                            --tokens_budget;
                            selected = true;
                        }
                    }
                }
                idx = (idx + 1u) % n;
                if (selected) {
                    idle_visits = 0;
                } else {
                    ++idle_visits;
                }
            }
            round_robin_cursor = idx;
        }
        prefill_scheduled.fetch_add((uint64_t)out_prefill.size(), std::memory_order_relaxed);
        decode_scheduled.fetch_add((uint64_t)out_decode.size(), std::memory_order_relaxed);
    }
};

static int moe_batch_scheduler_fetch_result(
    moe_continuous_batch_scheduler_t* scheduler,
    uint64_t request_id,
    int32_t* out_tokens,
    uint32_t out_capacity,
    uint32_t* out_count,
    uint32_t* out_prompt_tokens,
    int* out_failed,
    int remove_after_fetch
) {
    if (out_count) *out_count = 0;
    if (out_prompt_tokens) *out_prompt_tokens = 0;
    if (out_failed) *out_failed = 0;
    if (!scheduler) return 0;
    std::lock_guard<std::mutex> lock(scheduler->mtx);
    auto it = scheduler->completed_results.find(request_id);
    if (it == scheduler->completed_results.end()) return 0;
    const moe_batch_completed_request_t& completed = it->second;
    const uint32_t n = (uint32_t)completed.generated_tokens.size();
    if (out_count) *out_count = n;
    if (out_prompt_tokens) *out_prompt_tokens = completed.prompt_tokens;
    if (out_failed) *out_failed = completed.failed;
    if (out_tokens && out_capacity > 0) {
        const uint32_t copy_n = std::min<uint32_t>(out_capacity, n);
        std::copy(completed.generated_tokens.begin(), completed.generated_tokens.begin() + copy_n, out_tokens);
    }
    if (remove_after_fetch) {
        scheduler->completed_results.erase(it);
        // Leave the id in completed_result_order; FIFO eviction skips stale ids.
        // Avoiding O(n) removal keeps the hot result path cheap.
    }
    return 1;
}

static int moe_batch_scheduler_wait_for_result(
    moe_continuous_batch_scheduler_t* scheduler,
    uint64_t request_id,
    uint32_t timeout_ms
) {
    if (!scheduler) return 0;
    std::unique_lock<std::mutex> lock(scheduler->mtx);
    const auto pred = [&]() {
        return scheduler->completed_results.find(request_id) != scheduler->completed_results.end();
    };
    if (pred()) return 1;
    if (timeout_ms == 0) {
        scheduler->cv.wait(lock, pred);
        return 1;
    }
    return scheduler->cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), pred) ? 1 : 0;
}

static int moe_batch_scheduler_cancel_request(
    moe_continuous_batch_scheduler_t* scheduler,
    uint64_t request_id
) {
    if (!scheduler) return 0;
    std::lock_guard<std::mutex> lock(scheduler->mtx);
    for (auto it = scheduler->waiting_queue.begin(); it != scheduler->waiting_queue.end(); ++it) {
        moe_batch_request_t* req = *it;
        if (req && req->request_id == request_id) {
            delete req;
            scheduler->waiting_queue.erase(it);
            scheduler->cancelled_requests.fetch_add(1, std::memory_order_relaxed);
            scheduler->cv.notify_all();
            return 1;
        }
    }
    for (auto* req : scheduler->running_queue) {
        if (req && req->request_id == request_id) {
            req->phase = moe_REQ_FAILED;
            req->failed = 1;
            scheduler->cancelled_requests.fetch_add(1, std::memory_order_relaxed);
            scheduler->cv.notify_all();
            return 1;
        }
    }
    return 0;
}

#endif // STORAGELLM_GENERATION_BATCH_SCHEDULER_INC
