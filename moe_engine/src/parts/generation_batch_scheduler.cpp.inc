
// ============================================================================
// Continuous Batching Scheduler (Disaggregated Inference Ready)
// ============================================================================
// Handles dynamic request queuing, iteration-level scheduling, and prefill/decode
// phase separation. Uses Paged KV Cache to avoid memory fragmentation.

#include <vector>
#include <deque>
#include <mutex>
#include <string>

enum moe_request_phase_t {
    moe_REQ_WAITING,
    moe_REQ_PREFILL,
    moe_REQ_DECODE,
    moe_REQ_FINISHED
};

struct moe_batch_request_t {
    uint64_t request_id;
    std::vector<int32_t> prompt_tokens;
    std::vector<int32_t> generated_tokens;
    std::vector<uint32_t> kv_block_table; // Maps logical blocks to physical blocks
    uint32_t processed_tokens = 0;
    moe_request_phase_t phase = moe_REQ_WAITING;
    // BUGFIX 702: Skip counter for head-of-line blocking prevention ★★
    // Tracks how many scheduling iterations this request was bypassed.
    // When skip_count >= kMaxSkipCount, the request is force-admitted next.
    uint32_t skip_count = 0;
};

struct moe_continuous_batch_scheduler_t {
    // BUGFIX 702: Replace std::queue with std::deque ★★
    // std::queue only allows front-pop, causing head-of-line blocking when
    // a large request at the front can't be admitted but smaller ones behind it could.
    // std::deque allows random-access iteration + erase for skip-based scheduling.
    std::deque<moe_batch_request_t*> waiting_queue;
    std::vector<moe_batch_request_t*> running_queue;
    std::mutex mtx;
    uint32_t max_running_requests = 16;
    uint32_t max_tokens_per_iteration = 512;

    // BUGFIX 702: Maximum skip count before force-admission ★★
    // Prevents starvation of large-context requests that keep getting bypassed.
    // At 16 skips, the request is treated as highest priority and blocks further
    // admission until KV blocks become available for it.
    static constexpr uint32_t kMaxSkipCount = 16;

    // BUGFIX 229: Add destructor to clean up pending requests ★★
    // Problem: Requests in waiting_queue and running_queue never freed on scheduler destruction
    // Solution: Delete all pending requests in destructor
    // Impact: Prevents memory leak on engine shutdown
    ~moe_continuous_batch_scheduler_t() {
        std::lock_guard<std::mutex> lock(mtx);
        for (auto* req : waiting_queue) {
            delete req;
        }
        waiting_queue.clear();
        for (auto* req : running_queue) {
            delete req;
        }
        running_queue.clear();
    }

    void add_request(moe_batch_request_t* req) {
        // BUGFIX 308: Reject null requests to prevent queue corruption ★
        // Problem: Caller can pass nullptr, causing crash in schedule_next_iteration
        // Solution: Check req before adding to queue
        if (!req) {
            return;
        }
        std::lock_guard<std::mutex> lock(mtx);
        waiting_queue.push_back(req);
    }

    // Steps the scheduler: moves waiting requests to running if capacity allows,
    // and determines the batch of tokens to process this iteration.
    void schedule_next_iteration(moe_paged_kv_manager_t* kv_manager,
                                 std::vector<moe_batch_request_t*>& out_prefill,
                                 std::vector<moe_batch_request_t*>& out_decode) {
        std::lock_guard<std::mutex> lock(mtx);

        // 1. Evict finished requests
        // BUGFIX 229: Delete finished requests to prevent memory leak ★★
        // Problem: erase() removes from vector but doesn't free memory allocated by new
        // Solution: Call delete before erase to free the moe_batch_request_t object
        // Impact: Prevents memory leak (sizeof(moe_batch_request_t) + vectors) per request
        for (auto it = running_queue.begin(); it != running_queue.end(); ) {
            // BUGFIX 310: Check iterator validity before dereferencing ★
            // Problem: *it can be null if corruption occurs
            // Solution: Skip null entries
            if (!(*it)) {
                it = running_queue.erase(it);
                continue;
            }
            if ((*it)->phase == moe_REQ_FINISHED) {
                // BUGFIX 311: Check kv_manager before freeing blocks ★
                // Problem: kv_manager can be null, causes crash in free_block
                // Solution: Only free blocks if kv_manager valid
                if (kv_manager) {
                    for (uint32_t block_id : (*it)->kv_block_table) {
                        kv_manager->free_block(block_id);
                    }
                }
                delete *it;  // BUGFIX 229: Free the request object
                it = running_queue.erase(it);
            } else {
                ++it;
            }
        }

        // 2. Admit new requests (Iteration-level scheduling)
        // BUGFIX 702: Skip-based scheduling to eliminate head-of-line blocking ★★
        // Old: FIFO-only with break on first failure → large request blocks all behind it.
        // New: Iterate the entire waiting_queue. If a request can't fit, skip it and try
        //      the next one. Increment skip_count on each bypass. When skip_count reaches
        //      kMaxSkipCount, force-block: stop admitting others until this request can go.
        //      This preserves approximate FIFO fairness while allowing small requests through.

        // BUGFIX 286: Check kv_manager validity before accessing free_blocks ★
        if (!kv_manager || kv_manager->block_size == 0) {
            // Cannot admit requests without valid KV manager
        } else {
            auto it = waiting_queue.begin();
            while (it != waiting_queue.end() && running_queue.size() < max_running_requests) {
                auto* req = *it;

                // BUGFIX 306: Check req validity before accessing members ★
                if (!req) {
                    it = waiting_queue.erase(it);
                    continue;
                }

                uint32_t needed_blocks = (static_cast<uint32_t>(req->prompt_tokens.size()) + kv_manager->block_size - 1) / kv_manager->block_size;

                if (kv_manager->free_blocks.size() >= needed_blocks) {
                    // Enough KV blocks available — admit this request
                    req->phase = moe_REQ_PREFILL;
                    req->skip_count = 0;
                    running_queue.push_back(req);
                    it = waiting_queue.erase(it);
                } else {
                    // Not enough KV blocks for this request
                    req->skip_count++;

                    // BUGFIX 702: Starvation guard ★★
                    // If this request has been skipped too many times, STOP admitting
                    // any further requests. This forces the system to drain KV blocks
                    // until this starved request can be served, preserving SLO fairness.
                    if (req->skip_count >= kMaxSkipCount) {
                        break;  // Force-block: no more admissions until this one fits
                    }

                    ++it;  // Skip this request, try the next one
                }
            }
        }

        // 3. Separate into prefill and decode batches
        for (auto* req : running_queue) {
            // BUGFIX 309: Check req validity before accessing phase ★
            // Problem: running_queue can contain null if corruption occurs
            // Solution: Skip null entries
            if (!req) {
                continue;
            }
            if (req->phase == moe_REQ_PREFILL) out_prefill.push_back(req);
            else if (req->phase == moe_REQ_DECODE) out_decode.push_back(req);
        }
    }
};
