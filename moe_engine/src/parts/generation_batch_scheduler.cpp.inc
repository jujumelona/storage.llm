
// ============================================================================
// Continuous Batching Scheduler (Disaggregated Inference Ready)
// ============================================================================
// Handles dynamic request queuing, iteration-level scheduling, and prefill/decode
// phase separation. Uses Paged KV Cache to avoid memory fragmentation.

#include <vector>
#include <queue>
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
};

struct moe_continuous_batch_scheduler_t {
    std::queue<moe_batch_request_t*> waiting_queue;
    std::vector<moe_batch_request_t*> running_queue;
    std::mutex mtx;
    uint32_t max_running_requests = 16;
    uint32_t max_tokens_per_iteration = 512;

    void add_request(moe_batch_request_t* req) {
        std::lock_guard<std::mutex> lock(mtx);
        waiting_queue.push(req);
    }

    // Steps the scheduler: moves waiting requests to running if capacity allows,
    // and determines the batch of tokens to process this iteration.
    void schedule_next_iteration(moe_paged_kv_manager_t* kv_manager,
                                 std::vector<moe_batch_request_t*>& out_prefill,
                                 std::vector<moe_batch_request_t*>& out_decode) {
        std::lock_guard<std::mutex> lock(mtx);

        // 1. Evict finished requests
        for (auto it = running_queue.begin(); it != running_queue.end(); ) {
            if ((*it)->phase == moe_REQ_FINISHED) {
                for (uint32_t block_id : (*it)->kv_block_table) {
                    kv_manager->free_block(block_id);
                }
                it = running_queue.erase(it);
            } else {
                ++it;
            }
        }

        // 2. Admit new requests (Iteration-level scheduling)
        while (running_queue.size() < max_running_requests && !waiting_queue.empty()) {
            auto* req = waiting_queue.front();

            // Estimate if we have enough blocks for the prompt
            uint32_t needed_blocks = (req->prompt_tokens.size() + kv_manager->block_size - 1) / kv_manager->block_size;
            if (kv_manager->free_blocks.size() >= needed_blocks) {
                req->phase = moe_REQ_PREFILL;
                running_queue.push_back(req);
                waiting_queue.pop();
            } else {
                break; // KV Cache full
            }
        }

        // 3. Separate into prefill and decode batches
        for (auto* req : running_queue) {
            if (req->phase == moe_REQ_PREFILL) out_prefill.push_back(req);
            else if (req->phase == moe_REQ_DECODE) out_decode.push_back(req);
        }
    }
};
