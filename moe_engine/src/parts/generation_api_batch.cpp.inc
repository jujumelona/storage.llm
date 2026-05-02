// ============================================================================
// Continuous Batching API (Disaggregated Inference Core)
// ============================================================================
// Provides the entry point for submitting concurrent requests to the MoE engine.
// The engine schedules them automatically across the prefill and decode stages.

#include <vector>

// Forward declarations of normal inference loops that will be batched later.
// Currently this is a scaffold to demonstrate Continuous Batching queuing.

int moe_pc_engine_submit_request(
    moe_pc_engine_t* engine,
    uint64_t request_id,
    const int32_t* prompt_tokens,
    uint32_t token_count
) {
    if (!engine || !prompt_tokens || token_count == 0) return 0;

    // BUGFIX 634: Check new allocation failure ★
    moe_batch_request_t* req = new (std::nothrow) moe_batch_request_t();
    if (!req) {
        return 0;
    }
    req->request_id = request_id;
    req->prompt_tokens.assign(prompt_tokens, prompt_tokens + token_count);
    req->phase = moe_REQ_WAITING;

    engine->batch_scheduler.add_request(req);

    std::fprintf(stderr, "[storageLLM INFO] Request %llu submitted for continuous batching (tokens: %u)\n",
                 (unsigned long long)request_id, token_count);
    return 1;
}

int moe_pc_engine_step_batch(moe_pc_engine_t* engine) {
    if (!engine) return 0;

    std::lock_guard<std::mutex> lock(engine->generation_api_mutex);

    std::vector<moe_batch_request_t*> prefill_batch;
    std::vector<moe_batch_request_t*> decode_batch;

    // 1. Scheduler admits new requests and prepares batches
    engine->batch_scheduler.schedule_next_iteration(&engine->paged_kv_manager, prefill_batch, decode_batch);

    // 2. Disaggregated Prefill Phase
    // (Here we would launch a batched forward pass over all prefill_batch tokens)
    if (!prefill_batch.empty()) {
        std::fprintf(stderr, "[storageLLM BATCH] Running Disaggregated Prefill for %zu requests\n", prefill_batch.size());
        for (auto* req : prefill_batch) {
            req->phase = moe_REQ_DECODE; // Move to decode phase next iteration
            req->processed_tokens = (uint32_t)req->prompt_tokens.size();
        }
    }

    // 3. Decode Phase
    // (Here we would launch a batched forward pass over all decode_batch tokens)
    if (!decode_batch.empty()) {
        std::fprintf(stderr, "[storageLLM BATCH] Running Continuous Decode for %zu requests\n", decode_batch.size());
        for (auto* req : decode_batch) {
            // Simulated token generation for demo
            req->generated_tokens.push_back(0);
            req->processed_tokens++;

            // Randomly finish
            if (req->generated_tokens.size() > 50) {
                req->phase = moe_REQ_FINISHED;
            }
        }
    }

    return prefill_batch.empty() && decode_batch.empty() ? 0 : 1;
}

// ============================================================================
// Batched Forward Pass APIs (Continuous Batching)
// ============================================================================

static int moe_layer_attention_batched_f32(
    moe_pc_engine_t* engine,
    int32_t layer,
    moe_decode_scratch* s,
    const float** batch_hidden,
    uint32_t batch_size,
    uint32_t hidden_size,
    const uint32_t* batch_positions
) {
    // Future Implementation: GEMM-based batched attention with Paged KV
    return 1;
}

static int moe_layer_moe_mlp_batched_f32(
    moe_pc_engine_t* engine,
    int32_t layer,
    moe_decode_scratch* s,
    const float** batch_hidden,
    uint32_t batch_size,
    const moe_storage_constants_t* c
) {
    // Future Implementation: Batched Expert Routing and GEMM dispatch
    return 1;
}
