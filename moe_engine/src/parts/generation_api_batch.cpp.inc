// ============================================================================
// Continuous Batching API (Disaggregated Inference Core)
// ============================================================================
// Provides the entry point for submitting concurrent requests to the MoE engine.
// The engine schedules them automatically across the prefill and decode stages.

#include <vector>

#if defined(MOE_ENABLE_PAGED_KV) && MOE_ENABLE_PAGED_KV

// Forward declarations of normal inference loops that will be batched later.
// Currently this is a scaffold to demonstrate Continuous Batching queuing.

int moe_pc_engine_submit_request(
    moe_pc_engine_t* engine,
    uint64_t request_id,
    const int32_t* prompt_tokens,
    uint32_t token_count
) {
    if (!engine || !prompt_tokens || token_count == 0) return 0;
    moe_batch_request_t* req = new (std::nothrow) moe_batch_request_t();
    if (!req) return 0;
    req->request_id = request_id;
    req->prompt_tokens.assign(prompt_tokens, prompt_tokens + token_count);
    req->phase = moe_REQ_WAITING;

    engine->batch_scheduler.add_request(req);
    std::fprintf(stderr, "[storageLLM INFO] Request %llu submitted for continuous batching (tokens: %u)\n",
                 (unsigned long long)request_id, token_count);
    return 1;
}

static int moe_layer_attention_batched_f32(
    moe_pc_engine_t* engine,
    int32_t layer,
    moe_decode_scratch* s,
    const float** batch_hidden,
    uint32_t batch_size,
    uint32_t hidden_size,
    const uint32_t* batch_positions
) {
    // MVP implementation: loop over single-batch attention
    for (uint32_t i = 0; i < batch_size; ++i) {
        if (!moe_layer_attention_f32(engine, layer, s, const_cast<float*>(batch_hidden[i]), hidden_size, batch_positions[i])) {
            return 0;
        }
    }
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
    // MVP implementation: loop over single-batch MLP
    for (uint32_t i = 0; i < batch_size; ++i) {
        if (!moe_layer_moe_mlp_f32(engine, layer, s, const_cast<float*>(batch_hidden[i]), c)) {
            return 0;
        }
    }
    return 1;
}

int moe_pc_engine_step_batch(moe_pc_engine_t* engine) {
    if (!engine) return 0;

    std::vector<moe_batch_request_t*> prefill_reqs;
    std::vector<moe_batch_request_t*> decode_reqs;

    // Note: paged_kv_manager doesn't exist directly on engine in this codebase,
    // we pass nullptr or fetch it if it existed.
    engine->batch_scheduler.schedule_next_iteration(nullptr, prefill_reqs, decode_reqs);

    if (prefill_reqs.empty() && decode_reqs.empty()) {
        return 0; // No active requests
    }

    // MVP Continuous Batching execution
    // Currently we fallback to single-request generate_token_ids for each to drain the queue.
    // Full batched GEMM requires tensor_dot_cpu_kernels modifications.

    for (auto* req : prefill_reqs) {
        moe_generation_stats_t stats;
        std::vector<uint32_t> out_tokens(1024);
        moe_generation_config_t config{};
        config.max_new_tokens = 100;

        // This is a blocking call for the entire sequence as MVP
        moe_pc_engine_generate_token_ids(
            engine,
            req->prompt_tokens.data(),
            req->prompt_tokens.size(),
            &config,
            out_tokens.data(),
            out_tokens.size(),
            &stats
        );
        req->phase = moe_REQ_FINISHED;
    }

    for (auto* req : decode_reqs) {
        // Handled by prefill loop in MVP
        req->phase = moe_REQ_FINISHED;
    }

    return 1;
}

#endif
