// Raw forward - LM head operations

int glm5_pc_engine_lm_head_top1_f32(
    glm5_pc_engine_t* engine,
    const float* hidden,
    uint32_t hidden_count,
    uint32_t* out_token_id,
    float* out_logit
) {
    const glm5_storage_constants_t* c = glm5_storage_constants();
    if (!engine || !c || !hidden || !out_token_id || !out_logit ||
        hidden_count != c->hidden_size || c->vocab_size == 0) {
        return 0;
    }
    const int32_t lm_head_layer = (int32_t)c->num_hidden_layers + 1;
    const glm5_storage_raw_span_spec_t* span = find_raw_role_span(lm_head_layer, GLM5_RAW_ROLE_LM_HEAD);
    mmap_context_t* mapped = map_raw_span_part(engine, span);
    if (!raw_span_bounds_ok(span, mapped)) {
        return 0;
    }
    const uint64_t row_bytes = (uint64_t)c->hidden_size * 2ull;
    if (span->bytes < row_bytes * (uint64_t)c->vocab_size) {
        return 0;
    }

    const uint8_t* base = raw_span_ptr(span, mapped, 0, row_bytes * (uint64_t)c->vocab_size);
    if (!base) {
        return 0;
    }

    engine->generation_phase.store(GLM5_GEN_PHASE_LM_HEAD, std::memory_order_relaxed);
    engine->generation_layer.store(0xffffffffu, std::memory_order_relaxed);

    const uint32_t chunk_size = 1024;
    const uint32_t num_chunks = (c->vocab_size + chunk_size - 1) / chunk_size;

    struct chunk_result_t {
        uint32_t token;
        float score;
    };
    // Bug #27 Fix: Use regular local vector instead of thread_local.
    // thread_local causes UB: main thread resizes its instance, but worker threads
    // access their own uninitialized instances. Since each chunk is assigned to
    // exactly one thread (via cpu_next_row.fetch_add()), no mutex needed.
    std::vector<chunk_result_t> chunk_results(num_chunks);

    if (!glm5_parallel_rows(engine, num_chunks, 0, [&](uint32_t chunk) {
        const uint32_t start = chunk * chunk_size;
        const uint32_t end = std::min(start + chunk_size, c->vocab_size);
        uint32_t local_best = start;
        float local_max = -std::numeric_limits<float>::infinity();

        for (uint32_t t = start; t < end; ++t) {
            const uint8_t* row = base + row_bytes * (uint64_t)t;
            float s = glm5_dot_bf16_row(row, hidden, hidden_count);
            if (s > local_max) {
                local_max = s;
                local_best = t;
            }
        }

        chunk_results[chunk] = {local_best, local_max};
        return 1;
    })) {
        return 0;
    }

    // Single-thread final reduce over chunk_results
    uint32_t best_token = 0;
    float best_score = -std::numeric_limits<float>::infinity();
    for (uint32_t i = 0; i < num_chunks; ++i) {
        if (chunk_results[i].score > best_score) {
            best_score = chunk_results[i].score;
            best_token = chunk_results[i].token;
        }
    }

    *out_token_id = best_token;
    *out_logit = best_score;
    return 1;
}

int glm5_pc_engine_lm_head_token_logprob_f32(
    glm5_pc_engine_t* engine,
    const float* hidden,
    uint32_t hidden_count,
    uint32_t target_token_id,
    float* out_target_logit,
    float* out_logsumexp,
    float* out_logprob
) {
    const glm5_storage_constants_t* c = glm5_storage_constants();
    if (!engine || !c || !hidden || !out_target_logit || !out_logsumexp || !out_logprob ||
        hidden_count != c->hidden_size || c->vocab_size == 0 || target_token_id >= c->vocab_size) {
        return 0;
    }
    const int32_t lm_head_layer = (int32_t)c->num_hidden_layers + 1;
    const glm5_storage_raw_span_spec_t* span = find_raw_role_span(lm_head_layer, GLM5_RAW_ROLE_LM_HEAD);
    mmap_context_t* mapped = map_raw_span_part(engine, span);
    if (!raw_span_bounds_ok(span, mapped)) {
        return 0;
    }
    const uint64_t row_bytes = (uint64_t)c->hidden_size * 2ull;
    if (span->bytes < row_bytes * (uint64_t)c->vocab_size) {
        return 0;
    }

    const uint8_t* base = raw_span_ptr(span, mapped, 0, row_bytes * (uint64_t)c->vocab_size);
    if (!base) {
        return 0;
    }

    engine->generation_phase.store(GLM5_GEN_PHASE_LM_HEAD, std::memory_order_relaxed);
    engine->generation_layer.store(0xffffffffu, std::memory_order_relaxed);

    const uint32_t chunk_size = 1024;
    const uint32_t num_chunks = (c->vocab_size + chunk_size - 1) / chunk_size;

    struct chunk_logsum_t {
        float max_score;
        double exp_sum;
        float target_score;
        int has_target;
    };
    std::vector<chunk_logsum_t> chunk_results(num_chunks);

    if (!glm5_parallel_rows(engine, num_chunks, 0, [&](uint32_t chunk) {
        const uint32_t start = chunk * chunk_size;
        const uint32_t end = std::min(start + chunk_size, c->vocab_size);
        float local_max = -std::numeric_limits<float>::infinity();
        double local_sum = 0.0;
        float target_score = 0.0f;
        int has_target = 0;

        for (uint32_t t = start; t < end; ++t) {
            const uint8_t* row = base + row_bytes * (uint64_t)t;
            const float score = glm5_dot_bf16_row(row, hidden, hidden_count);
            if (score <= local_max) {
                local_sum += std::exp((double)score - (double)local_max);
            } else if (std::isfinite(local_max)) {
                local_sum = local_sum * std::exp((double)local_max - (double)score) + 1.0;
                local_max = score;
            } else {
                local_max = score;
                local_sum = 1.0;
            }
            if (t == target_token_id) {
                target_score = score;
                has_target = 1;
            }
        }

        chunk_results[chunk] = {local_max, local_sum, target_score, has_target};
        return 1;
    })) {
        return 0;
    }

    float global_max = -std::numeric_limits<float>::infinity();
    double exp_sum = 0.0;
    float target_score = 0.0f;
    int found_target = 0;
    for (uint32_t i = 0; i < num_chunks; ++i) {
        if (!(chunk_results[i].exp_sum > 0.0) || !std::isfinite(chunk_results[i].max_score)) {
            continue;
        }
        if (chunk_results[i].max_score <= global_max) {
            exp_sum += chunk_results[i].exp_sum *
                std::exp((double)chunk_results[i].max_score - (double)global_max);
        } else if (std::isfinite(global_max)) {
            exp_sum = exp_sum * std::exp((double)global_max - (double)chunk_results[i].max_score) +
                chunk_results[i].exp_sum;
            global_max = chunk_results[i].max_score;
        } else {
            global_max = chunk_results[i].max_score;
            exp_sum = chunk_results[i].exp_sum;
        }
        if (chunk_results[i].has_target) {
            target_score = chunk_results[i].target_score;
            found_target = 1;
        }
    }
    if (!found_target || !std::isfinite(global_max)) {
        return 0;
    }

    if (!(exp_sum > 0.0) || !std::isfinite(exp_sum)) {
        return 0;
    }

    const double logsumexp = (double)global_max + std::log(exp_sum);
    const double logprob = (double)target_score - logsumexp;
    *out_target_logit = target_score;
    *out_logsumexp = (float)logsumexp;
    *out_logprob = (float)logprob;
    return std::isfinite(logprob) ? 1 : 0;
}
