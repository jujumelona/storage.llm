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
