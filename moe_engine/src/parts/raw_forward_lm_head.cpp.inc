// Raw forward - LM head operations

static inline float moe_apply_final_logit_softcap_cached(float cap, float score) {
    if (!(cap > 0.0f) || !std::isfinite(score)) {
        return score;
    }
    return std::tanh(score / cap) * cap;
}

static int moe_map_lm_head_tensor(
    moe_pc_engine* engine,
    const moe_storage_constants_t* c,
    moe_raw_tensor_cache_entry* out
) {
    if (!engine || !c || !out) {
        return 0;
    }
    static const char* const lm_head_names[] = {
        "output.weight",
        "lm_head.weight",
        "token_embd.weight",
        "model.embed_tokens.weight",
        "embed_tokens.weight"
    };
    moe_raw_tensor_desc desc{};
    const moe_gguf_common_tensor_record* rec =
        find_common_tensor_by_names(engine, lm_head_names, sizeof(lm_head_names) / sizeof(lm_head_names[0]), &desc);
    if (!rec || desc.cols != c->hidden_size || desc.rows < c->vocab_size) {
        return 0;
    }
    return map_common_tensor_record(engine, *rec, desc, out);
}

int moe_pc_engine_lm_head_top1_f32(
    moe_pc_engine_t* engine,
    const float* hidden,
    uint32_t hidden_count,
    uint32_t* out_token_id,
    float* out_logit
) {
    const moe_storage_constants_t runtime_c = moe_engine_storage_constants_or_static(engine);
    const moe_storage_constants_t* c = &runtime_c;
    if (!engine || !c || !hidden || !out_token_id || !out_logit ||
        hidden_count != c->hidden_size || c->vocab_size == 0) {
        return 0;
    }
    moe_raw_tensor_cache_entry common{};
    const uint8_t* base = nullptr;
    uint64_t bytes = 0;
    uint8_t encoding = moe_RAW_SCALAR_BF16;
    uint8_t weight_format = moe_WEIGHT_ENCODING_RAW_FP16;
    uint64_t weight_row_bytes = 0;
    if (moe_map_lm_head_tensor(engine, c, &common)) {
        base = common.base;
        bytes = common.bytes;
        encoding = common.scalar_encoding;
        weight_format = common.weight_format;
        weight_row_bytes = common.weight_row_bytes;
    } else {
        const int32_t lm_head_layer = (int32_t)c->num_hidden_layers + 1;
        const moe_storage_raw_span_spec_t* span = find_raw_role_span(lm_head_layer, moe_RAW_ROLE_LM_HEAD);
        mmap_context_t* mapped = map_raw_span_part(engine, span);
        if (!raw_span_bounds_ok(span, mapped)) {
            return 0;
        }
        base = raw_span_ptr(span, mapped, 0, span->bytes);
        bytes = span->bytes;
        if (!base) {
            return 0;
        }
    }
    moe_raw_tensor_cache_entry raw{};
    raw.base = base;
    raw.bytes = bytes;
    raw.rows = c->vocab_size;
    raw.cols = c->hidden_size;
    raw.scalar_encoding = encoding;
    raw.weight_format = weight_format;
    raw.weight_row_bytes = weight_row_bytes;
    const uint64_t row_bytes = moe_raw_tensor_row_bytes(raw, c->hidden_size);
    if (bytes < row_bytes * (uint64_t)c->vocab_size) {
        return 0;
    }
    const float logit_softcap = moe_final_logit_softcap_value(engine);
    const uint64_t total_start_us = moe_profile_now_us();
    moe_profile_log(
        "lm_head_top1_begin",
        "vocab=%u hidden=%u fmt=%u scalar=%u row_bytes=%llu bytes=%llu softcap=%.6g",
        c->vocab_size,
        c->hidden_size,
        (unsigned)raw.weight_format,
        (unsigned)raw.scalar_encoding,
        (unsigned long long)row_bytes,
        (unsigned long long)bytes,
        (double)logit_softcap);

    engine->generation_phase.store(moe_GEN_PHASE_LM_HEAD, std::memory_order_relaxed);
    engine->generation_layer.store(0xffffffffu, std::memory_order_relaxed);

    static const unsigned int hw_raw = std::thread::hardware_concurrency();
    static const unsigned int hw_threads = hw_raw > 0 ? hw_raw : 4u;
    const uint32_t chunk_size = std::max(64u, (c->vocab_size + hw_threads * 8 - 1) / (hw_threads * 8));
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

    {
        moe_profile_parallel_label_scope rows_label("lm_head_top1");
        if (!moe_parallel_rows(engine, num_chunks, 0, [&](uint32_t chunk) {
            const uint32_t start = chunk * chunk_size;
            const uint32_t end = std::min(start + chunk_size, c->vocab_size);
            uint32_t local_best = start;
            float local_max = -std::numeric_limits<float>::infinity();

            for (uint32_t t = start; t < end; ++t) {
                const float s = moe_apply_final_logit_softcap_cached(
                    logit_softcap, moe_raw_tensor_dot_row(raw, t, hidden, hidden_count));
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
    moe_profile_log(
        "lm_head_top1_end",
        "best=%u score=%.6g chunks=%u total_ms=%.3f",
        best_token,
        (double)best_score,
        num_chunks,
        moe_profile_elapsed_ms(total_start_us));
    return 1;
}

int moe_pc_engine_lm_head_token_logprob_f32(
    moe_pc_engine_t* engine,
    const float* hidden,
    uint32_t hidden_count,
    uint32_t target_token_id,
    float* out_target_logit,
    float* out_logsumexp,
    float* out_logprob
) {
    const moe_storage_constants_t runtime_c = moe_engine_storage_constants_or_static(engine);
    const moe_storage_constants_t* c = &runtime_c;
    if (!engine || !c || !hidden || !out_target_logit || !out_logsumexp || !out_logprob ||
        hidden_count != c->hidden_size || c->vocab_size == 0 || target_token_id >= c->vocab_size) {
        return 0;
    }
    moe_raw_tensor_cache_entry common{};
    const uint8_t* base = nullptr;
    uint64_t bytes = 0;
    uint8_t encoding = moe_RAW_SCALAR_BF16;
    uint8_t weight_format = moe_WEIGHT_ENCODING_RAW_FP16;
    uint64_t weight_row_bytes = 0;
    if (moe_map_lm_head_tensor(engine, c, &common)) {
        base = common.base;
        bytes = common.bytes;
        encoding = common.scalar_encoding;
        weight_format = common.weight_format;
        weight_row_bytes = common.weight_row_bytes;
    } else {
        const int32_t lm_head_layer = (int32_t)c->num_hidden_layers + 1;
        const moe_storage_raw_span_spec_t* span = find_raw_role_span(lm_head_layer, moe_RAW_ROLE_LM_HEAD);
        mmap_context_t* mapped = map_raw_span_part(engine, span);
        if (!raw_span_bounds_ok(span, mapped)) {
            return 0;
        }
        base = raw_span_ptr(span, mapped, 0, span->bytes);
        bytes = span->bytes;
        if (!base) {
            return 0;
        }
    }
    moe_raw_tensor_cache_entry raw{};
    raw.base = base;
    raw.bytes = bytes;
    raw.rows = c->vocab_size;
    raw.cols = c->hidden_size;
    raw.scalar_encoding = encoding;
    raw.weight_format = weight_format;
    raw.weight_row_bytes = weight_row_bytes;
    const uint64_t row_bytes = moe_raw_tensor_row_bytes(raw, c->hidden_size);
    if (bytes < row_bytes * (uint64_t)c->vocab_size) {
        return 0;
    }
    const float logit_softcap = moe_final_logit_softcap_value(engine);
    const uint64_t total_start_us = moe_profile_now_us();
    moe_profile_log(
        "lm_head_logprob_begin",
        "target=%u vocab=%u hidden=%u fmt=%u scalar=%u row_bytes=%llu bytes=%llu softcap=%.6g",
        target_token_id,
        c->vocab_size,
        c->hidden_size,
        (unsigned)raw.weight_format,
        (unsigned)raw.scalar_encoding,
        (unsigned long long)row_bytes,
        (unsigned long long)bytes,
        (double)logit_softcap);

    engine->generation_phase.store(moe_GEN_PHASE_LM_HEAD, std::memory_order_relaxed);
    engine->generation_layer.store(0xffffffffu, std::memory_order_relaxed);

    static const unsigned int hw_raw_2 = std::thread::hardware_concurrency();
    static const unsigned int hw_threads_2 = hw_raw_2 > 0 ? hw_raw_2 : 4u;
    const uint32_t chunk_size = std::max(64u, (c->vocab_size + hw_threads_2 * 8 - 1) / (hw_threads_2 * 8));
    const uint32_t num_chunks = (c->vocab_size + chunk_size - 1) / chunk_size;

    struct chunk_logsum_t {
        float max_score;
        double exp_sum;
        float target_score;
        uint32_t best_token;
        float best_score;
        int has_target;
    };
    std::vector<chunk_logsum_t> chunk_results(num_chunks);

    {
        moe_profile_parallel_label_scope rows_label("lm_head_logprob");
        if (!moe_parallel_rows(engine, num_chunks, 0, [&](uint32_t chunk) {
            const uint32_t start = chunk * chunk_size;
            const uint32_t end = std::min(start + chunk_size, c->vocab_size);
            float local_max = -std::numeric_limits<float>::infinity();
            double local_sum = 0.0;
            float target_score = 0.0f;
            uint32_t local_best_token = start;
            float local_best_score = -std::numeric_limits<float>::infinity();
            int has_target = 0;

            for (uint32_t t = start; t < end; ++t) {
                const float score = moe_apply_final_logit_softcap_cached(
                    logit_softcap, moe_raw_tensor_dot_row(raw, t, hidden, hidden_count));
                if (score > local_best_score) {
                    local_best_score = score;
                    local_best_token = t;
                }
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

            chunk_results[chunk] = {local_max, local_sum, target_score, local_best_token, local_best_score, has_target};
            return 1;
        })) {
            return 0;
        }
    }

    float global_max = -std::numeric_limits<float>::infinity();
    double exp_sum = 0.0;
    float target_score = 0.0f;
    uint32_t best_token = 0;
    float best_score = -std::numeric_limits<float>::infinity();
    int found_target = 0;
    for (uint32_t i = 0; i < num_chunks; ++i) {
        if (!(chunk_results[i].exp_sum > 0.0) || !std::isfinite(chunk_results[i].max_score)) {
            continue;
        }
        if (chunk_results[i].best_score > best_score) {
            best_score = chunk_results[i].best_score;
            best_token = chunk_results[i].best_token;
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

    // BUGFIX 284: Add defensive check for exp_sum before log() ★
    // Problem: While loop checks chunk_results[i].exp_sum > 0.0, the accumulated exp_sum
    // can still become 0.0 due to floating point underflow in exp() operations
    // Solution: Check exp_sum > 0.0 and isfinite before log()
    // Impact: Prevents log(0) = -inf and subsequent NaN propagation
    if (!(exp_sum > 0.0) || !std::isfinite(exp_sum)) {
        return 0;
    }

    const double logsumexp = (double)global_max + std::log(exp_sum);
    const double logprob = (double)target_score - logsumexp;
    moe_profile_log(
        "lm_head_logprob_end",
        "target=%u target_score=%.6g best=%u best_score=%.6g global_max=%.6g logsumexp=%.6g logprob=%.6g softcap=%.6g",
        target_token_id,
        (double)target_score,
        best_token,
        (double)best_score,
        (double)global_max,
        logsumexp,
        logprob,
        (double)logit_softcap);
    moe_profile_log(
        "lm_head_logprob_time",
        "target=%u chunks=%u total_ms=%.3f",
        target_token_id,
        num_chunks,
        moe_profile_elapsed_ms(total_start_us));
    *out_target_logit = target_score;
    *out_logsumexp = (float)logsumexp;
    *out_logprob = (float)logprob;
    return std::isfinite(logprob) ? 1 : 0;
}
