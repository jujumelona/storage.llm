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
    // Fix 35: Cache lm_head mapping — tensor is immutable after model load
    // Eliminates per-token name lookup + mmap pointer recalculation
    static thread_local uint64_t tls_lm_head_engine_generation = 0;
    static thread_local moe_raw_tensor_cache_entry tls_lm_head_entry{};
    static thread_local bool tls_lm_head_valid = false;
    const uint64_t engine_generation = engine->engine_generation;

    moe_raw_tensor_cache_entry common{};
    const uint8_t* base = nullptr;
    uint64_t bytes = 0;
    uint8_t encoding = moe_RAW_SCALAR_BF16;
    uint8_t weight_format = moe_WEIGHT_ENCODING_RAW_FP16;
    uint64_t weight_row_bytes = 0;
    if (tls_lm_head_engine_generation == engine_generation && tls_lm_head_valid) {
        common = tls_lm_head_entry;
    } else if (moe_map_lm_head_tensor(engine, c, &common)) {
        tls_lm_head_engine_generation = engine_generation;
        tls_lm_head_entry = common;
        tls_lm_head_valid = true;
    }
    if (tls_lm_head_valid && tls_lm_head_engine_generation == engine_generation) {
        base = common.base;
        bytes = common.bytes;
        encoding = common.scalar_encoding;
        weight_format = common.weight_format;
        weight_row_bytes = common.weight_row_bytes;
    } else {
        if (engine->offload_gguf_valid.load(std::memory_order_relaxed)) {
            moe_profile_log("lm_head_common_tensor_missing_offload", "vocab=%u hidden=%u", c->vocab_size, c->hidden_size);
            return 0;
        }
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

    static const uint32_t hw_threads = std::max<uint32_t>(moe_cpu_thread_count_hint(), 4u);
    const uint32_t chunk_size = std::max(64u, (c->vocab_size + hw_threads * 8 - 1) / (hw_threads * 8));
    const uint32_t num_chunks = (c->vocab_size + chunk_size - 1) / chunk_size;
    const uint32_t chunk_block =
        std::max(1u, (num_chunks + (uint32_t)hw_threads - 1u) / (uint32_t)hw_threads);

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
        if (!moe_parallel_rows(engine, num_chunks, chunk_block, [&](uint32_t chunk) {
            const uint32_t start = chunk * chunk_size;
            const uint32_t end = std::min(start + chunk_size, c->vocab_size);
            uint32_t local_best = start;
            float local_max = -std::numeric_limits<float>::infinity();

            // Fix 15: Vectorize argmax search to reduce loop bottleneck
            constexpr uint32_t batch = 32;
            uint32_t t = start;
            for (; t + batch <= end; t += batch) {
                alignas(32) float scores[batch];
                for (uint32_t i = 0; i < batch; ++i) {
                    scores[i] = moe_apply_final_logit_softcap_cached(
                        logit_softcap, moe_raw_tensor_dot_row(raw, t + i, hidden, hidden_count));
                }
#if defined(__AVX512F__)
                __m512 v_max = _mm512_set1_ps(local_max);
                __m512i v_best = _mm512_set1_epi32(local_best);
                __m512i v_idx = _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
                for(int k=0; k<2; ++k) {
                    __m512 v_score = _mm512_loadu_ps(scores + k*16);
                    __mmask16 mask = _mm512_cmp_ps_mask(v_score, v_max, _CMP_GT_OQ);
                    v_max = _mm512_mask_blend_ps(mask, v_max, v_score);
                    __m512i curr_idx = _mm512_add_epi32(v_idx, _mm512_set1_epi32(t + k*16));
                    v_best = _mm512_mask_blend_epi32(mask, v_best, curr_idx);
                }
                float max_arr[16];
                uint32_t best_arr[16];
                _mm512_storeu_ps(max_arr, v_max);
                _mm512_storeu_epi32(best_arr, v_best);
                for(int i=0; i<16; ++i) {
                    if (max_arr[i] > local_max) {
                        local_max = max_arr[i];
                        local_best = best_arr[i];
                    }
                }
#elif defined(__AVX2__)
                __m256 v_max = _mm256_set1_ps(local_max);
                __m256i v_best = _mm256_set1_epi32(local_best);
                __m256i v_idx = _mm256_set_epi32(7,6,5,4,3,2,1,0);
                for(int k=0; k<4; ++k) {
                    __m256 v_score = _mm256_load_ps(scores + k*8);
                    __m256 v_cmp = _mm256_cmp_ps(v_score, v_max, _CMP_GT_OQ);
                    v_max = _mm256_blendv_ps(v_max, v_score, v_cmp);
                    __m256i curr_idx = _mm256_add_epi32(v_idx, _mm256_set1_epi32(t + k*8));
                    v_best = _mm256_blendv_epi8(v_best, curr_idx, _mm256_castps_si256(v_cmp));
                }
                float max_arr[8];
                uint32_t best_arr[8];
                _mm256_storeu_ps(max_arr, v_max);
                _mm256_storeu_si256((__m256i*)best_arr, v_best);
                for(int i=0; i<8; ++i) {
                    if (max_arr[i] > local_max) {
                        local_max = max_arr[i];
                        local_best = best_arr[i];
                    }
                }
#else
                for (uint32_t i = 0; i < batch; ++i) {
                    if (scores[i] > local_max) {
                        local_max = scores[i];
                        local_best = t + i;
                    }
                }
#endif
            }
            for (; t < end; ++t) {
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
    // Fix 35: Reuse cached lm_head mapping (same tls declared in top1)
    {
        static thread_local const moe_pc_engine* tls_lp_engine = nullptr;
        static thread_local moe_raw_tensor_cache_entry tls_lp_entry{};
        static thread_local bool tls_lp_valid = false;
        if (tls_lp_engine == engine && tls_lp_valid) {
            common = tls_lp_entry;
        } else if (moe_map_lm_head_tensor(engine, c, &common)) {
            tls_lp_engine = engine;
            tls_lp_entry = common;
            tls_lp_valid = true;
        }
    }
    if (common.base) {
        base = common.base;
        bytes = common.bytes;
        encoding = common.scalar_encoding;
        weight_format = common.weight_format;
        weight_row_bytes = common.weight_row_bytes;
    } else {
        if (engine->offload_gguf_valid.load(std::memory_order_relaxed)) {
            moe_profile_log("lm_head_common_tensor_missing_offload", "vocab=%u hidden=%u", c->vocab_size, c->hidden_size);
            return 0;
        }
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

    static const uint32_t hw_threads_2 = std::max<uint32_t>(moe_cpu_thread_count_hint(), 4u);
    const uint32_t chunk_size = std::max(64u, (c->vocab_size + hw_threads_2 * 8 - 1) / (hw_threads_2 * 8));
    const uint32_t num_chunks = (c->vocab_size + chunk_size - 1) / chunk_size;
    const uint32_t chunk_block =
        std::max(1u, (num_chunks + (uint32_t)hw_threads_2 - 1u) / (uint32_t)hw_threads_2);

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
        if (!moe_parallel_rows(engine, num_chunks, chunk_block, [&](uint32_t chunk) {
            const uint32_t start = chunk * chunk_size;
            const uint32_t end = std::min(start + chunk_size, c->vocab_size);
            float local_max = -std::numeric_limits<float>::infinity();
            double local_sum = 0.0;
            float target_score = 0.0f;
            uint32_t local_best_token = start;
            float local_best_score = -std::numeric_limits<float>::infinity();
            int has_target = 0;

            // Fix 15: Vectorize loop to improve cache hits and enable auto-vectorization
            constexpr uint32_t batch = 32;
            uint32_t t = start;
            for (; t + batch <= end; t += batch) {
                alignas(32) float scores[batch];
                for (uint32_t i = 0; i < batch; ++i) {
                    scores[i] = moe_apply_final_logit_softcap_cached(
                        logit_softcap, moe_raw_tensor_dot_row(raw, t + i, hidden, hidden_count));
                }
                for (uint32_t i = 0; i < batch; ++i) {
                    const float score = scores[i];
                    if (score > local_best_score) {
                        local_best_score = score;
                        local_best_token = t + i;
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
                    if (t + i == target_token_id) {
                        target_score = score;
                        has_target = 1;
                    }
                }
            }
            for (; t < end; ++t) {
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
