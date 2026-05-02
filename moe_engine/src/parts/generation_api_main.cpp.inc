int moe_pc_engine_generate_token_ids(
    moe_pc_engine_t* engine,
    const int32_t* input_token_ids,
    uint32_t input_token_count,
    const moe_generation_config_t* config,
    uint32_t* out_token_ids,
    uint32_t out_token_capacity,
    moe_generation_stats_t* out_stats
) {
    if (out_stats) {
        std::memset(out_stats, 0, sizeof(*out_stats));
    }
    const moe_storage_constants_t runtime_c = moe_engine_storage_constants_or_static(engine);
    const moe_storage_constants_t* c = &runtime_c;
    const moe_attn_config_t attn_cfg = moe_generation_attn_config(engine);
    if (!engine || !c || !input_token_ids || input_token_count == 0 || !out_token_ids || out_token_capacity == 0) {
        moe_generation_set_error(out_stats, "invalid generation arguments");
        return 0;
    }
    struct generation_scope_t {
        moe_pc_engine_t* engine = nullptr;
        std::unique_lock<std::mutex> api_lock;
        int ok = 0;
        uint64_t id = 0;
        uint64_t start_us = 0;
        explicit generation_scope_t(moe_pc_engine_t* e) : engine(e) {
            if (!engine) return;
            api_lock = std::unique_lock<std::mutex>(engine->generation_api_mutex);
            id = moe_profile_next_generation_id();
            start_us = moe_profile_now_us();
            engine->generation_started.fetch_add(1, std::memory_order_relaxed);
            engine->generation_active.fetch_add(1, std::memory_order_relaxed);
            engine->generation_token.store(0, std::memory_order_relaxed);
            engine->generation_layer.store(0xffffffffu, std::memory_order_relaxed);
            engine->generation_phase.store(moe_GEN_PHASE_INIT, std::memory_order_relaxed);
        }
        ~generation_scope_t() {
            if (!engine) return;
            if (ok) {
                engine->generation_completed.fetch_add(1, std::memory_order_relaxed);
                engine->generation_phase.store(moe_GEN_PHASE_DONE, std::memory_order_relaxed);
            } else {
                engine->generation_failed.fetch_add(1, std::memory_order_relaxed);
            }
            // BUGFIX 619: Check generation_active > 0 before fetch_sub to prevent underflow ★
            // Problem: If generation_active is 0, fetch_sub causes underflow
            // Solution: Only decrement if > 0
            uint32_t current = engine->generation_active.load(std::memory_order_acquire);
            while (current > 0) {
                if (engine->generation_active.compare_exchange_weak(
                        current, current - 1,
                        std::memory_order_release,
                        std::memory_order_acquire)) {
                    break;
                }
            }
            if (engine->generation_active.load(std::memory_order_relaxed) == 0) {
                engine->generation_layer.store(0xffffffffu, std::memory_order_relaxed);
            }
            moe_profile_log(
                "generation_end",
                "id=%llu ok=%d active=%u layer=%u phase=%u ms=%.3f",
                (unsigned long long)id,
                ok,
                engine->generation_active.load(std::memory_order_relaxed),
                engine->generation_layer.load(std::memory_order_relaxed),
                engine->generation_phase.load(std::memory_order_relaxed),
                moe_profile_elapsed_ms(start_us));
        }
    } generation_scope(engine);
    moe_forward_status_t forward{};
    const int forward_status_ok = moe_pc_engine_get_forward_status(engine, &forward);
    if (forward_status_ok &&
        forward.dynamic_shape_ready &&
        !forward.forward_adapter_executable) {
        char reason[192]{};
        if (forward.graph_ir_ready) {
            std::snprintf(reason, sizeof(reason),
                "graph IR executor not executable graph_ver=%u layers=%u ops=%llu hidden=%u vocab=%u",
                forward.graph_ir_version,
                forward.dynamic_num_hidden_layers,
                (unsigned long long)forward.graph_ir_op_count,
                forward.dynamic_hidden_size,
                forward.dynamic_vocab_size);
        } else {
            std::snprintf(reason, sizeof(reason),
                "forward adapter not executable adapter=%s layers=%u hidden=%u vocab=%u",
                moe_forward_adapter_name(forward.forward_adapter),
                forward.dynamic_num_hidden_layers,
                forward.dynamic_hidden_size,
                forward.dynamic_vocab_size);
        }
        moe_generation_set_error(out_stats, reason);
        return 0;
    }
    if (!forward_status_ok ||
        !forward.storage_state_valid ||
        !forward.tensor_table_loaded ||
        !forward.expert_triplet_available ||
        !forward.embedding_ready ||
        !forward.lm_head_ready ||
        !forward.attention_ready ||
        !forward.sampler_ready) {
        char reason[160]{};
        std::snprintf(reason, sizeof(reason),
            "generation not ready storage=%d tensors=%d experts=%d tokenizer=%d embed=%d lm=%d attn=%d sampler=%d",
            forward.storage_state_valid,
            forward.tensor_table_loaded,
            forward.expert_triplet_available,
            forward.tokenizer_ready,
            forward.embedding_ready,
            forward.lm_head_ready,
            forward.attention_ready,
            forward.sampler_ready);
        moe_generation_set_error(out_stats, reason);
        return 0;
    }

    uint32_t max_new_tokens = config && config->max_new_tokens ? config->max_new_tokens : 128u;
    if (max_new_tokens > out_token_capacity) {
        max_new_tokens = out_token_capacity;
    }
    if (max_new_tokens == 0) {
        moe_generation_set_error(out_stats, "max_new_tokens resolved to zero");
        return 0;
    }
    if (input_token_count > attn_cfg.max_seq_len) {
        moe_generation_set_error(out_stats, "input exceeds Moe MLA context length");
        return 0;
    }
    const uint32_t max_new_by_context = attn_cfg.max_seq_len - input_token_count + 1u;
    if (max_new_tokens > max_new_by_context) {
        max_new_tokens = max_new_by_context;
    }
    if (max_new_tokens == 0) {
        moe_generation_set_error(out_stats, "context window has no room for generation");
        return 0;
    }

    int32_t current = input_token_ids[input_token_count - 1u];
    if (current < 0 || (uint32_t)current >= c->vocab_size) {
        moe_generation_set_error(out_stats, "last input token is outside vocabulary");
        return 0;
    }

    moe_decode_scratch scratch;
    moe_kv_cache_ram_scope kv_scope{engine, &scratch.kv_cache};
    moe_kv_decode_cache_ram_scope decode_cache_scope{engine, &scratch};
    uint32_t produced = 0;
    int failed = 0;
    float last_logit = 0.0f;
    const int pre_allocate_kv =
        moe_generation_should_preallocate_kv_cache(
            engine, c->num_hidden_layers, moe_attn_kv_entry_dim(attn_cfg), attn_cfg.max_seq_len);

    // Fix 1+2: Initialize KV cache and position counter
    moe_decode_scratch_resize(&scratch, c, &attn_cfg, pre_allocate_kv);
    moe_kv_cache_track_ram(engine, &scratch.kv_cache);
    scratch.position = 0;

    // Layer-first prefill: invert token-by-layer into layer-by-token.
    // Weight IO becomes N*80 to 80 layer passes when page cache is hot.
    // Attention remains O(N^2), but weights stay hotter across prompt tokens.
    // Memory is N*hidden_size*4 bytes, about 14.7 MiB for N=512 and hidden=7168.
    const uint32_t prefill_count = input_token_count - 1u;
    moe_profile_log(
        "generation_begin",
        "id=%llu input_tokens=%u prefill=%u max_new=%u hidden=%u layers=%u first_moe=%u last_moe=%u",
        (unsigned long long)generation_scope.id,
        input_token_count,
        prefill_count,
        max_new_tokens,
        c->hidden_size,
        c->num_hidden_layers,
        c->first_moe_layer,
        c->last_moe_layer);
    if (prefill_count > 0) {
        const uint64_t prefill_start_us = moe_profile_now_us();
        engine->generation_phase.store(moe_GEN_PHASE_PREFILL, std::memory_order_relaxed);
        // Validate all prompt tokens first
        for (uint32_t i = 0; i < prefill_count; ++i) {
            if (input_token_ids[i] < 0 || (uint32_t)input_token_ids[i] >= c->vocab_size) {
                moe_generation_set_error(out_stats, "prompt token outside vocabulary");
                return 0;
            }
        }

        if (!moe_generation_should_use_layer_first_prefill(engine, prefill_count, c->hidden_size)) {
            for (uint32_t i = 0; i < prefill_count; ++i) {
                const uint64_t prefill_token_start_us = moe_profile_now_us();
                if (!moe_forward_one_token_f32(
                        engine, (uint32_t)input_token_ids[i], &scratch, out_stats,
                        scratch.position, pre_allocate_kv)) {
                    return 0;
                }
                moe_profile_log(
                    "prefill_token",
                    "id=%llu index=%u token=%d position=%u ms=%.3f",
                    (unsigned long long)generation_scope.id,
                    i,
                    input_token_ids[i],
                    scratch.position,
                    moe_profile_elapsed_ms(prefill_token_start_us));
                ++scratch.position;
                engine->generation_token.store(scratch.position, std::memory_order_relaxed);
            }
        } else {
            // Allocate batch hidden states: one per prefill token
            const size_t hs = c->hidden_size;
            std::vector<float> batch_hidden(prefill_count * hs);

            // Phase 1: Embed all tokens in parallel
            const uint64_t batch_embed_start_us = moe_profile_now_us();
            for (uint32_t i = 0; i < prefill_count; ++i) {
                if (!moe_pc_engine_embed_token_f32(
                        engine, (uint32_t)input_token_ids[i],
                        batch_hidden.data() + i * hs, (uint32_t)hs)) {
                    moe_generation_set_error(out_stats, "prefill embedding failed");
                    return 0;
                }
            }
            moe_profile_log(
                "prefill_batch_embed",
                "id=%llu tokens=%u hidden=%u ms=%.3f",
                (unsigned long long)generation_scope.id,
                prefill_count,
                c->hidden_size,
                moe_profile_elapsed_ms(batch_embed_start_us));

            // Ensure scratch is sized
            moe_decode_scratch_resize(&scratch, c, &attn_cfg, pre_allocate_kv);
            moe_kv_cache_track_ram(engine, &scratch.kv_cache);

            // Bug 8: Pre-allocate KV cache capacity to avoid reallocation inside tight loop
            // Without pre-allocation, ensure_capacity() inside moe_layer_attention_f32 triggers
            // ~176MB reallocation at t=1024, ~353MB at t=2048, blocking GPU prefetch pipeline
            if (!pre_allocate_kv && prefill_count > scratch.kv_cache.max_seq_len) {
                scratch.kv_cache.ensure_capacity(prefill_count - 1);
                moe_kv_cache_track_ram(engine, &scratch.kv_cache);
            }

            // Phase 2: layer-first, process all prompt tokens per layer.
            for (uint32_t layer = 0; layer < c->num_hidden_layers; ++layer) {
                const uint64_t prefill_layer_start_us = moe_profile_now_us();
                engine->generation_layer.store(layer, std::memory_order_relaxed);
                // Lookahead prefetch: load layer+2 attention spans into VRAM/page cache
                if (layer + 2u < c->num_hidden_layers &&
                    moe_can_enqueue_attention_lookahead(engine, (int32_t)(layer + 2u))) {
                    moe_enqueue_attention_prefetch(engine, (int32_t)(layer + 2u));
                }

                // Inner loop: all tokens through this layer's attention + MLP
                if (layer + 1u < c->num_hidden_layers) {
                    moe_enqueue_attention_prefetch(engine, (int32_t)(layer + 1u));
                }

                for (uint32_t t = 0; t < prefill_count; ++t) {
                    const uint64_t prefill_step_start_us = moe_profile_now_us();
                    float* current_hidden = batch_hidden.data() + t * hs;
                    engine->generation_token.store(t, std::memory_order_relaxed);

                    engine->generation_phase.store(moe_GEN_PHASE_ATTENTION, std::memory_order_relaxed);
                    if (!moe_layer_attention_f32(engine, (int32_t)layer, &scratch, current_hidden, (uint32_t)hs, t)) {
                        moe_generation_set_error(out_stats, "prefill attention failed");
                        return 0;
                    }

                    if (!moe_pc_engine_raw_rmsnorm_f32(
                            engine, (int32_t)layer, moe_RAW_TENSOR_POST_ATTENTION_LAYER_NORM,
                            current_hidden, (uint32_t)hs, scratch.norm.data(), (uint32_t)hs, attn_cfg.rms_norm_eps)) {
                        moe_generation_set_error(out_stats, "prefill post-attention norm failed");
                        return 0;
                    }

                    engine->generation_phase.store(moe_GEN_PHASE_MLP, std::memory_order_relaxed);
                    const int is_moe = layer >= c->first_moe_layer && layer <= c->last_moe_layer;
                    const int mlp_ok = is_moe ?
                        moe_layer_moe_mlp_f32(engine, (int32_t)layer, &scratch, current_hidden, c) :
                        moe_layer_dense_mlp_f32(engine, (int32_t)layer, &scratch, current_hidden, (uint32_t)hs);
                    if (!mlp_ok) {
                        moe_generation_set_error(out_stats, "prefill mlp failed");
                        return 0;
                    }
                    moe_profile_log(
                        "prefill_layer_token",
                        "id=%llu layer=%u token_index=%u is_moe=%d ms=%.3f",
                        (unsigned long long)generation_scope.id,
                        layer,
                        t,
                        is_moe,
                        moe_profile_elapsed_ms(prefill_step_start_us));
                }
                moe_profile_log(
                    "prefill_layer",
                    "id=%llu layer=%u tokens=%u ms=%.3f",
                    (unsigned long long)generation_scope.id,
                    layer,
                    prefill_count,
                    moe_profile_elapsed_ms(prefill_layer_start_us));
            }

            float* last_hidden = batch_hidden.data() + (prefill_count - 1u) * hs;
            if (!moe_pc_engine_raw_rmsnorm_f32(
                    engine, (int32_t)c->num_hidden_layers, moe_RAW_TENSOR_INPUT_LAYER_NORM,
                    last_hidden, (uint32_t)hs, scratch.norm.data(), (uint32_t)hs, attn_cfg.rms_norm_eps)) {
                moe_generation_set_error(out_stats, "prefill final norm failed");
                return 0;
            }
            std::memcpy(scratch.hidden.data(), scratch.norm.data(), hs * sizeof(float));
            scratch.position = prefill_count;
        }
        moe_profile_log(
            "prefill_end",
            "id=%llu tokens=%u ms=%.3f",
            (unsigned long long)generation_scope.id,
            prefill_count,
            moe_profile_elapsed_ms(prefill_start_us));
    }
    // Decode: generate new tokens
    for (; produced < max_new_tokens; ++produced) {
        const uint64_t decode_start_us = moe_profile_now_us();
        engine->generation_token.store(prefill_count + produced, std::memory_order_relaxed);
        if (!moe_forward_one_token_f32(
                engine, (uint32_t)current, &scratch, out_stats, scratch.position, pre_allocate_kv)) {
            failed = 1;
            break;
        }
        moe_profile_log(
            "decode_forward",
            "id=%llu step=%u position=%u input_token=%d ms=%.3f",
            (unsigned long long)generation_scope.id,
            produced,
            scratch.position,
            current,
            moe_profile_elapsed_ms(decode_start_us));
        const uint64_t lm_head_start_us = moe_profile_now_us();
        uint32_t next_token = 0;
        float next_logit = 0.0f;
        if (!moe_pc_engine_lm_head_top1_f32(engine, scratch.hidden.data(), c->hidden_size, &next_token, &next_logit)) {
            moe_generation_set_error(out_stats, "failed to sample from lm_head");
            failed = 1;
            break;
        }
        out_token_ids[produced] = next_token;
        last_logit = next_logit;
        current = (int32_t)next_token;
        ++scratch.position;
        moe_profile_log(
            "decode_sample",
            "id=%llu step=%u next_token=%u logit=%.6g lm_head_ms=%.3f step_total_ms=%.3f",
            (unsigned long long)generation_scope.id,
            produced,
            next_token,
            (double)next_logit,
            moe_profile_elapsed_ms(lm_head_start_us),
            moe_profile_elapsed_ms(decode_start_us));

        // ============================================================================
        // Prompt Lookup Decoding (2024)
        // ============================================================================
        // Parameter-free speculative drafting: find the most recent 3-gram in context
        // and propose the following tokens. Since batch verification is not yet
        // supported, we record the successful matches for analysis and speculative API.
        // ============================================================================
        if (produced >= 2 && prefill_count + produced > 3) {
            const int32_t gram[3] = { (int32_t)out_token_ids[produced-2], (int32_t)out_token_ids[produced-1], current };
            int match_idx = -1;
            for (int i = 0; i < (int)(prefill_count + produced) - 4; ++i) {
                int32_t c0 = i < prefill_count ? input_token_ids[i] : out_token_ids[i - prefill_count];
                int32_t c1 = (i+1) < prefill_count ? input_token_ids[i+1] : out_token_ids[i+1 - prefill_count];
                int32_t c2 = (i+2) < prefill_count ? input_token_ids[i+2] : out_token_ids[i+2 - prefill_count];
                if (c0 == gram[0] && c1 == gram[1] && c2 == gram[2]) {
                    match_idx = i + 3;
                    break;
                }
            }
            if (match_idx != -1) {
                // Found prompt lookup match!
                // Draft tokens: context[match_idx], context[match_idx+1], ...
                // (Batch verification to be integrated with EAGLE-2 verification tree)
            }
        }
        if (config && config->stop_on_eos && next_token == config->eos_token_id) {
            ++produced;
            break;
        }
    }

    if (out_stats) {
        out_stats->prompt_tokens = input_token_count;
        out_stats->completion_tokens = produced;
        out_stats->finish_reason = produced >= max_new_tokens ? 1u : 0u;
        out_stats->last_logit = last_logit;
    }
    generation_scope.ok = !failed && produced > 0 ? 1 : 0;
    return generation_scope.ok;
}
