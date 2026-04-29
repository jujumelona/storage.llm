int glm5_pc_engine_generate_token_ids(
    glm5_pc_engine_t* engine,
    const int32_t* input_token_ids,
    uint32_t input_token_count,
    const glm5_generation_config_t* config,
    uint32_t* out_token_ids,
    uint32_t out_token_capacity,
    glm5_generation_stats_t* out_stats
) {
    if (out_stats) {
        std::memset(out_stats, 0, sizeof(*out_stats));
    }
    const glm5_storage_constants_t* c = glm5_storage_constants();
    if (!engine || !c || !input_token_ids || input_token_count == 0 || !out_token_ids || out_token_capacity == 0) {
        glm5_generation_set_error(out_stats, "invalid generation arguments");
        return 0;
    }
    glm5_forward_status_t forward{};
    if (!glm5_pc_engine_get_forward_status(engine, &forward) ||
        !forward.storage_state_valid ||
        !forward.tensor_table_loaded ||
        !forward.expert_triplet_available ||
        !forward.embedding_ready ||
        !forward.lm_head_ready ||
        !forward.attention_ready ||
        !forward.sampler_ready) {
        glm5_generation_set_error(out_stats, "full generation path is not ready");
        return 0;
    }

    uint32_t max_new_tokens = config && config->max_new_tokens ? config->max_new_tokens : 128u;
    if (max_new_tokens > out_token_capacity) {
        max_new_tokens = out_token_capacity;
    }
    if (max_new_tokens == 0) {
        glm5_generation_set_error(out_stats, "max_new_tokens resolved to zero");
        return 0;
    }
    if (input_token_count > GLM5_MLA_MAX_SEQ_LEN) {
        glm5_generation_set_error(out_stats, "input exceeds GLM5 MLA context length");
        return 0;
    }
    const uint32_t max_new_by_context = GLM5_MLA_MAX_SEQ_LEN - input_token_count + 1u;
    if (max_new_tokens > max_new_by_context) {
        max_new_tokens = max_new_by_context;
    }
    if (max_new_tokens == 0) {
        glm5_generation_set_error(out_stats, "context window has no room for generation");
        return 0;
    }

    int32_t current = input_token_ids[input_token_count - 1u];
    if (current < 0 || (uint32_t)current >= c->vocab_size) {
        glm5_generation_set_error(out_stats, "last input token is outside vocabulary");
        return 0;
    }

    glm5_decode_scratch scratch;
    glm5_kv_cache_ram_scope kv_scope{engine, &scratch.kv_cache};
    glm5_kv_decode_cache_ram_scope decode_cache_scope{engine, &scratch};
    uint32_t produced = 0;
    int failed = 0;
    float last_logit = 0.0f;
    const int pre_allocate_kv =
        glm5_generation_should_preallocate_kv_cache(engine, c->num_hidden_layers);

    // Fix 1+2: Initialize KV cache and position counter
    glm5_decode_scratch_resize(&scratch, c, pre_allocate_kv);
    glm5_kv_cache_track_ram(engine, &scratch.kv_cache);
    scratch.position = 0;

    // Layer-first prefill: invert token-by-layer into layer-by-token.
    // Weight IO becomes N*80 to 80 layer passes when page cache is hot.
    // Attention remains O(N^2), but weights stay hotter across prompt tokens.
    // Memory is N*hidden_size*4 bytes, about 14.7 MiB for N=512 and hidden=7168.
    const uint32_t prefill_count = input_token_count - 1u;
    if (prefill_count > 0) {
        // Validate all prompt tokens first
        for (uint32_t i = 0; i < prefill_count; ++i) {
            if (input_token_ids[i] < 0 || (uint32_t)input_token_ids[i] >= c->vocab_size) {
                glm5_generation_set_error(out_stats, "prompt token outside vocabulary");
                return 0;
            }
        }

        if (!glm5_generation_should_use_layer_first_prefill(engine, prefill_count, c->hidden_size)) {
            for (uint32_t i = 0; i < prefill_count; ++i) {
                if (!glm5_forward_one_token_f32(
                        engine, (uint32_t)input_token_ids[i], &scratch, out_stats,
                        scratch.position, pre_allocate_kv)) {
                    return 0;
                }
                ++scratch.position;
            }
        } else {
            // Allocate batch hidden states: one per prefill token
            const size_t hs = c->hidden_size;
            std::vector<float> batch_hidden(prefill_count * hs);

            // Phase 1: Embed all tokens in parallel
            for (uint32_t i = 0; i < prefill_count; ++i) {
                if (!glm5_pc_engine_embed_token_f32(
                        engine, (uint32_t)input_token_ids[i],
                        batch_hidden.data() + i * hs, (uint32_t)hs)) {
                    glm5_generation_set_error(out_stats, "prefill embedding failed");
                    return 0;
                }
            }

            // Ensure scratch is sized
            glm5_decode_scratch_resize(&scratch, c, pre_allocate_kv);
            glm5_kv_cache_track_ram(engine, &scratch.kv_cache);

            // Bug 8: Pre-allocate KV cache capacity to avoid reallocation inside tight loop
            // Without pre-allocation, ensure_capacity() inside glm5_layer_attention_f32 triggers
            // ~176MB reallocation at t=1024, ~353MB at t=2048, blocking GPU prefetch pipeline
            if (!pre_allocate_kv && prefill_count > scratch.kv_cache.max_seq_len) {
                scratch.kv_cache.ensure_capacity(prefill_count - 1);
                glm5_kv_cache_track_ram(engine, &scratch.kv_cache);
            }

            // Phase 2: layer-first, process all prompt tokens per layer.
            for (uint32_t layer = 0; layer < c->num_hidden_layers; ++layer) {
                // Lookahead prefetch: load layer+2 attention spans into VRAM/page cache
                if (layer + 2u < c->num_hidden_layers &&
                    glm5_can_enqueue_attention_lookahead(engine, (int32_t)(layer + 2u))) {
                    glm5_enqueue_attention_prefetch(engine, (int32_t)(layer + 2u));
                }

                // Inner loop: all tokens through this layer's attention + MLP
                if (layer + 1u < c->num_hidden_layers) {
                    glm5_enqueue_attention_prefetch(engine, (int32_t)(layer + 1u));
                }

                for (uint32_t t = 0; t < prefill_count; ++t) {
                    float* current_hidden = batch_hidden.data() + t * hs;

                    if (!glm5_layer_attention_f32(engine, (int32_t)layer, &scratch, current_hidden, (uint32_t)hs, t)) {
                        glm5_generation_set_error(out_stats, "prefill attention failed");
                        return 0;
                    }

                    if (!glm5_pc_engine_raw_rmsnorm_f32(
                            engine, (int32_t)layer, GLM5_RAW_TENSOR_POST_ATTENTION_LAYER_NORM,
                            current_hidden, (uint32_t)hs, scratch.norm.data(), (uint32_t)hs, 1.0e-6f)) {
                        glm5_generation_set_error(out_stats, "prefill post-attention norm failed");
                        return 0;
                    }

                    const int is_moe = layer >= c->first_moe_layer && layer <= c->last_moe_layer;
                    const int mlp_ok = is_moe ?
                        glm5_layer_moe_mlp_f32(engine, (int32_t)layer, &scratch, current_hidden, c) :
                        glm5_layer_dense_mlp_f32(engine, (int32_t)layer, &scratch, current_hidden, (uint32_t)hs);
                    if (!mlp_ok) {
                        glm5_generation_set_error(out_stats, "prefill mlp failed");
                        return 0;
                    }
                }
            }

            float* last_hidden = batch_hidden.data() + (prefill_count - 1u) * hs;
            if (!glm5_pc_engine_raw_rmsnorm_f32(
                    engine, (int32_t)c->num_hidden_layers, GLM5_RAW_TENSOR_INPUT_LAYER_NORM,
                    last_hidden, (uint32_t)hs, scratch.norm.data(), (uint32_t)hs, 1.0e-6f)) {
                glm5_generation_set_error(out_stats, "prefill final norm failed");
                return 0;
            }
            std::memcpy(scratch.hidden.data(), scratch.norm.data(), hs * sizeof(float));
            scratch.position = prefill_count;
        }
    }
    // Decode: generate new tokens
    for (; produced < max_new_tokens; ++produced) {
        if (!glm5_forward_one_token_f32(
                engine, (uint32_t)current, &scratch, out_stats, scratch.position, pre_allocate_kv)) {
            failed = 1;
            break;
        }
        uint32_t next_token = 0;
        float next_logit = 0.0f;
        if (!glm5_pc_engine_lm_head_top1_f32(engine, scratch.hidden.data(), c->hidden_size, &next_token, &next_logit)) {
            glm5_generation_set_error(out_stats, "failed to sample from lm_head");
            failed = 1;
            break;
        }
        out_token_ids[produced] = next_token;
        last_logit = next_logit;
        current = (int32_t)next_token;
        ++scratch.position;
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
    return !failed && produced > 0 ? 1 : 0;
}
