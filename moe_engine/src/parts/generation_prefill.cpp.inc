static int moe_prefill_prompt_f32(
    moe_pc_engine_t* engine,
    const moe_storage_constants_t* c,
    const moe_attn_config_t& attn_cfg,
    const int32_t* input_token_ids,
    uint32_t prefill_count,
    uint32_t eos_token_id,
    moe_decode_scratch* scratch,
    moe_generation_stats_t* stats,
    int pre_allocate_kv,
    int qkv_only,
    uint64_t profile_id,
    const char* profile_prefix
) {
    if (!engine || !c || !input_token_ids || !scratch || prefill_count == 0) {
        return 1;
    }
    const uint64_t prefill_start_us = moe_profile_now_us();
    const char* prefix = profile_prefix ? profile_prefix : "prefill";
    engine->generation_phase.store(moe_GEN_PHASE_PREFILL, std::memory_order_relaxed);

    std::vector<int32_t> safe_input(prefill_count);
    for (uint32_t i = 0; i < prefill_count; ++i) {
        if (input_token_ids[i] < 0 || (uint32_t)input_token_ids[i] >= c->vocab_size) {
            safe_input[i] = eos_token_id ? (int32_t)eos_token_id : 0;
        } else {
            safe_input[i] = input_token_ids[i];
        }
    }

    if (!moe_generation_should_use_layer_first_prefill(engine, prefill_count, c->hidden_size)) {
        for (uint32_t i = 0; i < prefill_count; ++i) {
            const uint64_t token_start_us = moe_profile_now_us();
            if (!moe_forward_one_token_dispatch_f32(
                    engine, (uint32_t)safe_input[i], scratch, stats,
                    scratch->position, pre_allocate_kv)) {
                return 0;
            }
            moe_profile_log(
                "prefill_token",
                "scope=%s id=%llu index=%u token=%d position=%u ms=%.3f",
                prefix,
                (unsigned long long)profile_id,
                i,
                safe_input[i],
                scratch->position,
                moe_profile_elapsed_ms(token_start_us));
            ++scratch->position;
            engine->generation_token.store(scratch->position, std::memory_order_relaxed);
        }
        moe_profile_log(
            "prefill_end",
            "scope=%s id=%llu tokens=%u layer_first=0 ms=%.3f",
            prefix,
            (unsigned long long)profile_id,
            prefill_count,
            moe_profile_elapsed_ms(prefill_start_us));
        return 1;
    }

    const size_t hs = c->hidden_size;
    std::vector<float> batch_hidden((size_t)prefill_count * hs);
    std::vector<float> batch_norm((size_t)prefill_count * hs);

    const uint64_t batch_embed_start_us = moe_profile_now_us();
    std::atomic<int> embed_ok{1};
    if (prefill_count >= 2u && moe_generation_parallel_prefill_embed_enabled()) {
        const int rows_ok = moe_parallel_rows(engine, prefill_count, 0, [&](uint32_t i) {
            if (!moe_pc_engine_embed_token_f32(
                    engine, (uint32_t)safe_input[i],
                    batch_hidden.data() + (size_t)i * hs, (uint32_t)hs)) {
                embed_ok.store(0, std::memory_order_relaxed);
            }
        });
        if (!rows_ok) embed_ok.store(0, std::memory_order_relaxed);
    } else {
        for (uint32_t i = 0; i < prefill_count; ++i) {
            if (!moe_pc_engine_embed_token_f32(
                    engine, (uint32_t)safe_input[i],
                    batch_hidden.data() + (size_t)i * hs, (uint32_t)hs)) {
                embed_ok.store(0, std::memory_order_relaxed);
                break;
            }
        }
    }
    if (!embed_ok.load(std::memory_order_relaxed)) {
        moe_generation_set_error(stats, "prefill embedding failed");
        return 0;
    }
    moe_profile_log(
        "prefill_batch_embed",
        "scope=%s id=%llu tokens=%u hidden=%u parallel=%d ms=%.3f",
        prefix,
        (unsigned long long)profile_id,
        prefill_count,
        c->hidden_size,
        (prefill_count >= 2u && moe_generation_parallel_prefill_embed_enabled()) ? 1 : 0,
        moe_profile_elapsed_ms(batch_embed_start_us));

    if (!moe_decode_scratch_resize(scratch, c, &attn_cfg, pre_allocate_kv, engine)) {
        moe_generation_set_error(stats, "prefill scratch allocation failed");
        return 0;
    }
    if (!qkv_only) {
        moe_kv_cache_track_ram(engine, &scratch->kv_cache);
    }
    if (!pre_allocate_kv) {
        if (qkv_only) {
            if (!moe_qkv_only_ensure_capacity(engine, scratch, attn_cfg, prefill_count - 1u, 0)) {
                moe_generation_set_error(stats, "prefill QKV capacity allocation failed");
                return 0;
            }
        } else if (prefill_count > scratch->kv_cache.max_seq_len) {
            if (!scratch->kv_cache.ensure_capacity(prefill_count - 1)) {
                moe_generation_set_error(stats, "prefill KV cache allocation failed");
                return 0;
            }
            moe_kv_cache_track_ram(engine, &scratch->kv_cache);
        }
    }

    for (uint32_t layer = 0; layer < c->num_hidden_layers; ++layer) {
        const uint64_t layer_start_us = moe_profile_now_us();
        engine->generation_layer.store(layer, std::memory_order_relaxed);
        if (layer + 2u < c->num_hidden_layers &&
            moe_can_enqueue_attention_lookahead(engine, (int32_t)(layer + 2u))) {
            moe_enqueue_attention_prefetch(engine, (int32_t)(layer + 2u));
        }
        if (layer + 1u < c->num_hidden_layers) {
            moe_enqueue_attention_prefetch(engine, (int32_t)(layer + 1u));
        }

        const moe_attn_config_t layer_attn_cfg =
            moe_generation_layer_attn_config(engine, (int32_t)layer);
        const moe_decoder_layer_plan_t layer_plan =
            moe_build_decoder_layer_plan(engine, c, &attn_cfg, layer);
        for (uint32_t t = 0; t < prefill_count; ++t) {
            const uint64_t attention_start_us = moe_profile_now_us();
            float* current_hidden = batch_hidden.data() + (size_t)t * hs;
            engine->generation_token.store(t, std::memory_order_relaxed);
            engine->generation_phase.store(moe_GEN_PHASE_ATTENTION, std::memory_order_relaxed);
            if (layer_attn_cfg.type != moe_ATTN_TYPE_MLA) {
                std::memcpy(scratch->residual.data(), current_hidden, hs * sizeof(float));
                if (!moe_layer_standard_attention_f32(
                        engine, (int32_t)layer, scratch, current_hidden, (uint32_t)hs, t,
                        layer_attn_cfg, nullptr, 0)) {
                    moe_generation_set_error(stats, "prefill attention failed");
                    return 0;
                }
                if (!moe_apply_attention_residual_contract_f32(
                        engine, (int32_t)layer, layer_plan,
                        scratch->residual.data(), scratch->attn.data(), current_hidden,
                        scratch->norm.data(), scratch->shared.data(),
                        (uint32_t)hs, layer_attn_cfg.rms_norm_eps)) {
                    moe_generation_set_error(stats, "prefill post-attention norm failed");
                    return 0;
                }
                if (!moe_layer_ffn_norm_f32(
                        engine, (int32_t)layer, current_hidden, (uint32_t)hs,
                        scratch->norm.data(), layer_attn_cfg.rms_norm_eps)) {
                    if (layer_plan.ffn_norm_required) {
                        moe_generation_set_error(stats, "prefill ffn norm failed");
                        return 0;
                    }
                    std::memcpy(scratch->norm.data(), current_hidden, hs * sizeof(float));
                }
            } else {
                if (!moe_layer_attention_f32(engine, (int32_t)layer, scratch, current_hidden, (uint32_t)hs, t)) {
                    moe_generation_set_error(stats, "prefill attention failed");
                    return 0;
                }
                if (!moe_pc_engine_raw_rmsnorm_f32(
                        engine, (int32_t)layer, moe_RAW_TENSOR_POST_ATTENTION_LAYER_NORM,
                        current_hidden, (uint32_t)hs, scratch->norm.data(), (uint32_t)hs,
                        layer_attn_cfg.rms_norm_eps)) {
                    moe_generation_set_error(stats, "prefill post-attention norm failed");
                    return 0;
                }
            }
            std::memcpy(batch_norm.data() + (size_t)t * hs, scratch->norm.data(), hs * sizeof(float));
            moe_profile_log(
                "prefill_layer_attention_token",
                "scope=%s id=%llu layer=%u token_index=%u ms=%.3f",
                prefix,
                (unsigned long long)profile_id,
                layer,
                t,
                moe_profile_elapsed_ms(attention_start_us));
        }

        engine->generation_phase.store(moe_GEN_PHASE_MLP, std::memory_order_relaxed);
        const int is_moe = moe_layer_is_moe_runtime(engine, c, (int32_t)layer);
        const uint64_t mlp_start_us = moe_profile_now_us();
        int mlp_batch_ok = 0;
        if (is_moe && prefill_count >= 2u) {
            mlp_batch_ok = moe_layer_moe_mlp_prefill_batch_f32(
                engine, (int32_t)layer,
                batch_hidden.data(), batch_norm.data(),
                prefill_count, c);
        }
        if (mlp_batch_ok == moe_PREFILL_BATCH_MLP_FAST_KERNEL_REQUIRED) {
            moe_generation_set_error(
                stats,
                "internal backend policy requires moe_model_lib_run_expert_triplet_batch_f32");
            return 0;
        }
        if (!mlp_batch_ok) {
            for (uint32_t t = 0; t < prefill_count; ++t) {
                const uint64_t token_start_us = moe_profile_now_us();
                float* current_hidden = batch_hidden.data() + (size_t)t * hs;
                std::memcpy(scratch->norm.data(), batch_norm.data() + (size_t)t * hs, hs * sizeof(float));
                engine->generation_token.store(t, std::memory_order_relaxed);
                const int mlp_ok = is_moe ?
                    moe_layer_moe_mlp_f32(engine, (int32_t)layer, scratch, current_hidden, c) :
                    moe_layer_dense_mlp_f32(engine, (int32_t)layer, scratch, current_hidden, (uint32_t)hs);
                if (!mlp_ok) {
                    moe_generation_set_error(stats, "prefill mlp failed");
                    return 0;
                }
                moe_profile_log(
                    "prefill_layer_token",
                    "scope=%s id=%llu layer=%u token_index=%u is_moe=%d batch_fallback=1 ms=%.3f",
                    prefix,
                    (unsigned long long)profile_id,
                    layer,
                    t,
                    is_moe,
                    moe_profile_elapsed_ms(token_start_us));
            }
        }
        for (uint32_t t = 0; t < prefill_count; ++t) {
            float* current_hidden = batch_hidden.data() + (size_t)t * hs;
            if (!moe_layer_apply_ple_f32(
                    engine, (int32_t)layer, scratch, current_hidden, (uint32_t)hs,
                    c->num_hidden_layers, layer_attn_cfg.rms_norm_eps)) {
                moe_generation_set_error(stats, "prefill per-layer input block failed");
                return 0;
            }
            if (layer_plan.apply_layer_output_scale &&
                !moe_layer_output_scale_f32(engine, (int32_t)layer, current_hidden, (uint32_t)hs)) {
                moe_generation_set_error(stats, "prefill layer output scale failed");
                return 0;
            }
        }
        moe_profile_log(
            "prefill_layer_mlp",
            "scope=%s id=%llu layer=%u tokens=%u is_moe=%d batch=%d ms=%.3f",
            prefix,
            (unsigned long long)profile_id,
            layer,
            prefill_count,
            is_moe,
            mlp_batch_ok,
            moe_profile_elapsed_ms(mlp_start_us));
        moe_profile_log(
            "prefill_layer",
            "scope=%s id=%llu layer=%u tokens=%u ms=%.3f",
            prefix,
            (unsigned long long)profile_id,
            layer,
            prefill_count,
            moe_profile_elapsed_ms(layer_start_us));
    }

    float* last_hidden = batch_hidden.data() + (size_t)(prefill_count - 1u) * hs;
    if (!moe_final_norm_f32(
            engine, c, last_hidden, (uint32_t)hs,
            scratch->norm.data(), attn_cfg.rms_norm_eps)) {
        moe_generation_set_error(stats, "prefill final norm failed");
        return 0;
    }
    std::memcpy(scratch->hidden.data(), scratch->norm.data(), hs * sizeof(float));
    scratch->position = prefill_count;
    moe_profile_log(
        "prefill_end",
        "scope=%s id=%llu tokens=%u layer_first=1 ms=%.3f",
        prefix,
        (unsigned long long)profile_id,
        prefill_count,
        moe_profile_elapsed_ms(prefill_start_us));
    return 1;
}
