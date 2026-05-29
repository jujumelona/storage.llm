
static int moe_eval_exact_ppl_contract_ready(
    moe_pc_engine_t* engine,
    moe_eval_stats_t* out_stats
) {
    if (!engine) {
        moe_eval_set_error(out_stats, "engine is null");
        return 0;
    }
    if (engine->juju_required_features_contract_ready.load(std::memory_order_acquire) != 1 ||
        engine->juju_exact_ppl_contract_ready.load(std::memory_order_acquire) != 1) {
        moe_eval_set_error(out_stats, "JUJU exact_ppl_mode required_features contract is not loaded");
        return 0;
    }
    if (engine->juju_attention_contract_ready.load(std::memory_order_acquire) != 1 ||
        !(engine->model_config_attention_scale > 0.0f) ||
        !std::isfinite(engine->model_config_attention_scale)) {
        moe_eval_set_error(out_stats, "JUJU attention scale contract is not loaded");
        return 0;
    }
    if (engine->model_config_attention_unit_scale.load(std::memory_order_acquire) == 1) {
        const double actual = (double)engine->model_config_attention_scale;
        if (!std::isfinite(actual) || std::fabs(actual - 1.0) > 1.0e-8) {
            moe_eval_set_error(out_stats, "JUJU attention scale does not match runtime attention contract");
            return 0;
        }
    } else {
        if (!(engine->model_config_query_pre_attn_scalar > 0.0f) ||
            !std::isfinite(engine->model_config_query_pre_attn_scalar)) {
            moe_eval_set_error(out_stats, "JUJU attention scale contract is not loaded");
            return 0;
        }
    // Non-unit numeric attention-scale contracts are checked against
    // query_pre_attn_scalar; explicit unit-QK contracts are handled above.
        const double expected = 1.0 / std::sqrt((double)engine->model_config_query_pre_attn_scalar);
        const double actual = (double)engine->model_config_attention_scale;
        const double tol = std::max(1.0e-8, std::fabs(expected) * 1.0e-5);
        if (!std::isfinite(expected) || std::fabs(actual - expected) > tol) {
            moe_eval_set_error(out_stats, "JUJU attention scale does not match runtime attention contract");
            return 0;
        }
    }
    if (engine->storage_format_plan_available.load(std::memory_order_acquire) != 1) {
        moe_eval_set_error(out_stats, "JUJU storage format plan is not available for exact PPL");
        return 0;
    }
    if (engine->storage_expert_index_ready.load(std::memory_order_acquire) != 1) {
        moe_eval_set_error(out_stats, "JUJU expert index is not ready for exact PPL");
        return 0;
    }
    return 1;
}

static int moe_eval_layer_first_enabled(
    const moe_pc_engine_t* engine,
    uint32_t prefill_count,
    uint32_t hidden_size
) {
    const char* env = std::getenv("STORAGELLM_EVAL_LAYER_FIRST_PREFILL");
    if (!env ||
        env[0] == '\0' ||
        env[0] == '0' ||
        std::strcmp(env, "false") == 0 ||
        std::strcmp(env, "FALSE") == 0 ||
        std::strcmp(env, "off") == 0 ||
        std::strcmp(env, "OFF") == 0) {
        return 0;
    }
    return moe_generation_should_use_layer_first_prefill(engine, prefill_count, hidden_size);
}

static int moe_eval_layer_first_forward_all_f32(
    moe_pc_engine_t* engine,
    const int32_t* input_token_ids,
    uint32_t prefill_count,
    moe_decode_scratch* scratch,
    const moe_storage_constants_t* c,
    const moe_attn_config_t* attn_cfg,
    int pre_allocate_kv,
    int eval_qkv_only,
    std::vector<float>* out_hidden,
    moe_eval_stats_t* out_stats
) {
    if (!engine || !input_token_ids || !scratch || !c || !attn_cfg || !out_hidden ||
        prefill_count == 0 || c->hidden_size == 0 || c->num_hidden_layers == 0) {
        moe_eval_set_error(out_stats, "layer-first eval received invalid inputs");
        return 0;
    }
    const uint32_t hidden_size = c->hidden_size;
    const size_t hs = hidden_size;
    out_hidden->assign((size_t)prefill_count * hs, 0.0f);
    std::vector<float> batch_norm((size_t)prefill_count * hs, 0.0f);

    const uint64_t embed_start_us = moe_profile_now_us();
    std::atomic<int> embed_ok{1};
    if (prefill_count >= 2u && moe_generation_parallel_prefill_embed_enabled()) {
        const int rows_ok = moe_parallel_rows(engine, prefill_count, 0, [&](uint32_t i) {
            if (!moe_pc_engine_embed_token_f32(
                    engine,
                    (uint32_t)input_token_ids[i],
                    out_hidden->data() + (size_t)i * hs,
                    hidden_size)) {
                embed_ok.store(0, std::memory_order_relaxed);
            }
        });
        if (!rows_ok) {
            embed_ok.store(0, std::memory_order_relaxed);
        }
    } else {
        for (uint32_t i = 0; i < prefill_count; ++i) {
            if (!moe_pc_engine_embed_token_f32(
                    engine,
                    (uint32_t)input_token_ids[i],
                    out_hidden->data() + (size_t)i * hs,
                    hidden_size)) {
                embed_ok.store(0, std::memory_order_relaxed);
                break;
            }
        }
    }
    if (!embed_ok.load(std::memory_order_relaxed)) {
        moe_eval_set_error(out_stats, "layer-first eval embedding failed");
        return 0;
    }
    moe_profile_log(
        "eval_layer_first_embed",
        "tokens=%u hidden=%u parallel=%d ms=%.3f",
        prefill_count,
        hidden_size,
        (prefill_count >= 2u && moe_generation_parallel_prefill_embed_enabled()) ? 1 : 0,
        moe_profile_elapsed_ms(embed_start_us));

    if (!moe_decode_scratch_resize(scratch, c, attn_cfg, pre_allocate_kv, engine)) {
        moe_eval_set_error(out_stats, "layer-first eval scratch allocation failed");
        return 0;
    }
    if (!eval_qkv_only) {
        moe_kv_cache_track_ram(engine, &scratch->kv_cache);
    }
    if (!pre_allocate_kv) {
        if (eval_qkv_only) {
            if (!moe_qkv_only_ensure_capacity(engine, scratch, *attn_cfg, prefill_count - 1u, 0)) {
                moe_eval_set_error(out_stats, "layer-first eval QKV capacity allocation failed");
                return 0;
            }
        } else if (prefill_count > scratch->kv_cache.max_seq_len) {
            if (!scratch->kv_cache.ensure_capacity(prefill_count - 1u)) {
                moe_eval_set_error(out_stats, "layer-first eval KV capacity allocation failed");
                return 0;
            }
            moe_kv_cache_track_ram(engine, &scratch->kv_cache);
        }
    }
    engine->generation_phase.store(moe_GEN_PHASE_PREFILL, std::memory_order_relaxed);
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
            moe_build_decoder_layer_plan(engine, c, attn_cfg, layer);
        for (uint32_t t = 0; t < prefill_count; ++t) {
            const uint64_t attn_start_us = moe_profile_now_us();
            float* current_hidden = out_hidden->data() + (size_t)t * hs;
            engine->generation_token.store(t, std::memory_order_relaxed);
            scratch->position = t;
            engine->generation_phase.store(moe_GEN_PHASE_ATTENTION, std::memory_order_relaxed);
            if (layer_attn_cfg.type != moe_ATTN_TYPE_MLA) {
                std::memcpy(scratch->residual.data(), current_hidden, hs * sizeof(float));
                if (!moe_layer_standard_attention_f32(
                        engine, (int32_t)layer, scratch, current_hidden, hidden_size, t,
                        layer_attn_cfg, nullptr, 0)) {
                    moe_eval_set_error(out_stats, "layer-first eval attention failed");
                    return 0;
                }
                if (!moe_apply_attention_residual_contract_f32(
                        engine, (int32_t)layer, layer_plan,
                        scratch->residual.data(), scratch->attn.data(), current_hidden,
                        scratch->norm.data(), scratch->shared.data(),
                        hidden_size, layer_attn_cfg.rms_norm_eps)) {
                    moe_eval_set_error(out_stats, "layer-first eval post-attention norm failed");
                    return 0;
                }
                if (!moe_layer_ffn_norm_f32(
                        engine, (int32_t)layer, current_hidden, hidden_size,
                        scratch->norm.data(), layer_attn_cfg.rms_norm_eps)) {
                    if (layer_plan.ffn_norm_required) {
                        moe_eval_set_error(out_stats, "layer-first eval ffn norm failed");
                        return 0;
                    }
                    std::memcpy(scratch->norm.data(), current_hidden, hs * sizeof(float));
                }
            } else {
                if (!moe_layer_attention_f32(
                        engine, (int32_t)layer, scratch, current_hidden, hidden_size, t)) {
                    moe_eval_set_error(out_stats, "layer-first eval attention failed");
                    return 0;
                }
                if (!moe_pc_engine_raw_rmsnorm_f32(
                        engine, (int32_t)layer, moe_RAW_TENSOR_POST_ATTENTION_LAYER_NORM,
                        current_hidden, hidden_size,
                        scratch->norm.data(), hidden_size, layer_attn_cfg.rms_norm_eps)) {
                    moe_eval_set_error(out_stats, "layer-first eval post-attention norm failed");
                    return 0;
                }
            }
            std::memcpy(
                batch_norm.data() + (size_t)t * hs,
                scratch->norm.data(),
                hs * sizeof(float));
            moe_profile_log(
                "eval_layer_first_attention_token",
                "layer=%u token_index=%u ms=%.3f",
                layer, t, moe_profile_elapsed_ms(attn_start_us));
        }

        engine->generation_phase.store(moe_GEN_PHASE_MLP, std::memory_order_relaxed);
        const int is_moe = moe_layer_is_moe_runtime(engine, c, (int32_t)layer);
        const uint64_t mlp_start_us = moe_profile_now_us();
        int mlp_batch_ok = 0;
        if (is_moe && prefill_count >= 2u) {
            mlp_batch_ok = moe_layer_moe_mlp_prefill_batch_f32(
                engine,
                (int32_t)layer,
                out_hidden->data(),
                batch_norm.data(),
                prefill_count,
                c);
        }
        if (mlp_batch_ok == moe_PREFILL_BATCH_MLP_FAST_KERNEL_REQUIRED) {
            moe_eval_set_error(out_stats, "layer-first eval requires unavailable batch MLP backend");
            return 0;
        }
        if (!mlp_batch_ok) {
            for (uint32_t t = 0; t < prefill_count; ++t) {
                float* current_hidden = out_hidden->data() + (size_t)t * hs;
                std::memcpy(
                    scratch->norm.data(),
                    batch_norm.data() + (size_t)t * hs,
                    hs * sizeof(float));
                engine->generation_token.store(t, std::memory_order_relaxed);
                scratch->position = t;
                const int mlp_ok = is_moe ?
                    moe_layer_moe_mlp_f32(engine, (int32_t)layer, scratch, current_hidden, c) :
                    moe_layer_dense_mlp_f32(engine, (int32_t)layer, scratch, current_hidden, hidden_size);
                if (!mlp_ok) {
                    moe_eval_set_error(out_stats, "layer-first eval mlp failed");
                    return 0;
                }
            }
        }
        for (uint32_t t = 0; t < prefill_count; ++t) {
            float* current_hidden = out_hidden->data() + (size_t)t * hs;
            if (!moe_layer_apply_ple_f32(
                    engine, (int32_t)layer, scratch, current_hidden, hidden_size,
                    c->num_hidden_layers, layer_attn_cfg.rms_norm_eps)) {
                moe_eval_set_error(out_stats, "layer-first eval per-layer input block failed");
                return 0;
            }
            if (layer_plan.apply_layer_output_scale &&
                !moe_layer_output_scale_f32(engine, (int32_t)layer, current_hidden, hidden_size)) {
                moe_eval_set_error(out_stats, "layer-first eval layer output scale failed");
                return 0;
            }
            for (uint32_t h = 0; h < hidden_size; ++h) {
                if (!std::isfinite(current_hidden[h])) {
                    moe_eval_set_error(out_stats, "exact PPL aborted: non-finite layer-first hidden state");
                    return 0;
                }
            }
        }
        moe_profile_log(
            "eval_layer_first_layer",
            "layer=%u tokens=%u is_moe=%d batch=%d mlp_ms=%.3f total_ms=%.3f",
            layer,
            prefill_count,
            is_moe,
            mlp_batch_ok,
            moe_profile_elapsed_ms(mlp_start_us),
            moe_profile_elapsed_ms(layer_start_us));
    }
    scratch->position = prefill_count;
    return 1;
}

int moe_pc_engine_eval_token_ids_from(
    moe_pc_engine_t* engine,
    const int32_t* input_token_ids,
    uint32_t input_token_count,
    uint32_t first_target_index,
    moe_eval_stats_t* out_stats
) try {
    if (out_stats) {
        std::memset(out_stats, 0, sizeof(*out_stats));
    }
    if (!engine) {
        moe_eval_set_error(out_stats, "engine is null");
        return 0;
    }
    if (!input_token_ids || input_token_count < 2) {
        moe_eval_set_error(out_stats, "eval requires at least two input tokens");
        return 0;
    }
    // Build/select the CPU-storage runtime plan before checking plan readiness.
    // The readiness bits are produced by moe_cpu_storage_build_execution_plan(),
    // which is reached through moe_cpu_storage_ensure_runtime().  Checking them
    // first makes exact PPL fail closed before the engine has a chance to create
    // the required runtime tensor/expert index plan.
    moe_cpu_storage_ensure_runtime(engine);
    if (!moe_eval_exact_ppl_contract_ready(engine, out_stats)) {
        return 0;
    }
    const moe_storage_constants_t runtime_c = moe_engine_storage_constants_or_static(engine);
    const moe_storage_constants_t* c = &runtime_c;
    const moe_attn_config_t attn_cfg = moe_generation_attn_config(engine);
    if (!c) {
        moe_eval_set_error(out_stats, "engine storage constants unavailable");
        return 0;
    }
    struct eval_scope_t {
        moe_pc_engine_t* engine = nullptr;
        std::unique_lock<std::mutex> api_lock;
        int ok = 0;
        uint64_t id = 0;
        uint64_t start_us = 0;
        explicit eval_scope_t(moe_pc_engine_t* e) : engine(e) {
            if (!engine) return;
            api_lock = std::unique_lock<std::mutex>(engine->generation_api_mutex);
            id = moe_profile_next_generation_id();
            start_us = moe_profile_now_us();
#ifdef STORAGELLM_PPL_DIAG_ENABLED
            const uint32_t active_before =
                engine->generation_active.load(std::memory_order_acquire);
            const size_t qkv_slots_before = engine->qkv_head_state_slots.size();
            const size_t qkv_valid_before = engine->qkv_head_state_valid.size();
            const size_t qkv_epochs_before = engine->qkv_head_state_epochs.size();
            const uint64_t qkv_epoch_before = engine->qkv_context_epoch;
            const uint32_t qkv_last_ctx_before = engine->qkv_last_ctx;
#endif
            engine->generation_started.fetch_add(1, std::memory_order_relaxed);
            moe_engine_reset_qkv_cache(engine);
            engine->generation_active.fetch_add(1, std::memory_order_relaxed);
            engine->generation_token.store(0, std::memory_order_relaxed);
            engine->generation_layer.store(0xffffffffu, std::memory_order_relaxed);
            engine->generation_phase.store(moe_GEN_PHASE_INIT, std::memory_order_relaxed);
#ifdef STORAGELLM_PPL_DIAG_ENABLED
            moe_profile_log(
                "eval_begin",
                "id=%llu active_before=%u qkv_slots_before=%zu qkv_valid_before=%zu qkv_epochs_before=%zu qkv_epoch_before=%llu qkv_last_ctx_before=%u",
                (unsigned long long)id,
                active_before,
                qkv_slots_before,
                qkv_valid_before,
                qkv_epochs_before,
                (unsigned long long)qkv_epoch_before,
                qkv_last_ctx_before);
#endif
        }
        ~eval_scope_t() {
            if (!engine) return;
            if (ok) {
                engine->generation_completed.fetch_add(1, std::memory_order_relaxed);
                engine->generation_phase.store(moe_GEN_PHASE_DONE, std::memory_order_relaxed);
            } else {
                engine->generation_failed.fetch_add(1, std::memory_order_relaxed);
            }
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
                "eval_end",
                "id=%llu ok=%d active=%u layer=%u phase=%u ms=%.3f",
                (unsigned long long)id,
                ok,
                engine->generation_active.load(std::memory_order_relaxed),
                engine->generation_layer.load(std::memory_order_relaxed),
                engine->generation_phase.load(std::memory_order_relaxed),
                moe_profile_elapsed_ms(start_us));
        }
    } eval_scope(engine);
    if (first_target_index == 0) {
        first_target_index = 1;
    }
    if (first_target_index >= input_token_count) {
        moe_eval_set_error(out_stats, "eval target range is empty");
        return 0;
    }
    moe_forward_status_t forward{};
    const int forward_status_ok = moe_pc_engine_get_forward_status(engine, &forward);
    if (forward_status_ok && forward.graph_ir_required) {
        if (!forward.graph_ir_ready) {
            moe_eval_set_error(out_stats, "graph IR is required for exact eval but is not loaded");
            return 0;
        }
        if (!forward.runtime_execution_manifest_ready ||
            !forward.runtime_access_plan_ready ||
            !forward.kv_layout_contract_ready) {
            moe_eval_set_error(out_stats, "graph IR runtime contract is incomplete for exact eval");
            return 0;
        }
    }
    if (forward_status_ok &&
        forward.dynamic_shape_ready &&
        !forward.forward_adapter_executable) {
        char reason[320]{};
        if (forward.graph_ir_ready) {
            std::snprintf(reason, sizeof(reason),
                "graph IR executor not executable graph_ver=%u layers=%u ops=%llu hidden=%u vocab=%u runtime_manifest=%d access_plan=%d kv_layout=%d hotset=%llu file_groups=%llu tensor_refs=%llu layer_prefetch=%llu kv_entry=%llu kv_page=%llu",
                forward.graph_ir_version,
                forward.dynamic_num_hidden_layers,
                (unsigned long long)forward.graph_ir_op_count,
                forward.dynamic_hidden_size,
                forward.dynamic_vocab_size,
                forward.runtime_execution_manifest_ready,
                forward.runtime_access_plan_ready,
                forward.kv_layout_contract_ready,
                (unsigned long long)forward.runtime_access_hotset_count,
                (unsigned long long)forward.runtime_access_file_group_count,
                (unsigned long long)forward.runtime_access_executor_tensor_count,
                (unsigned long long)forward.runtime_access_layer_prefetch_plan_count,
                (unsigned long long)forward.kv_layout_max_entry_dim,
                (unsigned long long)forward.kv_layout_page_size_tokens);
        } else {
            std::snprintf(reason, sizeof(reason),
                "forward adapter not executable adapter=%s layers=%u hidden=%u vocab=%u",
                moe_forward_adapter_name(forward.forward_adapter),
                forward.dynamic_num_hidden_layers,
                forward.dynamic_hidden_size,
                forward.dynamic_vocab_size);
        }
        moe_eval_set_error(out_stats, reason);
        return 0;
    }
    if (!forward_status_ok ||
        !forward.storage_state_valid ||
        !forward.tensor_table_loaded ||
        !forward.expert_triplet_available ||
        !forward.embedding_ready ||
        !forward.lm_head_ready ||
        !forward.attention_ready ||
        !forward.storage_format_plan_available ||
        !forward.storage_expert_index_ready ||
        !forward.cpu_quant_registry_ready) {
        moe_eval_set_error(out_stats, "full exact eval path is not ready");
        return 0;
    }
    if (input_token_count > attn_cfg.max_seq_len) {
        moe_eval_set_error(out_stats, "input exceeds Moe MLA context length");
        return 0;
    }
    // BUGFIX 561: Check vocab_size > 0 before validation loop ★
    // Problem: vocab_size can be 0, causing all tokens to fail validation
    // Solution: Check vocab_size before loop
    if (c->vocab_size == 0) {
        moe_eval_set_error(out_stats, "vocabulary size is zero");
        return 0;
    }
    for (uint32_t i = 0; i < input_token_count; ++i) {
        if (input_token_ids[i] < 0 || (uint32_t)input_token_ids[i] >= c->vocab_size) {
            moe_eval_set_error(out_stats, "input token outside vocabulary");
            return 0;
        }
    }

    // BUGFIX 247: Initialize scratch to prevent undefined behavior ★★★
    // Problem: moe_decode_scratch declared but not initialized before use
    // scratch.kv_cache, scratch.hidden, etc. contain garbage values
    // Solution: Zero-initialize before using
    char eval_ids_preview[256]{};
    size_t eval_ids_offset = 0;
    const uint32_t eval_ids_limit = std::min<uint32_t>(input_token_count, 16u);
    for (uint32_t i = 0; i < eval_ids_limit && eval_ids_offset < sizeof(eval_ids_preview); ++i) {
        const int written = std::snprintf(
            eval_ids_preview + eval_ids_offset,
            sizeof(eval_ids_preview) - eval_ids_offset,
            "%s%d",
            i ? "," : "",
            input_token_ids[i]);
        if (written <= 0) {
            break;
        }
        eval_ids_offset += std::min<size_t>(
            (size_t)written,
            sizeof(eval_ids_preview) - eval_ids_offset);
    }
    moe_profile_log(
        "eval_input_ids",
        "input_tokens=%u first_target=%u preview=[%s]%s",
        input_token_count,
        first_target_index,
        eval_ids_preview,
        input_token_count > eval_ids_limit ? " truncated=1" : "");
    const uint64_t eval_start_us = moe_profile_now_us();
    moe_decode_scratch scratch;
    moe_qkv_force_plain_ab_scope_t eval_plain_kv_scope(
        moe_eval_uses_plain_kv_for_ppl_default());
    const int eval_qkv_only = moe_engine_qkv_only_required(engine);
    if (eval_qkv_only) {
        engine->qkv_ppl_qkv_only_evals.fetch_add(1, std::memory_order_relaxed);
    }
    moe_kv_cache_ram_scope kv_scope{engine, eval_qkv_only ? nullptr : &scratch.kv_cache};
    moe_kv_decode_cache_ram_scope decode_cache_scope{engine, eval_qkv_only ? nullptr : &scratch};
    // BUGFIX 562: Check num_hidden_layers > 0 before allocation ★
    // Problem: num_hidden_layers can be 0, causing invalid memory allocation
    // Solution: Check before calling moe_generation_should_preallocate_kv_cache
    if (c->num_hidden_layers == 0) {
        moe_eval_set_error(out_stats, "num_hidden_layers is zero");
        return 0;
    }
    // Eval/PPL walks a fixed prompt and does not benefit from reserving the
    // whole model context window up front. Large-context roots can otherwise
    // spend minutes in scratch/KV/RoPE preparation before scoring one token.
    const int pre_allocate_kv = 0;
    moe_profile_log(
        "eval_prepare_begin",
        "input_tokens=%u first_target=%u layers=%u hidden=%u max_seq=%u rope_len=%u kv_entry=%u prealloc=%d kv_backend=%s",
        input_token_count,
        first_target_index,
        c->num_hidden_layers,
        c->hidden_size,
        attn_cfg.max_seq_len,
        attn_cfg.rope_table_len,
        moe_attn_kv_entry_dim(attn_cfg),
        pre_allocate_kv,
        eval_qkv_only ? "qkv_contract" : "plain_kv_cache");
    // BUGFIX 563: Check moe_decode_scratch_resize success ★
    // Problem: moe_decode_scratch_resize can fail on allocation
    // Solution: Check return value (assuming it returns int)
    if (!moe_decode_scratch_resize(&scratch, c, &attn_cfg, pre_allocate_kv, engine)) {
        moe_eval_set_error(out_stats, "failed to allocate decode scratch");
        return 0;
    }
    {
        const float eval_softcap = moe_final_logit_softcap_value(engine);
        moe_profile_log(
            "eval_prepare_end",
            "kv_max_seq=%u qkv_cap=%u qkv_limit=%u plain_kv_bytes=%llu rope_q=%zu rope_k=%zu hidden=%zu final_logit_softcap=%.6g vocab=%u",
            eval_qkv_only ? 0u : scratch.kv_cache.max_seq_len,
            scratch.qkv_capacity_tokens,
            scratch.qkv_seq_limit,
            (unsigned long long)(eval_qkv_only ? 0ull : moe_kv_cache_allocated_bytes(scratch.kv_cache)),
            scratch.rope_q_cos.size(),
            scratch.rope_k_cos.size(),
            scratch.hidden.size(),
            (double)eval_softcap,
            c->vocab_size);
    }
    if (scratch.hidden.size() < c->hidden_size ||
        scratch.residual.size() < c->hidden_size ||
        scratch.norm.size() < c->hidden_size ||
        scratch.attn.size() < c->hidden_size) {
        moe_eval_set_error(out_stats, "failed to allocate decode scratch");
        return 0;
    }
    if (!eval_qkv_only) {
        moe_kv_cache_track_ram(engine, &scratch.kv_cache);
    }
    scratch.position = 0;

    double nll = 0.0;
    uint32_t evaluated = 0;
    const uint64_t linear_fallbacks_before =
        engine->storage_selected_expert_linear_fallbacks.load(std::memory_order_acquire);
    const uint32_t prefill_count = input_token_count - 1u;
    if (moe_eval_layer_first_enabled(engine, prefill_count, c->hidden_size)) {
        const uint64_t layer_first_start_us = moe_profile_now_us();
        std::vector<float> batch_hidden;
        if (!moe_eval_layer_first_forward_all_f32(
                engine, input_token_ids, prefill_count, &scratch, c, &attn_cfg,
                pre_allocate_kv, eval_qkv_only, &batch_hidden, out_stats)) {
            return 0;
        }
        const uint64_t linear_fallbacks_now =
            engine->storage_selected_expert_linear_fallbacks.load(std::memory_order_acquire);
        if (linear_fallbacks_now != linear_fallbacks_before) {
            moe_eval_set_error(out_stats, "exact PPL aborted: selected-expert linear fallback was used");
            return 0;
        }
        moe_profile_log(
            "eval_layer_first_forward_end",
            "positions=%u ms=%.3f",
            prefill_count,
            moe_profile_elapsed_ms(layer_first_start_us));
        for (uint32_t i = 0; i < prefill_count; ++i) {
            const uint32_t target_index = i + 1u;
            const int score_target = target_index >= first_target_index;
            moe_profile_log(
                "eval_forward_call_end",
                "pos=%u token=%u score=%d layer_first=1",
                i, (uint32_t)input_token_ids[i], score_target);
            if (!score_target) {
                continue;
            }
            const float* hidden_for_score =
                batch_hidden.data() + (size_t)i * (size_t)c->hidden_size;
            if (!moe_final_norm_f32(
                    engine, c, hidden_for_score, c->hidden_size,
                    scratch.norm.data(), attn_cfg.rms_norm_eps)) {
                moe_eval_set_error(out_stats, "failed to apply final norm for layer-first eval");
                return 0;
            }
            for (uint32_t h = 0; h < c->hidden_size; ++h) {
                if (!std::isfinite(scratch.norm[h])) {
                    moe_eval_set_error(out_stats, "exact PPL aborted: non-finite final hidden state");
                    return 0;
                }
            }
            float target_logit = 0.0f;
            float logsumexp = 0.0f;
            float logprob = 0.0f;
            if (!moe_pc_engine_lm_head_token_logprob_f32(
                    engine, scratch.norm.data(), c->hidden_size,
                    (uint32_t)input_token_ids[i + 1u],
                    &target_logit, &logsumexp, &logprob)) {
                moe_eval_set_error(out_stats, "failed to score target token");
                return 0;
            }
            if (!std::isfinite(target_logit) ||
                !std::isfinite(logsumexp) ||
                !std::isfinite(logprob)) {
                moe_eval_set_error(out_stats, "exact PPL aborted: non-finite lm_head logprob");
                return 0;
            }
            moe_profile_log(
                "eval_token_logprob",
                "pos=%u token=%u target=%u target_logit=%.6g logsumexp=%.6g logprob=%.6g layer_first=1",
                i,
                (uint32_t)input_token_ids[i],
                (uint32_t)input_token_ids[i + 1u],
                (double)target_logit,
                (double)logsumexp,
                (double)logprob);
            const double neg_logprob = -(double)logprob;
            if (nll > std::numeric_limits<double>::max() - neg_logprob) {
                moe_eval_set_error(out_stats, "NLL overflow on long sequence");
                return 0;
            }
            nll += neg_logprob;
            ++evaluated;
        }
    }
    moe_generation_stats_t gen_stats{};
    if (evaluated == 0) {
    for (uint32_t i = 0; i + 1u < input_token_count; ++i) {
        engine->generation_token.store(i, std::memory_order_relaxed);
        moe_profile_log(
            "eval_forward_call_begin",
            "pos=%u token=%u target=%u",
            i,
            (uint32_t)input_token_ids[i],
            (uint32_t)input_token_ids[i + 1u]);
        if (!moe_forward_one_token_dispatch_f32(
                engine,
                (uint32_t)input_token_ids[i],
                &scratch,
                &gen_stats,
                scratch.position,
                pre_allocate_kv)) {
            moe_eval_set_error(out_stats, gen_stats.error[0] ? gen_stats.error : "eval forward failed");
            return 0;
        }
        const uint64_t linear_fallbacks_now =
            engine->storage_selected_expert_linear_fallbacks.load(std::memory_order_acquire);
        if (linear_fallbacks_now != linear_fallbacks_before) {
            moe_eval_set_error(out_stats, "exact PPL aborted: selected-expert linear fallback was used");
            return 0;
        }
        for (uint32_t h = 0; h < c->hidden_size; ++h) {
            if (!std::isfinite(scratch.hidden[h])) {
                moe_eval_set_error(out_stats, "exact PPL aborted: non-finite hidden state");
                return 0;
            }
        }
        const uint32_t target_index = i + 1u;
        const int score_target = target_index >= first_target_index;
        moe_profile_log(
            "eval_forward_call_end",
            "pos=%u token=%u score=%d",
            i,
            (uint32_t)input_token_ids[i],
            score_target);
        if (!score_target) {
            ++scratch.position;
            continue;
        }

        float target_logit = 0.0f;
        float logsumexp = 0.0f;
        float logprob = 0.0f;
        // BUGFIX 564: Check scratch.hidden.data() is not null ★
        // Problem: scratch.hidden can be empty if allocation failed
        // Solution: Check before passing to lm_head
        if (scratch.hidden.empty() || !scratch.hidden.data()) {
            moe_eval_set_error(out_stats, "hidden state buffer is empty");
            return 0;
        }
        if (!moe_pc_engine_lm_head_token_logprob_f32(
                engine,
                scratch.hidden.data(),
                c->hidden_size,
                (uint32_t)input_token_ids[i + 1u],
                &target_logit,
                &logsumexp,
                &logprob)) {
            moe_eval_set_error(out_stats, "failed to score target token");
            return 0;
        }
        if (!std::isfinite(target_logit) ||
            !std::isfinite(logsumexp) ||
            !std::isfinite(logprob)) {
            moe_eval_set_error(out_stats, "exact PPL aborted: non-finite lm_head logprob");
            return 0;
        }
        // BUGFIX 248: Add overflow check for NLL accumulation ★
        // Problem: nll += -(double)logprob without overflow check
        // Very long sequences can cause nll to overflow to infinity
        // Solution: Check if accumulation would overflow before adding
        moe_profile_log(
            "eval_token_logprob",
            "pos=%u token=%u target=%u target_logit=%.6g logsumexp=%.6g logprob=%.6g",
            i,
            (uint32_t)input_token_ids[i],
            (uint32_t)input_token_ids[i + 1u],
            (double)target_logit,
            (double)logsumexp,
            (double)logprob);
        const double neg_logprob = -(double)logprob;
        if (nll > std::numeric_limits<double>::max() - neg_logprob) {
            moe_eval_set_error(out_stats, "NLL overflow on long sequence");
            return 0;
        }
        nll += neg_logprob;
        ++evaluated;
        ++scratch.position;
    }

    // BUGFIX 295: Check division by zero before computing mean_nll ★
    }

    // Problem: evaluated can be 0 if input_token_count == 1
    // But function already checks input_token_count < 2 at start
    // However, if input_token_count == 1, evaluated = 0, division by zero in mean_nll
    // Solution: Add explicit check (defense in depth)
    if (evaluated == 0) {
        moe_eval_set_error(out_stats, "no tokens evaluated");
        return 0;
    }
    if (!std::isfinite(nll)) {
        moe_eval_set_error(out_stats, "exact PPL aborted: non-finite NLL");
        return 0;
    }
    const double mean_nll = nll / (double)evaluated;
    if (!std::isfinite(mean_nll)) {
        moe_eval_set_error(out_stats, "exact PPL aborted: non-finite mean NLL");
        return 0;
    }
    double ppl = 0.0;
    if (mean_nll > std::log(std::numeric_limits<double>::max())) {
        ppl = std::numeric_limits<double>::max();
    } else {
        ppl = std::exp(mean_nll);
    }

    if (out_stats) {
        out_stats->input_tokens = input_token_count;
        out_stats->evaluated_tokens = evaluated;
        out_stats->nll = nll;
        out_stats->mean_nll = mean_nll;
        out_stats->perplexity = ppl;
    }
    // PPL sanity diagnostic: warn when mean_nll > ln(vocab_size)
    // This means the model performs worse than uniform random prediction.
    const double random_nll = c->vocab_size > 1 ? std::log((double)c->vocab_size) : 0.0;
    const int ppl_worse_than_random = std::isfinite(mean_nll) && random_nll > 0.0 && mean_nll > random_nll;
    if (ppl_worse_than_random) {
        moe_profile_log(
            "eval_ppl_sanity_warning",
            "mean_nll=%.6g exceeds ln(vocab=%u)=%.6g — model performs worse than uniform random. "
            "Possible causes: wrong test input, quantization degradation, or misconfigured softcap",
            mean_nll, c->vocab_size, random_nll);
    }
    moe_profile_log(
        "eval_done",
        "id=%llu input_tokens=%u first_target=%u evaluated=%u nll=%.9g mean_nll=%.9g ppl=%.9g random_nll=%.6g worse_than_random=%d total_ms=%.3f",
        (unsigned long long)eval_scope.id,
        input_token_count,
        first_target_index,
        evaluated,
        nll,
        mean_nll,
        ppl,
        random_nll,
        ppl_worse_than_random,
        moe_profile_elapsed_ms(eval_start_us));
    eval_scope.ok = 1;
    return 1;
} catch (const std::bad_alloc&) {
    moe_eval_set_error(out_stats, "eval aborted under RAM pressure");
    moe_engine_signal_memory_pressure_recovery(engine, "eval_bad_alloc");
    return 0;
} catch (...) {
    moe_eval_set_error(out_stats, "eval aborted by unexpected exception");
    moe_engine_signal_memory_pressure_recovery(engine, "eval_exception");
    return 0;
}

int moe_pc_engine_eval_token_ids(
    moe_pc_engine_t* engine,
    const int32_t* input_token_ids,
    uint32_t input_token_count,
    moe_eval_stats_t* out_stats
) {
    return moe_pc_engine_eval_token_ids_from(
        engine,
        input_token_ids,
        input_token_count,
        1u,
        out_stats);
}
