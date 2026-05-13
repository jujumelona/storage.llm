
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
        !(engine->model_config_query_pre_attn_scalar > 0.0f) ||
        !(engine->model_config_attention_scale > 0.0f) ||
        !std::isfinite(engine->model_config_query_pre_attn_scalar) ||
        !std::isfinite(engine->model_config_attention_scale)) {
        moe_eval_set_error(out_stats, "JUJU attention scale contract is not loaded");
        return 0;
    }
    const double expected = moe_engine_is_gemma4_text_contract(engine)
        ? 1.0
        : 1.0 / std::sqrt((double)engine->model_config_query_pre_attn_scalar);
    const double actual = moe_engine_is_gemma4_text_contract(engine)
        ? 1.0
        : (double)engine->model_config_attention_scale;
    const double tol = std::max(1.0e-8, std::fabs(expected) * 1.0e-5);
    if (!std::isfinite(expected) || std::fabs(actual - expected) > tol) {
        moe_eval_set_error(out_stats, "JUJU attention scale does not match runtime attention contract");
        return 0;
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
        (engine->config.kv_mode == moe_KV_MODE_QKV && qkv_is_enabled()) ? "qkv_contract" : "plain_kv_cache");
    // BUGFIX 563: Check moe_decode_scratch_resize success ★
    // Problem: moe_decode_scratch_resize can fail on allocation
    // Solution: Check return value (assuming it returns int)
    if (!moe_decode_scratch_resize(&scratch, c, &attn_cfg, pre_allocate_kv, engine)) {
        moe_eval_set_error(out_stats, "failed to allocate decode scratch");
        return 0;
    }
    moe_profile_log(
        "eval_prepare_end",
        "kv_max_seq=%u qkv_cap=%u qkv_limit=%u plain_kv_bytes=%llu rope_q=%zu rope_k=%zu hidden=%zu",
        eval_qkv_only ? 0u : scratch.kv_cache.max_seq_len,
        scratch.qkv_capacity_tokens,
        scratch.qkv_seq_limit,
        (unsigned long long)(eval_qkv_only ? 0ull : moe_kv_cache_allocated_bytes(scratch.kv_cache)),
        scratch.rope_q_cos.size(),
        scratch.rope_k_cos.size(),
        scratch.hidden.size());
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
    moe_generation_stats_t gen_stats{};
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
    moe_profile_log(
        "eval_done",
        "id=%llu input_tokens=%u first_target=%u evaluated=%u nll=%.9g mean_nll=%.9g ppl=%.9g total_ms=%.3f",
        (unsigned long long)eval_scope.id,
        input_token_count,
        first_target_index,
        evaluated,
        nll,
        mean_nll,
        ppl,
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
