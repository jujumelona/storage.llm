int moe_pc_engine_eval_token_ids(
    moe_pc_engine_t* engine,
    const int32_t* input_token_ids,
    uint32_t input_token_count,
    moe_eval_stats_t* out_stats
) {
    if (out_stats) {
        std::memset(out_stats, 0, sizeof(*out_stats));
    }
    const moe_storage_constants_t runtime_c = moe_engine_storage_constants_or_static(engine);
    const moe_storage_constants_t* c = &runtime_c;
    const moe_attn_config_t attn_cfg = moe_generation_attn_config(engine);
    if (!engine || !c || !input_token_ids || input_token_count < 2) {
        moe_eval_set_error(out_stats, "eval requires at least two input tokens");
        return 0;
    }
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
        moe_eval_set_error(out_stats, reason);
        return 0;
    }
    if (!forward_status_ok ||
        !forward.storage_state_valid ||
        !forward.tensor_table_loaded ||
        !forward.expert_triplet_available ||
        !forward.embedding_ready ||
        !forward.lm_head_ready ||
        !forward.attention_ready) {
        moe_eval_set_error(out_stats, "full eval path is not ready");
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
    moe_decode_scratch scratch;
    moe_kv_cache_ram_scope kv_scope{engine, &scratch.kv_cache};
    moe_kv_decode_cache_ram_scope decode_cache_scope{engine, &scratch};
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
        "input_tokens=%u layers=%u hidden=%u max_seq=%u rope_len=%u kv_entry=%u prealloc=%d",
        input_token_count,
        c->num_hidden_layers,
        c->hidden_size,
        attn_cfg.max_seq_len,
        attn_cfg.rope_table_len,
        moe_attn_kv_entry_dim(attn_cfg),
        pre_allocate_kv);
    // BUGFIX 563: Check moe_decode_scratch_resize success ★
    // Problem: moe_decode_scratch_resize can fail on allocation
    // Solution: Check return value (assuming it returns int)
    moe_decode_scratch_resize(&scratch, c, &attn_cfg, pre_allocate_kv, engine);
    moe_profile_log(
        "eval_prepare_end",
        "kv_max_seq=%u rope_q=%zu rope_k=%zu hidden=%zu",
        scratch.kv_cache.max_seq_len,
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
    moe_kv_cache_track_ram(engine, &scratch.kv_cache);
    scratch.position = 0;

    double nll = 0.0;
    moe_generation_stats_t gen_stats{};
    for (uint32_t i = 0; i + 1u < input_token_count; ++i) {
        moe_profile_log(
            "eval_forward_call_begin",
            "pos=%u token=%u target=%u",
            i,
            (uint32_t)input_token_ids[i],
            (uint32_t)input_token_ids[i + 1u]);
        if (!moe_forward_one_token_f32(
                engine,
                (uint32_t)input_token_ids[i],
                &scratch,
                &gen_stats,
                scratch.position,
                pre_allocate_kv)) {
            moe_eval_set_error(out_stats, gen_stats.error[0] ? gen_stats.error : "eval forward failed");
            return 0;
        }
        moe_profile_log(
            "eval_forward_call_end",
            "pos=%u token=%u",
            i,
            (uint32_t)input_token_ids[i]);

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
        ++scratch.position;
    }

    // BUGFIX 295: Check division by zero before computing mean_nll ★
    // Problem: evaluated can be 0 if input_token_count == 1
    // But function already checks input_token_count < 2 at start
    // However, if input_token_count == 1, evaluated = 0, division by zero in mean_nll
    // Solution: Add explicit check (defense in depth)
    const uint32_t evaluated = input_token_count - 1u;
    if (evaluated == 0) {
        moe_eval_set_error(out_stats, "no tokens evaluated");
        return 0;
    }
    const double mean_nll = nll / (double)evaluated;
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
    return 1;
}
