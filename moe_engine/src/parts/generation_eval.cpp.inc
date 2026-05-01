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
    for (uint32_t i = 0; i < input_token_count; ++i) {
        if (input_token_ids[i] < 0 || (uint32_t)input_token_ids[i] >= c->vocab_size) {
            moe_eval_set_error(out_stats, "input token outside vocabulary");
            return 0;
        }
    }

    moe_decode_scratch scratch;
    moe_kv_cache_ram_scope kv_scope{engine, &scratch.kv_cache};
    moe_kv_decode_cache_ram_scope decode_cache_scope{engine, &scratch};
    const int pre_allocate_kv =
        moe_generation_should_preallocate_kv_cache(
            engine, c->num_hidden_layers, moe_attn_kv_entry_dim(attn_cfg), attn_cfg.max_seq_len);
    moe_decode_scratch_resize(&scratch, c, &attn_cfg, pre_allocate_kv);
    moe_kv_cache_track_ram(engine, &scratch.kv_cache);
    scratch.position = 0;

    double nll = 0.0;
    moe_generation_stats_t gen_stats{};
    for (uint32_t i = 0; i + 1u < input_token_count; ++i) {
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

        float target_logit = 0.0f;
        float logsumexp = 0.0f;
        float logprob = 0.0f;
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
        nll += -(double)logprob;
        ++scratch.position;
    }

    const uint32_t evaluated = input_token_count - 1u;
    const double mean_nll = evaluated ? nll / (double)evaluated : 0.0;
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
