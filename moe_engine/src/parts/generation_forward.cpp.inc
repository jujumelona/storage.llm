static void moe_forward_trace_vec(
    const char* stage,
    uint32_t position,
    uint32_t layer,
    const float* values,
    uint32_t count
) {
    if (!moe_profile_enabled() || !values || count == 0) {
        return;
    }
    double ss = 0.0;
    double mean_abs = 0.0;
    float max_abs = 0.0f;
    for (uint32_t i = 0; i < count; ++i) {
        const float v = values[i];
        const float av = std::fabs(v);
        ss += (double)v * (double)v;
        mean_abs += (double)av;
        if (av > max_abs) {
            max_abs = av;
        }
    }
    moe_profile_log(
        "forward_vec",
        "position=%u layer=%u stage=%s count=%u rms=%.6g mean_abs=%.6g max_abs=%.6g first4=[%.6g,%.6g,%.6g,%.6g]",
        position, layer, stage ? stage : "",
        count,
        std::sqrt(ss / (double)count),
        mean_abs / (double)count,
        (double)max_abs,
        count > 0 ? (double)values[0] : 0.0,
        count > 1 ? (double)values[1] : 0.0,
        count > 2 ? (double)values[2] : 0.0,
        count > 3 ? (double)values[3] : 0.0);
}

static float moe_forward_embedding_scale_value(const moe_pc_engine* engine, uint32_t hidden_size) {
    // BUGFIX 500: hidden_size 범위 체크 강화
    if (!engine || hidden_size == 0 || hidden_size > 100000) {
        return 1.0f;
    }
    const uint32_t cached_hidden = engine->cached_embedding_scale_hidden.load(std::memory_order_acquire);
    const uint32_t cached_bits = engine->cached_embedding_scale_bits.load(std::memory_order_acquire);
    if (cached_hidden == hidden_size && cached_bits != 0u) {
        return moe_f32_from_bits(cached_bits);
    }
    double value = 0.0;
    float scale = 1.0f;
    if (!engine->offload_gguf_metadata_json.empty() &&
        (moe_json_get_double_local(engine->offload_gguf_metadata_json, "embedding_scale", &value) ||
         moe_json_get_double_local(engine->offload_gguf_metadata_json, "embed_scale", &value) ||
         moe_json_get_double_local(engine->offload_gguf_metadata_json, "scale_emb", &value)) &&
        value > 0.0) {
        scale = (float)value;
    }
    else if (moe_engine_graph_ir_mentions(engine, "sqrt_hidden_size") ||
        moe_engine_graph_ir_mentions(engine, "scale_embedding") ||
        moe_engine_metadata_mentions(engine, "sqrt_hidden_size") ||
        moe_engine_metadata_mentions(engine, "scale_embedding")) {
        scale = std::sqrt((float)hidden_size);
    }
    else if (moe_engine_has_common_tensor_suffix(engine, 0, "layer_output_scale.weight") &&
        moe_engine_has_common_tensor_suffix(engine, 0, "ffn_gate_inp.scale")) {
        scale = std::sqrt((float)hidden_size);
    }
    engine->cached_embedding_scale_hidden.store(hidden_size, std::memory_order_release);
    engine->cached_embedding_scale_bits.store(moe_f32_bits(scale), std::memory_order_release);
    return scale;
}

// CUDA Graph cached structures for decode
struct moe_cuda_graph_cache_t {
    void* graph = nullptr;
    void* exec = nullptr;
    bool is_captured = false;
};

static int moe_forward_one_token_f32(
    moe_pc_engine* engine,
    uint32_t token_id,
    moe_decode_scratch* s,
    moe_generation_stats_t* stats,
    uint32_t position,
    int pre_allocate_kv
) {
    const moe_storage_constants_t runtime_c = moe_engine_storage_constants_or_static(engine);
    const moe_storage_constants_t* c = &runtime_c;
    const moe_attn_config_t attn_cfg = moe_generation_attn_config(engine);

    if (!engine || !c || !s || c->hidden_size == 0 || token_id >= c->vocab_size) {
        moe_generation_set_error(stats, "invalid token forward arguments");
        return 0;
    }
    const uint64_t token_start_us = moe_profile_now_us();
    moe_profile_log(
        "forward_token_begin",
        "position=%u token=%u layers=%u hidden=%u vocab=%u prealloc_kv=%d",
        position, token_id, c->num_hidden_layers, c->hidden_size, c->vocab_size, pre_allocate_kv);
    moe_decode_scratch_resize(s, c, &attn_cfg, pre_allocate_kv, engine);
    moe_kv_cache_track_ram(engine, &s->kv_cache);  // ensure_capacity delta tracking
    float* hidden = s->hidden.data();
    engine->generation_phase.store(moe_GEN_PHASE_INIT, std::memory_order_relaxed);
    engine->generation_layer.store(0xffffffffu, std::memory_order_relaxed);
    engine->generation_token.store(position, std::memory_order_relaxed);

    // Fix 4: Reset adaptive prefetch depth for the new token
    engine->adaptive_prefetch_depth.store(2u, std::memory_order_relaxed);

    // Fix 10: Safely resize co_activation structures here (single-threaded forward loop)
    // BUGFIX 499: num_hidden_layers 범위 체크
    // BUGFIX 597: Check position == 0 before resize ★
    // Problem: Resize happens every token, causing memory allocation overhead
    // Solution: Only resize on first token (position == 0)
    if (position == 0 && c->num_hidden_layers > 0 &&
        c->num_hidden_layers < 10000 &&
        engine->co_activation_stats.size() < c->num_hidden_layers) {
        engine->co_activation_stats.resize(c->num_hidden_layers);
        engine->last_selected_experts.resize(c->num_hidden_layers);
    }
    const uint64_t embed_start_us = moe_profile_now_us();
    if (!moe_pc_engine_embed_token_f32(engine, token_id, hidden, c->hidden_size)) {
        moe_generation_set_error(stats, "failed to read token embedding");
        return 0;
    }
    const float embedding_scale = moe_forward_embedding_scale_value(engine, c->hidden_size);
    if (embedding_scale != 1.0f) {
        moe_scale_f32(hidden, embedding_scale, c->hidden_size);
    }
    moe_profile_log(
        "forward_embed",
        "position=%u token=%u hidden=%u scale=%.6g ms=%.3f",
        position, token_id, c->hidden_size, (double)embedding_scale,
        moe_profile_elapsed_ms(embed_start_us));
    moe_forward_trace_vec("embed", position, 0xffffffffu, hidden, c->hidden_size);

    for (uint32_t layer = 0; layer < c->num_hidden_layers; ++layer) {
        const uint64_t layer_start_us = moe_profile_now_us();
        // Pipeline Fix 2: Removed layer+2 lookahead to eliminate duplicate attention prefetch
        // layer+2 prefetch here + layer+1 prefetch after attention = same layer queued twice.
        // layer+1 prefetch alone provides sufficient DMA overlap during MLP compute.
        engine->generation_layer.store(layer, std::memory_order_relaxed);
        engine->generation_phase.store(moe_GEN_PHASE_ATTENTION, std::memory_order_relaxed);
        const int split_post_norm_block =
            attn_cfg.type != moe_ATTN_TYPE_MLA &&
            moe_layer_uses_split_post_norm_block(engine, (int32_t)layer);
        const moe_attn_config_t layer_attn_cfg =
            moe_generation_layer_attn_config(engine, (int32_t)layer);
        if (split_post_norm_block) {
            const uint64_t attn_start_us = moe_profile_now_us();
            std::memcpy(s->residual.data(), hidden, (size_t)c->hidden_size * sizeof(float));
            const uint64_t split_input_norm_start_us = moe_profile_now_us();
            if (!moe_pc_engine_raw_rmsnorm_f32(
                    engine, (int32_t)layer, moe_RAW_TENSOR_INPUT_LAYER_NORM,
                    hidden, c->hidden_size, s->norm.data(), c->hidden_size, layer_attn_cfg.rms_norm_eps)) {
                moe_generation_set_error(stats, "input rmsnorm failed");
                return 0;
            }
            moe_profile_log(
                "forward_split_input_norm",
                "position=%u layer=%u ms=%.3f",
                position, layer, moe_profile_elapsed_ms(split_input_norm_start_us));
            const uint64_t split_standard_attn_start_us = moe_profile_now_us();
            if (!moe_layer_standard_attention_f32(
                    engine, (int32_t)layer, s, hidden, c->hidden_size, position,
                    layer_attn_cfg, s->norm.data(), 0)) {
                moe_generation_set_error(stats, "attention block failed");
                return 0;
            }
            moe_profile_log(
                "forward_split_standard_attention",
                "position=%u layer=%u ms=%.3f",
                position, layer, moe_profile_elapsed_ms(split_standard_attn_start_us));
            const uint64_t split_post_norm_start_us = moe_profile_now_us();
            const float* attn_residual = s->attn.data();
            const int graph_residual_first =
                moe_graph_ir_attention_residual_before_ffn_norm(engine, (int32_t)layer);
            const int post_norm_used = graph_residual_first ? 0 :
                moe_layer_post_attention_output_norm_f32(
                    engine, (int32_t)layer, s->attn.data(), c->hidden_size,
                    s->norm.data(), layer_attn_cfg.rms_norm_eps);
            if (post_norm_used) {
                attn_residual = s->norm.data();
            }
            moe_profile_log(
                "forward_split_post_attention_norm",
                "position=%u layer=%u used=%d graph_residual_first=%d ms=%.3f",
                position, layer, post_norm_used, graph_residual_first,
                moe_profile_elapsed_ms(split_post_norm_start_us));
            const uint64_t split_residual_start_us = moe_profile_now_us();
            for (uint32_t i = 0; i < c->hidden_size; ++i) {
                hidden[i] = s->residual[i] + attn_residual[i];
            }
            moe_profile_log(
                "forward_split_residual",
                "position=%u layer=%u ms=%.3f",
                position, layer, moe_profile_elapsed_ms(split_residual_start_us));
            moe_forward_trace_vec("after_attention", position, layer, hidden, c->hidden_size);
            const double attention_ms = moe_profile_elapsed_ms(attn_start_us);
            const int is_moe = layer >= c->first_moe_layer && layer <= c->last_moe_layer;
            if (layer + 1u < c->num_hidden_layers &&
                moe_can_enqueue_attention_lookahead(engine, (int32_t)(layer + 1u))) {
                moe_enqueue_attention_prefetch(engine, (int32_t)(layer + 1u));
            }
            if (!moe_layer_ffn_norm_f32(
                    engine, (int32_t)layer, hidden, c->hidden_size,
                    s->norm.data(), layer_attn_cfg.rms_norm_eps)) {
                std::memcpy(s->norm.data(), hidden, (size_t)c->hidden_size * sizeof(float));
            }
            save_gate_input_snapshot(engine, layer, s->norm.data(), c->hidden_size);
            engine->generation_phase.store(moe_GEN_PHASE_MLP, std::memory_order_relaxed);
            const uint64_t mlp_start_us = moe_profile_now_us();
            const int ok = is_moe ?
                moe_layer_moe_mlp_f32(engine, (int32_t)layer, s, s->hidden.data(), c) :
                moe_layer_dense_mlp_f32(engine, (int32_t)layer, s, s->hidden.data(), c->hidden_size);
            if (!ok) {
                moe_generation_set_error(stats, "mlp/moe block failed");
                return 0;
            }
            const double mlp_ms = moe_profile_elapsed_ms(mlp_start_us);
            moe_forward_trace_vec("after_mlp", position, layer, hidden, c->hidden_size);
            if (!moe_layer_output_scale_f32(engine, (int32_t)layer, hidden, c->hidden_size)) {
                moe_generation_set_error(stats, "layer output scale failed");
                return 0;
            }
            moe_forward_trace_vec("after_output_scale", position, layer, hidden, c->hidden_size);
            moe_profile_log(
                "forward_layer",
                "position=%u layer=%u split=1 is_moe=%d attention_ms=%.3f mlp_ms=%.3f total_ms=%.3f",
                position, layer, is_moe, attention_ms, mlp_ms, moe_profile_elapsed_ms(layer_start_us));
            continue;
        }
        const uint64_t attn_start_us = moe_profile_now_us();
        if (!moe_layer_attention_f32(engine, (int32_t)layer, s, s->hidden.data(), c->hidden_size, position)) {
            moe_generation_set_error(stats, "attention block failed");
            return 0;
        }
        moe_forward_trace_vec("after_attention", position, layer, hidden, c->hidden_size);
        if (!moe_pc_engine_raw_rmsnorm_f32(
                engine, (int32_t)layer, moe_RAW_TENSOR_POST_ATTENTION_LAYER_NORM,
                hidden, c->hidden_size, s->norm.data(), c->hidden_size, layer_attn_cfg.rms_norm_eps)) {
            moe_generation_set_error(stats, "post-attention rmsnorm failed");
            return 0;
        }
        const double attention_ms = moe_profile_elapsed_ms(attn_start_us);
        const int is_moe = layer >= c->first_moe_layer && layer <= c->last_moe_layer;

        // Next-layer attention VRAM prefetch: enqueue layer+1 attention spans
        // so DMA overlaps with current layer's MLP compute.
        // Pipeline Fix 4: Add VRAM budget check to prevent evicting expert prefetch
        if (layer + 1u < c->num_hidden_layers &&
            moe_can_enqueue_attention_lookahead(engine, (int32_t)(layer + 1u))) {
            moe_enqueue_attention_prefetch(engine, (int32_t)(layer + 1u));
        }

        // Save hidden state snapshot for FATE predictor (gate input)
        // Note: s->norm holds the post-attention rmsnorm output which is the input to the gate
        save_gate_input_snapshot(engine, layer, s->norm.data(), c->hidden_size);

        engine->generation_phase.store(moe_GEN_PHASE_MLP, std::memory_order_relaxed);
        const uint64_t mlp_start_us = moe_profile_now_us();
        const int ok = is_moe ?
            moe_layer_moe_mlp_f32(engine, (int32_t)layer, s, s->hidden.data(), c) :
            moe_layer_dense_mlp_f32(engine, (int32_t)layer, s, s->hidden.data(), c->hidden_size);
        if (!ok) {
            moe_generation_set_error(stats, "mlp/moe block failed");
            return 0;
        }
        moe_forward_trace_vec("after_mlp", position, layer, hidden, c->hidden_size);
        moe_profile_log(
            "forward_layer",
            "position=%u layer=%u split=0 is_moe=%d attention_ms=%.3f mlp_ms=%.3f total_ms=%.3f",
            position, layer, is_moe, attention_ms,
            moe_profile_elapsed_ms(mlp_start_us), moe_profile_elapsed_ms(layer_start_us));
    }

    const uint64_t final_norm_start_us = moe_profile_now_us();
    if (!moe_pc_engine_raw_rmsnorm_f32(
            engine, (int32_t)c->num_hidden_layers, moe_RAW_TENSOR_INPUT_LAYER_NORM,
            hidden, c->hidden_size, s->norm.data(), c->hidden_size, attn_cfg.rms_norm_eps)) {
        moe_generation_set_error(stats, "final rmsnorm failed");
        return 0;
    }
    s->hidden.swap(s->norm);
    moe_forward_trace_vec("final_norm", position, c->num_hidden_layers, s->hidden.data(), c->hidden_size);
    moe_profile_log(
        "forward_token_end",
        "position=%u token=%u final_norm_ms=%.3f total_ms=%.3f",
        position, token_id, moe_profile_elapsed_ms(final_norm_start_us),
        moe_profile_elapsed_ms(token_start_us));

    return 1;
}
