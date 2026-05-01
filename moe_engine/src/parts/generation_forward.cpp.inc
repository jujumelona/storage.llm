static float moe_forward_embedding_scale_value(const moe_pc_engine* engine, uint32_t hidden_size) {
    if (!engine || hidden_size == 0) {
        return 1.0f;
    }
    double value = 0.0;
    if (!engine->offload_gguf_metadata_json.empty() &&
        (moe_json_get_double_local(engine->offload_gguf_metadata_json, "embedding_scale", &value) ||
         moe_json_get_double_local(engine->offload_gguf_metadata_json, "embed_scale", &value) ||
         moe_json_get_double_local(engine->offload_gguf_metadata_json, "scale_emb", &value)) &&
        value > 0.0) {
        return (float)value;
    }
    if (moe_engine_graph_ir_mentions(engine, "sqrt_hidden_size") ||
        moe_engine_graph_ir_mentions(engine, "scale_embedding") ||
        moe_engine_metadata_mentions(engine, "sqrt_hidden_size") ||
        moe_engine_metadata_mentions(engine, "scale_embedding")) {
        return std::sqrt((float)hidden_size);
    }
    if (moe_engine_has_common_tensor_suffix(engine, 0, "layer_output_scale.weight") &&
        moe_engine_has_common_tensor_suffix(engine, 0, "ffn_gate_inp.scale")) {
        return std::sqrt((float)hidden_size);
    }
    return 1.0f;
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
    moe_decode_scratch_resize(s, c, &attn_cfg, pre_allocate_kv, engine);
    moe_kv_cache_track_ram(engine, &s->kv_cache);  // ensure_capacity delta tracking
    float* hidden = s->hidden.data();
    engine->generation_phase.store(moe_GEN_PHASE_INIT, std::memory_order_relaxed);
    engine->generation_layer.store(0xffffffffu, std::memory_order_relaxed);
    engine->generation_token.store(position, std::memory_order_relaxed);

    // Fix 4: Reset adaptive prefetch depth for the new token
    engine->adaptive_prefetch_depth.store(2u, std::memory_order_relaxed);

    // Fix 10: Safely resize co_activation structures here (single-threaded forward loop)
    if (position == 0 && engine->co_activation_stats.size() < c->num_hidden_layers) {
        engine->co_activation_stats.resize(c->num_hidden_layers);
        engine->last_selected_experts.resize(c->num_hidden_layers);
    }
    if (!moe_pc_engine_embed_token_f32(engine, token_id, hidden, c->hidden_size)) {
        moe_generation_set_error(stats, "failed to read token embedding");
        return 0;
    }
    const float embedding_scale = moe_forward_embedding_scale_value(engine, c->hidden_size);
    if (embedding_scale != 1.0f) {
        moe_scale_f32(hidden, embedding_scale, c->hidden_size);
    }

    for (uint32_t layer = 0; layer < c->num_hidden_layers; ++layer) {
        // Pipeline Fix 2: Removed layer+2 lookahead to eliminate duplicate attention prefetch
        // layer+2 prefetch here + layer+1 prefetch after attention = same layer queued twice.
        // layer+1 prefetch alone provides sufficient DMA overlap during MLP compute.
        engine->generation_layer.store(layer, std::memory_order_relaxed);
        engine->generation_phase.store(moe_GEN_PHASE_ATTENTION, std::memory_order_relaxed);
        const int split_post_norm_block =
            attn_cfg.type != moe_ATTN_TYPE_MLA &&
            moe_layer_uses_split_post_norm_block(engine, (int32_t)layer);
        if (split_post_norm_block) {
            std::memcpy(s->residual.data(), hidden, (size_t)c->hidden_size * sizeof(float));
            if (!moe_pc_engine_raw_rmsnorm_f32(
                    engine, (int32_t)layer, moe_RAW_TENSOR_INPUT_LAYER_NORM,
                    hidden, c->hidden_size, s->norm.data(), c->hidden_size, attn_cfg.rms_norm_eps)) {
                moe_generation_set_error(stats, "input rmsnorm failed");
                return 0;
            }
            if (!moe_layer_standard_attention_f32(
                    engine, (int32_t)layer, s, hidden, c->hidden_size, position,
                    attn_cfg, s->norm.data(), 0)) {
                moe_generation_set_error(stats, "attention block failed");
                return 0;
            }
            const float* attn_residual = s->attn.data();
            if (moe_layer_post_attention_output_norm_f32(
                    engine, (int32_t)layer, s->attn.data(), c->hidden_size,
                    s->norm.data(), attn_cfg.rms_norm_eps)) {
                attn_residual = s->norm.data();
            }
            for (uint32_t i = 0; i < c->hidden_size; ++i) {
                hidden[i] = s->residual[i] + attn_residual[i];
            }
            const int is_moe = layer >= c->first_moe_layer && layer <= c->last_moe_layer;
            if (layer + 1u < c->num_hidden_layers &&
                moe_can_enqueue_attention_lookahead(engine, (int32_t)(layer + 1u))) {
                moe_enqueue_attention_prefetch(engine, (int32_t)(layer + 1u));
            }
            if (!moe_layer_ffn_norm_f32(
                    engine, (int32_t)layer, hidden, c->hidden_size,
                    s->norm.data(), attn_cfg.rms_norm_eps)) {
                std::memcpy(s->norm.data(), hidden, (size_t)c->hidden_size * sizeof(float));
            }
            save_gate_input_snapshot(engine, layer, s->norm.data(), c->hidden_size);
            engine->generation_phase.store(moe_GEN_PHASE_MLP, std::memory_order_relaxed);
            const int ok = is_moe ?
                moe_layer_moe_mlp_f32(engine, (int32_t)layer, s, s->hidden.data(), c) :
                moe_layer_dense_mlp_f32(engine, (int32_t)layer, s, s->hidden.data(), c->hidden_size);
            if (!ok || !moe_layer_output_scale_f32(engine, (int32_t)layer, hidden, c->hidden_size)) {
                moe_generation_set_error(stats, "mlp/moe block failed");
                return 0;
            }
            continue;
        }
        if (!moe_layer_attention_f32(engine, (int32_t)layer, s, s->hidden.data(), c->hidden_size, position)) {
            moe_generation_set_error(stats, "attention block failed");
            return 0;
        }
        if (!moe_pc_engine_raw_rmsnorm_f32(
                engine, (int32_t)layer, moe_RAW_TENSOR_POST_ATTENTION_LAYER_NORM,
                hidden, c->hidden_size, s->norm.data(), c->hidden_size, attn_cfg.rms_norm_eps)) {
            moe_generation_set_error(stats, "post-attention rmsnorm failed");
            return 0;
        }
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
        const int ok = is_moe ?
            moe_layer_moe_mlp_f32(engine, (int32_t)layer, s, s->hidden.data(), c) :
            moe_layer_dense_mlp_f32(engine, (int32_t)layer, s, s->hidden.data(), c->hidden_size);
        if (!ok) {
            moe_generation_set_error(stats, "mlp/moe block failed");
            return 0;
        }
    }

    if (!moe_pc_engine_raw_rmsnorm_f32(
            engine, (int32_t)c->num_hidden_layers, moe_RAW_TENSOR_INPUT_LAYER_NORM,
            hidden, c->hidden_size, s->norm.data(), c->hidden_size, attn_cfg.rms_norm_eps)) {
        moe_generation_set_error(stats, "final rmsnorm failed");
        return 0;
    }
    s->hidden.swap(s->norm);

    return 1;
}
