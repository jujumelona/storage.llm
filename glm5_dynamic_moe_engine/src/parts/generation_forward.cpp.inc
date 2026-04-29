static int glm5_forward_one_token_f32(
    glm5_pc_engine* engine,
    uint32_t token_id,
    glm5_decode_scratch* s,
    glm5_generation_stats_t* stats,
    uint32_t position,
    int pre_allocate_kv
) {
    const glm5_storage_constants_t* c = glm5_storage_constants();
    if (!engine || !c || !s || c->hidden_size == 0 || token_id >= c->vocab_size) {
        glm5_generation_set_error(stats, "invalid token forward arguments");
        return 0;
    }
    glm5_decode_scratch_resize(s, c, pre_allocate_kv);
    glm5_kv_cache_track_ram(engine, &s->kv_cache);
    float* hidden = s->hidden.data();
    engine->generation_phase.store(GLM5_GEN_PHASE_INIT, std::memory_order_relaxed);
    engine->generation_layer.store(0xffffffffu, std::memory_order_relaxed);
    engine->generation_token.store(position, std::memory_order_relaxed);
    if (!glm5_pc_engine_embed_token_f32(engine, token_id, hidden, c->hidden_size)) {
        glm5_generation_set_error(stats, "failed to read token embedding");
        return 0;
    }

    for (uint32_t layer = 0; layer < c->num_hidden_layers; ++layer) {
        // Pipeline Fix 2: Removed layer+2 lookahead to eliminate duplicate attention prefetch
        // layer+2 prefetch here + layer+1 prefetch after attention = same layer queued twice.
        // layer+1 prefetch alone provides sufficient DMA overlap during MLP compute.
        engine->generation_layer.store(layer, std::memory_order_relaxed);
        engine->generation_phase.store(GLM5_GEN_PHASE_ATTENTION, std::memory_order_relaxed);
        if (!glm5_layer_attention_f32(engine, (int32_t)layer, s, s->hidden.data(), c->hidden_size, position)) {
            glm5_generation_set_error(stats, "attention block failed");
            return 0;
        }
        if (!glm5_pc_engine_raw_rmsnorm_f32(
                engine, (int32_t)layer, GLM5_RAW_TENSOR_POST_ATTENTION_LAYER_NORM,
                hidden, c->hidden_size, s->norm.data(), c->hidden_size, 1.0e-6f)) {
            glm5_generation_set_error(stats, "post-attention rmsnorm failed");
            return 0;
        }
        const int is_moe = layer >= c->first_moe_layer && layer <= c->last_moe_layer;

        // Next-layer attention VRAM prefetch: enqueue layer+1 attention spans
        // so DMA overlaps with current layer's MLP compute.
        // Pipeline Fix 4: Add VRAM budget check to prevent evicting expert prefetch
        if (layer + 1u < c->num_hidden_layers &&
            glm5_can_enqueue_attention_lookahead(engine, (int32_t)(layer + 1u))) {
            glm5_enqueue_attention_prefetch(engine, (int32_t)(layer + 1u));
        }

        engine->generation_phase.store(GLM5_GEN_PHASE_MLP, std::memory_order_relaxed);
        const int ok = is_moe ?
            glm5_layer_moe_mlp_f32(engine, (int32_t)layer, s, s->hidden.data(), c) :
            glm5_layer_dense_mlp_f32(engine, (int32_t)layer, s, s->hidden.data(), c->hidden_size);
        if (!ok) {
            glm5_generation_set_error(stats, "mlp/moe block failed");
            return 0;
        }
    }

    if (!glm5_pc_engine_raw_rmsnorm_f32(
            engine, (int32_t)c->num_hidden_layers, GLM5_RAW_TENSOR_INPUT_LAYER_NORM,
            hidden, c->hidden_size, s->norm.data(), c->hidden_size, 1.0e-6f)) {
        glm5_generation_set_error(stats, "final rmsnorm failed");
        return 0;
    }
    s->hidden.swap(s->norm);
    return 1;
}
