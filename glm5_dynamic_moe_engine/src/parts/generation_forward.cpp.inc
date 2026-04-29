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
    if (!glm5_pc_engine_embed_token_f32(engine, token_id, hidden, c->hidden_size)) {
        glm5_generation_set_error(stats, "failed to read token embedding");
        return 0;
    }

    for (uint32_t layer = 0; layer < c->num_hidden_layers; ++layer) {
        // Fix #5: Lookahead attention prefetch: enqueue layer+2 attention spans
        // so DMA has time to complete before that layer's attention begins.
        // Current-layer synchronous mmap hints were too late.
        if (layer + 2u < c->num_hidden_layers &&
            glm5_can_enqueue_attention_lookahead(engine, (int32_t)(layer + 2u))) {
            glm5_enqueue_attention_prefetch(engine, (int32_t)(layer + 2u));
        }
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
        if (layer + 1u < c->num_hidden_layers) {
            glm5_enqueue_attention_prefetch(engine, (int32_t)(layer + 1u));
        }

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
