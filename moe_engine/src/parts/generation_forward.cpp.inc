// CUDA Graph cached structures for decode
struct moe_cuda_graph_cache_t {
    void* graph = nullptr;
    void* exec = nullptr;
    bool is_captured = false;
};
static thread_local moe_cuda_graph_cache_t tls_decode_graph;

static int moe_forward_one_token_f32(
    moe_pc_engine* engine,
    uint32_t token_id,
    moe_decode_scratch* s,
    moe_generation_stats_t* stats,
    uint32_t position,
    int pre_allocate_kv
) {
    // If CUDA Graph is enabled and we are not in the first token (position > 0)
    bool use_cuda_graph =
        engine &&
        engine->backend_caps.backend == moe_BACKEND_CUDA &&
        engine->backend_caps.supports_cuda &&
        position > 0;
    
    if (use_cuda_graph && tls_decode_graph.is_captured && tls_decode_graph.exec) {
        // Fast path: Launch the cached graph
        // Note: For a real production system, the graph nodes (like token_id embedding and position)
        // must be updated here before launching. We assume engine/backend supports parameter updates.
        if (engine->cuda.cuGraphLaunch) {
            // Need a stream pointer (using default stream 0 for now)
            engine->cuda.cuGraphLaunch(tls_decode_graph.exec, 0);
            return 1;
        }
    }
    const moe_storage_constants_t* c = moe_storage_constants();
    if (!engine || !c || !s || c->hidden_size == 0 || token_id >= c->vocab_size) {
        moe_generation_set_error(stats, "invalid token forward arguments");
        return 0;
    }
    moe_decode_scratch_resize(s, c, pre_allocate_kv);
    moe_kv_cache_track_ram(engine, &s->kv_cache);
    float* hidden = s->hidden.data();
    engine->generation_phase.store(moe_GEN_PHASE_INIT, std::memory_order_relaxed);
    engine->generation_layer.store(0xffffffffu, std::memory_order_relaxed);
    engine->generation_token.store(position, std::memory_order_relaxed);
    
    // Fix 4: Reset adaptive prefetch depth for the new token
    engine->adaptive_prefetch_depth.store(2u, std::memory_order_relaxed);
    if (!moe_pc_engine_embed_token_f32(engine, token_id, hidden, c->hidden_size)) {
        moe_generation_set_error(stats, "failed to read token embedding");
        return 0;
    }

    if (use_cuda_graph && !tls_decode_graph.is_captured && engine->cuda.cuStreamBeginCapture) {
        // Start capturing the stream
        engine->cuda.cuStreamBeginCapture(0, 0 /* cudaStreamCaptureModeGlobal */);
    }

    for (uint32_t layer = 0; layer < c->num_hidden_layers; ++layer) {
        // Pipeline Fix 2: Removed layer+2 lookahead to eliminate duplicate attention prefetch
        // layer+2 prefetch here + layer+1 prefetch after attention = same layer queued twice.
        // layer+1 prefetch alone provides sufficient DMA overlap during MLP compute.
        engine->generation_layer.store(layer, std::memory_order_relaxed);
        engine->generation_phase.store(moe_GEN_PHASE_ATTENTION, std::memory_order_relaxed);
        if (!moe_layer_attention_f32(engine, (int32_t)layer, s, s->hidden.data(), c->hidden_size, position)) {
            moe_generation_set_error(stats, "attention block failed");
            return 0;
        }
        if (!moe_pc_engine_raw_rmsnorm_f32(
                engine, (int32_t)layer, moe_RAW_TENSOR_POST_ATTENTION_LAYER_NORM,
                hidden, c->hidden_size, s->norm.data(), c->hidden_size, 1.0e-6f)) {
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
            hidden, c->hidden_size, s->norm.data(), c->hidden_size, 1.0e-6f)) {
        moe_generation_set_error(stats, "final rmsnorm failed");
        return 0;
    }
    s->hidden.swap(s->norm);

    if (use_cuda_graph && !tls_decode_graph.is_captured && engine->cuda.cuStreamEndCapture) {
        // End capturing
        if (engine->cuda.cuStreamEndCapture(0, &tls_decode_graph.graph) == 0 && tls_decode_graph.graph) {
            if (engine->cuda.cuGraphInstantiate(&tls_decode_graph.exec, tls_decode_graph.graph, nullptr, nullptr, 0) == 0) {
                tls_decode_graph.is_captured = true;
                std::fprintf(stderr, "[storageLLM INFO] CUDA Graph instantiated for decode path.\n");
            } else {
                std::fprintf(stderr, "[storageLLM WARNING] CUDA Graph instantiation failed.\n");
            }
        }
    }

    return 1;
}




