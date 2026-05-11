static void moe_forward_trace_vec(
    const char* stage,
    uint32_t position,
    uint32_t layer,
    const float* values,
    uint32_t count
) {
#ifdef STORAGELLM_PPL_DIAG_ENABLED
    storagellm_ppl_diag_forward_vec(stage, position, layer, values, count);
#endif
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

static void moe_forward_trace_generation_contract(
    const moe_pc_engine* engine,
    const moe_storage_constants_t* c
) {
    if (!engine || !c || !moe_profile_enabled()) {
        return;
    }
    static std::atomic<uint32_t> logged{0};
    if (logged.fetch_add(1, std::memory_order_relaxed) != 0) {
        return;
    }
    std::string token_embedding;
    std::string lm_head;
    std::string final_norm;
    int tied = 0;
    uint64_t post_attn = 0;
    uint64_t expert_norm = 0;
    uint64_t post_ffw = 0;
    uint64_t post_ffw1 = 0;
    uint64_t post_ffw2 = 0;
    uint64_t output_scale = 0;
    uint64_t no_v_proj = 0;
    uint64_t v_norm = 0;
    uint64_t router_scale = 0;
    const uint32_t runtime_manifest_ready = engine->juju_runtime_execution_manifest_ready.load(std::memory_order_relaxed);
    const uint32_t access_plan_ready = engine->juju_runtime_access_plan_ready.load(std::memory_order_relaxed);
    const uint64_t hotset_count = engine->juju_runtime_access_hotset_count.load(std::memory_order_relaxed);
    const uint64_t file_group_count = engine->juju_runtime_access_file_group_count.load(std::memory_order_relaxed);
    const uint64_t executor_tensor_count = engine->juju_runtime_executor_tensor_count.load(std::memory_order_relaxed);
    const uint64_t layer_prefetch_count = engine->juju_runtime_layer_prefetch_plan_count.load(std::memory_order_relaxed);
    const uint32_t kv_layout_ready = engine->juju_kv_layout_contract_ready.load(std::memory_order_relaxed);
    const uint64_t kv_entry_dim = engine->juju_kv_layout_max_entry_dim.load(std::memory_order_relaxed);
    const uint64_t kv_page_tokens = engine->juju_kv_layout_page_size_tokens.load(std::memory_order_relaxed);
    (void)moe_graph_ir_tensor_binding_name(engine, "token_embedding", &token_embedding);
    (void)moe_graph_ir_tensor_binding_name(engine, "lm_head", &lm_head);
    (void)moe_graph_ir_tensor_binding_name(engine, "final_norm", &final_norm);
    (void)moe_json_get_bool_local(engine->offload_graph_ir_json, "lm_head_tied_to_token_embedding", &tied);
    (void)moe_json_get_u64_local(engine->offload_graph_ir_json, "layers_with_post_attention_norm", &post_attn);
    (void)moe_json_get_u64_local(engine->offload_graph_ir_json, "layers_with_expert_ffn_norm", &expert_norm);
    (void)moe_json_get_u64_local(engine->offload_graph_ir_json, "layers_with_post_ffw_norm", &post_ffw);
    (void)moe_json_get_u64_local(engine->offload_graph_ir_json, "layers_with_post_ffw_norm_1", &post_ffw1);
    (void)moe_json_get_u64_local(engine->offload_graph_ir_json, "layers_with_post_ffw_norm_2", &post_ffw2);
    (void)moe_json_get_u64_local(engine->offload_graph_ir_json, "layers_with_layer_output_scale", &output_scale);
    (void)moe_json_get_u64_local(engine->offload_graph_ir_json, "layers_without_v_projection", &no_v_proj);
    (void)moe_json_get_u64_local(engine->offload_graph_ir_json, "layers_with_v_norm", &v_norm);
    (void)moe_json_get_u64_local(engine->offload_graph_ir_json, "layers_with_router_scale", &router_scale);
    moe_profile_log(
        "generation_contract",
        "layers=%u hidden=%u vocab=%u token_embedding=%s lm_head=%s tied=%d final_norm=%s post_attn=%llu expert_norm=%llu post_ffw=%llu post_ffw1=%llu post_ffw2=%llu output_scale=%llu no_v_proj=%llu v_norm=%llu router_scale=%llu runtime_manifest=%u access_plan=%u hotset=%llu file_groups=%llu tensor_refs=%llu layer_prefetch=%llu kv_layout=%u kv_entry_dim=%llu kv_page_tokens=%llu",
        c->num_hidden_layers,
        c->hidden_size,
        c->vocab_size,
        token_embedding.empty() ? "" : token_embedding.c_str(),
        lm_head.empty() ? "" : lm_head.c_str(),
        tied,
        final_norm.empty() ? "" : final_norm.c_str(),
        (unsigned long long)post_attn,
        (unsigned long long)expert_norm,
        (unsigned long long)post_ffw,
        (unsigned long long)post_ffw1,
        (unsigned long long)post_ffw2,
        (unsigned long long)output_scale,
        (unsigned long long)no_v_proj,
        (unsigned long long)v_norm,
        (unsigned long long)router_scale,
        runtime_manifest_ready,
        access_plan_ready,
        (unsigned long long)hotset_count,
        (unsigned long long)file_group_count,
        (unsigned long long)executor_tensor_count,
        (unsigned long long)layer_prefetch_count,
        kv_layout_ready,
        (unsigned long long)kv_entry_dim,
        (unsigned long long)kv_page_tokens);
}

static void moe_forward_trace_runtime_resources(
    const moe_pc_engine* engine,
    const moe_decode_scratch* s,
    uint32_t position,
    uint32_t layer,
    const char* stage
) {
    if (!engine || !moe_profile_enabled()) {
        return;
    }
    const uint64_t ram_tier = tier_used(engine, moe_TIER_RAM);
    const uint64_t vram_tier = tier_used(engine, moe_TIER_VRAM);
    const uint64_t db_tier = tier_used(engine, moe_TIER_DB);
    const uint64_t ram_used = engine->ram_used_bytes.load(std::memory_order_relaxed);
    const uint64_t vram_used = engine->vram_used_bytes.load(std::memory_order_relaxed);
    const uint64_t db_used = engine->db_used_bytes.load(std::memory_order_relaxed);
    const uint64_t available_ram = storagellm::available_ram_bytes();
    const int qkv_only = engine && moe_engine_qkv_only_required(engine);
    const uint64_t kv_bytes = (s && !qkv_only) ? moe_kv_cache_allocated_bytes(s->kv_cache) : 0;
    const uint32_t kv_seq = s ? (qkv_only ? s->qkv_capacity_tokens : s->kv_cache.max_seq_len) : 0;
    const uint32_t kv_entry_dim = s ? (qkv_only ? s->qkv_seq_limit : s->kv_cache.entry_dim) : 0;
    moe_profile_log(
        "cpu_ram",
        "position=%u layer=%u stage=%s ram_tier=%llu vram_tier=%llu db_tier=%llu ram_used=%llu vram_used=%llu db_used=%llu available_ram=%llu cpu_workers=%u",
        position,
        layer,
        stage ? stage : "",
        (unsigned long long)ram_tier,
        (unsigned long long)vram_tier,
        (unsigned long long)db_tier,
        (unsigned long long)ram_used,
        (unsigned long long)vram_used,
        (unsigned long long)db_used,
        (unsigned long long)available_ram,
        engine->cpu_active_workers.load(std::memory_order_relaxed));
    moe_profile_log(
        qkv_only ? "qkv_cache" : "kv_cache",
        "position=%u layer=%u stage=%s bytes=%llu tracked_ram=%llu seq_len=%u max_seq_len=%u entry_dim=%u layout=%s",
        position,
        layer,
        stage ? stage : "",
        (unsigned long long)kv_bytes,
        (unsigned long long)((s && !qkv_only) ? s->kv_cache.tracked_ram_bytes : 0),
        s ? (qkv_only ? s->qkv_seq_len : s->kv_cache.seq_len) : 0,
        kv_seq,
        kv_entry_dim,
        qkv_only ? "qkv_layer_head_state" : "position_major_layer_contiguous");
    moe_profile_log(
        "io_pipeline",
        "position=%u layer=%u stage=%s disk_depth=%u pinned_depth=%u gpu_depth=%u total_depth=%u disk_inflight=%u pinned_inflight=%u gpu_inflight=%u queued=%llu prefetch=%llu disk_to_ram=%llu ram_to_pinned=%llu pinned_to_gpu=%llu direct_to_gpu=%llu",
        position,
        layer,
        stage ? stage : "",
        engine->disk_stage.depth.load(std::memory_order_relaxed),
        engine->pinned_stage.depth.load(std::memory_order_relaxed),
        engine->gpu_stage.depth.load(std::memory_order_relaxed),
        engine->total_queue_depth.load(std::memory_order_relaxed),
        engine->disk_stage.inflight.load(std::memory_order_relaxed),
        engine->pinned_stage.inflight.load(std::memory_order_relaxed),
        engine->gpu_stage.inflight.load(std::memory_order_relaxed),
        (unsigned long long)engine->io_atomic.queued_requests.load(std::memory_order_relaxed),
        (unsigned long long)engine->io_atomic.bytes_prefetched.load(std::memory_order_relaxed),
        (unsigned long long)engine->io_atomic.bytes_disk_to_ram.load(std::memory_order_relaxed),
        (unsigned long long)engine->io_atomic.bytes_ram_to_pinned.load(std::memory_order_relaxed),
        (unsigned long long)engine->io_atomic.bytes_pinned_to_gpu.load(std::memory_order_relaxed),
        (unsigned long long)engine->io_atomic.bytes_direct_to_gpu.load(std::memory_order_relaxed));
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
    if (!moe_cpu_storage_forward_allowed(engine, stats)) {
        return 0;
    }
    const uint64_t token_start_us = moe_profile_now_us();
    moe_profile_log(
        "forward_token_begin",
        "position=%u token=%u layers=%u hidden=%u vocab=%u prealloc_kv=%d",
        position, token_id, c->num_hidden_layers, c->hidden_size, c->vocab_size, pre_allocate_kv);
    moe_forward_trace_generation_contract(engine, c);
    if (!moe_decode_scratch_resize(s, c, &attn_cfg, pre_allocate_kv, engine)) {
        moe_generation_set_error(stats, "decode scratch allocation failed");
        return 0;
    }
    if (!moe_engine_qkv_only_required(engine)) {
        moe_kv_cache_track_ram(engine, &s->kv_cache);  // ensure_capacity delta tracking
    }
    moe_forward_trace_runtime_resources(engine, s, position, 0xffffffffu, "after_scratch");
    moe_cpu_storage_begin_token(engine, position, token_id);
    float* hidden = s->hidden.data();
    engine->generation_phase.store(moe_GEN_PHASE_INIT, std::memory_order_relaxed);
    engine->generation_layer.store(0xffffffffu, std::memory_order_relaxed);
    engine->generation_token.store(position, std::memory_order_relaxed);

    // Fix 4: Reset adaptive prefetch depth for the new token
    engine->adaptive_prefetch_depth.store(2u, std::memory_order_relaxed);
    // BUGFIX Problem F stage 2: Reset per-session depth too ★★★
    s->session_adaptive_prefetch_depth = 2u;
    s->session_router_entropy_milli = 0u;

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
    // moe_pc_engine_embed_token_f32 returns an already scaled embedding row.
    const float embedding_scale = moe_embedding_scale_value(engine, c->hidden_size);
    moe_profile_log(
        "forward_embed",
        "position=%u token=%u hidden=%u scale=%.6g scale_applied=1 ms=%.3f",
        position, token_id, c->hidden_size, (double)embedding_scale,
        moe_profile_elapsed_ms(embed_start_us));
    moe_forward_trace_vec("embed", position, 0xffffffffu, hidden, c->hidden_size);
    if (!moe_prepare_ple_inputs_f32(engine, c, s, token_id, attn_cfg.rms_norm_eps)) {
        moe_generation_set_error(stats, "per-layer input prepare failed");
        return 0;
    }

    for (uint32_t layer = 0; layer < c->num_hidden_layers; ++layer) {
        moe_cpu_storage_begin_layer(engine, layer, c);
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
            const int post_attention_norm_required =
                moe_layer_has_post_attention_norm_weight(engine, (int32_t)layer);
            const int post_norm_used = graph_residual_first ? 0 :
                moe_layer_post_attention_output_norm_f32(
                    engine, (int32_t)layer, s->attn.data(), c->hidden_size,
                    s->norm.data(), layer_attn_cfg.rms_norm_eps);
            if (!graph_residual_first && post_attention_norm_required && !post_norm_used) {
                moe_generation_set_error(stats, "post-attention norm tensor is required but could not be mapped");
                return 0;
            }
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
            const int is_moe = moe_layer_is_moe_runtime(engine, c, (int32_t)layer);
            if (layer + 1u < c->num_hidden_layers &&
                moe_can_enqueue_attention_lookahead(engine, (int32_t)(layer + 1u))) {
                moe_enqueue_attention_prefetch(engine, (int32_t)(layer + 1u));
            }
            if (!moe_layer_ffn_norm_f32(
                    engine, (int32_t)layer, hidden, c->hidden_size,
                    s->norm.data(), layer_attn_cfg.rms_norm_eps)) {
                if (moe_layer_has_ffn_norm_weight(engine, (int32_t)layer)) {
                    moe_generation_set_error(stats, "ffn norm tensor is required but could not be mapped");
                    return 0;
                }
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
            if (!moe_layer_apply_ple_f32(
                    engine, (int32_t)layer, s, hidden, c->hidden_size,
                    c->num_hidden_layers, layer_attn_cfg.rms_norm_eps)) {
                moe_generation_set_error(stats, "per-layer input block failed");
                return 0;
            }
            moe_forward_trace_vec("after_ple", position, layer, hidden, c->hidden_size);
            if (!moe_graph_ir_skip_layer_output_scale(engine)) {
                if (!moe_layer_output_scale_f32(engine, (int32_t)layer, hidden, c->hidden_size)) {
                    moe_generation_set_error(stats, "layer output scale failed");
                    return 0;
                }
                moe_forward_trace_vec("after_output_scale", position, layer, hidden, c->hidden_size);
            }
            moe_profile_log(
                "forward_layer",
                "position=%u layer=%u split=1 is_moe=%d attention_ms=%.3f mlp_ms=%.3f total_ms=%.3f",
                position, layer, is_moe, attention_ms, mlp_ms, moe_profile_elapsed_ms(layer_start_us));
            moe_forward_trace_runtime_resources(engine, s, position, layer, "after_layer");
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
        const int is_moe = moe_layer_is_moe_runtime(engine, c, (int32_t)layer);

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
        if (!moe_layer_apply_ple_f32(
                engine, (int32_t)layer, s, hidden, c->hidden_size,
                c->num_hidden_layers, layer_attn_cfg.rms_norm_eps)) {
            moe_generation_set_error(stats, "per-layer input block failed");
            return 0;
        }
        moe_forward_trace_vec("after_ple", position, layer, hidden, c->hidden_size);
        // BUGFIX: Missing output scale in standard path
        // Problem: split_post_norm path applied moe_layer_output_scale_f32 but
        // standard path did NOT. Models with layer_output_scale.weight (Gemma4)
        // produced unscaled outputs on the standard attention path, causing
        // PPL drift that accumulated across all 79 layers.
        if (!moe_graph_ir_skip_layer_output_scale(engine)) {
            if (!moe_layer_output_scale_f32(engine, (int32_t)layer, hidden, c->hidden_size)) {
                moe_generation_set_error(stats, "layer output scale failed");
                return 0;
            }
            moe_forward_trace_vec("after_output_scale", position, layer, hidden, c->hidden_size);
        }
        moe_profile_log(
            "forward_layer",
            "position=%u layer=%u split=0 is_moe=%d attention_ms=%.3f mlp_ms=%.3f total_ms=%.3f",
            position, layer, is_moe, attention_ms,
            moe_profile_elapsed_ms(mlp_start_us), moe_profile_elapsed_ms(layer_start_us));
        moe_forward_trace_runtime_resources(engine, s, position, layer, "after_layer");
    }

    const uint64_t final_norm_start_us = moe_profile_now_us();
    if (!moe_final_norm_f32(
            engine, c, hidden, c->hidden_size,
            s->norm.data(), attn_cfg.rms_norm_eps)) {
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
    moe_forward_trace_runtime_resources(engine, s, position, c->num_hidden_layers, "after_token");

    return 1;
}
