// MLA constants for the legacy metadata fallback path.
static constexpr uint32_t moe_MLA_KV_COMPRESSED_DIM = 512u;
static constexpr uint32_t moe_MLA_K_ROPE_DIM = 64u;
static constexpr uint32_t moe_MLA_KV_A_PROJ_DIM = moe_MLA_KV_COMPRESSED_DIM + moe_MLA_K_ROPE_DIM; // 576
static constexpr uint32_t moe_MLA_KV_ENTRY_DIM = moe_MLA_KV_A_PROJ_DIM;
static constexpr uint32_t moe_MLA_MAX_SEQ_LEN = 8192u;
static constexpr uint32_t moe_MLA_ROPE_TABLE_LEN = 16384u;
// Bonus: Enforce consistency between ROPE table and KV cache max length
static_assert(moe_MLA_ROPE_TABLE_LEN >= moe_MLA_MAX_SEQ_LEN,
              "ROPE table must cover at least MAX_SEQ_LEN positions");
static constexpr uint32_t moe_MLA_KV_INITIAL_CHUNK = 1024u;
static constexpr uint32_t moe_MLA_NUM_HEADS = 64u;
static constexpr uint32_t moe_MLA_HEAD_DIM = 256u;
static constexpr uint32_t moe_MLA_NOPE_DIM = 192u;
static constexpr uint32_t moe_MLA_ROPE_DIM = 64u;
static constexpr uint32_t moe_MLA_V_HEAD_DIM = 256u;
static constexpr uint32_t moe_MLA_Q_B_DIM = moe_MLA_NUM_HEADS * moe_MLA_HEAD_DIM; // 16384
static constexpr uint32_t moe_MLA_K_NOPE_TOTAL_DIM = moe_MLA_NUM_HEADS * moe_MLA_NOPE_DIM; // 12288
static constexpr uint32_t moe_MLA_V_TOTAL_DIM = moe_MLA_NUM_HEADS * moe_MLA_V_HEAD_DIM; // 16384
static_assert(moe_MLA_ROPE_DIM == moe_MLA_K_ROPE_DIM, "Q/K RoPE dims must match");
static_assert(moe_MLA_Q_B_DIM == 16384u, "Moe q_b projection width mismatch");
static_assert(moe_MLA_K_NOPE_TOTAL_DIM == 12288u, "Moe K_nope width mismatch");
static_assert(moe_MLA_V_TOTAL_DIM == 16384u, "Moe V width mismatch");
static constexpr float moe_ROPE_THETA = 500000.0f;
static constexpr uint32_t moe_MOE_BASE_TOP_EXPERTS = 8u;
static constexpr uint32_t moe_MOE_TOP_EXPERTS = 12u;

static moe_attn_config_t moe_legacy_mla_attn_config() {
    moe_attn_config_t cfg{};
    cfg.type = moe_ATTN_TYPE_MLA;
    cfg.num_heads = moe_MLA_NUM_HEADS;
    cfg.num_kv_heads = moe_MLA_NUM_HEADS;
    cfg.head_dim = moe_MLA_HEAD_DIM;
    cfg.v_head_dim = moe_MLA_V_HEAD_DIM;
    cfg.kv_lora_rank = moe_MLA_KV_COMPRESSED_DIM;
    cfg.qk_nope_head_dim = moe_MLA_NOPE_DIM;
    cfg.qk_rope_head_dim = moe_MLA_ROPE_DIM;
    cfg.q_lora_rank = 2048u;
    cfg.max_seq_len = moe_MLA_MAX_SEQ_LEN;
    cfg.rope_table_len = moe_MLA_ROPE_TABLE_LEN;
    cfg.rope_theta = moe_ROPE_THETA;
    cfg.rms_norm_eps = 1.0e-6f;
    return cfg;
}

static moe_attn_config_t moe_generation_attn_config(const moe_pc_engine* engine) {
    moe_attn_config_t cfg{};
    if (engine && engine->attn_config_valid.load(std::memory_order_relaxed)) {
        cfg = engine->attn_config;
    } else {
        cfg = moe_legacy_mla_attn_config();
    }
    if (cfg.type == moe_ATTN_TYPE_UNKNOWN) {
        if (cfg.kv_lora_rank) {
            cfg.type = moe_ATTN_TYPE_MLA;
        } else if (cfg.num_heads && cfg.num_kv_heads == 1u) {
            cfg.type = moe_ATTN_TYPE_MQA;
        } else if (cfg.num_heads && cfg.num_kv_heads && cfg.num_kv_heads < cfg.num_heads) {
            cfg.type = moe_ATTN_TYPE_GQA;
        } else if (cfg.num_heads) {
            cfg.type = moe_ATTN_TYPE_MHA;
        } else {
            cfg.type = moe_ATTN_TYPE_MLA;
        }
    }
    if (cfg.type == moe_ATTN_TYPE_MLA) {
        if (!cfg.num_heads) cfg.num_heads = moe_MLA_NUM_HEADS;
        if (!cfg.num_kv_heads) cfg.num_kv_heads = cfg.num_heads;
        if (!cfg.qk_rope_head_dim) cfg.qk_rope_head_dim = moe_MLA_ROPE_DIM;
        if (!cfg.qk_nope_head_dim) cfg.qk_nope_head_dim = moe_MLA_NOPE_DIM;
        if (!cfg.head_dim) cfg.head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim;
        if (!cfg.v_head_dim) cfg.v_head_dim = moe_MLA_V_HEAD_DIM;
        if (!cfg.kv_lora_rank) cfg.kv_lora_rank = moe_MLA_KV_COMPRESSED_DIM;
        if (!cfg.q_lora_rank) cfg.q_lora_rank = 2048u;
    } else {
        if (!cfg.num_kv_heads) cfg.num_kv_heads = cfg.num_heads;
        if (!cfg.v_head_dim) cfg.v_head_dim = cfg.head_dim;
        if (!cfg.qk_rope_head_dim) cfg.qk_rope_head_dim = cfg.head_dim;
        if (cfg.qk_rope_head_dim > cfg.head_dim) cfg.qk_rope_head_dim = cfg.head_dim;
        if (!cfg.qk_nope_head_dim && cfg.head_dim > cfg.qk_rope_head_dim) {
            cfg.qk_nope_head_dim = cfg.head_dim - cfg.qk_rope_head_dim;
        }
    }
    if (cfg.rms_norm_eps <= 0.0f) cfg.rms_norm_eps = 1.0e-6f;
    if (cfg.rope_theta <= 0.0f) cfg.rope_theta = moe_ROPE_THETA;
    if (!cfg.max_seq_len) cfg.max_seq_len = moe_MLA_MAX_SEQ_LEN;
    if (!cfg.rope_table_len || cfg.rope_table_len < cfg.max_seq_len) {
        cfg.rope_table_len = std::max<uint32_t>(moe_MLA_ROPE_TABLE_LEN, cfg.max_seq_len);
    }
    return cfg;
}

static int moe_engine_uses_hybrid_local_global_attention(const moe_pc_engine* engine) {
    if (!engine) {
        return 0;
    }
    if (engine->model_config_attention_valid.load(std::memory_order_relaxed) ||
        engine->model_config_layer_types_valid.load(std::memory_order_relaxed)) {
        return 1;
    }
    if (moe_engine_graph_ir_mentions(engine, "hybrid_local_global_attention") ||
        moe_engine_graph_ir_mentions(engine, "local_global_attention") ||
        moe_engine_graph_ir_mentions(engine, "global_attention") ||
        moe_engine_graph_ir_mentions(engine, "sliding_window_attention")) {
        return 1;
    }
    if (moe_engine_metadata_mentions(engine, "num_global_key_value_heads") ||
        moe_engine_metadata_mentions(engine, "global_head_dim") ||
        moe_engine_metadata_mentions(engine, "sliding_window_pattern") ||
        moe_engine_metadata_mentions(engine, "attention_chunk_size") ||
        moe_engine_metadata_mentions(engine, "rope_local_base_freq") ||
        moe_engine_metadata_mentions(engine, "rope_global_base_freq")) {
        return 1;
    }
    const int has_q_norm =
        moe_engine_has_common_tensor_suffix(engine, 0, "attn_q_norm.weight") ||
        moe_engine_has_common_tensor_suffix(engine, 0, "q_norm.weight");
    const int has_k_norm =
        moe_engine_has_common_tensor_suffix(engine, 0, "attn_k_norm.weight") ||
        moe_engine_has_common_tensor_suffix(engine, 0, "k_norm.weight");
    const int has_scaled_blocks =
        moe_engine_has_common_tensor_suffix(engine, 0, "post_attention_norm.weight") &&
        moe_engine_has_common_tensor_suffix(engine, 0, "post_ffw_norm.weight") &&
        moe_engine_has_common_tensor_suffix(engine, 0, "layer_output_scale.weight");
    return has_q_norm && has_k_norm && has_scaled_blocks ? 1 : 0;
}

static uint32_t moe_hybrid_attention_period_from_metadata(
    const moe_pc_engine* engine,
    uint32_t fallback
) {
    if (!engine || engine->offload_gguf_metadata_json.empty()) {
        return fallback;
    }
    uint64_t value = 0;
    if ((moe_json_get_u64_local(engine->offload_gguf_metadata_json, "full_attention_interval", &value) ||
         moe_json_get_u64_local(engine->offload_gguf_metadata_json, "global_attention_interval", &value) ||
         moe_json_get_u64_local(engine->offload_gguf_metadata_json, "sliding_window_pattern", &value) ||
         moe_json_get_u64_local(engine->offload_gguf_metadata_json, "attention_pattern_interval", &value) ||
         moe_json_get_u64_local(engine->offload_gguf_metadata_json, "local_global_attention_period", &value)) &&
        value > 0) {
        return (uint32_t)std::min<uint64_t>(value, UINT32_MAX);
    }
    return fallback;
}

static uint32_t moe_hybrid_attention_offset_from_metadata(
    const moe_pc_engine* engine,
    uint32_t fallback
) {
    if (!engine || engine->offload_gguf_metadata_json.empty()) {
        return fallback;
    }
    uint64_t value = 0;
    if ((moe_json_get_u64_local(engine->offload_gguf_metadata_json, "full_attention_offset", &value) ||
         moe_json_get_u64_local(engine->offload_gguf_metadata_json, "global_attention_offset", &value) ||
         moe_json_get_u64_local(engine->offload_gguf_metadata_json, "local_global_attention_offset", &value)) &&
        value <= UINT32_MAX) {
        return (uint32_t)value;
    }
    return fallback;
}

static int moe_hybrid_layer_is_full_attention(
    const moe_pc_engine* engine,
    int32_t layer
) {
    if (layer < 0) {
        return 0;
    }
    if (engine &&
        engine->model_config_layer_types_valid.load(std::memory_order_relaxed) &&
        (uint32_t)layer < engine->model_config_full_attention_layers.size()) {
        return engine->model_config_full_attention_layers[(uint32_t)layer] ? 1 : 0;
    }
    const uint32_t period = moe_hybrid_attention_period_from_metadata(engine, 0u);
    if (period == 0) {
        return 0;
    }
    const uint32_t fallback_offset = period > 0 ? period - 1u : 0u;
    const uint32_t offset = moe_hybrid_attention_offset_from_metadata(engine, fallback_offset) % period;
    return ((uint32_t)layer % period) == offset ? 1 : 0;
}

static moe_attn_config_t moe_generation_layer_attn_config(
    const moe_pc_engine* engine,
    int32_t layer
) {
    moe_attn_config_t cfg = moe_generation_attn_config(engine);
    if (!engine || cfg.type == moe_ATTN_TYPE_MLA || !moe_engine_uses_hybrid_local_global_attention(engine)) {
        return cfg;
    }

    uint64_t value = 0;
    double dvalue = 0.0;
    uint32_t base_num_heads = cfg.num_heads;
    uint32_t base_num_kv_heads = cfg.num_kv_heads;
    uint32_t base_head_dim = cfg.head_dim;
    uint32_t global_head_dim = cfg.head_dim;
    uint32_t global_num_kv_heads = cfg.num_kv_heads;
    if (engine->model_config_num_attention_heads) {
        base_num_heads = engine->model_config_num_attention_heads;
    } else if (moe_json_get_u64_local(engine->offload_gguf_metadata_json, "num_attention_heads", &value) && value > 0) {
        base_num_heads = (uint32_t)std::min<uint64_t>(value, UINT32_MAX);
    }
    if (engine->model_config_num_key_value_heads) {
        base_num_kv_heads = engine->model_config_num_key_value_heads;
    } else if (moe_json_get_u64_local(engine->offload_gguf_metadata_json, "num_key_value_heads", &value) && value > 0) {
        base_num_kv_heads = (uint32_t)std::min<uint64_t>(value, UINT32_MAX);
    }
    if (engine->model_config_head_dim) {
        base_head_dim = engine->model_config_head_dim;
    } else if (moe_json_get_u64_local(engine->offload_gguf_metadata_json, "head_dim", &value) && value > 0) {
        base_head_dim = (uint32_t)std::min<uint64_t>(value, UINT32_MAX);
    }
    if (engine->model_config_global_head_dim) {
        global_head_dim = engine->model_config_global_head_dim;
    } else if (moe_json_get_u64_local(engine->offload_gguf_metadata_json, "global_head_dim", &value) && value > 0) {
        global_head_dim = (uint32_t)std::min<uint64_t>(value, UINT32_MAX);
    }
    if (engine->model_config_num_global_key_value_heads) {
        global_num_kv_heads = engine->model_config_num_global_key_value_heads;
    } else if (moe_json_get_u64_local(engine->offload_gguf_metadata_json, "num_global_key_value_heads", &value) && value > 0) {
        global_num_kv_heads = (uint32_t)std::min<uint64_t>(value, UINT32_MAX);
    }

    cfg.num_heads = base_num_heads ? base_num_heads : cfg.num_heads;
    cfg.v_head_dim = 0;
    if (moe_hybrid_layer_is_full_attention(engine, layer)) {
        cfg.head_dim = global_head_dim ? global_head_dim : cfg.head_dim;
        cfg.num_kv_heads = global_num_kv_heads ? global_num_kv_heads : cfg.num_kv_heads;
        const float partial =
            engine->model_config_partial_rotary_factor > 0.0f && engine->model_config_partial_rotary_factor <= 1.0f ?
            engine->model_config_partial_rotary_factor :
            (moe_json_get_double_local(engine->offload_gguf_metadata_json, "partial_rotary_factor", &dvalue) &&
            dvalue > 0.0 && dvalue <= 1.0 ? (float)dvalue : 0.0f);
        if (partial > 0.0f) {
            cfg.qk_rope_head_dim = std::max<uint32_t>(2u, (uint32_t)std::round((float)cfg.head_dim * partial));
        }
        if (engine->model_config_full_rope_theta > 0.0f) {
            cfg.rope_theta = engine->model_config_full_rope_theta;
        } else if (moe_json_get_double_local(engine->offload_gguf_metadata_json, "full_rope_theta", &dvalue) ||
                   moe_json_get_double_local(engine->offload_gguf_metadata_json, "full_attention_rope_theta", &dvalue)) {
            cfg.rope_theta = (float)dvalue;
        }
    } else {
        cfg.head_dim = base_head_dim ? base_head_dim : cfg.head_dim;
        cfg.num_kv_heads = base_num_kv_heads ? base_num_kv_heads : cfg.num_kv_heads;
        cfg.qk_rope_head_dim = cfg.head_dim;
        if (engine->model_config_sliding_rope_theta > 0.0f) {
            cfg.rope_theta = engine->model_config_sliding_rope_theta;
        } else if (moe_json_get_double_local(engine->offload_gguf_metadata_json, "sliding_rope_theta", &dvalue) ||
                   moe_json_get_double_local(engine->offload_gguf_metadata_json, "sliding_attention_rope_theta", &dvalue)) {
            cfg.rope_theta = (float)dvalue;
        }
    }
    cfg.v_head_dim = cfg.head_dim;
    return cfg;
}

static uint32_t moe_attn_q_b_dim(const moe_attn_config_t& cfg) {
    return cfg.num_heads * cfg.head_dim;
}

static uint32_t moe_attn_v_total_dim(const moe_attn_config_t& cfg) {
    return cfg.num_heads * cfg.v_head_dim;
}

static uint32_t moe_attn_k_nope_total_dim(const moe_attn_config_t& cfg) {
    return cfg.num_heads * cfg.qk_nope_head_dim;
}

static uint32_t moe_attn_kv_a_proj_dim(const moe_attn_config_t& cfg) {
    if (cfg.type != moe_ATTN_TYPE_MLA) {
        return cfg.num_kv_heads * cfg.head_dim;
    }
    return cfg.kv_lora_rank + cfg.qk_rope_head_dim;
}

static uint32_t moe_attn_kv_entry_dim(const moe_attn_config_t& cfg) {
    if (cfg.type != moe_ATTN_TYPE_MLA) {
        return cfg.num_kv_heads * (cfg.head_dim + cfg.v_head_dim);
    }
    return moe_attn_kv_a_proj_dim(cfg);
}
