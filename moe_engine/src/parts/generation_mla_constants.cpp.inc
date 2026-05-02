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

static int moe_engine_uses_gemma4_attention(const moe_pc_engine* engine) {
    if (!engine) {
        return 0;
    }
    const std::string arch = moe_lower_ascii_copy(engine->dynamic_architecture);
    const std::string model = moe_lower_ascii_copy(engine->dynamic_model_id);
    if (arch.find("gemma4") != std::string::npos ||
        arch.find("gemma-4") != std::string::npos ||
        model.find("gemma4") != std::string::npos ||
        model.find("gemma-4") != std::string::npos) {
        return 1;
    }
    return moe_engine_metadata_mentions(engine, "gemma4") ||
        moe_engine_metadata_mentions(engine, "gemma-4") ? 1 : 0;
}

static int moe_gemma4_layer_is_full_attention(int32_t layer) {
    return layer >= 0 && (layer % 6) == 5 ? 1 : 0;
}

static moe_attn_config_t moe_generation_layer_attn_config(
    const moe_pc_engine* engine,
    int32_t layer
) {
    moe_attn_config_t cfg = moe_generation_attn_config(engine);
    if (!engine || cfg.type == moe_ATTN_TYPE_MLA || !moe_engine_uses_gemma4_attention(engine)) {
        return cfg;
    }

    uint64_t value = 0;
    double dvalue = 0.0;
    const uint32_t base_num_heads =
        moe_json_get_u64_local(engine->offload_gguf_metadata_json, "num_attention_heads", &value) && value > 0 ?
        (uint32_t)std::min<uint64_t>(value, UINT32_MAX) : cfg.num_heads;
    const uint32_t base_num_kv_heads =
        moe_json_get_u64_local(engine->offload_gguf_metadata_json, "num_key_value_heads", &value) && value > 0 ?
        (uint32_t)std::min<uint64_t>(value, UINT32_MAX) : cfg.num_kv_heads;
    const uint32_t base_head_dim =
        moe_json_get_u64_local(engine->offload_gguf_metadata_json, "head_dim", &value) && value > 0 ?
        (uint32_t)std::min<uint64_t>(value, UINT32_MAX) : (cfg.head_dim ? cfg.head_dim : 256u);
    const uint32_t global_head_dim =
        moe_json_get_u64_local(engine->offload_gguf_metadata_json, "global_head_dim", &value) && value > 0 ?
        (uint32_t)std::min<uint64_t>(value, UINT32_MAX) : std::max<uint32_t>(base_head_dim, cfg.head_dim);
    const uint32_t global_num_kv_heads =
        moe_json_get_u64_local(engine->offload_gguf_metadata_json, "num_global_key_value_heads", &value) && value > 0 ?
        (uint32_t)std::min<uint64_t>(value, UINT32_MAX) : std::max<uint32_t>(1u, base_num_kv_heads / 4u);

    cfg.num_heads = base_num_heads ? base_num_heads : cfg.num_heads;
    cfg.v_head_dim = 0;
    if (moe_gemma4_layer_is_full_attention(layer)) {
        cfg.head_dim = global_head_dim ? global_head_dim : cfg.head_dim;
        cfg.num_kv_heads = global_num_kv_heads ? global_num_kv_heads : cfg.num_kv_heads;
        const float partial =
            moe_json_get_double_local(engine->offload_gguf_metadata_json, "partial_rotary_factor", &dvalue) &&
            dvalue > 0.0 && dvalue <= 1.0 ? (float)dvalue : 0.25f;
        cfg.qk_rope_head_dim = std::max<uint32_t>(2u, (uint32_t)std::round((float)cfg.head_dim * partial));
        cfg.rope_theta = 1000000.0f;
    } else {
        cfg.head_dim = base_head_dim ? base_head_dim : cfg.head_dim;
        cfg.num_kv_heads = base_num_kv_heads ? base_num_kv_heads : cfg.num_kv_heads;
        cfg.qk_rope_head_dim = cfg.head_dim;
        cfg.rope_theta = 10000.0f;
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
