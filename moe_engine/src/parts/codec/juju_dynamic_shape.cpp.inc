// JUJU dynamic shape inference

static void moe_dynamic_shape_finalize_adapter(moe_pc_engine* engine) {
    if (!engine || !engine->dynamic_shape_valid.load(std::memory_order_relaxed)) {
        return;
    }
    uint32_t adapter = moe_FORWARD_ADAPTER_GGUF_GENERIC;
    int executable = 0;
    if (engine->offload_graph_ir_ready.load(std::memory_order_relaxed)) {
        adapter = moe_FORWARD_ADAPTER_GRAPH_IR;
        executable = moe_graph_ir_executor_compiled(engine);
    } else if (engine->attn_config_valid.load(std::memory_order_relaxed) &&
               engine->attn_config.type == moe_ATTN_TYPE_MLA) {
        adapter = moe_FORWARD_ADAPTER_GGUF_GENERIC;
        executable = 1;
    } else if (!engine->gguf_common_tensor_slots.empty()) {
        adapter = moe_FORWARD_ADAPTER_GGUF_MOE;
        executable = 0;
    }
    engine->forward_adapter.store(adapter, std::memory_order_relaxed);
    engine->forward_adapter_executable.store(executable, std::memory_order_relaxed);
    engine->attention_projector_valid.store(-1, std::memory_order_relaxed);
}

struct moe_juju_runtime_probe_t {
    moe_model_shape_t shape{};
    moe_attn_config_t attn{};
    std::string architecture;
    std::string model_id;
    uint64_t q_proj_rows = 0;
    uint64_t k_proj_rows = 0;
    uint64_t v_proj_rows = 0;
    uint64_t o_proj_cols = 0;
    uint64_t q_norm_rows = 0;
    uint64_t k_norm_rows = 0;
    uint64_t q_a_rows = 0;
    uint64_t q_b_rows = 0;
    uint64_t q_b_cols = 0;
    uint64_t kv_a_rows = 0;
    uint64_t kv_b_rows = 0;
    uint64_t kv_b_cols = 0;
    bool saw_moe_tensor = false;
};

static int moe_name_has(const std::string& text, const char* needle) {
    return needle && needle[0] && text.find(needle) != std::string::npos;
}

static int moe_juju_name_is_mla_q_a(const std::string& lower) {
    return moe_name_has(lower, "q_a") && !moe_name_has(lower, "norm");
}

static int moe_juju_name_is_mla_q_b(const std::string& lower) {
    return moe_name_has(lower, "q_b") && !moe_name_has(lower, "norm");
}

static int moe_juju_name_is_mla_kv_a(const std::string& lower) {
    return moe_name_has(lower, "kv_a") && !moe_name_has(lower, "norm");
}

static int moe_juju_name_is_mla_kv_b(const std::string& lower) {
    return moe_name_has(lower, "kv_b") && !moe_name_has(lower, "norm");
}

static int moe_juju_name_is_attn_o(const std::string& lower) {
    return moe_name_has(lower, "o_proj.weight") ||
        moe_name_has(lower, "attn_o.weight") ||
        moe_name_has(lower, "attn_output.weight") ||
        moe_name_has(lower, "attention.wo.weight");
}

static int moe_juju_name_is_std_q(const std::string& lower) {
    return !moe_juju_name_is_mla_q_a(lower) &&
        !moe_juju_name_is_mla_q_b(lower) &&
        (moe_name_has(lower, "q_proj.weight") ||
         moe_name_has(lower, "attn_q.weight") ||
         moe_name_has(lower, "attention.wq.weight"));
}

static int moe_juju_name_is_std_k(const std::string& lower) {
    return !moe_juju_name_is_mla_kv_a(lower) &&
        !moe_juju_name_is_mla_kv_b(lower) &&
        (moe_name_has(lower, "k_proj.weight") ||
         moe_name_has(lower, "attn_k.weight") ||
         moe_name_has(lower, "attention.wk.weight"));
}

static int moe_juju_name_is_std_v(const std::string& lower) {
    return !moe_juju_name_is_mla_kv_a(lower) &&
        !moe_juju_name_is_mla_kv_b(lower) &&
        (moe_name_has(lower, "v_proj.weight") ||
         moe_name_has(lower, "self_attn.v_proj.weight") ||
         moe_name_has(lower, "value_proj.weight") ||
         moe_name_has(lower, "v_projection.weight") ||
         moe_name_has(lower, "attn_v.weight") ||
         moe_name_has(lower, "attention.wv.weight"));
}

static int moe_juju_name_is_std_q_norm(const std::string& lower) {
    return moe_name_has(lower, "attn_q_norm.weight") ||
        moe_name_has(lower, "q_norm.weight") ||
        moe_name_has(lower, "self_attn.q_norm.weight");
}

static int moe_juju_name_is_std_k_norm(const std::string& lower) {
    return moe_name_has(lower, "attn_k_norm.weight") ||
        moe_name_has(lower, "k_norm.weight") ||
        moe_name_has(lower, "self_attn.k_norm.weight");
}

static void moe_juju_probe_note_header(
    moe_juju_runtime_probe_t* probe,
    const moe_juju_header_t& header
) {
    if (!probe) {
        return;
    }
    if (header.layer_count) probe->shape.num_hidden_layers = header.layer_count;
    if (header.expert_count_per_layer) probe->shape.experts_per_moe_layer = header.expert_count_per_layer;
    if (header.hidden_dim) probe->shape.hidden_size = header.hidden_dim;
    if (header.expert_dim) probe->shape.expert_intermediate_size = header.expert_dim;
    if (header.expert_count_per_layer && !probe->shape.projection_count) {
        probe->shape.projection_count = 3;
    }
}

static void moe_juju_probe_note_index_json(
    moe_juju_runtime_probe_t* probe,
    const std::string& index_json
) {
    if (!probe || index_json.empty()) {
        return;
    }
    uint64_t value = 0;
    double double_value = 0.0;
    std::string string_value;
    if (moe_json_get_u64_local(index_json, "num_hidden_layers", &value) ||
        moe_json_get_u64_local(index_json, "n_layers", &value)) {
        probe->shape.num_hidden_layers = moe_u64_to_u32_clamped(value, probe->shape.num_hidden_layers);
    }
    if (moe_json_get_u64_local(index_json, "hidden_size", &value) ||
        moe_json_get_u64_local(index_json, "hidden_dim", &value)) {
        probe->shape.hidden_size = moe_u64_to_u32_clamped(value, probe->shape.hidden_size);
    }
    if (moe_json_get_u64_local(index_json, "vocab_size", &value)) {
        probe->shape.vocab_size = moe_u64_to_u32_clamped(value, probe->shape.vocab_size);
    }
    if (moe_json_get_u64_local(index_json, "experts_per_moe_layer", &value) ||
        moe_json_get_u64_local(index_json, "n_experts", &value)) {
        probe->shape.experts_per_moe_layer = moe_u64_to_u32_clamped(value, probe->shape.experts_per_moe_layer);
        probe->shape.projection_count = 3;
    }
    if (moe_json_get_u64_local(index_json, "expert_intermediate_size", &value) ||
        moe_json_get_u64_local(index_json, "expert_intermediate_dim", &value)) {
        probe->shape.expert_intermediate_size = moe_u64_to_u32_clamped(value, probe->shape.expert_intermediate_size);
    }
    if (moe_json_get_u64_local(index_json, "num_attention_heads", &value) ||
        moe_json_get_u64_local(index_json, "n_heads", &value) ||
        moe_json_get_u64_local(index_json, "head_count", &value)) {
        probe->attn.num_heads = moe_u64_to_u32_clamped(value, probe->attn.num_heads);
    }
    if (moe_json_get_u64_local(index_json, "num_key_value_heads", &value) ||
        moe_json_get_u64_local(index_json, "n_kv_heads", &value) ||
        moe_json_get_u64_local(index_json, "head_count_kv", &value)) {
        probe->attn.num_kv_heads = moe_u64_to_u32_clamped(value, probe->attn.num_kv_heads);
    }
    if (moe_json_get_u64_local(index_json, "head_dim", &value) ||
        moe_json_get_u64_local(index_json, "key_length", &value)) {
        probe->attn.head_dim = moe_u64_to_u32_clamped(value, probe->attn.head_dim);
    }
    if (moe_json_get_u64_local(index_json, "value_head_dim", &value) ||
        moe_json_get_u64_local(index_json, "value_length", &value)) {
        probe->attn.v_head_dim = moe_u64_to_u32_clamped(value, probe->attn.v_head_dim);
    }
    if (moe_json_get_u64_local(index_json, "kv_lora_rank", &value)) {
        probe->attn.kv_lora_rank = moe_u64_to_u32_clamped(value, probe->attn.kv_lora_rank);
        if (probe->attn.kv_lora_rank) {
            probe->attn.type = moe_ATTN_TYPE_MLA;
        }
    }
    if (moe_json_get_u64_local(index_json, "q_lora_rank", &value)) {
        probe->attn.q_lora_rank = moe_u64_to_u32_clamped(value, probe->attn.q_lora_rank);
        if (probe->attn.q_lora_rank) {
            probe->attn.type = moe_ATTN_TYPE_MLA;
        }
    }
    if (moe_json_get_u64_local(index_json, "qk_nope_head_dim", &value)) {
        probe->attn.qk_nope_head_dim = moe_u64_to_u32_clamped(value, probe->attn.qk_nope_head_dim);
        if (probe->attn.qk_nope_head_dim) {
            probe->attn.type = moe_ATTN_TYPE_MLA;
        }
    }
    if (moe_json_get_u64_local(index_json, "qk_rope_head_dim", &value)) {
        probe->attn.qk_rope_head_dim = moe_u64_to_u32_clamped(value, probe->attn.qk_rope_head_dim);
        if (probe->attn.qk_rope_head_dim) {
            probe->attn.type = moe_ATTN_TYPE_MLA;
        }
    }
    if (moe_json_get_u64_local(index_json, "context_length", &value) ||
        moe_json_get_u64_local(index_json, "max_position_embeddings", &value)) {
        probe->attn.max_seq_len = moe_u64_to_u32_clamped(value, probe->attn.max_seq_len);
    }
    if (moe_json_get_double_local(index_json, "rope_theta", &double_value) ||
        moe_json_get_double_local(index_json, "theta", &double_value)) {
        probe->attn.rope_theta = (float)double_value;
    }
    if (moe_json_get_double_local(index_json, "rms_norm_eps", &double_value) ||
        moe_json_get_double_local(index_json, "norm_eps", &double_value)) {
        probe->attn.rms_norm_eps = (float)double_value;
    }
    if (probe->architecture.empty() &&
        (moe_json_get_string_local(index_json, "declared_architecture", &string_value) ||
         moe_json_get_string_local(index_json, "architecture", &string_value) ||
         moe_json_get_string_local(index_json, "model_type", &string_value) ||
         moe_json_get_string_local(index_json, "family", &string_value))) {
        probe->architecture = string_value;
    }
    if (probe->model_id.empty() && moe_json_get_string_local(index_json, "model_id", &string_value)) {
        probe->model_id = string_value;
    }
}

static void moe_juju_probe_note_model_config_json(
    moe_pc_engine* engine,
    moe_juju_runtime_probe_t* probe,
    const std::string& config_json
) {
    if (!probe || config_json.empty()) {
        return;
    }
    size_t text_begin = 0;
    size_t text_end = config_json.size();
    if (!moe_json_find_object_local(config_json, "text_config", &text_begin, &text_end)) {
        text_begin = 0;
        text_end = config_json.size();
    }
    const std::string text_config = config_json.substr(text_begin, text_end - text_begin);
    uint64_t value = 0;
    double double_value = 0.0;
    std::string string_value;

    if (moe_json_get_u64_local(text_config, "num_hidden_layers", &value)) {
        // BUGFIX 854: Check value bounds before clamping ★★
        if (value > UINT32_MAX) {
            value = UINT32_MAX;
        }
        probe->shape.num_hidden_layers = moe_u64_to_u32_clamped(value, probe->shape.num_hidden_layers);
    }
    if (moe_json_get_u64_local(text_config, "hidden_size", &value)) {
        // BUGFIX 855: Check value bounds before clamping ★★
        if (value > UINT32_MAX) {
            value = UINT32_MAX;
        }
        probe->shape.hidden_size = moe_u64_to_u32_clamped(value, probe->shape.hidden_size);
    }
    if (moe_json_get_u64_local(text_config, "vocab_size", &value)) {
        // BUGFIX 856: Check value bounds before clamping ★★
        if (value > UINT32_MAX) {
            value = UINT32_MAX;
        }
        probe->shape.vocab_size = moe_u64_to_u32_clamped(value, probe->shape.vocab_size);
    }
    if (moe_json_get_u64_local(text_config, "num_experts", &value) ||
        moe_json_get_u64_local(text_config, "num_local_experts", &value) ||
        moe_json_get_u64_local(text_config, "experts_per_moe_layer", &value)) {
        probe->shape.experts_per_moe_layer = moe_u64_to_u32_clamped(value, probe->shape.experts_per_moe_layer);
        probe->shape.projection_count = 3;
    }
    if (moe_json_get_u64_local(text_config, "moe_intermediate_size", &value) ||
        moe_json_get_u64_local(text_config, "expert_intermediate_size", &value) ||
        moe_json_get_u64_local(text_config, "expert_intermediate_dim", &value)) {
        probe->shape.expert_intermediate_size = moe_u64_to_u32_clamped(value, probe->shape.expert_intermediate_size);
    }
    if (moe_json_get_u64_local(text_config, "num_attention_heads", &value)) {
        probe->attn.num_heads = moe_u64_to_u32_clamped(value, probe->attn.num_heads);
    }
    if (moe_json_get_u64_local(text_config, "num_key_value_heads", &value)) {
        probe->attn.num_kv_heads = moe_u64_to_u32_clamped(value, probe->attn.num_kv_heads);
    }
    if (moe_json_get_u64_local(text_config, "head_dim", &value)) {
        probe->attn.head_dim = moe_u64_to_u32_clamped(value, probe->attn.head_dim);
    }
