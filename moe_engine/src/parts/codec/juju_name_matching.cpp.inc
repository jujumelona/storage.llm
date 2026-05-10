// JUJU tensor name pattern matching
    if (moe_json_get_u64_local(text_config, "value_head_dim", &value) ||
        moe_json_get_u64_local(text_config, "v_head_dim", &value)) {
        probe->attn.v_head_dim = moe_u64_to_u32_clamped(value, probe->attn.v_head_dim);
    }
    if (moe_json_get_u64_local(text_config, "kv_lora_rank", &value)) {
        probe->attn.kv_lora_rank = moe_u64_to_u32_clamped(value, probe->attn.kv_lora_rank);
        if (probe->attn.kv_lora_rank) {
            probe->attn.type = moe_ATTN_TYPE_MLA;
        }
    }
    if (moe_json_get_u64_local(text_config, "q_lora_rank", &value)) {
        probe->attn.q_lora_rank = moe_u64_to_u32_clamped(value, probe->attn.q_lora_rank);
        if (probe->attn.q_lora_rank) {
            probe->attn.type = moe_ATTN_TYPE_MLA;
        }
    }
    if (moe_json_get_u64_local(text_config, "qk_nope_head_dim", &value)) {
        probe->attn.qk_nope_head_dim = moe_u64_to_u32_clamped(value, probe->attn.qk_nope_head_dim);
    }
    if (moe_json_get_u64_local(text_config, "qk_rope_head_dim", &value)) {
        probe->attn.qk_rope_head_dim = moe_u64_to_u32_clamped(value, probe->attn.qk_rope_head_dim);
    }
    if (moe_json_get_u64_local(text_config, "max_position_embeddings", &value) ||
        moe_json_get_u64_local(text_config, "context_length", &value)) {
        probe->attn.max_seq_len = moe_u64_to_u32_clamped(value, probe->attn.max_seq_len);
    }
    if (moe_json_get_double_local(text_config, "rope_theta", &double_value)) {
        probe->attn.rope_theta = (float)double_value;
    }
    if (moe_json_get_double_local(text_config, "rms_norm_eps", &double_value) ||
        moe_json_get_double_local(text_config, "norm_eps", &double_value)) {
        probe->attn.rms_norm_eps = (float)double_value;
    }
    if (probe->architecture.empty() &&
        (moe_json_get_string_local(text_config, "model_type", &string_value) ||
         moe_json_get_string_local(config_json, "model_type", &string_value))) {
        probe->architecture = string_value;
    }

    if (!engine) {
        return;
    }
    int attention_valid = 0;
    if (probe->attn.kv_lora_rank || probe->attn.q_lora_rank ||
        probe->attn.qk_nope_head_dim || probe->attn.qk_rope_head_dim) {
        attention_valid = 1;
    }
    if (moe_json_get_u64_local(text_config, "num_attention_heads", &value)) {
        engine->model_config_num_attention_heads = moe_u64_to_u32_clamped(value, 0);
        attention_valid = 1;
    }
    if (moe_json_get_u64_local(text_config, "num_key_value_heads", &value)) {
        engine->model_config_num_key_value_heads = moe_u64_to_u32_clamped(value, 0);
        attention_valid = 1;
    }
    if (moe_json_get_u64_local(text_config, "num_global_key_value_heads", &value)) {
        engine->model_config_num_global_key_value_heads = moe_u64_to_u32_clamped(value, 0);
        attention_valid = 1;
    }
    if (moe_json_get_u64_local(text_config, "head_dim", &value)) {
        engine->model_config_head_dim = moe_u64_to_u32_clamped(value, 0);
        attention_valid = 1;
    }
    if (moe_json_get_u64_local(text_config, "global_head_dim", &value)) {
        engine->model_config_global_head_dim = moe_u64_to_u32_clamped(value, 0);
        attention_valid = 1;
    }
    if (moe_json_get_u64_local(text_config, "sliding_window", &value)) {
        engine->model_config_sliding_window = moe_u64_to_u32_clamped(value, 0);
        attention_valid = 1;
    }
    if (moe_json_get_double_local(text_config, "query_pre_attn_scalar", &double_value) ||
        moe_json_get_double_local(config_json, "query_pre_attn_scalar", &double_value) ||
        moe_json_get_double_local(text_config, "attention_query_pre_attn_scalar", &double_value) ||
        moe_json_get_double_local(config_json, "attention_query_pre_attn_scalar", &double_value) ||
        moe_json_get_double_local(text_config, "attn_query_pre_attn_scalar", &double_value) ||
        moe_json_get_double_local(config_json, "attn_query_pre_attn_scalar", &double_value)) {
        if (double_value > 0.0 && double_value <= 65536.0) {
            engine->model_config_query_pre_attn_scalar = (float)double_value;
            attention_valid = 1;
        }
    }

    size_t rope_begin = 0;
    size_t rope_end = 0;
    if (moe_json_find_object_local(text_config, "rope_parameters", &rope_begin, &rope_end)) {
        const std::string rope = text_config.substr(rope_begin, rope_end - rope_begin);
        size_t full_begin = 0;
        size_t full_end = 0;
        if (moe_json_find_object_local(rope, "full_attention", &full_begin, &full_end)) {
            const std::string full = rope.substr(full_begin, full_end - full_begin);
            if (moe_json_get_double_local(full, "rope_theta", &double_value)) {
                engine->model_config_full_rope_theta = (float)double_value;
                attention_valid = 1;
            }
            if (moe_json_get_double_local(full, "partial_rotary_factor", &double_value)) {
                engine->model_config_partial_rotary_factor = (float)double_value;
                attention_valid = 1;
            }
        }
        size_t sliding_begin = 0;
        size_t sliding_end = 0;
        if (moe_json_find_object_local(rope, "sliding_attention", &sliding_begin, &sliding_end)) {
            const std::string sliding = rope.substr(sliding_begin, sliding_end - sliding_begin);
            if (moe_json_get_double_local(sliding, "rope_theta", &double_value)) {
                engine->model_config_sliding_rope_theta = (float)double_value;
                attention_valid = 1;
            }
        }
        if (engine->model_config_full_rope_theta <= 0.0f &&
            moe_json_get_double_local(rope, "rope_theta", &double_value)) {
            engine->model_config_full_rope_theta = (float)double_value;
            engine->model_config_sliding_rope_theta = (float)double_value;
            attention_valid = 1;
        }
    } else if (moe_json_get_double_local(text_config, "rope_theta", &double_value)) {
        engine->model_config_full_rope_theta = (float)double_value;
        engine->model_config_sliding_rope_theta = (float)double_value;
        attention_valid = 1;
    }

    std::vector<std::string> layer_types;
    if (moe_json_get_string_array_slice(config_json, text_begin, text_end, "layer_types", &layer_types)) {
        std::vector<uint8_t> full_layers;
        full_layers.reserve(layer_types.size());
        int saw_full = 0;
        int saw_local = 0;
        for (const std::string& item : layer_types) {
            const std::string lower = moe_lower_ascii_copy(item);
            const int is_full =
                lower.find("full") != std::string::npos ||
                lower.find("global") != std::string::npos;
            full_layers.push_back(is_full ? 1u : 0u);
            if (is_full) {
                saw_full = 1;
            } else {
                saw_local = 1;
            }
        }
        if (!full_layers.empty()) {
            engine->model_config_full_attention_layers = std::move(full_layers);
            engine->model_config_layer_types_valid.store(1, std::memory_order_relaxed);
            attention_valid = 1;
        }
        if (saw_full && saw_local) {
            engine->model_config_attention_valid.store(1, std::memory_order_relaxed);
        }
    }
    if (attention_valid) {
        engine->model_config_attention_valid.store(1, std::memory_order_relaxed);
        std::fprintf(stderr,
            "[storagellm] model config attention loaded: heads=%u kv=%u global_kv=%u head_dim=%u global_head_dim=%u kv_lora=%u q_lora=%u qk_nope=%u qk_rope=%u layers=%zu qpre=%.6g sliding_rope=%.6g full_rope=%.6g partial=%.6g\n",
            engine->model_config_num_attention_heads,
            engine->model_config_num_key_value_heads,
            engine->model_config_num_global_key_value_heads,
            engine->model_config_head_dim,
            engine->model_config_global_head_dim,
            probe->attn.kv_lora_rank,
            probe->attn.q_lora_rank,
            probe->attn.qk_nope_head_dim,
            probe->attn.qk_rope_head_dim,
            engine->model_config_full_attention_layers.size(),
            (double)engine->model_config_query_pre_attn_scalar,
            (double)engine->model_config_sliding_rope_theta,
            (double)engine->model_config_full_rope_theta,
            (double)engine->model_config_partial_rotary_factor);
    }
}

static void moe_juju_probe_note_tensor(
    moe_juju_runtime_probe_t* probe,
    const moe_juju_tensor_json_entry_t& entry
) {
    if (!probe || entry.name.empty() || !entry.shape[0]) {
        return;
    }
    const std::string lower = moe_lower_ascii_copy(entry.name);
    if (entry.dims == 1) {
        if (moe_juju_name_is_std_q_norm(lower)) {
            probe->q_norm_rows = std::max<uint64_t>(probe->q_norm_rows, entry.shape[0]);
        }
        if (moe_juju_name_is_std_k_norm(lower)) {
            probe->k_norm_rows = std::max<uint64_t>(probe->k_norm_rows, entry.shape[0]);
        }
        return;
    }
    if (entry.dims < 2 || !entry.shape[1]) {
        return;
    }
    const uint64_t cols = entry.math_in_features ? entry.math_in_features : entry.shape[0];
    const uint64_t rows = entry.math_out_features ? entry.math_out_features : entry.shape[1];
    if (lower == "token_embd.weight") {
        probe->shape.hidden_size = (uint32_t)std::min<uint64_t>(cols, UINT32_MAX);
        probe->shape.vocab_size = (uint32_t)std::min<uint64_t>(rows, UINT32_MAX);
    }
    uint32_t layer = 0;
    if (moe_parse_layer_from_gguf_name(lower, &layer)) {
        probe->shape.num_hidden_layers = std::max<uint32_t>(probe->shape.num_hidden_layers, layer + 1u);
        const bool expert_tensor =
            lower.find("ffn_gate_up_exps.weight") != std::string::npos ||
            lower.find("ffn_gate_exps.weight") != std::string::npos ||
            lower.find("ffn_up_exps.weight") != std::string::npos ||
            lower.find("ffn_down_exps.weight") != std::string::npos ||
            ((lower.find("gate_proj.weight") != std::string::npos ||
              lower.find("up_proj.weight") != std::string::npos ||
              lower.find("down_proj.weight") != std::string::npos) &&
             entry.dims >= 3);
        if (expert_tensor) {
            if (!probe->saw_moe_tensor) {
                probe->shape.first_moe_layer = layer;
                probe->shape.last_moe_layer = layer;
                probe->saw_moe_tensor = true;
            } else {
                probe->shape.first_moe_layer = std::min<uint32_t>(probe->shape.first_moe_layer, layer);
            }
            probe->shape.last_moe_layer = std::max<uint32_t>(probe->shape.last_moe_layer, layer);
            if (entry.dims >= 3 && entry.shape[2]) {
                probe->shape.experts_per_moe_layer =
                    std::max<uint32_t>(probe->shape.experts_per_moe_layer,
                        (uint32_t)std::min<uint64_t>(entry.shape[2], UINT32_MAX));
                probe->shape.projection_count = 3;
            }
            uint64_t expert_intermediate = 0;
            if (lower.find("ffn_gate_up_exps.weight") != std::string::npos) {
                expert_intermediate = rows >= 2 ? rows / 2ull : 0ull;
            } else if (lower.find("ffn_down_exps.weight") != std::string::npos ||
                       lower.find("down_proj.weight") != std::string::npos ||
                       lower.find("feed_forward.w2.weight") != std::string::npos ||
                       lower.find("ffn.w2.weight") != std::string::npos ||
                       lower.find("w2.weight") != std::string::npos) {
                expert_intermediate = cols;
            } else if (lower.find("ffn_gate_exps.weight") != std::string::npos ||
                       lower.find("ffn_up_exps.weight") != std::string::npos ||
                       lower.find("gate_proj.weight") != std::string::npos ||
                       lower.find("up_proj.weight") != std::string::npos ||
                       lower.find("feed_forward.w1.weight") != std::string::npos ||
                       lower.find("feed_forward.w3.weight") != std::string::npos ||
                       lower.find("ffn.w1.weight") != std::string::npos ||
                       lower.find("ffn.w3.weight") != std::string::npos ||
                       lower.find("w1.weight") != std::string::npos ||
                       lower.find("w3.weight") != std::string::npos) {
                expert_intermediate = rows;
            }
            if (expert_intermediate) {
                probe->shape.expert_intermediate_size =
                    (uint32_t)std::min<uint64_t>(expert_intermediate, UINT32_MAX);
            }
        }
    }
    if (moe_juju_name_is_mla_q_a(lower)) {
        probe->q_a_rows = std::max<uint64_t>(probe->q_a_rows, rows);
    } else if (moe_juju_name_is_mla_q_b(lower)) {
        probe->q_b_rows = std::max<uint64_t>(probe->q_b_rows, rows);
        probe->q_b_cols = std::max<uint64_t>(probe->q_b_cols, cols);
    } else if (moe_juju_name_is_mla_kv_a(lower)) {
        probe->kv_a_rows = std::max<uint64_t>(probe->kv_a_rows, rows);
    } else if (moe_juju_name_is_mla_kv_b(lower)) {
        probe->kv_b_rows = std::max<uint64_t>(probe->kv_b_rows, rows);
        probe->kv_b_cols = std::max<uint64_t>(probe->kv_b_cols, cols);
    } else if (moe_juju_name_is_std_q(lower)) {
        probe->q_proj_rows = std::max<uint64_t>(probe->q_proj_rows, rows);
    } else if (moe_juju_name_is_std_k(lower)) {
        probe->k_proj_rows = std::max<uint64_t>(probe->k_proj_rows, rows);
    } else if (moe_juju_name_is_std_v(lower)) {
        probe->v_proj_rows = std::max<uint64_t>(probe->v_proj_rows, rows);
    }
    if (moe_juju_name_is_std_q_norm(lower)) {
        probe->q_norm_rows = std::max<uint64_t>(probe->q_norm_rows, cols);
    }
    if (moe_juju_name_is_std_k_norm(lower)) {
        probe->k_norm_rows = std::max<uint64_t>(probe->k_norm_rows, cols);
    }
    if (moe_juju_name_is_attn_o(lower)) {
        probe->o_proj_cols = std::max<uint64_t>(probe->o_proj_cols, cols);
    }
}

static void moe_juju_probe_finalize_attention(moe_juju_runtime_probe_t* probe) {
    if (!probe) {
        return;
    }
    moe_attn_config_t& cfg = probe->attn;
    if (probe->kv_b_cols && probe->kv_a_rows > probe->kv_b_cols) {
        cfg.type = moe_ATTN_TYPE_MLA;
        cfg.kv_lora_rank = cfg.kv_lora_rank ? cfg.kv_lora_rank :
            (uint32_t)std::min<uint64_t>(probe->kv_b_cols, UINT32_MAX);
        cfg.qk_rope_head_dim = cfg.qk_rope_head_dim ? cfg.qk_rope_head_dim :
            (uint32_t)std::min<uint64_t>(probe->kv_a_rows - probe->kv_b_cols, UINT32_MAX);
        if (!cfg.q_lora_rank) {
            cfg.q_lora_rank = (uint32_t)std::min<uint64_t>(
                probe->q_b_cols ? probe->q_b_cols : probe->q_a_rows, UINT32_MAX);
        }
        if (!cfg.num_heads && cfg.head_dim && probe->q_b_rows >= cfg.head_dim) {
            cfg.num_heads = (uint32_t)std::min<uint64_t>(probe->q_b_rows / cfg.head_dim, UINT32_MAX);
        }
        if (!cfg.head_dim && cfg.num_heads && probe->q_b_rows >= cfg.num_heads) {
            cfg.head_dim = (uint32_t)std::min<uint64_t>(probe->q_b_rows / cfg.num_heads, UINT32_MAX);
        }
        if (!cfg.v_head_dim && cfg.num_heads && probe->o_proj_cols >= cfg.num_heads) {
            cfg.v_head_dim = (uint32_t)std::min<uint64_t>(probe->o_proj_cols / cfg.num_heads, UINT32_MAX);
        }
        if (!cfg.qk_nope_head_dim && cfg.num_heads && cfg.v_head_dim &&
            probe->kv_b_rows >= (uint64_t)cfg.num_heads * (uint64_t)cfg.v_head_dim) {
            cfg.qk_nope_head_dim = (uint32_t)std::min<uint64_t>(
                (probe->kv_b_rows / cfg.num_heads) - cfg.v_head_dim, UINT32_MAX);
        }
        if (!cfg.head_dim && cfg.qk_nope_head_dim && cfg.qk_rope_head_dim) {
            cfg.head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim;
        }
        if (!cfg.num_kv_heads) {
            cfg.num_kv_heads = cfg.num_heads;
        }
    } else {
        const uint64_t norm_head_dim =
            probe->q_norm_rows && probe->q_norm_rows == probe->k_norm_rows ?
            probe->q_norm_rows : std::max<uint64_t>(probe->q_norm_rows, probe->k_norm_rows);
        if (norm_head_dim &&
            probe->q_proj_rows >= norm_head_dim &&
            probe->k_proj_rows >= norm_head_dim &&
            probe->q_proj_rows % norm_head_dim == 0 &&
            probe->k_proj_rows % norm_head_dim == 0) {
            cfg.head_dim = (uint32_t)std::min<uint64_t>(norm_head_dim, UINT32_MAX);
            cfg.num_heads = (uint32_t)std::min<uint64_t>(probe->q_proj_rows / norm_head_dim, UINT32_MAX);
            cfg.num_kv_heads = (uint32_t)std::min<uint64_t>(probe->k_proj_rows / norm_head_dim, UINT32_MAX);
            if (probe->v_proj_rows >= cfg.num_kv_heads && cfg.num_kv_heads &&
                probe->v_proj_rows % cfg.num_kv_heads == 0) {
                cfg.v_head_dim = (uint32_t)std::min<uint64_t>(
                    probe->v_proj_rows / cfg.num_kv_heads, UINT32_MAX);
            }
        }
        if (!cfg.head_dim && cfg.num_heads && probe->q_proj_rows >= cfg.num_heads) {
            cfg.head_dim = (uint32_t)std::min<uint64_t>(probe->q_proj_rows / cfg.num_heads, UINT32_MAX);
        }
        if (!cfg.num_heads && cfg.head_dim && probe->q_proj_rows >= cfg.head_dim) {
            cfg.num_heads = (uint32_t)std::min<uint64_t>(probe->q_proj_rows / cfg.head_dim, UINT32_MAX);
        }
        if (!cfg.num_kv_heads && cfg.head_dim && probe->k_proj_rows >= cfg.head_dim) {
            cfg.num_kv_heads = (uint32_t)std::min<uint64_t>(probe->k_proj_rows / cfg.head_dim, UINT32_MAX);
        }
        if (!cfg.v_head_dim && cfg.num_kv_heads && probe->v_proj_rows >= cfg.num_kv_heads) {
            cfg.v_head_dim = (uint32_t)std::min<uint64_t>(probe->v_proj_rows / cfg.num_kv_heads, UINT32_MAX);
        }
        if (cfg.num_heads) {
            if (!cfg.num_kv_heads) {
                cfg.num_kv_heads = cfg.num_heads;
            }
            cfg.type = cfg.num_kv_heads == 1u ? moe_ATTN_TYPE_MQA :
                (cfg.num_kv_heads < cfg.num_heads ? moe_ATTN_TYPE_GQA : moe_ATTN_TYPE_MHA);
        }
        if (cfg.head_dim && probe->q_proj_rows >= cfg.head_dim &&
            probe->q_proj_rows % cfg.head_dim == 0 &&
            cfg.num_heads * cfg.head_dim != probe->q_proj_rows) {
            cfg.num_heads = (uint32_t)std::min<uint64_t>(probe->q_proj_rows / cfg.head_dim, UINT32_MAX);
        }
        if (cfg.head_dim && probe->k_proj_rows >= cfg.head_dim &&
            probe->k_proj_rows % cfg.head_dim == 0) {
            cfg.num_kv_heads = (uint32_t)std::min<uint64_t>(probe->k_proj_rows / cfg.head_dim, UINT32_MAX);
        }
        if (cfg.num_kv_heads && probe->v_proj_rows >= cfg.num_kv_heads &&
            probe->v_proj_rows % cfg.num_kv_heads == 0) {
            cfg.v_head_dim = (uint32_t)std::min<uint64_t>(
                probe->v_proj_rows / cfg.num_kv_heads, UINT32_MAX);
        }
        if (cfg.num_heads) {
            cfg.type = cfg.num_kv_heads == 1u ? moe_ATTN_TYPE_MQA :
                (cfg.num_kv_heads < cfg.num_heads ? moe_ATTN_TYPE_GQA : moe_ATTN_TYPE_MHA);
        }
    }
    if (!cfg.max_seq_len) cfg.max_seq_len = 8192u;
    if (!cfg.rope_table_len || cfg.rope_table_len < cfg.max_seq_len) {
        cfg.rope_table_len = std::max<uint32_t>(16384u, cfg.max_seq_len);
    }
    if (cfg.rope_theta <= 0.0f) cfg.rope_theta = 500000.0f;
    if (cfg.rms_norm_eps <= 0.0f) cfg.rms_norm_eps = 1.0e-6f;
    if (cfg.type == moe_ATTN_TYPE_UNKNOWN &&
        (cfg.kv_lora_rank || cfg.q_lora_rank || cfg.qk_nope_head_dim || cfg.qk_rope_head_dim)) {
        cfg.type = moe_ATTN_TYPE_MLA;
    }
}

static void moe_juju_probe_apply(moe_pc_engine* engine, moe_juju_runtime_probe_t* probe) {
    if (!engine || !probe) {
        return;
    }
    moe_juju_probe_finalize_attention(probe);
    if (probe->shape.experts_per_moe_layer && probe->shape.projection_count == 0) {
        probe->shape.projection_count = 3;
    }
    if (probe->shape.experts_per_moe_layer && probe->shape.num_hidden_layers &&
        probe->shape.first_moe_layer == 0 && probe->shape.last_moe_layer == 0) {
        probe->shape.last_moe_layer = probe->shape.num_hidden_layers - 1u;
    }
    engine->dynamic_shape = probe->shape;
    if (!probe->architecture.empty()) {
        engine->dynamic_architecture = probe->architecture;
    }
    if (!probe->model_id.empty()) {
        engine->dynamic_model_id = probe->model_id;
    }
    if (probe->shape.hidden_size && probe->shape.num_hidden_layers &&
        (probe->shape.vocab_size || probe->shape.experts_per_moe_layer)) {
        engine->dynamic_shape_valid.store(1, std::memory_order_relaxed);
    }
    if (probe->attn.type != moe_ATTN_TYPE_UNKNOWN &&
        probe->attn.num_heads && probe->attn.head_dim) {
        engine->attn_config = probe->attn;
        engine->attn_config_valid.store(1, std::memory_order_relaxed);
    }
}

static void moe_register_gguf_common_tensor(
    moe_pc_engine* engine,
    const moe_gguf_tensor_dir_entry_t& entry,
    uint16_t path_idx
) {
    if (!engine || entry.name.empty() || entry.bytes == 0) {
        return;
    }
    const std::string key = moe_lower_ascii_copy(entry.name);
    if (engine->gguf_common_tensor_slots.find(key) != engine->gguf_common_tensor_slots.end()) {
        return;
    }
    moe_gguf_common_tensor_record rec{};
    rec.name = key;
    rec.resolved_path_index = path_idx;
    rec.dims = std::min<uint32_t>(entry.dims, 4u);
    for (uint32_t i = 0; i < rec.dims; ++i) {
        rec.shape[i] = entry.shape[i];
    }
    rec.type = entry.type;
    rec.weight_encoding = entry.weight_encoding != moe_WEIGHT_ENCODING_UNKNOWN ?
        entry.weight_encoding : moe_weight_encoding_from_gguf_type(entry.type, moe_offload_gguf_contract_t{});
    rec.file_offset = entry.file_offset;
    rec.bytes = entry.bytes;
    rec.weight_row_bytes = entry.weight_row_bytes;
    engine->gguf_common_tensor_slots.emplace(rec.name, engine->gguf_common_tensors.size());
    engine->gguf_common_tensors.push_back(std::move(rec));
    moe_dynamic_shape_note_tensor(engine, key, entry.dims, entry.shape);
}
static int load_gguf_tensor_directory_as_tensor_index(
    moe_pc_engine* engine,
    const std::string& root,
    const moe_offload_gguf_contract_t& contract
) {
    if (!engine || root.empty()) {
        return 0;
    }
    std::vector<std::string> paths;
    moe_collect_gguf_files(root, 4, &paths);
    if (paths.empty()) {
        return 0;
    }
    std::sort(paths.begin(), paths.end());
    moe_juju_runtime_probe_t probe{};
    std::string model_config_json;
    const int has_model_config_json = moe_read_model_config_json(root, &model_config_json);
    if (has_model_config_json) {
        moe_juju_probe_note_model_config_json(engine, &probe, model_config_json);
    }
    for (const std::string& path : paths) {
        moe_juju_header_t header{};
        std::vector<moe_juju_section_entry_t> sections;
        uint64_t file_bytes = 0;
        if (!moe_juju_read_table_local(path, &header, &sections, &file_bytes)) {
            continue;
        }
        moe_ingest_juju_runtime_metadata_sections(engine, path, sections, file_bytes);
        moe_juju_probe_note_header(&probe, header);
        std::string index_json;
        if (!moe_read_juju_tensor_index_json(path, sections, file_bytes, &index_json)) {
            continue;
        }
        moe_juju_probe_note_index_json(&probe, index_json);
        std::vector<moe_juju_tensor_json_entry_t> tensors;
        if (moe_parse_juju_tensor_index_json(index_json, &tensors, nullptr, nullptr, nullptr)) {
            for (const moe_juju_tensor_json_entry_t& entry : tensors) {
                moe_juju_probe_note_tensor(&probe, entry);
            }
        }
    }
    moe_juju_probe_apply(engine, &probe);
    engine->expert_states.assign(expert_slot_count(engine), moe_block_state{});
    tensor_slots_reset(engine);
    engine->tensors.reserve(tensor_record_capacity(engine));
    std::unordered_map<std::string, uint16_t> path_index_map;
    uint64_t records = 0;
    for (const std::string& path : paths) {
        moe_gguf_tensor_dir_t dir{};
        if (!moe_read_gguf_tensor_directory(path, &dir)) {
            continue;
        }
        uint16_t path_idx = 0xFFFFu;
        auto pit = path_index_map.find(path);
        if (pit != path_index_map.end()) {
            path_idx = pit->second;
        } else {
            if (engine->tensor_paths.size() >= 0xFFFFu) {
                continue;
            }
            path_idx = (uint16_t)engine->tensor_paths.size();
            engine->tensor_paths.push_back(path);
