// JUJU format parsing and dynamic shape inference
static uint8_t moe_weight_encoding_from_juju_entry(
    const moe_juju_tensor_json_entry_t& entry,
    const moe_offload_gguf_contract_t& contract
) {
    if (entry.has_gguf_type) {
        const uint8_t by_type = moe_weight_encoding_from_gguf_type(entry.gguf_type, contract);
        if (by_type != moe_WEIGHT_ENCODING_UNKNOWN) {
            return by_type;
        }
    }
    if (entry.weight_encoding != moe_WEIGHT_ENCODING_UNKNOWN) {
        return entry.weight_encoding;
    }
    if (contract.weight_encoding != moe_WEIGHT_ENCODING_UNKNOWN) {
        return (uint8_t)std::min<uint32_t>(contract.weight_encoding, 255u);
    }
    return moe_WEIGHT_ENCODING_UNKNOWN;
}

static uint64_t moe_effective_juju_tensor_bytes(
    const moe_juju_tensor_json_entry_t& entry,
    const moe_offload_gguf_contract_t& contract
) {
    // BUGFIX 803: Check entry.juju_bytes for zero.
    if (entry.juju_bytes == 0 || entry.dims == 0 || entry.shape[0] == 0) {
        return entry.juju_bytes;
    }
    // BUGFIX 804: Check dims for reasonable upper bound.
    if (entry.dims > 8) {
        return entry.juju_bytes;
    }
    const uint32_t cols = (uint32_t)std::min<uint64_t>(entry.shape[0], UINT32_MAX);
    uint64_t rows64 = 1;
    for (uint32_t i = 1; i < std::min<uint32_t>(entry.dims, 4u); ++i) {
        if (entry.shape[i] == 0 || rows64 > UINT64_MAX / entry.shape[i]) {
            return entry.juju_bytes;
        }
        rows64 *= entry.shape[i];
    }
    if (rows64 == 0 || rows64 > UINT32_MAX || cols == 0) {
        return entry.juju_bytes;
    }
    if (entry.row_stride_bytes > 0) {
        if (rows64 <= UINT64_MAX / entry.row_stride_bytes) {
            const uint64_t physical = rows64 * entry.row_stride_bytes;
            if (physical > 0 && physical <= entry.juju_bytes) {
                return physical;
            }
        }
        return entry.juju_bytes;
    }
    moe_tensor_info_t info{};
    info.rows = (uint32_t)rows64;
    info.cols = cols;
    info.weight_byte_length = entry.juju_bytes;
    const uint8_t encoding = moe_weight_encoding_from_juju_entry(entry, contract);
    const uint64_t row_bytes = infer_weight_row_bytes(info, encoding);
    if (!row_bytes || rows64 > UINT64_MAX / row_bytes) {
        return entry.juju_bytes;
    }
    const uint64_t exact = rows64 * row_bytes;
    return exact > 0 && exact <= entry.juju_bytes ? exact : entry.juju_bytes;
}

static void moe_dynamic_shape_note_tensor(
    moe_pc_engine* engine,
    const std::string& key,
    uint32_t dims,
    const uint64_t* shape
) {
    if (!engine || key.empty() || !shape) {
        return;
    }
    // BUGFIX 802: Check dims for reasonable upper bound ★★
    if (dims > 8) {
        return;
    }
    moe_model_shape_t next = engine->dynamic_shape;
    if (key == "token_embd.weight" && dims >= 2 && shape[0] && shape[1]) {
        next.hidden_size = (uint32_t)std::min<uint64_t>(shape[0], UINT32_MAX);
        next.vocab_size = (uint32_t)std::min<uint64_t>(shape[1], UINT32_MAX);
    }
    uint32_t layer = 0;
    if (moe_parse_layer_from_gguf_name(key, &layer)) {
        if (layer + 1u > next.num_hidden_layers) {
            next.num_hidden_layers = layer + 1u;
        }
        if (key.find("ffn_gate_up_exps.weight") != std::string::npos ||
            key.find("ffn_gate_exps.weight") != std::string::npos ||
            key.find("ffn_up_exps.weight") != std::string::npos ||
            key.find("ffn_down_exps.weight") != std::string::npos) {
            const bool had_moe_tensor =
                next.first_moe_layer != 0 ||
                next.last_moe_layer != 0 ||
                next.expert_intermediate_size != 0;
            if (!had_moe_tensor) {
                next.first_moe_layer = layer;
                next.last_moe_layer = layer;
            } else if (layer < next.first_moe_layer) {
                next.first_moe_layer = layer;
            }
            if (layer > next.last_moe_layer) {
                next.last_moe_layer = layer;
            }
            if (dims >= 3 && shape[2]) {
                next.experts_per_moe_layer =
                    std::max<uint32_t>(next.experts_per_moe_layer,
                        (uint32_t)std::min<uint64_t>(shape[2], UINT32_MAX));
            }
            uint64_t expert_intermediate = 0;
            if (key.find("ffn_gate_up_exps.weight") != std::string::npos) {
                expert_intermediate = (dims >= 2 && shape[1] >= 2) ? shape[1] / 2ull : 0ull;
            } else if (key.find("ffn_down_exps.weight") != std::string::npos) {
                expert_intermediate = (dims >= 1) ? shape[0] : 0ull;
            } else if (key.find("ffn_gate_exps.weight") != std::string::npos ||
                       key.find("ffn_up_exps.weight") != std::string::npos) {
                expert_intermediate = (dims >= 2) ? shape[1] : 0ull;
            }
            if (expert_intermediate) {
                next.expert_intermediate_size =
                    (uint32_t)std::min<uint64_t>(expert_intermediate, UINT32_MAX);
            }
        }
    }
    if (next.hidden_size && next.vocab_size && next.num_hidden_layers) {
        next.projection_count = 3;
        engine->dynamic_shape = next;
        engine->dynamic_shape_valid.store(1, std::memory_order_relaxed);
    } else {
        engine->dynamic_shape = next;
    }
}

static int moe_graph_ir_executor_compiled(const moe_pc_engine* engine) {
    if (!engine ||
        !engine->offload_graph_ir_ready.load(std::memory_order_relaxed) ||
        !engine->dynamic_shape_valid.load(std::memory_order_relaxed) ||
        !engine->attn_config_valid.load(std::memory_order_relaxed) ||
        engine->gguf_common_tensor_slots.empty() ||
        engine->offload_graph_ir_json.empty()) {
        return 0;
    }
    const moe_model_shape_t& shape = engine->dynamic_shape;
    if (!shape.hidden_size || !shape.vocab_size || !shape.num_hidden_layers) {
        return 0;
    }
    const moe_attn_config_t& cfg = engine->attn_config;
    if (!cfg.num_heads || !cfg.num_kv_heads || !cfg.head_dim ||
        cfg.num_kv_heads > cfg.num_heads) {
        return 0;
    }
    if (cfg.type != moe_ATTN_TYPE_MLA &&
        cfg.type != moe_ATTN_TYPE_MHA &&
        cfg.type != moe_ATTN_TYPE_GQA &&
        cfg.type != moe_ATTN_TYPE_MQA) {
        return 0;
    }
    if (engine->offload_graph_ir_op_count.load(std::memory_order_relaxed) == 0 ||
        engine->offload_graph_ir_layer_count.load(std::memory_order_relaxed) == 0) {
        return 0;
    }
    const std::string& graph = engine->offload_graph_ir_json;
    static const char* kRequiredGraphKeys[] = {
        "JUJU_GRAPH_IR_V1",
        "\"generation_contract\"",
        "\"tensor_bindings\"",
        "\"token_embedding\"",
        "\"lm_head\"",
        "\"final_norm\"",
        "\"ops\"",
        "\"layers\""
    };
    for (const char* key : kRequiredGraphKeys) {
        if (graph.find(key) == std::string::npos) {
            return 0;
        }
    }
    const int graph_required = engine->offload_graph_ir_required.load(std::memory_order_relaxed);
    if ((graph_required || graph.find("\"runtime_execution_manifest\"") != std::string::npos) &&
        !engine->juju_runtime_execution_manifest_ready.load(std::memory_order_relaxed)) {
        return 0;
    }
    if ((graph_required || graph.find("\"runtime_access_plan\"") != std::string::npos) &&
        !engine->juju_runtime_access_plan_ready.load(std::memory_order_relaxed)) {
        return 0;
    }
    if ((graph_required || graph.find("\"kv_layout_contract\"") != std::string::npos) &&
        !engine->juju_kv_layout_contract_ready.load(std::memory_order_relaxed)) {
        return 0;
    }
    return 1;
}
