// GGUF format parsing utilities
static int moe_parse_layer_from_gguf_name(const std::string& name, uint32_t* layer) {
    if (!layer || name.empty()) {
        return 0;
    }
    const char* needles[] = {
        "blk.",
        "blocks.",
        "block.",
        "model.layers.",
        "layers.",
        "decoder.layers.",
        "transformer.h.",
        "transformer.blocks.",
        "h."
    };
    size_t pos = std::string::npos;
    size_t needle_len = 0;
    for (const char* needle : needles) {
        const size_t next = name.find(needle);
        if (next != std::string::npos &&
            (pos == std::string::npos || next < pos)) {
            pos = next;
            needle_len = std::strlen(needle);
        }
    }
    if (pos == std::string::npos) {
        return 0;
    }
    size_t p = pos + needle_len;
    if (p >= name.size()) {
        return 0;
    }
    uint64_t value = 0;
    bool any = false;
    while (p < name.size() && name[p] >= '0' && name[p] <= '9') {
        if (value > UINT64_MAX / 10ull) {
            return 0;
        }
        value = value * 10ull + (uint64_t)(name[p] - '0');
        any = true;
        ++p;
    }
    if (!any || value > UINT32_MAX) {
        return 0;
    }
    *layer = (uint32_t)value;
    return 1;
}

static int moe_gguf_expert_projection(const std::string& name, moe_projection_t* proj) {
    if (!proj || name.empty()) {
        return 0;
    }
    const std::string value = moe_lower_ascii_copy(name);
    if (value.find("ffn_gate_exps.weight") != std::string::npos ||
        value.find("gate_proj.weight") != std::string::npos ||
        value.find("experts.w1.weight") != std::string::npos ||
        value.find("experts.w1") != std::string::npos ||
        value.find("feed_forward.w1.weight") != std::string::npos ||
        value.find("ffn.w1.weight") != std::string::npos ||
        value.find("mlp.w1.weight") != std::string::npos) {
        *proj = moe_PROJ_GATE;
        return 1;
    }
    if (value.find("ffn_up_exps.weight") != std::string::npos ||
        value.find("up_proj.weight") != std::string::npos ||
        value.find("experts.w3.weight") != std::string::npos ||
        value.find("experts.w3") != std::string::npos ||
        value.find("feed_forward.w3.weight") != std::string::npos ||
        value.find("ffn.w3.weight") != std::string::npos ||
        value.find("mlp.w3.weight") != std::string::npos) {
        *proj = moe_PROJ_UP;
        return 1;
    }
    if (value.find("ffn_down_exps.weight") != std::string::npos ||
        value.find("down_proj.weight") != std::string::npos ||
        value.find("experts.w2.weight") != std::string::npos ||
        value.find("experts.w2") != std::string::npos ||
        value.find("feed_forward.w2.weight") != std::string::npos ||
        value.find("ffn.w2.weight") != std::string::npos ||
        value.find("mlp.w2.weight") != std::string::npos) {
        *proj = moe_PROJ_DOWN;
        return 1;
    }
    return 0;
}

static uint8_t moe_weight_encoding_from_gguf_type(uint32_t type, const moe_offload_gguf_contract_t& contract) {
    (void)contract;
    if (type == 0u) {
        return moe_WEIGHT_ENCODING_RAW_FP32;
    }
    if (type == 1u) {
        return moe_WEIGHT_ENCODING_RAW_FP16;
    }
    if (type == 2u) {
        return moe_WEIGHT_ENCODING_GGUF_Q4_0;
    }
    if (type == 3u) {
        return moe_WEIGHT_ENCODING_GGUF_Q4_1;
    }
    if (type == 6u) {
        return moe_WEIGHT_ENCODING_GGUF_Q5_0;
    }
    if (type == 7u) {
        return moe_WEIGHT_ENCODING_GGUF_Q5_1;
    }
    if (type == 8u) {
        return moe_WEIGHT_ENCODING_GGUF_Q8_0;
    }
    if (type == 9u) {
        return moe_WEIGHT_ENCODING_GGUF_Q8_1;
    }
    if (type == 10u) {
        return moe_WEIGHT_ENCODING_GGUF_Q2_K;
    }
    if (type == 11u) {
        return moe_WEIGHT_ENCODING_GGUF_Q3_K;
    }
    if (type == 12u) {
        return moe_WEIGHT_ENCODING_GGUF_Q4_K;
    }
    if (type == 13u) {
        return moe_WEIGHT_ENCODING_GGUF_Q5_K;
    }
    if (type == 14u) {
        return moe_WEIGHT_ENCODING_GGUF_Q6_K;
    }
    if (type == 15u) {
        return moe_WEIGHT_ENCODING_GGUF_Q8_K;
    }
    if (type == 39u) {
        return moe_WEIGHT_ENCODING_MXFP4;
    }
    if (type == 16u) {
        return moe_WEIGHT_ENCODING_GGUF_IQ2_XXS;
    }
    if (type == 18u) {
        return moe_WEIGHT_ENCODING_GGUF_IQ3_XXS;
    }
    if (type == 19u) {
        return moe_WEIGHT_ENCODING_GGUF_IQ1_S;
    }
    if (type == 17u) {
        return moe_WEIGHT_ENCODING_GGUF_IQ2_XS;
    }
    if (type == 22u) {
        return moe_WEIGHT_ENCODING_GGUF_IQ2_S;
    }
    if (type == 29u) {
        return moe_WEIGHT_ENCODING_GGUF_IQ1_M;
    }
    if (type == 21u) {
        return moe_WEIGHT_ENCODING_GGUF_IQ3_S;
    }
    if (type == 20u) {
        return moe_WEIGHT_ENCODING_GGUF_IQ4_NL;
    }
    if (type == 23u) {
        return moe_WEIGHT_ENCODING_GGUF_IQ4_XS;
    }
    if (type == 30u) {
        return moe_WEIGHT_ENCODING_RAW_BF16;
    }
    if (type == 31u || type == 32u || type == 33u) {
        return moe_WEIGHT_ENCODING_UNKNOWN;
    }
    if (type == 34u) {
        return moe_WEIGHT_ENCODING_GGUF_TQ1_0;
    }
    if (type == 35u) {
        return moe_WEIGHT_ENCODING_GGUF_TQ2_0;
    }
    return moe_WEIGHT_ENCODING_UNKNOWN;
}
