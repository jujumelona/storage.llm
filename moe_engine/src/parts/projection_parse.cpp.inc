static moe_projection_t parse_projection(const std::string& value) {
    std::string lower = value;
    for (char& ch : lower) ch = (char)std::tolower((unsigned char)ch);
    if (lower == "gate_proj" || lower == "gate" || lower == "w1" ||
        lower.find("gate_proj") != std::string::npos ||
        lower.find("ffn_gate") != std::string::npos) {
        return moe_PROJ_GATE;
    }
    if (lower == "up_proj" || lower == "up" || lower == "w3" ||
        lower.find("up_proj") != std::string::npos ||
        lower.find("ffn_up") != std::string::npos) {
        return moe_PROJ_UP;
    }
    return moe_PROJ_DOWN;
}


