static moe_projection_t parse_projection(const std::string& value) {
    if (value == "gate_proj" || value == "gate") {
        return moe_PROJ_GATE;
    }
    if (value == "up_proj" || value == "up") {
        return moe_PROJ_UP;
    }
    return moe_PROJ_DOWN;
}


