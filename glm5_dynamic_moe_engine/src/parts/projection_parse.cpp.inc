static glm5_projection_t parse_projection(const std::string& value) {
    if (value == "gate_proj" || value == "gate") {
        return GLM5_PROJ_GATE;
    }
    if (value == "up_proj" || value == "up") {
        return GLM5_PROJ_UP;
    }
    return GLM5_PROJ_DOWN;
}


