static const char* moe_tvm_backend_name(const moe_pc_engine* e) {
    if (!e) {
        return "cpu";
    }
    switch (e->backend_caps.backend) {
        case moe_BACKEND_CUDA:
            return "cuda";
        case moe_BACKEND_METAL:
            return "metal";
        case moe_BACKEND_VULKAN:
            return "vulkan";
        case moe_BACKEND_OPENCL:
            return "opencl";
        case moe_BACKEND_CPU:
        default:
            return "cpu";
    }
}

static const char* moe_tvm_target_string(const moe_pc_engine* e) {
    if (!e) {
        return "llvm";
    }
    switch (e->backend_caps.backend) {
        case moe_BACKEND_CUDA:
            return "cuda";
        case moe_BACKEND_METAL:
            return "metal";
        case moe_BACKEND_VULKAN:
            return "vulkan";
        case moe_BACKEND_OPENCL:
            return "opencl";
        case moe_BACKEND_CPU:
        default:
            return "llvm";
    }
}

static const char* moe_tvm_profile_dtype_from_storage(const moe_storage_constants_t* c) {
    (void)c;
    return "fp32";
}

static int moe_write_tvm_auto_profile(const moe_pc_engine* e, const char* out_path) {
    if (!e || !out_path || !out_path[0]) {
        return 0;
    }

    FILE* f = std::fopen(out_path, "wb");
    if (!f) {
        return 0;
    }

    const moe_storage_constants_t storage = moe_engine_storage_constants_or_static(e);
    const uint32_t hidden = storage.hidden_size;
    const uint32_t intermediate = storage.expert_intermediate_size ?
        storage.expert_intermediate_size : storage.hidden_size;
    const uint32_t experts = storage.experts_per_moe_layer ?
        storage.experts_per_moe_layer : 8u;
    const uint32_t max_experts = std::max<uint32_t>(1u, std::min<uint32_t>(8u, experts));

    std::fprintf(f,
        "{\n"
        "  \"source\": \"engine_auto\",\n"
        "  \"backend\": \"%s\",\n"
        "  \"target\": \"%s\",\n"
        "  \"hidden\": %u,\n"
        "  \"intermediate\": %u,\n"
        "  \"dtype\": \"%s\",\n"
        "  \"max_experts\": %u,\n"
        "  \"max_assignments\": %u\n"
        "}\n",
        moe_tvm_backend_name(e),
        moe_tvm_target_string(e),
        hidden,
        intermediate ? intermediate : hidden,
        moe_tvm_profile_dtype_from_storage(&storage),
        max_experts,
        256u);

    std::fclose(f);
    return 1;
}
