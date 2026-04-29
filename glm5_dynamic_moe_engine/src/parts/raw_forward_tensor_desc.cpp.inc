// Raw forward - Tensor descriptor definitions

struct glm5_raw_tensor_desc {
    uint32_t role;
    uint64_t span_bytes;
    uint32_t occurrence;
    uint64_t rel_offset;
    uint64_t bytes;
    uint32_t rows;
    uint32_t cols;
};

static int raw_tensor_desc_for_kind(glm5_raw_tensor_kind_t kind, glm5_raw_tensor_desc* out) {
    if (!out) {
        return 0;
    }
    switch (kind) {
        case GLM5_RAW_TENSOR_INPUT_LAYER_NORM:
            *out = {GLM5_RAW_ROLE_NORM, 12288ull, 0u, 0ull, 12288ull, 6144u, 0u};
            return 1;
        case GLM5_RAW_TENSOR_POST_ATTENTION_LAYER_NORM:
            *out = {GLM5_RAW_ROLE_NORM, 12288ull, 1u, 0ull, 12288ull, 6144u, 0u};
            return 1;
        case GLM5_RAW_TENSOR_Q_A_LAYER_NORM:
            *out = {GLM5_RAW_ROLE_NORM, 4096ull, 0u, 0ull, 4096ull, 2048u, 0u};
            return 1;
        case GLM5_RAW_TENSOR_KV_A_LAYER_NORM:
            *out = {GLM5_RAW_ROLE_NORM, 1024ull, 0u, 0ull, 1024ull, 512u, 0u};
            return 1;
        case GLM5_RAW_TENSOR_Q_A_PROJ:
            *out = {GLM5_RAW_ROLE_ATTENTION, 92274688ull, 0u, 0ull, 25165824ull, 2048u, 6144u};
            return 1;
        case GLM5_RAW_TENSOR_Q_B_PROJ:
            *out = {GLM5_RAW_ROLE_ATTENTION, 92274688ull, 0u, 25165824ull, 67108864ull, 16384u, 2048u};
            return 1;
        case GLM5_RAW_TENSOR_KV_A_PROJ_WITH_MQA:
            *out = {GLM5_RAW_ROLE_ATTENTION, 237764608ull, 0u, 0ull, 7077888ull, 576u, 6144u};
            return 1;
        case GLM5_RAW_TENSOR_KV_B_PROJ:
            *out = {GLM5_RAW_ROLE_ATTENTION, 237764608ull, 0u, 7077888ull, 29360128ull, 28672u, 512u};
            return 1;
        case GLM5_RAW_TENSOR_O_PROJ:
            *out = {GLM5_RAW_ROLE_ATTENTION, 237764608ull, 0u, 36438016ull, 201326592ull, 6144u, 16384u};
            return 1;
        case GLM5_RAW_TENSOR_INDEXER_K_NORM_BIAS:
            *out = {GLM5_RAW_ROLE_ATTENTION, 18743808ull, 0u, 0ull, 256ull, 128u, 0u};
            return 1;
        case GLM5_RAW_TENSOR_INDEXER_K_NORM_WEIGHT:
            *out = {GLM5_RAW_ROLE_ATTENTION, 18743808ull, 0u, 256ull, 256ull, 128u, 0u};
            return 1;
        case GLM5_RAW_TENSOR_INDEXER_WEIGHTS_PROJ:
            *out = {GLM5_RAW_ROLE_ATTENTION, 18743808ull, 0u, 512ull, 393216ull, 32u, 6144u};
            return 1;
        case GLM5_RAW_TENSOR_INDEXER_WK:
            *out = {GLM5_RAW_ROLE_ATTENTION, 18743808ull, 0u, 393728ull, 1572864ull, 128u, 6144u};
            return 1;
        case GLM5_RAW_TENSOR_INDEXER_WQ_B:
            *out = {GLM5_RAW_ROLE_ATTENTION, 18743808ull, 0u, 1966592ull, 16777216ull, 4096u, 2048u};
            return 1;
        default:
            return 0;
    }
}

static int find_raw_tensor_desc_and_span(
    int32_t layer,
    glm5_raw_tensor_kind_t kind,
    glm5_raw_tensor_desc* out_desc,
    const glm5_storage_raw_span_spec_t** out_span
) {
    glm5_raw_tensor_desc desc{};
    if (!raw_tensor_desc_for_kind(kind, &desc)) {
        return 0;
    }
    const glm5_storage_raw_span_spec_t* span =
        find_raw_role_span_bytes(layer, desc.role, desc.span_bytes, desc.occurrence);
    if (!span && desc.role == GLM5_RAW_ROLE_ATTENTION) {
        // Boundary layers can pack the 18,743,808-byte indexer span and the
        // 92,274,688-byte Q span into one 111,018,496-byte attention span.
        const uint64_t combined_indexer_q_bytes = 18743808ull + 92274688ull;
        if (desc.span_bytes == 92274688ull) {
            const glm5_storage_raw_span_spec_t* combined =
                find_raw_role_span_bytes(layer, GLM5_RAW_ROLE_ATTENTION, combined_indexer_q_bytes, 0);
            if (combined) {
                desc.rel_offset += 18743808ull;
                span = combined;
            }
        } else if (desc.span_bytes == 18743808ull) {
            span = find_raw_role_span_bytes(layer, GLM5_RAW_ROLE_ATTENTION, combined_indexer_q_bytes, 0);
        }
    }
    if (!span || desc.rel_offset > span->bytes || desc.bytes > span->bytes - desc.rel_offset) {
        return 0;
    }
    if (out_desc) {
        *out_desc = desc;
    }
    if (out_span) {
        *out_span = span;
    }
    return 1;
}
