static const moe_gguf_common_tensor_record* moe_find_layer_common_tensor(
    moe_pc_engine* engine,
    int32_t layer,
    const char* const* suffixes,
    size_t suffix_count,
    moe_raw_tensor_desc* out_desc
) {
    if (!engine || !suffixes || layer < 0) {
        return nullptr;
    }
    std::vector<std::string> names;
    moe_push_layer_tensor_candidates(&names, layer, suffixes, suffix_count);
    return moe_common_tensor_find_first_raw_compatible(engine, names, out_desc);
}

static uint32_t moe_fnv1a_hash_string(const char* str) {
    uint32_t hash = 2166136261u;
    while (*str) {
        hash ^= (uint8_t)*str++;
        hash *= 16777619u;
    }
    return hash;
}

static int moe_map_layer_common_tensor(
    moe_pc_engine* engine,
    int32_t layer,
    const char* const* suffixes,
    size_t suffix_count,
    moe_raw_tensor_cache_entry* out
) {
    if (!engine || !out || suffix_count == 0) {
        return 0;
    }
    // Fix 18: Cache tensor mapping to avoid std::string vector allocation per token
    const uint32_t suffix_hash = moe_fnv1a_hash_string(suffixes[0]);
    const uint64_t key = ((uint64_t)(uint32_t)layer << 32) | suffix_hash | 0x80000000ull;
    {
        std::shared_lock<std::shared_mutex> lock(engine->raw_tensor_cache_mutex);
        auto found = engine->raw_tensor_cache.find(key);
        if (found != engine->raw_tensor_cache.end() && found->second.base) {
            *out = found->second;
            return 1;
        }
    }
    std::vector<std::string> names;
    moe_push_layer_tensor_candidates(&names, layer, suffixes, suffix_count);
    moe_raw_tensor_desc desc{};
    const moe_gguf_common_tensor_record* rec =
        moe_common_tensor_find_first_raw_compatible(engine, names, &desc);
    if (!rec) {
        return 0;
    }
    if (!map_common_tensor_record(engine, *rec, desc, out)) {
        return 0;
    }
    {
        std::unique_lock<std::shared_mutex> lock(engine->raw_tensor_cache_mutex);
        engine->raw_tensor_cache[key] = *out;
    }
    return 1;
}

static int moe_layer_common_tensor_exists(
    moe_pc_engine* engine,
    int32_t layer,
    const char* const* suffixes,
    size_t suffix_count
) {
    if (!engine || !suffixes || layer < 0 || suffix_count == 0) {
        return 0;
    }
    moe_raw_tensor_desc desc{};
    return moe_find_layer_common_tensor(engine, layer, suffixes, suffix_count, &desc) ? 1 : 0;
}

static int moe_graph_ir_math_required(const moe_pc_engine* engine) {
    return engine &&
        engine->offload_graph_ir_required.load(std::memory_order_relaxed) &&
        engine->offload_graph_ir_ready.load(std::memory_order_relaxed);
}

static int moe_map_layer_tensor_by_roles_or_suffixes(
    moe_pc_engine* engine,
    int32_t layer,
    const char* const* roles,
    size_t role_count,
    const char* const* suffixes,
    size_t suffix_count,
    moe_raw_tensor_cache_entry* out,
    int* out_declared = nullptr
) {
    if (out_declared) {
        *out_declared = 0;
    }
    if (!engine || !out || layer < 0) {
        return 0;
    }
    if (moe_graph_ir_math_required(engine)) {
        if (out_declared) {
            *out_declared = moe_graph_ir_layer_has_op_role_any(engine, layer, roles, role_count);
        }
        return moe_graph_ir_map_layer_op_role_any(engine, layer, roles, role_count, out);
    }
    if (out_declared) {
        *out_declared = moe_layer_common_tensor_exists(engine, layer, suffixes, suffix_count);
    }
    return moe_map_layer_common_tensor(engine, layer, suffixes, suffix_count, out);
}

static int moe_common_linear_f32(
    moe_pc_engine* engine,
    const moe_raw_tensor_cache_entry& raw,
    const float* input,
    uint32_t input_count,
    float* output,
    uint32_t output_count
) {
    if (!engine || !input || !output ||
        raw.cols != input_count || raw.rows != output_count) {
        return 0;
    }
    // BUGFIX 506: Check raw.base validity before computing row_bytes ??
    // Problem: raw.base can be null if tensor mapping failed
    // moe_raw_tensor_row_bytes may dereference raw.base internally
    // Solution: Check raw.base before use
    if (!raw.base) {
        return 0;
    }
    const uint64_t row_bytes = moe_raw_tensor_row_bytes(raw, raw.cols);
    if (row_bytes == 0 || raw.bytes < row_bytes * (uint64_t)raw.rows) {
        return 0;
    }
    moe_profile_parallel_label_scope rows_label("common_linear");
    return moe_parallel_rows(engine, raw.rows, 0, [&](uint32_t row) {
        output[row] = moe_raw_tensor_dot_row(raw, row, input, input_count);
        return 1;
    });
}

static inline float moe_generation_gelu_tanh(float x) {
    // BUGFIX 718: GELU tanh NaN handling ★★
    // Problem: No validation of input x or tanh result
    // Impact: NaN in activation propagates to MLP output
    // Solution: Check for NaN in input and tanh result
    if (!std::isfinite(x)) {
        return 0.0f;
    }
    const float k = 0.7978845608028654f;
    const float inner = k * (x + 0.044715f * x * x * x);
    if (!std::isfinite(inner)) {
        return x > 0.0f ? x : 0.0f;
    }
    const float tanh_val = std::tanh(inner);
    if (!std::isfinite(tanh_val)) {
        return x > 0.0f ? x : 0.0f;
    }
    return 0.5f * x * (1.0f + tanh_val);
}

static inline float moe_generation_gated_activation_cached(int activation_mode, float gate, float up) {
    if (!std::isfinite(gate) || !std::isfinite(up)) {
        return 0.0f;
    }
    if (activation_mode == 2) {
        const float result = moe_generation_gelu_tanh(gate) * up;
        if (!std::isfinite(result)) {
            return 0.0f;
        }
        return result;
    }
    const float activated = (activation_mode == 1 ? moe_generation_gelu(gate) : moe_generation_silu(gate));
    if (!std::isfinite(activated)) {
        return 0.0f;
    }
    const float result = activated * up;
    if (!std::isfinite(result)) {
        return 0.0f;
    }
    return result;
}
static int moe_apply_rmsnorm_common_weight_f32(
    const moe_pc_engine* engine,
    const moe_raw_tensor_cache_entry& raw,
    const float* input,
    uint32_t count,
    float* output,
    float eps
);

static int moe_layer_rmsnorm_suffixes_f32(
    moe_pc_engine* engine,
    int32_t layer,
    const char* const* suffixes,
    size_t suffix_count,
    const float* input,
    uint32_t hidden_size,
    float* output,
    float eps
);

static int moe_graph_ir_layer_has_op_role_any(
    const moe_pc_engine* engine,
    int32_t layer,
    const char* const* roles,
    size_t role_count
);

static int moe_graph_ir_map_layer_op_role_any(
    moe_pc_engine* engine,
    int32_t layer,
    const char* const* roles,
    size_t role_count,
    moe_raw_tensor_cache_entry* out
);

static int moe_graph_ir_apply_rmsnorm_role_any_f32(
    moe_pc_engine* engine,
    int32_t layer,
    const char* const* roles,
    size_t role_count,
    const float* input,
    uint32_t hidden_size,
    float* output,
    float eps
);

static int moe_graph_ir_math_required(const moe_pc_engine* engine);
static int moe_graph_ir_layer_has_op_role_any(
    const moe_pc_engine* engine,
    int32_t layer,
    const char* const* roles,
    size_t role_count
);

static int moe_graph_ir_map_layer_op_role_any(
    moe_pc_engine* engine,
    int32_t layer,
    const char* const* roles,
    size_t role_count,
    moe_raw_tensor_cache_entry* out
);

struct moe_ple_binding_t {
    moe_raw_tensor_cache_entry token_embedding{};
    moe_raw_tensor_cache_entry model_projection{};
    moe_raw_tensor_cache_entry projection_norm{};
    uint32_t total_dim = 0;
    uint32_t layer_dim = 0;
    int has_token_embedding = 0;
};

static const moe_gguf_common_tensor_record* moe_find_common_tensor_by_roles_or_names(
    moe_pc_engine* engine,
    const char* const* roles,
    size_t role_count,
    const char* const* names,
    size_t name_count,
    moe_raw_tensor_desc* out_desc
) {
    if (!engine) {
        return nullptr;
    }
    if (roles) {
        for (size_t i = 0; i < role_count; ++i) {
            if (!roles[i] || !roles[i][0]) {
                continue;
            }
            if (const moe_gguf_common_tensor_record* rec =
                    find_common_tensor_by_op_role(engine, roles[i], INT32_MIN, out_desc)) {
                return rec;
            }
        }
    }
    return find_common_tensor_by_names(engine, names, name_count, out_desc);
}

static int moe_map_common_tensor_by_roles_or_names(
    moe_pc_engine* engine,
    const char* const* roles,
    size_t role_count,
    const char* const* names,
    size_t name_count,
    moe_raw_tensor_cache_entry* out,
    moe_raw_tensor_desc* out_desc = nullptr
) {
    moe_raw_tensor_desc desc{};
    const moe_gguf_common_tensor_record* rec =
        moe_find_common_tensor_by_roles_or_names(engine, roles, role_count, names, name_count, &desc);
    if (!rec) {
        return 0;
    }
    if (!map_common_tensor_record(engine, *rec, desc, out)) {
        return 0;
    }
    if (out_desc) {
        *out_desc = desc;
    }
    return 1;
}

static int moe_map_ple_binding(
    moe_pc_engine* engine,
    const moe_storage_constants_t* c,
    moe_ple_binding_t* out
) {
    if (!engine || !c || !out || c->hidden_size == 0 || c->num_hidden_layers == 0) {
        return 0;
    }
    static const char* const proj_roles[] = {
        "per_layer_model_projection",
        "per_layer_model_proj"
    };
    static const char* const proj_names[] = {
        "per_layer_model_proj.weight",
        "per_layer_model_projection.weight",
        "model.per_layer_model_projection.weight",
        "model.per_layer_model_proj.weight"
    };
    moe_raw_tensor_desc proj_desc{};
    moe_raw_tensor_cache_entry model_projection{};
    if (!moe_map_common_tensor_by_roles_or_names(
            engine, proj_roles, sizeof(proj_roles) / sizeof(proj_roles[0]),
            proj_names, sizeof(proj_names) / sizeof(proj_names[0]),
            &model_projection, &proj_desc)) {
        return 0;
    }
    if (proj_desc.cols != c->hidden_size ||
        proj_desc.rows == 0 ||
        (proj_desc.rows % c->num_hidden_layers) != 0u) {
        moe_profile_log(
            "ple_model_projection_rejected",
            "rows=%u cols=%u layers=%u hidden=%u",
            proj_desc.rows, proj_desc.cols, c->num_hidden_layers, c->hidden_size);
        return 0;
    }
    const uint32_t total_dim = proj_desc.rows;
    const uint32_t layer_dim = total_dim / c->num_hidden_layers;
    if (layer_dim == 0 || layer_dim > 100000u || total_dim > 10000000u) {
        return 0;
    }

    static const char* const norm_roles[] = {
        "per_layer_projection_norm",
        "per_layer_proj_norm"
    };
    static const char* const norm_names[] = {
        "per_layer_proj_norm.weight",
        "per_layer_projection_norm.weight",
        "model.per_layer_projection_norm.weight",
        "model.per_layer_proj_norm.weight"
    };
    moe_raw_tensor_desc norm_desc{};
    moe_raw_tensor_cache_entry projection_norm{};
    if (!moe_map_common_tensor_by_roles_or_names(
            engine, norm_roles, sizeof(norm_roles) / sizeof(norm_roles[0]),
            norm_names, sizeof(norm_names) / sizeof(norm_names[0]),
            &projection_norm, &norm_desc) ||
        norm_desc.rows < layer_dim) {
        moe_profile_log(
            "ple_projection_norm_missing",
            "layer_dim=%u norm_rows=%u",
            layer_dim, norm_desc.rows);
        return 0;
    }

    static const char* const token_roles[] = {
        "per_layer_token_embedding",
        "per_layer_token_embd",
        "embed_tokens_per_layer"
    };
    static const char* const token_names[] = {
        "per_layer_token_embd.weight",
        "embed_tokens_per_layer.weight",
        "model.embed_tokens_per_layer.weight",
        "model.per_layer_token_embd.weight"
    };
    moe_raw_tensor_desc token_desc{};
    moe_raw_tensor_cache_entry token_embedding{};
    const int has_token = moe_map_common_tensor_by_roles_or_names(
        engine, token_roles, sizeof(token_roles) / sizeof(token_roles[0]),
        token_names, sizeof(token_names) / sizeof(token_names[0]),
        &token_embedding, &token_desc);
    if (has_token &&
        (token_desc.cols != total_dim || token_desc.rows < c->vocab_size)) {
        moe_profile_log(
            "ple_token_embedding_rejected",
            "rows=%u cols=%u vocab=%u total_dim=%u",
            token_desc.rows, token_desc.cols, c->vocab_size, total_dim);
        return 0;
    }

    out->token_embedding = token_embedding;
    out->model_projection = model_projection;
    out->projection_norm = projection_norm;
    out->total_dim = total_dim;
    out->layer_dim = layer_dim;
    out->has_token_embedding = has_token ? 1 : 0;
    return 1;
}

static int moe_prepare_ple_inputs_f32(
    moe_pc_engine* engine,
    const moe_storage_constants_t* c,
    moe_decode_scratch* s,
    uint32_t token_id,
    float eps
) {
    if (!engine || !c || !s) {
        return 0;
    }
    moe_ple_binding_t ple{};
    if (!moe_map_ple_binding(engine, c, &ple)) {
        s->ple_inputs.clear();
        s->ple_projected.clear();
        return 1;
    }
    if (token_id >= c->vocab_size) {
        return 0;
    }
    try {
        s->ple_inputs.resize(ple.total_dim);
        s->ple_projected.resize(ple.total_dim);
    } catch (const std::bad_alloc&) {
        return 0;
    }

    if (ple.has_token_embedding) {
        if (!moe_raw_tensor_copy_row_f32(
                ple.token_embedding, token_id, s->ple_inputs.data(), ple.total_dim)) {
            return 0;
        }
        const float token_scale = std::sqrt((float)ple.layer_dim);
        if (std::isfinite(token_scale) && token_scale != 1.0f) {
            moe_scale_f32(s->ple_inputs.data(), token_scale, ple.total_dim);
        }
    } else {
        std::fill(s->ple_inputs.begin(), s->ple_inputs.end(), 0.0f);
    }

    const float projection_scale = 1.0f / std::sqrt((float)c->hidden_size);
    const int projected_ok = moe_parallel_rows(engine, ple.total_dim, 0, [&](uint32_t row) {
        float v = moe_raw_tensor_dot_row(ple.model_projection, row, s->hidden.data(), c->hidden_size);
        if (!std::isfinite(v)) {
            v = 0.0f;
        }
        s->ple_projected[row] = v * projection_scale;
        return 1;
    });
    if (!projected_ok) {
        return 0;
    }
    for (uint32_t layer = 0; layer < c->num_hidden_layers; ++layer) {
        float* slice = s->ple_projected.data() + (size_t)layer * (size_t)ple.layer_dim;
        if (!moe_apply_rmsnorm_common_weight_f32(
                engine, ple.projection_norm, slice, ple.layer_dim, slice,
                eps > 0.0f ? eps : 1.0e-6f)) {
            return 0;
        }
    }
    const float combine_scale = ple.has_token_embedding ? 0.7071067811865476f : 1.0f;
    for (uint32_t i = 0; i < ple.total_dim; ++i) {
        s->ple_inputs[i] = (s->ple_inputs[i] + s->ple_projected[i]) * combine_scale;
    }
    static std::atomic<uint32_t> logged{0};
    if (logged.fetch_add(1, std::memory_order_relaxed) == 0) {
        moe_profile_log(
            "ple_prepare",
            "total_dim=%u layer_dim=%u layers=%u token_embedding=%d projection_scale=%.9g",
            ple.total_dim, ple.layer_dim, c->num_hidden_layers, ple.has_token_embedding,
            (double)projection_scale);
    }
    return 1;
}

static int moe_layer_has_ple_tensors(
    const moe_pc_engine* engine,
    int32_t layer
) {
    return
        (moe_engine_has_common_tensor_suffix(engine, layer, "inp_gate.weight") ||
         moe_engine_has_common_tensor_suffix(engine, layer, "per_layer_input_gate.weight")) &&
        (moe_engine_has_common_tensor_suffix(engine, layer, "proj.weight") ||
         moe_engine_has_common_tensor_suffix(engine, layer, "per_layer_projection.weight")) &&
        (moe_engine_has_common_tensor_suffix(engine, layer, "post_norm.weight") ||
         moe_engine_has_common_tensor_suffix(engine, layer, "post_per_layer_input_norm.weight"));
}

static int moe_layer_apply_ple_f32(
    moe_pc_engine* engine,
    int32_t layer,
    moe_decode_scratch* s,
    float* hidden,
    uint32_t hidden_size,
    uint32_t num_layers,
    float eps
) {
    if (!engine || !s || !hidden || layer < 0 || hidden_size == 0 ||
        s->ple_inputs.empty()) {
        return 1;
    }
    if (num_layers == 0 ||
        (s->ple_inputs.size() % (size_t)num_layers) != 0u ||
        (uint32_t)layer >= num_layers) {
        return 0;
    }
    const uint32_t expected_layer_dim =
        (uint32_t)(s->ple_inputs.size() / (size_t)num_layers);
    static const char* const gate_suffixes[] = {
        "inp_gate.weight",
        "per_layer_input_gate.weight"
    };
    static const char* const proj_suffixes[] = {
        "proj.weight",
        "per_layer_projection.weight"
    };
    static const char* const norm_suffixes[] = {
        "post_norm.weight",
        "post_per_layer_input_norm.weight"
    };
    moe_raw_tensor_cache_entry gate_raw{};
    moe_raw_tensor_cache_entry proj_raw{};
    if (!moe_map_layer_common_tensor(
            engine, layer, gate_suffixes, sizeof(gate_suffixes) / sizeof(gate_suffixes[0]), &gate_raw) ||
        !moe_map_layer_common_tensor(
            engine, layer, proj_suffixes, sizeof(proj_suffixes) / sizeof(proj_suffixes[0]), &proj_raw)) {
        if (moe_layer_has_ple_tensors(engine, layer)) {
            moe_profile_log("ple_layer_map_failed", "layer=%d hidden=%u", layer, hidden_size);
            return 0;
        }
        return 1;
    }
    const uint32_t layer_dim = gate_raw.rows;
    const size_t offset = (size_t)layer * (size_t)layer_dim;
    if (gate_raw.cols != hidden_size ||
        layer_dim != expected_layer_dim ||
        proj_raw.cols != layer_dim ||
        proj_raw.rows != hidden_size ||
        layer_dim == 0 ||
        offset > s->ple_inputs.size() ||
        layer_dim > s->ple_inputs.size() - offset ||
        layer_dim > s->gate.size()) {
        moe_profile_log(
            "ple_layer_shape_mismatch",
            "layer=%d gate=%ux%u proj=%ux%u hidden=%u ple_size=%llu offset=%llu",
            layer, gate_raw.rows, gate_raw.cols, proj_raw.rows, proj_raw.cols, hidden_size,
            (unsigned long long)s->ple_inputs.size(), (unsigned long long)offset);
        return 0;
    }
    std::memcpy(s->residual.data(), hidden, (size_t)hidden_size * sizeof(float));
    const int activation_mode = moe_engine_gelu_activation_mode(engine);
    const float* ple_layer = s->ple_inputs.data() + offset;
    const int gate_ok = moe_parallel_rows(engine, layer_dim, 0, [&](uint32_t row) {
        float v = moe_raw_tensor_dot_row(gate_raw, row, hidden, hidden_size);
        v = moe_generation_gated_activation_cached(activation_mode, v, 1.0f);
        if (!std::isfinite(v)) {
            v = 0.0f;
        }
        s->gate[row] = v * ple_layer[row];
        return 1;
    });
    if (!gate_ok) {
        return 0;
    }
    if (!moe_common_linear_f32(engine, proj_raw, s->gate.data(), layer_dim, s->norm.data(), hidden_size)) {
        return 0;
    }
    if (!moe_layer_rmsnorm_suffixes_f32(
            engine, layer, norm_suffixes, sizeof(norm_suffixes) / sizeof(norm_suffixes[0]),
            s->norm.data(), hidden_size, s->attn.data(), eps)) {
        return 0;
    }
    for (uint32_t i = 0; i < hidden_size; ++i) {
        hidden[i] = s->residual[i] + s->attn[i];
    }
    moe_profile_log(
        "ple_layer_apply",
        "layer=%d hidden=%u layer_dim=%u",
        layer, hidden_size, layer_dim);
    return 1;
}

static int moe_raw_tensor_scalar_at_f32(
    const moe_raw_tensor_cache_entry& raw,
    uint64_t offset,
    float* out
) {
    if (!raw.base || !out) {
        return 0;
    }
    const uint32_t elem = moe_raw_scalar_size(raw.scalar_encoding);
    if (elem == 0 || offset > UINT64_MAX / (uint64_t)elem) {
        return 0;
    }
    const uint64_t byte_offset = offset * (uint64_t)elem;
    if (byte_offset > raw.bytes || (uint64_t)elem > raw.bytes - byte_offset) {
        return 0;
    }
    *out = moe_raw_scalar_to_f32(raw.base + byte_offset, raw.scalar_encoding);
    return 1;
}

static uint64_t moe_raw_tensor_scalar_count(const moe_raw_tensor_cache_entry& raw) {
    const uint32_t elem = moe_raw_scalar_size(raw.scalar_encoding);
    if (!elem) {
        return 0;
    }
    uint64_t logical_count = 0;
    if (raw.cols) {
        if (raw.rows > UINT64_MAX / (uint64_t)raw.cols) {
            return raw.bytes / (uint64_t)elem;
        }
        logical_count = (uint64_t)raw.rows * (uint64_t)raw.cols;
    } else {
        logical_count = raw.rows;
    }
    if (logical_count && logical_count <= raw.bytes / (uint64_t)elem) {
        return logical_count;
    }
    return raw.bytes / (uint64_t)elem;
}

static int moe_apply_rmsnorm_common_weight_f32(
    const moe_pc_engine* engine,
    const moe_raw_tensor_cache_entry& raw,
    const float* input,
    uint32_t count,
    float* output,
    float eps
) {
    // BUGFIX 710: RMSNorm input NaN handling ★★★
    // Problem: No validation of input values before computing sum of squares
    // Impact: NaN in input propagates to all normalized outputs
    // Solution: Check for NaN in input and skip NaN values in sum
    // BUGFIX 208: Division by zero when count == 0
    // Problem: ss / (double)count crashes if count == 0
    // Solution: Check count == 0 BEFORE division (already done), but clarify comment
    if (!input || !output || count == 0 || moe_raw_tensor_scalar_count(raw) < count) {
        return 0;
    }
    double ss = 0.0;
    uint32_t valid_count = 0;
    for (uint32_t i = 0; i < count; ++i) {
        if (!std::isfinite(input[i])) {
            return 0;
        }
        ss += (double)input[i] * (double)input[i];
        ++valid_count;
    }
    if (valid_count == 0) {
        return 0;
    }
    // Safe: valid_count > 0 guaranteed by check above
    const float norm = 1.0f / std::sqrt((float)(ss / (double)valid_count) + (eps > 0.0f ? eps : 1.0e-6f));
    const int unit_offset = moe_engine_rmsnorm_unit_offset(engine, raw);
    for (uint32_t i = 0; i < count; ++i) {
        float w = 0.0f;
        if (!moe_raw_tensor_scalar_at_f32(raw, i, &w)) {
            return 0;
        }
        if (!std::isfinite(w) || !std::isfinite(input[i])) {
            return 0;
        }
        const float result = input[i] * norm * (unit_offset ? (1.0f + w) : w);
        if (!std::isfinite(result)) {
            return 0;
        }
        output[i] = result;
    }
    return 1;
}

static int moe_layer_rmsnorm_suffixes_f32(
    moe_pc_engine* engine,
    int32_t layer,
    const char* const* suffixes,
    size_t suffix_count,
    const float* input,
    uint32_t hidden_size,
    float* output,
    float eps
) {
    moe_raw_tensor_cache_entry raw{};
    const int mapped = moe_map_layer_common_tensor(engine, layer, suffixes, suffix_count, &raw);
    moe_profile_log(
        "rmsnorm_common_map",
        "layer=%d suffix0=%s mapped=%d rows=%u cols=%u bytes=%llu hidden=%u input=%p output=%p",
        layer,
        (suffixes && suffix_count > 0 && suffixes[0]) ? suffixes[0] : "",
        mapped,
        raw.rows,
        raw.cols,
        (unsigned long long)raw.bytes,
        hidden_size,
        (const void*)input,
        (void*)output);
    if (!mapped) {
        return 0;
    }
    const int ok = moe_apply_rmsnorm_common_weight_f32(engine, raw, input, hidden_size, output, eps);
    moe_profile_log(
        "rmsnorm_common_apply",
        "layer=%d suffix0=%s ok=%d hidden=%u scalar=%u",
        layer,
        (suffixes && suffix_count > 0 && suffixes[0]) ? suffixes[0] : "",
        ok,
        hidden_size,
        (unsigned)raw.scalar_encoding);
    return ok;
}

static int moe_layer_ffn_norm_f32(
    moe_pc_engine* engine,
    int32_t layer,
    const float* input,
    uint32_t hidden_size,
    float* output,
    float eps
) {
    static const char* const suffixes[] = {
        "ffn_norm.weight",
        "ffn_pre_norm.weight",
        "pre_ffw_norm.weight",
        "mlp_norm.weight"
    };
    return moe_layer_rmsnorm_suffixes_f32(
        engine, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]),
        input, hidden_size, output, eps);
}

static int moe_layer_has_ffn_norm_weight(
    const moe_pc_engine* engine,
    int32_t layer
) {
    return
        moe_engine_has_common_tensor_suffix(engine, layer, "ffn_norm.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "ffn_pre_norm.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "pre_ffw_norm.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "mlp_norm.weight");
}

static int moe_layer_moe_norm_f32(
    moe_pc_engine* engine,
    int32_t layer,
    const float* input,
    uint32_t hidden_size,
    float* output,
    float eps
) {
    static const char* const suffixes[] = {
        "pre_ffw_norm_2.weight",
        "ffn_pre_norm_2.weight",
        "moe_norm.weight"
    };
    return moe_layer_rmsnorm_suffixes_f32(
        engine, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]),
        input, hidden_size, output, eps);
}

static int moe_layer_has_moe_norm_weight(
    const moe_pc_engine* engine,
    int32_t layer
) {
    return
        moe_engine_has_common_tensor_suffix(engine, layer, "pre_ffw_norm_2.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "ffn_pre_norm_2.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "moe_norm.weight");
}

static int moe_layer_post_attention_output_norm_f32(
    moe_pc_engine* engine,
    int32_t layer,
    const float* input,
    uint32_t hidden_size,
    float* output,
    float eps
) {
    static const char* const suffixes[] = {
        "post_attention_norm.weight",
        "post_attention_layernorm.weight",
        "post_attention_layer_norm.weight",
        "post_attn_norm.weight"
    };
    return moe_layer_rmsnorm_suffixes_f32(
        engine, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]),
        input, hidden_size, output, eps);
}

static int moe_layer_has_post_attention_norm_weight(
    const moe_pc_engine* engine,
    int32_t layer
) {
    return
        moe_engine_has_common_tensor_suffix(engine, layer, "post_attention_norm.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "post_attention_layernorm.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "post_attention_layer_norm.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "post_attn_norm.weight");
}

static int moe_final_norm_f32(
    moe_pc_engine* engine,
    const moe_storage_constants_t* c,
    const float* input,
    uint32_t hidden_size,
    float* output,
    float eps
) {
    if (!engine || !c || !input || !output || hidden_size == 0) {
        return 0;
    }
    std::string graph_name;
    if (moe_graph_ir_tensor_binding_name(engine, "final_norm", &graph_name)) {
        moe_raw_tensor_desc desc{};
        const char* exact_name = graph_name.c_str();
        const moe_gguf_common_tensor_record* rec =
            find_common_tensor_by_names(engine, &exact_name, 1, &desc);
        if (rec && desc.rows >= hidden_size) {
            moe_raw_tensor_cache_entry raw{};
            if (!map_common_tensor_record(engine, *rec, desc, &raw)) {
                return 0;
            }
            static std::atomic<uint32_t> logged{0};
            if (logged.fetch_add(1, std::memory_order_relaxed) == 0) {
                moe_profile_log(
                    "final_norm_contract_binding",
                    "name=%s rows=%u cols=%u hidden=%u",
                    graph_name.c_str(), desc.rows, desc.cols, hidden_size);
            }
            return moe_apply_rmsnorm_common_weight_f32(engine, raw, input, hidden_size, output, eps);
        }
        moe_profile_log(
            "final_norm_contract_binding_rejected",
            "name=%s rows=%u cols=%u hidden=%u",
            graph_name.c_str(), desc.rows, desc.cols, hidden_size);
        return 0;
    }
    static const char* const suffixes[] = {
        "output_norm.weight",
        "model.norm.weight",
        "norm.weight",
        "transformer.ln_f.weight"
    };
    {
        moe_raw_tensor_desc desc{};
        const moe_gguf_common_tensor_record* rec =
            find_common_tensor_by_op_role(engine, "final_norm", -1, &desc);
        if (rec && desc.rows >= hidden_size) {
            moe_raw_tensor_cache_entry raw{};
            if (!map_common_tensor_record(engine, *rec, desc, &raw)) {
                return 0;
            }
            return moe_apply_rmsnorm_common_weight_f32(engine, raw, input, hidden_size, output, eps);
        }
    }
    return moe_layer_rmsnorm_suffixes_f32(
        engine, (int32_t)c->num_hidden_layers, suffixes,
        sizeof(suffixes) / sizeof(suffixes[0]),
        input, hidden_size, output, eps);
}

// Cache split/post-norm detection per layer; layer 0 must not decide every layer.
static int moe_layer_uses_split_post_norm_block(const moe_pc_engine* engine, int32_t layer) {
    if (!engine || layer < 0) {
        return 0;
    }
    static thread_local const moe_pc_engine* tls_cached_engine = nullptr;
    static thread_local uint64_t tls_cached_generation = 0;
    static thread_local std::vector<int8_t> tls_cached_results;
    const uint64_t generation = engine->engine_generation;
    const size_t layer_index = (size_t)layer;
    int8_t* cached = nullptr;
    if (layer_index < 10000u) {
        if (tls_cached_engine != engine || tls_cached_generation != generation) {
            tls_cached_engine = engine;
            tls_cached_generation = generation;
            tls_cached_results.clear();
        }
        if (tls_cached_results.size() <= layer_index) {
            tls_cached_results.resize(layer_index + 1u, -1);
        }
        cached = &tls_cached_results[layer_index];
        if (*cached >= 0) {
            return *cached;
        }
    }
    int result = 0;
    if (moe_graph_ir_math_required(engine)) {
        static const char* const graph_roles[] = {
            "post_ffw_norm", "post_ffw_norm_1", "post_ffw_norm_2", "layer_output_scale"
        };
        result = moe_graph_ir_layer_has_op_role_any(
            engine, layer, graph_roles, sizeof(graph_roles) / sizeof(graph_roles[0]));
    } else if (moe_engine_has_common_tensor_suffix(engine, layer, "post_ffw_norm.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "post_ffw_norm_1.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "post_ffw_norm_2.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "layer_output_scale.weight")) {
        result = 1;
    } else {
        // Do not let a global GraphIR substring enable split-post-norm for every layer.
        // Per-layer behavior must come from a layer-local tensor/op binding.
        result = 0;
    }
    if (cached) {
        *cached = result ? 1 : 0;
    }
    return result;
}

static size_t moe_graph_ir_first_pos_any(
    const moe_pc_engine* engine,
    const char* const* needles,
    size_t needle_count,
    size_t begin = 0
) {
    if (!engine || !needles || engine->offload_graph_ir_json.empty()) {
        return std::string::npos;
    }
    size_t best = std::string::npos;
    for (size_t i = 0; i < needle_count; ++i) {
        if (!needles[i] || !needles[i][0]) {
            continue;
        }
        const size_t pos = moe_ascii_find_ci(engine->offload_graph_ir_json, needles[i], begin);
        if (pos != std::string::npos && (best == std::string::npos || pos < best)) {
            best = pos;
        }
    }
    return best;
}

static int moe_graph_ir_attention_residual_before_ffn_norm(const moe_pc_engine* engine, int32_t layer) {
    (void)engine;
    (void)layer;
    // Disabled: the previous implementation used whole-GraphIR substring ordering,
    // which is not a per-layer edge and can reorder every layer incorrectly.
    return 0;
}


static int moe_graph_ir_layer_has_op_role_any(
    const moe_pc_engine* engine,
    int32_t layer,
    const char* const* roles,
    size_t role_count
) {
    if (!engine || !roles || layer < 0) {
        return 0;
    }
    moe_raw_tensor_desc desc{};
    for (size_t i = 0; i < role_count; ++i) {
        if (!roles[i] || !roles[i][0]) {
            continue;
        }
        if (find_common_tensor_by_op_role(engine, roles[i], layer, &desc)) {
            return 1;
        }
    }
    return 0;
}

static int moe_graph_ir_map_layer_op_role_any(
    moe_pc_engine* engine,
    int32_t layer,
    const char* const* roles,
    size_t role_count,
    moe_raw_tensor_cache_entry* out
) {
    if (!engine || !roles || !out || layer < 0) {
        return 0;
    }
    for (size_t i = 0; i < role_count; ++i) {
        if (!roles[i] || !roles[i][0]) {
            continue;
        }
        moe_raw_tensor_desc desc{};
        const moe_gguf_common_tensor_record* rec =
            find_common_tensor_by_op_role(engine, roles[i], layer, &desc);
        if (!rec) {
            continue;
        }
        if (!map_common_tensor_record(engine, *rec, desc, out)) {
            return 0;
        }
        return 1;
    }
    return 0;
}

static int moe_graph_ir_apply_rmsnorm_role_any_f32(
    moe_pc_engine* engine,
    int32_t layer,
    const char* const* roles,
    size_t role_count,
    const float* input,
    uint32_t hidden_size,
    float* output,
    float eps
) {
    moe_raw_tensor_cache_entry raw{};
    if (!moe_graph_ir_map_layer_op_role_any(engine, layer, roles, role_count, &raw)) {
        return 0;
    }
    return moe_apply_rmsnorm_common_weight_f32(engine, raw, input, hidden_size, output, eps);
}

static int moe_layer_post_ffw_norm_suffixes_f32(
    moe_pc_engine* engine,
    int32_t layer,
    const char* const* suffixes,
    size_t suffix_count,
    const float* input,
    uint32_t hidden_size,
    float* output,
    float eps
) {
    return moe_layer_rmsnorm_suffixes_f32(
        engine, layer, suffixes, suffix_count, input, hidden_size, output, eps);
}

static int moe_layer_post_ffw_norm_f32(
    moe_pc_engine* engine,
    int32_t layer,
    const float* input,
    uint32_t hidden_size,
    float* output,
    float eps
) {
    static const char* const roles[] = { "post_ffw_norm" };
    if (moe_graph_ir_math_required(engine)) {
        return moe_graph_ir_apply_rmsnorm_role_any_f32(
            engine, layer, roles, sizeof(roles) / sizeof(roles[0]),
            input, hidden_size, output, eps);
    }
    static const char* const suffixes[] = {
        "post_ffw_norm.weight",
        "ffn_post_norm.weight"
    };
    return moe_layer_post_ffw_norm_suffixes_f32(
        engine, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]),
        input, hidden_size, output, eps);
}

static int moe_layer_has_post_ffw_norm_weight(
    const moe_pc_engine* engine,
    int32_t layer
) {
    static const char* const roles[] = { "post_ffw_norm" };
    if (moe_graph_ir_math_required(engine)) {
        return moe_graph_ir_layer_has_op_role_any(
            engine, layer, roles, sizeof(roles) / sizeof(roles[0]));
    }
    return
        moe_engine_has_common_tensor_suffix(engine, layer, "post_ffw_norm.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "ffn_post_norm.weight");
}

static int moe_layer_post_ffw_norm1_f32(
    moe_pc_engine* engine,
    int32_t layer,
    const float* input,
    uint32_t hidden_size,
    float* output,
    float eps
) {
    static const char* const roles[] = { "post_ffw_norm_1" };
    if (moe_graph_ir_math_required(engine)) {
        return moe_graph_ir_apply_rmsnorm_role_any_f32(
            engine, layer, roles, sizeof(roles) / sizeof(roles[0]),
            input, hidden_size, output, eps);
    }
    static const char* const suffixes[] = {
        "post_ffw_norm_1.weight",
        "ffn_post_norm_1.weight"
    };
    return moe_layer_post_ffw_norm_suffixes_f32(
        engine, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]),
        input, hidden_size, output, eps);
}

static int moe_layer_has_post_ffw_norm1_weight(
    const moe_pc_engine* engine,
    int32_t layer
) {
    static const char* const roles[] = { "post_ffw_norm_1" };
    if (moe_graph_ir_math_required(engine)) {
        return moe_graph_ir_layer_has_op_role_any(
            engine, layer, roles, sizeof(roles) / sizeof(roles[0]));
    }
    return
        moe_engine_has_common_tensor_suffix(engine, layer, "post_ffw_norm_1.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "ffn_post_norm_1.weight");
}

static int moe_layer_post_ffw_norm2_f32(
    moe_pc_engine* engine,
    int32_t layer,
    const float* input,
    uint32_t hidden_size,
    float* output,
    float eps
) {
    static const char* const roles[] = { "post_ffw_norm_2" };
    if (moe_graph_ir_math_required(engine)) {
        return moe_graph_ir_apply_rmsnorm_role_any_f32(
            engine, layer, roles, sizeof(roles) / sizeof(roles[0]),
            input, hidden_size, output, eps);
    }
    static const char* const suffixes[] = {
        "post_ffw_norm_2.weight",
        "ffn_post_norm_2.weight"
    };
    return moe_layer_post_ffw_norm_suffixes_f32(
        engine, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]),
        input, hidden_size, output, eps);
}

static int moe_layer_has_post_ffw_norm2_weight(
    const moe_pc_engine* engine,
    int32_t layer
) {
    static const char* const roles[] = { "post_ffw_norm_2" };
    if (moe_graph_ir_math_required(engine)) {
        return moe_graph_ir_layer_has_op_role_any(
            engine, layer, roles, sizeof(roles) / sizeof(roles[0]));
    }
    return
        moe_engine_has_common_tensor_suffix(engine, layer, "post_ffw_norm_2.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "ffn_post_norm_2.weight");
}

static int moe_layer_output_scale_f32(
    moe_pc_engine* engine,
    int32_t layer,
    float* hidden,
    uint32_t hidden_size
) {
    // BUGFIX 711: Layer output scale NaN handling ★★
    // Problem: No validation of scale values before multiplication
    // Impact: NaN in scale propagates to all hidden states
    // Solution: Check for NaN in scale values
    if (!engine || !hidden || hidden_size == 0) {
        return 0;
    }
    // BUGFIX 507: Check hidden_size upper bound to prevent overflow ??
    // Problem: hidden_size can be arbitrarily large from config
    // Loop for (i = 0; i < hidden_size; ++i) can run billions of iterations
    // Solution: Add reasonable upper bound check (e.g., 100000)
    if (hidden_size > 100000) {
        return 0;
    }
    moe_raw_tensor_cache_entry raw{};
    if (moe_graph_ir_math_required(engine)) {
        static const char* const roles[] = { "layer_output_scale" };
        if (!moe_graph_ir_layer_has_op_role_any(
                engine, layer, roles, sizeof(roles) / sizeof(roles[0]))) {
            return 1;
        }
        if (!moe_graph_ir_map_layer_op_role_any(
                engine, layer, roles, sizeof(roles) / sizeof(roles[0]), &raw)) {
            return 0;
        }
    } else {
        static const char* const suffixes[] = {
            "layer_output_scale.weight",
            "layer_scalar.weight",
            "layer_scalar"
        };
        if (!moe_map_layer_common_tensor(engine, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]), &raw)) {
            return 1;
        }
    }
    const uint64_t count = moe_raw_tensor_scalar_count(raw);
    if (count == 1u) {
        float scale = 1.0f;
        if (!moe_raw_tensor_scalar_at_f32(raw, 0, &scale)) {
            return 0;
        }
        if (!std::isfinite(scale)) {
            return 0;
        }
        moe_scale_f32(hidden, scale, hidden_size);
        return 1;
    }
    if (count < hidden_size) {
        return 0;
    }
    for (uint32_t i = 0; i < hidden_size; ++i) {
        float scale = 1.0f;
        if (!moe_raw_tensor_scalar_at_f32(raw, i, &scale)) {
            return 0;
        }
        if (!std::isfinite(scale)) {
            return 0;
        }
        hidden[i] *= scale;
    }
    return 1;
