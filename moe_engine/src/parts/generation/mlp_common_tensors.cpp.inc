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


static int moe_graph_ir_layer_contract_tensor_names_for_role(
    const moe_pc_engine* engine,
    int32_t layer,
    const char* section_key,
    const char* role_key,
    std::vector<std::string>* out_names
);

static int moe_graph_ir_layer_contract_declares_tensor_role(
    const moe_pc_engine* engine,
    int32_t layer,
    const char* section_key,
    const char* role_key
);

static int moe_graph_ir_map_layer_contract_tensor_role_f32(
    moe_pc_engine* engine,
    int32_t layer,
    const char* section_key,
    const char* role_key,
    moe_raw_tensor_cache_entry* out
);

struct moe_ple_binding_t {
    moe_raw_tensor_cache_entry token_embedding{};
    moe_raw_tensor_cache_entry model_projection{};
    moe_raw_tensor_cache_entry projection_norm{};
    uint32_t total_dim = 0;
    uint32_t layer_dim = 0;
    uint32_t token_vocab_size = 0;
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
        "per_layer_token_embedding.weight",
        "per_layer_embed.weight",
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
        (token_desc.cols != total_dim || token_desc.rows == 0)) {
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
    out->token_vocab_size = has_token ? token_desc.rows : 0;
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
        if (token_id < ple.token_vocab_size) {
            if (!moe_raw_tensor_copy_row_f32(
                    ple.token_embedding, token_id, s->ple_inputs.data(), ple.total_dim)) {
                return 0;
            }
        } else {
            std::fill(s->ple_inputs.begin(), s->ple_inputs.end(), 0.0f);
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
            "total_dim=%u layer_dim=%u layers=%u token_embedding=%d token_vocab=%u projection_scale=%.9g",
            ple.total_dim, ple.layer_dim, c->num_hidden_layers, ple.has_token_embedding, ple.token_vocab_size,
            (double)projection_scale);
    }
    return 1;
}

static int moe_layer_has_ple_tensors(
    const moe_pc_engine* engine,
    int32_t layer
) {
    static const char* const gate_roles[] = {
        "per_layer_inp_gate",
        "per_layer_input_gate",
        "ple_inp_gate",
        "ple_input_gate"
    };
    static const char* const proj_roles[] = {
        "per_layer_proj",
        "per_layer_projection",
        "ple_proj",
        "ple_projection"
    };
    static const char* const norm_roles[] = {
        "per_layer_post_norm",
        "post_per_layer_input_norm",
        "ple_post_norm"
    };
    const int has_gate =
        moe_graph_ir_layer_has_op_role_any(engine, layer, gate_roles, sizeof(gate_roles) / sizeof(gate_roles[0])) ||
        moe_engine_has_common_tensor_suffix(engine, layer, "inp_gate.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "per_layer_inp_gate.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "per_layer_input_gate.weight");
    const int has_proj =
        moe_graph_ir_layer_has_op_role_any(engine, layer, proj_roles, sizeof(proj_roles) / sizeof(proj_roles[0])) ||
        moe_engine_has_common_tensor_suffix(engine, layer, "proj.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "per_layer_proj.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "per_layer_projection.weight");
    const int has_norm =
        moe_graph_ir_layer_has_op_role_any(engine, layer, norm_roles, sizeof(norm_roles) / sizeof(norm_roles[0])) ||
        moe_engine_has_common_tensor_suffix(engine, layer, "post_norm.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "per_layer_post_norm.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "post_per_layer_input_norm.weight");
    return has_gate && has_proj && has_norm;
}

static int moe_map_layer_ple_tensor_role_or_suffix_f32(
    moe_pc_engine* engine,
    int32_t layer,
    const char* const* roles,
    size_t role_count,
    const char* const* suffixes,
    size_t suffix_count,
    moe_raw_tensor_cache_entry* out
) {
    if (!engine || !out) {
        return 0;
    }
    static const char* const ple_sections[] = {
        nullptr, "ple", "norms", "mlp"
    };
    if (roles) {
        for (size_t i = 0; i < role_count; ++i) {
            if (!roles[i] || !roles[i][0]) {
                continue;
            }
            for (const char* section : ple_sections) {
                if (moe_graph_ir_map_layer_contract_tensor_role_f32(engine, layer, section, roles[i], out)) {
                    return 1;
                }
            }
        }
    }
    if (moe_graph_ir_map_layer_op_role_any(engine, layer, roles, role_count, out)) {
        return 1;
    }
    return moe_map_layer_common_tensor(engine, layer, suffixes, suffix_count, out);
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
    if (!engine || !s || !hidden || layer < 0 || hidden_size == 0) {
        return 0;
    }
    const int ple_required = moe_layer_has_ple_tensors(engine, layer);
    if (s->ple_inputs.empty()) {
        if (ple_required) {
            moe_profile_log("ple_layer_missing_inputs", "layer=%d hidden=%u", layer, hidden_size);
            return 0;
        }
        return 1;
    }
    if (num_layers == 0 ||
        (s->ple_inputs.size() % (size_t)num_layers) != 0u ||
        (uint32_t)layer >= num_layers) {
        return 0;
    }
    const uint32_t expected_layer_dim =
        (uint32_t)(s->ple_inputs.size() / (size_t)num_layers);
    static const char* const gate_roles[] = {
        "per_layer_inp_gate",
        "per_layer_input_gate",
        "ple_inp_gate",
        "ple_input_gate"
    };
    static const char* const proj_roles[] = {
        "per_layer_proj",
        "per_layer_projection",
        "ple_proj",
        "ple_projection"
    };
    static const char* const norm_roles[] = {
        "per_layer_post_norm",
        "post_per_layer_input_norm",
        "ple_post_norm"
    };
    static const char* const gate_suffixes[] = {
        "inp_gate.weight",
        "per_layer_inp_gate.weight",
        "per_layer_input_gate.weight"
    };
    static const char* const proj_suffixes[] = {
        "proj.weight",
        "per_layer_proj.weight",
        "per_layer_projection.weight"
    };
    static const char* const norm_suffixes[] = {
        "post_norm.weight",
        "per_layer_post_norm.weight",
        "post_per_layer_input_norm.weight"
    };
    moe_raw_tensor_cache_entry gate_raw{};
    moe_raw_tensor_cache_entry proj_raw{};
    if (!moe_map_layer_ple_tensor_role_or_suffix_f32(
            engine, layer,
            gate_roles, sizeof(gate_roles) / sizeof(gate_roles[0]),
            gate_suffixes, sizeof(gate_suffixes) / sizeof(gate_suffixes[0]), &gate_raw) ||
        !moe_map_layer_ple_tensor_role_or_suffix_f32(
            engine, layer,
            proj_roles, sizeof(proj_roles) / sizeof(proj_roles[0]),
            proj_suffixes, sizeof(proj_suffixes) / sizeof(proj_suffixes[0]), &proj_raw)) {
        if (ple_required) {
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
    if (!moe_graph_ir_apply_rmsnorm_role_any_f32(
            engine, layer, norm_roles, sizeof(norm_roles) / sizeof(norm_roles[0]),
            s->norm.data(), hidden_size, s->attn.data(), eps) &&
        !moe_layer_rmsnorm_suffixes_f32(
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
    if (!engine || !suffixes || suffix_count == 0 || !input || !output || hidden_size == 0) {
        return 0;
    }

    std::vector<std::string> names;
    moe_push_layer_tensor_candidates(&names, layer, suffixes, suffix_count);
    uint32_t tried = 0;

    for (const std::string& name : names) {
        const char* exact = name.c_str();
        moe_raw_tensor_desc desc{};
        const moe_gguf_common_tensor_record* rec =
            find_common_tensor_by_names(engine, &exact, 1, &desc);
        if (!rec || !moe_common_tensor_record_raw_compatible(*rec, desc)) {
            continue;
        }
        moe_raw_tensor_cache_entry raw{};
        if (!map_common_tensor_record(engine, *rec, desc, &raw)) {
            continue;
        }
        ++tried;
        moe_profile_log(
            "rmsnorm_common_map",
            "layer=%d suffix0=%s mapped=1 name=%s rows=%u cols=%u bytes=%llu hidden=%u input=%p output=%p tried_before=%u",
            layer, suffixes[0] ? suffixes[0] : "", name.c_str(),
            raw.rows, raw.cols, (unsigned long long)raw.bytes, hidden_size,
            (const void*)input, (void*)output, tried - 1u);
        const int ok = moe_apply_rmsnorm_common_weight_f32(engine, raw, input, hidden_size, output, eps);
        moe_profile_log(
            "rmsnorm_common_apply",
            "layer=%d suffix0=%s ok=%d hidden=%u scalar=%u name=%s",
            layer, suffixes[0] ? suffixes[0] : "", ok, hidden_size,
            (unsigned)raw.scalar_encoding, name.c_str());
        return ok;
    }

    moe_profile_log(
        "rmsnorm_common_map",
        "layer=%d suffix0=%s mapped=0 tried=%u hidden=%u input=%p output=%p",
        layer, suffixes[0] ? suffixes[0] : "", tried,
        hidden_size, (const void*)input, (void*)output);
    return 0;
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
        if (rec && desc.rows >= hidden_size && desc.cols == 0) {
            moe_raw_tensor_cache_entry raw{};
            if (!map_common_tensor_record(engine, *rec, desc, &raw)) {
                return 0;
            }
            moe_profile_log(
                "final_norm_contract_binding",
                "name=%s rows=%u cols=%u hidden=%u source=tensor_binding",
                graph_name.c_str(), desc.rows, desc.cols, hidden_size);
            return moe_apply_rmsnorm_common_weight_f32(engine, raw, input, hidden_size, output, eps);
        }
        moe_profile_log(
            "final_norm_contract_binding_missing",
            "name=%s rows=%u cols=%u hidden=%u",
            graph_name.c_str(), desc.rows, desc.cols, hidden_size);
    }
    {
        moe_raw_tensor_desc desc{};
        const moe_gguf_common_tensor_record* rec =
            find_common_tensor_by_op_role(engine, "final_norm", -1, &desc);
        if (rec && desc.rows >= hidden_size && desc.cols == 0) {
            moe_raw_tensor_cache_entry raw{};
            if (!map_common_tensor_record(engine, *rec, desc, &raw)) {
                return 0;
            }
            moe_profile_log(
                "final_norm_contract_binding",
                "name=%s rows=%u cols=%u hidden=%u source=op_role",
                rec->name.c_str(), desc.rows, desc.cols, hidden_size);
            return moe_apply_rmsnorm_common_weight_f32(engine, raw, input, hidden_size, output, eps);
        }
    }
    static const char* const suffixes[] = {
        "output_norm.weight",
        "model.norm.weight",
        "norm.weight",
        "transformer.ln_f.weight"
    };
    if (const moe_gguf_common_tensor_record* rec =
            find_common_tensor_by_names(engine, suffixes, sizeof(suffixes) / sizeof(suffixes[0]), nullptr)) {
        moe_raw_tensor_desc desc{};
        if (moe_common_tensor_record_to_raw_desc(*rec, &desc) &&
            desc.rows >= hidden_size && desc.cols == 0) {
            moe_raw_tensor_cache_entry raw{};
            if (!map_common_tensor_record(engine, *rec, desc, &raw)) {
                return 0;
            }
            moe_profile_log(
                "final_norm_contract_binding",
                "name=%s rows=%u cols=%u hidden=%u source=global_name",
                rec->name.c_str(), desc.rows, desc.cols, hidden_size);
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
    static const char* const graph_roles[] = {
        "post_ffw_norm", "post_ffw_norm_1", "post_ffw_norm_2", "layer_output_scale"
    };
    const int has_role = moe_graph_ir_math_required(engine) &&
        moe_graph_ir_layer_has_op_role_any(
            engine, layer, graph_roles, sizeof(graph_roles) / sizeof(graph_roles[0]));
    const int has_contract_table = moe_graph_ir_math_required(engine) && (
        moe_graph_ir_layer_contract_declares_tensor_role(engine, layer, "norms", "post_ffw_norm") ||
        moe_graph_ir_layer_contract_declares_tensor_role(engine, layer, "norms", "post_ffw_norm_1") ||
        moe_graph_ir_layer_contract_declares_tensor_role(engine, layer, "norms", "post_ffw_norm_2") ||
        moe_graph_ir_layer_contract_declares_tensor_role(engine, layer, "tail", "layer_output_scale"));
    const int has_layer_suffix =
        moe_engine_has_common_tensor_suffix(engine, layer, "post_ffw_norm.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "ffn_post_norm.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "post_ffw_norm_1.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "ffn_post_norm_1.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "post_ffw_norm_2.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "ffn_post_norm_2.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "layer_output_scale.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "layer_scalar.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "layer_scalar");
    int result = has_role || has_contract_table || has_layer_suffix;
    // Do not let a global GraphIR substring enable split-post-norm for every layer.
    // Per-layer behavior must come from a layer-local tensor/op binding or suffix.
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

static int moe_graph_ir_op_name_is_any_f32(
    const std::string& lowered,
    const char* const* needles,
    size_t needle_count
) {
    if (lowered.empty() || !needles) {
        return 0;
    }
    for (size_t i = 0; i < needle_count; ++i) {
        if (needles[i] && needles[i][0] && lowered.find(needles[i]) != std::string::npos) {
            return 1;
        }
    }
    return 0;
}


static int moe_graph_ir_layer_contract_bool_key_f32(
    const moe_pc_engine* engine,
    int32_t layer,
    const char* section,
    const char* key,
    int* out,
    const char** out_source
) {
    if (out) {
        *out = 0;
    }
    if (out_source) {
        *out_source = "none";
    }
    if (!engine || layer < 0 || !key || !key[0] || !out) {
        return 0;
    }
    const std::string* docs[] = {
        &engine->offload_graph_ir_json,
        &engine->offload_gguf_metadata_json,
    };
    const char* doc_names[] = { "graph_ir", "metadata" };
    for (size_t doc_i = 0; doc_i < sizeof(docs) / sizeof(docs[0]); ++doc_i) {
        const std::string* docp = docs[doc_i];
        if (!docp || docp->empty()) {
            continue;
        }
        const std::string& doc = *docp;
        size_t arr_begin = 0;
        size_t arr_end = 0;
        if (!moe_json_find_array_local(doc, "layer_execution_contract_table", &arr_begin, &arr_end)) {
            continue;
        }
        size_t cursor = arr_begin + 1u;
        uint64_t row_ordinal = 0;
        while (cursor < arr_end) {
            size_t obj_begin = 0;
            size_t obj_end = 0;
            if (!moe_json_next_object_in_array(doc, &cursor, arr_end, &obj_begin, &obj_end)) {
                break;
            }
            const uint64_t this_row_ordinal = row_ordinal++;
            if (!moe_json_slice_matches_layer_index_f32(
                    doc, obj_begin, obj_end, layer, this_row_ordinal)) {
                continue;
            }
            int value = 0;
            if (section && section[0]) {
                std::string row = doc.substr(obj_begin, obj_end - obj_begin);
                size_t section_begin = 0;
                size_t section_end = 0;
                if (moe_json_find_object_local(row, section, &section_begin, &section_end) &&
                    moe_json_get_bool_slice(row, section_begin, section_end, key, &value)) {
                    *out = value ? 1 : 0;
                    if (out_source) {
                        *out_source = doc_names[doc_i];
                    }
                    return 1;
                }
            }
            if (moe_json_get_bool_slice(doc, obj_begin, obj_end, key, &value)) {
                *out = value ? 1 : 0;
                if (out_source) {
                    *out_source = doc_names[doc_i];
                }
                return 1;
            }
            break;
        }
    }
    return 0;
}

static int moe_graph_ir_attention_residual_before_ffn_norm(const moe_pc_engine* engine, int32_t layer) {
    if (!engine || layer < 0) {
        return 0;
    }
    const std::string* docs[] = {
        &engine->offload_graph_ir_json,
        &engine->offload_gguf_metadata_json,
    };
    const char* doc_names[] = {
        "graph_ir",
        "metadata",
    };
    for (size_t doc_i = 0; doc_i < sizeof(docs) / sizeof(docs[0]); ++doc_i) {
        const std::string* docp = docs[doc_i];
        if (!docp || docp->empty()) {
            continue;
        }
        const std::string& doc = *docp;
        size_t arr_begin = 0;
        size_t arr_end = 0;
        if (moe_json_find_array_local(doc, "layer_execution_contract_table", &arr_begin, &arr_end)) {
            size_t cursor = arr_begin + 1u;
            uint64_t row_ordinal = 0;
            while (cursor < arr_end) {
                size_t obj_begin = 0;
                size_t obj_end = 0;
                if (!moe_json_next_object_in_array(doc, &cursor, arr_end, &obj_begin, &obj_end)) {
                    break;
                }
                const uint64_t this_row_ordinal = row_ordinal++;
                if (!moe_json_slice_matches_layer_index_f32(
                        doc, obj_begin, obj_end, layer, this_row_ordinal)) {
                    continue;
                }
                int flag = 0;
                static const char* const bool_keys[] = {
                    "attention_residual_before_ffn_norm",
                    "post_attention_residual_before_norm",
                    "residual_before_post_attention_norm",
                    "post_attention_norm_after_residual",
                    "attention_norm_after_residual",
                    "graph_residual_first"
                };
                for (const char* key : bool_keys) {
                    if (moe_json_get_bool_slice(doc, obj_begin, obj_end, key, &flag)) {
                        moe_profile_log(
                            "graphir_attention_residual_order",
                            "layer=%d residual_first=%d source=%s.contract key=%s",
                            layer, flag ? 1 : 0, doc_names[doc_i], key);
                        return flag ? 1 : 0;
                    }
                }
                std::string order;
                if ((moe_json_get_string_slice(doc, obj_begin, obj_end, "post_attention_norm_input", &order) ||
                     moe_json_get_string_slice(doc, obj_begin, obj_end, "attention_norm_input", &order) ||
                     moe_json_get_string_slice(doc, obj_begin, obj_end, "post_attention_order", &order))) {
                    const std::string lowered = moe_lower_ascii_copy(order);
                    if (lowered.find("residual") != std::string::npos ||
                        lowered.find("attn_plus_hidden") != std::string::npos ||
                        lowered.find("hidden_plus_attn") != std::string::npos) {
                        moe_profile_log(
                            "graphir_attention_residual_order",
                            "layer=%d residual_first=1 source=%s.contract_string value=%s",
                            layer, doc_names[doc_i], order.c_str());
                        return 1;
                    }
                    if (lowered.find("attention_output") != std::string::npos ||
                        lowered.find("attn_output") != std::string::npos) {
                        moe_profile_log(
                            "graphir_attention_residual_order",
                            "layer=%d residual_first=0 source=%s.contract_string value=%s",
                            layer, doc_names[doc_i], order.c_str());
                        return 0;
                    }
                }
                break;
            }
        }

        size_t layers_begin = 0;
        size_t layers_end = 0;
        if (!moe_json_find_graph_layers_array_f32(doc, &layers_begin, &layers_end)) {
            continue;
        }
        size_t cursor = layers_begin + 1u;
        uint64_t layer_ordinal = 0;
        while (cursor < layers_end) {
            size_t layer_obj_begin = 0;
            size_t layer_obj_end = 0;
            if (!moe_json_next_object_in_array(doc, &cursor, layers_end, &layer_obj_begin, &layer_obj_end)) {
                break;
            }
            const uint64_t this_layer_ordinal = layer_ordinal++;
            if (!moe_json_slice_matches_layer_index_f32(
                    doc, layer_obj_begin, layer_obj_end, layer, this_layer_ordinal)) {
                continue;
            }
            const std::string layer_slice = doc.substr(layer_obj_begin, layer_obj_end - layer_obj_begin);
            size_t ops_begin = 0;
            size_t ops_end = 0;
            if (!moe_json_find_array_local(layer_slice, "ops", &ops_begin, &ops_end)) {
                break;
            }
            ops_begin += layer_obj_begin;
            ops_end += layer_obj_begin;
            size_t op_cursor = ops_begin + 1u;
            int32_t residual_index = -1;
            int32_t post_norm_index = -1;
            int32_t index = 0;
            static const char* const residual_needles[] = {
                "residual_add", "attention_residual", "attn_residual",
                "add_residual", "hidden_plus_attn", "attn_plus_hidden"
            };
            static const char* const post_norm_needles[] = {
                "post_attention_norm", "post_attn_norm", "attention_output_norm",
                "attn_output_norm"
            };
            while (op_cursor < ops_end) {
                size_t op_begin = 0;
                size_t op_end = 0;
                if (!moe_json_next_object_in_array(doc, &op_cursor, ops_end, &op_begin, &op_end)) {
                    break;
                }
                std::string text;
                std::string merged;
                static const char* const fields[] = {
                    "name", "execution_op", "op_role", "role", "graph_role", "bundle_member_role", "type"
                };
                for (const char* field : fields) {
                    if (moe_json_get_string_slice(doc, op_begin, op_end, field, &text)) {
                        merged += " ";
                        merged += moe_lower_ascii_copy(text);
                    }
                }
                if (residual_index < 0 &&
                    moe_graph_ir_op_name_is_any_f32(merged, residual_needles, sizeof(residual_needles) / sizeof(residual_needles[0]))) {
                    residual_index = index;
                }
                if (post_norm_index < 0 &&
                    moe_graph_ir_op_name_is_any_f32(merged, post_norm_needles, sizeof(post_norm_needles) / sizeof(post_norm_needles[0]))) {
                    post_norm_index = index;
                }
                ++index;
            }
            if (residual_index >= 0 && post_norm_index >= 0) {
                const int residual_first = residual_index < post_norm_index ? 1 : 0;
                moe_profile_log(
                    "graphir_attention_residual_order",
                    "layer=%d residual_first=%d source=%s.layers.ops residual_index=%d post_norm_index=%d",
                    layer, residual_first, doc_names[doc_i], residual_index, post_norm_index);
                return residual_first;
            }
            break;
        }
    }
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
    static const char* const sections[] = {
        nullptr, "attention", "norms", "router", "mlp", "tail", "ple"
    };
    moe_raw_tensor_desc desc{};
    for (size_t i = 0; i < role_count; ++i) {
        if (!roles[i] || !roles[i][0]) {
            continue;
        }
        if (find_common_tensor_by_op_role(engine, roles[i], layer, &desc)) {
            return 1;
        }
        for (const char* section : sections) {
            if (moe_graph_ir_layer_contract_declares_tensor_role(engine, layer, section, roles[i])) {
                return 1;
            }
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
    static const char* const sections[] = {
        nullptr, "attention", "norms", "router", "mlp", "tail", "ple"
    };
    for (size_t i = 0; i < role_count; ++i) {
        if (!roles[i] || !roles[i][0]) {
            continue;
        }
        moe_raw_tensor_desc desc{};
        const moe_gguf_common_tensor_record* rec =
            find_common_tensor_by_op_role(engine, roles[i], layer, &desc);
        if (rec) {
            if (!map_common_tensor_record(engine, *rec, desc, out)) {
                return 0;
            }
            moe_profile_log(
                "graphir_contract_tensor_bind",
                "layer=%d role=%s name=%s source=op_role rows=%u cols=%u bytes=%llu",
                layer, roles[i], rec->name.c_str(), desc.rows, desc.cols,
                (unsigned long long)rec->bytes);
            moe_profile_log(
                "rmsnorm_common_map",
                "layer=%d suffix0=%s mapped=1 name=%s rows=%u cols=%u bytes=%llu source=op_role",
                layer, roles[i], rec->name.c_str(), desc.rows, desc.cols,
                (unsigned long long)rec->bytes);
            return 1;
        }
        for (const char* section : sections) {
            if (moe_graph_ir_map_layer_contract_tensor_role_f32(engine, layer, section, roles[i], out)) {
                return 1;
            }
        }
    }
    return 0;
}

static int moe_graph_ir_string_field_matches_role_f32(
    const std::string& doc,
    size_t obj_begin,
    size_t obj_end,
    const char* field,
    const char* role_key
) {
    if (!field || !role_key || !role_key[0]) {
        return 0;
    }
    std::string value;
    if (!moe_json_get_string_slice(doc, obj_begin, obj_end, field, &value)) {
        return 0;
    }
    return moe_lower_ascii_copy(value) == moe_lower_ascii_copy(role_key) ? 1 : 0;
}

static int moe_graph_ir_layer_op_matches_tensor_role_f32(
    const std::string& doc,
    size_t obj_begin,
    size_t obj_end,
    const char* role_key
) {
    static const char* const fields[] = {
        "name", "execution_op", "op_role", "role", "graph_role", "bundle_member_role"
    };
    for (const char* field : fields) {
        if (moe_graph_ir_string_field_matches_role_f32(doc, obj_begin, obj_end, field, role_key)) {
            return 1;
        }
    }
    return 0;
}


static void moe_graph_ir_push_unique_names_f32(
    std::vector<std::string>* out_names,
    const std::vector<std::string>& names
) {
    if (!out_names) {
        return;
    }
    for (const std::string& name : names) {
        if (name.empty()) {
            continue;
        }
        int exists = 0;
        for (const std::string& cur : *out_names) {
            if (cur == name) {
                exists = 1;
                break;
            }
        }
        if (!exists) {
            out_names->push_back(name);
        }
    }
}

static size_t moe_json_find_next_contract_array_f32(
    const std::string& text,
    const char* key,
    size_t search_pos,
    size_t* begin,
    size_t* end
) {
    if (begin) *begin = 0;
    if (end) *end = 0;
    if (!key || !key[0] || !begin || !end || search_pos >= text.size()) {
        return std::string::npos;
    }
    const size_t key_len = std::strlen(key);
    size_t pos = search_pos;
    while ((pos = text.find('"', pos)) != std::string::npos) {
        if (pos + 1u + key_len < text.size() &&
            text.compare(pos + 1u, key_len, key) == 0 &&
            text[pos + 1u + key_len] == '"') {
            const size_t colon = text.find_first_not_of(" \t\r\n", pos + key_len + 2u);
            if (colon != std::string::npos && colon < text.size() && text[colon] == ':') {
                const size_t value = text.find_first_not_of(" \t\r\n", colon + 1u);
                if (value != std::string::npos && value < text.size() && text[value] == '[') {
                    const size_t close = moe_json_match_bracket_local(text, value, '[', ']');
                    if (close != std::string::npos) {
                        *begin = value;
                        *end = close;
                        return pos;
                    }
                }
            }
        }
        ++pos;
    }
    return std::string::npos;
}

static int moe_graph_ir_layer_ops_tensor_names_for_role(
    const moe_pc_engine* engine,
    int32_t layer,
    const char* role_key,
    std::vector<std::string>* out_names
) {
    if (!engine || layer < 0 || !role_key || !role_key[0] || !out_names) {
        return 0;
    }
    const size_t start_count = out_names->size();
    const std::string* docs[] = {
        &engine->offload_graph_ir_json,
        &engine->offload_gguf_metadata_json,
    };
    const char* doc_names[] = {
        "graph_ir",
        "metadata",
    };
    for (size_t doc_i = 0; doc_i < sizeof(docs) / sizeof(docs[0]); ++doc_i) {
        const std::string* docp = docs[doc_i];
        if (!docp || docp->empty()) {
            continue;
        }
        const std::string& doc = *docp;
        size_t layers_begin = 0;
        size_t layers_end = 0;
        if (!moe_json_find_graph_layers_array_f32(doc, &layers_begin, &layers_end)) {
            continue;
        }
        size_t cursor = layers_begin + 1u;
        uint64_t layer_ordinal = 0;
        while (cursor < layers_end) {
            size_t layer_obj_begin = 0;
            size_t layer_obj_end = 0;
            if (!moe_json_next_object_in_array(doc, &cursor, layers_end, &layer_obj_begin, &layer_obj_end)) {
                break;
            }
            const uint64_t this_layer_ordinal = layer_ordinal++;
            if (!moe_json_slice_matches_layer_index_f32(
                    doc, layer_obj_begin, layer_obj_end, layer, this_layer_ordinal)) {
                continue;
            }
            size_t ops_begin = 0;
            size_t ops_end = 0;
            const std::string layer_slice = doc.substr(layer_obj_begin, layer_obj_end - layer_obj_begin);
            if (!moe_json_find_array_local(layer_slice, "ops", &ops_begin, &ops_end)) {
                continue;
            }
            ops_begin += layer_obj_begin;
            ops_end += layer_obj_begin;
            size_t op_cursor = ops_begin + 1u;
            while (op_cursor < ops_end) {
                size_t op_begin = 0;
                size_t op_end = 0;
                if (!moe_json_next_object_in_array(doc, &op_cursor, ops_end, &op_begin, &op_end)) {
                    break;
                }
                if (!moe_graph_ir_layer_op_matches_tensor_role_f32(doc, op_begin, op_end, role_key)) {
                    continue;
                }
                std::vector<std::string> names;
                if (moe_json_get_string_array_slice(doc, op_begin, op_end, "weights", &names) && !names.empty()) {
                    const size_t before = out_names->size();
                    moe_graph_ir_push_unique_names_f32(out_names, names);
                    if (out_names->size() > before) {
                        moe_profile_log(
                            "graphir_contract_tensor_names",
                            "layer=%d role=%s count=%zu source=%s.layers.ops.weights",
                            layer, role_key, out_names->size() - before, doc_names[doc_i]);
                    }
                }
            }
        }
    }
    return out_names->size() > start_count ? 1 : 0;
}

static int moe_graph_ir_layer_contract_tensor_names_for_role(
    const moe_pc_engine* engine,
    int32_t layer,
    const char* section_key,
    const char* role_key,
    std::vector<std::string>* out_names
) {
    if (out_names) {
        out_names->clear();
    }
    if (!engine || layer < 0 || !role_key || !role_key[0] || !out_names) {
        return 0;
    }
    const size_t start_count = out_names->size();
    const std::string* docs[] = {
        &engine->offload_graph_ir_json,
        &engine->offload_gguf_metadata_json,
    };
    const char* doc_names[] = {
        "graph_ir",
        "metadata",
    };
    for (size_t doc_i = 0; doc_i < sizeof(docs) / sizeof(docs[0]); ++doc_i) {
        const std::string* docp = docs[doc_i];
        if (!docp || docp->empty()) {
            continue;
        }
        const std::string& doc = *docp;
        size_t search_pos = 0;
        size_t arr_begin = 0;
        size_t arr_end = 0;
        while (moe_json_find_next_contract_array_f32(
            doc, "layer_execution_contract_table", search_pos, &arr_begin, &arr_end) != std::string::npos) {
            size_t cursor = arr_begin + 1u;
            uint64_t row_ordinal = 0;
            while (cursor < arr_end) {
                size_t obj_begin = 0;
                size_t obj_end = 0;
                if (!moe_json_next_object_in_array(doc, &cursor, arr_end, &obj_begin, &obj_end)) {
                    break;
                }
                const uint64_t this_row_ordinal = row_ordinal++;
                if (!moe_json_slice_matches_layer_index_f32(
                        doc, obj_begin, obj_end, layer, this_row_ordinal)) {
                    continue;
                }
                size_t scope_begin = obj_begin;
                size_t scope_end = obj_end;
                int section_found = 0;
                if (section_key && section_key[0]) {
                    const std::string row = doc.substr(obj_begin, obj_end - obj_begin);
                    size_t local_scope_begin = 0;
                    size_t local_scope_end = 0;
                    if (moe_json_find_object_local(row, section_key, &local_scope_begin, &local_scope_end)) {
                        scope_begin = obj_begin + local_scope_begin;
                        scope_end = obj_begin + local_scope_end;
                        section_found = 1;
                    }
                }
                const size_t before = out_names->size();
                std::vector<std::string> names;
                if ((!section_key || !section_key[0] || section_found) &&
                    moe_json_get_string_array_slice(doc, scope_begin, scope_end, role_key, &names) &&
                    !names.empty()) {
                    moe_graph_ir_push_unique_names_f32(out_names, names);
                }
                std::string single_name;
                if ((!section_key || !section_key[0] || section_found) &&
                    moe_json_get_string_slice(doc, scope_begin, scope_end, role_key, &single_name) &&
                    !single_name.empty()) {
                    moe_graph_ir_push_unique_names_f32(out_names, std::vector<std::string>{single_name});
                }
                // Generator tables may place a role at row top-level even when
                // the semantic section exists.  Treat the row as a fallback only
                // after the requested section was checked.
                if (section_key && section_key[0]) {
                    names.clear();
                    if (moe_json_get_string_array_slice(doc, obj_begin, obj_end, role_key, &names) &&
                        !names.empty()) {
                        moe_graph_ir_push_unique_names_f32(out_names, names);
                    }
                    single_name.clear();
                    if (moe_json_get_string_slice(doc, obj_begin, obj_end, role_key, &single_name) &&
                        !single_name.empty()) {
                        moe_graph_ir_push_unique_names_f32(out_names, std::vector<std::string>{single_name});
                    }
                }
                if (out_names->size() > before) {
                    moe_profile_log(
                        "graphir_contract_tensor_names",
                        "layer=%d role=%s count=%zu source=%s.layer_execution_contract_table.%s",
                        layer, role_key, out_names->size() - before, doc_names[doc_i],
                        section_key && section_key[0] ? section_key : "row");
                }
            }
            search_pos = arr_end;
        }
    }
    if (out_names->size() > start_count) {
        return 1;
    }
    return moe_graph_ir_layer_ops_tensor_names_for_role(engine, layer, role_key, out_names);
}

static int moe_graph_ir_layer_contract_declares_tensor_role(
    const moe_pc_engine* engine,
    int32_t layer,
    const char* section_key,
    const char* role_key
) {
    std::vector<std::string> names;
    return moe_graph_ir_layer_contract_tensor_names_for_role(
        engine, layer, section_key, role_key, &names) && !names.empty();
}

static int moe_graph_ir_map_layer_contract_tensor_role_f32(
    moe_pc_engine* engine,
    int32_t layer,
    const char* section_key,
    const char* role_key,
    moe_raw_tensor_cache_entry* out
) {
    if (!engine || !out) {
        return 0;
    }
    std::vector<std::string> names;
    if (!moe_graph_ir_layer_contract_tensor_names_for_role(
            engine, layer, section_key, role_key, &names)) {
        return 0;
    }
    for (const std::string& name : names) {
        if (name.empty()) {
            continue;
        }
        moe_raw_tensor_desc desc{};
        const moe_gguf_common_tensor_record* rec = moe_common_tensor_find(engine, name.c_str());
        if (!rec || !moe_common_tensor_record_to_raw_desc(*rec, &desc) ||
            !moe_common_tensor_record_raw_compatible(*rec, desc)) {
            continue;
        }
        if (!map_common_tensor_record(engine, *rec, desc, out)) {
            return 0;
        }
        moe_profile_log(
            "graphir_contract_tensor_bind",
            "layer=%d role=%s name=%s source=layer_execution_contract_table.%s",
            layer,
            role_key,
            name.c_str(),
            section_key && section_key[0] ? section_key : "row");
        moe_profile_log(
            "rmsnorm_common_map",
            "layer=%d suffix0=%s mapped=1 name=%s rows=%u cols=%u bytes=%llu source=contract",
            layer, role_key, name.c_str(), desc.rows, desc.cols,
            (unsigned long long)rec->bytes);
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
    const int ok = moe_apply_rmsnorm_common_weight_f32(engine, raw, input, hidden_size, output, eps);
    moe_profile_log(
        "rmsnorm_common_apply",
        "layer=%d suffix0=%s ok=%d hidden=%u source=op_role",
        layer, (role_count && roles && roles[0]) ? roles[0] : "", ok, hidden_size);
    return ok;
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
    if (moe_graph_ir_math_required(engine) &&
        moe_graph_ir_apply_rmsnorm_role_any_f32(
            engine, layer, roles, sizeof(roles) / sizeof(roles[0]),
            input, hidden_size, output, eps)) {
        return 1;
    }
    if (moe_graph_ir_math_required(engine)) {
        moe_raw_tensor_cache_entry raw{};
        if (moe_graph_ir_map_layer_contract_tensor_role_f32(
                engine, layer, "norms", "post_ffw_norm", &raw)) {
            const int ok = moe_apply_rmsnorm_common_weight_f32(engine, raw, input, hidden_size, output, eps);
            moe_profile_log(
                "rmsnorm_common_apply",
                "layer=%d suffix0=post_ffw_norm ok=%d hidden=%u source=contract",
                layer, ok, hidden_size);
            return ok;
        }
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
    if (moe_graph_ir_math_required(engine) &&
        moe_graph_ir_layer_has_op_role_any(
            engine, layer, roles, sizeof(roles) / sizeof(roles[0]))) {
        return 1;
    }
    if (moe_graph_ir_math_required(engine) &&
        moe_graph_ir_layer_contract_declares_tensor_role(engine, layer, "norms", "post_ffw_norm")) {
        return 1;
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
    if (moe_graph_ir_math_required(engine) &&
        moe_graph_ir_apply_rmsnorm_role_any_f32(
            engine, layer, roles, sizeof(roles) / sizeof(roles[0]),
            input, hidden_size, output, eps)) {
        return 1;
    }
    if (moe_graph_ir_math_required(engine)) {
        moe_raw_tensor_cache_entry raw{};
        if (moe_graph_ir_map_layer_contract_tensor_role_f32(
                engine, layer, "norms", "post_ffw_norm_1", &raw)) {
            const int ok = moe_apply_rmsnorm_common_weight_f32(engine, raw, input, hidden_size, output, eps);
            moe_profile_log(
                "rmsnorm_common_apply",
                "layer=%d suffix0=post_ffw_norm_1 ok=%d hidden=%u source=contract",
                layer, ok, hidden_size);
            return ok;
        }
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
    if (moe_graph_ir_math_required(engine) &&
        moe_graph_ir_layer_has_op_role_any(
            engine, layer, roles, sizeof(roles) / sizeof(roles[0]))) {
        return 1;
    }
    if (moe_graph_ir_math_required(engine) &&
        moe_graph_ir_layer_contract_declares_tensor_role(engine, layer, "norms", "post_ffw_norm_1")) {
        return 1;
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
    if (moe_graph_ir_math_required(engine) &&
        moe_graph_ir_apply_rmsnorm_role_any_f32(
            engine, layer, roles, sizeof(roles) / sizeof(roles[0]),
            input, hidden_size, output, eps)) {
        return 1;
    }
    if (moe_graph_ir_math_required(engine)) {
        moe_raw_tensor_cache_entry raw{};
        if (moe_graph_ir_map_layer_contract_tensor_role_f32(
                engine, layer, "norms", "post_ffw_norm_2", &raw)) {
            const int ok = moe_apply_rmsnorm_common_weight_f32(engine, raw, input, hidden_size, output, eps);
            moe_profile_log(
                "rmsnorm_common_apply",
                "layer=%d suffix0=post_ffw_norm_2 ok=%d hidden=%u source=contract",
                layer, ok, hidden_size);
            return ok;
        }
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
    if (moe_graph_ir_math_required(engine) &&
        moe_graph_ir_layer_has_op_role_any(
            engine, layer, roles, sizeof(roles) / sizeof(roles[0]))) {
        return 1;
    }
    if (moe_graph_ir_math_required(engine) &&
        moe_graph_ir_layer_contract_declares_tensor_role(engine, layer, "norms", "post_ffw_norm_2")) {
        return 1;
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
    if (!engine || !hidden || hidden_size == 0 || hidden_size > 100000) {
        return 0;
    }
    moe_raw_tensor_cache_entry raw{};
    int mapped = 0;
    if (moe_graph_ir_math_required(engine)) {
        static const char* const roles[] = { "layer_output_scale" };
        mapped = moe_graph_ir_map_layer_op_role_any(
            engine, layer, roles, sizeof(roles) / sizeof(roles[0]), &raw);
        if (!mapped) {
            mapped = moe_graph_ir_map_layer_contract_tensor_role_f32(
                engine, layer, "tail", "layer_output_scale", &raw);
        }
    }
    if (!mapped) {
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
        if (!moe_raw_tensor_scalar_at_f32(raw, 0, &scale) || !std::isfinite(scale)) {
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
        if (!moe_raw_tensor_scalar_at_f32(raw, i, &scale) || !std::isfinite(scale)) {
            return 0;
        }
        hidden[i] *= scale;
    }
    return 1;
}
