// Raw forward - Utility functions

static float moe_raw_bf16_to_f32(uint16_t v) {
    const uint32_t bits = (uint32_t)v << 16u;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

static uint32_t moe_f32_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float moe_f32_from_bits(uint32_t bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

static size_t moe_ascii_find_ci(const std::string& haystack, const char* needle, size_t begin = 0) {
    if (!needle || !needle[0]) {
        return std::string::npos;
    }
    const size_t n = std::strlen(needle);
    if (n == 0 || begin >= haystack.size() || haystack.size() < n) {
        return std::string::npos;
    }
    for (size_t i = begin; i + n <= haystack.size(); ++i) {
        size_t j = 0;
        for (; j < n; ++j) {
            const unsigned char a = static_cast<unsigned char>(haystack[i + j]);
            const unsigned char b = static_cast<unsigned char>(needle[j]);
            if (std::tolower(a) != std::tolower(b)) {
                break;
            }
        }
        if (j == n) {
            return i;
        }
    }
    return std::string::npos;
}

static int moe_ascii_contains_ci(const std::string& haystack, const char* needle) {
    if (!needle || !needle[0]) {
        return 0;
    }
    if (moe_ascii_find_ci(haystack, needle) != std::string::npos) {
        return 1;
    }
    return 0;
}

static int moe_graph_ir_tensor_binding_name(
    const moe_pc_engine* engine,
    const char* key,
    std::string* out
) {
    if (out) {
        out->clear();
    }
    if (!engine || !key || !key[0] || !out || engine->offload_graph_ir_json.empty()) {
        return 0;
    }
    size_t begin = 0;
    size_t end = 0;
    if (moe_json_find_object_local(engine->offload_graph_ir_json, "tensor_bindings", &begin, &end) &&
        moe_json_get_string_slice(engine->offload_graph_ir_json, begin, end, key, out) &&
        !out->empty()) {
        return 1;
    }
    return moe_json_get_string_local(engine->offload_graph_ir_json, key, out) && !out->empty();
}

static int moe_graph_ir_lm_head_tied_to_token_embedding(const moe_pc_engine* engine) {
    if (!engine || engine->offload_graph_ir_json.empty()) {
        return 0;
    }
    int tied = 0;
    size_t begin = 0;
    size_t end = 0;
    if (moe_json_find_object_local(engine->offload_graph_ir_json, "tensor_bindings", &begin, &end) &&
        moe_json_get_bool_slice(engine->offload_graph_ir_json, begin, end, "lm_head_tied_to_token_embedding", &tied) &&
        tied) {
        return 1;
    }
    if (moe_json_get_bool_local(engine->offload_graph_ir_json, "lm_head_tied_to_token_embedding", &tied) && tied) {
        return 1;
    }
    std::string lm_head;
    std::string token_embedding;
    if (moe_graph_ir_tensor_binding_name(engine, "lm_head", &lm_head)) {
        if (moe_graph_ir_tensor_binding_name(engine, "token_embedding", &token_embedding) &&
            !token_embedding.empty() &&
            lm_head == token_embedding) {
            return 1;
        }
        if (lm_head == "token_embd.weight" ||
            lm_head == "model.embed_tokens.weight" ||
            lm_head == "embed_tokens.weight") {
            return 1;
        }
    }
    return 0;
}

static uint32_t moe_raw_scalar_size(uint8_t encoding) {
    if (encoding == moe_RAW_SCALAR_F32) {
        return 4u;
    }
    if (encoding == moe_RAW_SCALAR_F16 || encoding == moe_RAW_SCALAR_BF16) {
        return 2u;
    }
    return 0u;
}

static float moe_raw_scalar_to_f32(const uint8_t* ptr, uint8_t encoding) {
    if (!ptr) {
        return 0.0f;
    }
    if (encoding == moe_RAW_SCALAR_F32) {
        float v = 0.0f;
        std::memcpy(&v, ptr, sizeof(v));
        return v;
    }
    uint16_t v = 0;
    std::memcpy(&v, ptr, sizeof(v));
    if (encoding == moe_RAW_SCALAR_F16) {
        return moe_raw_fp16_to_f32(v);
    }
    if (encoding == moe_RAW_SCALAR_BF16) {
        return moe_raw_bf16_to_f32(v);
    }
    return 0.0f;
}

static void moe_trace_raw_dot_unsupported(const moe_raw_tensor_cache_entry& raw) {
    static std::atomic<uint32_t> logged{0};
    const uint32_t n = logged.fetch_add(1, std::memory_order_relaxed);
    if (n >= 32u) {
        return;
    }
    moe_profile_log(
        "raw_dot_unsupported",
        "format=%u scalar=%u rows=%u cols=%u bytes=%llu row_bytes=%llu",
        (uint32_t)raw.weight_format,
        (uint32_t)raw.scalar_encoding,
        raw.rows,
        raw.cols,
        (unsigned long long)raw.bytes,
        (unsigned long long)raw.weight_row_bytes);
}

static const moe_storage_raw_span_spec_t* find_raw_role_span(int32_t layer, uint32_t role) {
    const uint32_t begin = moe_storage_raw_span_find_first(layer);
    const uint32_t end = moe_storage_raw_span_find_end(layer);
    for (uint32_t i = begin; i < end; ++i) {
        const moe_storage_raw_span_spec_t* s = moe_storage_raw_span_at(i);
        if (s && (s->role & role) != 0 && s->bytes > 0) {
            return s;
        }
    }
    return nullptr;
}

static mmap_context_t* map_raw_span_part(moe_pc_engine* engine, const moe_storage_raw_span_spec_t* span) {
    if (!engine || !span || engine->model_root.empty()) {
        return nullptr;
    }
    const moe_storage_part_spec_t* part = moe_storage_part_by_id(span->part);
    return part && part->path ? get_or_map_shard_partial_by_id(
        engine,
        span->part,
        resolve_model_file_path(engine->model_root, part->path, ""),
        span->offset,
        span->bytes
    ) : nullptr;
}

static int raw_span_bounds_ok(const moe_storage_raw_span_spec_t* span, const mmap_context_t* mapped) {
    return span && mapped && mmap_get_ptr(mapped, 0) &&
        span->bytes <= (uint64_t)mmap_get_size(mapped);
}

static const uint8_t* raw_span_ptr(
    const moe_storage_raw_span_spec_t* span,
    const mmap_context_t* mapped,
    uint64_t rel_offset,
    uint64_t bytes
) {
    if (!raw_span_bounds_ok(span, mapped)) {
        return nullptr;
    }
    const uint64_t mapped_size = (uint64_t)mmap_get_size(mapped);
    // BUGFIX 495: rel_offset + bytes overflow 泥댄겕
    if (rel_offset > mapped_size || bytes > mapped_size - rel_offset ||
        rel_offset > (uint64_t)SIZE_MAX ||
        (bytes > 0 && rel_offset > UINT64_MAX - bytes)) {
        return nullptr;
    }
    return static_cast<const uint8_t*>(mmap_get_ptr(mapped, (size_t)rel_offset));
}

static const moe_storage_raw_span_spec_t* find_raw_role_span_bytes(
    int32_t layer,
    uint32_t role,
    uint64_t bytes,
    uint32_t occurrence
) {
    const uint32_t begin = moe_storage_raw_span_find_first(layer);
    const uint32_t end = moe_storage_raw_span_find_end(layer);
    uint32_t seen = 0;
    for (uint32_t i = begin; i < end; ++i) {
        const moe_storage_raw_span_spec_t* s = moe_storage_raw_span_at(i);
        if (s && (s->role & role) != 0 && s->bytes == bytes) {
            if (seen == occurrence) {
                return s;
            }
            ++seen;
        }
    }
    return nullptr;
}

static float moe_dot_bf16_row(const uint8_t* row, const float* hidden, uint32_t hidden_count) {
    // BUGFIX 712: BF16 dot product NaN handling ★★
    // Problem: No validation of row or hidden input, no NaN check on result
    // Impact: NaN in weights or activations propagates to PPL
    // Solution: Check inputs and final result for NaN/Inf
    if (!row || !hidden) {
        return 0.0f;
    }
    uint32_t i = 0;
    float acc = 0.0f;
#if defined(__AVX2__)
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    for (; i + 15 < hidden_count; i += 16) {
        __m128i h0 = _mm_loadu_si128((const __m128i*)(row + (uint64_t)i * 2ull));
        __m128i h1 = _mm_loadu_si128((const __m128i*)(row + (uint64_t)(i + 8) * 2ull));
        __m256i exp0 = _mm256_cvtepu16_epi32(h0);
        __m256i exp1 = _mm256_cvtepu16_epi32(h1);
        __m256 f0 = _mm256_castsi256_ps(_mm256_slli_epi32(exp0, 16));
        __m256 f1 = _mm256_castsi256_ps(_mm256_slli_epi32(exp1, 16));
        __m256 x0 = _mm256_loadu_ps(hidden + i);
        __m256 x1 = _mm256_loadu_ps(hidden + i + 8);
        sum0 = moe_madd_ps(f0, x0, sum0);
        sum1 = moe_madd_ps(f1, x1, sum1);
    }
    sum0 = _mm256_add_ps(sum0, sum1);
    float buf[8];
    _mm256_storeu_ps(buf, sum0);
    acc += buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];
#endif
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (; i + 3u < hidden_count; i += 4u) {
        uint16_t v0, v1, v2, v3;
        std::memcpy(&v0, row + (uint64_t)(i + 0u) * 2ull, sizeof(v0));
        std::memcpy(&v1, row + (uint64_t)(i + 1u) * 2ull, sizeof(v1));
        std::memcpy(&v2, row + (uint64_t)(i + 2u) * 2ull, sizeof(v2));
        std::memcpy(&v3, row + (uint64_t)(i + 3u) * 2ull, sizeof(v3));
        a0 += moe_raw_bf16_to_f32(v0) * hidden[i + 0u];
        a1 += moe_raw_bf16_to_f32(v1) * hidden[i + 1u];
        a2 += moe_raw_bf16_to_f32(v2) * hidden[i + 2u];
        a3 += moe_raw_bf16_to_f32(v3) * hidden[i + 3u];
    }
    acc += (a0 + a1) + (a2 + a3);
    for (; i < hidden_count; ++i) {
        uint16_t v;
        std::memcpy(&v, row + (uint64_t)i * 2ull, sizeof(v));
        acc += moe_raw_bf16_to_f32(v) * hidden[i];
    }
    // BUGFIX 712: Final NaN/Inf check
    if (!std::isfinite(acc)) {
        return 0.0f;
    }
    return acc;
}

static float moe_dot_raw_row(
    const uint8_t* row,
    uint8_t encoding,
    const float* hidden,
    uint32_t hidden_count
) {
    // BUGFIX 529: Check hidden_count upper bound to prevent excessive iteration ??    // Problem: hidden_count can be arbitrarily large from config
    // Loop can run billions of iterations
    // Solution: Add reasonable upper bound check (e.g., 100000)
    if (!row || !hidden || hidden_count > 100000) {
        return 0.0f;
    }
    if (encoding == moe_RAW_SCALAR_BF16) {
        return moe_dot_bf16_row(row, hidden, hidden_count);
    }
    const uint32_t elem = moe_raw_scalar_size(encoding);
    if (!elem) {
        return 0.0f;
    }
    float acc = 0.0f;
    for (uint32_t i = 0; i < hidden_count; ++i) {
        acc += moe_raw_scalar_to_f32(row + (uint64_t)i * elem, encoding) * hidden[i];
    }
    return acc;
}

static int moe_engine_has_common_tensor_suffix(
    const moe_pc_engine* engine,
    int32_t layer,
    const char* suffix
) {
    if (!engine || !suffix || layer < 0) {
        return 0;
    }
    const std::string suffix_lower = moe_lower_ascii_copy(std::string(suffix));
    char p0[64], p1[64], p2[64], p3[64], p4[64];
    std::snprintf(p0, sizeof(p0), "blk.%d.", (int)layer);
    std::snprintf(p1, sizeof(p1), "model.layers.%d.", (int)layer);
    std::snprintf(p2, sizeof(p2), "layers.%d.", (int)layer);
    std::snprintf(p3, sizeof(p3), "transformer.h.%d.", (int)layer);
    std::snprintf(p4, sizeof(p4), "h.%d.", (int)layer);
    const char* prefixes[] = {p0, p1, p2, p3, p4};
    std::string cache_key = std::to_string(layer);
    cache_key.push_back('\n');
    cache_key.append(suffix_lower);
    {
        std::shared_lock<std::shared_mutex> lock(engine->common_tensor_suffix_cache_mutex);
        auto found = engine->common_tensor_suffix_cache.find(cache_key);
        if (found != engine->common_tensor_suffix_cache.end()) {
            return found->second;
        }
    }
    int result = 0;
    for (const auto& it : engine->gguf_common_tensor_slots) {
        const std::string& key = it.first;
        for (const char* prefix : prefixes) {
            const size_t prefix_len = std::strlen(prefix);
            if (key.size() >= prefix_len + suffix_lower.size() &&
                key.compare(0, prefix_len, prefix) == 0 &&
                key.compare(key.size() - suffix_lower.size(), suffix_lower.size(), suffix_lower) == 0) {
                result = 1;
                break;
            }
        }
        if (result) break;
    }
    {
        std::unique_lock<std::shared_mutex> lock(engine->common_tensor_suffix_cache_mutex);
        engine->common_tensor_suffix_cache.emplace(std::move(cache_key), result);
    }
    return result;
}

static int moe_engine_graph_ir_mentions(const moe_pc_engine* engine, const char* needle) {
    if (!engine || !needle || !needle[0] || engine->offload_graph_ir_json.empty()) {
        return 0;
    }
    const std::string key = moe_lower_ascii_copy(std::string(needle));
    {
        std::shared_lock<std::shared_mutex> lock(engine->text_mention_cache_mutex);
        auto found = engine->graph_ir_mention_cache.find(key);
        if (found != engine->graph_ir_mention_cache.end()) {
            return found->second;
        }
    }
    const int result = moe_ascii_contains_ci(engine->offload_graph_ir_json, needle);
    {
        std::unique_lock<std::shared_mutex> lock(engine->text_mention_cache_mutex);
        engine->graph_ir_mention_cache.emplace(key, result);
    }
    return result;
}

static int moe_engine_metadata_mentions(const moe_pc_engine* engine, const char* needle) {
    if (!engine || !needle || !needle[0] || engine->offload_gguf_metadata_json.empty()) {
        return 0;
    }
    const std::string key = moe_lower_ascii_copy(std::string(needle));
    {
        std::shared_lock<std::shared_mutex> lock(engine->text_mention_cache_mutex);
        auto found = engine->metadata_mention_cache.find(key);
        if (found != engine->metadata_mention_cache.end()) {
            return found->second;
        }
    }
    const int result = moe_ascii_contains_ci(engine->offload_gguf_metadata_json, needle);
    {
        std::unique_lock<std::shared_mutex> lock(engine->text_mention_cache_mutex);
        engine->metadata_mention_cache.emplace(key, result);
    }
    return result;
}

static int moe_engine_contract_has_common_tensor_suffix_global(
    const moe_pc_engine* engine,
    const char* suffix
) {
    if (!engine || !suffix || !suffix[0]) {
        return 0;
    }
    const std::string suffix_lower = moe_lower_ascii_copy(std::string(suffix));
    std::string cache_key("global\n");
    cache_key.append(suffix_lower);
    {
        std::shared_lock<std::shared_mutex> lock(engine->common_tensor_suffix_cache_mutex);
        auto found = engine->common_tensor_suffix_cache.find(cache_key);
        if (found != engine->common_tensor_suffix_cache.end()) {
            return found->second;
        }
    }
    int result = 0;
    for (const auto& it : engine->gguf_common_tensor_slots) {
        const std::string& key = it.first;
        if (key.size() >= suffix_lower.size() &&
            key.compare(key.size() - suffix_lower.size(), suffix_lower.size(), suffix_lower) == 0) {
            result = 1;
            break;
        }
    }
    {
        std::unique_lock<std::shared_mutex> lock(engine->common_tensor_suffix_cache_mutex);
        engine->common_tensor_suffix_cache.emplace(std::move(cache_key), result);
    }
    return result;
}

static int moe_engine_contract_has_layer_tensor_suffix_any(
    const moe_pc_engine* engine,
    int32_t layer,
    const char* const* suffixes,
    size_t suffix_count
) {
    if (!engine || !suffixes || suffix_count == 0) {
        return 0;
    }
    for (size_t i = 0; i < suffix_count; ++i) {
        const char* suffix = suffixes[i];
        if (!suffix || !suffix[0]) {
            continue;
        }
        if (layer >= 0) {
            if (moe_engine_has_common_tensor_suffix(engine, layer, suffix)) {
                return 1;
            }
        } else if (moe_engine_contract_has_common_tensor_suffix_global(engine, suffix)) {
            return 1;
        }
    }
    return 0;
}

static int moe_engine_contract_has_global_tensor_suffix_any(
    const moe_pc_engine* engine,
    const char* const* suffixes,
    size_t suffix_count
) {
    if (!engine || !suffixes || suffix_count == 0) {
        return 0;
    }
    for (size_t i = 0; i < suffix_count; ++i) {
        if (moe_engine_contract_has_common_tensor_suffix_global(engine, suffixes[i])) {
            return 1;
        }
    }
    return 0;
}

static int moe_engine_contract_uses_split_ffn_norm(
    const moe_pc_engine* engine,
    int32_t layer
) {
    if (!engine) {
        return 0;
    }
    static const char* const split_norm_suffixes[] = {
        "pre_ffw_norm_2.weight",
        "post_ffw_norm_1.weight",
        "post_ffw_norm_2.weight",
        "post_ffw_norm.weight",
        "ffn_norm_2.weight",
        "mlp_norm_2.weight"
    };
    // Do not use graph-wide counters/summary strings such as
    // layers_with_post_ffw_norm_1 to force a layer into the split-FFN path.
    // The runtime path is layer-local: actual tensors/roles must be present for
    // this layer.
    return moe_engine_contract_has_layer_tensor_suffix_any(
        engine, layer, split_norm_suffixes,
        sizeof(split_norm_suffixes) / sizeof(split_norm_suffixes[0]));
}

static int moe_engine_contract_has_router_input_scale(
    const moe_pc_engine* engine,
    int32_t layer
) {
    if (!engine) {
        return 0;
    }
    static const char* const router_scale_suffixes[] = {
        // Direct router-input activation scales. Keep this list narrow: generic
        // router.scale/moe.gate.scale names are ambiguous and may describe a
        // projection scale or a quantization sidecar, not the residual router
        // input transform.  ffn_gate_inp.scale/scales are router-weight
        // quantization sidecars in GGUF/MXFP artifacts.
        "router_input_scale",
        "router.input_scale",
        "mlp.router_input_scale",
        "moe.gate_input_scale",
        // Router input RMSNorm gamma. These imply a router-input transform but
        // must not be treated as bare direct activation scales.
        "router_norm.weight",
        "router.input_layernorm.weight",
        "router.input_norm.weight",
        "moe_router_norm.weight",
        "moe.gate_norm.weight",
        "mlp.router_norm.weight"
    };
    // Do not infer router input scaling from graph-wide text mentions, generic
    // router.scale/moe.gate.scale names, or concrete weight sidecars such as
    // ffn_gate_inp.scale/scales.
    return moe_engine_contract_has_layer_tensor_suffix_any(
        engine, layer, router_scale_suffixes,
        sizeof(router_scale_suffixes) / sizeof(router_scale_suffixes[0]));
}

static int moe_engine_contract_uses_direct_router_input_scale(
    const moe_pc_engine* engine,
    int32_t layer
) {
    if (!engine) {
        return 0;
    }
    static const char* const direct_scale_suffixes[] = {
        "router_input_scale",
        "router.input_scale",
        "mlp.router_input_scale",
        "moe.gate_input_scale"
    };
    // Direct router-input scale is narrower than "has router input scale".
    // Router_norm.weight-style tensors are RMSNorm gamma weights and must not
    // force hidden_size^-0.5 direct-scale semantics or unit_offset=0.
    return moe_engine_contract_has_layer_tensor_suffix_any(
        engine, layer, direct_scale_suffixes,
        sizeof(direct_scale_suffixes) / sizeof(direct_scale_suffixes[0])) &&
        moe_engine_contract_uses_split_ffn_norm(engine, layer);
}

static int moe_engine_contract_uses_direct_per_expert_scale(
    const moe_pc_engine* engine,
    int32_t layer
) {
    if (!engine) {
        return 0;
    }
    static const char* const direct_suffixes[] = {
        "per_expert_scale",
        "moe.per_expert_scale",
        "mlp.moe.per_expert_scale",
        "experts.per_expert_scale",
        "ffn_gate_inp.per_expert_scale",
        "router.per_expert_scale",
        "mlp.router.per_expert_scale",
        "moe.gate.per_expert_scale"
    };
    // Do not enable direct per-expert scaling from graph-wide mentions.  It is
    // valid only when a layer-local compatible tensor exists.
    return moe_engine_contract_has_layer_tensor_suffix_any(
        engine, layer, direct_suffixes,
        sizeof(direct_suffixes) / sizeof(direct_suffixes[0]));
}

static int moe_engine_contract_uses_unit_qk_norm_global(
    const moe_pc_engine* engine
) {
    if (!engine) {
        return 0;
    }
    if (moe_engine_graph_ir_mentions(engine, "unit_qk_norm") ||
        moe_engine_graph_ir_mentions(engine, "qk_norm_unit_scale") ||
        moe_engine_graph_ir_mentions(engine, "unit_attention_scale")) {
        return 1;
    }
    static const char* const q_norm_suffixes[] = {"attn_q_norm.weight", "q_norm.weight"};
    static const char* const k_norm_suffixes[] = {"attn_k_norm.weight", "k_norm.weight"};
    static const char* const block_scale_suffixes[] = {
        "post_attention_norm.weight",
        "post_ffw_norm.weight",
        "layer_output_scale.weight"
    };
    return moe_engine_contract_has_global_tensor_suffix_any(
               engine, q_norm_suffixes, sizeof(q_norm_suffixes) / sizeof(q_norm_suffixes[0])) &&
        moe_engine_contract_has_global_tensor_suffix_any(
               engine, k_norm_suffixes, sizeof(k_norm_suffixes) / sizeof(k_norm_suffixes[0])) &&
        moe_engine_contract_has_global_tensor_suffix_any(
               engine, block_scale_suffixes, sizeof(block_scale_suffixes) / sizeof(block_scale_suffixes[0]));
}


static int moe_engine_contract_uses_rmsnorm_unit_offset_tensor_contract(
    const moe_pc_engine* engine
) {
    (void)engine;
    // RMSNorm weight semantics are not inferable from tensor topology.  Models
    // with q/k norm, post-FFW norm, layer output scales, or PLE tensors may use
    // either direct gamma (x * weight) or unit-offset gamma (x * (1 + weight)).
    // Treat this helper as a hard no-op unless an explicit metadata/GraphIR
    // semantics flag says otherwise in moe_engine_contract_uses_rmsnorm_unit_offset().
    return 0;
}

static int moe_engine_contract_uses_direct_rmsnorm_weight(
    const moe_pc_engine* engine
) {
    if (!engine) {
        return 0;
    }
    int direct = 0;
    static const char* const direct_bool_keys[] = {
        "direct_rmsnorm_weight",
        "rmsnorm_direct_weight",
        "rms_norm_direct_weight",
        "rmsnorm_weight_direct",
        "rms_norm_weight_direct"
    };
    for (const char* key : direct_bool_keys) {
        if (moe_json_get_bool_local(engine->offload_gguf_metadata_json, key, &direct) ||
            moe_json_get_bool_local(engine->offload_graph_ir_json, key, &direct)) {
            return direct ? 1 : 0;
        }
    }
    if (moe_engine_graph_ir_mentions(engine, "direct_rmsnorm_weight") ||
        moe_engine_graph_ir_mentions(engine, "rmsnorm_direct_weight") ||
        moe_engine_graph_ir_mentions(engine, "rms_norm_direct_weight") ||
        moe_engine_graph_ir_mentions(engine, "rmsnorm_weight_semantics\":\"direct") ||
        moe_engine_graph_ir_mentions(engine, "rms_norm_weight_semantics\":\"direct") ||
        moe_engine_metadata_mentions(engine, "direct_rmsnorm_weight") ||
        moe_engine_metadata_mentions(engine, "rmsnorm_direct_weight") ||
        moe_engine_metadata_mentions(engine, "rms_norm_direct_weight") ||
        moe_engine_metadata_mentions(engine, "rmsnorm_weight_semantics\":\"direct") ||
        moe_engine_metadata_mentions(engine, "rms_norm_weight_semantics\":\"direct")) {
        return 1;
    }
    return 0;
}

static int moe_engine_contract_uses_rmsnorm_unit_offset(
    const moe_pc_engine* engine
) {
    if (!engine) {
        return 0;
    }
    int unit_offset = 0;
    static const char* const unit_bool_keys[] = {
        "rms_norm_unit_offset",
        "rmsnorm_unit_offset",
        "rms_norm_weight_unit_offset",
        "rmsnorm_weight_unit_offset",
        "rms_norm_weight_plus_one",
        "rmsnorm_weight_plus_one",
        "rms_norm_add_unit",
        "rmsnorm_add_unit"
    };
    for (const char* key : unit_bool_keys) {
        if (moe_json_get_bool_local(engine->offload_gguf_metadata_json, key, &unit_offset) ||
            moe_json_get_bool_local(engine->offload_graph_ir_json, key, &unit_offset)) {
            return unit_offset ? 1 : 0;
        }
    }
    // Do not treat the mere presence of a bool key name (for example
    // "rmsnorm_unit_offset": false) as a true contract. Only explicit string
    // semantics may opt into x * (1 + weight).
    if (moe_engine_graph_ir_mentions(engine, "rmsnorm_weight_semantics\":\"unit_offset") ||
        moe_engine_graph_ir_mentions(engine, "rms_norm_weight_semantics\":\"unit_offset") ||
        moe_engine_graph_ir_mentions(engine, "rmsnorm_weight_semantics\": \"unit_offset") ||
        moe_engine_graph_ir_mentions(engine, "rms_norm_weight_semantics\": \"unit_offset") ||
        moe_engine_graph_ir_mentions(engine, "rmsnorm_weight_semantics\":\"plus_one") ||
        moe_engine_graph_ir_mentions(engine, "rms_norm_weight_semantics\":\"plus_one") ||
        moe_engine_graph_ir_mentions(engine, "rmsnorm_weight_semantics\": \"plus_one") ||
        moe_engine_graph_ir_mentions(engine, "rms_norm_weight_semantics\": \"plus_one") ||
        moe_engine_metadata_mentions(engine, "rmsnorm_weight_semantics\":\"unit_offset") ||
        moe_engine_metadata_mentions(engine, "rms_norm_weight_semantics\":\"unit_offset") ||
        moe_engine_metadata_mentions(engine, "rmsnorm_weight_semantics\": \"unit_offset") ||
        moe_engine_metadata_mentions(engine, "rms_norm_weight_semantics\": \"unit_offset") ||
        moe_engine_metadata_mentions(engine, "rmsnorm_weight_semantics\":\"plus_one") ||
        moe_engine_metadata_mentions(engine, "rms_norm_weight_semantics\":\"plus_one") ||
        moe_engine_metadata_mentions(engine, "rmsnorm_weight_semantics\": \"plus_one") ||
        moe_engine_metadata_mentions(engine, "rms_norm_weight_semantics\": \"plus_one")) {
        return 1;
    }
    return moe_engine_contract_uses_rmsnorm_unit_offset_tensor_contract(engine);
}

static int moe_engine_contract_uses_gelu_tanh_mlp(
    const moe_pc_engine* engine
) {
    if (!engine) {
        return 0;
    }
    static const char* const gelu_tanh_terms[] = {
        "gelu_pytorch_tanh",
        "gelu_tanh",
        "gelu_new",
        "gelu_fast",
        "approximate_gelu"
    };
    for (const char* term : gelu_tanh_terms) {
        if (moe_engine_graph_ir_mentions(engine, term) ||
            moe_engine_metadata_mentions(engine, term)) {
            return 1;
        }
    }
    return 0;
}

static int moe_engine_contract_allows_missing_v_projection(
    const moe_pc_engine* engine,
    int32_t layer
) {
    if (!engine) {
        return 0;
    }
    // Missing V is a layer-local K-as-V contract.  A global summary/string such
    // as layers_with_no_v_proj must not hide a failed V-projection bind on other
    // layers.
    static const char* const q_suffixes[] = {"attn_q.weight", "q_proj.weight", "self_attn.q_proj.weight"};
    static const char* const k_suffixes[] = {"attn_k.weight", "k_proj.weight", "self_attn.k_proj.weight"};
    static const char* const v_suffixes[] = {"attn_v.weight", "v_proj.weight", "self_attn.v_proj.weight"};
    static const char* const o_suffixes[] = {"attn_output.weight", "o_proj.weight", "self_attn.o_proj.weight"};
    const int has_q = moe_engine_contract_has_layer_tensor_suffix_any(engine, layer, q_suffixes, sizeof(q_suffixes) / sizeof(q_suffixes[0]));
    const int has_k = moe_engine_contract_has_layer_tensor_suffix_any(engine, layer, k_suffixes, sizeof(k_suffixes) / sizeof(k_suffixes[0]));
    const int has_v = moe_engine_contract_has_layer_tensor_suffix_any(engine, layer, v_suffixes, sizeof(v_suffixes) / sizeof(v_suffixes[0]));
    const int has_o = moe_engine_contract_has_layer_tensor_suffix_any(engine, layer, o_suffixes, sizeof(o_suffixes) / sizeof(o_suffixes[0]));
    return has_q && has_k && has_o && !has_v;
}
