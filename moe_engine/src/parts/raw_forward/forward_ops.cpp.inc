
static float moe_round_f32_to_bf16_value(float value) {
    if (!std::isfinite(value)) {
        return value;
    }
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t lsb = (bits >> 16u) & 1u;
    bits += 0x7fffu + lsb;
    bits &= 0xffff0000u;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

static uint16_t moe_f32_to_fp16_bits_rne(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16u) & 0x8000u;
    uint32_t mant = bits & 0x007fffffu;
    int32_t exp = (int32_t)((bits >> 23u) & 0xffu) - 127 + 15;
    if (exp <= 0) {
        if (exp < -10) {
            return (uint16_t)sign;
        }
        mant |= 0x00800000u;
        const uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        const uint32_t round_bit = (mant >> (shift - 1u)) & 1u;
        const uint32_t sticky = mant & ((1u << (shift - 1u)) - 1u);
        if (round_bit && (sticky || (half_mant & 1u))) {
            ++half_mant;
        }
        return (uint16_t)(sign | half_mant);
    }
    if (exp >= 31) {
        if (((bits >> 23u) & 0xffu) == 0xffu && mant) {
            return (uint16_t)(sign | 0x7e00u);
        }
        return (uint16_t)(sign | 0x7c00u);
    }
    uint32_t half_mant = mant >> 13u;
    const uint32_t round = mant & 0x1fffu;
    if (round > 0x1000u || (round == 0x1000u && (half_mant & 1u))) {
        ++half_mant;
        if (half_mant == 0x400u) {
            half_mant = 0;
            ++exp;
            if (exp >= 31) {
                return (uint16_t)(sign | 0x7c00u);
            }
        }
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10u) | half_mant);
}

static float moe_round_f32_to_fp16_value(float value) {
    if (!std::isfinite(value)) {
        return value;
    }
    return moe_raw_fp16_to_f32(moe_f32_to_fp16_bits_rne(value));
}

static float moe_cast_embedding_scale_to_tensor_dtype(
    float scale,
    uint8_t scalar_encoding,
    uint8_t weight_format,
    const char** out_cast_source
) {
    const char* cast_source = "f32";
    float cast_scale = scale;
    if (scalar_encoding == moe_RAW_SCALAR_BF16 || weight_format == moe_WEIGHT_ENCODING_RAW_BF16) {
        cast_scale = moe_round_f32_to_bf16_value(scale);
        cast_source = "bf16";
    } else if (scalar_encoding == moe_RAW_SCALAR_F16 || weight_format == moe_WEIGHT_ENCODING_RAW_FP16) {
        cast_scale = moe_round_f32_to_fp16_value(scale);
        cast_source = "fp16";
    }
    if (out_cast_source) {
        *out_cast_source = cast_source;
    }
    return std::isfinite(cast_scale) && cast_scale > 0.0f ? cast_scale : scale;
}

static float moe_embedding_scale_value(const moe_pc_engine* engine, uint32_t hidden_size) {
    if (!engine || hidden_size == 0 || hidden_size > 100000u) {
        return 1.0f;
    }
    const uint32_t cached_hidden = engine->cached_embedding_scale_hidden.load(std::memory_order_acquire);
    const uint32_t cached_bits = engine->cached_embedding_scale_bits.load(std::memory_order_acquire);
    if (cached_hidden == hidden_size && cached_bits != 0u) {
        return moe_f32_from_bits(cached_bits);
    }

    double value = 0.0;
    int bool_value = 0;
    float scale = 1.0f;
    const char* source = "default";
    if (!engine->offload_gguf_metadata_json.empty() &&
        (moe_json_get_double_local(engine->offload_gguf_metadata_json, "embedding_scale", &value) ||
         moe_json_get_double_local(engine->offload_gguf_metadata_json, "embed_scale", &value) ||
         moe_json_get_double_local(engine->offload_gguf_metadata_json, "scale_emb", &value) ||
         moe_json_get_double_local(engine->offload_gguf_metadata_json, "scale_embedding", &value)) &&
        value > 0.0 && std::isfinite(value)) {
        scale = (float)value;
        source = "metadata_double";
    } else if (!engine->offload_graph_ir_json.empty() &&
        (moe_json_get_double_local(engine->offload_graph_ir_json, "embedding_scale", &value) ||
         moe_json_get_double_local(engine->offload_graph_ir_json, "embed_scale", &value) ||
         moe_json_get_double_local(engine->offload_graph_ir_json, "scale_emb", &value) ||
         moe_json_get_double_local(engine->offload_graph_ir_json, "scale_embedding", &value)) &&
        value > 0.0 && std::isfinite(value)) {
        scale = (float)value;
        source = "graph_ir_double";
    } else if ((!engine->offload_graph_ir_json.empty() &&
                (moe_json_get_bool_local(engine->offload_graph_ir_json, "scale_embedding", &bool_value) ||
                 moe_json_get_bool_local(engine->offload_graph_ir_json, "sqrt_hidden_size", &bool_value)) &&
                bool_value) ||
               (!engine->offload_gguf_metadata_json.empty() &&
                (moe_json_get_bool_local(engine->offload_gguf_metadata_json, "scale_embedding", &bool_value) ||
                 moe_json_get_bool_local(engine->offload_gguf_metadata_json, "sqrt_hidden_size", &bool_value)) &&
                bool_value)) {
        scale = std::sqrt((float)hidden_size);
        source = "explicit_bool_sqrt_hidden";
    } else if (moe_engine_metadata_mentions(engine, "sqrt_hidden_size")) {
        scale = std::sqrt((float)hidden_size);
        source = "metadata_text_sqrt_hidden";
    }
    moe_profile_log(
        "embedding_scale_contract",
        "source=%s scale=%.9g hidden=%u",
        source,
        (double)scale,
        hidden_size);
    engine->cached_embedding_scale_hidden.store(hidden_size, std::memory_order_release);
    engine->cached_embedding_scale_bits.store(moe_f32_bits(scale), std::memory_order_release);
    return scale;
}

static float moe_embedding_scale_value_for_tensor(
    const moe_pc_engine* engine,
    uint32_t hidden_size,
    uint8_t scalar_encoding,
    uint8_t weight_format,
    const char** out_cast_source
) {
    const float scale = moe_embedding_scale_value(engine, hidden_size);
    return moe_cast_embedding_scale_to_tensor_dtype(scale, scalar_encoding, weight_format, out_cast_source);
}

static void moe_apply_embedding_scale_f32(
    const moe_pc_engine* engine,
    const moe_storage_constants_t* c,
    float* hidden,
    uint8_t scalar_encoding = moe_RAW_SCALAR_UNKNOWN,
    uint8_t weight_format = moe_WEIGHT_ENCODING_UNKNOWN
) {
    if (!engine || !c || !hidden || c->hidden_size == 0) {
        return;
    }
    const char* cast_source = "f32";
    const float base_scale = moe_embedding_scale_value(engine, c->hidden_size);
    const float scale = moe_cast_embedding_scale_to_tensor_dtype(
        base_scale, scalar_encoding, weight_format, &cast_source);
    if (scale == 1.0f || !std::isfinite(scale)) {
        return;
    }
    static std::atomic<uint32_t> cast_logged{0};
    if (cast_logged.fetch_add(1, std::memory_order_relaxed) < 4u) {
        moe_profile_log(
            "embedding_scale_contract",
            "cast=%s applied_scale=%.9g base_scale=%.9g scalar=%u format=%u hidden=%u",
            cast_source,
            (double)scale,
            (double)base_scale,
            (unsigned)scalar_encoding,
            (unsigned)weight_format,
            c->hidden_size);
    }
    for (uint32_t i = 0; i < c->hidden_size; ++i) {
        hidden[i] *= scale;
    }
}

static int moe_engine_gelu_activation_mode(const moe_pc_engine* engine) {
    if (!engine) {
        return 0;
    }
    const int cached = engine->cached_gelu_activation_mode.load(std::memory_order_acquire);
    if (cached >= 0) {
        return cached;
    }

    auto activation_mode_from_value = [](const std::string& value, int* out_mode) -> int {
        if (!out_mode) {
            return 0;
        }
        const std::string lower = moe_lower_ascii_copy(value);
        if (lower.find("gelu_pytorch_tanh") != std::string::npos ||
            lower.find("gelu_tanh") != std::string::npos ||
            lower.find("gelu_new") != std::string::npos ||
            lower.find("gelu_fast") != std::string::npos ||
            lower.find("approximate_gelu") != std::string::npos) {
            *out_mode = 2;
            return 1;
        }
        if (lower.find("gelu") != std::string::npos) {
            *out_mode = 1;
            return 1;
        }
        if (lower.find("silu") != std::string::npos ||
            lower.find("swiglu") != std::string::npos ||
            lower.find("swi_glu") != std::string::npos) {
            *out_mode = 0;
            return 1;
        }
        return 0;
    };

    int mode = 0;
    int resolved = 0;
    const char* source = "default_silu";
    const char* explicit_key = "-";
    std::string explicit_value;
    if (!engine->offload_gguf_metadata_json.empty()) {
        static const char* activation_keys[] = {
            "hidden_activation",
            "hidden_act",
            "activation_function",
            "activation_fn",
            "act_fn",
            "feed_forward_activation",
            "mlp_activation",
            "intermediate_act_fn",
        };
        for (const char* key : activation_keys) {
            std::string value;
            if (!moe_json_get_string_local(engine->offload_gguf_metadata_json, key, &value)) {
                continue;
            }
            explicit_key = key;
            explicit_value = value;
            if (activation_mode_from_value(value, &mode)) {
                source = "metadata_key";
                resolved = 1;
                break;
            }
        }
    }
    if (!resolved && moe_engine_contract_uses_gelu_tanh_mlp(engine)) {
        mode = 2;
        source = "gelu_tanh_mlp_contract";
        resolved = 1;
    } else if (!resolved &&
        (moe_engine_graph_ir_mentions(engine, "gelu") ||
         moe_engine_metadata_mentions(engine, "gelu"))) {
        mode = 1;
        source = "text_mention_gelu";
        resolved = 1;
    }
    engine->cached_gelu_activation_mode.store(mode, std::memory_order_release);
    moe_profile_log(
        "activation_mode",
        "mode=%d name=%s source=%s explicit_key=%s explicit_value=%s",
        mode,
        mode == 2 ? "gelu_pytorch_tanh" : (mode == 1 ? "gelu" : "silu"),
        source,
        explicit_key,
        explicit_value.empty() ? "-" : explicit_value.c_str());
    return mode;
}

static int moe_engine_prefers_gelu_activation(const moe_pc_engine* engine) {
    return moe_engine_gelu_activation_mode(engine) != 0 ? 1 : 0;
}

static float moe_final_logit_softcap_value(const moe_pc_engine* engine) {
    if (!engine) {
        return 0.0f;
    }
    struct final_logit_softcap_cache_entry {
        const moe_pc_engine* engine = nullptr;
        uint64_t generation = 0;
        uint32_t value_bits = 0;
        int present = 0;
    };
    static thread_local std::vector<final_logit_softcap_cache_entry> tls_cache;
    const uint64_t generation = engine->engine_generation;
    for (const final_logit_softcap_cache_entry& e : tls_cache) {
        if (e.engine == engine && e.generation == generation) {
            return e.present ? moe_f32_from_bits(e.value_bits) : 0.0f;
        }
    }
    double value = 0.0;
    float cap = 0.0f;
    int present = 0;
    const char* source = "absent";
    if (!engine->offload_gguf_metadata_json.empty() &&
        (moe_json_get_double_local(engine->offload_gguf_metadata_json, "final_logit_softcapping", &value) ||
         moe_json_get_double_local(engine->offload_gguf_metadata_json, "final_logit_softcap", &value) ||
         moe_json_get_double_local(engine->offload_gguf_metadata_json, "logit_softcap", &value)) &&
        value > 0.0 && std::isfinite(value)) {
        cap = (float)value;
        present = 1;
        source = "gguf_metadata";
    } else if (!engine->offload_graph_ir_json.empty() &&
        (moe_json_get_double_local(engine->offload_graph_ir_json, "final_logit_softcapping", &value) ||
         moe_json_get_double_local(engine->offload_graph_ir_json, "final_logit_softcap", &value) ||
         moe_json_get_double_local(engine->offload_graph_ir_json, "logit_softcap", &value)) &&
        value > 0.0 && std::isfinite(value)) {
        cap = (float)value;
        present = 1;
        source = "graph_ir";
    }
    // One-time trace of softcap configuration for PPL diagnostics
    static std::atomic<uint32_t> softcap_traced{0};
    if (softcap_traced.fetch_add(1, std::memory_order_relaxed) == 0) {
        moe_profile_log(
            "final_logit_softcap_resolved",
            "present=%d value=%.6g source=%s",
            present, (double)cap, source);
    }
    if (tls_cache.size() > 64u) {
        tls_cache.erase(tls_cache.begin(), tls_cache.begin() + 16);
    }
    final_logit_softcap_cache_entry entry{};
    entry.engine = engine;
    entry.generation = generation;
    entry.value_bits = moe_f32_bits(cap);
    entry.present = present;
    tls_cache.push_back(entry);
    return present ? cap : 0.0f;
}

static float moe_apply_final_logit_softcap(const moe_pc_engine* engine, float score) {
    const float cap = moe_final_logit_softcap_value(engine);
    if (!(cap > 0.0f) || !std::isfinite(score)) {
        return score;
    }
    return std::tanh(score / cap) * cap;
}

static uint64_t moe_weight_format_row_bytes_for_cols(
    uint8_t weight_format,
    uint32_t cols
) {
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q8_0) {
        return ((uint64_t)cols + 31ull) / 32ull * 34ull;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q8_1) {
        return ((uint64_t)cols + 31ull) / 32ull * 36ull;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q8_K) {
        return ((uint64_t)cols + 255ull) / 256ull * 292ull;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q4_0) {
        return ((uint64_t)cols + 31ull) / 32ull * 18ull;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q4_1) {
        return ((uint64_t)cols + 31ull) / 32ull * 20ull;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q5_0) {
        return ((uint64_t)cols + 31ull) / 32ull * 22ull;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q5_1) {
        return ((uint64_t)cols + 31ull) / 32ull * 24ull;
    }
    if (weight_format == moe_WEIGHT_ENCODING_MXFP4) {
        return ((uint64_t)cols + 31ull) / 32ull * 17ull;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q2_K) {
        return ((uint64_t)cols + 255ull) / 256ull * 84ull;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q3_K) {
        return ((uint64_t)cols + 255ull) / 256ull * 110ull;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q4_K) {
        return ((uint64_t)cols + 255ull) / 256ull * 144ull;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q5_K) {
        return ((uint64_t)cols + 255ull) / 256ull * 176ull;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q6_K) {
        return ((uint64_t)cols + 255ull) / 256ull * 210ull;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_IQ2_XXS) {
        return ((uint64_t)cols + 255ull) / 256ull * 66ull;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_IQ2_XS) {
        return ((uint64_t)cols + 255ull) / 256ull * 74ull;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_IQ2_S) {
        return ((uint64_t)cols + 255ull) / 256ull * 82ull;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_IQ3_XXS) {
        return ((uint64_t)cols + 255ull) / 256ull * 98ull;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_IQ3_S) {
        return ((uint64_t)cols + 255ull) / 256ull * 110ull;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_IQ4_NL) {
        return ((uint64_t)cols + 31ull) / 32ull * 18ull;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_IQ4_XS) {
        return ((uint64_t)cols + 255ull) / 256ull * 136ull;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_IQ1_S) {
        return ((uint64_t)cols + 255ull) / 256ull * 50ull;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_IQ1_M) {
        return ((uint64_t)cols + 255ull) / 256ull * 56ull;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_TQ1_0) {
        return ((uint64_t)cols + 255ull) / 256ull * 54ull;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_TQ2_0) {
        return ((uint64_t)cols + 255ull) / 256ull * 66ull;
    }
    return 0;
}

static uint8_t moe_effective_weight_format_for_row_bytes(
    const moe_raw_tensor_cache_entry& raw,
    uint32_t cols
) {
    if (!raw.weight_row_bytes || cols == 0 ||
        raw.weight_format != moe_WEIGHT_ENCODING_MXFP4) {
        return raw.weight_format;
    }
    const uint64_t actual = raw.weight_row_bytes;
    if (actual == moe_weight_format_row_bytes_for_cols(raw.weight_format, cols)) {
        return raw.weight_format;
    }
    (void)actual;
    // Row stride can differ from logical codec row bytes when the JUJU index
    // uses physical row padding. The codec itself is contractual; never infer a
    // different quant family from byte counts.
    return raw.weight_format;
}

static uint64_t moe_raw_tensor_row_bytes(
    const moe_raw_tensor_cache_entry& raw,
    uint32_t cols
) {
    if (raw.weight_row_bytes) {
        return raw.weight_row_bytes;
    }
    const uint64_t quant_bytes = moe_weight_format_row_bytes_for_cols(raw.weight_format, cols);
    if (quant_bytes) {
        return quant_bytes;
    }
    const uint32_t elem = moe_raw_scalar_size(raw.scalar_encoding);
    return (uint64_t)cols * (uint64_t)elem;
}

static float moe_raw_tensor_invalid_dot_value(const char* reason, const moe_raw_tensor_cache_entry& raw, uint32_t row, uint32_t hidden_count) {
    static std::atomic<uint32_t> logs{0};
    const uint32_t n = logs.fetch_add(1, std::memory_order_relaxed);
    if (n < 64u) {
        moe_profile_log(
            "raw_dot_invalid",
            "reason=%s row=%u rows=%u cols=%u physical_rows=%u physical_cols=%u hidden=%u fmt=%u scalar=%u transposed=%u bytes=%llu row_bytes=%llu",
            reason ? reason : "unknown",
            row,
            raw.rows,
            raw.cols,
            raw.physical_rows,
            raw.physical_cols,
            hidden_count,
            (unsigned)raw.weight_format,
            (unsigned)raw.scalar_encoding,
            (unsigned)raw.math_transposed,
            (unsigned long long)raw.bytes,
            (unsigned long long)raw.weight_row_bytes);
    }
    return std::numeric_limits<float>::quiet_NaN();
}

static float moe_raw_tensor_dot_row_transposed(
    const moe_raw_tensor_cache_entry& raw,
    uint32_t row,
    const float* hidden,
    uint32_t hidden_count
) {
    if (!raw.base || !hidden || row >= raw.rows || hidden_count == 0) {
        return moe_raw_tensor_invalid_dot_value("transposed_bounds", raw, row, hidden_count);
    }
    if (raw.weight_format != moe_WEIGHT_ENCODING_UNKNOWN &&
        raw.weight_format != moe_WEIGHT_ENCODING_RAW_FP16 &&
        raw.weight_format != moe_WEIGHT_ENCODING_RAW_BF16 &&
        raw.weight_format != moe_WEIGHT_ENCODING_RAW_FP32) {
        return moe_raw_tensor_invalid_dot_value("transposed_quant_unsupported", raw, row, hidden_count);
    }
    const uint32_t elem = moe_raw_scalar_size(raw.scalar_encoding);
    if (!elem) {
        return moe_raw_tensor_invalid_dot_value("transposed_scalar_unknown", raw, row, hidden_count);
    }
    const uint32_t physical_cols = raw.physical_cols ? raw.physical_cols : raw.rows;
    const uint32_t physical_rows = raw.physical_rows ? raw.physical_rows : raw.cols;
    if (row >= physical_cols || hidden_count > physical_rows) {
        return moe_raw_tensor_invalid_dot_value("transposed_shape_mismatch", raw, row, hidden_count);
    }
    const uint64_t physical_row_bytes = raw.weight_row_bytes ? raw.weight_row_bytes : (uint64_t)physical_cols * (uint64_t)elem;
    if (physical_row_bytes == 0 || physical_row_bytes > raw.bytes) {
        return moe_raw_tensor_invalid_dot_value("transposed_row_bytes", raw, row, hidden_count);
    }
    double acc = 0.0;
    for (uint32_t i = 0; i < hidden_count; ++i) {
        if ((uint64_t)i > UINT64_MAX / physical_row_bytes) {
            return moe_raw_tensor_invalid_dot_value("transposed_offset_overflow", raw, row, hidden_count);
        }
        const uint64_t off = (uint64_t)i * physical_row_bytes + (uint64_t)row * (uint64_t)elem;
        if (off > raw.bytes || (uint64_t)elem > raw.bytes - off) {
            return moe_raw_tensor_invalid_dot_value("transposed_element_bounds", raw, row, hidden_count);
        }
        const float w = moe_raw_scalar_to_f32(raw.base + off, raw.scalar_encoding);
        if (!std::isfinite(w) || !std::isfinite(hidden[i])) {
            return moe_raw_tensor_invalid_dot_value("transposed_nonfinite", raw, row, hidden_count);
        }
        acc += (double)w * (double)hidden[i];
    }
    return (float)acc;
}

static float moe_raw_tensor_dot_row(
    const moe_raw_tensor_cache_entry& raw,
    uint32_t row,
    const float* hidden,
    uint32_t hidden_count
) {
    if (raw.math_transposed) {
        return moe_raw_tensor_dot_row_transposed(raw, row, hidden, hidden_count);
    }
    const uint32_t stride_cols = raw.cols ? raw.cols : hidden_count;
    const uint64_t row_bytes = moe_raw_tensor_row_bytes(raw, stride_cols);
    if (!raw.base || !hidden || row >= raw.rows || row_bytes == 0 ||
        row > UINT64_MAX / row_bytes ||
        (uint64_t)row * row_bytes > raw.bytes ||
        row_bytes > raw.bytes - (uint64_t)row * row_bytes) {
        static std::atomic<uint32_t> raw_dot_bounds_logs{0};
        const uint32_t n = raw_dot_bounds_logs.fetch_add(1, std::memory_order_relaxed);
        if (n < 32u) {
            moe_profile_log(
                "raw_dot_bounds_failed",
                "row=%u rows=%u cols=%u hidden=%u row_bytes=%llu bytes=%llu base=%p hidden=%p",
                row,
                raw.rows,
                raw.cols,
                hidden_count,
                (unsigned long long)row_bytes,
                (unsigned long long)raw.bytes,
                (const void*)raw.base,
                (const void*)hidden);
        }
        return moe_raw_tensor_invalid_dot_value("row_bounds", raw, row, hidden_count);
    }
    const uint8_t* row_ptr = raw.base + (uint64_t)row * row_bytes;
    const uint8_t effective_weight_format = moe_effective_weight_format_for_row_bytes(raw, stride_cols);
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_Q8_0) {
        return moe_dot_gguf_q8_0_row(row_ptr, hidden, hidden_count);
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_Q8_1) {
        return moe_dot_gguf_q8_1_row(row_ptr, hidden, hidden_count);
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_Q8_K) {
        return moe_dot_gguf_q8_k_row(row_ptr, hidden, hidden_count);
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_Q4_0) {
        return moe_dot_gguf_q4_0_row(row_ptr, hidden, hidden_count);
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_Q4_1) {
        return moe_dot_gguf_q4_1_row(row_ptr, hidden, hidden_count);
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_Q5_0) {
        return moe_dot_gguf_q5_0_row(row_ptr, hidden, hidden_count);
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_Q5_1) {
        return moe_dot_gguf_q5_1_row(row_ptr, hidden, hidden_count);
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_MXFP4) {
        return moe_dot_gguf_mxfp4_row_strided(row_ptr, hidden, hidden_count, row_bytes);
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_Q2_K) {
        return moe_dot_gguf_q2_k_row(row_ptr, hidden, hidden_count);
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_Q3_K) {
        return moe_dot_gguf_q3_k_row(row_ptr, hidden, hidden_count);
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_Q4_K) {
        return moe_dot_gguf_q4_k_row(row_ptr, hidden, hidden_count);
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_Q5_K) {
        return moe_dot_gguf_q5_k_row(row_ptr, hidden, hidden_count);
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_Q6_K) {
        return moe_dot_gguf_q6_k_row(row_ptr, hidden, hidden_count);
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_IQ2_XXS) {
        return moe_dot_gguf_iq2_xxs_row(row_ptr, hidden, hidden_count);
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_IQ2_XS) {
        return moe_dot_gguf_iq2_xs_row(row_ptr, hidden, hidden_count);
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_IQ2_S) {
        return moe_dot_gguf_iq2_s_row(row_ptr, hidden, hidden_count);
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_IQ3_XXS) {
        return moe_dot_gguf_iq3_xxs_row(row_ptr, hidden, hidden_count);
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_IQ3_S) {
        return moe_dot_gguf_iq3_s_row(row_ptr, hidden, hidden_count);
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_IQ4_NL) {
        return moe_dot_gguf_iq4_nl_row(row_ptr, hidden, hidden_count);
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_IQ4_XS) {
        return moe_dot_gguf_iq4_xs_row(row_ptr, hidden, hidden_count);
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_IQ1_S) {
        return moe_dot_gguf_iq1_s_row(row_ptr, hidden, hidden_count);
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_IQ1_M) {
        return moe_dot_gguf_iq1_m_row(row_ptr, hidden, hidden_count);
    }
    if (effective_weight_format != moe_WEIGHT_ENCODING_UNKNOWN &&
        effective_weight_format != moe_WEIGHT_ENCODING_RAW_FP16 &&
        effective_weight_format != moe_WEIGHT_ENCODING_RAW_BF16 &&
        effective_weight_format != moe_WEIGHT_ENCODING_RAW_FP32) {
        moe_trace_raw_dot_unsupported(raw);
        return moe_raw_tensor_invalid_dot_value("unsupported_weight_format", raw, row, hidden_count);
    }
    const float raw_dot = moe_dot_raw_row(row_ptr, raw.scalar_encoding, hidden, hidden_count);
    return std::isfinite(raw_dot) ? raw_dot : moe_raw_tensor_invalid_dot_value("raw_scalar_nonfinite", raw, row, hidden_count);
}
