// Raw forward - Linear and RMSNorm operations

static int moe_rmsnorm_unit_offset_from_json(
    const std::string& text,
    int* out
) {
    if (!out || text.empty()) {
        return 0;
    }
    static const char* const bool_keys[] = {
        "rms_norm_unit_offset",
        "rmsnorm_unit_offset",
        "rms_norm_weight_unit_offset",
        "rmsnorm_weight_unit_offset",
        "rms_norm_weight_plus_one",
        "rmsnorm_weight_plus_one",
        "rms_norm_add_unit",
        "rmsnorm_add_unit"
    };
    int bool_value = 0;
    for (const char* key : bool_keys) {
        if (moe_json_get_bool_local(text, key, &bool_value)) {
            *out = bool_value ? 1 : 0;
            return 1;
        }
    }
    static const char* const offset_keys[] = {
        "rms_norm_weight_offset",
        "rmsnorm_weight_offset",
        "rms_norm_gamma_offset",
        "rmsnorm_gamma_offset"
    };
    double offset = 0.0;
    for (const char* key : offset_keys) {
        if (moe_json_get_double_local(text, key, &offset) && std::isfinite(offset)) {
            *out = std::fabs(offset) >= 0.5 ? 1 : 0;
            return 1;
        }
    }
    return 0;
}

static uint64_t moe_rmsnorm_unit_offset_cache_key(
    const moe_raw_tensor_cache_entry& raw
) {
    if (!raw.base || raw.rows == 0 || raw.cols != 0 || raw.bytes == 0) {
        return 0;
    }
    uint64_t key = (uint64_t)(uintptr_t)raw.base;
    key ^= raw.bytes + 0x9e3779b97f4a7c15ull + (key << 6) + (key >> 2);
    key ^= ((uint64_t)raw.rows << 32) ^ ((uint64_t)raw.scalar_encoding << 24);
    return key ? key : 1u;
}

static int moe_engine_rmsnorm_unit_offset(
    const moe_pc_engine* engine,
    const moe_raw_tensor_cache_entry& raw
) {
    if (!engine) {
        return 0;
    }
    const uint64_t cache_key = moe_rmsnorm_unit_offset_cache_key(raw);
    if (cache_key) {
        std::shared_lock<std::shared_mutex> lock(engine->rmsnorm_unit_offset_cache_mutex);
        const auto found = engine->rmsnorm_unit_offset_cache.find(cache_key);
        if (found != engine->rmsnorm_unit_offset_cache.end()) {
            return found->second ? 1 : 0;
        }
    }

    int value = 0;
    int metadata_value = -1;
    const char* metadata_source = "none";
    const char* source = "contract_default";
    int parsed_value = 0;

    // Contract rule: unit-offset RMSNorm is architecture/config math, not a
    // tensor-statistics heuristic.  Previous code inferred (1+w) from weight
    // magnitudes; that made layer-local decisions drift across norm tensors and
    // caused the same model to run different math depending on quant sidecars.
    // Use only explicit metadata/GraphIR.  If absent, use normal RMSNorm weight.
    if (moe_rmsnorm_unit_offset_from_json(engine->offload_gguf_metadata_json, &parsed_value)) {
        metadata_value = parsed_value ? 1 : 0;
        metadata_source = "metadata";
        value = metadata_value;
        source = "metadata";
    } else if (moe_rmsnorm_unit_offset_from_json(engine->offload_graph_ir_json, &parsed_value)) {
        metadata_value = parsed_value ? 1 : 0;
        metadata_source = "graph_ir";
        value = metadata_value;
        source = "graph_ir";
    } else if (moe_engine_contract_uses_rmsnorm_unit_offset(engine)) {
        value = 1;
        source = "unit_offset_contract";
    } else if (moe_engine_contract_uses_direct_rmsnorm_weight(engine)) {
        value = 0;
        source = "direct_rmsnorm_contract";
    }

    moe_profile_log(
        "rmsnorm_unit_offset",
        "source=%s value=%d metadata=%d metadata_source=%s cache_key=%llu",
        source,
        value ? 1 : 0,
        metadata_value,
        metadata_source,
        (unsigned long long)cache_key);
    if (cache_key) {
        std::unique_lock<std::shared_mutex> lock(engine->rmsnorm_unit_offset_cache_mutex);
        engine->rmsnorm_unit_offset_cache[cache_key] = value ? 1 : 0;
    }
    return value ? 1 : 0;
}

int moe_pc_engine_find_raw_tensor(
    moe_pc_engine_t* engine,
    int32_t layer,
    moe_raw_tensor_kind_t kind,
    moe_raw_tensor_info_t* out_info
) {
    if (!engine || !out_info) {
        return 0;
    }
    moe_raw_tensor_desc desc{};
    if (const moe_gguf_common_tensor_record* rec =
            find_common_raw_tensor_record(engine, layer, kind, &desc)) {
        std::memset(out_info, 0, sizeof(*out_info));
        out_info->layer = layer;
        out_info->part = (uint32_t)rec->resolved_path_index + 1u;
        out_info->offset = rec->file_offset;
        out_info->bytes = rec->bytes;
        out_info->rows = desc.rows;
        out_info->cols = desc.cols;
        const uint8_t scalar_encoding =
            moe_raw_scalar_encoding_from_common_tensor(*rec, desc.rows, desc.cols);
        uint8_t weight_format = rec->weight_encoding != moe_WEIGHT_ENCODING_UNKNOWN ?
            rec->weight_encoding : moe_weight_encoding_from_gguf_type(rec->type, moe_offload_gguf_contract_t{});
        if (weight_format == moe_WEIGHT_ENCODING_UNKNOWN) {
            weight_format = moe_raw_weight_encoding_from_common_tensor(
                *rec, desc.rows, desc.cols, scalar_encoding);
        }
        if (weight_format == moe_WEIGHT_ENCODING_UNKNOWN) {
            return 0;
        }
        if (weight_format != moe_WEIGHT_ENCODING_RAW_FP32 &&
            weight_format != moe_WEIGHT_ENCODING_RAW_FP16 &&
            weight_format != moe_WEIGHT_ENCODING_RAW_BF16) {
            return 0;
        }
        out_info->bf16 = weight_format == moe_WEIGHT_ENCODING_RAW_BF16 ? 1 : 0;
        return 1;
    }
    if (engine->offload_gguf_valid.load(std::memory_order_relaxed)) {
        moe_trace_raw_tensor_static_fallback_blocked(layer, kind);
        return 0;
    }
    const moe_storage_raw_span_spec_t* span = nullptr;
    if (!find_raw_tensor_desc_and_span(layer, kind, &desc, &span)) {
        return 0;
    }
    std::memset(out_info, 0, sizeof(*out_info));
    out_info->layer = layer;
    out_info->part = span->part;
    out_info->offset = span->offset + desc.rel_offset;
    out_info->bytes = desc.bytes;
    out_info->rows = desc.rows;
    out_info->cols = desc.cols;
    out_info->bf16 = 1;
    return 1;
}

int moe_pc_engine_raw_linear_f32(
    moe_pc_engine_t* engine,
    int32_t layer,
    moe_raw_tensor_kind_t kind,
    const float* input,
    uint32_t input_count,
    float* output,
    uint32_t output_count
) {
    if (!engine || !input || !output) {
        return 0;
    }
    moe_raw_tensor_cache_entry raw{};
    if (!get_cached_raw_tensor(engine, layer, kind, &raw) ||
        raw.cols == 0 || input_count != raw.cols || output_count != raw.rows) {
        return 0;
    }
    const uint64_t raw_row_bytes = moe_raw_tensor_row_bytes(raw, raw.cols);
    if (raw.bytes < raw_row_bytes * (uint64_t)raw.rows) {
        return 0;
    }
    return moe_parallel_rows(engine, raw.rows, 0, [&](uint32_t row) {
        output[row] = moe_raw_tensor_dot_row(raw, row, input, input_count);
        return 1;
    });
}



static int moe_apply_unweighted_rmsnorm_f32(
    const float* input,
    uint32_t count,
    float* output,
    float eps
) {
    if (!input || !output || count == 0) {
        return 0;
    }
    double ss = 0.0;
    for (uint32_t i = 0; i < count; ++i) {
        if (!std::isfinite(input[i])) {
            return 0;
        }
        ss += (double)input[i] * (double)input[i];
    }
    if (!std::isfinite(ss) || ss < 0.0) {
        return 0;
    }
    const float norm = 1.0f / std::sqrt((float)(ss / (double)count) + (eps > 0.0f ? eps : 1.0e-6f));
    if (!std::isfinite(norm)) {
        return 0;
    }
    for (uint32_t i = 0; i < count; ++i) {
        const float v = input[i] * norm;
        if (!std::isfinite(v)) {
            return 0;
        }
        output[i] = v;
    }
    return 1;
}

int moe_pc_engine_raw_rmsnorm_f32(
    moe_pc_engine_t* engine,
    int32_t layer,
    moe_raw_tensor_kind_t kind,
    const float* input,
    uint32_t input_count,
    float* output,
    uint32_t output_count,
    float eps
) {
    if (!engine || !input || !output || input_count == 0 || output_count != input_count) {
        return 0;
    }
    moe_raw_tensor_cache_entry raw{};
    if (!get_cached_raw_tensor(engine, layer, kind, &raw) ||
        raw.cols != 0 || raw.rows != input_count) {
        return 0;
    }
    const uint32_t elem = moe_raw_scalar_size(raw.scalar_encoding);
    if (!elem) {
        return 0;
    }
    if (raw.bytes < (uint64_t)input_count * (uint64_t)elem) {
        return 0;
    }
    double ss = 0.0;
    uint32_t i = 0;
#if defined(__AVX2__)
    // Use double precision accumulator for better accuracy with large hidden_size
    __m256d sum0 = _mm256_setzero_pd();
    __m256d sum1 = _mm256_setzero_pd();
    // BUGFIX 276: Process 16 elements per iteration to match x0 (8) + x1 (8) loads
    // This prevents out-of-bounds read on x1 when input_count is not multiple of 16
    // Example: input_count=12 → without this check, x1 would read [8..15] → OOB at [12..15]
    for (; i + 15 < input_count; i += 16) {
        __m256 x0 = _mm256_loadu_ps(input + i);
        __m256 x1 = _mm256_loadu_ps(input + i + 8);
        // Convert float to double for higher precision accumulation
        __m256d xd0 = _mm256_cvtps_pd(_mm256_castps256_ps128(x0));
        __m256d xd1 = _mm256_cvtps_pd(_mm256_extractf128_ps(x0, 1));
        __m256d xd2 = _mm256_cvtps_pd(_mm256_castps256_ps128(x1));
        __m256d xd3 = _mm256_cvtps_pd(_mm256_extractf128_ps(x1, 1));
#if defined(__FMA__)
        sum0 = _mm256_fmadd_pd(xd0, xd0, sum0);
        sum1 = _mm256_fmadd_pd(xd1, xd1, sum1);
        sum0 = _mm256_fmadd_pd(xd2, xd2, sum0);
        sum1 = _mm256_fmadd_pd(xd3, xd3, sum1);
#else
        sum0 = _mm256_add_pd(_mm256_mul_pd(xd0, xd0), sum0);
        sum1 = _mm256_add_pd(_mm256_mul_pd(xd1, xd1), sum1);
        sum0 = _mm256_add_pd(_mm256_mul_pd(xd2, xd2), sum0);
        sum1 = _mm256_add_pd(_mm256_mul_pd(xd3, xd3), sum1);
#endif
    }
    sum0 = _mm256_add_pd(sum0, sum1);
    double buf[4];
    _mm256_storeu_pd(buf, sum0);
    ss = buf[0] + buf[1] + buf[2] + buf[3];
#endif
    for (; i < input_count; ++i) {
        if (!std::isfinite(input[i])) {
            return 0;
        }
        ss += (double)input[i] * (double)input[i];
    }
    if (!std::isfinite(ss) || ss < 0.0) {
        return 0;
    }
    const float norm = 1.0f / std::sqrt((float)(ss / (double)input_count) + (eps > 0.0f ? eps : 1.0e-6f));
    if (!std::isfinite(norm)) {
        return 0;
    }
    const int unit_offset = moe_engine_rmsnorm_unit_offset(engine, raw);
    if (moe_profile_level() >= 2) {
        double weight_ss = 0.0;
        double weight_abs = 0.0;
        float weight_min = std::numeric_limits<float>::infinity();
        float weight_max = -std::numeric_limits<float>::infinity();
        float input_max_abs = 0.0f;
        for (uint32_t j = 0; j < input_count; ++j) {
            const float w_raw = moe_raw_scalar_to_f32(raw.base + (uint64_t)j * (uint64_t)elem, raw.scalar_encoding);
            const float w = unit_offset ? (1.0f + w_raw) : w_raw;
            const float aw = std::fabs(w);
            weight_ss += (double)w * (double)w;
            weight_abs += (double)aw;
            if (w < weight_min) weight_min = w;
            if (w > weight_max) weight_max = w;
            const float ai = std::fabs(input[j]);
            if (ai > input_max_abs) input_max_abs = ai;
        }
        moe_profile_log(
            "rmsnorm_raw_stats_begin",
            "layer=%d kind=%u hidden=%u scalar=%u bytes=%llu unit_offset=%d eps=%.6g norm=%.6g input_rms=%.6g input_max_abs=%.6g weight_rms=%.6g weight_mean_abs=%.6g weight_min=%.6g weight_max=%.6g",
            (int)layer,
            (uint32_t)kind,
            input_count,
            (uint32_t)raw.scalar_encoding,
            (unsigned long long)raw.bytes,
            unit_offset,
            (double)eps,
            (double)norm,
            std::sqrt(ss / (double)input_count),
            (double)input_max_abs,
            std::sqrt(weight_ss / (double)input_count),
            weight_abs / (double)input_count,
            (double)weight_min,
            (double)weight_max);
    }
    const int ok = moe_parallel_rows(engine, input_count, 0, [&](uint32_t i) {
        const float w = moe_raw_scalar_to_f32(raw.base + (uint64_t)i * (uint64_t)elem, raw.scalar_encoding);
        if (!std::isfinite(w)) {
            return 0;
        }
        const float result = input[i] * norm * (unit_offset ? (1.0f + w) : w);
        if (!std::isfinite(result)) {
            return 0;
        }
        output[i] = result;
        return 1;
    });
    if (ok && moe_profile_level() >= 2) {
        double output_ss = 0.0;
        double output_abs = 0.0;
        float output_max_abs = 0.0f;
        for (uint32_t j = 0; j < input_count; ++j) {
            const float v = output[j];
            const float av = std::fabs(v);
            output_ss += (double)v * (double)v;
            output_abs += (double)av;
            if (av > output_max_abs) output_max_abs = av;
        }
        moe_profile_log(
            "rmsnorm_raw_stats_end",
            "layer=%d kind=%u hidden=%u output_rms=%.6g output_mean_abs=%.6g output_max_abs=%.6g first4=[%.6g,%.6g,%.6g,%.6g]",
            (int)layer,
            (uint32_t)kind,
            input_count,
            std::sqrt(output_ss / (double)input_count),
            output_abs / (double)input_count,
            (double)output_max_abs,
            input_count > 0 ? (double)output[0] : 0.0,
            input_count > 1 ? (double)output[1] : 0.0,
            input_count > 2 ? (double)output[2] : 0.0,
            input_count > 3 ? (double)output[3] : 0.0);
    }
    return ok;
}
