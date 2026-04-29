// Raw forward - Linear and RMSNorm operations

int glm5_pc_engine_find_raw_tensor(
    glm5_pc_engine_t* engine,
    int32_t layer,
    glm5_raw_tensor_kind_t kind,
    glm5_raw_tensor_info_t* out_info
) {
    (void)engine;
    if (!out_info) {
        return 0;
    }
    glm5_raw_tensor_desc desc{};
    const glm5_storage_raw_span_spec_t* span = nullptr;
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

int glm5_pc_engine_raw_linear_f32(
    glm5_pc_engine_t* engine,
    int32_t layer,
    glm5_raw_tensor_kind_t kind,
    const float* input,
    uint32_t input_count,
    float* output,
    uint32_t output_count
) {
    if (!engine || !input || !output) {
        return 0;
    }
    glm5_raw_tensor_cache_entry raw{};
    if (!get_cached_raw_tensor(engine, layer, kind, &raw) ||
        raw.cols == 0 || input_count != raw.cols || output_count != raw.rows) {
        return 0;
    }
    const uint64_t row_bytes = (uint64_t)raw.cols * 2ull;
    return glm5_parallel_rows(engine, raw.rows, 0, [&](uint32_t row) {
        output[row] = glm5_dot_bf16_row(raw.base + row_bytes * (uint64_t)row, input, input_count);
        return 1;
    });
}

int glm5_pc_engine_raw_rmsnorm_f32(
    glm5_pc_engine_t* engine,
    int32_t layer,
    glm5_raw_tensor_kind_t kind,
    const float* input,
    uint32_t input_count,
    float* output,
    uint32_t output_count,
    float eps
) {
    if (!engine || !input || !output || input_count == 0 || output_count != input_count) {
        return 0;
    }
    glm5_raw_tensor_cache_entry raw{};
    if (!get_cached_raw_tensor(engine, layer, kind, &raw) ||
        raw.cols != 0 || raw.rows != input_count) {
        return 0;
    }
    double ss = 0.0;
    uint32_t i = 0;
#if defined(__AVX2__)
    // Use double precision accumulator for better accuracy with large hidden_size
    __m256d sum0 = _mm256_setzero_pd();
    __m256d sum1 = _mm256_setzero_pd();
    // Fix: Process 16 elements at a time to avoid OOB read on x1
    for (; i + 15 < input_count; i += 16) {
        __m256 x0 = _mm256_loadu_ps(input + i);
        __m256 x1 = _mm256_loadu_ps(input + i + 8);
        // Convert float to double for higher precision accumulation
        __m256d xd0 = _mm256_cvtps_pd(_mm256_castps256_ps128(x0));
        __m256d xd1 = _mm256_cvtps_pd(_mm256_extractf128_ps(x0, 1));
        __m256d xd2 = _mm256_cvtps_pd(_mm256_castps256_ps128(x1));
        __m256d xd3 = _mm256_cvtps_pd(_mm256_extractf128_ps(x1, 1));
        sum0 = _mm256_fmadd_pd(xd0, xd0, sum0);
        sum1 = _mm256_fmadd_pd(xd1, xd1, sum1);
        sum0 = _mm256_fmadd_pd(xd2, xd2, sum0);
        sum1 = _mm256_fmadd_pd(xd3, xd3, sum1);
    }
    sum0 = _mm256_add_pd(sum0, sum1);
    double buf[4];
    _mm256_storeu_pd(buf, sum0);
    ss = buf[0] + buf[1] + buf[2] + buf[3];
#endif
    for (; i < input_count; ++i) {
        ss += (double)input[i] * (double)input[i];
    }
    const float norm = 1.0f / std::sqrt((float)(ss / (double)input_count) + (eps > 0.0f ? eps : 1.0e-6f));
    return glm5_parallel_rows(engine, input_count, 0, [&](uint32_t i) {
        uint16_t w;
        std::memcpy(&w, raw.base + (uint64_t)i * 2ull, sizeof(w));
        output[i] = input[i] * norm * glm5_raw_bf16_to_f32(w);
        return 1;
    });
}
