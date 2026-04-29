static int glm5_raw_linear_span_f32(
    glm5_pc_engine* engine,
    const glm5_storage_raw_span_spec_t* span,
    uint64_t rel_offset,
    uint32_t rows,
    uint32_t cols,
    const float* input,
    uint32_t input_count,
    float* output,
    uint32_t output_count
) {
    if (!engine || !span || !input || !output || cols == 0 ||
        input_count != cols || output_count != rows) {
        return 0;
    }
    mmap_context_t* mapped = map_raw_span_part(engine, span);
    if (!raw_span_bounds_ok(span, mapped)) {
        return 0;
    }
    const uint64_t bytes = (uint64_t)rows * (uint64_t)cols * 2ull;
    const uint8_t* base = raw_span_ptr(span, mapped, rel_offset, bytes);
    if (!base) {
        return 0;
    }
    const uint64_t row_bytes = (uint64_t)cols * 2ull;
    return glm5_parallel_rows(engine, rows, 0, [&](uint32_t row) {
        output[row] = glm5_dot_bf16_row(base + row_bytes * (uint64_t)row, input, input_count);
        return 1;
    });
}

static int glm5_raw_mlp_span_f32(
    glm5_pc_engine* engine,
    const glm5_storage_raw_span_spec_t* span,
    const float* hidden,
    uint32_t hidden_size,
    uint32_t intermediate_size,
    float* scratch_gate,
    float* scratch_up,
    float* out_hidden
) {
    if (!engine || !span || !hidden || !scratch_gate || !scratch_up || !out_hidden ||
        hidden_size == 0 || intermediate_size == 0) {
        return 0;
    }
    const uint64_t matrix_bytes = (uint64_t)intermediate_size * (uint64_t)hidden_size * 2ull;
    const uint64_t expected_bytes = matrix_bytes * 2ull + (uint64_t)hidden_size * (uint64_t)intermediate_size * 2ull;
    if (span->bytes < expected_bytes) {
        return 0;
    }
    mmap_context_t* mapped = map_raw_span_part(engine, span);
    if (!raw_span_bounds_ok(span, mapped)) {
        return 0;
    }
    const uint8_t* gate_base = raw_span_ptr(span, mapped, 0, matrix_bytes);
    const uint8_t* up_base = raw_span_ptr(span, mapped, matrix_bytes, matrix_bytes);
    if (!gate_base || !up_base) {
        return 0;
    }
    const uint64_t row_bytes = (uint64_t)hidden_size * 2ull;
    if (!glm5_parallel_rows(engine, intermediate_size, 0, [&](uint32_t row) {
            const float gate = glm5_dot_bf16_row(gate_base + row_bytes * (uint64_t)row, hidden, hidden_size);
            const float up = glm5_dot_bf16_row(up_base + row_bytes * (uint64_t)row, hidden, hidden_size);
            scratch_up[row] = up;
            scratch_gate[row] = glm5_generation_silu(gate) * up;
            return 1;
        })) {
        return 0;
    }
    return glm5_raw_linear_span_f32(
        engine,
        span,
        matrix_bytes * 2ull,
        hidden_size,
        intermediate_size,
        scratch_gate,
        intermediate_size,
        out_hidden,
        hidden_size
    );
}
