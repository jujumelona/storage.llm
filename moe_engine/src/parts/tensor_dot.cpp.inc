static int moe_dot_tensor_record_row(
    moe_pc_engine_t* engine,
    const moe_tensor_record* rec,
    const mmap_context_t* shard,
    uint32_t row,
    const float* x,
    uint32_t x_count,
    float* out_value
) {
    if (!engine || !rec || !x || !out_value ||
        x_count != rec->info.cols || !row_bounds_ok(*rec, row)) {
        return 0;
    }
    if (!shard || !mmap_get_ptr(shard, 0)) {
        return 0;
    }
    const int weight_format = tensor_weight_format(*rec);
    const uint64_t row_offset = rec->info.weight_byte_offset + (uint64_t)row * rec->weight_row_bytes;
    const uint64_t shard_size = (uint64_t)mmap_get_size(shard);
    if (row_offset > shard_size || rec->weight_row_bytes > shard_size - row_offset) {
        return 0;
    }
    const uint8_t* packed = static_cast<const uint8_t*>(mmap_get_ptr(shard, static_cast<size_t>(row_offset)));
    if (!packed) {
        return 0;
    }
    if (weight_format == moe_WEIGHT_ENCODING_RAW_FP16) {
        *out_value = moe_dot_raw_fp16_row(packed, x, rec->info.cols);
        return 1;
    }
    if (weight_format == moe_WEIGHT_ENCODING_RAW_FP32) {
        *out_value = moe_dot_raw_fp32_row(packed, x, rec->info.cols);
        return 1;
    }
    if (weight_format == moe_WEIGHT_ENCODING_NVFP4) {
        *out_value = moe_dot_fp4_row(*rec, shard, packed, x, engine, row);
        return 1;
    }
    if (weight_format == moe_WEIGHT_ENCODING_MXFP4) {
        *out_value = moe_dot_gguf_mxfp4_row(packed, x, rec->info.cols);
        return 1;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q8_0) {
        *out_value = moe_dot_gguf_q8_0_row(packed, x, rec->info.cols);
        return 1;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q5_1) {
        *out_value = moe_dot_gguf_q5_1_row(packed, x, rec->info.cols);
        return 1;
    }
    if (weight_format == moe_WEIGHT_ENCODING_LINEAR_Q2 ||
        weight_format == moe_WEIGHT_ENCODING_LINEAR_Q3 ||
        weight_format == moe_WEIGHT_ENCODING_LINEAR_Q4 ||
        weight_format == moe_WEIGHT_ENCODING_LINEAR_Q8) {
        *out_value = moe_dot_linear_quant_row(
            *rec,
            shard,
            packed,
            x,
            rec->weight_bits,
            row);
        return 1;
    }
    return 0;
}

// ============================================================================
// Batched Matrix-Vector Multiplication (GEMM Wrapper)
// ============================================================================
// Computes dot product of a weight row against multiple token hidden states.
// This is the foundational math kernel for Continuous Batching.
// ============================================================================
static int moe_dot_tensor_record_row_batch(
    moe_pc_engine_t* engine,
    const moe_tensor_record* rec,
    const mmap_context_t* shard,
    uint32_t row,
    const float** batch_x,
    uint32_t batch_size,
    uint32_t x_count,
    float* out_values // Size must be [batch_size]
) {
    if (!engine || !rec || !batch_x || !out_values || batch_size == 0 || x_count != rec->info.cols || !row_bounds_ok(*rec, row)) {
        return 0;
    }
    if (!shard || !mmap_get_ptr(shard, 0)) {
        return 0;
    }

    // Fast Path: Load the packed weight row exactly ONCE
    // and compute dot products for all tokens in the batch.
    // This massively increases arithmetic intensity (FLOPs/byte) and
    // overcomes the memory-bandwidth bottleneck of single-token decode.

    const int weight_format = tensor_weight_format(*rec);
    const uint64_t row_offset = rec->info.weight_byte_offset + (uint64_t)row * rec->weight_row_bytes;
    const uint64_t shard_size = (uint64_t)mmap_get_size(shard);
    if (row_offset > shard_size || rec->weight_row_bytes > shard_size - row_offset) {
        return 0;
    }
    const uint8_t* packed = static_cast<const uint8_t*>(mmap_get_ptr(shard, static_cast<size_t>(row_offset)));
    if (!packed) return 0;

    // TODO: A fully optimized SIMD SGEMM kernel would interleave the token vectors
    // to maximize register usage. For now, we reuse the L1 cache for the packed weights.
    for (uint32_t b = 0; b < batch_size; ++b) {
        if (weight_format == moe_WEIGHT_ENCODING_RAW_FP16) {
            out_values[b] = moe_dot_raw_fp16_row(packed, batch_x[b], rec->info.cols);
        }
        else if (weight_format == moe_WEIGHT_ENCODING_RAW_FP32) {
            out_values[b] = moe_dot_raw_fp32_row(packed, batch_x[b], rec->info.cols);
        }
        else if (weight_format == moe_WEIGHT_ENCODING_NVFP4) {
            out_values[b] = moe_dot_fp4_row(*rec, shard, packed, batch_x[b], engine, row);
        }
        else if (weight_format == moe_WEIGHT_ENCODING_MXFP4) {
            out_values[b] = moe_dot_gguf_mxfp4_row(packed, batch_x[b], rec->info.cols);
        }
        else if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q8_0) {
            out_values[b] = moe_dot_gguf_q8_0_row(packed, batch_x[b], rec->info.cols);
        }
        else if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q5_1) {
            out_values[b] = moe_dot_gguf_q5_1_row(packed, batch_x[b], rec->info.cols);
        }
        else {
            out_values[b] = moe_dot_linear_quant_row(*rec, shard, packed, batch_x[b], rec->weight_bits, row);
        }
    }
    return 1;
}
