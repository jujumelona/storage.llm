static int moe_dot_weight_format_supported(int weight_format) {
    return weight_format == moe_WEIGHT_ENCODING_RAW_FP16 ||
        weight_format == moe_WEIGHT_ENCODING_RAW_BF16 ||
        weight_format == moe_WEIGHT_ENCODING_RAW_FP32 ||
        weight_format == moe_WEIGHT_ENCODING_NVFP4 ||
        weight_format == moe_WEIGHT_ENCODING_MXFP4 ||
        weight_format == moe_WEIGHT_ENCODING_GGUF_Q4_0 ||
        weight_format == moe_WEIGHT_ENCODING_GGUF_Q4_1 ||
        weight_format == moe_WEIGHT_ENCODING_GGUF_Q5_0 ||
        weight_format == moe_WEIGHT_ENCODING_GGUF_Q8_0 ||
        weight_format == moe_WEIGHT_ENCODING_GGUF_Q8_1 ||
        weight_format == moe_WEIGHT_ENCODING_GGUF_Q8_K ||
        weight_format == moe_WEIGHT_ENCODING_GGUF_IQ4_NL ||
        weight_format == moe_WEIGHT_ENCODING_GGUF_IQ4_XS ||
        weight_format == moe_WEIGHT_ENCODING_GGUF_Q5_1 ||
        weight_format == moe_WEIGHT_ENCODING_GGUF_Q2_K ||
        weight_format == moe_WEIGHT_ENCODING_GGUF_Q3_K ||
        weight_format == moe_WEIGHT_ENCODING_GGUF_Q4_K ||
        weight_format == moe_WEIGHT_ENCODING_GGUF_Q5_K ||
        weight_format == moe_WEIGHT_ENCODING_GGUF_Q6_K ||
        weight_format == moe_WEIGHT_ENCODING_GGUF_IQ2_XXS ||
        weight_format == moe_WEIGHT_ENCODING_GGUF_IQ2_XS ||
        weight_format == moe_WEIGHT_ENCODING_GGUF_IQ2_S ||
        weight_format == moe_WEIGHT_ENCODING_GGUF_IQ3_XXS ||
        weight_format == moe_WEIGHT_ENCODING_GGUF_IQ3_S ||
        weight_format == moe_WEIGHT_ENCODING_GGUF_IQ1_S ||
        weight_format == moe_WEIGHT_ENCODING_GGUF_IQ1_M ||
        weight_format == moe_WEIGHT_ENCODING_LINEAR_Q2 ||
        weight_format == moe_WEIGHT_ENCODING_LINEAR_Q3 ||
        weight_format == moe_WEIGHT_ENCODING_LINEAR_Q4 ||
        weight_format == moe_WEIGHT_ENCODING_LINEAR_Q8;
}

static void moe_trace_tensor_dot_unsupported(const moe_tensor_record* rec, int weight_format) {
    static std::atomic<uint32_t> logged{0};
    const uint32_t n = logged.fetch_add(1, std::memory_order_relaxed);
    if (n >= 32u) {
        return;
    }
    moe_profile_log(
        "tensor_dot_unsupported",
        "format=%d layer=%u expert=%u proj=%u rows=%u cols=%u row_bytes=%llu",
        weight_format,
        rec ? rec->info.layer : 0u,
        rec ? rec->info.expert : 0u,
        rec ? (uint32_t)rec->info.proj : 0u,
        rec ? rec->info.rows : 0u,
        rec ? rec->info.cols : 0u,
        rec ? (unsigned long long)rec->weight_row_bytes : 0ull);
}

static uint64_t moe_tensor_weight_format_row_bytes_for_cols(
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

static int moe_effective_tensor_weight_format_for_row_bytes(
    const moe_tensor_record* rec,
    int weight_format
) {
    if (!rec || !rec->weight_row_bytes ||
        weight_format != moe_WEIGHT_ENCODING_MXFP4) {
        return weight_format;
    }
    if (rec->weight_row_bytes == moe_tensor_weight_format_row_bytes_for_cols((uint8_t)weight_format, rec->info.cols)) {
        return weight_format;
    }
    static const uint8_t candidates[] = {
        moe_WEIGHT_ENCODING_GGUF_Q8_0,
        moe_WEIGHT_ENCODING_GGUF_Q8_1,
        moe_WEIGHT_ENCODING_GGUF_Q8_K,
        moe_WEIGHT_ENCODING_GGUF_Q4_0,
        moe_WEIGHT_ENCODING_GGUF_Q4_1,
        moe_WEIGHT_ENCODING_GGUF_Q5_0,
        moe_WEIGHT_ENCODING_GGUF_Q5_1,
        moe_WEIGHT_ENCODING_GGUF_Q2_K,
        moe_WEIGHT_ENCODING_GGUF_Q3_K,
        moe_WEIGHT_ENCODING_GGUF_Q4_K,
        moe_WEIGHT_ENCODING_GGUF_Q5_K,
        moe_WEIGHT_ENCODING_GGUF_Q6_K,
        moe_WEIGHT_ENCODING_GGUF_IQ2_XXS,
        moe_WEIGHT_ENCODING_GGUF_IQ2_XS,
        moe_WEIGHT_ENCODING_GGUF_IQ2_S,
        moe_WEIGHT_ENCODING_GGUF_IQ3_XXS,
        moe_WEIGHT_ENCODING_GGUF_IQ3_S,
        moe_WEIGHT_ENCODING_GGUF_IQ4_NL,
        moe_WEIGHT_ENCODING_GGUF_IQ4_XS,
        moe_WEIGHT_ENCODING_GGUF_IQ1_S,
        moe_WEIGHT_ENCODING_GGUF_IQ1_M,
        moe_WEIGHT_ENCODING_GGUF_TQ1_0,
        moe_WEIGHT_ENCODING_GGUF_TQ2_0
    };
    int match = weight_format;
    uint32_t matches = 0;
    for (uint8_t candidate : candidates) {
        if (rec->weight_row_bytes == moe_tensor_weight_format_row_bytes_for_cols(candidate, rec->info.cols)) {
            match = (int)candidate;
            ++matches;
        }
    }
    if (matches == 1u) {
        return match;
    }
    return weight_format;
}

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
    const int declared_weight_format = tensor_weight_format(*rec);
    const int weight_format = moe_effective_tensor_weight_format_for_row_bytes(rec, declared_weight_format);
    if (!moe_dot_weight_format_supported(weight_format)) {
        moe_trace_tensor_dot_unsupported(rec, weight_format);
        return 0;
    }
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
    if (weight_format == moe_WEIGHT_ENCODING_RAW_BF16) {
        *out_value = moe_dot_raw_bf16_row(packed, x, rec->info.cols);
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
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q4_0) {
        *out_value = moe_dot_gguf_q4_0_row(packed, x, rec->info.cols);
        return 1;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q4_1) {
        *out_value = moe_dot_gguf_q4_1_row(packed, x, rec->info.cols);
        return 1;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q5_0) {
        *out_value = moe_dot_gguf_q5_0_row(packed, x, rec->info.cols);
        return 1;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q8_0) {
        *out_value = moe_dot_gguf_q8_0_row(packed, x, rec->info.cols);
        return 1;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q8_1) {
        *out_value = moe_dot_gguf_q8_1_row(packed, x, rec->info.cols);
        return 1;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q8_K) {
        *out_value = moe_dot_gguf_q8_k_row(packed, x, rec->info.cols);
        return 1;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_IQ4_NL) {
        *out_value = moe_dot_gguf_iq4_nl_row(packed, x, rec->info.cols);
        return 1;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_IQ4_XS) {
        *out_value = moe_dot_gguf_iq4_xs_row(packed, x, rec->info.cols);
        return 1;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q5_1) {
        *out_value = moe_dot_gguf_q5_1_row(packed, x, rec->info.cols);
        return 1;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q2_K) {
        *out_value = moe_dot_gguf_q2_k_row(packed, x, rec->info.cols);
        return 1;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q3_K) {
        *out_value = moe_dot_gguf_q3_k_row(packed, x, rec->info.cols);
        return 1;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q4_K) {
        *out_value = moe_dot_gguf_q4_k_row(packed, x, rec->info.cols);
        return 1;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q5_K) {
        *out_value = moe_dot_gguf_q5_k_row(packed, x, rec->info.cols);
        return 1;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q6_K) {
        *out_value = moe_dot_gguf_q6_k_row(packed, x, rec->info.cols);
        return 1;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_IQ2_XXS) {
        *out_value = moe_dot_gguf_iq2_xxs_row(packed, x, rec->info.cols);
        return 1;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_IQ2_XS) {
        *out_value = moe_dot_gguf_iq2_xs_row(packed, x, rec->info.cols);
        return 1;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_IQ2_S) {
        *out_value = moe_dot_gguf_iq2_s_row(packed, x, rec->info.cols);
        return 1;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_IQ3_XXS) {
        *out_value = moe_dot_gguf_iq3_xxs_row(packed, x, rec->info.cols);
        return 1;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_IQ3_S) {
        *out_value = moe_dot_gguf_iq3_s_row(packed, x, rec->info.cols);
        return 1;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_IQ1_S) {
        *out_value = moe_dot_gguf_iq1_s_row(packed, x, rec->info.cols);
        return 1;
    }
    if (weight_format == moe_WEIGHT_ENCODING_GGUF_IQ1_M) {
        *out_value = moe_dot_gguf_iq1_m_row(packed, x, rec->info.cols);
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

    const int declared_weight_format = tensor_weight_format(*rec);
    const int weight_format = moe_effective_tensor_weight_format_for_row_bytes(rec, declared_weight_format);
    if (!moe_dot_weight_format_supported(weight_format)) {
        moe_trace_tensor_dot_unsupported(rec, weight_format);
        return 0;
    }
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
        // BUGFIX 283: Check batch_x[b] for null before dereferencing ★
        // Problem: batch_x array is checked for null, but individual batch_x[b] pointers are not
        // If caller passes array with null elements, segfault occurs
        // Solution: Skip null pointers and set output to 0.0f
        if (!batch_x[b]) {
            out_values[b] = 0.0f;
            continue;
        }

        if (weight_format == moe_WEIGHT_ENCODING_RAW_FP16) {
            out_values[b] = moe_dot_raw_fp16_row(packed, batch_x[b], rec->info.cols);
        }
        else if (weight_format == moe_WEIGHT_ENCODING_RAW_BF16) {
            out_values[b] = moe_dot_raw_bf16_row(packed, batch_x[b], rec->info.cols);
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
        else if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q4_0) {
            out_values[b] = moe_dot_gguf_q4_0_row(packed, batch_x[b], rec->info.cols);
        }
        else if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q4_1) {
            out_values[b] = moe_dot_gguf_q4_1_row(packed, batch_x[b], rec->info.cols);
        }
        else if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q5_0) {
            out_values[b] = moe_dot_gguf_q5_0_row(packed, batch_x[b], rec->info.cols);
        }
        else if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q8_0) {
            out_values[b] = moe_dot_gguf_q8_0_row(packed, batch_x[b], rec->info.cols);
        }
        else if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q8_1) {
            out_values[b] = moe_dot_gguf_q8_1_row(packed, batch_x[b], rec->info.cols);
        }
        else if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q8_K) {
            out_values[b] = moe_dot_gguf_q8_k_row(packed, batch_x[b], rec->info.cols);
        }
        else if (weight_format == moe_WEIGHT_ENCODING_GGUF_IQ4_NL) {
            out_values[b] = moe_dot_gguf_iq4_nl_row(packed, batch_x[b], rec->info.cols);
        }
        else if (weight_format == moe_WEIGHT_ENCODING_GGUF_IQ4_XS) {
            out_values[b] = moe_dot_gguf_iq4_xs_row(packed, batch_x[b], rec->info.cols);
        }
        else if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q5_1) {
            out_values[b] = moe_dot_gguf_q5_1_row(packed, batch_x[b], rec->info.cols);
        }
        else if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q2_K) {
            out_values[b] = moe_dot_gguf_q2_k_row(packed, batch_x[b], rec->info.cols);
        }
        else if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q3_K) {
            out_values[b] = moe_dot_gguf_q3_k_row(packed, batch_x[b], rec->info.cols);
        }
        else if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q4_K) {
            out_values[b] = moe_dot_gguf_q4_k_row(packed, batch_x[b], rec->info.cols);
        }
        else if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q5_K) {
            out_values[b] = moe_dot_gguf_q5_k_row(packed, batch_x[b], rec->info.cols);
        }
        else if (weight_format == moe_WEIGHT_ENCODING_GGUF_Q6_K) {
            out_values[b] = moe_dot_gguf_q6_k_row(packed, batch_x[b], rec->info.cols);
        }
        else if (weight_format == moe_WEIGHT_ENCODING_GGUF_IQ2_XXS) {
            out_values[b] = moe_dot_gguf_iq2_xxs_row(packed, batch_x[b], rec->info.cols);
        }
        else if (weight_format == moe_WEIGHT_ENCODING_GGUF_IQ2_XS) {
            out_values[b] = moe_dot_gguf_iq2_xs_row(packed, batch_x[b], rec->info.cols);
        }
        else if (weight_format == moe_WEIGHT_ENCODING_GGUF_IQ2_S) {
            out_values[b] = moe_dot_gguf_iq2_s_row(packed, batch_x[b], rec->info.cols);
        }
        else if (weight_format == moe_WEIGHT_ENCODING_GGUF_IQ3_XXS) {
            out_values[b] = moe_dot_gguf_iq3_xxs_row(packed, batch_x[b], rec->info.cols);
        }
        else if (weight_format == moe_WEIGHT_ENCODING_GGUF_IQ3_S) {
            out_values[b] = moe_dot_gguf_iq3_s_row(packed, batch_x[b], rec->info.cols);
        }
        else if (weight_format == moe_WEIGHT_ENCODING_GGUF_IQ1_S) {
            out_values[b] = moe_dot_gguf_iq1_s_row(packed, batch_x[b], rec->info.cols);
        }
        else if (weight_format == moe_WEIGHT_ENCODING_GGUF_IQ1_M) {
            out_values[b] = moe_dot_gguf_iq1_m_row(packed, batch_x[b], rec->info.cols);
        }
        else if (weight_format == moe_WEIGHT_ENCODING_LINEAR_Q2 ||
                 weight_format == moe_WEIGHT_ENCODING_LINEAR_Q3 ||
                 weight_format == moe_WEIGHT_ENCODING_LINEAR_Q4 ||
                 weight_format == moe_WEIGHT_ENCODING_LINEAR_Q8) {
            out_values[b] = moe_dot_linear_quant_row(*rec, shard, packed, batch_x[b], rec->weight_bits, row);
        }
        else {
            return 0;
        }
    }
    return 1;
}
