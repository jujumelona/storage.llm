static uint64_t packed_bits_row_bytes(uint32_t cols, uint32_t bits) {
    return ((uint64_t)cols * (uint64_t)bits + 7ull) / 8ull;
}

static uint8_t weight_bits_for_encoding(uint8_t encoding) {
    switch (encoding) {
        case moe_WEIGHT_ENCODING_RAW_FP16:
            return 16;
        case moe_WEIGHT_ENCODING_RAW_FP32:
            return 32;
        case moe_WEIGHT_ENCODING_NVFP4:
        case moe_WEIGHT_ENCODING_MXFP4:
        case moe_WEIGHT_ENCODING_LINEAR_Q4:
        case moe_WEIGHT_ENCODING_GGUF_IQ4:
            return 4;
        case moe_WEIGHT_ENCODING_LINEAR_Q3:
        case moe_WEIGHT_ENCODING_GGUF_IQ3:
            return 3;
        case moe_WEIGHT_ENCODING_LINEAR_Q2:
        case moe_WEIGHT_ENCODING_GGUF_IQ2:
            return 2;
        case moe_WEIGHT_ENCODING_LINEAR_Q8:
            return 8;
        default:
            return 0;
    }
}

// BUGFIX 50: Issue 21 - Weight format inference correctness bug ?…â˜…??// Problem: infer_weight_format returns NVFP4 for any 4-bit format (NVFP4, MXFP4, LINEAR_Q4)
// because they all have identical byte counts. MXFP4 tensors get tagged as NVFP4, causing
// wrong decode table in raw_forward_qkv. This is a critical correctness bug.
// Solution: Accept optional weight_encoding hint from GGUF metadata. When available, use it
// directly instead of byte-count inference. Set weight_format_inferred flag for fallback cases.
static uint8_t infer_weight_format(const moe_tensor_info_t& info, uint8_t gguf_encoding_hint) {
    // Priority 1: Use GGUF metadata if available (offload_weight_encoding)
    if (gguf_encoding_hint != moe_WEIGHT_ENCODING_UNKNOWN &&
        gguf_encoding_hint != 0) {
        return gguf_encoding_hint;
    }

    // Priority 2: Byte-count inference (ambiguous for 4-bit formats)
    const uint64_t elems = (uint64_t)info.rows * info.cols;
    if (elems == 0 || info.rows == 0) {
        return moe_WEIGHT_ENCODING_UNKNOWN;
    }
    if (info.weight_byte_length == elems * 2ull) {
        return moe_WEIGHT_ENCODING_RAW_FP16;
    }
    if (info.weight_byte_length == elems * 4ull) {
        return moe_WEIGHT_ENCODING_RAW_FP32;
    }
    const uint64_t rows = info.rows;
    const uint64_t q8_bytes = rows * packed_bits_row_bytes(info.cols, 8);
    const uint64_t q4_bytes = rows * packed_bits_row_bytes(info.cols, 4);
    const uint64_t q3_bytes = rows * packed_bits_row_bytes(info.cols, 3);
    const uint64_t q2_bytes = rows * packed_bits_row_bytes(info.cols, 2);
    if (info.weight_byte_length == q8_bytes) {
        return moe_WEIGHT_ENCODING_LINEAR_Q8;
    }
    if (info.weight_byte_length == q4_bytes) {
        // AMBIGUOUS: Could be NVFP4, MXFP4, or LINEAR_Q4
        // Default to NVFP4 for backward compatibility, but caller should check
        // weight_format_inferred flag and validate with GGUF metadata
        return moe_WEIGHT_ENCODING_NVFP4;
    }
    if (info.weight_byte_length == q3_bytes) {
        return moe_WEIGHT_ENCODING_LINEAR_Q3;
    }
    if (info.weight_byte_length == q2_bytes) {
        return moe_WEIGHT_ENCODING_LINEAR_Q2;
    }
    return moe_WEIGHT_ENCODING_UNKNOWN;
}

static uint64_t infer_weight_row_bytes(const moe_tensor_info_t& info, uint8_t format) {
    if (format == moe_WEIGHT_ENCODING_RAW_FP16) {
        return (uint64_t)info.cols * 2ull;
    }
    if (format == moe_WEIGHT_ENCODING_RAW_FP32) {
        return (uint64_t)info.cols * 4ull;
    }
    const uint8_t bits = weight_bits_for_encoding(format);
    return bits ? packed_bits_row_bytes(info.cols, bits) : 0;
}

static uint8_t infer_scale_format(const moe_tensor_info_t& info) {
    const uint64_t count = (uint64_t)info.rows * info.groups;
    if (count == 0 || info.scale_byte_length == 0) {
        return 0;
    }
    if (info.scale_byte_length == count * 4ull) {
        return 1;
    }
    if (info.scale_byte_length == count * 2ull) {
        return 2;
    }
    return info.scale_byte_length == count ? 3 : 0;
}

static void finalize_tensor_layout(moe_tensor_record* rec, uint8_t gguf_encoding_hint = moe_WEIGHT_ENCODING_UNKNOWN) {
    if (!rec) {
        return;
    }
    rec->weight_format = infer_weight_format(rec->info, gguf_encoding_hint);
    rec->weight_format_inferred = (gguf_encoding_hint == moe_WEIGHT_ENCODING_UNKNOWN ||
                                    gguf_encoding_hint == 0) ? 1 : 0;
    rec->weight_row_bytes = infer_weight_row_bytes(rec->info, rec->weight_format);
    rec->weight_bits = weight_bits_for_encoding(rec->weight_format);
    rec->weight_block_size = rec->info.group_size > 0 && rec->info.group_size <= 255u ?
        (uint8_t)rec->info.group_size : 0;
    rec->scale_format = infer_scale_format(rec->info);
    rec->cpu_executable =
        rec->weight_format == moe_WEIGHT_ENCODING_RAW_FP16 ||
        rec->weight_format == moe_WEIGHT_ENCODING_RAW_FP32 ||
        rec->weight_format == moe_WEIGHT_ENCODING_NVFP4 ||
        rec->weight_format == moe_WEIGHT_ENCODING_LINEAR_Q2 ||
        rec->weight_format == moe_WEIGHT_ENCODING_LINEAR_Q3 ||
        rec->weight_format == moe_WEIGHT_ENCODING_LINEAR_Q4 ||
        rec->weight_format == moe_WEIGHT_ENCODING_LINEAR_Q8;
    rec->gpu_executable = 0;
    rec->info.weight_encoding = rec->weight_format;
    rec->info.weight_bits = rec->weight_bits;
    rec->info.weight_block_size = rec->weight_block_size;
}

static int row_bounds_ok(const moe_tensor_record& rec, uint32_t row) {
    if (row >= rec.info.rows || rec.info.cols == 0 || rec.weight_row_bytes == 0) {
        return 0;
    }
    const uint64_t rel = (uint64_t)row * rec.weight_row_bytes;
    return rel + rec.weight_row_bytes <= rec.info.weight_byte_length;
}

static int tensor_weight_format(const moe_tensor_record& rec) {
    return rec.weight_format;
}
