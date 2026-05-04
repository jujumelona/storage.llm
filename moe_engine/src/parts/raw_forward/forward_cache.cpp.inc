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
        return 0.0f;
    }
    return moe_dot_raw_row(row_ptr, raw.scalar_encoding, hidden, hidden_count);
}

static int moe_raw_tensor_copy_row_f32(
    const moe_raw_tensor_cache_entry& raw,
    uint32_t row,
    float* out,
    uint32_t out_count
) {
    // BUGFIX 530: Check out_count upper bound to prevent excessive iteration ??    // Problem: out_count can be arbitrarily large from config
    // Loop can run billions of iterations
    // Solution: Add reasonable upper bound check (e.g., 100000)
    if (!out || out_count > 100000) {
        return 0;
    }
    const uint32_t stride_cols = raw.cols ? raw.cols : out_count;
    const uint64_t row_bytes = moe_raw_tensor_row_bytes(raw, stride_cols);
    // BUGFIX 493: row * row_bytes overflow 泥댄겕
    if (!raw.base || !out || row >= raw.rows || row_bytes == 0 ||
        row > UINT64_MAX / row_bytes ||
        (uint64_t)row * row_bytes > raw.bytes ||
        row_bytes > raw.bytes - (uint64_t)row * row_bytes) {
        return 0;
    }
    const uint8_t* row_ptr = raw.base + (uint64_t)row * row_bytes;
    const uint8_t effective_weight_format = moe_effective_weight_format_for_row_bytes(raw, stride_cols);
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_Q8_0) {
        const uint32_t block_cols = 32u;
        const uint32_t block_bytes = 34u;
        uint32_t dst = 0;
        const uint8_t* block = row_ptr;
        while (dst < out_count) {
            uint16_t d16 = 0;
            std::memcpy(&d16, block, sizeof(d16));
            const float d = moe_raw_fp16_to_f32(d16);
            const int8_t* q = reinterpret_cast<const int8_t*>(block + 2u);
            const uint32_t limit = std::min<uint32_t>(block_cols, out_count - dst);
            for (uint32_t i = 0; i < limit; ++i) {
                out[dst + i] = d * (float)q[i];
            }
            dst += limit;
            block += block_bytes;
        }
        return 1;
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_Q8_1) {
        const uint32_t block_cols = 32u;
        const uint32_t block_bytes = 36u;
        uint32_t dst = 0;
        const uint8_t* block = row_ptr;
        while (dst < out_count) {
            uint16_t d16 = 0;
            std::memcpy(&d16, block, sizeof(d16));
            const float d = moe_raw_fp16_to_f32(d16);
            const int8_t* q = reinterpret_cast<const int8_t*>(block + 4u);
            const uint32_t limit = std::min<uint32_t>(block_cols, out_count - dst);
            for (uint32_t i = 0; i < limit; ++i) {
                out[dst + i] = d * (float)q[i];
            }
            dst += limit;
            block += block_bytes;
        }
        return 1;
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_Q8_K) {
        const uint32_t block_cols = 256u;
        const uint32_t block_bytes = 292u;
        uint32_t dst = 0;
        const uint8_t* block = row_ptr;
        while (dst < out_count) {
            float d = 0.0f;
            std::memcpy(&d, block, sizeof(d));
            const int8_t* q = reinterpret_cast<const int8_t*>(block + 4u);
            const uint32_t limit = std::min<uint32_t>(block_cols, out_count - dst);
            for (uint32_t i = 0; i < limit; ++i) {
                out[dst + i] = d * (float)q[i];
            }
            dst += limit;
            block += block_bytes;
        }
        return 1;
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_Q4_0) {
        const uint32_t block_cols = 32u;
        const uint32_t block_bytes = 18u;
        uint32_t dst = 0;
        const uint8_t* block = row_ptr;
        while (dst < out_count) {
            uint16_t d16 = 0;
            std::memcpy(&d16, block, sizeof(d16));
            const float d = moe_raw_fp16_to_f32(d16);
            const uint8_t* qs = block + 2u;
            const uint32_t limit = std::min<uint32_t>(block_cols, out_count - dst);
            for (uint32_t i = 0; i < limit; ++i) {
                const uint8_t byte = qs[i & 15u];
                const uint8_t q = (i < 16u) ? (byte & 0x0fu) : (byte >> 4);
                out[dst + i] = d * (float)((int)q - 8);
            }
            dst += limit;
            block += block_bytes;
        }
        return 1;
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_Q4_1) {
        const uint32_t block_cols = 32u;
        const uint32_t block_bytes = 20u;
        uint32_t dst = 0;
        const uint8_t* block = row_ptr;
        while (dst < out_count) {
            uint16_t d16 = 0;
            uint16_t m16 = 0;
            std::memcpy(&d16, block, sizeof(d16));
            std::memcpy(&m16, block + 2u, sizeof(m16));
            const float d = moe_raw_fp16_to_f32(d16);
            const float m = moe_raw_fp16_to_f32(m16);
            const uint8_t* qs = block + 4u;
            const uint32_t limit = std::min<uint32_t>(block_cols, out_count - dst);
            for (uint32_t i = 0; i < limit; ++i) {
                const uint8_t byte = qs[i & 15u];
                const uint8_t q = (i < 16u) ? (byte & 0x0fu) : (byte >> 4);
                out[dst + i] = d * (float)q + m;
            }
            dst += limit;
            block += block_bytes;
        }
        return 1;
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_Q5_0) {
        const uint32_t block_cols = 32u;
        const uint32_t block_bytes = 22u;
        uint32_t dst = 0;
        const uint8_t* block = row_ptr;
        while (dst < out_count) {
            uint16_t d16 = 0;
            uint32_t qh = 0;
            std::memcpy(&d16, block, sizeof(d16));
            std::memcpy(&qh, block + 2u, sizeof(qh));
            const float d = moe_raw_fp16_to_f32(d16);
            const uint8_t* qs = block + 6u;
            const uint32_t limit = std::min<uint32_t>(block_cols, out_count - dst);
            for (uint32_t i = 0; i < limit; ++i) {
                const uint8_t byte = qs[i & 15u];
                const uint32_t low = (i < 16u) ? (byte & 0x0fu) : (byte >> 4);
                const uint32_t high = (qh >> i) & 1u;
                out[dst + i] = d * (float)((int)(low | (high << 4u)) - 16);
            }
            dst += limit;
            block += block_bytes;
        }
        return 1;
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_Q5_1) {
        const uint32_t block_cols = 32u;
        const uint32_t block_bytes = 24u;
        uint32_t dst = 0;
        const uint8_t* block = row_ptr;
        while (dst < out_count) {
            uint16_t d16 = 0;
            uint16_t m16 = 0;
            uint32_t qh = 0;
            std::memcpy(&d16, block, sizeof(d16));
            std::memcpy(&m16, block + 2u, sizeof(m16));
            std::memcpy(&qh, block + 4u, sizeof(qh));
            const float d = moe_raw_fp16_to_f32(d16);
            const float m = moe_raw_fp16_to_f32(m16);
            const uint8_t* qs = block + 8u;
            const uint32_t limit = std::min<uint32_t>(block_cols, out_count - dst);
            for (uint32_t i = 0; i < limit; ++i) {
                const uint8_t byte = qs[i & 15u];
                const uint32_t low = (i < 16u) ? (byte & 0x0fu) : (byte >> 4);
                const uint32_t high = (qh >> i) & 1u;
                out[dst + i] = d * (float)(low | (high << 4u)) + m;
            }
            dst += limit;
            block += block_bytes;
        }
        return 1;
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_MXFP4) {
        const uint32_t block_cols = 32u;
        const uint32_t block_bytes = 17u;
        uint32_t dst = 0;
        const uint8_t* block = row_ptr;
        while (dst < out_count) {
            const float scale = moe_e8m0_to_f32_half(block[0]);
            const uint8_t* qs = block + 1u;
            const uint32_t limit = std::min<uint32_t>(block_cols, out_count - dst);
            for (uint32_t i = 0; i < limit; ++i) {
                out[dst + i] = moe_mxfp4_value_from_code(moe_mxfp4_code_at(qs, i)) * scale;
            }
            dst += limit;
            block += block_bytes;
        }
        return 1;
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_Q2_K) {
        moe_dequant_gguf_q2_k_row(row_ptr, out, out_count);
        return 1;
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_Q3_K) {
        moe_dequant_gguf_q3_k_row(row_ptr, out, out_count);
        return 1;
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_Q4_K) {
        moe_dequant_gguf_q4_k_row(row_ptr, out, out_count);
        return 1;
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_Q5_K) {
        moe_dequant_gguf_q5_k_row(row_ptr, out, out_count);
        return 1;
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_Q6_K) {
        moe_dequant_gguf_q6_k_row(row_ptr, out, out_count);
        return 1;
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_IQ2_XXS) {
        moe_dequant_gguf_iq2_xxs_row(row_ptr, out, out_count);
        return 1;
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_IQ2_XS) {
        moe_dequant_gguf_iq2_xs_row(row_ptr, out, out_count);
        return 1;
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_IQ2_S) {
        moe_dequant_gguf_iq2_s_row(row_ptr, out, out_count);
        return 1;
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_IQ3_XXS) {
        moe_dequant_gguf_iq3_xxs_row(row_ptr, out, out_count);
        return 1;
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_IQ3_S) {
        moe_dequant_gguf_iq3_s_row(row_ptr, out, out_count);
        return 1;
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_IQ4_NL) {
        moe_dequant_gguf_iq4_nl_row(row_ptr, out, out_count);
        return 1;
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_IQ4_XS) {
        moe_dequant_gguf_iq4_xs_row(row_ptr, out, out_count);
        return 1;
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_IQ1_S) {
        moe_dequant_gguf_iq1_s_row(row_ptr, out, out_count);
        return 1;
    }
    if (effective_weight_format == moe_WEIGHT_ENCODING_GGUF_IQ1_M) {
        moe_dequant_gguf_iq1_m_row(row_ptr, out, out_count);
        return 1;
    }
    if (effective_weight_format != moe_WEIGHT_ENCODING_UNKNOWN &&
        effective_weight_format != moe_WEIGHT_ENCODING_RAW_FP16 &&
        effective_weight_format != moe_WEIGHT_ENCODING_RAW_BF16 &&
        effective_weight_format != moe_WEIGHT_ENCODING_RAW_FP32) {
        moe_trace_raw_dot_unsupported(raw);
        return 0;
    }
    const uint32_t elem = moe_raw_scalar_size(raw.scalar_encoding);
    if (!elem) {
        return 0;
    }
    for (uint32_t i = 0; i < out_count; ++i) {
        out[i] = moe_raw_scalar_to_f32(row_ptr + (uint64_t)i * (uint64_t)elem, raw.scalar_encoding);
    }
    return 1;
}
