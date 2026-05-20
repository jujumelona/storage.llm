// Q8/Q4/Q5 quantization kernels
static float moe_dot_gguf_q8_0_row(
    const uint8_t* packed,
    const float* x,
    uint32_t cols
) {
    if (!packed || !x) {
        return 0.0f;
    }
    float acc = 0.0f;
    const uint32_t block_cols = 32u;
    const uint32_t block_bytes = 34u;
    for (uint32_t start = 0; start < cols; start += block_cols) {
        uint16_t d16 = 0;
        std::memcpy(&d16, packed, sizeof(d16));
        const float d = moe_raw_fp16_to_f32(d16);
        const int8_t* q = reinterpret_cast<const int8_t*>(packed + 2u);
        const uint32_t limit = std::min<uint32_t>(block_cols, cols - start);
        if (!std::isfinite(d)) {
            return std::numeric_limits<float>::quiet_NaN();
        }
#if defined(__AVX2__)
        const __m256 vscale = _mm256_set1_ps(d);
        __m256 vacc = _mm256_setzero_ps();
        uint32_t i = 0;
        for (; i + 7u < limit; i += 8u) {
            const __m256 qv = moe_i8x8_to_ps(q + i);
            const __m256 xv = _mm256_loadu_ps(x + start + i);
            vacc = moe_madd_ps(_mm256_mul_ps(qv, vscale), xv, vacc);
        }
        acc += moe_dot_hsum256_ps(vacc);
        for (; i < limit; ++i) {
            acc += d * (float)q[i] * x[start + i];
        }
#else
        for (uint32_t i = 0; i < limit; ++i) {
            acc += d * (float)q[i] * x[start + i];
        }
#endif
        packed += block_bytes;
    }
    return acc;
}

static float moe_dot_gguf_q8_1_row(
    const uint8_t* packed,
    const float* x,
    uint32_t cols
) {
    if (!packed || !x) {
        return 0.0f;
    }
    float acc = 0.0f;
    const uint32_t block_cols = 32u;
    const uint32_t block_bytes = 36u;
    for (uint32_t start = 0; start < cols; start += block_cols) {
        uint16_t d16 = 0;
        std::memcpy(&d16, packed, sizeof(d16));
        const float d = moe_raw_fp16_to_f32(d16);
        const int8_t* q = reinterpret_cast<const int8_t*>(packed + 4u);
        const uint32_t limit = std::min<uint32_t>(block_cols, cols - start);
        if (!std::isfinite(d)) {
            return std::numeric_limits<float>::quiet_NaN();
        }
#if defined(__AVX2__)
        const __m256 vscale = _mm256_set1_ps(d);
        __m256 vacc = _mm256_setzero_ps();
        uint32_t i = 0;
        for (; i + 7u < limit; i += 8u) {
            const __m256 qv = moe_i8x8_to_ps(q + i);
            const __m256 xv = _mm256_loadu_ps(x + start + i);
            vacc = moe_madd_ps(_mm256_mul_ps(qv, vscale), xv, vacc);
        }
        acc += moe_dot_hsum256_ps(vacc);
        for (; i < limit; ++i) {
            acc += d * (float)q[i] * x[start + i];
        }
#else
        for (uint32_t i = 0; i < limit; ++i) {
            acc += d * (float)q[i] * x[start + i];
        }
#endif
        packed += block_bytes;
    }
    return acc;
}

static float moe_dot_gguf_q8_k_row(
    const uint8_t* packed,
    const float* x,
    uint32_t cols
) {
    if (!packed || !x) {
        return 0.0f;
    }
    float acc = 0.0f;
    const uint32_t block_cols = 256u;
    const uint32_t block_bytes = 292u;
    for (uint32_t start = 0; start < cols; start += block_cols) {
        float d = 0.0f;
        std::memcpy(&d, packed, sizeof(d));
        const int8_t* q = reinterpret_cast<const int8_t*>(packed + 4u);
        const uint32_t limit = std::min<uint32_t>(block_cols, cols - start);
        if (!std::isfinite(d)) {
            return std::numeric_limits<float>::quiet_NaN();
        }
#if defined(__AVX2__)
        const __m256 vscale = _mm256_set1_ps(d);
        __m256 vacc = _mm256_setzero_ps();
        uint32_t i = 0;
        for (; i + 7u < limit; i += 8u) {
            const __m256 qv = moe_i8x8_to_ps(q + i);
            const __m256 xv = _mm256_loadu_ps(x + start + i);
            vacc = moe_madd_ps(_mm256_mul_ps(qv, vscale), xv, vacc);
        }
        acc += moe_dot_hsum256_ps(vacc);
        for (; i < limit; ++i) {
            acc += d * (float)q[i] * x[start + i];
        }
#else
        for (uint32_t i = 0; i < limit; ++i) {
            acc += d * (float)q[i] * x[start + i];
        }
#endif
        packed += block_bytes;
    }
    return acc;
}

static float moe_dot_gguf_q4_0_row(
    const uint8_t* packed,
    const float* x,
    uint32_t cols
) {
    if (!packed || !x) {
        return 0.0f;
    }
    float acc = 0.0f;
    const uint32_t block_cols = 32u;
    const uint32_t block_bytes = 18u;
    for (uint32_t start = 0; start < cols; start += block_cols) {
        uint16_t d16 = 0;
        std::memcpy(&d16, packed, sizeof(d16));
        const float d = moe_raw_fp16_to_f32(d16);
        const uint8_t* qs = packed + 2u;
        const uint32_t limit = std::min<uint32_t>(block_cols, cols - start);
        if (!std::isfinite(d)) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        for (uint32_t i = 0; i < limit; ++i) {
            const uint8_t byte = qs[i & 15u];
            const uint8_t q = (i < 16u) ? (byte & 0x0fu) : (byte >> 4);
            acc += d * ((float)((int)q - 8)) * x[start + i];
        }
        packed += block_bytes;
    }
    return acc;
}

static float moe_dot_gguf_q4_1_row(
    const uint8_t* packed,
    const float* x,
    uint32_t cols
) {
    if (!packed || !x) {
        return 0.0f;
    }
    float acc = 0.0f;
    const uint32_t block_cols = 32u;
    const uint32_t block_bytes = 20u;
    for (uint32_t start = 0; start < cols; start += block_cols) {
        uint16_t d16 = 0;
        uint16_t m16 = 0;
        std::memcpy(&d16, packed, sizeof(d16));
        std::memcpy(&m16, packed + 2u, sizeof(m16));
        const float d = moe_raw_fp16_to_f32(d16);
        const float m = moe_raw_fp16_to_f32(m16);
        const uint8_t* qs = packed + 4u;
        const uint32_t limit = std::min<uint32_t>(block_cols, cols - start);
        if (!std::isfinite(d) || !std::isfinite(m)) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        for (uint32_t i = 0; i < limit; ++i) {
            const uint8_t byte = qs[i & 15u];
            const uint8_t q = (i < 16u) ? (byte & 0x0fu) : (byte >> 4);
            acc += (d * (float)q + m) * x[start + i];
        }
        packed += block_bytes;
    }
    return acc;
}

static float moe_dot_gguf_q5_0_row(
    const uint8_t* packed,
    const float* x,
    uint32_t cols
) {
    if (!packed || !x) {
        return 0.0f;
    }
    float acc = 0.0f;
    const uint32_t block_cols = 32u;
    const uint32_t block_bytes = 22u;
    for (uint32_t start = 0; start < cols; start += block_cols) {
        uint16_t d16 = 0;
        uint32_t qh = 0;
        std::memcpy(&d16, packed, sizeof(d16));
        std::memcpy(&qh, packed + 2u, sizeof(qh));
        const float d = moe_raw_fp16_to_f32(d16);
        const uint8_t* qs = packed + 6u;
        const uint32_t limit = std::min<uint32_t>(block_cols, cols - start);
        if (!std::isfinite(d)) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        for (uint32_t i = 0; i < limit; ++i) {
            const uint8_t byte = qs[i & 15u];
            const uint32_t low = (i < 16u) ? (byte & 0x0fu) : (byte >> 4);
            const uint32_t high = (qh >> i) & 1u;
            acc += d * ((float)((int)(low | (high << 4u)) - 16)) * x[start + i];
        }
        packed += block_bytes;
    }
    return acc;
}

static const int8_t* moe_iq4nl_value_table() {
    static constexpr int8_t kValues[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10,
           1,   13,  25,  38,  53,  69,  89, 113,
    };
    return kValues;
}

static float moe_gguf_iq4_nl_row_impl(const uint8_t* packed, const float* x, float* out, uint32_t cols) {
    float acc = 0.0f;
    const uint32_t block_cols = 32u;
    const uint32_t block_bytes = 18u;
    const int8_t* values = moe_iq4nl_value_table();
    for (uint32_t start = 0; start < cols; start += block_cols) {
        const float d = moe_load_f16le(packed);
        const uint8_t* qs = packed + 2u;
        const uint32_t limit = std::min<uint32_t>(block_cols, cols - start);
        for (uint32_t i = 0; i < limit; ++i) {
            const uint8_t byte = qs[i & 15u];
            const uint8_t code = (i < 16u) ? (byte & 0x0fu) : (byte >> 4);
            const float v = d * (float)values[code];
            moe_accum_or_store(v, x, out, start + i, cols, &acc);
        }
        packed += block_bytes;
    }
    return acc;
}

static float moe_dot_gguf_iq4_nl_row(const uint8_t* packed, const float* x, uint32_t cols) {
    return moe_gguf_iq4_nl_row_impl(packed, x, nullptr, cols);
}

static void moe_dequant_gguf_iq4_nl_row(const uint8_t* packed, float* out, uint32_t cols) {
    (void)moe_gguf_iq4_nl_row_impl(packed, nullptr, out, cols);
}

static float moe_gguf_iq4_xs_row_impl(const uint8_t* packed, const float* x, float* out, uint32_t cols) {
    float acc = 0.0f;
    const uint32_t block_cols = 256u;
    const uint32_t block_bytes = 136u;
    const int8_t* values = moe_iq4nl_value_table();
    for (uint32_t base = 0; base < cols; base += block_cols) {
        const float d = moe_load_f16le(packed);
        const uint16_t scales_h = moe_load_u16le(packed + 2u);
        const uint8_t* scales_l = packed + 4u;
        const uint8_t* qs = packed + 8u;
        const uint32_t limit = std::min<uint32_t>(block_cols, cols - base);
        for (uint32_t ib = 0; ib < 8u && ib * 32u < limit; ++ib) {
            const uint32_t scale_l = (scales_l[ib / 2u] >> (4u * (ib & 1u))) & 0x0fu;
            const uint32_t scale_h = (scales_h >> (2u * ib)) & 0x03u;
            const float dl = d * (float)((int)(scale_l | (scale_h << 4u)) - 32);
            const uint8_t* block_qs = qs + ib * 16u;
            const uint32_t local_limit = std::min<uint32_t>(32u, limit - ib * 32u);
            for (uint32_t j = 0; j < local_limit; ++j) {
                const uint8_t byte = block_qs[j & 15u];
                const uint8_t code = (j < 16u) ? (byte & 0x0fu) : (byte >> 4);
                const float v = dl * (float)values[code];
                moe_accum_or_store(v, x, out, base + ib * 32u + j, cols, &acc);
            }
        }
        packed += block_bytes;
    }
    return acc;
}

static float moe_dot_gguf_iq4_xs_row(const uint8_t* packed, const float* x, uint32_t cols) {
    return moe_gguf_iq4_xs_row_impl(packed, x, nullptr, cols);
}

static void moe_dequant_gguf_iq4_xs_row(const uint8_t* packed, float* out, uint32_t cols) {
    (void)moe_gguf_iq4_xs_row_impl(packed, nullptr, out, cols);
}
static float moe_e8m0_to_f32_half(uint8_t v) {
    if (v == 0u) {
        uint32_t bits = 0x00400000u;
        float out = 0.0f;
        std::memcpy(&out, &bits, sizeof(out));
        return out;
    }
    uint32_t bits = (uint32_t)v << 23u;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

static const float* moe_mxfp4_value_table() {
    static constexpr float kMxfp4Table[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    return kMxfp4Table;
}

static float moe_mxfp4_value_from_code(uint8_t code) {
    return moe_mxfp4_value_table()[code & 0x0fu];
}

static int moe_mxfp4_pairwise_nibble_order_enabled() {
    // GGML MXFP4 stores a 32-value block as sixteen payload bytes:
    // low nibbles decode logical columns 0..15 and high nibbles decode
    // columns 16..31.  JUJU materialization preserves source quant rows, so
    // use that split-16 layout unless a repaired artifact explicitly asks for
    // the older pairwise experiment.
    static int mode = -1;
    if (mode < 0) {
        const char* env = std::getenv("STORAGELLM_MXFP4_NIBBLE_ORDER");
        mode = (env && *env &&
                (std::strcmp(env, "pairwise") == 0 ||
                 std::strcmp(env, "juju_pairwise") == 0)) ? 1 : 0;
    }
    return mode != 0;
}

static uint8_t moe_mxfp4_code_at(const uint8_t* qs, uint32_t index) {
    if (moe_mxfp4_pairwise_nibble_order_enabled()) {
        const uint8_t byte = qs[index >> 1u];
        return (index & 1u) ? (byte >> 4) : (byte & 0x0fu);
    }
    const uint8_t byte = qs[index & 15u];
    return (index < 16u) ? (byte & 0x0fu) : (byte >> 4);
}

struct moe_mxfp4_row_layout_t {
    uint32_t block_cols;
    uint32_t block_bytes;
    uint8_t bytewise_codes;
};

static inline uint64_t moe_mxfp4_row_bytes_for_block_cols(uint32_t cols, uint32_t block_cols) {
    return ((uint64_t)cols + (uint64_t)block_cols - 1ull) / (uint64_t)block_cols * 17ull;
}

static moe_mxfp4_row_layout_t moe_mxfp4_row_layout_for_bytes(uint32_t cols, uint64_t physical_row_bytes) {
    // Standard GGML MXFP4 is a 32-value block: one E8M0 scale byte plus
    // sixteen packed E2M1 bytes.  Some JUJU/GGUF_MXFP4_MOE artifacts carry an
    // expanded physical row layout instead: one scale byte plus sixteen
    // bytewise low-nibble FP4 codes for each 16 logical values.  The tensor
    // index row stride is the only reliable runtime discriminator here; do not
    // force either layout from model names or global feature flags.
    moe_mxfp4_row_layout_t layout{32u, 17u, 0u};
    if (cols == 0 || physical_row_bytes == 0) {
        return layout;
    }
    const uint64_t bytes_16_expanded = moe_mxfp4_row_bytes_for_block_cols(cols, 16u);
    if (physical_row_bytes == bytes_16_expanded) {
        layout.block_cols = 16u;
        layout.bytewise_codes = 1u;
        return layout;
    }
    const uint64_t bytes_32 = moe_mxfp4_row_bytes_for_block_cols(cols, 32u);
    if (physical_row_bytes >= bytes_32) {
        layout.block_cols = 32u;
        layout.bytewise_codes = 0u;
        return layout;
    }
    return layout;
}

static inline uint8_t moe_mxfp4_code_at_layout(
    const uint8_t* qs,
    uint32_t index,
    const moe_mxfp4_row_layout_t& layout
) {
    return layout.bytewise_codes ? (qs[index] & 0x0fu) : moe_mxfp4_code_at(qs, index);
}

static float moe_dot_gguf_mxfp4_row_strided(
    const uint8_t* packed,
    const float* x,
    uint32_t cols,
    uint64_t physical_row_bytes
);

static void moe_dot_gguf_mxfp4_two_rows_strided(
    const uint8_t* packed0,
    const uint8_t* packed1,
    const float* x,
    uint32_t cols,
    uint64_t row_bytes0,
    uint64_t row_bytes1,
    float* out0,
    float* out1
) {
    if (!out0 || !out1) {
        return;
    }
    if (!packed0 || !packed1 || !x || cols == 0) {
        *out0 = 0.0f;
        *out1 = 0.0f;
        return;
    }
    const moe_mxfp4_row_layout_t layout0 = moe_mxfp4_row_layout_for_bytes(cols, row_bytes0);
    const moe_mxfp4_row_layout_t layout1 = moe_mxfp4_row_layout_for_bytes(cols, row_bytes1);
    if (layout0.block_cols != layout1.block_cols ||
        layout0.bytewise_codes != layout1.bytewise_codes) {
        *out0 = moe_dot_gguf_mxfp4_row_strided(packed0, x, cols, row_bytes0);
        *out1 = moe_dot_gguf_mxfp4_row_strided(packed1, x, cols, row_bytes1);
        return;
    }
    const float* table = moe_mxfp4_value_table();
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    uint32_t start = 0;
    while (start < cols) {
        const uint32_t limit0 = std::min<uint32_t>(layout0.block_cols, cols - start);
        const uint32_t limit1 = std::min<uint32_t>(layout1.block_cols, cols - start);
        const uint32_t limit = std::min<uint32_t>(limit0, limit1);
        const float scale0 = moe_e8m0_to_f32_half(packed0[0]);
        const float scale1 = moe_e8m0_to_f32_half(packed1[0]);
        const uint8_t* qs0 = packed0 + 1u;
        const uint8_t* qs1 = packed1 + 1u;
        float block0 = 0.0f;
        float block1 = 0.0f;
        for (uint32_t j = 0; j < limit; ++j) {
            const float xv = x[start + j];
            block0 += table[moe_mxfp4_code_at_layout(qs0, j, layout0)] * xv;
            block1 += table[moe_mxfp4_code_at_layout(qs1, j, layout1)] * xv;
        }
        acc0 += block0 * scale0;
        acc1 += block1 * scale1;
        start += limit;
        packed0 += layout0.block_bytes;
        packed1 += layout1.block_bytes;
    }
    *out0 = acc0;
    *out1 = acc1;
}

static void moe_dot_gguf_mxfp4_two_rows(
    const uint8_t* packed0,
    const uint8_t* packed1,
    const float* x,
    uint32_t cols,
    float* out0,
    float* out1
) {
    moe_dot_gguf_mxfp4_two_rows_strided(packed0, packed1, x, cols, 0ull, 0ull, out0, out1);
}

static float moe_dot_gguf_mxfp4_row_strided(
    const uint8_t* packed,
    const float* x,
    uint32_t cols,
    uint64_t physical_row_bytes
) {
    if (!packed || !x) {
        return 0.0f;
    }
    const moe_mxfp4_row_layout_t layout = moe_mxfp4_row_layout_for_bytes(cols, physical_row_bytes);
    const float* table = moe_mxfp4_value_table();
    float acc = 0.0f;
    uint32_t start = 0;
    while (start < cols) {
        const float scale = moe_e8m0_to_f32_half(packed[0]);
        const uint8_t* qs = packed + 1u;
        const uint32_t limit = std::min<uint32_t>(layout.block_cols, cols - start);
        float block_acc = 0.0f;
        for (uint32_t j = 0; j < limit; ++j) {
            block_acc += table[moe_mxfp4_code_at_layout(qs, j, layout)] * x[start + j];
        }
        acc += block_acc * scale;
        start += limit;
        packed += layout.block_bytes;
    }
    return acc;
}

static float moe_dot_gguf_mxfp4_row(
    const uint8_t* packed,
    const float* x,
    uint32_t cols
) {
    return moe_dot_gguf_mxfp4_row_strided(packed, x, cols, 0ull);
}
