static int row_bounds_ok(const moe_tensor_record& rec, uint32_t row);
static int tensor_weight_format(const moe_tensor_record& rec);

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
    // The index codec is authoritative. Row-byte mismatches can be physical
    // stride padding; they must not retag MXFP4 as another 4-bit family.
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
    // BUGFIX 858: Check engine pointer for NULL ★★
    if (!engine || !rec || !x || !out_value ||
        x_count != rec->info.cols || !row_bounds_ok(*rec, row)) {
        return 0;
    }
    if (!shard || !mmap_get_ptr(shard, 0)) {
        return 0;
    }
    if (x_count == 0) {
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
        if (!std::isfinite(*out_value)) {
            return 0;
        }
        return 1;
    }
    if (weight_format == moe_WEIGHT_ENCODING_RAW_BF16) {
        *out_value = moe_dot_raw_bf16_row(packed, x, rec->info.cols);
        if (!std::isfinite(*out_value)) {
            return 0;
        }
        return 1;
    }
    if (weight_format == moe_WEIGHT_ENCODING_RAW_FP32) {
        *out_value = moe_dot_raw_fp32_row(packed, x, rec->info.cols);
        if (!std::isfinite(*out_value)) {
            return 0;
        }
        return 1;
    }
    if (weight_format == moe_WEIGHT_ENCODING_NVFP4) {
        *out_value = moe_dot_fp4_row(*rec, shard, packed, x, engine, row);
        return 1;
    }
    if (weight_format == moe_WEIGHT_ENCODING_MXFP4) {
        *out_value = moe_dot_gguf_mxfp4_row_strided(packed, x, rec->info.cols, rec->weight_row_bytes);
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



typedef float (*moe_simple_row_dot_fn)(const uint8_t*, const float*, uint32_t);

struct moe_tensor_row_fast_view_t {
    const moe_tensor_record* rec = nullptr;
    const mmap_context_t* shard = nullptr;
    const uint8_t* base = nullptr;
    uint64_t shard_size = 0;
    uint64_t base_offset = 0;
    uint64_t row_bytes = 0;
    uint32_t rows = 0;
    uint32_t cols = 0;
    int weight_format = 0;
    moe_simple_row_dot_fn simple_dot = nullptr;
};

static inline void moe_cpu_prefetch_read(const void* p) {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(p, 0, 3);
#elif defined(_MSC_VER)
    _mm_prefetch(reinterpret_cast<const char*>(p), _MM_HINT_T0);
#else
    (void)p;
#endif
}


static inline moe_simple_row_dot_fn moe_cpu_simple_row_dot_for_format(int weight_format) {
    switch (weight_format) {
        case moe_WEIGHT_ENCODING_RAW_FP16: return &moe_dot_raw_fp16_row;
        case moe_WEIGHT_ENCODING_RAW_BF16: return &moe_dot_raw_bf16_row;
        case moe_WEIGHT_ENCODING_RAW_FP32: return &moe_dot_raw_fp32_row;
        // MXFP4 row layout depends on the physical row stride carried by the
        // tensor index.  The simple-dot ABI has no row-bytes argument, so using
        // it would silently decode 16-value JUJU rows as legacy 32-value rows.
        case moe_WEIGHT_ENCODING_MXFP4: return nullptr;
        case moe_WEIGHT_ENCODING_GGUF_Q4_0: return &moe_dot_gguf_q4_0_row;
        case moe_WEIGHT_ENCODING_GGUF_Q4_1: return &moe_dot_gguf_q4_1_row;
        case moe_WEIGHT_ENCODING_GGUF_Q5_0: return &moe_dot_gguf_q5_0_row;
        case moe_WEIGHT_ENCODING_GGUF_Q5_1: return &moe_dot_gguf_q5_1_row;
        case moe_WEIGHT_ENCODING_GGUF_Q8_0: return &moe_dot_gguf_q8_0_row;
        case moe_WEIGHT_ENCODING_GGUF_Q8_1: return &moe_dot_gguf_q8_1_row;
        case moe_WEIGHT_ENCODING_GGUF_Q8_K: return &moe_dot_gguf_q8_k_row;
        case moe_WEIGHT_ENCODING_GGUF_Q2_K: return &moe_dot_gguf_q2_k_row;
        case moe_WEIGHT_ENCODING_GGUF_Q3_K: return &moe_dot_gguf_q3_k_row;
        case moe_WEIGHT_ENCODING_GGUF_Q4_K: return &moe_dot_gguf_q4_k_row;
        case moe_WEIGHT_ENCODING_GGUF_Q5_K: return &moe_dot_gguf_q5_k_row;
        case moe_WEIGHT_ENCODING_GGUF_Q6_K: return &moe_dot_gguf_q6_k_row;
        case moe_WEIGHT_ENCODING_GGUF_IQ4_NL: return &moe_dot_gguf_iq4_nl_row;
        case moe_WEIGHT_ENCODING_GGUF_IQ4_XS: return &moe_dot_gguf_iq4_xs_row;
        case moe_WEIGHT_ENCODING_GGUF_IQ2_XXS: return &moe_dot_gguf_iq2_xxs_row;
        case moe_WEIGHT_ENCODING_GGUF_IQ2_XS: return &moe_dot_gguf_iq2_xs_row;
        case moe_WEIGHT_ENCODING_GGUF_IQ2_S: return &moe_dot_gguf_iq2_s_row;
        case moe_WEIGHT_ENCODING_GGUF_IQ3_XXS: return &moe_dot_gguf_iq3_xxs_row;
        case moe_WEIGHT_ENCODING_GGUF_IQ3_S: return &moe_dot_gguf_iq3_s_row;
        case moe_WEIGHT_ENCODING_GGUF_IQ1_S: return &moe_dot_gguf_iq1_s_row;
        case moe_WEIGHT_ENCODING_GGUF_IQ1_M: return &moe_dot_gguf_iq1_m_row;
        default: return nullptr;
    }
}

static inline const uint8_t* moe_tensor_row_fast_ptr_unchecked(
    const moe_tensor_row_fast_view_t* view,
    uint32_t row
) {
    return view->base + view->base_offset + (uint64_t)row * view->row_bytes;
}

static int moe_tensor_row_fast_view_init(
    moe_pc_engine_t* engine,
    const moe_tensor_record* rec,
    const mmap_context_t* shard,
    uint32_t expected_cols,
    moe_tensor_row_fast_view_t* out
) {
    if (!out) {
        return 0;
    }
    *out = moe_tensor_row_fast_view_t{};
    if (!engine || !rec || !shard || expected_cols == 0 || expected_cols != rec->info.cols) {
        return 0;
    }
    const uint8_t* base = static_cast<const uint8_t*>(mmap_get_ptr(shard, 0));
    const uint64_t shard_size = (uint64_t)mmap_get_size(shard);
    if (!base || shard_size == 0 || rec->weight_row_bytes == 0 || rec->info.rows == 0) {
        return 0;
    }
    const int declared_weight_format = tensor_weight_format(*rec);
    const int weight_format = moe_effective_tensor_weight_format_for_row_bytes(rec, declared_weight_format);
    if (!moe_dot_weight_format_supported(weight_format)) {
        moe_trace_tensor_dot_unsupported(rec, weight_format);
        return 0;
    }
    const uint64_t last_row = rec->info.rows - 1u;
    const uint64_t last_offset = rec->info.weight_byte_offset + last_row * rec->weight_row_bytes;
    if (rec->info.weight_byte_offset > shard_size ||
        last_offset > shard_size ||
        rec->weight_row_bytes > shard_size - last_offset) {
        return 0;
    }
    out->rec = rec;
    out->shard = shard;
    out->base = base;
    out->shard_size = shard_size;
    out->base_offset = rec->info.weight_byte_offset;
    out->row_bytes = rec->weight_row_bytes;
    out->rows = rec->info.rows;
    out->cols = rec->info.cols;
    out->weight_format = weight_format;
    out->simple_dot = moe_cpu_simple_row_dot_for_format(weight_format);
    return 1;
}

static inline const uint8_t* moe_tensor_row_fast_ptr(
    const moe_tensor_row_fast_view_t* view,
    uint32_t row
) {
    if (!view || !view->base || !view->rec || row >= view->rows) {
        return nullptr;
    }
    // Bounds were verified once in moe_tensor_row_fast_view_init or copied from
    // CPUStorageEngine's persistent hot view.  Keep hot-loop pointer lookup to
    // one multiply and one add.
    return moe_tensor_row_fast_ptr_unchecked(view, row);
}

static inline void moe_tensor_row_fast_prefetch(
    const moe_tensor_row_fast_view_t* view,
    uint32_t row
) {
    const uint8_t* p = moe_tensor_row_fast_ptr(view, row);
    if (!p) {
        return;
    }
    moe_cpu_prefetch_read(p);
    if (view->row_bytes > 64u) moe_cpu_prefetch_read(p + 64u);
    if (view->row_bytes > 256u) moe_cpu_prefetch_read(p + 256u);
    if (view->row_bytes > 512u) moe_cpu_prefetch_read(p + 512u);
    if (view->row_bytes > 1024u) moe_cpu_prefetch_read(p + 1024u);
}

static int moe_dot_tensor_fast_view_row(
    moe_pc_engine_t* engine,
    const moe_tensor_row_fast_view_t* view,
    uint32_t row,
    const float* x,
    uint32_t x_count,
    float* out_value
) {
    if (!engine || !view || !view->rec || !x || !out_value || x_count != view->cols) {
        return 0;
    }
    const uint8_t* packed = moe_tensor_row_fast_ptr(view, row);
    if (!packed) {
        return 0;
    }
    if (view->simple_dot) {
        *out_value = view->simple_dot(packed, x, view->cols);
        return std::isfinite(*out_value) ? 1 : 0;
    }
    switch (view->weight_format) {
        case moe_WEIGHT_ENCODING_RAW_FP16:
            *out_value = moe_dot_raw_fp16_row(packed, x, view->cols);
            break;
        case moe_WEIGHT_ENCODING_RAW_BF16:
            *out_value = moe_dot_raw_bf16_row(packed, x, view->cols);
            break;
        case moe_WEIGHT_ENCODING_RAW_FP32:
            *out_value = moe_dot_raw_fp32_row(packed, x, view->cols);
            break;
        case moe_WEIGHT_ENCODING_NVFP4:
            *out_value = moe_dot_fp4_row(*view->rec, view->shard, packed, x, engine, row);
            break;
        case moe_WEIGHT_ENCODING_MXFP4:
            *out_value = moe_dot_gguf_mxfp4_row_strided(packed, x, view->cols, view->row_bytes);
            break;
        case moe_WEIGHT_ENCODING_GGUF_Q4_0:
            *out_value = moe_dot_gguf_q4_0_row(packed, x, view->cols);
            break;
        case moe_WEIGHT_ENCODING_GGUF_Q4_1:
            *out_value = moe_dot_gguf_q4_1_row(packed, x, view->cols);
            break;
        case moe_WEIGHT_ENCODING_GGUF_Q5_0:
            *out_value = moe_dot_gguf_q5_0_row(packed, x, view->cols);
            break;
        case moe_WEIGHT_ENCODING_GGUF_Q5_1:
            *out_value = moe_dot_gguf_q5_1_row(packed, x, view->cols);
            break;
        case moe_WEIGHT_ENCODING_GGUF_Q8_0:
            *out_value = moe_dot_gguf_q8_0_row(packed, x, view->cols);
            break;
        case moe_WEIGHT_ENCODING_GGUF_Q8_1:
            *out_value = moe_dot_gguf_q8_1_row(packed, x, view->cols);
            break;
        case moe_WEIGHT_ENCODING_GGUF_Q8_K:
            *out_value = moe_dot_gguf_q8_k_row(packed, x, view->cols);
            break;
        case moe_WEIGHT_ENCODING_GGUF_Q2_K:
            *out_value = moe_dot_gguf_q2_k_row(packed, x, view->cols);
            break;
        case moe_WEIGHT_ENCODING_GGUF_Q3_K:
            *out_value = moe_dot_gguf_q3_k_row(packed, x, view->cols);
            break;
        case moe_WEIGHT_ENCODING_GGUF_Q4_K:
            *out_value = moe_dot_gguf_q4_k_row(packed, x, view->cols);
            break;
        case moe_WEIGHT_ENCODING_GGUF_Q5_K:
            *out_value = moe_dot_gguf_q5_k_row(packed, x, view->cols);
            break;
        case moe_WEIGHT_ENCODING_GGUF_Q6_K:
            *out_value = moe_dot_gguf_q6_k_row(packed, x, view->cols);
            break;
        case moe_WEIGHT_ENCODING_GGUF_IQ4_NL:
            *out_value = moe_dot_gguf_iq4_nl_row(packed, x, view->cols);
            break;
        case moe_WEIGHT_ENCODING_GGUF_IQ4_XS:
            *out_value = moe_dot_gguf_iq4_xs_row(packed, x, view->cols);
            break;
        case moe_WEIGHT_ENCODING_GGUF_IQ2_XXS:
            *out_value = moe_dot_gguf_iq2_xxs_row(packed, x, view->cols);
            break;
        case moe_WEIGHT_ENCODING_GGUF_IQ2_XS:
            *out_value = moe_dot_gguf_iq2_xs_row(packed, x, view->cols);
            break;
        case moe_WEIGHT_ENCODING_GGUF_IQ2_S:
            *out_value = moe_dot_gguf_iq2_s_row(packed, x, view->cols);
            break;
        case moe_WEIGHT_ENCODING_GGUF_IQ3_XXS:
            *out_value = moe_dot_gguf_iq3_xxs_row(packed, x, view->cols);
            break;
        case moe_WEIGHT_ENCODING_GGUF_IQ3_S:
            *out_value = moe_dot_gguf_iq3_s_row(packed, x, view->cols);
            break;
        case moe_WEIGHT_ENCODING_GGUF_IQ1_S:
            *out_value = moe_dot_gguf_iq1_s_row(packed, x, view->cols);
            break;
        case moe_WEIGHT_ENCODING_GGUF_IQ1_M:
            *out_value = moe_dot_gguf_iq1_m_row(packed, x, view->cols);
            break;
        case moe_WEIGHT_ENCODING_LINEAR_Q2:
        case moe_WEIGHT_ENCODING_LINEAR_Q3:
        case moe_WEIGHT_ENCODING_LINEAR_Q4:
        case moe_WEIGHT_ENCODING_LINEAR_Q8:
            *out_value = moe_dot_linear_quant_row(
                *view->rec, view->shard, packed, x, view->rec->weight_bits, row);
            break;
        default:
            return 0;
    }
    return std::isfinite(*out_value) ? 1 : 0;
}

static int moe_dot_tensor_fast_view_two_rows(
    moe_pc_engine_t* engine,
    const moe_tensor_row_fast_view_t* a,
    const moe_tensor_row_fast_view_t* b,
    uint32_t row,
    const float* x,
    uint32_t x_count,
    float* out_a,
    float* out_b
) {
    if (!engine || !a || !b || !out_a || !out_b || a->cols != b->cols || x_count != a->cols) {
        return 0;
    }
    const uint8_t* pa = moe_tensor_row_fast_ptr(a, row);
    const uint8_t* pb = moe_tensor_row_fast_ptr(b, row);
    if (!pa || !pb) {
        return 0;
    }
    if (a->weight_format == moe_WEIGHT_ENCODING_MXFP4 &&
        b->weight_format == moe_WEIGHT_ENCODING_MXFP4) {
        moe_dot_gguf_mxfp4_two_rows_strided(pa, pb, x, a->cols, a->row_bytes, b->row_bytes, out_a, out_b);
        return std::isfinite(*out_a) && std::isfinite(*out_b) ? 1 : 0;
    }
    return moe_dot_tensor_fast_view_row(engine, a, row, x, x_count, out_a) &&
           moe_dot_tensor_fast_view_row(engine, b, row, x, x_count, out_b);
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
            out_values[b] = moe_dot_gguf_mxfp4_row_strided(packed, batch_x[b], rec->info.cols, rec->weight_row_bytes);
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
