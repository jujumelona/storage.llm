static int moe_apply_rmsnorm_weight_entry_f32(
    const moe_raw_tensor_cache_entry& raw,
    const float* input,
    uint32_t count,
    float* output,
    float eps
) {
    // BUGFIX 214: Division by zero clarification (same pattern as BUGFIX 208)
    // Problem: count == 0 check exists but no comment explaining division safety
    // Solution: Add explicit comment for code clarity
    if (!raw.base || !input || !output || count == 0 || raw.cols != 0 || raw.rows != count) {
        return 0;
    }
    const uint32_t elem = moe_raw_scalar_size(raw.scalar_encoding);
    if (raw.bytes < (uint64_t)count * (uint64_t)elem) {
        return 0;
    }
    double ss = 0.0;
    for (uint32_t i = 0; i < count; ++i) {
        ss += (double)input[i] * (double)input[i];
    }
    // Safe: count > 0 guaranteed by check above
    const float norm = 1.0f / std::sqrt((float)(ss / (double)count) + (eps > 0.0f ? eps : 1.0e-6f));
    for (uint32_t i = 0; i < count; ++i) {
        const float w = moe_raw_scalar_to_f32(raw.base + (uint64_t)i * (uint64_t)elem, raw.scalar_encoding);
        output[i] = input[i] * norm * w;
    }
    return 1;
}

static int moe_apply_optional_head_rmsnorm_f32(
    moe_pc_engine* engine,
    int32_t layer,
    moe_raw_tensor_kind_t kind,
    float* values,
    uint32_t head_count,
    uint32_t head_dim,
    float eps
) {
    if (!engine || !values || head_count == 0 || head_dim == 0) {
        return 0;
    }
    moe_raw_tensor_cache_entry raw{};
    if (!get_cached_raw_tensor(engine, layer, kind, &raw)) {
        return 1;
    }
    if (raw.rows != head_dim || raw.cols != 0) {
        return 0;
    }
    std::vector<float> tmp(head_dim);
    for (uint32_t h = 0; h < head_count; ++h) {
        float* head = values + (uint64_t)h * (uint64_t)head_dim;
        if (!moe_apply_rmsnorm_weight_entry_f32(raw, head, head_dim, tmp.data(), eps)) {
            return 0;
        }
        std::memcpy(head, tmp.data(), (size_t)head_dim * sizeof(float));
    }
    return 1;
}

static int moe_apply_unweighted_head_rmsnorm_f32(
    float* values,
    uint32_t head_count,
    uint32_t head_dim,
    float eps
) {
    // BUGFIX 215: Division by zero clarification
    // Problem: head_dim == 0 check exists but no comment explaining division safety
    // Solution: Add explicit comment for code clarity
    if (!values || head_count == 0 || head_dim == 0) {
        return 0;
    }
    const float norm_eps = eps > 0.0f ? eps : 1.0e-6f;
    for (uint32_t h = 0; h < head_count; ++h) {
        float* head = values + (uint64_t)h * (uint64_t)head_dim;
        double ss = 0.0;
        for (uint32_t i = 0; i < head_dim; ++i) {
            ss += (double)head[i] * (double)head[i];
        }
        // Safe: head_dim > 0 guaranteed by check above
        const float norm = 1.0f / std::sqrt((float)(ss / (double)head_dim) + norm_eps);
        for (uint32_t i = 0; i < head_dim; ++i) {
            head[i] *= norm;
        }
    }
    return 1;
}

static int moe_layer_uses_unweighted_v_norm(
    const moe_pc_engine* engine,
    int32_t layer
) {
    if (!engine || layer < 0) {
        return 0;
    }
    if (moe_engine_graph_ir_mentions(engine, "attn_v_norm") ||
        moe_engine_graph_ir_mentions(engine, "value_norm") ||
        moe_engine_graph_ir_mentions(engine, "v_norm")) {
        return 1;
    }
    if (moe_engine_has_common_tensor_suffix(engine, layer, "attn_v_norm.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "value_norm.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "v_norm.weight")) {
        return 1;
    }
    return moe_engine_has_common_tensor_suffix(engine, layer, "ffn_gate_inp.scale") &&
        (moe_engine_has_common_tensor_suffix(engine, layer, "post_ffw_norm.weight") ||
         moe_engine_has_common_tensor_suffix(engine, layer, "post_ffw_norm_1.weight") ||
         moe_engine_has_common_tensor_suffix(engine, layer, "post_ffw_norm_2.weight") ||
         moe_engine_has_common_tensor_suffix(engine, layer, "layer_output_scale.weight")) ? 1 : 0;
}

static int moe_layer_has_qk_norm_scaled_standard_layout(
    const moe_pc_engine* engine,
    int32_t layer
) {
    if (!engine || layer < 0) {
        return 0;
    }
    const int has_q_norm =
        moe_engine_has_common_tensor_suffix(engine, layer, "attn_q_norm.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "q_norm.weight");
    const int has_k_norm =
        moe_engine_has_common_tensor_suffix(engine, layer, "attn_k_norm.weight") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "k_norm.weight");
    const int has_scaled_block_norms =
        moe_engine_has_common_tensor_suffix(engine, layer, "post_attention_norm.weight") &&
        moe_engine_has_common_tensor_suffix(engine, layer, "post_ffw_norm.weight") &&
        moe_engine_has_common_tensor_suffix(engine, layer, "layer_output_scale.weight");
    return has_q_norm && has_k_norm && has_scaled_block_norms;
}

static float moe_standard_attention_score_scale(
    const moe_pc_engine* engine,
    int32_t layer,
    uint32_t head_dim
) {
    if (head_dim == 0) {
        return 1.0f;
    }
    if (moe_layer_has_qk_norm_scaled_standard_layout(engine, layer)) {
        return 1.0f;
    }
    return 1.0f / std::sqrt((float)head_dim);
}

static void moe_standard_attention_rope_params(
    const moe_pc_engine* engine,
    int32_t layer,
    int has_v,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t* io_rope_dim,
    float* io_rope_theta
) {
    if (!io_rope_dim || !io_rope_theta || head_dim == 0) {
        return;
    }
    if (!moe_layer_has_qk_norm_scaled_standard_layout(engine, layer)) {
        return;
    }
    const int full_attention_layout =
        !has_v ||
        (head_dim > 256u) ||
        (num_heads > 0 && num_kv_heads > 0 && num_kv_heads * 4u <= num_heads);
    if (full_attention_layout) {
        uint32_t partial_dim = (uint32_t)std::round((float)head_dim * 0.25f);
        if (partial_dim < 2u) partial_dim = 2u;
        if (partial_dim > head_dim) partial_dim = head_dim;
        *io_rope_dim = partial_dim;
        *io_rope_theta = 1000000.0f;
    } else {
        *io_rope_dim = head_dim;
        *io_rope_theta = 10000.0f;
    }
}

static int moe_standard_kv_cache_ensure(
    moe_pc_engine* engine,
    moe_decode_scratch* s,
    uint32_t position,
    uint32_t kv_entry_dim
) {
    if (!engine || !s || kv_entry_dim == 0) {
        return 0;
    }
    if (position < s->kv_cache.max_seq_len) {
        return 1;
    }
    try {
        s->kv_cache.ensure_capacity(position);
        moe_kv_cache_track_ram(engine, &s->kv_cache);
    } catch (const std::bad_alloc&) {
        return 0;
    } catch (...) {
        return 0;
    }
    return position < s->kv_cache.max_seq_len ? 1 : 0;
}

static int moe_layer_standard_attention_f32(
    moe_pc_engine* engine,
    int32_t layer,
    moe_decode_scratch* s,
    float* hidden,
    uint32_t hidden_size,
    uint32_t position,
    const moe_attn_config_t& cfg,
    const float* norm_input = nullptr,
    int add_residual = 1
) {
    uint32_t num_heads = cfg.num_heads;
    uint32_t num_kv_heads = cfg.num_kv_heads;
    uint32_t head_dim = cfg.head_dim;
    uint32_t v_head_dim = cfg.v_head_dim ? cfg.v_head_dim : cfg.head_dim;
    moe_raw_tensor_cache_entry q_shape{};
    moe_raw_tensor_cache_entry k_shape{};
    moe_raw_tensor_cache_entry v_shape{};
    moe_raw_tensor_cache_entry o_shape{};
    moe_raw_tensor_cache_entry q_norm_shape{};
    moe_raw_tensor_cache_entry k_norm_shape{};
    const int has_q = get_cached_raw_tensor(engine, layer, moe_RAW_TENSOR_Q_A_PROJ, &q_shape);
    const int has_k = get_cached_raw_tensor(engine, layer, moe_RAW_TENSOR_KV_A_PROJ_WITH_MQA, &k_shape);
    const int has_v = get_cached_raw_tensor(engine, layer, moe_RAW_TENSOR_KV_B_PROJ, &v_shape);
    const int has_o = get_cached_raw_tensor(engine, layer, moe_RAW_TENSOR_O_PROJ, &o_shape);
    const int has_q_norm = get_cached_raw_tensor(engine, layer, moe_RAW_TENSOR_Q_A_LAYER_NORM, &q_norm_shape);
    const int has_k_norm = get_cached_raw_tensor(engine, layer, moe_RAW_TENSOR_KV_A_LAYER_NORM, &k_norm_shape);
    if (!has_q || !has_k || !has_o) {
        std::fprintf(stderr, "[storagellm attn] missing q/k/o layer=%d q=%d k=%d v=%d o=%d\n",
            layer, has_q, has_k, has_v, has_o);
        return 0;
    }
    if (has_q_norm && has_k_norm &&
        q_norm_shape.rows && q_norm_shape.rows == k_norm_shape.rows &&
        q_shape.rows % q_norm_shape.rows == 0 &&
        k_shape.rows % q_norm_shape.rows == 0) {
        head_dim = q_norm_shape.rows;
    } else if (head_dim == 0 && num_heads && q_shape.rows % num_heads == 0) {
        head_dim = q_shape.rows / num_heads;
    }
    if (head_dim && q_shape.rows % head_dim == 0) {
        num_heads = q_shape.rows / head_dim;
    }
    if (head_dim && k_shape.rows % head_dim == 0) {
        num_kv_heads = k_shape.rows / head_dim;
    }
    if (has_v && num_kv_heads && v_shape.rows % num_kv_heads == 0) {
        v_head_dim = v_shape.rows / num_kv_heads;
    } else if (num_heads && o_shape.cols % num_heads == 0) {
        v_head_dim = o_shape.cols / num_heads;
    }
    const uint32_t q_dim = q_shape.rows;
    const uint32_t k_dim = k_shape.rows;
    const uint32_t kv_v_dim = has_v ? v_shape.rows : num_kv_heads * v_head_dim;
    const uint32_t v_total_dim = o_shape.cols ? o_shape.cols : num_heads * v_head_dim;
    const uint32_t kv_entry_dim = k_dim + kv_v_dim;
    const uint32_t base_rope_dim = std::min<uint32_t>(cfg.qk_rope_head_dim ? cfg.qk_rope_head_dim : head_dim, head_dim);
    uint32_t rope_dim = base_rope_dim;
    float rope_theta = cfg.rope_theta;
    moe_standard_attention_rope_params(
        engine, layer, has_v, num_heads, num_kv_heads, head_dim, &rope_dim, &rope_theta);
    const int use_dynamic_rope =
        moe_engine_uses_hybrid_local_global_attention(engine) ||
        rope_dim != base_rope_dim ||
        std::fabs(rope_theta - cfg.rope_theta) > 1.0e-6f;
    moe_profile_log(
        "attn_standard_cfg",
        "layer=%d pos=%u heads=%u kv_heads=%u head_dim=%u v_head_dim=%u q_dim=%u k_dim=%u kv_v_dim=%u v_total=%u rope_dim=%u theta=%.6g has_v=%d dynamic_rope=%d",
        layer,
        position,
        num_heads,
        num_kv_heads,
        head_dim,
        v_head_dim,
        q_dim,
        k_dim,
        kv_v_dim,
        v_total_dim,
        rope_dim,
        (double)rope_theta,
        has_v,
        use_dynamic_rope);
    if (!engine || !s || !hidden || hidden_size == 0 ||
        num_heads == 0 || num_kv_heads == 0 || num_kv_heads > num_heads ||
        head_dim == 0 || v_head_dim == 0 || q_dim == 0 || k_dim == 0 ||
        kv_v_dim == 0 || v_total_dim == 0 || kv_entry_dim == 0 ||
        o_shape.rows != hidden_size || v_total_dim != num_heads * v_head_dim ||
        (!has_v && kv_v_dim != k_dim)) {
        std::fprintf(stderr,
            "[storagellm attn] invalid standard cfg layer=%d hidden=%u heads=%u kv_heads=%u head_dim=%u v_head_dim=%u q_dim=%u k_dim=%u v_dim=%u o=%ux%u has_v=%d\n",
            layer, hidden_size, num_heads, num_kv_heads, head_dim, v_head_dim, q_dim, k_dim, kv_v_dim,
            o_shape.rows, o_shape.cols, has_v);
        return 0;
    }
    const float* attn_input = norm_input;
    if (!attn_input) {
        if (!moe_pc_engine_raw_rmsnorm_f32(
                engine, layer, moe_RAW_TENSOR_INPUT_LAYER_NORM,
                hidden, hidden_size, s->norm.data(), hidden_size, cfg.rms_norm_eps)) {
            std::fprintf(stderr, "[storagellm attn] input rmsnorm failed layer=%d hidden=%u\n", layer, hidden_size);
            return 0;
        }
        attn_input = s->norm.data();
    }

    // BUGFIX 164: Parallelize Q, K, V projections ★★
    // Problem: Q, K, V projections run serially, leaving CPU cores idle
    // Standard attention: Q (7168), K (1024), V (1024) - Q is 7x larger
    // Solution: Fuse Q/K/V into single moe_parallel_rows dispatch
    // Same pattern as BUGFIX 82 for MLA attention
    // Impact: Better CPU utilization, reduced latency
    moe_raw_tensor_cache_entry q_raw{};
    moe_raw_tensor_cache_entry k_raw{};
    moe_raw_tensor_cache_entry v_raw{};
    const int has_q_raw = get_cached_raw_tensor(engine, layer, moe_RAW_TENSOR_Q_A_PROJ, &q_raw);
    const int has_k_raw = get_cached_raw_tensor(engine, layer, moe_RAW_TENSOR_KV_A_PROJ_WITH_MQA, &k_raw);
    const int has_v_raw = has_v ? get_cached_raw_tensor(engine, layer, moe_RAW_TENSOR_KV_B_PROJ, &v_raw) : 0;

    if (has_q_raw && has_k_raw && q_raw.cols == hidden_size && k_raw.cols == hidden_size &&
        (!has_v || (has_v_raw && v_raw.cols == hidden_size))) {
        // BUGFIX 216: Add scratch buffer size validation before parallel Q/K/V
        // Problem: s->q_b, s->attn_kv_a, s->attn_value used without size check
        // Solution: Validate buffer sizes before parallel dispatch
        // Impact: Prevents buffer overflow, improves diagnostics
        if (s->q_b.size() < q_dim || s->attn_kv_a.size() < k_dim ||
            (has_v && s->attn_value.size() < kv_v_dim)) {
            std::fprintf(stderr, "[storagellm attn] scratch buffer size mismatch: "
                "layer=%d, need q_b=%u attn_kv_a=%u attn_value=%u, "
                "have q_b=%zu attn_kv_a=%zu attn_value=%zu\n",
                layer, q_dim, k_dim, kv_v_dim,
                s->q_b.size(), s->attn_kv_a.size(), s->attn_value.size());
            return 0;
        }

        // Parallel path: fuse Q/K/V projections
        const uint32_t total_rows = q_dim + k_dim + (has_v ? kv_v_dim : 0);
        if (!moe_parallel_rows(engine, total_rows, 0, [&](uint32_t row) {
            if (row < q_dim) {
                // Q projection
                s->q_b[row] = moe_raw_tensor_dot_row(q_raw, row, attn_input, hidden_size);
            } else if (row < q_dim + k_dim) {
                // K projection
                const uint32_t k_row = row - q_dim;
                s->attn_kv_a[k_row] = moe_raw_tensor_dot_row(k_raw, k_row, attn_input, hidden_size);
            } else if (has_v) {
                // V projection
                const uint32_t v_row = row - q_dim - k_dim;
                s->attn_value[v_row] = moe_raw_tensor_dot_row(v_raw, v_row, attn_input, hidden_size);
            }
            return 1;
        })) {
            std::fprintf(stderr, "[storagellm attn] parallel Q/K/V projection failed layer=%d\n", layer);
            return 0;
        }
        if (!has_v) {
            std::memcpy(s->attn_value.data(), s->attn_kv_a.data(), (size_t)kv_v_dim * sizeof(float));
        }
    } else {
        // Fallback: serial projections (original path)
        if (!moe_pc_engine_raw_linear_f32(
                engine, layer, moe_RAW_TENSOR_Q_A_PROJ,
                attn_input, hidden_size, s->q_b.data(), q_dim)) {
            std::fprintf(stderr, "[storagellm attn] q proj failed layer=%d in=%u out=%u heads=%u kv_heads=%u head_dim=%u v_head_dim=%u type=%d\n",
                layer, hidden_size, q_dim, num_heads, num_kv_heads, head_dim, v_head_dim, (int)cfg.type);
            return 0;
        }
        if (!moe_pc_engine_raw_linear_f32(
                engine, layer, moe_RAW_TENSOR_KV_A_PROJ_WITH_MQA,
                attn_input, hidden_size, s->attn_kv_a.data(), k_dim)) {
            std::fprintf(stderr, "[storagellm attn] k proj failed layer=%d in=%u out=%u\n", layer, hidden_size, k_dim);
            return 0;
        }
        if (has_v) {
            if (!moe_pc_engine_raw_linear_f32(
                engine, layer, moe_RAW_TENSOR_KV_B_PROJ,
                attn_input, hidden_size, s->attn_value.data(), kv_v_dim)) {
                std::fprintf(stderr, "[storagellm attn] v proj failed layer=%d in=%u out=%u\n", layer, hidden_size, kv_v_dim);
                return 0;
            }
        } else {
            std::memcpy(s->attn_value.data(), s->attn_kv_a.data(), (size_t)kv_v_dim * sizeof(float));
        }
    }
    if (!moe_apply_optional_head_rmsnorm_f32(
            engine, layer, moe_RAW_TENSOR_Q_A_LAYER_NORM,
            s->q_b.data(), num_heads, head_dim, cfg.rms_norm_eps) ||
        !moe_apply_optional_head_rmsnorm_f32(
            engine, layer, moe_RAW_TENSOR_KV_A_LAYER_NORM,
            s->attn_kv_a.data(), num_kv_heads, head_dim, cfg.rms_norm_eps)) {
        std::fprintf(stderr, "[storagellm attn] q/k norm failed layer=%d heads=%u kv_heads=%u head_dim=%u\n",
            layer, num_heads, num_kv_heads, head_dim);
        return 0;
    }
    if (has_v && moe_layer_uses_unweighted_v_norm(engine, layer) &&
        !moe_apply_unweighted_head_rmsnorm_f32(
            s->attn_value.data(), num_kv_heads, v_head_dim, cfg.rms_norm_eps)) {
        std::fprintf(stderr, "[storagellm attn] v norm failed layer=%d kv_heads=%u v_head_dim=%u\n",
            layer, num_kv_heads, v_head_dim);
        return 0;
    }
    if (rope_dim >= 2u) {
        static const std::vector<float> empty_rope_table;
        const std::vector<float>& q_cos = use_dynamic_rope ? empty_rope_table : s->rope_q_cos;
        const std::vector<float>& q_sin = use_dynamic_rope ? empty_rope_table : s->rope_q_sin;
        const std::vector<float>& k_cos = use_dynamic_rope ? empty_rope_table : s->rope_k_cos;
        const std::vector<float>& k_sin = use_dynamic_rope ? empty_rope_table : s->rope_k_sin;
        for (uint32_t h = 0; h < num_heads; ++h) {
            moe_standard_rope_apply_inplace(
                engine,
                cfg,
                s->q_b.data() + (uint64_t)h * head_dim,
                rope_dim,
                position,
                q_cos,
                q_sin,
                rope_theta);
        }
        for (uint32_t h = 0; h < num_kv_heads; ++h) {
            moe_standard_rope_apply_inplace(
                engine,
                cfg,
                s->attn_kv_a.data() + (uint64_t)h * head_dim,
                rope_dim,
                position,
                k_cos,
                k_sin,
                rope_theta);
        }
    }
    if (!moe_standard_kv_cache_ensure(engine, s, position, kv_entry_dim)) {
        std::fprintf(stderr, "[storagellm attn] kv ensure failed layer=%d pos=%u entry=%u\n", layer, position, kv_entry_dim);
        return 0;
    }
    if (s->kv_entry.size() < kv_entry_dim) {
        s->kv_entry.resize(kv_entry_dim);
    }
    std::memcpy(s->kv_entry.data(), s->attn_kv_a.data(), (size_t)k_dim * sizeof(float));
    std::memcpy(s->kv_entry.data() + k_dim, s->attn_value.data(), (size_t)kv_v_dim * sizeof(float));
    s->kv_cache.store_entry((uint32_t)layer, position, s->kv_entry.data(), kv_entry_dim);
    s->kv_cache.seq_len = std::min(position + 1u, s->kv_cache.max_seq_len);
    const uint32_t seq_len = std::min(position + 1u, s->kv_cache.max_seq_len);

    float* layer_decode = nullptr;
    uint32_t decode_start = 0;
    if (position > 0 &&
        !moe_kv_decode_layer_cache_prepare(
            engine, s, (uint32_t)layer, position, &layer_decode, &decode_start)) {
        std::fprintf(stderr, "[storagellm attn] kv decode cache failed layer=%d pos=%u entry=%u\n", layer, position, kv_entry_dim);
        return 0;
    }
    if (position > 0) {
        for (uint32_t p = decode_start; p < position && p < seq_len; ++p) {
            s->kv_cache.load_entry((uint32_t)layer, p, layer_decode + (size_t)p * kv_entry_dim, kv_entry_dim);
        }
        if (s->kv_decoded_up_to.size() > (size_t)layer) {
            s->kv_decoded_up_to[(uint32_t)layer] =
                std::max(s->kv_decoded_up_to[(uint32_t)layer], std::min(position, seq_len));
        }
    }

    const float scale = moe_standard_attention_score_scale(engine, layer, head_dim);
    std::fill(s->attn_value.begin(), s->attn_value.begin() + v_total_dim, 0.0f);
    if (!moe_parallel_rows(engine, num_heads, 0, [&](uint32_t h) {
        const uint32_t kv_h = (uint32_t)(((uint64_t)h * (uint64_t)num_kv_heads) / (uint64_t)num_heads);
        const float* q = s->q_b.data() + (uint64_t)h * head_dim;
        float* out_h = s->attn_value.data() + (uint64_t)h * v_head_dim;
        float max_score = -std::numeric_limits<float>::infinity();
        float denom = 0.0f;
        for (uint32_t p = 0; p < seq_len; ++p) {
            const float* entry = p == position ? s->kv_entry.data() :
                layer_decode + (size_t)p * kv_entry_dim;
            const float* k = entry + (uint64_t)kv_h * head_dim;
            float score = moe_dot_f32(q, k, head_dim) * scale;
            if (score > max_score) {
                const float rescale = std::exp(max_score - score);
                denom *= rescale;
                moe_scale_f32(out_h, rescale, v_head_dim);
                max_score = score;
            }
            const float w = std::exp(score - max_score);
            denom += w;
            const float* v = entry + k_dim + (uint64_t)kv_h * v_head_dim;
            moe_axpy_f32(out_h, v, w, v_head_dim);
        }
        if (denom > 0.0f) {
            moe_scale_f32(out_h, 1.0f / denom, v_head_dim);
        }
        return 1;
    })) {
        return 0;
    }

    if (!moe_pc_engine_raw_linear_f32(
            engine, layer, moe_RAW_TENSOR_O_PROJ,
            s->attn_value.data(), v_total_dim, s->attn.data(), hidden_size)) {
        std::fprintf(stderr, "[storagellm attn] o proj failed layer=%d in=%u out=%u\n", layer, v_total_dim, hidden_size);
        return 0;
    }
    if (add_residual) {
        for (uint32_t i = 0; i < hidden_size; ++i) {
            hidden[i] += s->attn[i];
        }
    }
    return 1;
}

static int moe_layer_attention_f32(
    moe_pc_engine* engine,
    int32_t layer,
    moe_decode_scratch* s,
    float* hidden,
    uint32_t hidden_size,
    uint32_t position
) {
    const moe_attn_config_t cfg = moe_generation_layer_attn_config(engine, layer);
    if (cfg.type != moe_ATTN_TYPE_MLA) {
        return moe_layer_standard_attention_f32(engine, layer, s, hidden, hidden_size, position, cfg);
    }
    const uint32_t cdim = cfg.kv_lora_rank;
    const uint32_t num_heads = cfg.num_heads;
    const uint32_t head_dim = cfg.head_dim;
    const uint32_t nope_dim = cfg.qk_nope_head_dim;
    const uint32_t rope_dim = cfg.qk_rope_head_dim;
    const uint32_t v_head_dim = cfg.v_head_dim;
    const uint32_t q_lora_rank = cfg.q_lora_rank;
    const uint32_t kv_a_dim = moe_attn_kv_a_proj_dim(cfg);
    const uint32_t kv_entry_dim = moe_attn_kv_entry_dim(cfg);
    const uint32_t q_b_dim = moe_attn_q_b_dim(cfg);
    const uint32_t k_nope_total_dim = moe_attn_k_nope_total_dim(cfg);
    const uint32_t v_total_dim = moe_attn_v_total_dim(cfg);
    const uint64_t kv_b_row_bytes = (uint64_t)cdim * 2ull; // bf16
    if (cdim == 0 || num_heads == 0 || head_dim == 0 || nope_dim + rope_dim != head_dim ||
        v_head_dim == 0 || q_lora_rank == 0 || kv_a_dim == 0 || kv_entry_dim == 0 ||
        q_b_dim == 0 || k_nope_total_dim == 0 || v_total_dim == 0) {
        return 0;
    }

    // --- Pre-attention RMSNorm ---
    if (!moe_pc_engine_raw_rmsnorm_f32(
            engine, layer, moe_RAW_TENSOR_INPUT_LAYER_NORM,
            hidden, hidden_size, s->norm.data(), hidden_size, cfg.rms_norm_eps)) {
        return 0;
    }

    // BUGFIX 82: Issue Y - Parallelize Q and KV projections ?★??
    // Problem: q_a_proj and kv_a_proj are independent but run serially, leaving cores idle
    // Solution: Fuse them into a single moe_parallel_rows dispatch
    moe_raw_tensor_cache_entry kv_a_raw{};
    moe_raw_tensor_cache_entry q_a_raw{};
    if (!get_cached_raw_tensor(engine, layer, moe_RAW_TENSOR_KV_A_PROJ_WITH_MQA, &kv_a_raw) ||
        !get_cached_raw_tensor(engine, layer, moe_RAW_TENSOR_Q_A_PROJ, &q_a_raw)) {
        return 0;
    }
    if (q_a_raw.rows < q_lora_rank || q_a_raw.cols != hidden_size ||
        kv_a_raw.rows < kv_a_dim || kv_a_raw.cols != hidden_size) {
        return 0;
    }
    if (!moe_parallel_rows(engine, q_lora_rank + kv_a_dim, 0, [&](uint32_t row) {
        if (row < q_lora_rank) {
            s->q_a[row] = moe_raw_tensor_dot_row(q_a_raw, row, s->norm.data(), hidden_size);
        } else {
            const uint32_t kv_row = row - q_lora_rank;
            s->attn_kv_a[kv_row] = moe_raw_tensor_dot_row(kv_a_raw, kv_row, s->norm.data(), hidden_size);
        }
        return 1;
    })) {
        return 0;
    }

    if (!moe_pc_engine_raw_rmsnorm_f32(
            engine, layer, moe_RAW_TENSOR_KV_A_LAYER_NORM,
            s->attn_kv_a.data(), cdim, s->attn_kv_norm.data(), cdim, cfg.rms_norm_eps)) {
        return 0;
    }
    float* k_rope = s->attn_kv_a.data() + cdim;

    // --- Q path: q_a_norm -> q_b_proj ---
    if (!moe_pc_engine_raw_rmsnorm_f32(
            engine, layer, moe_RAW_TENSOR_Q_A_LAYER_NORM,
            s->q_a.data(), q_lora_rank, s->q_a_norm.data(), q_lora_rank, cfg.rms_norm_eps) ||
        !moe_pc_engine_raw_linear_f32(
            engine, layer, moe_RAW_TENSOR_Q_B_PROJ,
            s->q_a_norm.data(), q_lora_rank, s->q_b.data(), q_b_dim)) {
        return 0;
    }

    // Apply K RoPE before storing the current KV entry. Q RoPE is fused below
    // with Q-side absorption and historical KV decode in one barrier.
    moe_rope_inplace(k_rope, rope_dim, position, s->rope_k_cos, s->rope_k_sin, cfg.rope_theta);

    // BUGFIX 45: Issue 11 - Proactive KV cache pre-allocation ?★??
    // Problem: ensure_capacity triggers demote_cold_until in critical path every time
    // KV cache grows, evicting unrelated MoE experts during token generation.
    // Solution: Pre-allocate full KV cache when RAM budget allows, eliminating
    // reallocation and eviction from the hot path entirely.
    // Background pre-allocation at 75% threshold prevents late-stage realloc spikes.
    const uint32_t seq_len_threshold = (cfg.max_seq_len * 3u) / 4u;
    const bool should_preallocate_now =
        s->kv_cache.max_seq_len < cfg.max_seq_len &&
        position >= seq_len_threshold &&
        moe_generation_should_preallocate_kv_cache(engine, s->kv_cache.num_layers, kv_entry_dim, cfg.max_seq_len);

    if (should_preallocate_now) {
        // Pre-allocate to max capacity to avoid future reallocations
        const uint64_t old_bytes = moe_kv_cache_bytes(s->kv_cache.num_layers, s->kv_cache.max_seq_len, kv_entry_dim);
        const uint64_t new_bytes = moe_kv_cache_bytes(s->kv_cache.num_layers, cfg.max_seq_len, kv_entry_dim);
        // BUGFIX 108: Fix double-counting of old_bytes in KV cache reallocation
        // Old: realloc_peak = old_bytes + new_bytes. Since ram_used already includes old_bytes,
        //      checking ram_used + realloc_peak > ram_budget double-counted old_bytes,
        //      causing aggressive unnecessary eviction of MoE experts from RAM.
        // New: realloc_peak is just new_bytes, because peak usage during realloc is ram_used + new_bytes.
        const uint64_t realloc_peak = new_bytes;
        const uint64_t ram_budget = tier_budget(engine, moe_TIER_RAM);
        if (ram_budget > 0) {
            const uint64_t ram_used = tier_used(engine, moe_TIER_RAM);
            if (ram_used + realloc_peak > ram_budget) {
                std::vector<std::pair<int,int>> to_free;
                demote_cold_until(engine, moe_TIER_RAM, realloc_peak, to_free);
                for (auto& p : to_free) {
                    device_free_expert(engine, p.first, p.second);
                }
            }
        }
        s->kv_cache.ensure_capacity(cfg.max_seq_len - 1u);
        moe_kv_cache_track_ram(engine, &s->kv_cache);
    } else if (position >= s->kv_cache.max_seq_len) {
        // Fallback: incremental growth when pre-allocation not possible
        uint32_t new_cap = s->kv_cache.max_seq_len > 0 ? s->kv_cache.max_seq_len : 1u;
        while (new_cap <= position && new_cap < cfg.max_seq_len) {
            new_cap *= 2u;
        }
        if (new_cap > cfg.max_seq_len) {
            new_cap = cfg.max_seq_len;
        }
        const uint64_t old_bytes = moe_kv_cache_bytes(s->kv_cache.num_layers, s->kv_cache.max_seq_len, kv_entry_dim);
        const uint64_t new_bytes = moe_kv_cache_bytes(s->kv_cache.num_layers, new_cap, kv_entry_dim);
        // BUGFIX 108: Fix double-counting of old_bytes in KV cache reallocation
        // Old: realloc_peak = old_bytes + new_bytes. Since ram_used already includes old_bytes,
        //      checking ram_used + realloc_peak > ram_budget double-counted old_bytes,
        //      causing aggressive unnecessary eviction of MoE experts from RAM.
        // New: realloc_peak is just new_bytes, because peak usage during realloc is ram_used + new_bytes.
        const uint64_t realloc_peak = new_bytes;
        const uint64_t ram_budget = tier_budget(engine, moe_TIER_RAM);
        if (ram_budget > 0) {
            const uint64_t ram_used = tier_used(engine, moe_TIER_RAM);
            if (ram_used + realloc_peak > ram_budget) {
                std::vector<std::pair<int,int>> to_free;
                demote_cold_until(engine, moe_TIER_RAM, realloc_peak, to_free);
                for (auto& p : to_free) {
                    device_free_expert(engine, p.first, p.second);
                }
            }
        }
        // BUGFIX 65: Issue 9 - Handle ensure_capacity failure ?★??
        // Problem: ensure_capacity can throw std::bad_alloc if RAM exhausted
        // Exception crosses C API boundary ??undefined behavior
        // Or demote succeeds but another thread allocates ??capacity fails ??KV entry lost
        // Solution: Try-catch around ensure_capacity, return 0 on failure
        // Impact: Explicit error handling instead of silent data corruption
        try {
            s->kv_cache.ensure_capacity(position);
            moe_kv_cache_track_ram(engine, &s->kv_cache);
        } catch (const std::bad_alloc&) {
            fprintf(stderr, "[storageLLM ERROR] KV cache allocation failed at position %u, "
                "layer %d. RAM exhausted. Aborting generation.\n", position, layer);
            return 0;
        } catch (...) {
            fprintf(stderr, "[storageLLM ERROR] KV cache allocation failed at position %u, "
                "layer %d. Unknown exception. Aborting generation.\n", position, layer);
            return 0;
        }
    }
    // BUGFIX 65: Issue 9 - Verify capacity before store ?★??
    // Problem: If ensure_capacity failed silently, position >= max_seq_len
    // store_entry is skipped ??KV entry lost ??attention corrupted
    // Solution: Explicit check, return error if capacity insufficient
    if (position >= s->kv_cache.max_seq_len) {
        fprintf(stderr, "[storageLLM ERROR] KV cache capacity insufficient: "
            "position %u >= max_seq_len %u, layer %d. Aborting generation.\n",
            position, s->kv_cache.max_seq_len, layer);
        return 0;
    }
    if (position < s->kv_cache.max_seq_len) {
        // Build float32 entry, then store as bf16
        if (s->kv_entry.size() < kv_entry_dim) {
            s->kv_entry.resize(kv_entry_dim);
        }
        std::memcpy(s->kv_entry.data(), s->attn_kv_norm.data(), cdim * sizeof(float));
        std::memcpy(s->kv_entry.data() + cdim, k_rope, rope_dim * sizeof(float));
        s->kv_cache.store_entry((uint32_t)layer, position, s->kv_entry.data(), kv_entry_dim);
    }
    s->kv_cache.seq_len = std::min(position + 1u, s->kv_cache.max_seq_len);
    const uint32_t seq_len = std::min(position + 1u, s->kv_cache.max_seq_len);

    // --- Cached W_kv_b for KV absorption ---
    moe_raw_tensor_cache_entry kv_b_raw{};
    if (!get_cached_raw_tensor(engine, layer, moe_RAW_TENSOR_KV_B_PROJ, &kv_b_raw) ||
        kv_b_raw.cols != cdim || kv_b_raw.rows < (k_nope_total_dim + v_total_dim)) {
        return 0;
    }
    const uint8_t* kv_b_base = kv_b_raw.base;
    if (!kv_b_base) return 0;

    // Q RoPE, Q-side absorption, and historical KV decode are independent.
    // Run them in one flat dispatch to avoid three separate row-worker barriers.
    std::fill(s->q_absorbed.begin(), s->q_absorbed.end(), 0.0f);
    const uint32_t decoded_kv_count = position < seq_len ? position : seq_len;
    float* layer_decode = nullptr;
    uint32_t decode_start = 0;
    if (decoded_kv_count > 0 &&
        !moe_kv_decode_layer_cache_prepare(
            engine, s, (uint32_t)layer, decoded_kv_count, &layer_decode, &decode_start)) {
        return 0;
    }
    const uint32_t decode_delta = decoded_kv_count > decode_start ? decoded_kv_count - decode_start : 0u;
    const uint32_t fused_rows = num_heads + decode_delta;
    if (fused_rows > 0) {
        if (!moe_parallel_rows(engine, fused_rows, 0, [&](uint32_t row) {
            if (row < num_heads) {
                const uint32_t h = row;
                moe_rope_inplace(
                    s->q_b.data() + h * head_dim + nope_dim,
                    rope_dim,
                    position,
                    s->rope_q_cos,
                    s->rope_q_sin
                );
                float* qa_h = s->q_absorbed.data() + h * cdim;
                // BUGFIX 292: Check h * cdim doesn't overflow before accessing ★
                if ((uint64_t)h * (uint64_t)cdim > s->q_absorbed.size()) {
                    return 0;
                }
                for (uint32_t d = 0; d < nope_dim; ++d) {
                    const float qv = s->q_b[h * head_dim + d];
                    if (std::abs(qv) < 1e-12f) continue;
                    const uint8_t* w_row = kv_b_base + kv_b_row_bytes * (h * nope_dim + d);
                    moe_axpy_bf16_row_f32(qa_h, w_row, qv, cdim);
                }
            } else {
                const uint32_t p = decode_start + (row - num_heads);
                float* decode = layer_decode + (size_t)p * kv_entry_dim;
                s->kv_cache.load_entry((uint32_t)layer, p, decode, kv_entry_dim);
            }
            return 1;
        })) {
            return 0;
        }
        if (decoded_kv_count > 0) {
            s->kv_decoded_up_to[(uint32_t)layer] = std::max(
                s->kv_decoded_up_to[(uint32_t)layer],
                decoded_kv_count);
        }
    }

    // --- Online softmax: head-outer (parallel), position-inner ---
    // Each head has independent running max/denom/c_weighted; decoded KV is read-only shared.

    // BUGFIX 70: Issue 8 - Remove QKV dead code stub ?★??
    // Problem: BUGFIX 47 added qkv_is_enabled() branch but lambda does nothing (return 1)
    // Standard softmax path always executes ??no memory savings, only overhead
    // QKV integration requires qkv_state_t* in moe_decode_scratch (not implemented)
    // Solution: Remove misleading stub until full QKV state integration is complete
    // When implementing: Add qkv_state_t* to scratch, initialize in generation init,
    // call qkv_attention_decode(engine, h, q_h, qkv_state, decoded_kv_count, out_h)
    // Status: Stub removed, TODO documented for future implementation

    // Standard path: bf16 KV cache with full materialization
    const float scale = 1.0f / std::sqrt((float)head_dim);
    std::fill(s->c_weighted.begin(), s->c_weighted.end(), 0.0f);
    s->head_max.assign(num_heads, -1e30f);
    s->head_denom.assign(num_heads, 0.0f);

    moe_parallel_rows(engine, num_heads, 0, [&](uint32_t h) {
        const float* qa_h = s->q_absorbed.data() + h * cdim;
        const float* q_rope_h = s->q_b.data() + h * head_dim + nope_dim;
        float* cw_h = s->c_weighted.data() + h * cdim;
        float h_max = -1e30f;
        float h_denom = 0.0f;

        // BUGFIX 42: Fix softmax loop boundary to prevent out-of-bounds access ?★??
        // Old: loop_end = position < max_seq_len ? seq_len : seq_len + 1
        //      When position >= max_seq_len, loop_end = seq_len + 1
        //      But layer_decode size = decoded_kv_count = min(position, seq_len)
        //      ??p = seq_len accesses layer_decode[seq_len] which is out of bounds
        // New: loop_end = seq_len (always), current token handled separately in loop
        // The condition (p == position || (position >= max_seq_len && p == seq_len))
        // already handles current token correctly, no need for seq_len + 1
        uint32_t loop_end = seq_len;
        for (uint32_t p = 0; p < loop_end; ++p) {
            const float* c_kv;
            const float* p_k_rope;
            if (p == position || (position >= s->kv_cache.max_seq_len && p == seq_len)) {
                c_kv = s->attn_kv_norm.data();
                p_k_rope = k_rope;
            } else {
                const float* decode = layer_decode + (size_t)p * kv_entry_dim;
                c_kv = decode;
                p_k_rope = decode + cdim;
            }

            float score = moe_dot_f32(qa_h, c_kv, cdim);
            score += moe_dot_f32(q_rope_h, p_k_rope, rope_dim);
            score *= scale;

            if (score > h_max) {
                const float rescale = std::exp(h_max - score);
                h_denom *= rescale;
                moe_scale_f32(cw_h, rescale, cdim);
                h_max = score;
            }
            const float w = std::exp(score - h_max);
            h_denom += w;
            moe_axpy_f32(cw_h, c_kv, w, cdim);
        }

        // Normalize
        if (h_denom > 0.0f) {
            const float inv = 1.0f / h_denom;
            moe_scale_f32(cw_h, inv, cdim);
        }
        s->head_max[h] = h_max;
        s->head_denom[h] = h_denom;
        return 1;
    });

    // --- V-side absorption: head-parallel expansion ---
    const uint8_t* kv_b_v_base = kv_b_base + kv_b_row_bytes * k_nope_total_dim;
    moe_parallel_rows(engine, num_heads, 0, [&](uint32_t h) {
        float* out_h = s->attn_value.data() + h * v_head_dim;
        const float* cw_h = s->c_weighted.data() + h * cdim;
        for (uint32_t d = 0; d < v_head_dim; ++d) {
            const uint8_t* w_row = kv_b_v_base + kv_b_row_bytes * (h * v_head_dim + d);
            out_h[d] = moe_dot_bf16_row(w_row, cw_h, cdim);
        }
        return 1;
    });

    // --- O projection: attention value width -> hidden_size ---
    if (!moe_pc_engine_raw_linear_f32(
            engine, layer, moe_RAW_TENSOR_O_PROJ,
            s->attn_value.data(), v_total_dim, s->attn.data(), hidden_size)) {
        return 0;
    }
    for (uint32_t i = 0; i < hidden_size; ++i) {
        hidden[i] += s->attn[i];
    }
    return 1;
}
