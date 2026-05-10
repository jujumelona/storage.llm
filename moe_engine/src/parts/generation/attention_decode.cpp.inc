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
    moe_profile_log(
        "attn_standard_map",
        "layer=%d pos=%u q=%d k=%d v=%d o=%d q_norm=%d k_norm=%d ms=%.3f",
        layer, position, has_q, has_k, has_v, has_o, has_q_norm, has_k_norm,
        moe_profile_elapsed_ms(map_start_us));
    const uint64_t input_norm_start_us = moe_profile_now_us();
    const int input_norm_supplied = norm_input ? 1 : 0;
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
    moe_profile_log(
        "attn_standard_input_norm",
        "layer=%d pos=%u supplied=%d ms=%.3f",
        layer, position, input_norm_supplied,
        moe_profile_elapsed_ms(input_norm_start_us));
#ifdef STORAGELLM_PPL_DIAG_ENABLED
    storagellm_ppl_diag_print_vec_stats(
        "attn_input", layer, position, 0, "hidden", hidden, hidden_size);
    storagellm_ppl_diag_print_vec_stats(
        "attn_input", layer, position, 0, "attn_input", attn_input, hidden_size);
#endif

    // BUGFIX 164: Parallelize Q, K, V projections ?끸쁾
    // Problem: Q, K, V projections run serially, leaving CPU cores idle
    // Standard attention: Q (7168), K (1024), V (1024) - Q is 7x larger
    // Solution: Fuse Q/K/V into single moe_parallel_rows dispatch
    // Same pattern as BUGFIX 82 for MLA attention
    // Impact: Better CPU utilization, reduced latency
    moe_raw_tensor_cache_entry q_raw{};
    moe_raw_tensor_cache_entry k_raw{};
    moe_raw_tensor_cache_entry v_raw{};
    const uint64_t qkv_start_us = moe_profile_now_us();
    const int has_q_raw = get_cached_raw_tensor(engine, layer, moe_RAW_TENSOR_Q_A_PROJ, &q_raw);
    const int has_k_raw = get_cached_raw_tensor(engine, layer, moe_RAW_TENSOR_KV_A_PROJ_WITH_MQA, &k_raw);
    const int has_v_raw = has_v ? get_cached_raw_tensor(engine, layer, moe_RAW_TENSOR_KV_B_PROJ, &v_raw) : 0;
    const int qkv_parallel_path = has_q_raw && has_k_raw && q_raw.cols == hidden_size && k_raw.cols == hidden_size &&
        (!has_v || (has_v_raw && v_raw.cols == hidden_size));
    const int allow_v_equals_k = !engine || !engine->offload_graph_ir_required.load(std::memory_order_relaxed) ||
        moe_engine_graph_ir_mentions(engine, "v_equals_k") ||
        moe_engine_graph_ir_mentions(engine, "k_equals_v") ||
        moe_engine_graph_ir_mentions(engine, "shared_kv_projection") ||
        moe_engine_graph_ir_mentions(engine, "attention_k_eq_v") ||
        moe_engine_graph_ir_mentions(engine, "raw_k_projection_before_k_norm");

    if (qkv_parallel_path) {
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
        {
            moe_profile_parallel_label_scope rows_label("attn_qkv");
            if (!moe_parallel_rows(engine, total_rows, 0, [&](uint32_t row) {
                if (row < q_dim) {
                    // Q projection
                    s->q_b[row] = moe_raw_tensor_dot_row(q_raw, row, attn_input, hidden_size);
                    if (!std::isfinite(s->q_b[row])) return 0;
                } else if (row < q_dim + k_dim) {
                    // K projection
                    const uint32_t k_row = row - q_dim;
                    s->attn_kv_a[k_row] = moe_raw_tensor_dot_row(k_raw, k_row, attn_input, hidden_size);
                    if (!std::isfinite(s->attn_kv_a[k_row])) return 0;
                } else if (has_v) {
                    // V projection
                    const uint32_t v_row = row - q_dim - k_dim;
                    s->attn_value[v_row] = moe_raw_tensor_dot_row(v_raw, v_row, attn_input, hidden_size);
                    if (!std::isfinite(s->attn_value[v_row])) return 0;
                }
                return 1;
            })) {
                std::fprintf(stderr, "[storagellm attn] parallel Q/K/V projection failed layer=%d\n", layer);
                return 0;
            }
        }
        if (!has_v) {
            if (!allow_v_equals_k) {
                moe_profile_log("attn_v_equals_k_blocked", "layer=%d pos=%u k_dim=%u v_dim=%u", layer, position, k_dim, kv_v_dim);
                return 0;
            }
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
            if (!allow_v_equals_k) {
                moe_profile_log("attn_v_equals_k_blocked", "layer=%d pos=%u k_dim=%u v_dim=%u", layer, position, k_dim, kv_v_dim);
                return 0;
            }
            std::memcpy(s->attn_value.data(), s->attn_kv_a.data(), (size_t)kv_v_dim * sizeof(float));
        }
    }
    moe_profile_log(
        "attn_standard_qkv",
        "layer=%d pos=%u parallel=%d q_rows=%u k_rows=%u v_rows=%u input=%u ms=%.3f",
        layer, position, qkv_parallel_path, q_dim, k_dim, kv_v_dim, hidden_size,
        moe_profile_elapsed_ms(qkv_start_us));
#ifdef STORAGELLM_PPL_DIAG_ENABLED
    storagellm_ppl_diag_qkv_block(
        "after_qkv_proj",
        layer,
        position,
        s->q_b.data(),
        q_dim,
        s->attn_kv_a.data(),
        k_dim,
        s->attn_value.data(),
        kv_v_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        v_head_dim);
#endif
    const uint64_t qkv_norm_start_us = moe_profile_now_us();
    moe_raw_tensor_cache_entry v_norm_shape{};
    const int has_v_norm =
        has_v ? get_cached_raw_tensor(engine, layer, moe_RAW_TENSOR_V_A_LAYER_NORM, &v_norm_shape) : 0;
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
    int v_norm_mode = 0;
    if (has_v_norm) {
        if (!moe_apply_optional_head_rmsnorm_f32(
                engine, layer, moe_RAW_TENSOR_V_A_LAYER_NORM,
                s->attn_value.data(), num_kv_heads, v_head_dim, cfg.rms_norm_eps)) {
            std::fprintf(stderr, "[storagellm attn] weighted v norm failed layer=%d kv_heads=%u v_head_dim=%u\n",
                layer, num_kv_heads, v_head_dim);
            return 0;
        }
        v_norm_mode = 2;
    } else if (moe_layer_uses_unweighted_v_norm(engine, layer)) {
        if (!moe_apply_unweighted_head_rmsnorm_f32(
                s->attn_value.data(), num_kv_heads, v_head_dim, cfg.rms_norm_eps)) {
            std::fprintf(stderr, "[storagellm attn] unweighted v norm failed layer=%d kv_heads=%u v_head_dim=%u\n",
                layer, num_kv_heads, v_head_dim);
            return 0;
        }
        v_norm_mode = 1;
    }
    moe_profile_log(
        "attn_standard_qkv_norm",
        "layer=%d pos=%u q_norm=%d k_norm=%d has_v=%d v_norm=%d v_norm_weighted=%d ms=%.3f",
        layer, position, has_q_norm, has_k_norm, has_v,
        v_norm_mode != 0 ? 1 : 0,
        v_norm_mode == 2 ? 1 : 0,
        moe_profile_elapsed_ms(qkv_norm_start_us));
#ifdef STORAGELLM_PPL_DIAG_ENABLED
    storagellm_ppl_diag_qkv_block(
        "after_qkv_norm",
        layer,
        position,
        s->q_b.data(),
        q_dim,
        s->attn_kv_a.data(),
        k_dim,
        s->attn_value.data(),
        kv_v_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        v_head_dim);
#endif
    const uint64_t rope_start_us = moe_profile_now_us();
    const int rope_rotate_half = moe_standard_rope_uses_rotate_half_cached(engine, cfg);
    const int rope_partial_half_split = (rope_rotate_half && rope_dim < head_dim) ? 1 : 0;
    const int rope_proportional_freq =
        rope_partial_half_split &&
        moe_hybrid_layer_is_full_attention(engine, layer) &&
        moe_engine_metadata_mentions(engine, "proportional");
    const uint32_t rope_frequency_dim = rope_proportional_freq ? head_dim : rope_dim;
    if (rope_dim >= 2u) {
        static const std::vector<float> empty_rope_table;
        const std::vector<float>& q_cos = use_dynamic_rope ? empty_rope_table : s->rope_q_cos;
        const std::vector<float>& q_sin = use_dynamic_rope ? empty_rope_table : s->rope_q_sin;
        const std::vector<float>& k_cos = use_dynamic_rope ? empty_rope_table : s->rope_k_cos;
        const std::vector<float>& k_sin = use_dynamic_rope ? empty_rope_table : s->rope_k_sin;
        for (uint32_t h = 0; h < num_heads; ++h) {
            moe_standard_rope_apply_inplace_with_full_dim(
                rope_rotate_half,
                s->q_b.data() + (uint64_t)h * head_dim,
                head_dim,
                rope_dim,
                rope_frequency_dim,
                position,
                q_cos,
                q_sin,
                rope_theta);
        }
        for (uint32_t h = 0; h < num_kv_heads; ++h) {
            moe_standard_rope_apply_inplace_with_full_dim(
                rope_rotate_half,
                s->attn_kv_a.data() + (uint64_t)h * head_dim,
                head_dim,
                rope_dim,
                rope_frequency_dim,
                position,
                k_cos,
                k_sin,
                rope_theta);
        }
    }
    moe_profile_log(
        "attn_standard_rope",
        "layer=%d pos=%u rope_dim=%u rope_full_dim=%u freq_dim=%u rotate_half=%d partial_half_split=%d proportional=%d dynamic=%d ms=%.3f",
        layer, position, rope_dim, head_dim,
        rope_frequency_dim,
        rope_rotate_half,
        rope_partial_half_split,
        rope_proportional_freq,
        use_dynamic_rope,
        moe_profile_elapsed_ms(rope_start_us));
#ifdef STORAGELLM_PPL_DIAG_ENABLED
    storagellm_ppl_diag_qkv_block(
        "after_rope",
        layer,
        position,
        s->q_b.data(),
        q_dim,
        s->attn_kv_a.data(),
        k_dim,
        s->attn_value.data(),
        kv_v_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        v_head_dim);
#endif
    const int qkv_configured = engine && moe_engine_qkv_only_required(engine);
    const int qkv_decode = qkv_configured && qkv_is_enabled();
    if (qkv_configured && !qkv_decode) {
        std::fprintf(stderr,
            "[storagellm attn] qkv mode is configured but qkv runtime is disabled layer=%d pos=%u\n",
            layer, position);
        return 0;
    }

    const uint64_t kv_start_us = moe_profile_now_us();
    uint32_t seq_len = 0;
    uint32_t qkv_capacity_tokens = 0;

    float* layer_decode = nullptr;
    uint32_t decode_start = 0;
    if (qkv_decode) {
        qkv_capacity_tokens = moe_qkv_only_ensure_capacity(engine, s, cfg, position, 0);
        if (qkv_capacity_tokens == 0) {
            std::fprintf(stderr,
                "[storagellm attn] qkv-only capacity failed layer=%d pos=%u limit=%u\n",
                layer, position, s ? s->qkv_seq_limit : 0u);
            return 0;
        }
        seq_len = s->qkv_seq_len;
    } else {
        if (moe_engine_qkv_only_required(engine)) {
            moe_qkv_only_plain_touch(engine, "store");
            std::fprintf(stderr,
                "[storagellm attn] plain KV branch reached in QKV-only model layer=%d pos=%u\n",
                layer, position);
            return 0;
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
        seq_len = std::min(position + 1u, s->kv_cache.max_seq_len);
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
    }
    moe_profile_log(
        "attn_standard_kv",
        "layer=%d pos=%u seq=%u decode_start=%u entry=%u backend=%s qkv_cap=%u plain_max_seq=%u ms=%.3f",
        layer, position, seq_len, decode_start, kv_entry_dim,
        qkv_decode ? "qkv_only" : "plain_kv_cache",
        qkv_capacity_tokens,
        qkv_decode ? 0u : s->kv_cache.max_seq_len,
        moe_profile_elapsed_ms(kv_start_us));

    const char* scale_source = nullptr;
    const float scale = moe_standard_attention_score_scale(engine, layer, head_dim, &scale_source);
    const float attn_logit_softcap = moe_attention_logit_softcap_value(engine);
    if (moe_attention_score_scale_violates_qk_norm_contract(engine, layer, scale, scale_source)) {
        std::fprintf(
            stderr,
            "[storagellm attn] qk-norm scale contract violation layer=%d head_dim=%u scale=%.9g source=%s\n",
            layer,
            head_dim,
            (double)scale,
            scale_source ? scale_source : "unknown");
        return 0;
    }
    const uint32_t qkv_head_stride = std::max<uint32_t>(num_heads, num_kv_heads);
    if (qkv_decode &&
        !moe_qkv_prepare_layer_head_storage(
            engine, (uint32_t)layer, qkv_head_stride, head_dim)) {
        std::fprintf(stderr,
            "[storagellm attn] qkv storage prepare failed layer=%d stride=%u head_dim=%u\n",
            layer, qkv_head_stride, head_dim);
        return 0;
    }
#ifdef STORAGELLM_PPL_DIAG_ENABLED
    storagellm_ppl_diag_attention_config(
        "before_attention_heads",
        layer,
        position,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        v_head_dim,
        q_dim,
        k_dim,
        kv_v_dim,
        seq_len,
        qkv_capacity_tokens,
        rope_dim,
        rope_frequency_dim,
        qkv_decode,
        scale,
        attn_logit_softcap);
#endif
    if (qkv_decode) {
        const uint64_t qkv_append_start_us = moe_profile_now_us();
        if (v_head_dim != head_dim) {
            std::fprintf(stderr,
                "[storagellm attn] qkv requires matching key/value dims layer=%d head_dim=%u v_head_dim=%u\n",
                layer, head_dim, v_head_dim);
            return 0;
        }
        // TurboQuant/QJL KV mode must use the same quantize/dequantize contract
        // for every generated token, including the seq_len==1 self-attention
        // case. Bypassing the just-appended current value with the scratch FP32
        // V made PPL evaluate a hybrid path that is not the paper algorithm and
        // hid cache/dequant errors for the first token. Append first, then read
        // it back through the QKV decode path below.
        for (uint32_t kv_h = 0; kv_h < num_kv_heads; ++kv_h) {
            const float* key = s->attn_kv_a.data() + (uint64_t)kv_h * head_dim;
            const float* value = s->attn_value.data() + (uint64_t)kv_h * v_head_dim;
            if (!moe_qkv_append_layer_head_token(
                    engine,
                    (uint32_t)layer,
                    kv_h,
                    qkv_head_stride,
                    qkv_capacity_tokens,
                    seq_len,
                    head_dim,
                    key,
                    value)) {
                std::fprintf(stderr,
                    "[storagellm attn] qkv append failed layer=%d pos=%u kv_head=%u seq=%u cap=%u head_dim=%u\n",
                    layer, position, kv_h, seq_len, qkv_capacity_tokens, head_dim);
                return 0;
            }
        }
        moe_profile_log(
            "attn_standard_qkv_append",
            "layer=%d pos=%u kv_heads=%u seq=%u cap=%u head_dim=%u "
            "k_bits=%d v_bits=%d outlier_channels=%d outlier_bits=%d normal_bits=%d "
            "key_outlier_bits=%d key_normal_bits=%d value_outlier_bits=%d value_normal_bits=%d "
            "current_exact=%d scale=%.6g scale_source=%s attn_softcap=%.6g ms=%.3f",
            layer, position, num_kv_heads, seq_len, qkv_capacity_tokens, head_dim,
            engine->qkv_cfg.k_bits,
            engine->qkv_cfg.v_bits,
            engine->qkv_cfg.outlier_channels,
            engine->qkv_cfg.outlier_bits,
            engine->qkv_cfg.normal_bits,
            engine->qkv_cfg.key_outlier_bits,
            engine->qkv_cfg.key_normal_bits,
            engine->qkv_cfg.value_outlier_bits,
            engine->qkv_cfg.value_normal_bits,
            0,
            (double)scale,
            scale_source ? scale_source : "unknown",
            (double)attn_logit_softcap,
            moe_profile_elapsed_ms(qkv_append_start_us));
    }
    const uint64_t heads_start_us = moe_profile_now_us();
    std::fill(s->attn_value.begin(), s->attn_value.begin() + v_total_dim, 0.0f);
    {
        moe_profile_parallel_label_scope rows_label("attn_heads");
        const uint32_t head_block_rows =
            ((uint64_t)seq_len * (uint64_t)head_dim >= 1024ull && num_heads > 1u) ? 1u : 0u;
        if (!moe_parallel_rows(engine, num_heads, head_block_rows, [&](uint32_t h) {
        const uint32_t kv_h = (uint32_t)(((uint64_t)h * (uint64_t)num_kv_heads) / (uint64_t)num_heads);
        const float* q = s->q_b.data() + (uint64_t)h * head_dim;
        float* out_h = s->attn_value.data() + (uint64_t)h * v_head_dim;
        if (qkv_decode) {
            return moe_pc_engine_attention_decode_layer_head_qkv_f32(
                engine,
                (uint32_t)layer,
                kv_h,
                qkv_head_stride,
                q,
                seq_len,
                qkv_capacity_tokens,
                head_dim,
                scale,
                attn_logit_softcap,
                out_h);
        }
        float max_score = -std::numeric_limits<float>::infinity();
        float denom = 0.0f;
        for (uint32_t p = 0; p < seq_len; ++p) {
            const float* entry = p == position ? s->kv_entry.data() :
                layer_decode + (size_t)p * kv_entry_dim;
            const float* k = entry + (uint64_t)kv_h * head_dim;
            float score = moe_apply_attention_logit_softcap(
                attn_logit_softcap,
                moe_dot_f32(q, k, head_dim) * scale);
            if (!std::isfinite(score)) {
                return 0;
            }
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
        if (denom > 1e-10f && std::isfinite(denom)) {
            moe_scale_f32(out_h, 1.0f / denom, v_head_dim);
        } else {
            return 0;
        }
        return 1;
        })) {
#ifdef STORAGELLM_PPL_DIAG_ENABLED
            storagellm_ppl_diag_attention_heads_failed(
                layer,
                position,
                num_heads,
                num_kv_heads,
                seq_len,
                head_dim,
                v_head_dim,
                qkv_decode,
                scale,
                attn_logit_softcap);
#endif
            return 0;
        }
    }
    moe_profile_log(
        "attn_standard_heads",
        "layer=%d pos=%u heads=%u seq=%u head_dim=%u v_head_dim=%u qkv=%d kv_backend=%s scale=%.6g scale_source=%s attn_softcap=%.6g ms=%.3f",
        layer, position, num_heads, seq_len, head_dim, v_head_dim,
        qkv_decode,
        qkv_decode ? "qkv_layer_head_append" : "plain_kv_cache",
        (double)scale,
        scale_source ? scale_source : "unknown",
        (double)attn_logit_softcap,
        moe_profile_elapsed_ms(heads_start_us));
#ifdef STORAGELLM_PPL_DIAG_ENABLED
    storagellm_ppl_diag_print_vec_stats(
        "after_attention_heads", layer, position, 0, "attn_value_all", s->attn_value.data(), v_total_dim);
#endif

    const uint64_t o_start_us = moe_profile_now_us();
    if (!moe_pc_engine_raw_linear_f32(
            engine, layer, moe_RAW_TENSOR_O_PROJ,
            s->attn_value.data(), v_total_dim, s->attn.data(), hidden_size)) {
        std::fprintf(stderr, "[storagellm attn] o proj failed layer=%d in=%u out=%u\n", layer, v_total_dim, hidden_size);
        return 0;
    }
    moe_profile_log(
        "attn_standard_o_proj",
        "layer=%d pos=%u in=%u out=%u ms=%.3f",
        layer, position, v_total_dim, hidden_size,
        moe_profile_elapsed_ms(o_start_us));
#ifdef STORAGELLM_PPL_DIAG_ENABLED
    storagellm_ppl_diag_print_vec_stats(
        "after_o_proj", layer, position, 0, "attn_output", s->attn.data(), hidden_size);
#endif
    const uint64_t residual_start_us = moe_profile_now_us();
    if (add_residual) {
        for (uint32_t i = 0; i < hidden_size; ++i) {
            if (!std::isfinite(s->attn[i])) {
                return 0;
            }
            hidden[i] += s->attn[i];
        }
    }
    moe_profile_log(
        "attn_standard_total",
        "layer=%d pos=%u add_residual=%d residual_ms=%.3f total_ms=%.3f",
        layer, position, add_residual,
        moe_profile_elapsed_ms(residual_start_us),
        moe_profile_elapsed_ms(total_start_us));
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
