#ifndef STORAGELLM_PPL_DIAG_HOOKS_CPP_INC
#define STORAGELLM_PPL_DIAG_HOOKS_CPP_INC

#define STORAGELLM_PPL_DIAG_ENABLED 1

static int storagellm_ppl_diag_verbose_enabled() {
    static int cached = -1;
    if (cached >= 0) {
        return cached;
    }
    const char* env = std::getenv("STORAGELLM_PPL_DIAG_VERBOSE");
    if (!env || !*env) {
        cached = 0;
        return cached;
    }
    cached =
        std::strcmp(env, "0") != 0 &&
        std::strcmp(env, "false") != 0 &&
        std::strcmp(env, "FALSE") != 0 &&
        std::strcmp(env, "off") != 0 &&
        std::strcmp(env, "OFF") != 0;
    return cached;
}

struct storagellm_ppl_diag_vec_stats {
    uint32_t count = 0;
    uint32_t finite_count = 0;
    uint32_t bad_index = UINT32_MAX;
    float bad_value = 0.0f;
    float min_value = 0.0f;
    float max_value = 0.0f;
    float max_abs = 0.0f;
    double sum = 0.0;
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    float first[4] = {0.0f, 0.0f, 0.0f, 0.0f};
};

static storagellm_ppl_diag_vec_stats storagellm_ppl_diag_collect_vec(
    const float* data,
    uint32_t count
) {
    storagellm_ppl_diag_vec_stats st{};
    st.count = count;
    if (!data || count == 0) {
        return st;
    }
    for (uint32_t i = 0; i < count && i < 4u; ++i) {
        st.first[i] = data[i];
    }
    for (uint32_t i = 0; i < count; ++i) {
        const float v = data[i];
        if (!std::isfinite(v)) {
            if (st.bad_index == UINT32_MAX) {
                st.bad_index = i;
                st.bad_value = v;
            }
            continue;
        }
        if (st.finite_count == 0) {
            st.min_value = v;
            st.max_value = v;
        } else {
            st.min_value = std::min(st.min_value, v);
            st.max_value = std::max(st.max_value, v);
        }
        const float av = (float)std::fabs((double)v);
        st.max_abs = std::max(st.max_abs, av);
        st.sum += (double)v;
        st.sum_abs += (double)av;
        st.sum_sq += (double)v * (double)v;
        ++st.finite_count;
    }
    return st;
}

static void storagellm_ppl_diag_print_vec_stats(
    const char* tag,
    int32_t layer,
    uint32_t position,
    uint32_t index,
    const char* name,
    const float* data,
    uint32_t count
) {
    if (!storagellm_ppl_diag_verbose_enabled()) {
        return;
    }
    const storagellm_ppl_diag_vec_stats st = storagellm_ppl_diag_collect_vec(data, count);
    const double denom = st.finite_count ? (double)st.finite_count : 1.0;
    std::fprintf(stderr,
        "[storagellm ppl_diag] vec tag=%s layer=%d pos=%u index=%u name=%s ptr=%p "
        "count=%u finite=%u bad=%u bad_value=%.9g min=%.9g max=%.9g "
        "mean=%.9g mean_abs=%.9g rms=%.9g max_abs=%.9g "
        "first4=[%.9g,%.9g,%.9g,%.9g]\n",
        tag ? tag : "unknown",
        layer,
        position,
        index,
        name ? name : "unknown",
        (const void*)data,
        st.count,
        st.finite_count,
        st.bad_index,
        (double)st.bad_value,
        (double)st.min_value,
        (double)st.max_value,
        st.sum / denom,
        st.sum_abs / denom,
        std::sqrt(st.sum_sq / denom),
        (double)st.max_abs,
        (double)st.first[0],
        (double)st.first[1],
        (double)st.first[2],
        (double)st.first[3]);
}

static void storagellm_ppl_diag_attention_config(
    const char* tag,
    int32_t layer,
    uint32_t position,
    uint32_t hidden_size,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t v_head_dim,
    uint32_t q_dim,
    uint32_t k_dim,
    uint32_t kv_v_dim,
    uint32_t seq_len,
    uint32_t capacity_tokens,
    uint32_t rope_dim,
    uint32_t rope_frequency_dim,
    int qkv_decode,
    float scale,
    float attn_logit_softcap
) {
    if (!storagellm_ppl_diag_verbose_enabled()) {
        return;
    }
    std::fprintf(stderr,
        "[storagellm ppl_diag] attention_config tag=%s layer=%d pos=%u hidden=%u "
        "heads=%u kv_heads=%u head_dim=%u v_head_dim=%u q_dim=%u k_dim=%u v_dim=%u "
        "seq=%u cap=%u rope_dim=%u rope_freq_dim=%u qkv=%d scale=%.9g softcap=%.9g\n",
        tag ? tag : "unknown",
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
        capacity_tokens,
        rope_dim,
        rope_frequency_dim,
        qkv_decode,
        (double)scale,
        (double)attn_logit_softcap);
}

static void storagellm_ppl_diag_qkv_block(
    const char* tag,
    int32_t layer,
    uint32_t position,
    const float* q,
    uint32_t q_dim,
    const float* k,
    uint32_t k_dim,
    const float* v,
    uint32_t v_dim,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t v_head_dim
) {
    if (!storagellm_ppl_diag_verbose_enabled()) {
        return;
    }
    storagellm_ppl_diag_print_vec_stats(tag, layer, position, 0, "q_all", q, q_dim);
    storagellm_ppl_diag_print_vec_stats(tag, layer, position, 0, "k_all", k, k_dim);
    storagellm_ppl_diag_print_vec_stats(tag, layer, position, 0, "v_all", v, v_dim);
    const uint32_t q_heads_to_log = std::min<uint32_t>(num_heads, 4u);
    for (uint32_t h = 0; h < q_heads_to_log; ++h) {
        storagellm_ppl_diag_print_vec_stats(
            tag, layer, position, h, "q_head", q + (size_t)h * (size_t)head_dim, head_dim);
    }
    const uint32_t kv_heads_to_log = std::min<uint32_t>(num_kv_heads, 4u);
    for (uint32_t h = 0; h < kv_heads_to_log; ++h) {
        storagellm_ppl_diag_print_vec_stats(
            tag, layer, position, h, "k_head", k + (size_t)h * (size_t)head_dim, head_dim);
        storagellm_ppl_diag_print_vec_stats(
            tag, layer, position, h, "v_head", v + (size_t)h * (size_t)v_head_dim, v_head_dim);
    }
}

static void storagellm_ppl_diag_scan_vector(
    const float* data,
    uint32_t count,
    uint32_t* bad_index,
    float* bad_value,
    float* max_abs
) {
    if (bad_index) *bad_index = UINT32_MAX;
    if (bad_value) *bad_value = 0.0f;
    if (max_abs) *max_abs = 0.0f;
    if (!data || count == 0) {
        return;
    }
    for (uint32_t i = 0; i < count; ++i) {
        const float v = data[i];
        if (!std::isfinite(v)) {
            if (bad_index && *bad_index == UINT32_MAX) *bad_index = i;
            if (bad_value) *bad_value = v;
            return;
        }
        if (max_abs) {
            const float av = (float)std::fabs((double)v);
            *max_abs = std::max(*max_abs, av);
        }
    }
}

static void storagellm_ppl_diag_qkv_append_probe(
    uint32_t layer,
    uint32_t kv_head,
    uint32_t token_index,
    uint32_t head_dim,
    const float* key,
    const float* value,
    const qkv_state_t* slot,
    const char* stage
) {
    const int verbose = storagellm_ppl_diag_verbose_enabled();
    uint32_t key_bad = UINT32_MAX;
    uint32_t value_bad = UINT32_MAX;
    float key_bad_value = 0.0f;
    float value_bad_value = 0.0f;
    float key_max_abs = 0.0f;
    float value_max_abs = 0.0f;
    storagellm_ppl_diag_scan_vector(key, head_dim, &key_bad, &key_bad_value, &key_max_abs);
    storagellm_ppl_diag_scan_vector(value, head_dim, &value_bad, &value_bad_value, &value_max_abs);

    uint32_t sink_key_bad = UINT32_MAX;
    uint32_t sink_value_bad = UINT32_MAX;
    float sink_key_bad_value = 0.0f;
    float sink_value_bad_value = 0.0f;
    float sink_key_max_abs = 0.0f;
    float sink_value_max_abs = 0.0f;
    const int sink_present =
        slot && token_index < slot->sink_tokens && slot->k_sink && slot->v_sink &&
        slot->head_dim == (int)head_dim;
    if (sink_present) {
        const float* sink_key = slot->k_sink + (size_t)token_index * (size_t)head_dim;
        const float* sink_value = slot->v_sink + (size_t)token_index * (size_t)head_dim;
        storagellm_ppl_diag_scan_vector(
            sink_key, head_dim, &sink_key_bad, &sink_key_bad_value, &sink_key_max_abs);
        storagellm_ppl_diag_scan_vector(
            sink_value, head_dim, &sink_value_bad, &sink_value_bad_value, &sink_value_max_abs);
    }

    if (!verbose &&
        key_bad == UINT32_MAX && value_bad == UINT32_MAX &&
        sink_key_bad == UINT32_MAX && sink_value_bad == UINT32_MAX) {
        return;
    }

    std::fprintf(stderr,
        "[storagellm ppl_diag] qkv_append_probe stage=%s layer=%u kv_head=%u token=%u "
        "head_dim=%u sink_present=%d key_bad=%u key_bad_value=%.9g key_max_abs=%.9g "
        "value_bad=%u value_bad_value=%.9g value_max_abs=%.9g "
        "sink_key_bad=%u sink_key_bad_value=%.9g sink_key_max_abs=%.9g "
        "sink_value_bad=%u sink_value_bad_value=%.9g sink_value_max_abs=%.9g\n",
        stage ? stage : "unknown",
        layer,
        kv_head,
        token_index,
        head_dim,
        sink_present,
        key_bad,
        (double)key_bad_value,
        (double)key_max_abs,
        value_bad,
        (double)value_bad_value,
        (double)value_max_abs,
        sink_key_bad,
        (double)sink_key_bad_value,
        (double)sink_key_max_abs,
        sink_value_bad,
        (double)sink_value_bad_value,
        (double)sink_value_max_abs);
}

static void storagellm_ppl_diag_qkv_decode_enter(
    uint32_t layer,
    uint32_t kv_head,
    uint32_t token_position,
    uint32_t context_tokens,
    uint32_t capacity_tokens,
    uint32_t head_dim,
    float score_scale,
    float attn_logit_softcap,
    const float* query,
    const qkv_state_t* slot,
    const qkv_config_t* cfg
) {
    if (!storagellm_ppl_diag_verbose_enabled()) {
        return;
    }
    std::fprintf(stderr,
        "[storagellm ppl_diag] qkv_decode_enter layer=%u kv_head=%u pos=%u ctx=%u cap=%u "
        "head_dim=%u slot=%p slot_tokens=%d slot_dim=%d sink=%u score_scale=%.9g "
        "softcap=%.9g k_bits=%d v_bits=%d qjl=%d rotation=%d\n",
        layer,
        kv_head,
        token_position,
        context_tokens,
        capacity_tokens,
        head_dim,
        (const void*)slot,
        slot ? slot->n_tokens : -1,
        slot ? slot->head_dim : -1,
        slot ? slot->sink_tokens : 0u,
        (double)score_scale,
        (double)attn_logit_softcap,
        cfg ? cfg->k_bits : 0,
        cfg ? cfg->v_bits : 0,
        cfg && cfg->enable_qjl ? 1 : 0,
        cfg && cfg->enable_rotation ? 1 : 0);
    storagellm_ppl_diag_print_vec_stats(
        "qkv_decode_enter", (int32_t)layer, token_position, kv_head, "query", query, head_dim);
    if (slot && context_tokens > 0 && context_tokens <= slot->sink_tokens && slot->k_sink && slot->v_sink) {
        const uint32_t tokens_to_log = std::min<uint32_t>(context_tokens, 4u);
        for (uint32_t t = 0; t < tokens_to_log; ++t) {
            storagellm_ppl_diag_print_vec_stats(
                "qkv_decode_enter", (int32_t)layer, token_position, t, "sink_k",
                slot->k_sink + (size_t)t * (size_t)head_dim, head_dim);
            storagellm_ppl_diag_print_vec_stats(
                "qkv_decode_enter", (int32_t)layer, token_position, t, "sink_v",
                slot->v_sink + (size_t)t * (size_t)head_dim, head_dim);
        }
    }
}

static void storagellm_ppl_diag_qkv_decode_failed(
    uint32_t layer,
    uint32_t kv_head,
    uint32_t head_stride,
    const float* query,
    const qkv_state_t* slot,
    const qkv_config_t* cfg,
    uint32_t context_tokens,
    uint32_t capacity_tokens,
    uint32_t head_dim,
    float score_scale,
    float attn_logit_softcap
) {
    if (!query || !slot || !cfg || head_dim == 0) {
        std::fprintf(stderr,
            "[storagellm ppl_diag] qkv_decode_failed invalid_hook_args layer=%u kv_head=%u ctx=%u cap=%u head_dim=%u\n",
            layer, kv_head, context_tokens, capacity_tokens, head_dim);
        return;
    }

    double q_ss = 0.0;
    float q_max_abs = 0.0f;
    uint32_t q_bad = UINT32_MAX;
    for (uint32_t i = 0; i < head_dim; ++i) {
        const float v = query[i];
        if (!std::isfinite(v)) {
            q_bad = i;
            break;
        }
        q_ss += (double)v * (double)v;
        const float av = (float)std::fabs((double)v);
        q_max_abs = std::max(q_max_abs, av);
    }

    const int sink_covered =
        context_tokens <= slot->sink_tokens && slot->k_sink && slot->v_sink;
    uint32_t sink_bad = UINT32_MAX;
    uint32_t sink_bad_dim = UINT32_MAX;
    char sink_bad_kind = '-';
    float sink_bad_value = 0.0f;
    float sink_max_score = -std::numeric_limits<float>::infinity();
    float sink_k0_max_abs = 0.0f;
    float sink_v0_max_abs = 0.0f;
    if (sink_covered) {
        for (uint32_t t = 0; t < context_tokens && sink_bad == UINT32_MAX; ++t) {
            const float* k = slot->k_sink + (size_t)t * (size_t)head_dim;
            const float* v = slot->v_sink + (size_t)t * (size_t)head_dim;
            float dot = 0.0f;
            for (uint32_t i = 0; i < head_dim; ++i) {
                if (t == 0) {
                    sink_k0_max_abs = std::max(sink_k0_max_abs, (float)std::fabs((double)k[i]));
                    sink_v0_max_abs = std::max(sink_v0_max_abs, (float)std::fabs((double)v[i]));
                }
                if (!std::isfinite(k[i])) {
                    sink_bad = t;
                    sink_bad_dim = i;
                    sink_bad_kind = 'k';
                    sink_bad_value = k[i];
                    break;
                }
                if (!std::isfinite(v[i])) {
                    sink_bad = t;
                    sink_bad_dim = i;
                    sink_bad_kind = 'v';
                    sink_bad_value = v[i];
                    break;
                }
                dot += query[i] * k[i];
            }
            if (sink_bad != UINT32_MAX) {
                break;
            }
            const float score = dot * score_scale;
            if (!std::isfinite(score)) {
                sink_bad = t;
                sink_bad_kind = 's';
                sink_bad_value = score;
                break;
            }
            sink_max_score = std::max(sink_max_score, score);
        }
    }

    std::fprintf(stderr,
        "[storagellm ppl_diag] qkv_decode_failed layer=%u kv_head=%u stride=%u "
        "ctx=%u cap=%u head_dim=%u slot_tokens=%d slot_dim=%d sink=%u "
        "sink_covered=%d q_bad=%u q_rms=%.9g q_max_abs=%.9g "
        "sink_bad=%u sink_bad_kind=%c sink_bad_dim=%u sink_bad_value=%.9g "
        "sink_max_score=%.9g sink_k0_max_abs=%.9g sink_v0_max_abs=%.9g "
        "k_norm0=%.9g v_norm0=%.9g "
        "score_scale=%.9g softcap=%.9g k_bits=%d v_bits=%d qjl=%d rotation=%d\n",
        layer,
        kv_head,
        head_stride,
        context_tokens,
        capacity_tokens,
        head_dim,
        slot->n_tokens,
        slot->head_dim,
        slot->sink_tokens,
        sink_covered,
        q_bad,
        head_dim ? std::sqrt(q_ss / (double)head_dim) : 0.0,
        (double)q_max_abs,
        sink_bad,
        sink_bad_kind,
        sink_bad_dim,
        (double)sink_bad_value,
        (double)sink_max_score,
        (double)sink_k0_max_abs,
        (double)sink_v0_max_abs,
        slot->k_norms ? (double)slot->k_norms[0] : 0.0,
        slot->v_norms ? (double)slot->v_norms[0] : 0.0,
        (double)score_scale,
        (double)attn_logit_softcap,
        cfg->k_bits,
        cfg->v_bits,
        cfg->enable_qjl ? 1 : 0,
        cfg->enable_rotation ? 1 : 0);
}

static void storagellm_ppl_diag_attention_heads_failed(
    int32_t layer,
    uint32_t position,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    uint32_t seq_len,
    uint32_t head_dim,
    uint32_t v_head_dim,
    int qkv_decode,
    float scale,
    float attn_logit_softcap
) {
    std::fprintf(stderr,
        "[storagellm ppl_diag] attention_heads_failed layer=%d pos=%u heads=%u kv_heads=%u "
        "seq=%u head_dim=%u v_head_dim=%u qkv=%d scale=%.9g softcap=%.9g\n",
        layer,
        position,
        num_heads,
        num_kv_heads,
        seq_len,
        head_dim,
        v_head_dim,
        qkv_decode,
        (double)scale,
        (double)attn_logit_softcap);
}

#endif
