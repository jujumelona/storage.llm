#ifndef STORAGELLM_PPL_DIAG_HOOKS_CPP_INC
#define STORAGELLM_PPL_DIAG_HOOKS_CPP_INC

#define STORAGELLM_PPL_DIAG_ENABLED 1

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
    float sink_max_score = -std::numeric_limits<float>::infinity();
    if (sink_covered) {
        for (uint32_t t = 0; t < context_tokens && sink_bad == UINT32_MAX; ++t) {
            const float* k = slot->k_sink + (size_t)t * (size_t)head_dim;
            const float* v = slot->v_sink + (size_t)t * (size_t)head_dim;
            float dot = 0.0f;
            for (uint32_t i = 0; i < head_dim; ++i) {
                if (!std::isfinite(k[i]) || !std::isfinite(v[i])) {
                    sink_bad = t;
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
                break;
            }
            sink_max_score = std::max(sink_max_score, score);
        }
    }

    std::fprintf(stderr,
        "[storagellm ppl_diag] qkv_decode_failed layer=%u kv_head=%u stride=%u "
        "ctx=%u cap=%u head_dim=%u slot_tokens=%d slot_dim=%d sink=%u "
        "sink_covered=%d q_bad=%u q_rms=%.9g q_max_abs=%.9g "
        "sink_bad=%u sink_max_score=%.9g k_norm0=%.9g v_norm0=%.9g "
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
        (double)sink_max_score,
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
