// QKV-only runtime helpers.
// These helpers close the offload-native QKV invariant without replacing the
// surrounding forward implementation. They deliberately keep plain KV available
// only for legacy/debug non-QKV models while making every QKV-required touch
// explicit and countable.

static int moe_engine_qkv_only_required(const moe_pc_engine* engine) {
    return engine &&
        (engine->qkv_forced_by_format.load(std::memory_order_acquire) ||
         engine->config.kv_mode == moe_KV_MODE_QKV);
}

static int moe_engine_qkv_only_active(const moe_pc_engine* engine) {
    return moe_engine_qkv_only_required(engine) && qkv_is_enabled();
}

static void moe_qkv_only_plain_touch(moe_pc_engine* engine, const char* site) {
    if (!engine || !moe_engine_qkv_only_required(engine)) {
        return;
    }
    engine->qkv_plain_kv_forbidden_touches.fetch_add(1, std::memory_order_relaxed);
    if (site) {
        if (std::strcmp(site, "readiness") == 0) {
            engine->qkv_plain_kv_readiness_touches.fetch_add(1, std::memory_order_relaxed);
        } else if (std::strcmp(site, "capacity") == 0) {
            engine->qkv_plain_kv_capacity_touches.fetch_add(1, std::memory_order_relaxed);
        } else if (std::strcmp(site, "store") == 0) {
            engine->qkv_plain_kv_store_touches.fetch_add(1, std::memory_order_relaxed);
        } else if (std::strcmp(site, "decode_cache") == 0) {
            engine->qkv_plain_kv_decode_cache_touches.fetch_add(1, std::memory_order_relaxed);
        }
    }
    moe_profile_log("qkv_only_plain_kv_touch", "site=%s touches=%llu",
        site ? site : "unknown",
        (unsigned long long)engine->qkv_plain_kv_forbidden_touches.load(std::memory_order_relaxed));
}

static uint32_t moe_qkv_only_seq_limit_from_cfg(const moe_attn_config_t& cfg, uint32_t min_needed) {
    uint32_t limit = cfg.max_seq_len ? cfg.max_seq_len : moe_MLA_MAX_SEQ_LEN;
    if (limit == 0) {
        limit = min_needed ? min_needed : 1u;
    }
    if (limit < min_needed) {
        limit = min_needed;
    }
    return limit;
}

static uint32_t moe_qkv_only_ensure_capacity(
    moe_pc_engine* engine,
    moe_decode_scratch* s,
    const moe_attn_config_t& cfg,
    uint32_t position,
    int pre_allocate
) {
    if (!s) {
        return 0;
    }
    const uint32_t seq_len = position + 1u;
    const uint32_t limit = moe_qkv_only_seq_limit_from_cfg(cfg, seq_len);
    if (position >= limit) {
        return 0;
    }
    s->qkv_only_runtime = 1;
    s->qkv_seq_limit = limit;
    uint32_t cap = s->qkv_capacity_tokens;
    if (cap == 0) {
        cap = pre_allocate ? limit : std::min<uint32_t>(moe_MLA_KV_INITIAL_CHUNK, limit);
        if (cap == 0) cap = seq_len;
    }
    while (cap < seq_len && cap < limit) {
        const uint32_t next_cap = cap > UINT32_MAX / 2u ? limit : std::min<uint32_t>(limit, cap * 2u);
        if (next_cap <= cap) break;
        cap = next_cap;
    }
    if (cap < seq_len) {
        return 0;
    }
    s->qkv_capacity_tokens = cap;
    s->qkv_seq_len = seq_len;
    if (engine) {
        // Prove that capacity bookkeeping is QKV-local, not borrowed from moe_kv_cache.
        moe_profile_log("qkv_only_capacity", "pos=%u seq=%u cap=%u limit=%u plain_kv_bytes=%llu",
            position, s->qkv_seq_len, s->qkv_capacity_tokens, s->qkv_seq_limit,
            (unsigned long long)moe_kv_cache_allocated_bytes(s->kv_cache));
    }
    return cap;
}
