// Raw forward - QKV attention integration

// GAP 1A: Quantize plain float K/V into persistent per-head qkv_state_t
// Called once per head when context changes; state persists across decode calls
static void moe_qkv_ensure_head_storage_locked(moe_pc_engine* engine, size_t slot_count) {
    if (!engine || slot_count == 0) {
        return;
    }
    if (slot_count > 1000000) {
        return;
    }
    if (engine->qkv_head_state_slots.size() < slot_count) {
        engine->qkv_head_state_slots.resize(slot_count);
    }
    if (engine->qkv_head_state_valid.size() < slot_count) {
        engine->qkv_head_state_valid.resize(slot_count, 0);
    }
    if (engine->qkv_head_state_locks.size() < slot_count) {
        engine->qkv_head_state_locks.resize(slot_count);
    }
    if (engine->qkv_head_state_layers.size() < slot_count) {
        engine->qkv_head_state_layers.resize(slot_count, UINT32_MAX);
    }
    if (engine->qkv_head_state_heads.size() < slot_count) {
        engine->qkv_head_state_heads.resize(slot_count, UINT32_MAX);
    }
    if (engine->qkv_head_state_fingerprints.size() < slot_count) {
        engine->qkv_head_state_fingerprints.resize(slot_count, 0);
    }
    if (engine->qkv_head_state_epochs.size() < slot_count) {
        engine->qkv_head_state_epochs.resize(slot_count, 0);
    }
    if (engine->qkv_head_state_filled_tokens.size() < slot_count) {
        engine->qkv_head_state_filled_tokens.resize(slot_count, 0);
    }
    for (size_t i = 0; i < slot_count; ++i) {
        if (!engine->qkv_head_state_locks[i]) {
            engine->qkv_head_state_locks[i] = std::make_unique<std::mutex>();
        }
    }
}

static void moe_qkv_free_head_states_locked(moe_pc_engine* engine) {
    if (!engine) {
        return;
    }
    for (size_t i = 0; i < engine->qkv_head_state_slots.size(); ++i) {
        if (i < engine->qkv_head_state_valid.size() && engine->qkv_head_state_valid[i]) {
            qkv_free(&engine->qkv_head_state_slots[i]);
        }
    }
    engine->qkv_head_state_slots.clear();
    engine->qkv_head_state_valid.clear();
    engine->qkv_head_state_locks.clear();
    engine->qkv_head_state_layers.clear();
    engine->qkv_head_state_heads.clear();
    engine->qkv_head_state_fingerprints.clear();
    engine->qkv_head_state_epochs.clear();
    engine->qkv_head_state_filled_tokens.clear();
    engine->qkv_last_ctx = 0;
}

static int moe_qkv_env_bool_or_current(const char* name, int current) {
    const char* env = name ? std::getenv(name) : nullptr;
    if (!env || !*env) {
        return current;
    }
    return std::atoi(env) != 0 ? 1 : 0;
}

static int moe_qkv_env_bits_or_current(const char* name, int current, int* changed) {
    const uint32_t raw = moe_parse_env_u32(name);
    if (raw == 0) {
        return current;
    }
    const int bits = (int)raw;
    if (!qkv_bits_valid(bits)) {
        moe_profile_log("qkv_config_override_rejected", "name=%s value=%u current=%d", name, raw, current);
        return current;
    }
    if (changed) {
        *changed = 1;
    }
    return bits;
}

static uint32_t moe_qkv_rotation_backend_from_env_or_current(const char* name, uint32_t current) {
    const char* env = name ? std::getenv(name) : nullptr;
    if (!env || !*env) {
        return current;
    }
    if (std::strcmp(env, "hadamard_sign_fast") == 0 ||
        std::strcmp(env, "hadamard") == 0 ||
        std::strcmp(env, "fast_hadamard") == 0) {
        return QKV_ROTATION_BACKEND_HADAMARD_SIGN_FAST;
    }
    if (std::strcmp(env, "gaussian_qr_orthogonal") == 0 ||
        std::strcmp(env, "gaussian_qr") == 0 ||
        std::strcmp(env, "paper") == 0) {
        return QKV_ROTATION_BACKEND_GAUSSIAN_QR_ORTHOGONAL;
    }
    return current;
}

static uint32_t moe_qkv_codebook_distribution_from_env_or_current(const char* name, uint32_t current) {
    const char* env = name ? std::getenv(name) : nullptr;
    if (!env || !*env) {
        return current;
    }
    if (std::strcmp(env, "gaussian_approx") == 0 ||
        std::strcmp(env, "gaussian") == 0 ||
        std::strcmp(env, "normal_approx") == 0) {
        return QKV_CODEBOOK_DISTRIBUTION_GAUSSIAN_APPROX;
    }
    if (std::strcmp(env, "exact_beta") == 0 ||
        std::strcmp(env, "beta") == 0 ||
        std::strcmp(env, "paper") == 0) {
        return QKV_CODEBOOK_DISTRIBUTION_EXACT_BETA;
    }
    return current;
}

static uint64_t moe_qkv_mix_u64(uint64_t h, uint64_t v) {
    h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdull;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ull;
    h ^= h >> 33;
    return h;
}

static uint64_t moe_qkv_config_policy_hash(const qkv_config_t& cfg) {
    uint64_t h = 0x51564b545152554eull;
    h = moe_qkv_mix_u64(h, (uint64_t)(uint32_t)cfg.k_bits);
    h = moe_qkv_mix_u64(h, (uint64_t)(uint32_t)cfg.v_bits);
    h = moe_qkv_mix_u64(h, (uint64_t)(uint32_t)cfg.head_dim);
    h = moe_qkv_mix_u64(h, cfg.enable_qjl ? 1ull : 0ull);
    h = moe_qkv_mix_u64(h, cfg.enable_rotation ? 1ull : 0ull);
    h = moe_qkv_mix_u64(h, cfg.rotation_seed);
    h = moe_qkv_mix_u64(h, cfg.qjl_seed);
    h = moe_qkv_mix_u64(h, cfg.rotation_backend);
    h = moe_qkv_mix_u64(h, cfg.codebook_distribution);
    h = moe_qkv_mix_u64(h, (uint64_t)(uint32_t)cfg.outlier_channels);
    h = moe_qkv_mix_u64(h, (uint64_t)(uint32_t)cfg.outlier_bits);
    h = moe_qkv_mix_u64(h, (uint64_t)(uint32_t)cfg.normal_bits);
    h = moe_qkv_mix_u64(h, (uint64_t)(uint32_t)cfg.key_outlier_bits);
    h = moe_qkv_mix_u64(h, (uint64_t)(uint32_t)cfg.key_normal_bits);
    h = moe_qkv_mix_u64(h, (uint64_t)(uint32_t)cfg.value_outlier_bits);
    h = moe_qkv_mix_u64(h, (uint64_t)(uint32_t)cfg.value_normal_bits);
    h = moe_qkv_mix_u64(h, cfg.group_size);
    h = moe_qkv_mix_u64(h, cfg.page_size_tokens);
    h = moe_qkv_mix_u64(h, cfg.sink_tokens);
    h = moe_qkv_mix_u64(h, cfg.plain_kv_persistent_storage ? 1ull : 0ull);
    return h ? h : 1ull;
}

static int moe_qkv_drop_unbound_outlier_split(qkv_config_t& cfg) {
    if (cfg.outlier_channels <= 0 || cfg.outlier_channel_indices) {
        return 0;
    }
    const int dropped = cfg.outlier_channels;
    cfg.outlier_channels = 0;
    cfg.policy_hash = 0;
    return dropped;
}

static qkv_config_t moe_qkv_contract_config_for_head_dim(
    const moe_pc_engine* engine,
    uint32_t head_dim
) {
    qkv_config_t cfg = qkv_config_default((int)head_dim);
    if (engine && engine->offload_qkv_cfg_available.load(std::memory_order_acquire)) {
        cfg = engine->offload_qkv_cfg;
        cfg.head_dim = (int)head_dim;
        if (cfg.outlier_channels < 0) {
            cfg.outlier_channels = 0;
        }
        if (cfg.outlier_channels > (int)head_dim) {
            cfg.outlier_channels = (int)head_dim;
        }
        if (cfg.group_size == 0) {
            cfg.group_size = 64;
        }
        if (cfg.page_size_tokens == 0) {
            cfg.page_size_tokens = 16;
        }
        if (cfg.sink_tokens == 0) {
            cfg.sink_tokens = 4;
        }
    }
    int bits_overridden = 0;
    const int qkv_forced = engine && engine->qkv_forced_by_format.load(std::memory_order_acquire);
    const char* allow_override_env = std::getenv("STORAGELLM_ALLOW_QKV_ENV_OVERRIDE");
    const int allow_env_override = allow_override_env && allow_override_env[0] &&
        std::strcmp(allow_override_env, "0") != 0 &&
        std::strcmp(allow_override_env, "false") != 0 &&
        std::strcmp(allow_override_env, "FALSE") != 0;
    const int apply_env_overrides = !qkv_forced || allow_env_override;
    if (apply_env_overrides) {
        cfg.k_bits = moe_qkv_env_bits_or_current("STORAGELLM_QKV_K_BITS", cfg.k_bits, &bits_overridden);
        cfg.v_bits = moe_qkv_env_bits_or_current("STORAGELLM_QKV_V_BITS", cfg.v_bits, &bits_overridden);
        cfg.normal_bits = moe_qkv_env_bits_or_current("STORAGELLM_QKV_NORMAL_BITS", cfg.normal_bits, &bits_overridden);
        cfg.outlier_bits = moe_qkv_env_bits_or_current("STORAGELLM_QKV_OUTLIER_BITS", cfg.outlier_bits, &bits_overridden);
        cfg.key_normal_bits = moe_qkv_env_bits_or_current("STORAGELLM_QKV_KEY_NORMAL_BITS", cfg.key_normal_bits, &bits_overridden);
        cfg.value_normal_bits = moe_qkv_env_bits_or_current("STORAGELLM_QKV_VALUE_NORMAL_BITS", cfg.value_normal_bits, &bits_overridden);
        cfg.key_outlier_bits = moe_qkv_env_bits_or_current("STORAGELLM_QKV_KEY_OUTLIER_BITS", cfg.key_outlier_bits, &bits_overridden);
        cfg.value_outlier_bits = moe_qkv_env_bits_or_current("STORAGELLM_QKV_VALUE_OUTLIER_BITS", cfg.value_outlier_bits, &bits_overridden);
        const uint32_t outlier_channels_override = moe_parse_env_u32("STORAGELLM_QKV_OUTLIER_CHANNELS");
        if (outlier_channels_override > 0 || std::getenv("STORAGELLM_QKV_OUTLIER_CHANNELS")) {
            cfg.outlier_channels = (int)std::min<uint32_t>(outlier_channels_override, head_dim);
        } else if (bits_overridden && (qkv_bits_raw(cfg.k_bits) || qkv_bits_raw(cfg.v_bits))) {
            cfg.outlier_channels = 0;
            cfg.normal_bits = cfg.v_bits;
            cfg.outlier_bits = cfg.k_bits;
            cfg.key_normal_bits = cfg.normal_bits;
            cfg.value_normal_bits = cfg.normal_bits;
            cfg.key_outlier_bits = cfg.outlier_bits;
            cfg.value_outlier_bits = cfg.outlier_bits;
        }
        cfg.enable_rotation = moe_qkv_env_bool_or_current("STORAGELLM_QKV_ROTATION", cfg.enable_rotation ? 1 : 0) != 0;
        cfg.enable_qjl = moe_qkv_env_bool_or_current("STORAGELLM_QKV_QJL", cfg.enable_qjl ? 1 : 0) != 0;
        cfg.rotation_backend = moe_qkv_rotation_backend_from_env_or_current(
            "STORAGELLM_QKV_ROTATION_BACKEND", cfg.rotation_backend);
        cfg.codebook_distribution = moe_qkv_codebook_distribution_from_env_or_current(
            "STORAGELLM_QKV_CODEBOOK_DISTRIBUTION", cfg.codebook_distribution);
    } else if (std::getenv("STORAGELLM_QKV_K_BITS") ||
               std::getenv("STORAGELLM_QKV_V_BITS") ||
               std::getenv("STORAGELLM_QKV_NORMAL_BITS") ||
               std::getenv("STORAGELLM_QKV_OUTLIER_BITS") ||
               std::getenv("STORAGELLM_QKV_KEY_NORMAL_BITS") ||
               std::getenv("STORAGELLM_QKV_VALUE_NORMAL_BITS") ||
               std::getenv("STORAGELLM_QKV_KEY_OUTLIER_BITS") ||
               std::getenv("STORAGELLM_QKV_VALUE_OUTLIER_BITS") ||
               std::getenv("STORAGELLM_QKV_OUTLIER_CHANNELS") ||
               std::getenv("STORAGELLM_QKV_ROTATION") ||
               std::getenv("STORAGELLM_QKV_QJL") ||
               std::getenv("STORAGELLM_QKV_ROTATION_BACKEND") ||
               std::getenv("STORAGELLM_QKV_CODEBOOK_DISTRIBUTION")) {
        if (engine) {
            const_cast<moe_pc_engine*>(engine)->qkv_env_override_blocked.fetch_add(1, std::memory_order_relaxed);
        }
        moe_profile_log("qkv_env_override_blocked", "head_dim=%u policy_hash=%llu",
            head_dim, (unsigned long long)cfg.policy_hash);
    }
    if (qkv_bits_raw(cfg.k_bits) || qkv_bits_raw(cfg.v_bits) ||
        qkv_bits_raw(cfg.normal_bits) || qkv_bits_raw(cfg.outlier_bits) ||
        qkv_bits_raw(cfg.key_normal_bits) || qkv_bits_raw(cfg.value_normal_bits) ||
        qkv_bits_raw(cfg.key_outlier_bits) || qkv_bits_raw(cfg.value_outlier_bits)) {
        cfg.enable_qjl = false;
    }
    if (cfg.outlier_channels < 0) {
        cfg.outlier_channels = 0;
    }
    if (cfg.outlier_channels > (int)head_dim) {
        cfg.outlier_channels = (int)head_dim;
    }
    moe_qkv_drop_unbound_outlier_split(cfg);
    if (cfg.rotation_backend != QKV_ROTATION_BACKEND_GAUSSIAN_QR_ORTHOGONAL &&
        cfg.rotation_backend != QKV_ROTATION_BACKEND_HADAMARD_SIGN_FAST) {
        cfg.rotation_backend = QKV_ROTATION_BACKEND_GAUSSIAN_QR_ORTHOGONAL;
    }
    if (cfg.codebook_distribution != QKV_CODEBOOK_DISTRIBUTION_EXACT_BETA &&
        cfg.codebook_distribution != QKV_CODEBOOK_DISTRIBUTION_GAUSSIAN_APPROX) {
        cfg.codebook_distribution = QKV_CODEBOOK_DISTRIBUTION_EXACT_BETA;
    }
    if (qkv_forced) {
        cfg.plain_kv_persistent_storage = false;
    }
    if (cfg.policy_hash == 0) {
        cfg.policy_hash = moe_qkv_config_policy_hash(cfg);
    }
    cfg.engine_io_thread_count =
        runtime_disk_workers(engine) + runtime_pinned_workers(engine) + runtime_gpu_workers(engine);
    return cfg;
}

static bool moe_qkv_config_matches_contract(
    const qkv_config_t& current,
    const qkv_config_t& desired
) {
    return current.k_bits == desired.k_bits &&
        current.v_bits == desired.v_bits &&
        current.head_dim == desired.head_dim &&
        current.enable_qjl == desired.enable_qjl &&
        current.enable_rotation == desired.enable_rotation &&
        current.rotation_seed == desired.rotation_seed &&
        current.qjl_seed == desired.qjl_seed &&
        current.rotation_backend == desired.rotation_backend &&
        current.codebook_distribution == desired.codebook_distribution &&
        current.policy_hash == desired.policy_hash &&
        current.outlier_channels == desired.outlier_channels &&
        current.outlier_bits == desired.outlier_bits &&
        current.normal_bits == desired.normal_bits &&
        current.key_outlier_bits == desired.key_outlier_bits &&
        current.key_normal_bits == desired.key_normal_bits &&
        current.value_outlier_bits == desired.value_outlier_bits &&
        current.value_normal_bits == desired.value_normal_bits &&
        current.group_size == desired.group_size &&
        current.page_size_tokens == desired.page_size_tokens &&
        current.sink_tokens == desired.sink_tokens &&
        current.attention_score_scale == desired.attention_score_scale &&
        current.attention_logit_softcap == desired.attention_logit_softcap &&
        current.plain_kv_persistent_storage == desired.plain_kv_persistent_storage;
}

static void moe_qkv_apply_contract_config_locked(
    moe_pc_engine* engine,
    uint32_t head_dim
) {
    if (!engine) {
        return;
    }
    const qkv_config_t desired = moe_qkv_contract_config_for_head_dim(engine, head_dim);
    if (!engine->qkv_cfg_initialized.load(std::memory_order_acquire) ||
        !moe_qkv_config_matches_contract(engine->qkv_cfg, desired)) {
        // Mixed-attention models such as Gemma variants can alternate head_dim
        // across layers (for example sliding/local vs global attention).  The
        // QKV slot fingerprint already includes head_dim and policy_hash, so a
        // per-slot mismatch is enough to rebuild only the affected layer/head.
        // Do not clear the whole persistent KV cache merely because the next
        // layer uses a different head_dim: that drops previous-token KV for
        // other layers and makes PPL explode while the request still reports
        // forward_ok=1.
        engine->qkv_cfg = desired;
        engine->qkv_cfg_initialized.store(true, std::memory_order_release);
        const int disabled_outlier_channels =
            engine->offload_qkv_cfg_available.load(std::memory_order_acquire) &&
            engine->offload_qkv_cfg.outlier_channels > 0 &&
            !engine->offload_qkv_cfg.outlier_channel_indices &&
            desired.outlier_channels == 0
                ? engine->offload_qkv_cfg.outlier_channels
                : 0;
        if (disabled_outlier_channels > 0) {
            moe_profile_log(
                "qkv_outlier_split_disabled",
                "head_dim=%u requested_channels=%d reason=missing_explicit_outlier_channel_indices",
                head_dim,
                disabled_outlier_channels);
        }
        moe_profile_log(
            "qkv_config",
            "head_dim=%u k_bits=%d v_bits=%d group=%u page=%u sink=%u "
            "rotation=%d rotation_backend=%u rotation_seed=%llu qjl=%d qjl_seed=%llu "
            "codebook_distribution=%u policy_hash=%llu "
            "outlier_channels=%d outlier_bits=%d normal_bits=%d "
            "key_outlier_bits=%d key_normal_bits=%d value_outlier_bits=%d value_normal_bits=%d "
            "effective_bits=%.3f plain_kv=%d "
            "env_k=%u env_v=%u env_normal=%u env_outlier=%u "
            "env_key_normal=%u env_value_normal=%u env_key_outlier=%u env_value_outlier=%u",
            head_dim,
            desired.k_bits,
            desired.v_bits,
            desired.group_size,
            desired.page_size_tokens,
            desired.sink_tokens,
            desired.enable_rotation ? 1 : 0,
            desired.rotation_backend,
            (unsigned long long)desired.rotation_seed,
            desired.enable_qjl ? 1 : 0,
            (unsigned long long)desired.qjl_seed,
            desired.codebook_distribution,
            (unsigned long long)desired.policy_hash,
            desired.outlier_channels,
            desired.outlier_bits,
            desired.normal_bits,
            desired.key_outlier_bits,
            desired.key_normal_bits,
            desired.value_outlier_bits,
            desired.value_normal_bits,
            (double)qkv_effective_bits(&desired),
            desired.plain_kv_persistent_storage ? 1 : 0,
            moe_parse_env_u32("STORAGELLM_QKV_K_BITS"),
            moe_parse_env_u32("STORAGELLM_QKV_V_BITS"),
            moe_parse_env_u32("STORAGELLM_QKV_NORMAL_BITS"),
            moe_parse_env_u32("STORAGELLM_QKV_OUTLIER_BITS"),
            moe_parse_env_u32("STORAGELLM_QKV_KEY_NORMAL_BITS"),
            moe_parse_env_u32("STORAGELLM_QKV_VALUE_NORMAL_BITS"),
            moe_parse_env_u32("STORAGELLM_QKV_KEY_OUTLIER_BITS"),
            moe_parse_env_u32("STORAGELLM_QKV_VALUE_OUTLIER_BITS"));
    }
}

static int moe_qkv_prepare_runtime_config(
    moe_pc_engine* engine,
    uint32_t head_dim
) {
    if (!engine || head_dim == 0) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(engine->qkv_mutex);
    moe_qkv_apply_contract_config_locked(engine, head_dim);
    return 1;
}

static int moe_qkv_prepare_layer_head_storage(
    moe_pc_engine* engine,
    uint32_t layer,
    uint32_t head_stride,
    uint32_t head_dim
) {
    if (!engine || head_stride == 0 || head_dim == 0) {
        return 0;
    }
    const uint64_t layers = (uint64_t)layer + 1ull;
    if (layers > (UINT64_MAX / (uint64_t)head_stride)) {
        return 0;
    }
    const uint64_t slot_count_u64 = layers * (uint64_t)head_stride;
    if (slot_count_u64 > (uint64_t)SIZE_MAX) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(engine->qkv_mutex);
    moe_qkv_apply_contract_config_locked(engine, head_dim);
    moe_qkv_ensure_head_storage_locked(engine, (size_t)slot_count_u64);
    return engine->qkv_head_state_slots.size() >= (size_t)slot_count_u64 &&
        engine->qkv_head_state_locks.size() >= (size_t)slot_count_u64;
}

static int moe_qkv_slot_index(
    uint32_t layer,
    uint32_t head,
    uint32_t head_stride,
    size_t* out_slot
) {
    if (!out_slot || head_stride == 0 || head >= head_stride) {
        return 0;
    }
    if ((uint64_t)layer > (UINT64_MAX - (uint64_t)head) / (uint64_t)head_stride) {
        return 0;
    }
    const uint64_t slot = (uint64_t)layer * (uint64_t)head_stride + (uint64_t)head;
    if (slot > (uint64_t)SIZE_MAX) {
        return 0;
    }
    *out_slot = (size_t)slot;
    return 1;
}

static void moe_qkv_note_context_progress_locked(moe_pc_engine* engine, uint32_t context_tokens) {
    if (!engine || context_tokens == 0) {
        return;
    }
    if (context_tokens == 1u && engine->qkv_last_ctx != 1u) {
        ++engine->qkv_context_epoch;
        if (engine->qkv_context_epoch == 0) {
            engine->qkv_context_epoch = 1;
        }
    }
    engine->qkv_last_ctx = context_tokens;
}

static uint64_t moe_qkv_layer_head_fingerprint(
    uint32_t layer,
    uint32_t head,
    uint32_t capacity_tokens,
    uint32_t head_dim,
    uint64_t epoch,
    uint64_t policy_hash
) {
    uint64_t h = policy_hash ? policy_hash : 0x51564b5354415445ull;
    h = moe_qkv_mix_u64(h, layer);
    h = moe_qkv_mix_u64(h, head);
    h = moe_qkv_mix_u64(h, capacity_tokens);
    h = moe_qkv_mix_u64(h, head_dim);
    h = moe_qkv_mix_u64(h, epoch);
    return h ? h : 1ull;
}

static qkv_state_t* moe_qkv_ensure_layer_head_state_locked(
    moe_pc_engine* engine,
    uint32_t layer,
    uint32_t head,
    uint32_t head_stride,
    uint32_t capacity_tokens,
    uint32_t head_dim,
    size_t* out_slot_index
) {
    if (!engine || head_stride == 0 || capacity_tokens == 0 || head_dim == 0) {
        return nullptr;
    }
    size_t slot_index = 0;
    if (!moe_qkv_slot_index(layer, head, head_stride, &slot_index)) {
        return nullptr;
    }
    const size_t slot_count = slot_index + 1u;
    {
        std::lock_guard<std::mutex> lock(engine->qkv_mutex);
        moe_qkv_apply_contract_config_locked(engine, head_dim);
        moe_qkv_ensure_head_storage_locked(engine, slot_count);
    }
    if (slot_index >= engine->qkv_head_state_locks.size() ||
        !engine->qkv_head_state_locks[slot_index]) {
        return nullptr;
    }
    if (out_slot_index) {
        *out_slot_index = slot_index;
    }
    qkv_state_t* slot = &engine->qkv_head_state_slots[slot_index];
    const uint64_t epoch = engine->qkv_context_epoch ? engine->qkv_context_epoch : 1ull;
    const uint64_t desired_fingerprint = moe_qkv_layer_head_fingerprint(
        layer, head, capacity_tokens, head_dim, epoch, engine->qkv_cfg.policy_hash);
    const int valid =
        slot_index < engine->qkv_head_state_valid.size() &&
        engine->qkv_head_state_valid[slot_index] &&
        slot->head_dim == (int)head_dim &&
        slot->n_tokens >= (int)capacity_tokens &&
        slot_index < engine->qkv_head_state_layers.size() &&
        slot_index < engine->qkv_head_state_heads.size() &&
        engine->qkv_head_state_layers[slot_index] == layer &&
        engine->qkv_head_state_heads[slot_index] == head &&
        slot_index < engine->qkv_head_state_fingerprints.size() &&
        engine->qkv_head_state_fingerprints[slot_index] == desired_fingerprint &&
        slot_index < engine->qkv_head_state_epochs.size() &&
        engine->qkv_head_state_epochs[slot_index] == epoch;
    if (valid) {
        return slot;
    }
    if (slot_index < engine->qkv_head_state_valid.size() &&
        engine->qkv_head_state_valid[slot_index]) {
        qkv_free(slot);
        *slot = qkv_state_t{};
        engine->qkv_head_state_valid[slot_index] = 0;
        if (slot_index < engine->qkv_head_state_filled_tokens.size()) {
            engine->qkv_head_state_filled_tokens[slot_index] = 0;
        }
    }
    qkv_state_t st{};
    if (capacity_tokens > (uint32_t)std::numeric_limits<int>::max() ||
        !qkv_init(&st, &engine->qkv_cfg, (int)capacity_tokens)) {
        return nullptr;
    }
    *slot = st;
    engine->qkv_head_state_valid[slot_index] = 1;
    engine->qkv_head_state_layers[slot_index] = layer;
    engine->qkv_head_state_heads[slot_index] = head;
    engine->qkv_head_state_fingerprints[slot_index] = desired_fingerprint;
    engine->qkv_head_state_epochs[slot_index] = epoch;
    if (slot_index < engine->qkv_head_state_filled_tokens.size()) {
        engine->qkv_head_state_filled_tokens[slot_index] = 0;
    }
    return slot;
}

static int moe_qkv_quantize_current_token_mse(
    qkv_state_t* slot,
    const qkv_config_t* cfg,
    const float* key,
    const float* value,
    uint32_t token_index,
    uint32_t head_dim
) {
    if (!slot || !cfg || !key || !value || head_dim == 0 ||
        token_index >= (uint32_t)std::max(slot->n_tokens, 0)) {
        return 0;
    }
    if (!qkv_quantize_token(slot, cfg, key, value, (int)token_index)) {
        return 0;
    }
    return 1;
}

static int moe_qkv_append_layer_head_token(
    moe_pc_engine* engine,
    uint32_t layer,
    uint32_t head,
    uint32_t head_stride,
    uint32_t capacity_tokens,
    uint32_t context_tokens,
    uint32_t head_dim,
    const float* key,
    const float* value
) {
    if (!engine || !key || !value || context_tokens == 0 || capacity_tokens < context_tokens) {
        return 0;
    }
    {
        std::lock_guard<std::mutex> lock(engine->qkv_mutex);
        moe_qkv_note_context_progress_locked(engine, context_tokens);
    }
    size_t slot_index = 0;
    qkv_state_t* slot = moe_qkv_ensure_layer_head_state_locked(
        engine, layer, head, head_stride, capacity_tokens, head_dim, &slot_index);
    if (!slot || slot_index >= engine->qkv_head_state_locks.size() ||
        !engine->qkv_head_state_locks[slot_index]) {
        return 0;
    }
    std::lock_guard<std::mutex> head_guard(*engine->qkv_head_state_locks[slot_index]);
    slot = moe_qkv_ensure_layer_head_state_locked(
        engine, layer, head, head_stride, capacity_tokens, head_dim, &slot_index);
    if (!slot) {
        return 0;
    }
    const uint32_t token_index = context_tokens - 1u;
#ifdef STORAGELLM_PPL_DIAG_ENABLED
    storagellm_ppl_diag_qkv_append_probe(
        layer, head, token_index, head_dim, key, value, slot, "before_quantize");
#endif
    if (!moe_qkv_quantize_current_token_mse(
            slot, &engine->qkv_cfg, key, value, token_index, head_dim)) {
        return 0;
    }
    if (slot_index < engine->qkv_head_state_filled_tokens.size()) {
        engine->qkv_head_state_filled_tokens[slot_index] = std::max(
            engine->qkv_head_state_filled_tokens[slot_index],
            token_index + 1u);
    }
#ifdef STORAGELLM_PPL_DIAG_ENABLED
    storagellm_ppl_diag_qkv_append_probe(
        layer, head, token_index, head_dim, key, value, slot, "after_quantize");
#endif
    return 1;
}

static int moe_pc_engine_attention_decode_layer_head_qkv_f32(
    moe_pc_engine_t* engine,
    uint32_t layer,
    uint32_t kv_head,
    uint32_t head_stride,
    const float* query,
    const float* current_key,
    const float* current_value,
    uint32_t context_tokens,
    uint32_t capacity_tokens,
    uint32_t head_dim,
    float score_scale,
    float attn_logit_softcap,
    float* output
) {
    if (!engine || !query || !output || context_tokens == 0 || capacity_tokens < context_tokens || head_dim == 0) {
        return 0;
    }
    size_t slot_index = 0;
    qkv_state_t* slot = moe_qkv_ensure_layer_head_state_locked(
        engine, layer, kv_head, head_stride, capacity_tokens, head_dim, &slot_index);
    if (!slot || slot_index >= engine->qkv_head_state_locks.size() ||
        !engine->qkv_head_state_locks[slot_index]) {
        return 0;
    }
    if (!(score_scale > 0.0f) || !std::isfinite(score_scale)) {
        return 0;
    }
    std::lock_guard<std::mutex> head_guard(*engine->qkv_head_state_locks[slot_index]);
    slot = &engine->qkv_head_state_slots[slot_index];
    if (slot_index >= engine->qkv_head_state_valid.size() ||
        !engine->qkv_head_state_valid[slot_index] ||
        slot_index >= engine->qkv_head_state_filled_tokens.size() ||
        engine->qkv_head_state_filled_tokens[slot_index] < context_tokens ||
        slot->n_tokens < (int)context_tokens) {
        return 0;
    }
    qkv_config_t qkv_decode_cfg = engine->qkv_cfg;
    qkv_decode_cfg.attention_score_scale = score_scale;
    qkv_decode_cfg.attention_logit_softcap = attn_logit_softcap;
#ifdef STORAGELLM_PPL_DIAG_ENABLED
    storagellm_ppl_diag_qkv_decode_enter(
        layer,
        kv_head,
        context_tokens - 1u,
        context_tokens,
        capacity_tokens,
        head_dim,
        score_scale,
        attn_logit_softcap,
        query,
        slot,
        &qkv_decode_cfg);
#endif
    const int ok = (current_key && current_value)
        ? qkv_attention_decode_exact_current(
            query, slot, &qkv_decode_cfg, context_tokens, head_dim,
            current_key, current_value, output)
        : qkv_attention_decode(query, slot, &qkv_decode_cfg, context_tokens, head_dim, output);
    if (!ok) {
#ifdef STORAGELLM_PPL_DIAG_ENABLED
        storagellm_ppl_diag_qkv_decode_failed(
            layer,
            kv_head,
            head_stride,
            query,
            slot,
            &qkv_decode_cfg,
            context_tokens,
            capacity_tokens,
            head_dim,
            score_scale,
            attn_logit_softcap);
#endif
    }
    return ok;
}

int moe_pc_engine_attention_decode_f32(
    moe_pc_engine_t* engine,
    const float* query,
    const float* key_cache,
    const float* value_cache,
    uint32_t context_tokens,
    uint32_t head_count,
    uint32_t head_dim,
    float* output
) {
    if (!engine || !query || !key_cache || !value_cache || !output ||
        context_tokens == 0 || head_count == 0 || head_dim == 0) {
        return 0;
    }
    // BUGFIX 640: head_dim 체크를 sqrt 전에 수행 ★
    // Problem: head_dim == 0 체크가 sqrt 호출 후에 있음
    // 위 if문에서 head_dim == 0 체크하므로 안전하지만, 명시적으로 확인
    // Solution: 이미 체크되어 있으므로 안전함 (주석 추가)
    const float inv_sqrt = 1.0f / std::sqrt((float)head_dim);  // Safe: head_dim > 0 guaranteed by check above
    if (context_tokens > (uint32_t)std::numeric_limits<int>::max() ||
        head_dim > (uint32_t)std::numeric_limits<int>::max()) {
        return 0;
    }
    // BUGFIX 630: Check for overflow in pointer arithmetic ★
    // Problem: head * context_tokens * head_dim can overflow uint64_t
    // Solution: Check multiplication overflow before pointer arithmetic
    if (head_count > 0 && context_tokens > 0 && head_dim > 0) {
        if (context_tokens > UINT64_MAX / head_dim) {
            return 0;  // context_tokens * head_dim would overflow
        }
        uint64_t tokens_times_dim = (uint64_t)context_tokens * (uint64_t)head_dim;
        if (head_count > UINT64_MAX / tokens_times_dim) {
            return 0;  // head_count * context_tokens * head_dim would overflow
        }
    }
    if (engine->config.kv_mode == moe_KV_MODE_QKV && qkv_is_enabled() &&
        !moe_qkv_prepare_runtime_config(engine, head_dim)) {
        return 0;
    }
    return moe_parallel_rows(engine, head_count, 0, [&](uint32_t head) {
        const float* q = query + (uint64_t)head * (uint64_t)head_dim;
        float* out = output + (uint64_t)head * (uint64_t)head_dim;
        const float* head_keys = key_cache + (uint64_t)head * (uint64_t)context_tokens * (uint64_t)head_dim;
        const float* head_values = value_cache + (uint64_t)head * (uint64_t)context_tokens * (uint64_t)head_dim;
        // Legacy flat API: use a per-call state so head-only slots cannot
        // contaminate the layer/head persistent generation cache.
        if (engine->config.kv_mode == moe_KV_MODE_QKV && qkv_is_enabled()) {
            qkv_state_t st{};
            if (context_tokens > (uint32_t)std::numeric_limits<int>::max() ||
                !qkv_init(&st, &engine->qkv_cfg, (int)context_tokens)) {
                return 0;
            }
            const int ok = qkv_quantize(&st, &engine->qkv_cfg, head_keys, head_values, (int)context_tokens) &&
                qkv_attention_decode(q, &st, &engine->qkv_cfg, context_tokens, head_dim, out);
            qkv_free(&st);
            return ok;
        }
        float max_score = -std::numeric_limits<float>::infinity();

        static thread_local std::vector<float> tls_scores;
        if (tls_scores.size() < context_tokens) {
            tls_scores.resize(context_tokens);
        }
        float* scores = tls_scores.data();

        for (uint32_t t = 0; t < context_tokens; ++t) {
            // [head][t][d] layout
            const float* k = head_keys + (uint64_t)t * (uint64_t)head_dim;
            float score = 0.0f;
            for (uint32_t d = 0; d < head_dim; ++d) {
                score += q[d] * k[d];
            }
            score *= inv_sqrt;
            scores[t] = score;
            if (score > max_score) {
                max_score = score;
            }
        }
        for (uint32_t d = 0; d < head_dim; ++d) {
            out[d] = 0.0f;
        }
        float denom = 0.0f;
        for (uint32_t t = 0; t < context_tokens; ++t) {
            // [head][t][d] layout
            const float* v = head_values + (uint64_t)t * (uint64_t)head_dim;
            const float weight = std::exp(scores[t] - max_score);
            denom += weight;
            for (uint32_t d = 0; d < head_dim; ++d) {
                out[d] += weight * v[d];
            }
        }
        // BUGFIX 650: Softmax fallback consistency with QKV mode ★★★
        // Problem: Old code returned error (0) when denom <= 0, causing generation abort
        //          QKV mode uses uniform distribution fallback instead (qkv_attention.cpp:362)
        // Impact: Non-QKV mode fails on edge cases where QKV succeeds → PPL inconsistency
        // Solution: Match QKV behavior with uniform fallback, use same threshold (1e-10f)
        if (denom > 1e-10f) {
            const float scale = 1.0f / denom;
            for (uint32_t d = 0; d < head_dim; ++d) {
                out[d] *= scale;
            }
        } else {
            return 0;
        }
        return 1;
    });
}

// FATE Predictor: Save hidden state snapshot just before gate
static void save_gate_input_snapshot(moe_pc_engine* engine, int layer, const float* hidden, uint32_t dim) {
    if (!engine || layer < 0 || !hidden || dim == 0) return;
    const size_t layer_u = (size_t)layer;
    // BUGFIX 503: layer_u 범위 체크
    if (layer_u > moe_GATE_SNAPSHOT_MAX_LAYER) return;
    const size_t req_size = (layer_u + 1u) * 2u;
    if (req_size > moe_GATE_SNAPSHOT_EPOCH_SLOTS) return;
    {
        std::lock_guard<std::mutex> lock(engine->gate_input_snapshot_mutex);
        if (engine->gate_input_snapshots.size() < req_size) {
            engine->gate_input_snapshots.resize(req_size);
        }

        const size_t slot0 = layer_u * 2u;
        const size_t slot1 = slot0 + 1u;
        std::atomic<uint64_t>& epoch0_ref = engine->gate_input_snapshot_epochs[slot0];
        std::atomic<uint64_t>& epoch1_ref = engine->gate_input_snapshot_epochs[slot1];
        const uint64_t e0 = epoch0_ref.load(std::memory_order_acquire);
        const uint64_t e1 = epoch1_ref.load(std::memory_order_acquire);
        const int write_idx = (e0 >= e1) ? 1 : 0;
        const size_t write_slot = layer_u * 2u + (size_t)write_idx;

        engine->gate_input_snapshots[write_slot].assign(hidden, hidden + dim);
        const uint64_t new_epoch = std::max(e0, e1) + 1u;
        if (write_idx == 0) {
            epoch0_ref.store(new_epoch, std::memory_order_release);
        } else {
            epoch1_ref.store(new_epoch, std::memory_order_release);
        }
    }
}
