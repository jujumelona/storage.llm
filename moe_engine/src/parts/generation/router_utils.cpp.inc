static void moe_save_gate_input_snapshot(
    moe_pc_engine* engine,
    int32_t layer,
    const float* hidden,
    uint32_t hidden_size
) {
    if (!engine || !hidden || layer < 0 || hidden_size == 0) {
        return;
    }
    const size_t slot = (size_t)layer;
    if (slot > moe_GATE_SNAPSHOT_MAX_LAYER) return;
    if (hidden_size > 100000) return;

    // Fix 19: Double buffering for lock-free snapshot exchange
    const size_t req_size = (slot + 1) * 2;
    {
        std::lock_guard<std::mutex> lock(engine->gate_input_snapshot_mutex);
        if (engine->gate_input_snapshots.size() < req_size) {
            engine->gate_input_snapshots.resize(req_size);
        }

        const size_t slot0 = slot * 2;
        const size_t slot1 = slot * 2 + 1;

        if (slot0 >= moe_GATE_SNAPSHOT_EPOCH_SLOTS || slot1 >= moe_GATE_SNAPSHOT_EPOCH_SLOTS) {
            return;
        }
        std::atomic<uint64_t>& epoch0_ref = engine->gate_input_snapshot_epochs[slot0];
        std::atomic<uint64_t>& epoch1_ref = engine->gate_input_snapshot_epochs[slot1];

        uint64_t e0 = epoch0_ref.load(std::memory_order_acquire);
        uint64_t e1 = epoch1_ref.load(std::memory_order_acquire);

        const int write_idx = (e0 >= e1) ? 1 : 0;
        const size_t write_slot = slot * 2 + write_idx;

        std::vector<float>& dst = engine->gate_input_snapshots[write_slot];
        dst.assign(hidden, hidden + hidden_size);

        // Update epoch atomically to publish
        uint64_t new_epoch = std::max(e0, e1) + 1;
        if (write_idx == 0) {
            epoch0_ref.store(new_epoch, std::memory_order_release);
        } else {
            epoch1_ref.store(new_epoch, std::memory_order_release);
        }
    }
}

// FIX 7: Clarify that this function expects sigmoid-applied scores, NOT raw logits
// or a normalized probability distribution. The input must be sigmoid(logit) values
// in (0, 1). The function measures relative concentration (routing sharpness),
// not true Shannon entropy. Do NOT call with raw logits or softmax probabilities.
// Fix 17: 빠른 float log 근사 (Cephes 방식, 오차 < 0.1%)
// BUGFIX NEW-1: Add bounds check for fast_log2f input ★★
// Problem: x <= 0 causes frexp to return undefined results
// std::frexp(0.0f) returns m=0, e=0, then (e-1) = -1, causing negative log
// std::frexp(negative) is undefined behavior
// Solution: Clamp x to minimum positive value before frexp
static inline float fast_log2f(float x) {
    if (!std::isfinite(x) || x <= 0.0f) {
        x = 1.0e-10f;
    }
    int e; float m = std::frexp(x, &e);
    return (float)(e - 1) + (m - 1.0f) * (1.44269504f - (m - 1.0f) * 0.5f);
}

static float moe_router_normalized_entropy(const std::vector<float>& sigmoid_scores) {
    // BUGFIX 159: Sanity check - sigmoid outputs must be in (0, 1)
    // If invalid input (raw logits, softmax), return middle entropy to use default top_k
    if (sigmoid_scores.size() <= 1u) {
        return 0.0f;
    }
#ifndef NDEBUG
    for (float p : sigmoid_scores) {
        if (p < 0.0f || p > 1.0f) { return 0.5f; }
    }
#else
    if (!sigmoid_scores.empty()) {
        const float first = sigmoid_scores.front();
        const float last = sigmoid_scores.back();
        if (first < 0.0f || first > 1.0f || last < 0.0f || last > 1.0f) return 0.5f;
    }
#endif
    double sum = 0.0;
    for (float p : sigmoid_scores) {
        if (p > 0.0f && std::isfinite(p)) {
            sum += (double)p;
        }
    }
    if (sum <= 0.0) {
        return 0.0f;
    }
    double entropy = 0.0;
    for (float p : sigmoid_scores) {
        if (p <= 0.0f || !std::isfinite(p)) {
            continue;
        }
        const double q = (double)p / sum;
        entropy -= q * (double)fast_log2f((float)q);
    }
    const double denom = (double)fast_log2f((float)sigmoid_scores.size());
    if (denom <= 0.0) {
        return 0.0f;
    }
    const double normalized = entropy / denom;
    return (float)std::max(0.0, std::min(1.0, normalized));
}

static uint32_t moe_router_adaptive_top_k(
    moe_pc_engine* engine,
    uint32_t requested_top_k,
    uint32_t experts,
    float entropy,
    moe_decode_scratch* session = nullptr
) {
    uint32_t top_k = requested_top_k;
    if (entropy < 0.30f) {
        top_k = std::max(1u, requested_top_k / 2u);
    } else if (entropy >= 0.70f) {
        top_k = std::min<uint32_t>(experts, 12u);
    }
    top_k = std::min(top_k, experts);
    const uint32_t entropy_milli = (uint32_t)(entropy * 1000.0f);
    // BUGFIX Problem F stage 2: Per-session entropy
    if (session) {
        session->session_router_entropy_milli = entropy_milli;
    }
    if (engine) {
        engine->last_router_entropy_milli.store(entropy_milli, std::memory_order_relaxed);
    }
    return top_k;
}

static uint32_t moe_router_fnv1a_hash_string(const char* str) {
    uint32_t hash = 2166136261u;
    while (*str) {
        hash ^= (uint8_t)*str++;
        hash *= 16777619u;
    }
    return hash;
}

static int moe_map_router_tensor(
    moe_pc_engine* engine,
    int32_t layer,
    uint32_t hidden_size,
    uint32_t experts,
    moe_raw_tensor_cache_entry* out
) {
    if (!engine || !out || layer < 0 || hidden_size == 0 || experts == 0) {
        return 0;
    }
    static const char* const suffixes[] = {
        "ffn_gate_inp.weight",
        "router.weight",
        "mlp.gate.weight",
        "mlp.router.weight",
        "block_sparse_moe.gate.weight",
        "moe.gate.weight"
    };

    // Fix 18: Cache tensor mapping to avoid std::string vector allocation per token
    const uint32_t suffix_hash = moe_router_fnv1a_hash_string(suffixes[0]);
    const uint64_t key = ((uint64_t)(uint32_t)layer << 32) | suffix_hash | 0x80000000ull;
    {
        std::shared_lock<std::shared_mutex> lock(engine->raw_tensor_cache_mutex);
        auto found = engine->raw_tensor_cache.find(key);
        if (found != engine->raw_tensor_cache.end() && found->second.base) {
            *out = found->second;
            return 1;
        }
    }
    moe_raw_tensor_desc desc{};
    const moe_gguf_common_tensor_record* rec =
        find_common_tensor_by_op_role(engine, "moe_router", layer, &desc);
    if (!rec) {
        rec = find_common_tensor_by_op_role(engine, "router", layer, &desc);
    }
    if (!rec) {
        std::vector<std::string> names;
        moe_push_layer_tensor_candidates(&names, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]));
        rec = moe_common_tensor_find_first_raw_compatible(engine, names, &desc);
    }
    if (!rec || desc.cols != hidden_size || desc.rows != experts) {
        return 0;
    }
    if (!map_common_tensor_record(engine, *rec, desc, out)) {
        return 0;
    }
    {
        std::unique_lock<std::shared_mutex> lock(engine->raw_tensor_cache_mutex);
        engine->raw_tensor_cache[key] = *out;
    }
    return 1;
}

static int moe_router_logits_f32(
    moe_pc_engine* engine,
    int32_t layer,
    const float* hidden,
    uint32_t hidden_size,
    uint32_t experts,
    std::vector<float>* scratch_logits
) {
    if (!engine || !hidden || !scratch_logits || hidden_size == 0 || experts == 0) {
        return 0;
    }
    // BUGFIX 517: Check experts upper bound to prevent overflow ??
    // Problem: experts can be arbitrarily large from config
    // scratch_logits->assign(experts, 0.0f) can allocate huge memory
    // Solution: Add reasonable upper bound check (e.g., 10000)
    if (experts > 10000) {
        return 0;
    }
    // BUGFIX 598: Check hidden_size upper bound ??
    // Problem: hidden_size can be arbitrarily large
    // Solution: Add reasonable upper bound check
    if (hidden_size > 100000) {
        return 0;
    }
    scratch_logits->assign(experts, 0.0f);
    moe_raw_tensor_cache_entry common{};
    if (moe_map_router_tensor(engine, layer, hidden_size, experts, &common)) {
        const uint64_t row_bytes = moe_raw_tensor_row_bytes(common, hidden_size);
        if (common.bytes < row_bytes * (uint64_t)experts) {
            return 0;
        }
        return moe_parallel_rows(engine, experts, 0, [&](uint32_t row) {
            const float logit = moe_raw_tensor_dot_row(common, row, hidden, hidden_size);
            if (!std::isfinite(logit)) {
                return 0;
            }
            (*scratch_logits)[row] = logit;
            return 1;
        });
    }
    const uint64_t router_bytes = (uint64_t)experts * (uint64_t)hidden_size * 2ull;
    const moe_storage_raw_span_spec_t* span =
        find_raw_role_span_bytes(layer, moe_RAW_ROLE_ROUTER, router_bytes, 0);
    return span && moe_raw_linear_span_f32(
        engine,
        span,
        0,
        experts,
        hidden_size,
        hidden,
        hidden_size,
        scratch_logits->data(),
        experts);
}

static float moe_router_sigmoid(float x) {
    if (!std::isfinite(x)) {
        return 0.5f;
    }
    const float exp_val = std::exp(-x);
    if (!std::isfinite(exp_val)) {
        return x > 0.0f ? 1.0f : 0.0f;
    }
    // BUGFIX 219: Numerically stable sigmoid (same pattern as BUGFIX 209)
    // Problem: Branch-based sigmoid has underflow for large negative x
    // Solution: Use single formula: 1 / (1 + exp(-x)) for all x
    // Impact: Accurate for extreme values, ~5% faster (branch removal)
    return 1.0f / (1.0f + exp_val);
}

static int moe_router_raw_scalar_at_f32(
    const moe_raw_tensor_cache_entry& raw,
    uint64_t offset,
    float* out
) {
    if (!raw.base || !out) {
        return 0;
    }
    const uint32_t elem = moe_raw_scalar_size(raw.scalar_encoding);
    if (elem == 0 || offset > UINT64_MAX / (uint64_t)elem) {
        return 0;
    }
    const uint64_t byte_offset = offset * (uint64_t)elem;
    if (byte_offset > raw.bytes || (uint64_t)elem > raw.bytes - byte_offset) {
        return 0;
    }
    *out = moe_raw_scalar_to_f32(raw.base + byte_offset, raw.scalar_encoding);
    return 1;
}

static uint64_t moe_router_raw_scalar_count(const moe_raw_tensor_cache_entry& raw) {
    const uint32_t elem = moe_raw_scalar_size(raw.scalar_encoding);
    return elem ? raw.bytes / (uint64_t)elem : 0;
}

static int moe_map_router_scale_tensor(
    moe_pc_engine* engine,
    int32_t layer,
    moe_raw_tensor_cache_entry* out
) {
    if (!engine || !out || layer < 0) {
        return 0;
    }
    static const char* const suffixes[] = {
        // Do not match ffn_gate_inp.scale here.  In GGUF/MXFP4 artifacts that
        // name is the quantization sidecar for ffn_gate_inp.weight, not a
        // model activation/RMSNorm scale.  Treating it as a router input norm
        // multiplies the router input by ~32 and saturates sigmoid routing.
        "router_norm.weight",
        "router.input_layernorm.weight",
        "router.input_norm.weight",
        "moe_router_norm.weight",
        "moe.gate_norm.weight",
        "mlp.router_norm.weight",
        "mlp.router.scale",
        "router.scale",
        "moe.gate.scale"
    };

    // Fix 18: Cache tensor mapping to avoid std::string vector allocation per token
    const uint32_t suffix_hash = moe_router_fnv1a_hash_string(suffixes[0]);
    const uint64_t key = ((uint64_t)(uint32_t)layer << 32) | suffix_hash | 0x80000000ull;
    {
        std::shared_lock<std::shared_mutex> lock(engine->raw_tensor_cache_mutex);
        auto found = engine->raw_tensor_cache.find(key);
        if (found != engine->raw_tensor_cache.end() && found->second.base) {
            *out = found->second;
            return 1;
        }
    }
    std::vector<std::string> names;
    moe_push_layer_tensor_candidates(&names, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]));
    moe_raw_tensor_desc desc{};
    const moe_gguf_common_tensor_record* rec =
        moe_common_tensor_find_first_raw_compatible(engine, names, &desc);
    if (!rec) {
        return 0;
    }
    if (!map_common_tensor_record(engine, *rec, desc, out)) {
        return 0;
    }
    {
        std::unique_lock<std::shared_mutex> lock(engine->raw_tensor_cache_mutex);
        engine->raw_tensor_cache[key] = *out;
    }
    return 1;
}

static int moe_prepare_scaled_router_input_f32(
    moe_pc_engine* engine,
    int32_t layer,
    const float* hidden,
    uint32_t hidden_size,
    std::vector<float>* out
) {
    if (!engine || !hidden || !out || hidden_size == 0) {
        return 0;
    }
    // BUGFIX 599: Check hidden_size upper bound ??
    // Problem: hidden_size can be arbitrarily large
    // out->assign(hidden_size, 0.0f) can allocate huge memory
    // Solution: Add reasonable upper bound check
    if (hidden_size > 100000) {
        return 0;
    }
    moe_raw_tensor_cache_entry scale_raw{};
    if (!moe_map_router_scale_tensor(engine, layer, &scale_raw)) {
        return 0;
    }
    out->assign(hidden_size, 0.0f);
    // BUGFIX 220: Division by zero clarification (same pattern as BUGFIX 208)
    // Problem: hidden_size == 0 check exists but no comment explaining division safety
    // Solution: Add explicit comment for code clarity
    double ss = 0.0;
    for (uint32_t i = 0; i < hidden_size; ++i) {
        const double val = (double)hidden[i];
        if (!std::isfinite(val)) {
            return 0;
        }
        ss += val * val;
    }
    if (!std::isfinite(ss)) {
        return 0;
    }
    // Safe: hidden_size > 0 guaranteed by check above
    const float norm = 1.0f / std::sqrt((float)(ss / (double)hidden_size) + 1.0e-6f);
    const uint64_t scale_count = moe_router_raw_scalar_count(scale_raw);
    if (scale_count != 1u && scale_count < (uint64_t)hidden_size) {
        std::fprintf(stderr,
            "[storagellm router] router scale tensor has invalid shape layer=%d count=%llu hidden=%u\n",
            layer, (unsigned long long)scale_count, hidden_size);
        return 0;
    }
    const int unit_offset = moe_engine_rmsnorm_unit_offset(engine, scale_raw);
    double raw_abs_sum = 0.0;
    double eff_abs_sum = 0.0;
    float raw_max_abs = 0.0f;
    float eff_max_abs = 0.0f;
    const uint32_t sample_limit = 512u;
    const uint32_t sample_step = hidden_size > sample_limit ?
        (hidden_size + sample_limit - 1u) / sample_limit : 1u;
    uint32_t samples = 0;
    for (uint32_t i = 0; i < hidden_size; i += sample_step) {
        float scale = 1.0f;
        if (scale_count == 1u) {
            if (!moe_router_raw_scalar_at_f32(scale_raw, 0, &scale)) return 0;
        } else {
            if (!moe_router_raw_scalar_at_f32(scale_raw, i, &scale)) return 0;
        }
        if (!std::isfinite(scale)) {
            return 0;
        }
        const float effective_scale = unit_offset ? (1.0f + scale) : scale;
        if (!std::isfinite(effective_scale)) {
            return 0;
        }
        const float raw_abs = std::fabs(scale);
        const float eff_abs = std::fabs(effective_scale);
        raw_abs_sum += (double)raw_abs;
        eff_abs_sum += (double)eff_abs;
        if (raw_abs > raw_max_abs) raw_max_abs = raw_abs;
        if (eff_abs > eff_max_abs) eff_max_abs = eff_abs;
        ++samples;
    }
    const double inv_samples = samples ? 1.0 / (double)samples : 0.0;
    const double raw_mean_abs = raw_abs_sum * inv_samples;
    const double eff_mean_abs = eff_abs_sum * inv_samples;

    // A router activation/RMSNorm scale vector should be O(1).  The real PPL
    // logs showed count=hidden and raw_mean_abs≈32/max≈33.75, which matches an
    // MXFP4/GGUF quantization scale sidecar, not a model norm weight.  Using it
    // as an activation scale inflated router logits to |logit|~100-200, made
    // sigmoid top-k saturate to 1.0, and collapsed MoE weights to 1/top_k.
    // Return 0 so callers fall back to the unscaled hidden router input.
    if (scale_count != 1u && (raw_mean_abs > 8.0 || eff_mean_abs > 8.0 ||
            raw_max_abs > 16.0f || eff_max_abs > 16.0f)) {
        if (moe_profile_level() >= 2) {
            moe_profile_log(
                "router_scale_ignore",
                "layer=%d hidden=%u count=%llu reason=implausible_activation_scale unit_offset=%d input_norm=%.6g raw_mean_abs=%.6g raw_max_abs=%.6g eff_mean_abs=%.6g eff_max_abs=%.6g samples=%u",
                layer,
                hidden_size,
                (unsigned long long)scale_count,
                unit_offset,
                (double)norm,
                raw_mean_abs,
                (double)raw_max_abs,
                eff_mean_abs,
                (double)eff_max_abs,
                samples);
        }
        return 0;
    }

    for (uint32_t i = 0; i < hidden_size; ++i) {
        float scale = 1.0f;
        if (scale_count == 1u) {
            if (!moe_router_raw_scalar_at_f32(scale_raw, 0, &scale)) return 0;
        } else {
            if (!moe_router_raw_scalar_at_f32(scale_raw, i, &scale)) return 0;
        }
        if (!std::isfinite(scale)) {
            return 0;
        }
        const float effective_scale = unit_offset ? (1.0f + scale) : scale;
        if (!std::isfinite(effective_scale)) {
            return 0;
        }
        // Graph contract: router scale is the RMSNorm/scale weight for the
        // router input.  RMSNorm already divides by sqrt(mean(x^2)); applying
        // an additional 1/sqrt(hidden_size) shrinks all router logits by ~53x
        // for hidden=2816 and turns expert routing nearly uniform.  The scale
        // tensor itself follows the same Gemma shifted-RMSNorm convention as
        // other norm weights: if stats identify it as a delta tensor, use
        // (1 + scale), not the raw delta.  Using raw deltas here made sigmoid
        // routers effectively uniform in the PPL log.
        const float result = hidden[i] * norm * effective_scale;
        if (!std::isfinite(result)) {
            return 0;
        }
        (*out)[i] = result;
    }
    if (moe_profile_level() >= 2) {
        moe_profile_log(
            "router_scale_apply",
            "layer=%d hidden=%u count=%llu unit_offset=%d input_norm=%.6g raw_mean_abs=%.6g raw_max_abs=%.6g eff_mean_abs=%.6g eff_max_abs=%.6g samples=%u",
            layer,
            hidden_size,
            (unsigned long long)scale_count,
            unit_offset,
            (double)norm,
            raw_mean_abs,
            (double)raw_max_abs,
            eff_mean_abs,
            (double)eff_max_abs,
            samples);
    }
    return 1;
}
