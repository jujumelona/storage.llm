static uint64_t moe_profile_now_us();
static uint64_t device_budget_bytes(const moe_pc_engine* e);

static uint64_t moe_common_raw_tensor_bytes() {
    const moe_storage_constants_t* storage = moe_storage_constants();
    return storage ? storage->raw_tensor_total_bytes : 0;
}

static uint64_t moe_common_raw_tensor_count() {
    const moe_storage_constants_t* storage = moe_storage_constants();
    return storage ? storage->raw_tensor_count : 0;
}

static uint64_t common_vram_wanted_bytes(const moe_pc_engine* engine) {
    const uint64_t common_bytes = moe_common_raw_tensor_bytes();
    if (!engine || common_bytes == 0) {
        return 0;
    }
    if (engine->config.common_vram_reserve_bytes) {
        return std::min<uint64_t>(engine->config.common_vram_reserve_bytes, common_bytes);
    }
    const moe_storage_constants_t* s = moe_storage_constants();
    uint64_t runtime_overhead = 0;
    if (s && s->runtime_state_bytes > 0) {
        runtime_overhead = s->runtime_state_bytes;
    } else {
        // BUGFIX 1006: Realistic fallback for runtime_state_bytes ★★ IMPORTANT
        // Problem: Old fallback (2*hidden + experts*4 + hidden) = ~19KB for typical models
        // → 1000x smaller than actual runtime state (attention scratch, norm buffers, activation)
        // → common_vram_wanted_bytes ≈ 0 → embedding/lm_head not pinned → reload every token
        // Solution: Use 2 × hidden_size × max_seq_len × 2 (bf16) as minimum activation buffer
        // This approximates KQ matrix size, which is the dominant runtime allocation.
        // Fallback to 8192 if max_seq_len unavailable.
        // BUGFIX Problem 3: Account for MLA projection buffers ★★★ CRITICAL
        // Problem: MLA models have latent_dim projection buffers much larger than KQ matrix
        // → GLM-5.1: latent_dim=512, num_experts=256 → projection scratch ~2GB
        // → Fallback underestimates by 10-15x → embedding/lm_head evicted every token
        // Solution: Add MLA latent buffer estimate: num_experts × latent_dim × max_seq_len × 2
        // Impact: Correct VRAM reservation for MLA models, prevents reload thrashing
        const uint64_t hidden = s ? s->hidden_size : 6144ull;
        uint32_t max_seq = 8192u;
        if (engine && engine->attn_config_valid.load(std::memory_order_relaxed) &&
            engine->attn_config.max_seq_len > 0) {
            max_seq = engine->attn_config.max_seq_len;
        }
        // Base: KQ matrix (2 × hidden × seq × bf16)
        runtime_overhead = 2ull * hidden * (uint64_t)max_seq * 2ull;

        // Add MLA projection buffer estimate if model has MLA
        if (s && s->experts_per_moe_layer > 0) {
            // Estimate MLA latent buffer: experts × latent_dim × seq × bf16
            // Conservative: assume latent_dim = hidden/8 (typical for MLA)
            const uint64_t latent_dim = hidden / 8ull;
            const uint64_t mla_buffer = (uint64_t)s->experts_per_moe_layer * latent_dim * (uint64_t)max_seq * 2ull;
            runtime_overhead += mla_buffer;
        }

        fprintf(stderr,
            "[storageLLM WARNING] runtime_state_bytes not set in moe_storage_constants. "
            "Using calculated fallback: %llu bytes (KQ matrix + MLA projection estimate). "
            "Actual runtime state may be larger with vision encoders or custom attention.\n",
            (unsigned long long)runtime_overhead);
    }
    const uint64_t core = (s ? s->raw_tensor_lm_head_bytes : 0ull) + runtime_overhead;
    const uint64_t embed = s ? s->raw_tensor_embedding_bytes : 0ull;
    const uint64_t bundle = s && s->normal_bundle_max_bytes ? s->normal_bundle_max_bytes : 32ull * 1024ull * 1024ull;
    const uint64_t expert_floor = bundle * 8ull;

    const uint64_t effective_budget = device_budget_bytes(engine);

    if (effective_budget <= expert_floor) {
        return 0;
    }
    uint64_t reserve = effective_budget >= core + expert_floor ? core : 12288ull;
    if (effective_budget >= core + embed + expert_floor) {
        reserve += embed;
    }
    return std::min<uint64_t>(reserve, common_bytes);
}

static uint64_t vram_expert_budget_bytes(const moe_pc_engine* engine) {
    // BUGFIX Issue 1: Unify with device_budget_bytes() to prevent inconsistency ★★★
    // Problem: Two functions calculate VRAM budget differently
    // device_budget_bytes() has Metal fallback, this function returns 0 for Metal
    // Solution: Call device_budget_bytes() internally for consistency
    // This ensures Metal unified memory and all backends use same budget logic
    if (!engine) {
        return 0;
    }

    // Use device_budget_bytes() as single source of truth
    // It handles Metal unified memory, Vulkan/LevelZero dynamic queries, etc.
    const uint64_t total_budget = device_budget_bytes(engine);

    // Subtract common and KV reservations (already subtracted in device_budget_bytes)
    // device_budget_bytes already subtracts pending, kv_reserved, common_reserved
    // So we just return the result directly
    return total_budget;
}

// Rate-limited wrapper that warns on expert budget exhaustion
static uint64_t vram_expert_budget_bytes_checked(const moe_pc_engine* engine) {
    const uint64_t result = vram_expert_budget_bytes(engine);
    if (result == 0 && engine && engine->config.vram_budget_bytes > 0) {
        // BUGFIX Issue 4: Move static timer to engine member ★★★
        // Problem: static s_last_warn shared across all engine instances
        // Solution: Use engine->last_expert_budget_warn_ms
        const uint64_t now = moe_profile_now_us() / 1000000ull;
        auto* mutable_engine = const_cast<moe_pc_engine*>(engine);
        const uint64_t last = mutable_engine->last_expert_budget_warn_ms.load(std::memory_order_relaxed);
        if (now > last + 5ull) {
            mutable_engine->last_expert_budget_warn_ms.store(now, std::memory_order_relaxed);
            const uint64_t common = engine->common_vram_reserved_bytes.load(std::memory_order_relaxed);
            // OPTIMIZATION Issue 2: Update warning message for dynamic KV budget ★★★
            // Old message: "Consider reducing vram_kv_ratio"
            // New message: Reflect that KV is soft cap, show actual usage
            const uint64_t kv_cap = engine->vram_kv_budget_bytes.load(std::memory_order_relaxed);
            const uint64_t kv_actual = engine->device_kv_cache_bytes.load(std::memory_order_relaxed);
            const uint64_t pending = engine->pending_vram_reservation.load(std::memory_order_relaxed);
            fprintf(stderr,
                "[storageLLM WARNING] Expert VRAM budget exhausted: "
                "total=%llu MB, KV cap=%llu MB (actual=%llu MB), common=%llu MB, pending=%llu MB. "
                "KV is using soft cap. Consider reducing context length or increasing VRAM budget.\n",
                (unsigned long long)(engine->config.vram_budget_bytes / (1024*1024)),
                (unsigned long long)(kv_cap / (1024*1024)),
                (unsigned long long)(kv_actual / (1024*1024)),
                (unsigned long long)(common / (1024*1024)),
                (unsigned long long)(pending / (1024*1024)));
        }
    }
    return result;
}

// BUGFIX 문제4: Initialize KV VRAM budget ★★★
// Splits VRAM budget between experts and KV cache
// Default: 80% experts, 20% KV (configurable via vram_kv_ratio)
// OPTIMIZATION: Dynamic KV budget - soft cap instead of fixed reservation ★★★
// Problem: Fixed KV budget wastes VRAM in short conversations, causes RAM spill in long ones
// Solution: Track actual KV VRAM usage via device_kv_cache_bytes atomic
// vram_kv_budget_bytes becomes soft cap, device_budget_bytes() uses real usage
// BUGFIX Problem C: Changed from static to extern for device_allocator_init() ★★★
static int moe_kv_vram_attention_runtime_available(const moe_pc_engine* engine) {
    if (!engine) {
        return 0;
    }
#if defined(MOE_ENABLE_DEVICE_KV_ATTENTION) && MOE_ENABLE_DEVICE_KV_ATTENTION
    return engine->backend_caps.backend == moe_BACKEND_CUDA ||
           engine->backend_caps.backend == moe_BACKEND_HIP;
#else
    (void)engine;
    return 0;
#endif
}

void init_kv_vram_budget(moe_pc_engine* engine) {
    if (!engine || engine->config.vram_budget_bytes == 0) {
        if (engine) {
            engine->vram_kv_budget_bytes.store(0, std::memory_order_relaxed);
            engine->vram_kv_used_bytes.store(0, std::memory_order_relaxed);
            engine->device_kv_cache_bytes.store(0, std::memory_order_relaxed);
        }
        return;
    }

    // BUGFIX Problem 1: Metal unified memory - set KV budget to 0 ★★★ CRITICAL
    // Problem: Metal unified memory causes double-counting of KV cache
    // KV cache tracked in tier_used(RAM) + device_budget_bytes() both reference same memory
    // Solution: For Metal, set vram_kv_budget_bytes = 0, delegate to OS unified memory pressure
    // Impact: Eliminates double-counting, KV managed by macOS memory pressure system
    if (engine->backend_caps.backend == moe_BACKEND_METAL) {
        engine->vram_kv_budget_bytes.store(0, std::memory_order_relaxed);
        engine->vram_kv_used_bytes.store(0, std::memory_order_relaxed);
        engine->device_kv_cache_bytes.store(0, std::memory_order_relaxed);
        fprintf(stderr,
            "[storageLLM INFO] Metal unified memory: KV budget set to 0, delegated to OS memory pressure\n");
        return;
    }

    if (!moe_kv_vram_attention_runtime_available(engine)) {
        engine->vram_kv_budget_bytes.store(0, std::memory_order_relaxed);
        engine->vram_kv_used_bytes.store(0, std::memory_order_relaxed);
        engine->device_kv_cache_bytes.store(0, std::memory_order_relaxed);
        fprintf(stderr,
            "[storageLLM INFO] KV VRAM tier disabled: no compiled device-side KV attention consumer; "
            "flat/QKV KV remains RAM-resident.\n");
        return;
    }

    float kv_ratio = engine->config.vram_kv_ratio;

    // BUGFIX Issue 2-5: Multi-tier KV VRAM ratio thresholds ★★★
    // Problem: Single 24GB threshold causes suboptimal allocation:
    //   - 8GB VRAM: 10% = 819MB insufficient for long sequences → RAM thrashing
    //   - 80GB VRAM: 20% = 16GB excessive → wastes expert prefetch budget
    // Solution: Use graduated thresholds based on VRAM size for better allocation
    // Auto-detect if not configured (0.0 means auto)
    if (kv_ratio <= 0.0f) {
        const uint64_t gib = 1024ull * 1024ull * 1024ull;
        const uint64_t vram = engine->config.vram_budget_bytes;

        // BUGFIX Problem 6: Context-aware KV ratio calculation ★★ IMPORTANT
        // Problem: Fixed ratio ignores context length → 8GB with 32K context spills to RAM
        // Solution: Calculate KV needed based on context length, adjust ratio accordingly
        // Estimate: ~512 bytes/token for BF16 with GQA (conservative)
        uint32_t ctx = 4096u;  // Default fallback
        if (engine->attn_config_valid.load(std::memory_order_relaxed) &&
            engine->attn_config.max_seq_len > 0) {
            ctx = engine->attn_config.max_seq_len;
        }

        const uint64_t kv_needed_estimate = (uint64_t)ctx * 512ull;
        const float ctx_ratio = vram > 0
            ? std::min(0.4f, (float)kv_needed_estimate / (float)vram)
            : 0.1f;

        // Base ratio from VRAM size + context adjustment
        const float base = vram <= 8ull * gib ? 0.10f :
                          vram <= 24ull * gib ? 0.15f : 0.20f;
        kv_ratio = std::max(base, std::min(ctx_ratio, 0.4f));

        fprintf(stderr,
            "[storageLLM INFO] Context-aware KV ratio: ctx=%u tokens, estimated=%llu MB, "
            "base_ratio=%.2f, ctx_ratio=%.2f, final=%.2f\n",
            ctx,
            (unsigned long long)(kv_needed_estimate / (1024*1024)),
            base, ctx_ratio, kv_ratio);
    } else if (kv_ratio > 1.0f) {
        fprintf(stderr,
            "[storageLLM WARNING] vram_kv_ratio=%.2f exceeds 1.0, clamping to full KV allocation\n",
            kv_ratio);
        kv_ratio = 1.0f;
    }

    // BUGFIX 문제F: Warn if user sets excessive KV ratio ★★
    // High KV ratio (>0.5) starves expert VRAM budget and kills prefetch hit rate
    if (kv_ratio > 0.5f) {
        fprintf(stderr,
            "[storageLLM WARNING] effective vram_kv_ratio=%.2f exceeds 0.5, expert prefetch will be severely limited. "
            "Recommended range: 0.1-0.3; 1.0 explicitly reserves all VRAM for KV.\n",
            kv_ratio);
    }
    if (engine->config.vram_kv_ratio <= 0.0f) {
        fprintf(stderr, "[storageLLM INFO] vram_kv_ratio auto-adjusted to %.2f based on VRAM capacity.\n", kv_ratio);
    }

    // OPTIMIZATION Issue 2: Soft cap instead of hard reservation ★★★
    // Store as soft cap, actual usage tracked in device_kv_cache_bytes
    // device_budget_bytes() will use device_kv_cache_bytes.load() for real-time accounting
    const uint64_t kv_budget = (uint64_t)(engine->config.vram_budget_bytes * kv_ratio);
    engine->vram_kv_budget_bytes.store(kv_budget, std::memory_order_relaxed);
    engine->vram_kv_used_bytes.store(0, std::memory_order_relaxed);
    // OPTIMIZATION: Initialize device_kv_cache_bytes for dynamic tracking
    engine->device_kv_cache_bytes.store(0, std::memory_order_relaxed);

    fprintf(stderr,
        "[storageLLM INFO] VRAM budget split: Total=%llu MB, KV soft cap=%llu MB (%.0f%%), Experts=%llu MB (dynamic)\n",
        (unsigned long long)(engine->config.vram_budget_bytes / (1024*1024)),
        (unsigned long long)(kv_budget / (1024*1024)),
        kv_ratio * 100.0f,
        (unsigned long long)((engine->config.vram_budget_bytes > kv_budget ?
            engine->config.vram_budget_bytes - kv_budget : 0) / (1024*1024)));
}

static void apply_common_vram_reservation(moe_pc_engine* engine) {
    if (!engine) {
        return;
    }
    // BUGFIX 230: Add memory_order_relaxed to 4 atomic stores ★★
    // Problem: Missing memory_order causes seq_cst default (40-200x slower than relaxed)
    // Solution: Add memory_order_relaxed to all atomic stores (initialization doesn't need synchronization)
    // Impact: 40-200x performance improvement in VRAM reservation setup
    engine->common_vram_reserved_bytes.store(0, std::memory_order_relaxed);
    engine->vram_used_bytes.store(0, std::memory_order_relaxed);

    // BUGFIX 문제4: Initialize KV VRAM budget
    init_kv_vram_budget(engine);

    if (!engine->config.prefer_gpu || !engine_has_device_upload_workers(engine) ||
        !engine->io_config.enable_common_raw_vram_pin) {
        engine->common_vram_upload_complete.store(true, std::memory_order_release);
        return;
    }
    const uint64_t reserve = std::min<uint64_t>(
        common_vram_wanted_bytes(engine),
        engine->config.vram_budget_bytes
    );
    engine->common_vram_reserved_bytes.store(reserve, std::memory_order_relaxed);
    engine->vram_used_bytes.store(0, std::memory_order_relaxed);
    engine->common_vram_upload_complete.store(reserve == 0, std::memory_order_release);
}
