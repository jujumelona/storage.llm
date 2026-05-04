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

    // BUGFIX 82: Issue Y - Parallelize Q and KV projections ????
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

    // BUGFIX 45: Issue 11 - Proactive KV cache pre-allocation ????
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
        if (!s->kv_cache.ensure_capacity(cfg.max_seq_len - 1u)) {
            fprintf(stderr, "[storageLLM ERROR] KV cache pre-allocation failed at layer %d. RAM exhausted. Aborting generation.\n", layer);
            return 0;
        }
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
        // BUGFIX 65: Issue 9 - Handle ensure_capacity failure ★★★
        // Problem: KV growth can fail when RAM is exhausted or another thread allocates first.
        // Solution: return 0 on allocation failure instead of letting generation continue.
        // Impact: Explicit error handling instead of silent data corruption
        // BUGFIX Issue ⑬: Reduce ensure_capacity call overhead ★★
        // Problem: ensure_capacity called every token, even when capacity sufficient
        // Solution: Only call when position approaches max_seq_len
        if (position >= s->kv_cache.max_seq_len) {
            if (!s->kv_cache.ensure_capacity(position)) {
                fprintf(stderr, "[storageLLM ERROR] KV cache allocation failed at position %u, "
                    "layer %d. RAM exhausted. Aborting generation.\n", position, layer);
                return 0;
            }
            moe_kv_cache_track_ram(engine, &s->kv_cache);
        }
    }
    // BUGFIX 651: Fix KV cache storage logic - remove redundant check ★★★
    // Problem: Old code checked position >= max_seq_len, returned error, then checked position < max_seq_len
    //          If ensure_capacity failed, max_seq_len not updated → entry not stored → attention corrupted
    //          Redundant checks caused confusion and potential storage skip
    // Solution: Single check with clear error path - if capacity insufficient, abort; otherwise store
    // Impact: Ensures KV entries are always stored when capacity is available → fixes PPL corruption
    if (position >= s->kv_cache.max_seq_len) {
        fprintf(stderr, "[storageLLM ERROR] KV cache capacity insufficient: "
            "position %u >= max_seq_len %u, layer %d. Aborting generation.\n",
            position, s->kv_cache.max_seq_len, layer);
        return 0;
    }
    // Build float32 entry, then store as bf16
    if (s->kv_entry.size() < kv_entry_dim) {
        s->kv_entry.resize(kv_entry_dim);
    }
    std::memcpy(s->kv_entry.data(), s->attn_kv_norm.data(), cdim * sizeof(float));
    std::memcpy(s->kv_entry.data() + cdim, k_rope, rope_dim * sizeof(float));
    s->kv_cache.store_entry((uint32_t)layer, position, s->kv_entry.data(), kv_entry_dim);
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
                // BUGFIX 292: Check (h+1)*cdim doesn't exceed buffer before accessing.
                // Off-by-one fix: original checked h*cdim > size, but we access [h*cdim .. (h+1)*cdim-1]
                if ((uint64_t)(h + 1) * (uint64_t)cdim > s->q_absorbed.size()) {
                    return 0;
                }
                float* qa_h = s->q_absorbed.data() + h * cdim;
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

    // BUGFIX 70: Issue 8 - Remove QKV dead code stub ????
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

        // BUGFIX 42: Fix softmax loop boundary to prevent out-of-bounds access ????
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
                // BUGFIX 652: Prevent numerical underflow in softmax rescaling ★★
                // Problem: exp(h_max - score) with h_max << score causes underflow
                //          Repeated underflow → h_denom approaches 0 → inaccurate normalization
                // Solution: Clamp rescale to prevent extreme underflow (min 1e-30f)
                // Impact: Improves numerical stability in attention with large score ranges
                const float rescale_exp = h_max - score;
                const float rescale = rescale_exp > -69.0f ? std::exp(rescale_exp) : 1e-30f;
                h_denom *= rescale;
                moe_scale_f32(cw_h, rescale, cdim);
                h_max = score;
            }
            const float w = std::exp(score - h_max);
            h_denom += w;
            moe_axpy_f32(cw_h, c_kv, w, cdim);
        }

        // Normalize
        // BUGFIX 653: Softmax normalization threshold consistency ★
        // Problem: Used h_denom > 0.0f check, but QKV mode uses > 1e-10f
        // Solution: Match QKV threshold for consistency across modes
        // Impact: Prevents division by near-zero values, consistent PPL behavior
        if (h_denom > 1e-10f) {
            const float inv = 1.0f / h_denom;
            moe_scale_f32(cw_h, inv, cdim);
        } else {
            // Uniform fallback when denominator too small (should be rare after rescaling fix)
            const float uniform = 1.0f / (float)loop_end;
            for (uint32_t i = 0; i < cdim; ++i) {
                cw_h[i] = uniform;
            }
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
