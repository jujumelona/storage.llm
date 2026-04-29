static int glm5_layer_attention_f32(
    glm5_pc_engine* engine,
    int32_t layer,
    glm5_decode_scratch* s,
    float* hidden,
    uint32_t hidden_size,
    uint32_t position
) {
    constexpr uint32_t cdim = GLM5_MLA_KV_COMPRESSED_DIM; // 512
    constexpr uint32_t num_heads = GLM5_MLA_NUM_HEADS;     // 64
    constexpr uint32_t head_dim = GLM5_MLA_HEAD_DIM;       // 256
    constexpr uint32_t nope_dim = GLM5_MLA_NOPE_DIM;       // 192
    constexpr uint32_t rope_dim = GLM5_MLA_ROPE_DIM;       // 64
    constexpr uint32_t v_head_dim = GLM5_MLA_V_HEAD_DIM;   // 256
    constexpr uint64_t kv_b_row_bytes = cdim * 2ull;       // bf16

    // --- Pre-attention RMSNorm ---
    if (!glm5_pc_engine_raw_rmsnorm_f32(
            engine, layer, GLM5_RAW_TENSOR_INPUT_LAYER_NORM,
            hidden, hidden_size, s->norm.data(), hidden_size, 1.0e-6f)) {
        return 0;
    }

    // --- KV path: kv_a_proj -> kv_a_norm (NO kv_b_proj; absorbed) ---
    // kv_a is small (576 outputs) and runs inline. No std::async: creating
    // OS threads for 80 layers times N tokens is too expensive per inference.
    if (!glm5_pc_engine_raw_linear_f32(
            engine, layer, GLM5_RAW_TENSOR_KV_A_PROJ_WITH_MQA,
            s->norm.data(), hidden_size, s->attn_kv_a.data(), GLM5_MLA_KV_A_PROJ_DIM)) {
        return 0;
    }
    if (!glm5_pc_engine_raw_rmsnorm_f32(
            engine, layer, GLM5_RAW_TENSOR_KV_A_LAYER_NORM,
            s->attn_kv_a.data(), cdim, s->attn_kv_norm.data(), cdim, 1.0e-6f)) {
        return 0;
    }
    float* k_rope = s->attn_kv_a.data() + cdim;

    // --- Q path: q_a_proj -> q_a_norm -> q_b_proj ---
    if (!glm5_pc_engine_raw_linear_f32(
            engine, layer, GLM5_RAW_TENSOR_Q_A_PROJ,
            s->norm.data(), hidden_size, s->q_a.data(), 2048u) ||
        !glm5_pc_engine_raw_rmsnorm_f32(
            engine, layer, GLM5_RAW_TENSOR_Q_A_LAYER_NORM,
            s->q_a.data(), 2048u, s->q_a_norm.data(), 2048u, 1.0e-6f) ||
        !glm5_pc_engine_raw_linear_f32(
            engine, layer, GLM5_RAW_TENSOR_Q_B_PROJ,
            s->q_a_norm.data(), 2048u, s->q_b.data(), GLM5_MLA_Q_B_DIM)) {
        return 0;
    }

    // Apply K RoPE before storing the current KV entry. Q RoPE is fused below
    // with Q-side absorption and historical KV decode in one barrier.
    glm5_rope_inplace(k_rope, GLM5_MLA_K_ROPE_DIM, position, s->rope_k_cos, s->rope_k_sin);

    // Cache compressed KV + RoPE'd k_rope as bf16
    if (position >= s->kv_cache.max_seq_len) {
        uint32_t new_cap = s->kv_cache.max_seq_len > 0 ? s->kv_cache.max_seq_len : 1u;
        while (new_cap <= position && new_cap < GLM5_MLA_MAX_SEQ_LEN) {
            new_cap *= 2u;
        }
        if (new_cap > GLM5_MLA_MAX_SEQ_LEN) {
            new_cap = GLM5_MLA_MAX_SEQ_LEN;
        }
        const uint64_t old_bytes = glm5_kv_cache_bytes(s->kv_cache.num_layers, s->kv_cache.max_seq_len);
        const uint64_t new_bytes = glm5_kv_cache_bytes(s->kv_cache.num_layers, new_cap);
        // Bug 8: Account for realloc peak (old + new simultaneously in memory during memcpy)
        // ensure_capacity allocates new_data, copies from data, then moves
        // Peak memory = old_bytes + new_bytes during the copy phase
        const uint64_t realloc_peak = old_bytes + new_bytes;
        const uint64_t ram_budget = tier_budget(engine, GLM5_TIER_RAM);
        if (ram_budget > 0) {
            const uint64_t ram_used = tier_used(engine, GLM5_TIER_RAM);
            // Reserve peak memory to prevent underestimating RAM pressure
            if (ram_used + realloc_peak > ram_budget) {
                std::vector<std::pair<int,int>> to_free;
                demote_cold_until(engine, GLM5_TIER_RAM, realloc_peak, to_free);
                for (auto& p : to_free) {
                    device_free_expert(engine, p.first, p.second);
                }
            }
        }
    }
    s->kv_cache.ensure_capacity(position);
    glm5_kv_cache_track_ram(engine, &s->kv_cache);
    if (position < s->kv_cache.max_seq_len) {
        // Build float32 entry, then store as bf16
        float kv_entry[GLM5_MLA_KV_ENTRY_DIM];
        std::memcpy(kv_entry, s->attn_kv_norm.data(), cdim * sizeof(float));
        std::memcpy(kv_entry + cdim, k_rope, GLM5_MLA_K_ROPE_DIM * sizeof(float));
        s->kv_cache.store_entry((uint32_t)layer, position, kv_entry, GLM5_MLA_KV_ENTRY_DIM);
    }
    s->kv_cache.seq_len = std::min(position + 1u, s->kv_cache.max_seq_len);
    const uint32_t seq_len = std::min(position + 1u, s->kv_cache.max_seq_len);

    // --- Cached W_kv_b for KV absorption ---
    glm5_raw_tensor_cache_entry kv_b_raw{};
    if (!get_cached_raw_tensor(engine, layer, GLM5_RAW_TENSOR_KV_B_PROJ, &kv_b_raw) ||
        kv_b_raw.cols != cdim || kv_b_raw.rows < (GLM5_MLA_K_NOPE_TOTAL_DIM + GLM5_MLA_V_TOTAL_DIM)) {
        return 0;
    }
    const uint8_t* kv_b_base = kv_b_raw.base;
    if (!kv_b_base) return 0;

    // Q RoPE, Q-side absorption, and historical KV decode are independent.
    // Run them in one flat dispatch to avoid three separate row-worker barriers.
    std::fill(s->q_absorbed.begin(), s->q_absorbed.end(), 0.0f);
    const uint32_t decoded_kv_count = position < seq_len ? position : seq_len;
    std::vector<float>* layer_decode = nullptr;
    uint32_t decode_start = 0;
    if (decoded_kv_count > 0 &&
        !glm5_kv_decode_layer_cache_prepare(
            engine, s, (uint32_t)layer, decoded_kv_count, &layer_decode, &decode_start)) {
        return 0;
    }
    const uint32_t decode_delta = decoded_kv_count > decode_start ? decoded_kv_count - decode_start : 0u;
    const uint32_t fused_rows = num_heads + decode_delta;
    if (fused_rows > 0) {
        if (!glm5_parallel_rows(engine, fused_rows, 0, [&](uint32_t row) {
            if (row < num_heads) {
                const uint32_t h = row;
                glm5_rope_inplace(
                    s->q_b.data() + h * head_dim + nope_dim,
                    rope_dim,
                    position,
                    s->rope_q_cos,
                    s->rope_q_sin
                );
                float* qa_h = s->q_absorbed.data() + h * cdim;
                for (uint32_t d = 0; d < nope_dim; ++d) {
                    const float qv = s->q_b[h * head_dim + d];
                    if (std::abs(qv) < 1e-12f) continue;
                    const uint8_t* w_row = kv_b_base + kv_b_row_bytes * (h * nope_dim + d);
                    glm5_axpy_bf16_row_f32(qa_h, w_row, qv, cdim);
                }
            } else {
                const uint32_t p = decode_start + (row - num_heads);
                float* decode = layer_decode->data() + (size_t)p * GLM5_MLA_KV_ENTRY_DIM;
                s->kv_cache.load_entry((uint32_t)layer, p, decode, GLM5_MLA_KV_ENTRY_DIM);
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
    const float scale = 1.0f / std::sqrt((float)head_dim);
    std::fill(s->c_weighted.begin(), s->c_weighted.end(), 0.0f);
    s->head_max.assign(num_heads, -1e30f);
    s->head_denom.assign(num_heads, 0.0f);

    glm5_parallel_rows(engine, num_heads, 0, [&](uint32_t h) {
        const float* qa_h = s->q_absorbed.data() + h * cdim;
        const float* q_rope_h = s->q_b.data() + h * head_dim + nope_dim;
        float* cw_h = s->c_weighted.data() + h * cdim;
        float h_max = -1e30f;
        float h_denom = 0.0f;

        uint32_t loop_end = position < s->kv_cache.max_seq_len ? seq_len : seq_len + 1;
        for (uint32_t p = 0; p < loop_end; ++p) {
            const float* c_kv;
            const float* p_k_rope;
            if (p == position || (position >= s->kv_cache.max_seq_len && p == seq_len)) {
                c_kv = s->attn_kv_norm.data();
                p_k_rope = k_rope;
            } else {
                const float* decode = layer_decode->data() + (size_t)p * GLM5_MLA_KV_ENTRY_DIM;
                c_kv = decode;
                p_k_rope = decode + cdim;
            }

            float score = glm5_dot_f32(qa_h, c_kv, cdim);
            score += glm5_dot_f32(q_rope_h, p_k_rope, std::min(rope_dim, GLM5_MLA_K_ROPE_DIM));
            score *= scale;

            if (score > h_max) {
                const float rescale = std::exp(h_max - score);
                h_denom *= rescale;
                glm5_scale_f32(cw_h, rescale, cdim);
                h_max = score;
            }
            const float w = std::exp(score - h_max);
            h_denom += w;
            glm5_axpy_f32(cw_h, c_kv, w, cdim);
        }

        // Normalize
        if (h_denom > 0.0f) {
            const float inv = 1.0f / h_denom;
            glm5_scale_f32(cw_h, inv, cdim);
        }
        s->head_max[h] = h_max;
        s->head_denom[h] = h_denom;
        return 1;
    });

    // --- V-side absorption: head-parallel expansion ---
    const uint8_t* kv_b_v_base = kv_b_base + kv_b_row_bytes * GLM5_MLA_K_NOPE_TOTAL_DIM;
    glm5_parallel_rows(engine, num_heads, 0, [&](uint32_t h) {
        float* out_h = s->attn_value.data() + h * v_head_dim;
        const float* cw_h = s->c_weighted.data() + h * cdim;
        for (uint32_t d = 0; d < v_head_dim; ++d) {
            const uint8_t* w_row = kv_b_v_base + kv_b_row_bytes * (h * v_head_dim + d);
            out_h[d] = glm5_dot_bf16_row(w_row, cw_h, cdim);
        }
        return 1;
    });

    // --- O projection: attention value width -> hidden_size ---
    if (!glm5_pc_engine_raw_linear_f32(
            engine, layer, GLM5_RAW_TENSOR_O_PROJ,
            s->attn_value.data(), GLM5_MLA_V_TOTAL_DIM, s->attn.data(), hidden_size)) {
        return 0;
    }
    for (uint32_t i = 0; i < hidden_size; ++i) {
        hidden[i] += s->attn[i];
    }
    return 1;
}
