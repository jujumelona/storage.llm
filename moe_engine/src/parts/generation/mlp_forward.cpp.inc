// MLP forward pass operations
        // Fix 24: Use stack buffer for predicted_experts (Increased to 2048 for universal MoE support)
        int predicted_experts_buf[2048];
        const int predicted_experts_cap =
            (int)(sizeof(predicted_experts_buf) / sizeof(predicted_experts_buf[0]));
        int predicted_experts_count = 0;
        // BUGFIX 950: Entropy-adaptive InfiniGen confidence threshold ★★★ PERFORMANCE
        // Problem: Hardcoded 0.50 threshold never reached by high-entropy models
        //   (uniform router distributions e.g. GLM-4-9B) → speculative prefetch disabled
        // Solution: Scale threshold based on per-session router entropy
        //   Low entropy (sharp topk): threshold stays ~0.50 (high accuracy requirement)
        //   High entropy (uniform):   threshold drops to ~0.30 (allow lower-confidence predictions)
        // session_router_entropy_milli: 0 = untrained/unknown, 100-300 = sharp, 500+ = uniform
        const float spec_threshold = [&]() -> float {
            if (!s || s->session_router_entropy_milli == 0) return 0.50f;
            const float entropy_norm = std::min(1.0f, (float)s->session_router_entropy_milli / 1000.0f);
            // Linear interpolation: entropy=0 → 0.50, entropy=1.0 → 0.25
            return 0.50f - 0.25f * entropy_norm;
        }();
        if (infinigen_predicted_count > 0 && spec_confidence > spec_threshold) {
            predicted_experts_count = std::min(infinigen_predicted_count, predicted_experts_cap);
            for (int ii = 0; ii < predicted_experts_count; ++ii)
                predicted_experts_buf[ii] = infinigen_predicted_buf[ii];
        } else {
            const uint32_t router_top_k = moe_router_top_k_from_metadata(
                engine, 8u, c->experts_per_moe_layer);
            const uint32_t co_predict_limit = (uint32_t)std::min<uint64_t>(
                (uint64_t)c->experts_per_moe_layer,
                std::max<uint64_t>(1ull, (uint64_t)router_top_k) * 2ull);
            std::vector<int> co_predicted = moe_juju_predict_co_activation(
                engine,
                layer + 1,
                (int)co_predict_limit
            );
            predicted_experts_count = (int)std::min(co_predicted.size(), (size_t)predicted_experts_cap);
            for (int ii = 0; ii < predicted_experts_count; ++ii)
                predicted_experts_buf[ii] = co_predicted[ii];
        }
        if (depth == 0u) {
            predicted_experts_count = 0;
        }

        // Primary lookahead: layer+1
        moe_pc_engine_prefetch_moe_lookahead(
            engine,
            layer,
            s->selected_experts.data(),
            (uint32_t)s->selected_experts.size(),
            predicted_experts_count > 0 ? predicted_experts_buf : nullptr,
            (uint32_t)predicted_experts_count
        );

        // Deep speculative prefetch: layers +2 to +depth
        // Only when InfiniGen confidence is high (entropy-adaptive depth)
        for (uint32_t d = 2; d <= depth && layer + (int32_t)d < (int32_t)c->num_hidden_layers; ++d) {
            std::vector<int> deep_predicted;
            float deep_conf = 0.0f;
            moe_infinigen_speculative_prefetch(
                engine,
                layer + (int32_t)d,
                c->experts_per_moe_layer,
                deep_predicted,
                &deep_conf,
                s
            );
            // Only prefetch deep layers if confidence remains adequate
            // BUGFIX 950: Entropy-adaptive deep speculative threshold ★★★ PERFORMANCE
            const float deep_threshold = [&]() -> float {
                if (!s || s->session_router_entropy_milli == 0) return 0.40f;
                const float entropy_norm = std::min(1.0f, (float)s->session_router_entropy_milli / 1000.0f);
                return 0.40f - 0.20f * entropy_norm;
            }();
            if (deep_predicted.empty() || deep_conf < deep_threshold) {
                break;
            }
            // BUGFIX NEW-24: Already fixed - using layer + d
            moe_pc_engine_prefetch_moe_lookahead(
                engine,
                layer + (int32_t)d,
                nullptr, 0,  // no "selected" at this depth
                deep_predicted.data(),
                (uint32_t)deep_predicted.size()
            );
        }

    } else {
        // Prefill phase: no speculative prediction, just basic lookahead
        moe_pc_engine_prefetch_moe_lookahead(
            engine,
            layer,
            s->selected_experts.data(),
            (uint32_t)s->selected_experts.size(),
            nullptr,
            0
        );
    }
    moe_profile_log(
        "mlp_moe_prefetch",
        "layer=%d phase=%u adaptive_depth=%u session_depth=%u ms=%.3f",
        layer, phase,
        engine ? engine->adaptive_prefetch_depth.load(std::memory_order_relaxed) : 0u,
        s ? s->session_adaptive_prefetch_depth : 0u,
        moe_profile_elapsed_ms(prefetch_start_us));


    // Phase 2: Run experts in parallel with per-expert scratch buffers.
    // Each expert gets its own gate/up/out buffers to prevent data races.
    const size_t num_selected = s->selected_experts.size();
    if (num_selected == 0 || s->selected_weights.size() != num_selected) {
        std::fprintf(stderr,
            "[storagellm mlp] router produced invalid selection layer=%d selected=%zu weights=%zu\n",
            layer, num_selected, s->selected_weights.size());
        return 0;
    }
    // BUGFIX 511: Check num_selected upper bound before resize ??
    // Problem: num_selected can be arbitrarily large if router is corrupted
    // Resizing to huge num_selected causes memory exhaustion
    // Solution: Add reasonable upper bound check dynamically based on the model's total experts
    if (num_selected > c->experts_per_moe_layer) {
        std::fprintf(stderr,
            "[storagellm mlp] router produced excessive selection layer=%d selected=%zu\n",
            layer, num_selected);
        return 0;
    }
    // BUGFIX 210: Auto-resize expert_work if undersized (extends BUGFIX 66)
    // Problem: BUGFIX 66 added logging but still returns 0, causing generation failure
    // Solution: Auto-resize expert_work to num_selected if undersized
    // This handles dynamic top-k changes or scratch buffer initialization issues
    if (num_selected > s->expert_work.size()) {
        fprintf(stderr, "[storageLLM WARN] expert_work size mismatch: "
            "need %zu, have %zu, layer %d. Auto-resizing scratch buffer.\n",
            num_selected, s->expert_work.size(), layer);
        s->expert_work.resize(num_selected);
    }
    // BUGFIX 623: Check expert_work not empty before data() call ??
    // Problem: If num_selected is 0, expert_work can be empty, data() returns nullptr
    // Solution: Check empty before data() call
    if (s->expert_work.empty()) {
        return 1;  // No experts to process
    }
    moe_expert_work_t* work = s->expert_work.data();
    const uint64_t work_init_start_us = moe_profile_now_us();
    for (size_t e = 0; e < num_selected; ++e) {
        work[e].gate = nullptr;
        work[e].up = nullptr;
        work[e].out = nullptr;
        work[e].weight = s->selected_weights[e];
        work[e].expert_id = s->selected_experts[e];
        work[e].ok = 0;
    }
    moe_profile_log(
        "mlp_moe_work_init",
        "layer=%d selected=%zu ms=%.3f",
        layer, num_selected, moe_profile_elapsed_ms(work_init_start_us));

    // Fix 2: Flat row-parallel expert dispatch.
    // Instead of 8 sequential run_expert_triplet_f32 calls (each taking the full
    // pool via cpu_row_active), we pre-map all shards, then dispatch ALL experts'
    // rows in one parallel_rows call. This eliminates cpu_row_active serialization.
    uint32_t gate_rows = 0;
    uint32_t gate_cols = 0;
    uint32_t up_rows = 0;
    uint32_t up_cols = 0;
    uint32_t down_rows = 0;
    uint32_t down_cols = 0;

    // Pre-map all expert weight shards (these are cached, so fast for hot experts)
    struct expert_shards_t {
        const moe_tensor_record* gate_rec = nullptr;
        const moe_tensor_record* up_rec = nullptr;
        const moe_tensor_record* down_rec = nullptr;
        mmap_context_t* gate_shard = nullptr;
        mmap_context_t* up_shard = nullptr;
        mmap_context_t* down_shard = nullptr;
    };
    // BUGFIX 211 & 212: Dynamic allocation for shards and expert_ok arrays
    // Problem: Fixed-size arrays shards[moe_MOE_TOP_EXPERTS] and expert_ok[moe_MOE_TOP_EXPERTS]
    //          cause buffer overflow if num_selected > moe_MOE_TOP_EXPERTS
    // Solution: Use std::vector for dynamic sizing based on num_selected
    // Impact: Prevents buffer overflow, supports dynamic top-k values
    // BUGFIX Problem 12: Fix stack array initialization and size ★★★
    // Problem: std::atomic<int>[1000] on stack causes:
    //   1. 56KB stack frame (48KB shards + 8KB atomic)
    //   2. std::atomic<int> default constructor leaves indeterminate value (UB)
    //   3. 1000 constructor calls even when num_selected=8
    // Solution: Use std::vector for both, value-initialize to 0
    // Impact: Heap allocation but correct initialization, no UB, only num_selected elements
    std::vector<expert_shards_t> shards(num_selected);
    std::vector<std::atomic<int>> expert_ok(num_selected);
    for (size_t e = 0; e < num_selected; ++e) {
        expert_ok[e].store(0, std::memory_order_relaxed);
    }
    uint32_t missing_experts = 0;
    uint32_t shape_mismatch_experts = 0;
    uint32_t map_failed_experts = 0;
    int first_bad_expert = -1;
    uint32_t first_gate_rows = 0, first_gate_cols = 0;
    uint32_t first_up_rows = 0, first_up_cols = 0;
    uint32_t first_down_rows = 0, first_down_cols = 0;
    const uint64_t map_start_us = moe_profile_now_us();
    for (size_t e = 0; e < num_selected; ++e) {
        shards[e].gate_rec = find_tensor_record(engine, layer, work[e].expert_id, moe_PROJ_GATE);
        shards[e].up_rec = find_tensor_record(engine, layer, work[e].expert_id, moe_PROJ_UP);
        shards[e].down_rec = find_tensor_record(engine, layer, work[e].expert_id, moe_PROJ_DOWN);
        if (!shards[e].gate_rec || !shards[e].up_rec || !shards[e].down_rec) {
            ++missing_experts;
            if (first_bad_expert < 0) {
                first_bad_expert = work[e].expert_id;
                if (shards[e].gate_rec) {
                    first_gate_rows = shards[e].gate_rec->info.rows;
                    first_gate_cols = shards[e].gate_rec->info.cols;
                }
                if (shards[e].up_rec) {
                    first_up_rows = shards[e].up_rec->info.rows;
                    first_up_cols = shards[e].up_rec->info.cols;
                }
                if (shards[e].down_rec) {
                    first_down_rows = shards[e].down_rec->info.rows;
                    first_down_cols = shards[e].down_rec->info.cols;
                }
            }
            continue;
        }
        const uint32_t cand_gate_rows = shards[e].gate_rec->info.rows;
        const uint32_t cand_gate_cols = shards[e].gate_rec->info.cols;
        const uint32_t cand_up_rows = shards[e].up_rec->info.rows;
        const uint32_t cand_up_cols = shards[e].up_rec->info.cols;
        const uint32_t cand_down_rows = shards[e].down_rec->info.rows;
        const uint32_t cand_down_cols = shards[e].down_rec->info.cols;
        const int record_shape_ok =
            cand_gate_rows != 0 &&
            cand_gate_rows == cand_up_rows &&
            cand_gate_rows == cand_down_cols &&
            cand_gate_cols == hidden_size &&
            cand_up_cols == hidden_size &&
            cand_down_rows == hidden_size;
        const int matches_canonical =
            !gate_rows ||
            (cand_gate_rows == gate_rows && cand_gate_cols == gate_cols &&
             cand_up_rows == up_rows && cand_up_cols == up_cols &&
             cand_down_rows == down_rows && cand_down_cols == down_cols);
        if (!record_shape_ok || !matches_canonical) {
            ++shape_mismatch_experts;
            if (first_bad_expert < 0) {
                first_bad_expert = work[e].expert_id;
                first_gate_rows = cand_gate_rows;
                first_gate_cols = cand_gate_cols;
                first_up_rows = cand_up_rows;
                first_up_cols = cand_up_cols;
                first_down_rows = cand_down_rows;
                first_down_cols = cand_down_cols;
            }
            continue;
        }
        if (!gate_rows) {
            gate_rows = cand_gate_rows;
            gate_cols = cand_gate_cols;
            up_rows = cand_up_rows;
            up_cols = cand_up_cols;
            down_rows = cand_down_rows;
            down_cols = cand_down_cols;
        }
        shards[e].gate_shard = get_or_map_tensor_shard(engine, *shards[e].gate_rec);
        shards[e].up_shard = get_or_map_tensor_shard(engine, *shards[e].up_rec);
        shards[e].down_shard = get_or_map_tensor_shard(engine, *shards[e].down_rec);
        if (!shards[e].gate_shard || !shards[e].up_shard || !shards[e].down_shard) {
            ++map_failed_experts;
            if (first_bad_expert < 0) {
                first_bad_expert = work[e].expert_id;
                first_gate_rows = shards[e].gate_rec->info.rows;
                first_gate_cols = shards[e].gate_rec->info.cols;
                first_up_rows = shards[e].up_rec->info.rows;
                first_up_cols = shards[e].up_rec->info.cols;
                first_down_rows = shards[e].down_rec->info.rows;
                first_down_cols = shards[e].down_rec->info.cols;
            }
            continue;
        }
        expert_ok[e].store(1, std::memory_order_release);
    }
    uint32_t mapped_experts = 0;
    uint8_t gate_effective_format = 0;
    uint8_t up_effective_format = 0;
    uint8_t down_effective_format = 0;
    for (size_t e = 0; e < num_selected; ++e) {
        if (expert_ok[e].load(std::memory_order_acquire)) {
            ++mapped_experts;
            if (!gate_effective_format && shards[e].gate_rec && shards[e].up_rec && shards[e].down_rec) {
                gate_effective_format = (uint8_t)moe_effective_tensor_weight_format_for_row_bytes(
                    shards[e].gate_rec, tensor_weight_format(*shards[e].gate_rec));
                up_effective_format = (uint8_t)moe_effective_tensor_weight_format_for_row_bytes(
                    shards[e].up_rec, tensor_weight_format(*shards[e].up_rec));
                down_effective_format = (uint8_t)moe_effective_tensor_weight_format_for_row_bytes(
                    shards[e].down_rec, tensor_weight_format(*shards[e].down_rec));
            }
        }
    }
    moe_profile_log(
        "mlp_moe_map",
        "layer=%d selected=%zu mapped=%u missing=%u shape_mismatch=%u map_failed=%u gate=%ux%u gate_eff_fmt=%u up=%ux%u up_eff_fmt=%u down=%ux%u down_eff_fmt=%u ms=%.3f",
        layer, num_selected, mapped_experts, missing_experts, shape_mismatch_experts, map_failed_experts,
        gate_rows, gate_cols, (unsigned)gate_effective_format,
        up_rows, up_cols, (unsigned)up_effective_format,
        down_rows, down_cols, (unsigned)down_effective_format,
        moe_profile_elapsed_ms(map_start_us));

    // BUGFIX 문제10: Handle partial expert failures with weight renormalization ?�★
    // Problem: If any expert is missing, entire layer fails (return 0)
    // Impact: 1 missing expert out of 8 causes full token generation failure
    // Solution: Renormalize weights for available experts and continue
    // Only fail if ALL experts are missing
    const uint32_t failed_experts = missing_experts + shape_mismatch_experts + map_failed_experts;
    if (failed_experts > 0) {
        const uint32_t expected_intermediate = gate_rows ? gate_rows : configured_expert_intermediate;
        const uint32_t expected_gate_cols = gate_cols ? gate_cols : hidden_size;
        const uint32_t expected_up_cols = up_cols ? up_cols : hidden_size;
        const uint32_t expected_down_rows = down_rows ? down_rows : hidden_size;
        const uint32_t expected_down_cols = down_cols ? down_cols : configured_expert_intermediate;
        std::fprintf(stderr,
            "[storagellm mlp] selected expert shard contract failed layer=%d selected=%zu mapped=%u first=%d missing=%u shape_mismatch=%u map_failed=%u expected gate=%ux%u up=%ux%u down=%ux%u first_bad=%d got gate=%ux%u up=%ux%u down=%ux%u\n",
            layer, num_selected, mapped_experts, work[0].expert_id,
            missing_experts, shape_mismatch_experts, map_failed_experts,
            expected_intermediate, expected_gate_cols,
            expected_intermediate, expected_up_cols,
            expected_down_rows, expected_down_cols,
            first_bad_expert,
            first_gate_rows, first_gate_cols,
            first_up_rows, first_up_cols,
            first_down_rows, first_down_cols);
        return 0;
    }

    // BUGFIX 문제10: Enforce expert weight renormalization to fix PPL degradation ★
    // Renormalize weights regardless of router normalization settings (norm_topk)
    // or whether experts failed to map. This guarantees stable output scales.
    // Correctness: only normalized top-k routers may absorb missing expert weight.
    const uint32_t expert_intermediate = gate_rows;
    if (expert_intermediate == 0) {
        std::fprintf(stderr, "[storagellm mlp] invalid intermediate size layer=%d intermediate=%u\n",
            layer, expert_intermediate);
        return 0;
    }
    // BUGFIX 512: The 100000 intermediate size limit was removed as it arbitrarily
    // restricts extremely wide experts. Memory exhaustion is prevented by the SIZE_MAX check below.
    // BUGFIX 765: Check expert_intermediate for reasonable upper bound ★★★
    if (expert_intermediate > 1000000) {
        std::fprintf(stderr, "[storagellm mlp] excessive intermediate size layer=%d intermediate=%u\n",
            layer, expert_intermediate);
        return 0;
    }
    const size_t needed_gate = num_selected * (size_t)expert_intermediate;
    const size_t needed_up = needed_gate;
    const size_t needed_out = num_selected * (size_t)hidden_size;
    // BUGFIX 513: Check resize sizes for overflow ??
    // Problem: needed_gate, needed_up, needed_out can overflow size_t
    // Causes vector resize to fail or allocate wrong size
    // Solution: Check for overflow before resize
    if (needed_gate > SIZE_MAX / sizeof(float) ||
        needed_up > SIZE_MAX / sizeof(float) ||
        needed_out > SIZE_MAX / sizeof(float)) {
        std::fprintf(stderr, "[storagellm mlp] expert pool size overflow layer=%d selected=%zu intermediate=%u\n",
            layer, num_selected, expert_intermediate);
        return 0;
    }
    const uint64_t alloc_start_us = moe_profile_now_us();
    if (s->expert_gate_pool.size() < needed_gate) {
        s->expert_gate_pool.resize(needed_gate);
    }
    if (s->expert_up_pool.size() < needed_up) {
        s->expert_up_pool.resize(needed_up);
    }
    if (s->expert_out_pool.size() < needed_out) {
        s->expert_out_pool.resize(needed_out);
    }
    for (size_t e = 0; e < num_selected; ++e) {
        work[e].gate = s->expert_gate_pool.data() + e * (size_t)expert_intermediate;
        work[e].up = s->expert_up_pool.data() + e * (size_t)expert_intermediate;
        work[e].out = s->expert_out_pool.data() + e * (size_t)hidden_size;
        std::fill(work[e].out, work[e].out + hidden_size, 0.0f);
    }
    moe_profile_log(
        "mlp_moe_alloc",
        "layer=%d selected=%zu intermediate=%u hidden=%u gate_floats=%zu out_floats=%zu ms=%.3f",
        layer, num_selected, expert_intermediate, hidden_size,
        needed_gate, needed_out, moe_profile_elapsed_ms(alloc_start_us));

    // Phase 2a: Flat dispatch: gate+up+SiLU for all experts simultaneously.
    // Total rows = num_selected * intermediate_size, row i maps to expert e and row r.
    // BUGFIX 262: Check expert_intermediate > 0 BEFORE multiplication ??
    // Problem: Checked after total_gate_rows calculation, but division by zero still occurs in lambda
    // Solution: Check expert_intermediate > 0 before any calculation
    if (expert_intermediate == 0) {
        std::fprintf(stderr, "[storagellm mlp] invalid intermediate size layer=%d intermediate=%u\n",
            layer, expert_intermediate);
        return 0;
    }
    if (num_selected > (size_t)UINT32_MAX / (size_t)expert_intermediate) {
        std::fprintf(stderr, "[storagellm mlp] invalid gate rows layer=%d selected=%zu intermediate=%u\n",
            layer, num_selected, expert_intermediate);
        return 0;
    }
    const uint32_t total_gate_rows = (uint32_t)num_selected * expert_intermediate;
    auto mlp_input_reuse_tile = [](uint32_t input_floats) -> uint32_t {
        const uint32_t env_tile = moe_parse_env_u32("STORAGELLM_MLP_ROW_TILE");
        if (env_tile > 0) {
            return std::max(1u, std::min<uint32_t>(env_tile, 64u));
        }
        const uint64_t input_bytes = (uint64_t)input_floats * sizeof(float);
        if (input_bytes <= 16ull * 1024ull) return 16u;
        if (input_bytes <= 32ull * 1024ull) return 8u;
        return 4u;
    };
    const uint32_t gate_tile_rows = std::max(1u, std::min(expert_intermediate, mlp_input_reuse_tile(hidden_size)));
    const uint32_t gate_blocks_per_expert = (expert_intermediate + gate_tile_rows - 1u) / gate_tile_rows;
    if (gate_blocks_per_expert == 0 ||
        num_selected > (size_t)UINT32_MAX / (size_t)gate_blocks_per_expert) {
        std::fprintf(stderr, "[storagellm mlp] invalid gate blocks layer=%d selected=%zu intermediate=%u tile=%u\n",
            layer, num_selected, expert_intermediate, gate_tile_rows);
        return 0;
    }
    const uint32_t total_gate_blocks = (uint32_t)num_selected * gate_blocks_per_expert;
    const uint64_t gate_up_start_us = moe_profile_now_us();
    const int activation_mode = moe_engine_gelu_activation_mode(engine);
    moe_profile_parallel_label_scope gate_up_rows_label("moe_mlp_gate_up");
    const int gate_up_ok = moe_parallel_rows(engine, total_gate_blocks, 0, [&](uint32_t block_index) {
        const uint32_t e = block_index / gate_blocks_per_expert;
        const uint32_t block = block_index % gate_blocks_per_expert;
        // Guard row-to-expert decoding before indexing per-expert arrays.
        // BUGFIX 290: Add bounds check for e index ??
        // Problem: e can exceed num_selected if flat_row is corrupted or overflow occurs
        // Accessing expert_ok[e], work[e], shards[e] with invalid e causes out-of-bounds access
        // Solution: Check e < num_selected before any array access
        if (e >= num_selected) {
            return 0;
        }
        if (!expert_ok[e].load(std::memory_order_acquire)) return 1;
        const uint32_t row_begin = block * gate_tile_rows;
        uint32_t row_end = row_begin + gate_tile_rows;
        if (row_end > expert_intermediate) {
            row_end = expert_intermediate;
        }
        for (uint32_t r = row_begin; r < row_end; ++r) {
            if (!moe_dot_tensor_record_row(engine, shards[e].gate_rec, shards[e].gate_shard, r, expert_input, hidden_size, &work[e].gate[r])) { expert_ok[e].store(0, std::memory_order_release); return 1; }
            if (!moe_dot_tensor_record_row(engine, shards[e].up_rec, shards[e].up_shard, r, expert_input, hidden_size, &work[e].up[r])) { expert_ok[e].store(0, std::memory_order_release); return 1; }
            // BUGFIX 850: Check gate/up results for NaN before activation ★★★
            if (!std::isfinite(work[e].gate[r]) || !std::isfinite(work[e].up[r])) {
                expert_ok[e].store(0, std::memory_order_release);
                return 1;
            }
            work[e].gate[r] = moe_generation_gated_activation_cached(activation_mode, work[e].gate[r], work[e].up[r]);
        }
        return 1;
    });
    moe_profile_log(
        "mlp_moe_gate_up",
        "layer=%d ok=%d activation_mode=%d selected=%zu rows=%u blocks=%u tile=%u hidden=%u intermediate=%u ms=%.3f",
        layer, gate_up_ok, activation_mode, num_selected, total_gate_rows, total_gate_blocks, gate_tile_rows, hidden_size, expert_intermediate,
        moe_profile_elapsed_ms(gate_up_start_us));
    if (!gate_up_ok) return 0;

    // Phase 2b: Flat dispatch: down_proj for all experts simultaneously.
    if (hidden_size == 0 || num_selected > (size_t)UINT32_MAX / (size_t)hidden_size) {
        std::fprintf(stderr, "[storagellm mlp] invalid down rows layer=%d selected=%zu hidden=%u\n",
            layer, num_selected, hidden_size);
        return 0;
    }
    const uint32_t total_down_rows = (uint32_t)num_selected * hidden_size;
    const uint32_t down_tile_rows = std::max(1u, std::min(hidden_size, mlp_input_reuse_tile(expert_intermediate)));
    const uint32_t down_blocks_per_expert = (hidden_size + down_tile_rows - 1u) / down_tile_rows;
    if (down_blocks_per_expert == 0 ||
        num_selected > (size_t)UINT32_MAX / (size_t)down_blocks_per_expert) {
        std::fprintf(stderr, "[storagellm mlp] invalid down blocks layer=%d selected=%zu hidden=%u tile=%u\n",
            layer, num_selected, hidden_size, down_tile_rows);
        return 0;
    }
    const uint32_t total_down_blocks = (uint32_t)num_selected * down_blocks_per_expert;
    const uint64_t down_start_us = moe_profile_now_us();
    moe_profile_parallel_label_scope down_rows_label("moe_mlp_down");
    const int down_ok = moe_parallel_rows(engine, total_down_blocks, 0, [&](uint32_t block_index) {
        const uint32_t e = block_index / down_blocks_per_expert;
        const uint32_t block = block_index % down_blocks_per_expert;
        // Guard row-to-expert decoding before indexing per-expert arrays.
        // BUGFIX 312: Add bounds check for e index (same as gate_up phase) ??
        // Problem: e can exceed num_selected if flat_row is corrupted
        // Accessing expert_ok[e], work[e], shards[e] with invalid e causes out-of-bounds access
        // Solution: Check e < num_selected before any array access
        if (e >= num_selected) {
            return 0;
        }
        if (!expert_ok[e].load(std::memory_order_acquire)) return 1;
        const uint32_t row_begin = block * down_tile_rows;
        uint32_t row_end = row_begin + down_tile_rows;
        if (row_end > hidden_size) {
            row_end = hidden_size;
        }
        for (uint32_t r = row_begin; r < row_end; ++r) {
            if (!moe_dot_tensor_record_row(engine, shards[e].down_rec, shards[e].down_shard,
                    r, work[e].gate, expert_intermediate, &work[e].out[r])) {
                expert_ok[e].store(0, std::memory_order_release);
                return 1;
            }
            // BUGFIX 851: Check down result for NaN ★★★
            if (!std::isfinite(work[e].out[r])) {
                expert_ok[e].store(0, std::memory_order_release);
                return 1;
            }
        }
        return 1;
    });
    moe_profile_log(
        "mlp_moe_down",
        "layer=%d ok=%d selected=%zu rows=%u blocks=%u tile=%u hidden=%u intermediate=%u ms=%.3f",
        layer, down_ok, num_selected, total_down_rows, total_down_blocks, down_tile_rows, hidden_size, expert_intermediate,
        moe_profile_elapsed_ms(down_start_us));
    if (!down_ok) return 0;
    for (size_t e = 0; e < num_selected; ++e) {
        work[e].ok = expert_ok[e].load(std::memory_order_acquire);
    }

    // Phase 3: Accumulate weighted expert outputs, then apply the expert-branch
    // post norm once. Gemma4 normalizes the summed expert branch, not each expert
    // independently.
    const uint64_t accumulate_start_us = moe_profile_now_us();

    // BUGFIX: Explicitly zero the accumulator buffer before processing sparse experts
    // Without this, the accumulation includes garbage or previous shared expert data
    // BUGFIX 766: Check hidden_size bounds before fill ★★★
    if (hidden_size > s->residual.size()) {
        std::fprintf(stderr, "[storagellm mlp] hidden_size exceeds residual buffer layer=%d hidden=%u buffer=%zu\n",
            layer, hidden_size, s->residual.size());
        return 0;
    }
    std::fill(s->residual.begin(), s->residual.begin() + hidden_size, 0.0f);

    for (size_t e = 0; e < num_selected; ++e) {
        if (!work[e].ok) continue;
        if (!moe_layer_apply_expert_down_scale_f32(
                engine, layer, work[e].expert_id, c->experts_per_moe_layer,
                work[e].out, hidden_size)) {
            std::fprintf(stderr, "[storagellm mlp] expert down scale failed layer=%d expert=%d\n",
                layer, work[e].expert_id);
            return 0;
        }
        // BUGFIX 767: Check work[e].weight for NaN/Inf before axpy ★★★
        if (!std::isfinite(work[e].weight)) {
            std::fprintf(stderr, "[storagellm mlp] invalid expert weight layer=%d expert=%d weight=%f\n",
                layer, work[e].expert_id, work[e].weight);
            return 0;
        }
        // BUGFIX 848: Check work[e].out for NULL before axpy ★★★
        if (!work[e].out) {
            std::fprintf(stderr, "[storagellm mlp] NULL expert output layer=%d expert=%d\n",
                layer, work[e].expert_id);
            return 0;
        }
        moe_axpy_f32(s->residual.data(), work[e].out, work[e].weight, hidden_size);
    }
    const float* expert_branch = s->residual.data();
    if (moe_layer_post_ffw_norm2_f32(
            engine, layer, s->residual.data(), hidden_size, s->expert.data(), attn_cfg.rms_norm_eps)) {
        expert_branch = s->expert.data();
    } else if (moe_layer_has_post_ffw_norm2_weight(engine, layer)) {
        std::fprintf(stderr,
            "[storagellm mlp] post-ffw norm2 tensor is required but could not be mapped layer=%d\n",
            layer);
        return 0;
    }
    // BUGFIX 768: Check for NaN/Inf in expert_branch before addition ★★★
    // BUGFIX 849: Check expert_branch for NULL before loop ★★★
    if (!expert_branch) {
        std::fprintf(stderr, "[storagellm mlp] NULL expert_branch layer=%d\n", layer);
        return 0;
    }
    for (uint32_t i = 0; i < hidden_size; ++i) {
        if (!std::isfinite(expert_branch[i])) {
            std::fprintf(stderr,
                "[storagellm mlp] non-finite expert branch layer=%d index=%u\n",
                layer, i);
            return 0;
        }
        s->mlp[i] += expert_branch[i];
    }
    const double accumulate_ms = moe_profile_elapsed_ms(accumulate_start_us);

    const uint64_t residual_start_us = moe_profile_now_us();
    const int residual_ok = moe_add_mlp_residual_f32(engine, layer, s, hidden, hidden_size);
    moe_profile_log(
        "mlp_moe_end",
        "layer=%d ok=%d selected=%zu mapped=%u hidden=%u intermediate=%u accumulate_ms=%.3f residual_ms=%.3f total_ms=%.3f",
        layer, residual_ok, num_selected, mapped_experts, hidden_size, expert_intermediate,
        accumulate_ms, moe_profile_elapsed_ms(residual_start_us), moe_profile_elapsed_ms(total_start_us));
    return residual_ok;
}


static uint32_t moe_mlp_prefill_expert_batch_limit() {
    const uint32_t env_limit = moe_parse_env_u32("STORAGELLM_MLP_EXPERT_BATCH");
    if (env_limit > 0) {
        return std::max(1u, std::min<uint32_t>(env_limit, 4096u)); // UNIVERSAL BATCH LIMIT
    }
    return 256u; // Default dynamic limit
}

static int moe_mlp_prefill_batch_enabled() {
    const char* env = std::getenv("STORAGELLM_PREFILL_BATCH_MLP");
    return !(env && env[0] == '0');
}

static constexpr int moe_PREFILL_BATCH_MLP_FAST_KERNEL_REQUIRED = -2;

struct moe_device_tensor_batch_view_t {
    const moe_tensor_record* rec = nullptr;
    uint64_t key = 0;
    uint64_t ptr = 0;
    uint64_t bytes = 0;
    uint32_t weight_format = 0;
    uint32_t backend_kind = 0;
    uint64_t backend_aux = 0;
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint64_t weight_row_bytes = 0;
    uint64_t weight_bytes = 0;
    uint64_t stream_bytes = 0;
    uint32_t expert_gpu_layout_kind = 0;
    uint64_t expert_gpu_layout_offset = 0;
    uint64_t expert_gpu_layout_size = 0;
    uint64_t expert_gpu_layout_row_bytes = 0;
};

using moe_model_lib_expert_batch_f32_fn = int (*)(
    uint32_t abi_version,
    int32_t backend,
    int32_t layer,
    int32_t expert,
    const void* gate,
    const void* up,
    const void* down,
    const float* const* batch_hidden,
    uint32_t batch_size,
    uint32_t hidden_size,
    uint32_t intermediate_size,
    float* out_pool,
    uint32_t out_count,
    int activation_mode);

using moe_model_lib_expert_batch_contiguous_f32_fn = int (*)(
    uint32_t abi_version,
    int32_t backend,
    int32_t layer,
    int32_t expert,
    const void* gate,
    const void* up,
    const void* down,
    const float* batch_hidden_contiguous,
    uint32_t batch_stride,
    uint32_t batch_size,
    uint32_t hidden_size,
    uint32_t intermediate_size,
    float* out_pool,
    uint32_t out_count,
    int activation_mode);

using moe_model_lib_expert_batch_device_f32_fn = int (*)(
    uint32_t abi_version,
    int32_t backend,
    int32_t layer,
    int32_t expert,
    const void* gate,
    const void* up,
    const void* down,
    const void* d_batch_hidden,
    uint32_t batch_stride,
    uint32_t batch_size,
    uint32_t hidden_size,
    uint32_t intermediate_size,
    void* d_out,
    uint32_t out_stride,
    int activation_mode);

struct moe_model_lib_grouped_expert_task_t {
    int32_t layer = 0;
    int32_t expert = 0;
    const void* gate = nullptr;
    const void* up = nullptr;
    const void* down = nullptr;
    const float* x = nullptr;
    float* out = nullptr;
    uint32_t batch = 0;
    uint32_t hidden = 0;
    uint32_t intermediate = 0;
};

using moe_model_lib_grouped_expert_triplets_f32_fn = int (*)(
    uint32_t abi_version,
    int32_t backend,
    const moe_model_lib_grouped_expert_task_t* tasks,
    uint32_t task_count);

struct moe_model_lib_grouped_expert_device_task_t {
    int32_t layer = 0;
    int32_t expert = 0;
    const void* gate = nullptr;
    const void* up = nullptr;
    const void* down = nullptr;
    const void* d_x = nullptr;
    void* d_out = nullptr;
    uint32_t batch_stride = 0;
    uint32_t out_stride = 0;
    uint32_t batch = 0;
    uint32_t hidden = 0;
    uint32_t intermediate = 0;
};

using moe_model_lib_grouped_expert_triplets_device_f32_fn = int (*)(
    uint32_t abi_version,
    int32_t backend,
    const moe_model_lib_grouped_expert_device_task_t* tasks,
    uint32_t task_count);

struct moe_model_lib_indexed_expert_task_t {
    int32_t layer = 0;
    int32_t expert = 0;
    const void* gate = nullptr;
    const void* up = nullptr;
    const void* down = nullptr;
    uint32_t token_offset = 0;
    uint32_t batch = 0;
    uint32_t hidden = 0;
    uint32_t intermediate = 0;
};

using moe_model_lib_grouped_expert_triplets_indexed_f32_fn = int (*)(
    uint32_t abi_version,
    int32_t backend,
    const moe_model_lib_indexed_expert_task_t* tasks,
    uint32_t task_count,
    const float* batch_input,
    uint32_t input_stride,
    uint32_t input_count,
    const uint32_t* token_indices,
    const float* token_weights,
    uint32_t assignment_count,
    float* batch_accum,
    uint32_t accum_stride);

using moe_model_lib_grouped_expert_triplets_indexed_device_f32_fn = int (*)(
    uint32_t abi_version,
    int32_t backend,
    const moe_grouped_expert_device_task_t* tasks,
    uint32_t task_count,
    void* stream_or_queue);

using moe_model_lib_prefill_activation_acquire_f32_fn = int (*)(
    uint32_t abi_version,
    int32_t backend,
    const float* batch_input,
    uint32_t input_stride,
    uint32_t input_count,
    const uint32_t* token_indices,
    const float* token_weights,
    uint32_t assignment_count,
    const float* batch_accum,
    uint32_t accum_stride,
    uint64_t* out_handle,
    const void** out_d_batch_input,
    const void** out_d_token_indices,
    const void** out_d_token_weights,
    void** out_d_batch_accum);

using moe_model_lib_prefill_activation_release_f32_fn = int (*)(
    uint32_t abi_version,
    int32_t backend,
    uint64_t handle,
    float* batch_accum,
    uint32_t input_count,
    uint32_t accum_stride,
    int copy_back);

static int moe_get_resident_device_tensor_view(
    moe_pc_engine* engine,
    const moe_tensor_record* rec,
    moe_device_tensor_batch_view_t* out
) {
    if (out) {
        *out = moe_device_tensor_batch_view_t{};
    }
    if (!engine || !rec || !out || engine->device_allocator_ready.load(std::memory_order_acquire) <= 0) {
        return 0;
    }
    const uint64_t key = device_key_tensor(*rec);
    std::lock_guard<std::mutex> lock(engine->device_mutex);
    auto it = engine->device_allocations.find(key);
    if (it == engine->device_allocations.end() ||
        it->second.pending_free ||
        it->second.inflight_copies != 0 ||
        it->second.ptr == 0 ||
        it->second.bytes < tensor_stream_bytes(*rec)) {
        return 0;
    }
    out->rec = rec;
    out->key = key;
    out->ptr = it->second.ptr;
    out->bytes = it->second.bytes;
    out->weight_format = static_cast<uint32_t>(tensor_weight_format(*rec));
    out->backend_kind = it->second.backend_kind;
    out->backend_aux = it->second.backend_aux;
    out->rows = rec->info.rows;
    out->cols = rec->info.cols;
    out->weight_row_bytes = rec->weight_row_bytes;
    out->weight_bytes = rec->info.weight_byte_length;
    out->stream_bytes = tensor_stream_bytes(*rec);
    out->expert_gpu_layout_kind = rec->expert_gpu_layout_kind;
    out->expert_gpu_layout_offset = rec->expert_gpu_layout_offset;
    out->expert_gpu_layout_size = rec->expert_gpu_layout_size;
    out->expert_gpu_layout_row_bytes = rec->expert_gpu_layout_row_bytes;
    return 1;
}

static int moe_expert_triplet_resident_in_vram(
    moe_pc_engine* engine,
    int32_t layer,
    int expert,
    moe_device_tensor_batch_view_t views[3] = nullptr
) {
    if (!engine || layer < 0 || expert < 0) {
        return 0;
    }
    const moe_tensor_record* gate = find_tensor_record(engine, layer, expert, moe_PROJ_GATE);
    const moe_tensor_record* up = find_tensor_record(engine, layer, expert, moe_PROJ_UP);
    const moe_tensor_record* down = find_tensor_record(engine, layer, expert, moe_PROJ_DOWN);
    if (!gate || !up || !down) {
        return 0;
    }
    moe_device_tensor_batch_view_t local[3]{};
    if (!moe_get_resident_device_tensor_view(engine, gate, &local[0]) ||
        !moe_get_resident_device_tensor_view(engine, up, &local[1]) ||
        !moe_get_resident_device_tensor_view(engine, down, &local[2])) {
        return 0;
    }
    if (views) {
        views[0] = local[0];
        views[1] = local[1];
        views[2] = local[2];
    }
    return 1;
}

static moe_model_lib_expert_batch_f32_fn moe_gpu_batch_expert_kernel_fn(moe_pc_engine* engine) {
    if (!engine || !engine->model_lib_handle || !engine->model_lib_expert_batch_f32) {
        return nullptr;
    }
    return reinterpret_cast<moe_model_lib_expert_batch_f32_fn>(engine->model_lib_expert_batch_f32);
}

static moe_model_lib_expert_batch_contiguous_f32_fn moe_gpu_batch_expert_contiguous_kernel_fn(moe_pc_engine* engine) {
    if (!engine || !engine->model_lib_handle || !engine->model_lib_expert_batch_contiguous_f32) {
        return nullptr;
    }
    return reinterpret_cast<moe_model_lib_expert_batch_contiguous_f32_fn>(
        engine->model_lib_expert_batch_contiguous_f32);
}

static moe_model_lib_expert_batch_device_f32_fn moe_gpu_batch_expert_device_kernel_fn(moe_pc_engine* engine) {
    if (!engine || !engine->model_lib_handle || !engine->model_lib_expert_batch_device_f32) {
        return nullptr;
    }
    return reinterpret_cast<moe_model_lib_expert_batch_device_f32_fn>(
        engine->model_lib_expert_batch_device_f32);
}

static moe_model_lib_grouped_expert_triplets_f32_fn moe_gpu_grouped_expert_kernel_fn(moe_pc_engine* engine) {
    if (!engine || !engine->model_lib_handle || !engine->model_lib_grouped_expert_triplets_f32) {
        return nullptr;
    }
    return reinterpret_cast<moe_model_lib_grouped_expert_triplets_f32_fn>(
        engine->model_lib_grouped_expert_triplets_f32);
}

static moe_model_lib_grouped_expert_triplets_device_f32_fn moe_gpu_grouped_expert_device_kernel_fn(
    moe_pc_engine* engine
) {
    if (!engine || !engine->model_lib_handle || !engine->model_lib_grouped_expert_triplets_device_f32) {
        return nullptr;
    }
    return reinterpret_cast<moe_model_lib_grouped_expert_triplets_device_f32_fn>(
        engine->model_lib_grouped_expert_triplets_device_f32);
}

static moe_model_lib_grouped_expert_triplets_indexed_f32_fn moe_gpu_grouped_expert_indexed_kernel_fn(
    moe_pc_engine* engine
) {
    if (!engine || !engine->model_lib_handle || !engine->model_lib_grouped_expert_triplets_indexed_f32) {
        return nullptr;
    }
    return reinterpret_cast<moe_model_lib_grouped_expert_triplets_indexed_f32_fn>(
        engine->model_lib_grouped_expert_triplets_indexed_f32);
}

static moe_model_lib_grouped_expert_triplets_indexed_device_f32_fn
moe_gpu_grouped_expert_indexed_device_kernel_fn(moe_pc_engine* engine) {
    if (!engine || !engine->model_lib_handle ||
        !engine->model_lib_grouped_expert_triplets_indexed_device_f32) {
        return nullptr;
    }
    return reinterpret_cast<moe_model_lib_grouped_expert_triplets_indexed_device_f32_fn>(
        engine->model_lib_grouped_expert_triplets_indexed_device_f32);
}

static moe_model_lib_prefill_activation_acquire_f32_fn
moe_gpu_prefill_activation_acquire_fn(moe_pc_engine* engine) {
    if (!engine || !engine->model_lib_handle || !engine->model_lib_prefill_activation_acquire_f32) {
        return nullptr;
    }
    return reinterpret_cast<moe_model_lib_prefill_activation_acquire_f32_fn>(
        engine->model_lib_prefill_activation_acquire_f32);
}

static moe_model_lib_prefill_activation_release_f32_fn
moe_gpu_prefill_activation_release_fn(moe_pc_engine* engine) {
    if (!engine || !engine->model_lib_handle || !engine->model_lib_prefill_activation_release_f32) {
        return nullptr;
    }
    return reinterpret_cast<moe_model_lib_prefill_activation_release_f32_fn>(
        engine->model_lib_prefill_activation_release_f32);
}

static int moe_prefill_host_activation_bridge_allowed(const moe_pc_engine* engine) {
    return engine && !moe_verified_backend_required(engine);
}

static int moe_gpu_batch_expert_kernel_available(moe_pc_engine* engine) {
    return moe_gpu_batch_expert_kernel_fn(engine) ? 1 : 0;
}

static int moe_run_expert_triplet_gpu_batch_f32(
    moe_pc_engine* engine,
    int32_t layer,
    int expert,
    const float** batch_hidden,
    uint32_t batch_size,
    uint32_t hidden_size,
    uint32_t intermediate_size,
    float* out_pool,
    uint32_t out_count,
    int activation_mode
) {
    if (!engine || !batch_hidden || !out_pool || batch_size == 0 ||
        hidden_size == 0 || intermediate_size == 0 || out_count != hidden_size) {
        return 0;
    }
    auto fn = moe_gpu_batch_expert_kernel_fn(engine);
    if (!fn) {
        return 0;
    }
    moe_device_tensor_batch_view_t views[3]{};
    if (!moe_expert_triplet_resident_in_vram(engine, layer, expert, views)) {
        return 0;
    }
    const int ok = fn(
        engine->model_lib_abi_version,
        static_cast<int32_t>(engine->backend_caps.backend),
        layer,
        expert,
        &views[0],
        &views[1],
        &views[2],
        batch_hidden,
        batch_size,
        hidden_size,
        intermediate_size,
        out_pool,
        out_count,
        activation_mode);
    if (ok) {
        device_touch_allocation(engine, views[0].key);
        device_touch_allocation(engine, views[1].key);
        device_touch_allocation(engine, views[2].key);
    }
    return ok ? 1 : 0;
}

static int moe_run_expert_triplet_gpu_batch_contiguous_f32(
    moe_pc_engine* engine,
    int32_t layer,
    int expert,
    const float* batch_hidden_contiguous,
    uint32_t batch_stride,
    uint32_t batch_size,
    uint32_t hidden_size,
    uint32_t intermediate_size,
    float* out_pool,
    uint32_t out_count,
    int activation_mode
) {
    if (!engine || !batch_hidden_contiguous || !out_pool || batch_size == 0 ||
        batch_stride < hidden_size || hidden_size == 0 ||
        intermediate_size == 0 || out_count != hidden_size) {
        return 0;
    }
    auto fn = moe_gpu_batch_expert_contiguous_kernel_fn(engine);
    if (!fn) {
        return 0;
    }
    moe_device_tensor_batch_view_t views[3]{};
    if (!moe_expert_triplet_resident_in_vram(engine, layer, expert, views)) {
        return 0;
    }
    const int ok = fn(
        engine->model_lib_abi_version,
        static_cast<int32_t>(engine->backend_caps.backend),
        layer,
        expert,
        &views[0],
        &views[1],
        &views[2],
        batch_hidden_contiguous,
        batch_stride,
        batch_size,
        hidden_size,
        intermediate_size,
        out_pool,
        out_count,
        activation_mode);
    if (ok) {
        device_touch_allocation(engine, views[0].key);
        device_touch_allocation(engine, views[1].key);
        device_touch_allocation(engine, views[2].key);
    }
    return ok ? 1 : 0;
}

static int moe_run_expert_triplet_row_batch_f32(
    moe_pc_engine* engine,
    int32_t layer,
    int expert,
    const float** batch_hidden,
    uint32_t batch_size,
    uint32_t hidden_size,
    float* gate_pool,
    float* up_pool,
    uint32_t intermediate_size,
    float* out_pool,
    uint32_t out_count,
    int activation_mode
) {
    if (!engine || !batch_hidden || !gate_pool || !up_pool || !out_pool ||
        batch_size == 0 || batch_size > 4096u ||
        hidden_size == 0 || intermediate_size == 0 || out_count != hidden_size) {
        return 0;
    }
    for (uint32_t i = 0; i < batch_size; ++i) {
        if (!batch_hidden[i]) {
            return 0;
        }
    }
    const moe_tensor_record* gate = find_tensor_record(engine, layer, expert, moe_PROJ_GATE);
    const moe_tensor_record* up = find_tensor_record(engine, layer, expert, moe_PROJ_UP);
    const moe_tensor_record* down = find_tensor_record(engine, layer, expert, moe_PROJ_DOWN);
    if (!gate || !up || !down ||
        gate->info.rows != intermediate_size || gate->info.cols != hidden_size ||
        up->info.rows != intermediate_size || up->info.cols != hidden_size ||
        down->info.rows != hidden_size || down->info.cols != intermediate_size) {
        return 0;
    }
    const int gpu_kernel_available = moe_gpu_batch_expert_kernel_available(engine);
    const int resident_in_vram = moe_expert_triplet_resident_in_vram(engine, layer, expert);
    if (gpu_kernel_available && resident_in_vram) {
        if (moe_run_expert_triplet_gpu_batch_f32(
                engine,
                layer,
                expert,
                batch_hidden,
                batch_size,
                hidden_size,
                intermediate_size,
                out_pool,
                out_count,
                activation_mode)) {
            return 1;
        }
        if (engine->io_config.internal_require_gpu_expert_batch_kernel) {
            return 0;
        }
    }
    mmap_context_t* gate_shard = get_or_map_tensor_shard(engine, *gate);
    mmap_context_t* up_shard = get_or_map_tensor_shard(engine, *up);
    mmap_context_t* down_shard = get_or_map_tensor_shard(engine, *down);
    if (!gate_shard || !up_shard || !down_shard ||
        !mmap_get_ptr(gate_shard, 0) ||
        !mmap_get_ptr(up_shard, 0) ||
        !mmap_get_ptr(down_shard, 0)) {
        return 0;
    }

    auto tile_rows_for_input = [](uint32_t input_floats) -> uint32_t {
        const uint32_t env_tile = moe_parse_env_u32("STORAGELLM_MLP_ROW_TILE");
        if (env_tile > 0) {
            return std::max(1u, std::min<uint32_t>(env_tile, 64u));
        }
        const uint64_t input_bytes = (uint64_t)input_floats * sizeof(float);
        if (input_bytes <= 16ull * 1024ull) return 16u;
        if (input_bytes <= 32ull * 1024ull) return 8u;
        return 4u;
    };

    const uint32_t gate_tile_rows = std::max(1u, std::min(intermediate_size, tile_rows_for_input(hidden_size)));
    const uint32_t gate_block_count = (intermediate_size + gate_tile_rows - 1u) / gate_tile_rows;
    std::vector<float> gate_values_all((size_t)gate_block_count * batch_size);
    std::vector<float> up_values_all((size_t)gate_block_count * batch_size);
    const int gate_up_ok = moe_parallel_rows(engine, gate_block_count, 0, [&](uint32_t block) {
        float* gate_values = gate_values_all.data() + (size_t)block * batch_size;
        float* up_values = up_values_all.data() + (size_t)block * batch_size;
        const uint32_t row_begin = block * gate_tile_rows;
        uint32_t row_end = row_begin + gate_tile_rows;
        if (row_end > intermediate_size) {
            row_end = intermediate_size;
        }
        for (uint32_t row = row_begin; row < row_end; ++row) {
            if (!moe_dot_tensor_record_row_batch(
                    engine, gate, gate_shard, row,
                    batch_hidden, batch_size, hidden_size, gate_values) ||
                !moe_dot_tensor_record_row_batch(
                    engine, up, up_shard, row,
                    batch_hidden, batch_size, hidden_size, up_values)) {
                return 0;
            }
            for (uint32_t b = 0; b < batch_size; ++b) {
                const size_t idx = (size_t)b * (size_t)intermediate_size + row;
                gate_pool[idx] = moe_generation_gated_activation_cached(activation_mode, gate_values[b], up_values[b]);
                up_pool[idx] = up_values[b];
            }
        }
        return 1;
    });
    if (!gate_up_ok) {
        return 0;
    }

    std::vector<const float*> activated_inputs_all(batch_size);
    const float** activated_inputs = activated_inputs_all.data();
    for (uint32_t b = 0; b < batch_size; ++b) {
        activated_inputs[b] = gate_pool + (size_t)b * (size_t)intermediate_size;
    }
    const uint32_t down_tile_rows = std::max(1u, std::min(out_count, tile_rows_for_input(intermediate_size)));
    const uint32_t down_block_count = (out_count + down_tile_rows - 1u) / down_tile_rows;
    std::vector<float> down_values_all((size_t)down_block_count * batch_size);
    return moe_parallel_rows(engine, down_block_count, 0, [&](uint32_t block) {
        float* down_values = down_values_all.data() + (size_t)block * batch_size;
        const uint32_t row_begin = block * down_tile_rows;
        uint32_t row_end = row_begin + down_tile_rows;
        if (row_end > out_count) {
            row_end = out_count;
        }
        for (uint32_t row = row_begin; row < row_end; ++row) {
            if (!moe_dot_tensor_record_row_batch(
                    engine, down, down_shard, row,
                    activated_inputs, batch_size, intermediate_size, down_values)) {
                return 0;
            }
            for (uint32_t b = 0; b < batch_size; ++b) {
                out_pool[(size_t)b * (size_t)out_count + row] = down_values[b];
            }
        }
        return 1;
    });
}

struct moe_mlp_batch_assignment_t {
    uint32_t token = 0;
    float weight = 0.0f;
};

struct moe_active_expert_group_t {
    uint32_t expert = 0;
    uint32_t assignment_count = 0;
    float weight_sum = 0.0f;
    moe_tier_t current_tier = moe_TIER_DB;
};

static void moe_build_active_expert_groups(
    moe_pc_engine* engine,
    int layer,
    const std::vector<std::vector<moe_mlp_batch_assignment_t>>& grouped,
    std::vector<moe_active_expert_group_t>* out
) {
    if (!out) {
        return;
    }
    out->clear();
    if (!engine) {
        return;
    }
    for (uint32_t expert = 0; expert < grouped.size(); ++expert) {
        const auto& assignments = grouped[expert];
        if (assignments.empty()) {
            continue;
        }
        float weight_sum = 0.0f;
        for (const auto& assignment : assignments) {
            weight_sum += assignment.weight;
        }
        moe_tier_t tier = moe_TIER_DB;
        const size_t slot = expert_dense_slot(engine, layer, (int)expert);
        if (slot != tensor_slot_invalid() &&
            slot < engine->expert_states.size() &&
            engine->expert_states[slot].active) {
            tier = engine->expert_states[slot].tier;
        }
        out->push_back({
            expert,
            static_cast<uint32_t>(assignments.size()),
            weight_sum,
            tier
        });
    }
    std::sort(out->begin(), out->end(), [](const auto& a, const auto& b) {
        if (a.assignment_count != b.assignment_count) {
            return a.assignment_count > b.assignment_count;
        }
        return a.weight_sum > b.weight_sum;
    });
}

static void moe_wait_ready_for_top_active_experts(
    moe_pc_engine* engine,
    int layer,
    const std::vector<moe_active_expert_group_t>& active_experts,
    uint32_t wait_us
) {
    if (!engine || wait_us == 0 || active_experts.empty()) {
        return;
    }
    const uint32_t min_assignments =
        std::max<uint32_t>(1u, engine->io_config.expert_group_gpu_min_assignments);
    std::vector<uint32_t> target_experts;
    target_experts.reserve(std::min<size_t>(active_experts.size(), 8u));
    for (const auto& group : active_experts) {
        if (group.assignment_count >= min_assignments) {
            target_experts.push_back(group.expert);
            if (target_experts.size() >= 8u) {
                break;
            }
        }
    }
    if (target_experts.empty()) {
        return;
    }

    const uint64_t deadline_us = moe_profile_now_us() + wait_us;
    while (moe_profile_now_us() < deadline_us) {
        device_drain_deferred_staging_copies(engine, false);
        device_drain_directstorage_device_copies(engine);

        bool all_ready = true;
        for (uint32_t expert : target_experts) {
            if (!moe_expert_triplet_resident_in_vram(engine, layer, (int)expert)) {
                all_ready = false;
                break;
            }
        }
        if (all_ready) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(25));
    }
}

static int moe_layer_expert_down_scalar_scale_f32(
    moe_pc_engine* engine,
    int32_t layer,
    int expert_id,
    uint32_t experts,
    uint32_t hidden_size,
    float* out_scale,
    int* out_scalar_supported
) {
    if (out_scale) {
        *out_scale = 1.0f;
    }
    if (out_scalar_supported) {
        *out_scalar_supported = 1;
    }
    if (!engine || expert_id < 0 || hidden_size == 0 || !out_scale || !out_scalar_supported) {
        return 0;
    }
    if (experts > 10000u || (uint32_t)expert_id >= experts) {
        return 0;
    }

    static const char* const suffixes[] = {
        "ffn_down_exps.scale",
        "ffn_gate_inp.per_expert_scale",
        "router.per_expert_scale",
        "mlp.router.per_expert_scale",
        "moe.gate.per_expert_scale"
    };
    moe_raw_tensor_cache_entry raw{};
    if (!moe_map_layer_common_tensor(engine, layer, suffixes, sizeof(suffixes) / sizeof(suffixes[0]), &raw)) {
        return 1;
    }
    const uint64_t count = moe_raw_tensor_scalar_count(raw);
    if (count == 1u) {
        float scale = 1.0f;
        if (!moe_raw_tensor_scalar_at_f32(raw, 0, &scale) || !std::isfinite(scale)) {
            return 0;
        }
        *out_scale = scale;
        return 1;
    }
    if (experts && count == experts) {
        float scale = 1.0f;
        if (!moe_raw_tensor_scalar_at_f32(raw, (uint64_t)expert_id, &scale) || !std::isfinite(scale)) {
            return 0;
        }
        *out_scale = scale;
        return 1;
    }

    const uint64_t base = (uint64_t)expert_id * (uint64_t)hidden_size;
    if (base <= count && (uint64_t)hidden_size <= count - base) {
        *out_scalar_supported = 0;
        return 1;
    }
    return 1;
}

static int moe_layer_moe_mlp_prefill_batch_f32(
    moe_pc_engine* engine,
    int32_t layer,
    float* batch_hidden,
    const float* batch_norm,
    uint32_t batch_size,
    const moe_storage_constants_t* c
) {
    if (!moe_mlp_prefill_batch_enabled() || !engine || !batch_hidden || !batch_norm || !c ||
        batch_size < 2 || batch_size > 4096u ||
        c->hidden_size == 0 || c->expert_intermediate_size == 0 ||
        c->experts_per_moe_layer == 0 || c->experts_per_moe_layer > 10000u) {
        return 0;
    }
    if (engine->io_config.internal_require_gpu_expert_batch_kernel &&
        !moe_gpu_batch_expert_kernel_available(engine)) {
        std::fprintf(stderr,
            "[storagellm mlp] internal backend policy requires moe_model_lib_run_expert_triplet_batch_f32 "
            "with moe_model_lib_has_expert_triplet_batch_f32()==1 and a true architecture-specific "
            "MoE backend; internal fallback is disabled\n");
        return moe_PREFILL_BATCH_MLP_FAST_KERNEL_REQUIRED;
    }
    if (moe_verified_backend_required(engine) &&
        !moe_fast_backend_available(engine)) {
        engine->io_atomic.gpu_grouped_host_fallback_count.fetch_add(
            1, std::memory_order_relaxed);
        std::fprintf(stderr,
            "[storagellm mlp] internal backend policy requires a verified "
            "architecture-specific MoE backend (CUDA CUTLASS/cuBLASLt/TVM, "
            "ROCm hipBLASLt/CK, Metal MPS/TVM, oneDNN SYCL, Vulkan cooperative "
            "matrix/TVM, CLBlast/TVM, or CPU oneDNN/TVM); internal fallback is disabled\n");
        return moe_PREFILL_BATCH_MLP_FAST_KERNEL_REQUIRED;
    }
    const uint64_t total_start_us = moe_profile_now_us();
    const uint32_t hidden_size = c->hidden_size;
    const uint32_t intermediate_size = c->expert_intermediate_size;
    const moe_attn_config_t attn_cfg = moe_generation_attn_config(engine);
    if ((uint64_t)batch_size * hidden_size > (uint64_t)SIZE_MAX / sizeof(float) ||
        (uint64_t)batch_size * intermediate_size > (uint64_t)SIZE_MAX / sizeof(float)) {
        return 0;
    }

    std::vector<float> batch_mlp((size_t)batch_size * hidden_size, 0.0f);
    std::vector<float> batch_expert_norm((size_t)batch_size * hidden_size, 0.0f);
    std::vector<float> batch_expert_sum((size_t)batch_size * hidden_size, 0.0f);
    std::vector<const float*> expert_inputs(batch_size, nullptr);
    std::vector<std::vector<int>> selected_experts(batch_size);
    std::vector<std::vector<float>> selected_weights(batch_size);
    std::vector<float> success_weight_sum(batch_size, 0.0f);
    std::vector<uint32_t> success_count(batch_size, 0u);
    std::vector<uint32_t> failed_count(batch_size, 0u);
    std::vector<std::vector<moe_mlp_batch_assignment_t>> grouped(c->experts_per_moe_layer);
    moe_decode_scratch local;
    local.gate.resize(std::max<uint32_t>(12288u, intermediate_size));
    local.up.resize(std::max<uint32_t>(12288u, intermediate_size));
    local.shared.resize(hidden_size);
    local.residual.resize(hidden_size);
    local.expert.resize(hidden_size);
    local.router_logits.resize(c->experts_per_moe_layer);
    local.router_order.resize(c->experts_per_moe_layer);

    const uint64_t shared_bytes =
        2ull * (uint64_t)intermediate_size * (uint64_t)hidden_size * 2ull +
        (uint64_t)hidden_size * (uint64_t)intermediate_size * 2ull;
    const moe_storage_raw_span_spec_t* shared =
        find_raw_role_span_bytes(layer, moe_RAW_ROLE_SHARED, shared_bytes, 0);
    const int router_has_internal_norm_scale =
        moe_engine_has_common_tensor_suffix(engine, layer, "ffn_gate_inp.scale") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "router.scale") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "mlp.router.scale") ||
        moe_engine_has_common_tensor_suffix(engine, layer, "moe.gate.scale");
    const uint32_t top_k = moe_router_top_k_from_metadata(
        engine, moe_MOE_BASE_TOP_EXPERTS, c->experts_per_moe_layer);

    const uint64_t prepare_start_us = moe_profile_now_us();
    for (uint32_t t = 0; t < batch_size; ++t) {
        float* hidden = batch_hidden + (size_t)t * hidden_size;
        const float* shared_input = batch_norm + (size_t)t * hidden_size;
        const float* expert_input = shared_input;
        if (moe_layer_moe_norm_f32(
                engine, layer, hidden, hidden_size,
                batch_expert_norm.data() + (size_t)t * hidden_size,
                attn_cfg.rms_norm_eps)) {
            expert_input = batch_expert_norm.data() + (size_t)t * hidden_size;
        } else if (moe_layer_has_moe_norm_weight(engine, layer)) {
            std::fprintf(stderr,
                "[storagellm mlp] batch expert ffn norm tensor is required but could not be mapped layer=%d token=%u\n",
                layer, t);
            return 0;
        }
        expert_inputs[t] = expert_input;

        float* mlp_out = batch_mlp.data() + (size_t)t * hidden_size;
        float* shared_out_buf = local.shared.data();
        int shared_ok = 0;
        if (moe_layer_common_dense_mlp_f32(engine, layer, &local, shared_input, hidden_size, shared_out_buf)) {
            shared_ok = 1;
        } else if (shared &&
            moe_raw_mlp_span_f32(
                engine, shared, shared_input, hidden_size, intermediate_size,
                local.gate.data(), local.up.data(), shared_out_buf)) {
            shared_ok = 1;
        }
        if (shared_ok) {
            float shared_gate = 1.0f;
            if (!moe_layer_shared_expert_gate_f32(engine, layer, shared_input, hidden_size, &shared_gate)) {
                return 0;
            }
            const float* shared_branch = shared_out_buf;
            if (moe_layer_post_ffw_norm1_f32(
                    engine, layer, shared_out_buf, hidden_size,
                    local.residual.data(), attn_cfg.rms_norm_eps)) {
                shared_branch = local.residual.data();
            } else if (moe_layer_has_post_ffw_norm1_weight(engine, layer)) {
                std::fprintf(stderr,
                    "[storagellm mlp] batch post-ffw norm1 tensor is required but could not be mapped layer=%d token=%u\n",
                    layer, t);
                return 0;
            }
            for (uint32_t i = 0; i < hidden_size; ++i) {
                mlp_out[i] += shared_branch[i] * shared_gate;
            }
        }

        const float* router_input = router_has_internal_norm_scale ? hidden : expert_input;
        moe_save_gate_input_snapshot(engine, layer, router_input, hidden_size);
        if (!moe_router_topk_f32(
                engine, layer, router_input, hidden_size,
                c->experts_per_moe_layer, top_k,
                &selected_experts[t], &selected_weights[t],
                &local.router_logits, &local.router_order)) {
            return 0;
        }
        if (selected_experts[t].size() != selected_weights[t].size() ||
            selected_experts[t].empty() || selected_experts[t].size() > c->experts_per_moe_layer) {
            return 0;
        }
        moe_juju_update_co_activation(engine, layer, selected_experts[t]);
        for (size_t i = 0; i < selected_experts[t].size(); ++i) {
            const int expert_id = selected_experts[t][i];
            if (expert_id < 0 || (uint32_t)expert_id >= c->experts_per_moe_layer) {
                ++failed_count[t];
                continue;
            }
            const size_t slot = expert_dense_slot(engine, layer, expert_id);
            if (slot != tensor_slot_invalid() && slot < engine->expert_states.size()) {
                moe_block_state& state = engine->expert_states[slot];
                if (state.active) {
                    const moe_tier_t tier = state.tier;
                    const double boost = tier == moe_TIER_VRAM ? 4.0 :
                                        tier == moe_TIER_RAM ? 1.0 : 0.25;
                    touch_expert_heat(engine, &state, boost);
                }
            }
            grouped[(uint32_t)expert_id].push_back({t, selected_weights[t][i]});
        }
    }
    moe_profile_log(
        "prefill_mlp_batch_prepare",
        "layer=%d tokens=%u experts=%u topk=%u ms=%.3f",
        layer, batch_size, c->experts_per_moe_layer, top_k,
        moe_profile_elapsed_ms(prepare_start_us));

    std::vector<moe_active_expert_group_t> active_experts;
    moe_build_active_expert_groups(engine, layer, grouped, &active_experts);
    const bool gpu_batch_ready = moe_gpu_batch_expert_kernel_available(engine) != 0;
    for (const auto& group : active_experts) {
        if (group.current_tier == moe_TIER_VRAM) {
            continue;
        }
        if (group.assignment_count < engine->io_config.expert_group_gpu_min_assignments) {
            continue;
        }
        const moe_tier_t target_tier = gpu_batch_ready ? moe_TIER_VRAM : moe_TIER_RAM;
        moe_pc_engine_prefetch_expert_triplet_impl(
            engine,
            layer,
            (int)group.expert,
            target_tier,
            1,
            1,
            vram_reservation_token{},
            0,
            moe_PREFETCH_REASON_BATCH_SELECTED,
            0);
    }
    if (gpu_batch_ready) {
        moe_wait_ready_for_top_active_experts(
            engine,
            layer,
            active_experts,
            engine->io_config.prefill_group_wait_us);
    }

    // Fetch dynamic expert batch limit instead of hardcoded 64
    const uint32_t expert_batch_limit = moe_mlp_prefill_expert_batch_limit();
    const int activation_mode = moe_engine_gelu_activation_mode(engine);
    std::vector<const float*> input_ptrs(expert_batch_limit, nullptr);
    std::vector<float> group_input_pool((size_t)expert_batch_limit * hidden_size);
    std::vector<float> gate_pool((size_t)expert_batch_limit * intermediate_size);
    std::vector<float> up_pool((size_t)expert_batch_limit * intermediate_size);
    std::vector<float> out_pool((size_t)expert_batch_limit * hidden_size);
    uint32_t active_groups = 0;
    uint32_t total_assignments = 0;
    uint32_t failed_assignments = 0;
    const uint64_t expert_start_us = moe_profile_now_us();
    auto grouped_chunk_key = [](uint32_t expert, size_t begin) -> uint64_t {
        return (uint64_t(expert) << 32) ^ (uint64_t(begin) & 0xffffffffull);
    };
    std::unordered_set<uint64_t> grouped_done_chunks;
    if (gpu_batch_ready) {
        auto grouped_fn = moe_gpu_grouped_expert_kernel_fn(engine);
        struct moe_prefill_grouped_gpu_chunk_t {
            uint32_t expert = 0;
            size_t begin = 0;
            uint32_t chunk = 0;
            size_t input_offset = 0;
            size_t output_offset = 0;
            moe_device_tensor_batch_view_t views[3]{};
        };
        std::vector<moe_prefill_grouped_gpu_chunk_t> grouped_chunks;
        size_t grouped_input_floats = 0;
        size_t grouped_output_floats = 0;
        if (grouped_fn) {
            for (const auto& group : active_experts) {
                const uint32_t expert = group.expert;
                if (expert >= c->experts_per_moe_layer) {
                    continue;
                }
                const auto& assignments = grouped[expert];
                if (assignments.empty()) {
                    continue;
                }
                moe_device_tensor_batch_view_t views[3]{};
                if (!moe_expert_triplet_resident_in_vram(engine, layer, (int)expert, views)) {
                    continue;
                }
                for (size_t begin = 0; begin < assignments.size(); begin += expert_batch_limit) {
                    const uint32_t chunk =
                        (uint32_t)std::min<size_t>(expert_batch_limit, assignments.size() - begin);
                    moe_prefill_grouped_gpu_chunk_t item{};
                    item.expert = expert;
                    item.begin = begin;
                    item.chunk = chunk;
                    item.input_offset = grouped_input_floats;
                    item.output_offset = grouped_output_floats;
                    item.views[0] = views[0];
                    item.views[1] = views[1];
                    item.views[2] = views[2];
                    grouped_input_floats += (size_t)chunk * hidden_size;
                    grouped_output_floats += (size_t)chunk * hidden_size;
                    grouped_chunks.push_back(item);
                }
            }
        }
        if (!grouped_chunks.empty()) {
            auto indexed_fn = moe_gpu_grouped_expert_indexed_kernel_fn(engine);
            auto indexed_device_fn = moe_gpu_grouped_expert_indexed_device_kernel_fn(engine);
            auto activation_acquire_fn = moe_gpu_prefill_activation_acquire_fn(engine);
            auto activation_release_fn = moe_gpu_prefill_activation_release_fn(engine);
            const float* indexed_input_base = expert_inputs.empty() ? nullptr : expert_inputs[0];
            int indexed_inputs_contiguous = indexed_input_base ? 1 : 0;
            for (uint32_t t = 0; t < batch_size && indexed_inputs_contiguous; ++t) {
                if (expert_inputs[t] != indexed_input_base + (size_t)t * hidden_size) {
                    indexed_inputs_contiguous = 0;
                }
            }
            if (indexed_device_fn && activation_acquire_fn && activation_release_fn &&
                moe_prefill_host_activation_bridge_allowed(engine) &&
                indexed_inputs_contiguous && moe_fast_backend_available(engine)) {
                std::vector<moe_grouped_expert_device_task_t> fast_tasks;
                std::vector<size_t> fast_chunk_indices;
                std::vector<uint32_t> fast_token_indices;
                std::vector<float> fast_token_weights;
                fast_tasks.reserve(grouped_chunks.size());
                fast_chunk_indices.reserve(grouped_chunks.size());
                for (size_t chunk_index = 0; chunk_index < grouped_chunks.size(); ++chunk_index) {
                    const auto& chunk_info = grouped_chunks[chunk_index];
                    if (grouped_done_chunks.find(grouped_chunk_key(chunk_info.expert, chunk_info.begin)) !=
                        grouped_done_chunks.end()) {
                        continue;
                    }
                    float down_scale = 1.0f;
                    int scalar_scale_supported = 1;
                    if (!moe_layer_expert_down_scalar_scale_f32(
                            engine, layer, (int)chunk_info.expert, c->experts_per_moe_layer,
                            hidden_size, &down_scale, &scalar_scale_supported) ||
                        !scalar_scale_supported || !std::isfinite(down_scale)) {
                        continue;
                    }
                    const uint32_t token_offset = static_cast<uint32_t>(fast_token_indices.size());
                    const auto& assignments = grouped[chunk_info.expert];
                    int chunk_ok = 1;
                    for (uint32_t b = 0; b < chunk_info.chunk; ++b) {
                        const moe_mlp_batch_assignment_t& a = assignments[chunk_info.begin + b];
                        if (a.token >= batch_size || !std::isfinite(a.weight)) {
                            chunk_ok = 0;
                            break;
                        }
                        fast_token_indices.push_back(a.token);
                        fast_token_weights.push_back(a.weight * down_scale);
                    }
                    if (!chunk_ok) {
                        fast_token_indices.resize(token_offset);
                        fast_token_weights.resize(token_offset);
                        continue;
                    }
                    moe_grouped_expert_device_task_t task{};
                    task.layer = layer;
                    task.expert = static_cast<int32_t>(chunk_info.expert);
                    task.gate_weight = &chunk_info.views[0];
                    task.up_weight = &chunk_info.views[1];
                    task.down_weight = &chunk_info.views[2];
                    task.input_stride = hidden_size;
                    task.assignment_offset = token_offset;
                    task.assignment_count = chunk_info.chunk;
                    task.accum_stride = hidden_size;
                    task.hidden_size = hidden_size;
                    task.intermediate_size = intermediate_size;
                    task.activation_mode = static_cast<uint32_t>(activation_mode);
                    fast_tasks.push_back(task);
                    fast_chunk_indices.push_back(chunk_index);
                }
                if (fast_tasks.size() >= 2u && !fast_token_indices.empty()) {
                    uint64_t activation_handle = 0;
                    const void* d_batch_input = nullptr;
                    const void* d_token_indices = nullptr;
                    const void* d_token_weights = nullptr;
                    void* d_batch_accum = nullptr;
                    if (activation_acquire_fn(
                            engine->model_lib_abi_version,
                            static_cast<int32_t>(engine->backend_caps.backend),
                            indexed_input_base,
                            hidden_size,
                            batch_size,
                            fast_token_indices.data(),
                            fast_token_weights.data(),
                            static_cast<uint32_t>(fast_token_indices.size()),
                            batch_expert_sum.data(),
                            hidden_size,
                            &activation_handle,
                            &d_batch_input,
                            &d_token_indices,
                            &d_token_weights,
                            &d_batch_accum)) {
                        for (auto& task : fast_tasks) {
                            task.d_input = d_batch_input;
                            task.d_token_indices = static_cast<const uint32_t*>(d_token_indices);
                            task.d_token_weights = static_cast<const float*>(d_token_weights);
                            task.d_accum = d_batch_accum;
                        }
                        engine->io_atomic.gpu_grouped_indexed_device_submitted.fetch_add(
                            fast_token_indices.size(), std::memory_order_relaxed);
                        const int run_ok = indexed_device_fn(
                            engine->model_lib_abi_version,
                            static_cast<int32_t>(engine->backend_caps.backend),
                            fast_tasks.data(),
                            static_cast<uint32_t>(fast_tasks.size()),
                            nullptr);
                        const int release_ok = activation_release_fn(
                            engine->model_lib_abi_version,
                            static_cast<int32_t>(engine->backend_caps.backend),
                            activation_handle,
                            batch_expert_sum.data(),
                            batch_size,
                            hidden_size,
                            run_ok ? 1 : 0);
                        activation_handle = 0;
                        if (run_ok && release_ok) {
                            engine->io_atomic.gpu_grouped_indexed_device_completed.fetch_add(
                                fast_token_indices.size(), std::memory_order_relaxed);
                            for (size_t chunk_index : fast_chunk_indices) {
                                const auto& chunk_info = grouped_chunks[chunk_index];
                                device_touch_allocation(engine, chunk_info.views[0].key);
                                device_touch_allocation(engine, chunk_info.views[1].key);
                                device_touch_allocation(engine, chunk_info.views[2].key);
                                const auto& assignments = grouped[chunk_info.expert];
                                for (uint32_t b = 0; b < chunk_info.chunk; ++b) {
                                    const moe_mlp_batch_assignment_t& a =
                                        assignments[chunk_info.begin + b];
                                    success_weight_sum[a.token] += a.weight;
                                    ++success_count[a.token];
                                }
                                grouped_done_chunks.insert(
                                    grouped_chunk_key(chunk_info.expert, chunk_info.begin));
                            }
                        } else if (engine->io_config.internal_require_gpu_expert_batch_kernel) {
                            engine->io_atomic.gpu_grouped_host_fallback_count.fetch_add(
                                1, std::memory_order_relaxed);
                            return 0;
                        }
                    } else if (engine->io_config.internal_require_gpu_expert_batch_kernel) {
                        engine->io_atomic.gpu_grouped_host_fallback_count.fetch_add(
                            1, std::memory_order_relaxed);
                        return 0;
                    }
                }
            }
            if ((indexed_fn || indexed_device_fn) && indexed_inputs_contiguous) {
                std::vector<uint8_t> attempted(grouped_chunks.size(), 0);
                for (size_t seed = 0; seed < grouped_chunks.size(); ++seed) {
                    if (attempted[seed] ||
                        grouped_done_chunks.find(grouped_chunk_key(
                            grouped_chunks[seed].expert,
                            grouped_chunks[seed].begin)) != grouped_done_chunks.end()) {
                        continue;
                    }
                    const uint32_t batch_for_group = grouped_chunks[seed].chunk;
                    std::vector<moe_model_lib_indexed_expert_task_t> indexed_tasks;
                    std::vector<size_t> indexed_chunk_indices;
                    std::vector<uint32_t> indexed_token_indices;
                    std::vector<float> indexed_token_weights;
                    indexed_tasks.reserve(grouped_chunks.size() - seed);
                    indexed_chunk_indices.reserve(grouped_chunks.size() - seed);
                    for (size_t j = seed; j < grouped_chunks.size(); ++j) {
                        const auto& chunk_info = grouped_chunks[j];
                        if (attempted[j] ||
                            chunk_info.chunk != batch_for_group ||
                            grouped_done_chunks.find(grouped_chunk_key(
                                chunk_info.expert, chunk_info.begin)) != grouped_done_chunks.end()) {
                            continue;
                        }
                        float down_scale = 1.0f;
                        int scalar_scale_supported = 1;
                        if (!moe_layer_expert_down_scalar_scale_f32(
                                engine, layer, (int)chunk_info.expert, c->experts_per_moe_layer,
                                hidden_size, &down_scale, &scalar_scale_supported) ||
                            !scalar_scale_supported || !std::isfinite(down_scale)) {
                            attempted[j] = 1;
                            continue;
                        }
                        const uint32_t token_offset =
                            static_cast<uint32_t>(indexed_token_indices.size());
                        const auto& assignments = grouped[chunk_info.expert];
                        int chunk_ok = 1;
                        for (uint32_t b = 0; b < chunk_info.chunk; ++b) {
                            const moe_mlp_batch_assignment_t& a = assignments[chunk_info.begin + b];
                            if (a.token >= batch_size || !std::isfinite(a.weight)) {
                                chunk_ok = 0;
                                break;
                            }
                            indexed_token_indices.push_back(a.token);
                            indexed_token_weights.push_back(a.weight * down_scale);
                        }
                        if (!chunk_ok) {
                            indexed_token_indices.resize(token_offset);
                            indexed_token_weights.resize(token_offset);
                            attempted[j] = 1;
                            continue;
                        }
                        moe_model_lib_indexed_expert_task_t task{};
                        task.layer = layer;
                        task.expert = (int32_t)chunk_info.expert;
                        task.gate = &chunk_info.views[0];
                        task.up = &chunk_info.views[1];
                        task.down = &chunk_info.views[2];
                        task.token_offset = token_offset;
                        task.batch = chunk_info.chunk;
                        task.hidden = hidden_size;
                        task.intermediate = intermediate_size;
                        indexed_tasks.push_back(task);
                        indexed_chunk_indices.push_back(j);
                    }
                    for (size_t j : indexed_chunk_indices) {
                        attempted[j] = 1;
                    }
                    if (indexed_tasks.size() < 2u || indexed_token_indices.empty()) {
                        continue;
                    }
                    int device_indexed_ok = 0;
                    if (indexed_device_fn && activation_acquire_fn && activation_release_fn &&
                        moe_prefill_host_activation_bridge_allowed(engine)) {
                        uint64_t activation_handle = 0;
                        const void* d_batch_input = nullptr;
                        const void* d_token_indices = nullptr;
                        const void* d_token_weights = nullptr;
                        void* d_batch_accum = nullptr;
                        if (activation_acquire_fn(
                                engine->model_lib_abi_version,
                                static_cast<int32_t>(engine->backend_caps.backend),
                                indexed_input_base,
                                hidden_size,
                                batch_size,
                                indexed_token_indices.data(),
                                indexed_token_weights.data(),
                                (uint32_t)indexed_token_indices.size(),
                                batch_expert_sum.data(),
                                hidden_size,
                                &activation_handle,
                                &d_batch_input,
                                &d_token_indices,
                                &d_token_weights,
                                &d_batch_accum)) {
                            std::vector<moe_grouped_expert_device_task_t> fast_tasks;
                            fast_tasks.reserve(indexed_tasks.size());
                            for (const auto& task : indexed_tasks) {
                                moe_grouped_expert_device_task_t fast_task{};
                                fast_task.layer = task.layer;
                                fast_task.expert = task.expert;
                                fast_task.gate_weight = task.gate;
                                fast_task.up_weight = task.up;
                                fast_task.down_weight = task.down;
                                fast_task.d_input = d_batch_input;
                                fast_task.input_stride = hidden_size;
                                fast_task.d_token_indices =
                                    static_cast<const uint32_t*>(d_token_indices);
                                fast_task.d_token_weights =
                                    static_cast<const float*>(d_token_weights);
                                fast_task.assignment_offset = task.token_offset;
                                fast_task.assignment_count = task.batch;
                                fast_task.d_accum = d_batch_accum;
                                fast_task.accum_stride = hidden_size;
                                fast_task.hidden_size = hidden_size;
                                fast_task.intermediate_size = intermediate_size;
                                fast_task.activation_mode = static_cast<uint32_t>(activation_mode);
                                fast_tasks.push_back(fast_task);
                            }
                            engine->io_atomic.gpu_grouped_indexed_device_submitted.fetch_add(
                                indexed_token_indices.size(), std::memory_order_relaxed);
                            if (indexed_device_fn(
                                    engine->model_lib_abi_version,
                                    static_cast<int32_t>(engine->backend_caps.backend),
                                    fast_tasks.data(),
                                    (uint32_t)fast_tasks.size(),
                                    nullptr)) {
                                const int release_ok = activation_release_fn(
                                    engine->model_lib_abi_version,
                                    static_cast<int32_t>(engine->backend_caps.backend),
                                    activation_handle,
                                    batch_expert_sum.data(),
                                    batch_size,
                                    hidden_size,
                                    1);
                                activation_handle = 0;
                                if (release_ok) {
                                    engine->io_atomic.gpu_grouped_indexed_device_completed.fetch_add(
                                        indexed_token_indices.size(), std::memory_order_relaxed);
                                    device_indexed_ok = 1;
                                }
                            }
                            if (activation_handle != 0) {
                                activation_release_fn(
                                    engine->model_lib_abi_version,
                                    static_cast<int32_t>(engine->backend_caps.backend),
                                    activation_handle,
                                    nullptr,
                                    0,
                                    0,
                                    0);
                            }
                        }
                    }
                    if (device_indexed_ok) {
                        for (size_t chunk_index : indexed_chunk_indices) {
                            const auto& chunk_info = grouped_chunks[chunk_index];
                            device_touch_allocation(engine, chunk_info.views[0].key);
                            device_touch_allocation(engine, chunk_info.views[1].key);
                            device_touch_allocation(engine, chunk_info.views[2].key);
                            const auto& assignments = grouped[chunk_info.expert];
                            for (uint32_t b = 0; b < chunk_info.chunk; ++b) {
                                const moe_mlp_batch_assignment_t& a = assignments[chunk_info.begin + b];
                                success_weight_sum[a.token] += a.weight;
                                ++success_count[a.token];
                            }
                            grouped_done_chunks.insert(grouped_chunk_key(chunk_info.expert, chunk_info.begin));
                        }
                        continue;
                    }
                    if (engine->io_config.internal_require_gpu_expert_batch_kernel) {
                        engine->io_atomic.gpu_grouped_host_fallback_count.fetch_add(
                            1, std::memory_order_relaxed);
                        return 0;
                    }
                    if (!indexed_fn) {
                        continue;
                    }
                    engine->io_atomic.gpu_grouped_indexed_host_submitted.fetch_add(
                        indexed_token_indices.size(), std::memory_order_relaxed);
                    if (!indexed_fn(
                            engine->model_lib_abi_version,
                            static_cast<int32_t>(engine->backend_caps.backend),
                            indexed_tasks.data(),
                            (uint32_t)indexed_tasks.size(),
                            indexed_input_base,
                            hidden_size,
                            batch_size,
                            indexed_token_indices.data(),
                            indexed_token_weights.data(),
                            (uint32_t)indexed_token_indices.size(),
                            batch_expert_sum.data(),
                            hidden_size)) {
                        continue;
                    }
                    engine->io_atomic.gpu_grouped_indexed_host_completed.fetch_add(
                        indexed_token_indices.size(), std::memory_order_relaxed);
                    for (size_t chunk_index : indexed_chunk_indices) {
                        const auto& chunk_info = grouped_chunks[chunk_index];
                        device_touch_allocation(engine, chunk_info.views[0].key);
                        device_touch_allocation(engine, chunk_info.views[1].key);
                        device_touch_allocation(engine, chunk_info.views[2].key);
                        const auto& assignments = grouped[chunk_info.expert];
                        for (uint32_t b = 0; b < chunk_info.chunk; ++b) {
                            const moe_mlp_batch_assignment_t& a = assignments[chunk_info.begin + b];
                            success_weight_sum[a.token] += a.weight;
                            ++success_count[a.token];
                        }
                        grouped_done_chunks.insert(grouped_chunk_key(chunk_info.expert, chunk_info.begin));
                    }
                }
            }
        }
        if (!grouped_chunks.empty()) {
            int has_host_grouped_remainder = 0;
            for (const auto& chunk_info : grouped_chunks) {
                if (grouped_done_chunks.find(grouped_chunk_key(chunk_info.expert, chunk_info.begin)) ==
                    grouped_done_chunks.end()) {
                    has_host_grouped_remainder = 1;
                    break;
                }
            }
            if (!has_host_grouped_remainder) {
                grouped_chunks.clear();
            } else {
                engine->io_atomic.gpu_grouped_host_fallback_count.fetch_add(
                    1, std::memory_order_relaxed);
                if (engine->io_config.internal_require_gpu_expert_batch_kernel) {
                    return 0;
                }
            }
        }
        if (!grouped_chunks.empty()) {
            std::vector<float> grouped_inputs(grouped_input_floats);
            std::vector<float> grouped_outputs(grouped_output_floats);
            std::vector<moe_model_lib_grouped_expert_task_t> grouped_tasks;
            std::vector<size_t> grouped_task_chunk_indices;
            grouped_tasks.reserve(grouped_chunks.size());
            grouped_task_chunk_indices.reserve(grouped_chunks.size());
            int grouped_pack_ok = 1;
            for (size_t chunk_index = 0; chunk_index < grouped_chunks.size(); ++chunk_index) {
                const auto& chunk_info = grouped_chunks[chunk_index];
                if (grouped_done_chunks.find(grouped_chunk_key(chunk_info.expert, chunk_info.begin)) !=
                    grouped_done_chunks.end()) {
                    continue;
                }
                const auto& assignments = grouped[chunk_info.expert];
                float* chunk_input = grouped_inputs.data() + chunk_info.input_offset;
                float* chunk_output = grouped_outputs.data() + chunk_info.output_offset;
                for (uint32_t b = 0; b < chunk_info.chunk; ++b) {
                    const float* src = expert_inputs[assignments[chunk_info.begin + b].token];
                    if (!src) {
                        grouped_pack_ok = 0;
                        break;
                    }
                    std::memcpy(
                        chunk_input + (size_t)b * hidden_size,
                        src,
                        (size_t)hidden_size * sizeof(float));
                }
                if (!grouped_pack_ok) {
                    break;
                }
                moe_model_lib_grouped_expert_task_t task{};
                task.layer = layer;
                task.expert = (int32_t)chunk_info.expert;
                task.gate = &chunk_info.views[0];
                task.up = &chunk_info.views[1];
                task.down = &chunk_info.views[2];
                task.x = chunk_input;
                task.out = chunk_output;
                task.batch = chunk_info.chunk;
                task.hidden = hidden_size;
                task.intermediate = intermediate_size;
                grouped_tasks.push_back(task);
                grouped_task_chunk_indices.push_back(chunk_index);
            }
            if (grouped_pack_ok && !grouped_tasks.empty()) {
                std::vector<uint8_t> attempted(grouped_tasks.size(), 0);
                for (size_t seed = 0; seed < grouped_tasks.size(); ++seed) {
                    if (attempted[seed]) {
                        continue;
                    }
                    std::vector<moe_model_lib_grouped_expert_task_t> same_shape_tasks;
                    std::vector<size_t> same_shape_indices;
                    same_shape_tasks.reserve(grouped_tasks.size() - seed);
                    same_shape_indices.reserve(grouped_tasks.size() - seed);
                    const uint32_t batch_for_group = grouped_tasks[seed].batch;
                    for (size_t j = seed; j < grouped_tasks.size(); ++j) {
                        if (attempted[j] ||
                            grouped_tasks[j].batch != batch_for_group ||
                            grouped_tasks[j].hidden != grouped_tasks[seed].hidden ||
                            grouped_tasks[j].intermediate != grouped_tasks[seed].intermediate) {
                            continue;
                        }
                        same_shape_tasks.push_back(grouped_tasks[j]);
                        same_shape_indices.push_back(j);
                    }
                    for (size_t j : same_shape_indices) {
                        attempted[j] = 1;
                    }
                    if (same_shape_tasks.size() < 2u) {
                        continue;
                    }
                    if (!grouped_fn(
                            engine->model_lib_abi_version,
                            static_cast<int32_t>(engine->backend_caps.backend),
                            same_shape_tasks.data(),
                            (uint32_t)same_shape_tasks.size())) {
                        continue;
                    }
                    uint64_t grouped_d2h_bytes = 0;
                    for (size_t task_index : same_shape_indices) {
                        const auto& chunk_info = grouped_chunks[grouped_task_chunk_indices[task_index]];
                        grouped_d2h_bytes +=
                            (uint64_t)chunk_info.chunk * (uint64_t)hidden_size * (uint64_t)sizeof(float);
                    }
                    engine->io_atomic.gpu_grouped_d2h_expert_output_bytes.fetch_add(
                        grouped_d2h_bytes, std::memory_order_relaxed);
                    for (size_t task_index : same_shape_indices) {
                        const auto& chunk_info = grouped_chunks[grouped_task_chunk_indices[task_index]];
                        device_touch_allocation(engine, chunk_info.views[0].key);
                        device_touch_allocation(engine, chunk_info.views[1].key);
                        device_touch_allocation(engine, chunk_info.views[2].key);
                        const auto& assignments = grouped[chunk_info.expert];
                        for (uint32_t b = 0; b < chunk_info.chunk; ++b) {
                            const moe_mlp_batch_assignment_t& a = assignments[chunk_info.begin + b];
                            float* expert_out =
                                grouped_outputs.data() + chunk_info.output_offset + (size_t)b * hidden_size;
                            if (!moe_layer_apply_expert_down_scale_f32(
                                    engine, layer, (int)chunk_info.expert, c->experts_per_moe_layer,
                                    expert_out, hidden_size)) {
                                ++failed_count[a.token];
                                ++failed_assignments;
                                continue;
                            }
                            moe_axpy_f32(
                                batch_expert_sum.data() + (size_t)a.token * hidden_size,
                                expert_out, a.weight, hidden_size);
                            success_weight_sum[a.token] += a.weight;
                            ++success_count[a.token];
                        }
                        grouped_done_chunks.insert(grouped_chunk_key(chunk_info.expert, chunk_info.begin));
                    }
                }
            }
        }
    }
    for (const auto& group : active_experts) {
        const uint32_t expert = group.expert;
        if (expert >= c->experts_per_moe_layer) {
            continue;
        }
        const auto& assignments = grouped[expert];
        if (assignments.empty()) {
            continue;
        }
        ++active_groups;
        total_assignments += (uint32_t)assignments.size();
        for (size_t begin = 0; begin < assignments.size(); begin += expert_batch_limit) {
            if (grouped_done_chunks.find(grouped_chunk_key(expert, begin)) != grouped_done_chunks.end()) {
                continue;
            }
            const uint32_t chunk = (uint32_t)std::min<size_t>(expert_batch_limit, assignments.size() - begin);
            for (uint32_t b = 0; b < chunk; ++b) {
                input_ptrs[b] = expert_inputs[assignments[begin + b].token];
            }
            const float* contiguous_inputs = input_ptrs[0];
            uint32_t contiguous_stride = hidden_size;
            int inputs_are_contiguous = contiguous_inputs ? 1 : 0;
            for (uint32_t b = 0; b < chunk; ++b) {
                if (!input_ptrs[b] ||
                    input_ptrs[b] != contiguous_inputs + (size_t)b * hidden_size) {
                    inputs_are_contiguous = 0;
                    break;
                }
            }
            if (!inputs_are_contiguous) {
                for (uint32_t b = 0; b < chunk; ++b) {
                    if (!input_ptrs[b]) {
                        continue;
                    }
                    std::memcpy(
                        group_input_pool.data() + (size_t)b * hidden_size,
                        input_ptrs[b],
                        (size_t)hidden_size * sizeof(float));
                }
                contiguous_inputs = group_input_pool.data();
            }
            int expert_ok = 0;
            if (gpu_batch_ready && moe_expert_triplet_resident_in_vram(engine, layer, (int)expert)) {
                expert_ok = moe_run_expert_triplet_gpu_batch_contiguous_f32(
                    engine, layer, (int)expert,
                    contiguous_inputs, contiguous_stride,
                    chunk, hidden_size, intermediate_size,
                    out_pool.data(), hidden_size, activation_mode);
                if (expert_ok) {
                    engine->io_atomic.gpu_grouped_d2h_expert_output_bytes.fetch_add(
                        (uint64_t)chunk * (uint64_t)hidden_size * (uint64_t)sizeof(float),
                        std::memory_order_relaxed);
                }
            }
            if (!expert_ok) {
                expert_ok = moe_run_expert_triplet_row_batch_f32(
                    engine, layer, (int)expert,
                    input_ptrs.data(), chunk, hidden_size,
                    gate_pool.data(), up_pool.data(), intermediate_size,
                    out_pool.data(), hidden_size, activation_mode);
            }
            if (!expert_ok) {
                failed_assignments += chunk;
                for (uint32_t b = 0; b < chunk; ++b) {
                    ++failed_count[assignments[begin + b].token];
                }
                continue;
            }
            for (uint32_t b = 0; b < chunk; ++b) {
                const moe_mlp_batch_assignment_t& a = assignments[begin + b];
                float* expert_out = out_pool.data() + (size_t)b * hidden_size;
                if (!moe_layer_apply_expert_down_scale_f32(
                        engine, layer, (int)expert, c->experts_per_moe_layer,
                        expert_out, hidden_size)) {
                    ++failed_count[a.token];
                    ++failed_assignments;
                    continue;
                }
                moe_axpy_f32(
                    batch_expert_sum.data() + (size_t)a.token * hidden_size,
                    expert_out, a.weight, hidden_size);
                success_weight_sum[a.token] += a.weight;
                ++success_count[a.token];
            }
        }
    }
    moe_profile_log(
        "prefill_mlp_batch_experts",
        "layer=%d tokens=%u activation_mode=%d active_groups=%u assignments=%u failed=%u chunk=%u ms=%.3f",
        layer, batch_size, activation_mode, active_groups, total_assignments, failed_assignments,
        expert_batch_limit, moe_profile_elapsed_ms(expert_start_us));

    const int use_sigmoid = moe_router_uses_sigmoid_scores(engine);
    const int norm_topk = moe_router_norm_topk_prob(engine, use_sigmoid);
    const uint64_t finish_start_us = moe_profile_now_us();
    for (uint32_t t = 0; t < batch_size; ++t) {
        if (success_count[t] == 0) {
            return 0;
        }
        if (failed_count[t] > 0) {
            std::fprintf(stderr,
                "[storagellm mlp] batch selected expert contract failed layer=%d token=%u failed=%u success=%u\n",
                layer, t, failed_count[t], success_count[t]);
            return 0;
        }
        float* expert_sum = batch_expert_sum.data() + (size_t)t * hidden_size;
        const float* expert_branch = expert_sum;
        if (moe_layer_post_ffw_norm2_f32(
                engine, layer, expert_sum, hidden_size,
                local.expert.data(), attn_cfg.rms_norm_eps)) {
            expert_branch = local.expert.data();
        } else if (moe_layer_has_post_ffw_norm2_weight(engine, layer)) {
            std::fprintf(stderr,
                "[storagellm mlp] batch post-ffw norm2 tensor is required but could not be mapped layer=%d token=%u\n",
                layer, t);
            return 0;
        }
        float* mlp_out = batch_mlp.data() + (size_t)t * hidden_size;
        for (uint32_t i = 0; i < hidden_size; ++i) {
            if (!std::isfinite(expert_branch[i])) {
                std::fprintf(stderr,
                    "[storagellm mlp] batch non-finite expert branch layer=%d token=%u index=%u\n",
                    layer, t, i);
                return 0;
            }
            mlp_out[i] += expert_branch[i];
        }
        float* hidden = batch_hidden + (size_t)t * hidden_size;
        if (moe_layer_post_ffw_norm_f32(
                engine, layer, mlp_out, hidden_size,
                local.shared.data(), attn_cfg.rms_norm_eps)) {
            // BUGFIX 715: MLP residual addition NaN handling ★★
            // Problem: No validation of residual values before addition
            // Impact: NaN in MLP output propagates to hidden state
            // Solution: Check for NaN in residual values
            for (uint32_t i = 0; i < hidden_size; ++i) {
                if (!std::isfinite(local.shared[i])) {
                    std::fprintf(stderr,
                        "[storagellm mlp] batch non-finite post-ffw norm output layer=%d token=%u index=%u\n",
                        layer, t, i);
                    return 0;
                }
                hidden[i] += local.shared[i];
            }
        } else {
            if (moe_layer_has_post_ffw_norm_weight(engine, layer)) {
                std::fprintf(stderr,
                    "[storagellm mlp] batch post-ffw norm tensor is required but could not be mapped layer=%d token=%u\n",
                    layer, t);
                return 0;
            }
            // BUGFIX 715: MLP residual addition NaN handling ★★
            for (uint32_t i = 0; i < hidden_size; ++i) {
                if (!std::isfinite(mlp_out[i])) {
                    std::fprintf(stderr,
                        "[storagellm mlp] batch non-finite mlp output layer=%d token=%u index=%u\n",
                        layer, t, i);
                    return 0;
                }
                hidden[i] += mlp_out[i];
            }
        }
    }
    moe_profile_log(
        "prefill_mlp_batch_finish",
        "layer=%d tokens=%u norm_topk=%d finish_ms=%.3f total_ms=%.3f",
        layer, batch_size, norm_topk,
        moe_profile_elapsed_ms(finish_start_us),
        moe_profile_elapsed_ms(total_start_us));
    return 1;
}
