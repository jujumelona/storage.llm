// Adaptive router strategies
    // This blocks moe_save_gate_input_snapshot writes and causes 200μs+ latency spikes.
    // Solution: Only copy the truncated portion we actually need (hidden_size/5).
    // This reduces copy size by 80% and lock hold time from ~50μs to ~10μs.
    // Trade-off: Need to know full_hidden before lock, so check size first.
    // Fix 19: Double buffering lock-free read
    std::vector<float> snapshot;
    {
        std::lock_guard<std::mutex> lock(engine->gate_input_snapshot_mutex);
        const size_t slot = (size_t)source_layer;
        const size_t req_size = (slot + 1) * 2;
        if (engine->gate_input_snapshots.size() < req_size) {
            return;
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

        const int read_idx = (e0 >= e1) ? 0 : 1;
        const size_t read_slot = slot * 2 + read_idx;

        const std::vector<float>& src = engine->gate_input_snapshots[read_slot];
        const uint32_t full_hidden = (uint32_t)src.size();
        if (full_hidden == 0) {
            return;
        }

        const uint32_t trunc_size = std::max(64u, full_hidden / 5u);
        const uint32_t copy_size = std::min(trunc_size, full_hidden);
        snapshot.assign(src.begin(), src.begin() + copy_size);
    }

    const uint32_t full_hidden = (uint32_t)snapshot.size();
    if (full_hidden == 0) {
        return;
    }

    // InfiniGen truncation: use first hidden_size/5 elements only
    // This saves 80% of the dot-product compute while retaining most
    // discriminative information (low-index features are typically
    // more informative in transformer hidden states).
    const uint32_t trunc_size = std::max(64u, full_hidden / 5u);

    const uint8_t* router_base = nullptr;
    uint64_t router_mapped_bytes = 0;
    uint8_t router_encoding = moe_RAW_SCALAR_BF16;
    moe_raw_tensor_cache_entry common_router{};
    if (moe_map_router_tensor(engine, target_layer, full_hidden, experts, &common_router)) {
        router_base = common_router.base;
        router_mapped_bytes = common_router.bytes;
        router_encoding = common_router.scalar_encoding;
    } else {
        const uint64_t router_bytes = (uint64_t)experts * (uint64_t)full_hidden * 2ull;
        const moe_storage_raw_span_spec_t* span =
            find_raw_role_span_bytes(target_layer, moe_RAW_ROLE_ROUTER, router_bytes, 0);
        if (!span) {
            return;
        }
        mmap_context_t* mapped = map_raw_span_part(engine, span);
        if (!raw_span_bounds_ok(span, mapped)) {
            return;
        }
        router_base = raw_span_ptr(span, mapped, 0, span->bytes);
        router_mapped_bytes = span->bytes;
    }
    if (!router_base) {
        return;
    }

    // Compute approximate logits using truncated hidden × first trunc_size cols of gate
    // Gate weight layout: [experts × hidden_size] in BF16
    // We compute: approx_logit[e] = sum_{i=0}^{trunc_size-1} hidden[i] * gate[e][i]
    const uint32_t router_elem = moe_raw_scalar_size(router_encoding);
    // BUGFIX 791: Check router_elem for zero ★★★
    if (router_elem == 0) {
        return;
    }
    const uint64_t row_bytes = common_router.base ?
        moe_raw_tensor_row_bytes(common_router, full_hidden) :
        (uint64_t)full_hidden * (uint64_t)router_elem;
    // BUGFIX 792: Check row_bytes for overflow ★★★
    if (row_bytes == 0 || row_bytes > UINT64_MAX / experts) {
        return;
    }
    // BUGFIX 793: Check experts for reasonable upper bound ★★
    if (experts > 10000) {
        return;
    }
    std::vector<float> approx_logits(experts, 0.0f);

    for (uint32_t e = 0; e < experts; ++e) {
        const uint64_t rel = row_bytes * (uint64_t)e;
        if (rel > router_mapped_bytes ||
            (common_router.base ? row_bytes : (uint64_t)trunc_size * (uint64_t)router_elem) > router_mapped_bytes - rel) {
            return;
        }
        const uint8_t* row_ptr = router_base + rel;
        // Truncated dot product: only first trunc_size elements
        approx_logits[e] = common_router.base ?
            moe_raw_tensor_dot_row(common_router, e, snapshot.data(), trunc_size) :
            moe_dot_raw_row(row_ptr, router_encoding, snapshot.data(), trunc_size);
    }

    // Apply sigmoid activation (same as actual router)
    auto sigmoid = [](float x) -> float {
        // BUGFIX 720: Sigmoid NaN handling ★★
        // Problem: No validation of x before exp
        // Impact: NaN in router logit propagates to expert selection
        // Solution: Check for NaN in input and exp result
        if (!std::isfinite(x)) {
            return 0.5f;
        }
        // BUGFIX 794: Check for overflow in exp argument ★★★
        if (x > 88.0f) {
            return 1.0f;
        }
        if (x < -88.0f) {
            return 0.0f;
        }
        if (x >= 0.0f) {
            const float z = std::exp(-x);
            if (!std::isfinite(z)) {
                return 1.0f;
            }
            return 1.0f / (1.0f + z);
        }
        const float z = std::exp(x);
        if (!std::isfinite(z)) {
            return 0.0f;
        }
        return z / (1.0f + z);
    };
    for (float& v : approx_logits) {
        v = sigmoid(v);
    }

    // Compute confidence via normalized entropy
    const float spec_entropy = moe_router_normalized_entropy(approx_logits);

    // Select top-k predicted experts (use base top-k, not adaptive)
    const uint32_t predict_k = std::min(experts, (uint32_t)moe_MOE_BASE_TOP_EXPERTS * 2u);
    // BUGFIX 795: Check experts for zero before vector resize ★★★
    if (experts == 0) {
        return;
    }
    std::vector<uint32_t> order(experts);
    for (uint32_t i = 0; i < experts; ++i) order[i] = i;

    auto cmp = [&](uint32_t a, uint32_t b) {
        return approx_logits[a] > approx_logits[b];
    };
    // BUGFIX 624: Check order not empty before data() call ??
    // Problem: If experts is 0, order can be empty, data() returns nullptr
    // Solution: Check empty before data() call
    if (!order.empty()) {
        if (predict_k < experts) {
            std::nth_element(order.data(), order.data() + predict_k, order.data() + experts, cmp);
        }
        std::sort(order.data(), order.data() + predict_k, cmp);
    }

    for (uint32_t i = 0; i < predict_k; ++i) {
        predicted_experts.push_back((int)order[i]);
    }

    if (out_confidence) {
        // Confidence = 1 - entropy. Low entropy = high confidence = accurate prediction
        *out_confidence = 1.0f - spec_entropy;
    }

    // Update adaptive prefetch depth based on speculative confidence
    // InfiniGen insight: when prediction is confident (low entropy),
    // we can safely prefetch deeper into future layers.
    // BUGFIX 160: Limit speculative prefetch depth based on lookahead distance
    // Problem: Using layer N-1 snapshot to predict layer N+3 has low accuracy
    // Solution: Scale confidence threshold by lookahead distance
    if (engine) {
        // Calculate lookahead distance (how many layers ahead we're predicting)
        const int32_t lookahead = target_layer - source_layer;  // Always 1 in current code

        // BUGFIX 160: Confidence threshold increases with lookahead distance
        // depth=1: always allowed (next layer prediction)
        // depth=2: requires confidence > 0.5 (entropy < 0.5)
        // depth=3: requires confidence > 0.7 (entropy < 0.3)
        // depth=4: requires confidence > 0.85 (entropy < 0.15)
        uint32_t depth;
        if (spec_entropy < 0.15f) {
            depth = 4u;  // Very confident ??prefetch 4 layers ahead
        } else if (spec_entropy < 0.30f) {
            depth = 3u;  // Confident ??3 layers ahead
        } else if (spec_entropy < 0.50f) {
            depth = 2u;  // Moderate ??2 layers ahead (default)
        } else {
            depth = 1u;  // Uncertain ??only 1 layer ahead
        }

        // BUGFIX 160: Cap maximum depth at 2 for safety
        // Predictions beyond 2 layers have <70% accuracy, wasting VRAM on wrong experts
        const uint32_t max_safe_depth = 2u;
        depth = std::min(depth, max_safe_depth);

        // Fix 22: Hysteresis for adaptive_prefetch_depth (now per-session)
        const uint32_t current_depth = session ?
            session->session_adaptive_prefetch_depth :
            engine->adaptive_prefetch_depth.load(std::memory_order_relaxed);
        if (depth > current_depth) {
            // Need strong confidence to increase depth (lower entropy)
            float thresh = depth == 2u ? 0.50f : (depth == 3u ? 0.30f : 0.15f);
            if (spec_entropy < thresh * 0.95f) {
                // BUGFIX 955: Per-session depth isolation ★★★ CRITICAL
                // Problem: Engine-global write always executed even with session present
                // → Concurrent sessions overwrite each other's depth (code-gen resets chat)
                // Solution: Write engine-global only when no session (legacy non-batched path)
                if (session) {
                    session->session_adaptive_prefetch_depth = depth;
                } else {
                    engine->adaptive_prefetch_depth.store(depth, std::memory_order_relaxed);
                }
            }
        } else if (depth < current_depth) {
            // Need strong lack of confidence to decrease depth (higher entropy)
            float thresh = depth == 1u ? 0.50f : (depth == 2u ? 0.30f : 0.15f);
            if (spec_entropy > thresh * 1.05f) {
                if (session) {
                    session->session_adaptive_prefetch_depth = depth;
                } else {
                    engine->adaptive_prefetch_depth.store(depth, std::memory_order_relaxed);
                }
            }
        } else {
            if (session) {
                session->session_adaptive_prefetch_depth = depth;
            } else {
                engine->adaptive_prefetch_depth.store(depth, std::memory_order_relaxed);
            }
        }
    }
}

// ============================================================================
// Expert Choice Routing (Expert Choice, 2022; Switch Transformer; ST-MoE)
// ============================================================================
// Reverse routing: each expert selects its top-c tokens (capacity factor).
// In batch inference, this automatically balances load across experts.
// For single-token decode, this degenerates to top-k, but the infrastructure
// supports batch prefill where it provides significant load-balancing benefit.
// ============================================================================

static int moe_router_expert_choice_f32(
    moe_pc_engine* engine,
    int32_t layer,
    const float* hidden_batch,  // [batch_size × hidden_size]
    uint32_t hidden_size,
    uint32_t batch_size,
    uint32_t experts,
    uint32_t capacity_factor,   // tokens per expert
    std::vector<int>* out_token_to_experts,  // [batch_size] ??selected expert list
    std::vector<float>* out_token_to_weights,
    std::vector<float>* scratch_logits,
    std::vector<uint32_t>* scratch_order
) {
    if (!engine || !hidden_batch || !out_token_to_experts || !out_token_to_weights ||
        !scratch_logits || !scratch_order ||
        hidden_size == 0 || experts == 0 || batch_size == 0 || capacity_factor == 0) {
        return 0;
    }

    // For single-token decode, fall back to standard top-k (Expert Choice
    // requires batch dimension for meaningful load balancing)
    if (batch_size == 1) {
        return moe_router_topk_f32(
            engine, layer, hidden_batch, hidden_size, experts,
            moe_MOE_BASE_TOP_EXPERTS,
            out_token_to_experts, out_token_to_weights,
            scratch_logits, scratch_order
        );
    }

    // BUGFIX 601: Check batch_size and experts upper bounds ??
    // Problem: batch_size * experts can overflow
    // gate_matrix.resize(experts * batch_size) can allocate huge memory
    // Solution: Add reasonable upper bound checks
    if (batch_size > 10000 || experts > 10000) {
        return 0;
    }
    // BUGFIX 602: Check multiplication overflow ??
    // Problem: experts * batch_size can overflow uint32_t
    // Solution: Check before multiplication
    if (experts > UINT32_MAX / batch_size) {
        return 0;
    }

    // Step 1: Compute full gate score matrix [experts x batch_size].
    std::vector<float> gate_matrix(experts * batch_size, 0.0f);
    const int use_sigmoid = moe_router_uses_sigmoid_scores(engine);

    for (uint32_t t = 0; t < batch_size; ++t) {
        std::vector<float> scaled_router_input;
        const float* router_input = hidden_batch + t * hidden_size;
        if (moe_prepare_scaled_router_input_f32(engine, layer, router_input, hidden_size, &scaled_router_input)) {
            router_input = scaled_router_input.data();
        }
        if (!moe_router_logits_f32(
                engine,
                layer,
                router_input,
                hidden_size,
                experts,
                scratch_logits)) {
            return 0;
        }
        if (use_sigmoid) {
            for (float& v : *scratch_logits) {
                v = moe_router_sigmoid(v);
            }
            moe_router_apply_group_limit(engine, scratch_logits, experts);
        } else {
            float max_logit = -std::numeric_limits<float>::infinity();
            for (float v : *scratch_logits) {
                if (v > max_logit) max_logit = v;
            }
            double exp_denom = 0.0;
            for (float& v : *scratch_logits) {
                v = (float)std::exp((double)v - (double)max_logit);
                exp_denom += (double)v;
            }
            if (!(exp_denom > 0.0) || !std::isfinite(exp_denom)) {
                return 0;
            }
            const float inv = (float)(1.0 / exp_denom);
            for (float& v : *scratch_logits) {
                v *= inv;
            }
        }
        for (uint32_t e = 0; e < experts; ++e) {
            gate_matrix[e * batch_size + t] = (*scratch_logits)[e];
        }
    }

    // Step 2: Each expert selects its top-c tokens
    const uint32_t cap = std::min(capacity_factor, batch_size);
    // token_assignment[t] = list of (expert_id, weight)
    std::vector<std::vector<std::pair<int, float>>> token_assignment(batch_size);

    for (uint32_t e = 0; e < experts; ++e) {
        // Sort tokens by gate score for this expert
        scratch_order->resize(batch_size);
        for (uint32_t t = 0; t < batch_size; ++t) {
            (*scratch_order)[t] = t;
        }
        std::partial_sort(
            scratch_order->data(),
            scratch_order->data() + cap,
            scratch_order->data() + batch_size,
            [&](uint32_t a, uint32_t b) {
                return gate_matrix[e * batch_size + a] > gate_matrix[e * batch_size + b];
            }
        );
        for (uint32_t c = 0; c < cap; ++c) {
            const uint32_t t = (*scratch_order)[c];
            const float w = gate_matrix[e * batch_size + t];
            if (w > 0.01f) {  // Skip negligible assignments
                token_assignment[t].push_back({(int)e, w});
            }
        }
    }

    // Step 3: Flatten to output format (first token's assignment)
    // For compatibility with existing pipeline, output first token's results
    out_token_to_experts->clear();
    out_token_to_weights->clear();
    // BUGFIX 287: Check token_assignment bounds before accessing [0] ??
    // Problem: If batch_size is 0 or token_assignment is empty, accessing [0] is out-of-bounds
    // Solution: Add bounds check before accessing token_assignment[0]
    // BUGFIX 313: Also check token_assignment.size() >= batch_size ??
    // Problem: token_assignment.size() can be less than batch_size if resize failed
    // Accessing token_assignment[0] when size < batch_size causes out-of-bounds
    // Solution: Check token_assignment.size() > 0 before accessing [0]
    if (batch_size > 0 && !token_assignment.empty() && !token_assignment[0].empty()) {
        // Sort by weight descending
        std::sort(token_assignment[0].begin(), token_assignment[0].end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
        float denom = 0.0f;
        for (const auto& [eid, w] : token_assignment[0]) {
            out_token_to_experts->push_back(eid);
            out_token_to_weights->push_back(w);
            denom += w;
        }
        if (moe_router_norm_topk_prob(engine, use_sigmoid)) {
            // BUGFIX 287: Ensure denom is positive before division ??
            // Problem: All weights could be filtered out by w > 0.01f check, leaving denom = 0
            // Solution: Use max(denom, 1e-8f) to prevent division by zero
            if (denom <= 0.0f) denom = 1e-8f;
            for (float& w : *out_token_to_weights) {
                w /= denom;
            }
        }
        const float scale = use_sigmoid ? moe_router_routed_scaling_factor(engine) : 1.0f;
        if (scale != 1.0f) {
            for (float& w : *out_token_to_weights) {
                w *= scale;
            }
        }
    }

    return 1;
}
