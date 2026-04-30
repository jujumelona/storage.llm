static int moe_router_topk_f32(
    moe_pc_engine* engine,
    int32_t layer,
    const float* hidden,
    uint32_t hidden_size,
    uint32_t experts,
    uint32_t top_k,
    std::vector<int>* out_experts,
    std::vector<float>* out_weights,
    std::vector<float>* scratch_logits,
    std::vector<uint32_t>* scratch_order
) {
    if (!engine || !hidden || !out_experts || !out_weights || !scratch_logits || !scratch_order ||
        hidden_size == 0 || experts == 0 || top_k == 0) {
        return 0;
    }
    const uint64_t router_bytes = (uint64_t)experts * (uint64_t)hidden_size * 2ull;
    const moe_storage_raw_span_spec_t* span =
        find_raw_role_span_bytes(layer, moe_RAW_ROLE_ROUTER, router_bytes, 0);
    if (!span) {
        return 0;
    }
    scratch_logits->assign(experts, 0.0f);
    if (!moe_raw_linear_span_f32(
            engine,
            span,
            0,
            experts,
            hidden_size,
            hidden,
            hidden_size,
            scratch_logits->data(),
            experts)) {
        return 0;
    }

    auto sigmoid = [](float x) {
        if (x >= 0.0f) {
            const float z = std::exp(-x);
            return 1.0f / (1.0f + z);
        }
        const float z = std::exp(x);
        return z / (1.0f + z);
    };
    for (float& v : *scratch_logits) {
        v = sigmoid(v);
    }

    top_k = std::min(top_k, experts);
    scratch_order->resize(experts);
    uint32_t* order = scratch_order->data();
    for (uint32_t i = 0; i < experts; ++i) {
        order[i] = i;
    }
    // Fix 9: nth_element O(N) avg + sort O(K log K) beats partial_sort O(N log K)
    auto cmp = [&](uint32_t a, uint32_t b) {
        return (*scratch_logits)[a] > (*scratch_logits)[b];
    };
    if (top_k < experts) {
        std::nth_element(order, order + top_k, order + experts, cmp);
    }
    std::sort(order, order + top_k, cmp);

    float denom = 0.0f;
    out_experts->clear();
    out_weights->clear();
    for (uint32_t i = 0; i < top_k; ++i) {
        const float w = (*scratch_logits)[order[i]];
        out_experts->push_back((int)order[i]);
        out_weights->push_back(w);
        denom += w;
    }
    // Bug 10: Clamp denom to prevent silent failure on underflow
    // When all sigmoid outputs are near zero, denom can underflow to 0.0f
    if (denom <= 0.0f) {
        denom = 1e-8f;  // Clamp to small positive value, maintain weight ratios
    }
    constexpr float routed_scaling_factor = 2.5f;
    for (float& w : *out_weights) {
        w = (w / denom) * routed_scaling_factor;
    }
    return 1;
}
