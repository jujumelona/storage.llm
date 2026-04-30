struct grouped_request_tokens {
    std::vector<int> experts;
    uint32_t active_tokens = 0;
};

static grouped_request_tokens group_request_tokens(int layer, const std::vector<int>& experts) {
    grouped_request_tokens grouped{};
    if (layer < 0 || experts.empty()) {
        return grouped;
    }
    std::vector<moe_token_expert_assignment_t> assignments;
    assignments.reserve(experts.size());
    for (uint32_t i = 0; i < experts.size(); ++i) {
        if (experts[i] < 0) {
            continue;
        }
        assignments.push_back({i, static_cast<uint32_t>(layer), static_cast<uint32_t>(experts[i]), 1.0f});
    }
    grouped.active_tokens = static_cast<uint32_t>(assignments.size());
    if (assignments.empty()) {
        return grouped;
    }
    std::vector<moe_token_expert_assignment_t> sorted(assignments.size());
    std::vector<moe_expert_batch_t> batches(assignments.size());
    uint32_t batch_count = 0;
    if (!moe_pc_group_tokens_by_expert(assignments.data(), static_cast<uint32_t>(assignments.size()),
            sorted.data(), batches.data(), static_cast<uint32_t>(batches.size()), &batch_count)) {
        return grouped;
    }
    grouped.experts.reserve(batch_count);
    for (uint32_t i = 0; i < batch_count; ++i) {
        if (batches[i].layer == static_cast<uint32_t>(layer)) {
            grouped.experts.push_back(static_cast<int>(batches[i].expert));
        }
    }
    return grouped;
}

static void server_orchestrate_request_prefetch(
    moe_pc_engine_t* engine,
    const server_options& opts,
    const std::string& body
) {
    int layer = static_cast<int>(moe_pc_Moe1_model_shape().first_moe_layer);
    storagellm::json_read_int(body, "current_layer", &layer);
    std::vector<int> selected = storagellm::json_read_int_array(body, "selected_experts");
    std::vector<int> next = storagellm::json_read_int_array(body, "next_experts");
    grouped_request_tokens grouped = group_request_tokens(layer, selected);
    if (grouped.experts.empty()) return;
    moe_pc_engine_prepare_moe_activation(engine, layer);
    moe_runtime_orchestrator_state_t state{};
    state.policy = opts.policy;
    state.cpu_load_pct = 50.0f;
    state.gpu_load_pct = 50.0f;
    state.active_tokens = grouped.active_tokens;
    moe_runtime_orchestrator_plan_t plan{};
    moe_expert_prefetch_item_t items[64];
    uint32_t item_count = 0;
    moe_pc_engine_orchestrate_moe_step(engine, layer, grouped.experts.data(),
        static_cast<uint32_t>(grouped.experts.size()), next.data(),
        static_cast<uint32_t>(next.size()), &state, &plan, items, 64, &item_count);
}
