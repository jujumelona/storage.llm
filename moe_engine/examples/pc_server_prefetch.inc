struct grouped_request_tokens {
    std::vector<int> experts;
    uint32_t active_tokens = 0;
};

// Semantic validation for prefetch requests (Bug B2)
static bool validate_prefetch_request(
    moe_pc_engine_t* engine,
    int current_layer,
    std::vector<int>& selected_experts,
    std::vector<int>& next_experts,
    std::string* err
) {
    // Get model shape
    const auto shape = moe_pc_Moe1_model_shape();
    int layer_count = static_cast<int>(shape.num_hidden_layers);
    if (layer_count <= 0 && engine) {
        moe_forward_status_t status{};
        if (moe_pc_engine_get_forward_status(engine, &status) &&
            status.dynamic_num_hidden_layers > 0) {
            layer_count = static_cast<int>(status.dynamic_num_hidden_layers);
        }
    }

    // Validate layer range
    if (current_layer < 0 || current_layer >= layer_count) {
        if (err) *err = "current_layer out of range [0, " + std::to_string(layer_count) + ")";
        return false;
    }

    // Get expert count for this layer
    const int expert_count = static_cast<int>(shape.experts_per_moe_layer);

    auto normalize = [&](std::vector<int>& ids, const char* field) -> bool {
        // Sort and deduplicate
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());

        // Check array length limit (policy: max 64 experts)
        if (ids.size() > 64) {
            if (err) *err = std::string(field) + " too long (max 64)";
            return false;
        }

        // Validate each expert ID
        for (int id : ids) {
            if (id < 0 || (expert_count > 0 && id >= expert_count)) {
                if (err) *err = std::string(field) + " contains invalid expert id " +
                                std::to_string(id) + " (valid range: [0, " +
                                std::to_string(expert_count) + "))";
                return false;
            }
        }
        return true;
    };

    return normalize(selected_experts, "selected_experts") &&
           normalize(next_experts, "next_experts");
}

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
    size_t key_colon = 0;
    const bool has_current_layer = storagellm::find_top_level_key_colon(body, "current_layer", key_colon);
    const bool has_selected_experts = storagellm::find_top_level_key_colon(body, "selected_experts", key_colon);
    const bool has_next_experts = storagellm::find_top_level_key_colon(body, "next_experts", key_colon);
    if (!has_current_layer && !has_selected_experts && !has_next_experts) {
        return;
    }

    int layer = static_cast<int>(moe_pc_Moe1_model_shape().first_moe_layer);
    if (layer < 0) layer = 0;
    storagellm::json_read_int(body, "current_layer", &layer);
    std::vector<int> selected = storagellm::json_read_int_array(body, "selected_experts");
    std::vector<int> next = storagellm::json_read_int_array(body, "next_experts");
    if (selected.empty() && next.empty()) {
        return;
    }

    // Validate request (Bug B2: semantic validation)
    std::string validation_error;
    if (!validate_prefetch_request(engine, layer, selected, next, &validation_error)) {
        fprintf(stderr, "[storageLLM ERROR] Invalid prefetch request: %s\n", validation_error.c_str());
        return;  // Reject invalid request
    }

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
