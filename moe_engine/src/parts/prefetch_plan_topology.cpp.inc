static void prefetch_plan_topology_candidates(
    moe_pc_engine* engine,
    int layer,
    const int* experts,
    uint32_t count,
    std::vector<std::tuple<uint64_t, int, int>>* out
) {
    if (!out) {
        return;
    }
    out->clear();
    if (!engine || !engine->io_config.enable_topology_prefetch || !experts) {
        return;
    }
    // BUGFIX 552: Check for overflow in reserve calculation ★
    // Problem: count * topology_max_edges_per_source can overflow size_t
    // If count = 1000 and max_edges = 10000000, multiplication overflows
    // Solution: Check for overflow before reserve
    const size_t max_edges = (size_t)engine->io_config.topology_max_edges_per_source;
    if (count > 0 && max_edges > SIZE_MAX / count) {
        // Overflow would occur, use smaller reserve
        out->reserve(10000);
    } else {
        out->reserve((size_t)count * max_edges);
    }
    std::unordered_map<int, std::unordered_map<int, uint64_t>> aggregated_dst;

    for (uint32_t i = 0; i < count; ++i) {
        const uint64_t src_key = topology_key(layer, experts[i]);
        if (src_key == UINT64_MAX) {
            continue;
        }
        std::shared_lock<std::shared_mutex> lock(engine->topology_mutex);
        const auto src = engine->topology_edges.find(src_key);
        if (src == engine->topology_edges.end()) {
            continue;
        }
        for (const auto& dst : src->second) {
            aggregated_dst[topology_key_layer(dst.first)][topology_key_expert(dst.first)] += dst.second;
        }
    }

    // ============================================================================
    // Hierarchical Expert Grouping (HMoE / Lory 2024)
    // ============================================================================
    // Instead of 1:1 topological edges, we expand the destination experts to
    // their full semantic domain (clusters). If an expert in domain D is a
    // topological neighbor, all experts in domain D are prefetched.
    // ============================================================================
    for (const auto& layer_pair : aggregated_dst) {
        const int dst_layer = layer_pair.first;
        std::vector<int> explicit_experts;
        for (const auto& exp_pair : layer_pair.second) {
            explicit_experts.push_back(exp_pair.first);
        }

        std::vector<int> expanded_domain;
        moe_expand_merged_experts(engine, dst_layer, explicit_experts, expanded_domain);

        for (int e : expanded_domain) {
            uint64_t freq = 0;
            auto it = layer_pair.second.find(e);
            if (it != layer_pair.second.end()) {
                freq = it->second;
            } else {
                // Domain member that wasn't an explicit topological neighbor.
                // Give it a baseline priority derived from the domain's average or a small constant
                freq = 1;
            }
            out->emplace_back(freq, dst_layer, e);
        }
    }
}

static void prefetch_plan_sort_topology(std::vector<std::tuple<uint64_t, int, int>>& candidates) {
    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        if (std::get<0>(a) != std::get<0>(b)) {
            return std::get<0>(a) > std::get<0>(b);
        }
        if (std::get<1>(a) != std::get<1>(b)) {
            return std::get<1>(a) < std::get<1>(b);
        }
        return std::get<2>(a) < std::get<2>(b);
    });
}

static void prefetch_plan_add_topology(
    moe_pc_engine* engine,
    std::vector<moe_expert_prefetch_item_t>& items,
    int layer,
    const int* experts,
    uint32_t count
) {
    if (!engine) {
        return;
    }
    static thread_local std::vector<std::tuple<uint64_t, int, int>> candidates;
    prefetch_plan_topology_candidates(engine, layer, experts, count, &candidates);
    prefetch_plan_sort_topology(candidates);
    uint32_t added = 0;
    for (const auto& candidate : candidates) {
        if (added >= engine->io_config.topology_max_candidates) {
            break;
        }
        const size_t before = items.size();
        prefetch_plan_add_item(
            engine,
            items,
            std::get<1>(candidate),
            std::get<2>(candidate),
            moe_TIER_RAM,
            moe_PREFETCH_REASON_TOPOLOGY,
            50.0f + (float)std::get<0>(candidate)
        );
        if (items.size() != before) {
            ++added;
        }
    }
}
