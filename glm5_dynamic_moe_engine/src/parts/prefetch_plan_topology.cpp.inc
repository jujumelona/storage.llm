static void prefetch_plan_topology_candidates(
    glm5_pc_engine* engine,
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
    out->reserve((size_t)count * (size_t)engine->io_config.topology_max_edges_per_source);
    std::shared_lock<std::shared_mutex> lock(engine->topology_mutex);
    for (uint32_t i = 0; i < count; ++i) {
        const uint64_t src_key = topology_key(layer, experts[i]);
        if (src_key == UINT64_MAX) {
            continue;
        }
        const auto src = engine->topology_edges.find(src_key);
        if (src == engine->topology_edges.end()) {
            continue;
        }
        for (const auto& dst : src->second) {
            out->emplace_back(dst.second, topology_key_layer(dst.first), topology_key_expert(dst.first));
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
    glm5_pc_engine* engine,
    std::vector<glm5_expert_prefetch_item_t>& items,
    int layer,
    const int* experts,
    uint32_t count
) {
    if (!engine) {
        return;
    }
    std::lock_guard<std::mutex> scratch_lock(engine->topology_candidates_mutex);
    auto& candidates = engine->topology_candidates_scratch;
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
            GLM5_TIER_RAM,
            GLM5_PREFETCH_REASON_TOPOLOGY,
            50.0f + (float)std::get<0>(candidate)
        );
        if (items.size() != before) {
            ++added;
        }
    }
}
