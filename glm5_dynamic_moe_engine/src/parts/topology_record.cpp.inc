void glm5_pc_engine_apply_topology_updates(glm5_pc_engine_t* engine) {
    if (!engine) return;
    std::vector<std::pair<uint64_t, uint64_t>> updates;
    {
        std::lock_guard<std::mutex> qlock(engine->topology_queue_mutex);
        updates.swap(engine->topology_update_queue);
    }
    if (updates.empty()) return;
    std::unique_lock<std::shared_mutex> lock(engine->topology_mutex);
    for (const auto& u : updates) {
        const uint64_t src_key = u.first;
        const uint64_t dst_key = u.second;
        auto& edges = engine->topology_edges[src_key];
        edges[dst_key] += 1;
        if (edges.size() > engine->io_config.topology_max_edges_per_source) {
            uint64_t min_count = UINT64_MAX;
            uint64_t min_key = dst_key;
            for (const auto& e : edges) {
                if (e.second < min_count) {
                    min_count = e.second;
                    min_key = e.first;
                }
            }
            edges.erase(min_key);
        }
    }
}

int glm5_pc_engine_record_expert_transition(
    glm5_pc_engine_t* engine,
    int layer,
    int expert,
    int next_layer,
    int next_expert
) {
    if (!engine || layer < 0 || expert < 0 || next_layer < 0 || next_expert < 0) {
        return 0;
    }
    const uint64_t src_key = topology_key(layer, expert);
    const uint64_t dst_key = topology_key(next_layer, next_expert);
    if (src_key == UINT64_MAX || dst_key == UINT64_MAX) {
        return 0;
    }
    {
        std::lock_guard<std::mutex> lock(engine->topology_queue_mutex);
        // Fix: Add size limit to prevent unbounded queue growth
        if (engine->topology_update_queue.size() < 65536) {
            engine->topology_update_queue.emplace_back(src_key, dst_key);
        }
    }
    engine->io_atomic.topology_updates.fetch_add(1);
    return 1;
}
