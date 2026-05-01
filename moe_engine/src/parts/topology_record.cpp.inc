void moe_pc_engine_apply_topology_updates(moe_pc_engine_t* engine) {
    if (!engine) return;
    std::vector<std::pair<uint64_t, uint64_t>> updates;
    {
        std::lock_guard<std::mutex> qlock(engine->topology_queue_mutex);
        updates.swap(engine->topology_update_queue);
    }
    if (updates.empty()) return;
    
    // BUGFIX 174: Limit total topology_edges size to prevent unbounded memory growth
    constexpr size_t MAX_TOPOLOGY_SOURCES = 32768;
    
    std::unique_lock<std::shared_mutex> lock(engine->topology_mutex);
    for (const auto& u : updates) {
        const uint64_t src_key = u.first;
        const uint64_t dst_key = u.second;
        
        // Check if we're about to add a new source and would exceed limit
        if (engine->topology_edges.find(src_key) == engine->topology_edges.end()) {
            if (engine->topology_edges.size() >= MAX_TOPOLOGY_SOURCES) {
                // Remove oldest/least-used source to make room
                // Find source with smallest total edge count
                uint64_t min_total = UINT64_MAX;
                uint64_t min_src = src_key;
                for (const auto& [src, edges] : engine->topology_edges) {
                    uint64_t total = 0;
                    for (const auto& [dst, count] : edges) {
                        total += count;
                    }
                    if (total < min_total) {
                        min_total = total;
                        min_src = src;
                    }
                }
                engine->topology_edges.erase(min_src);
            }
        }
        
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

int moe_pc_engine_record_expert_transition(
    moe_pc_engine_t* engine,
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
    // BUGFIX 151: Add memory_order_relaxed for statistics counter
    engine->io_atomic.topology_updates.fetch_add(1, std::memory_order_relaxed);
    return 1;
}
