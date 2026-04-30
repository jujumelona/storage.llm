int moe_pc_engine_load_topology(moe_pc_engine_t* engine, const char* path) {
    if (!engine) {
        return 0;
    }
    const std::string chosen = path && path[0] ? path : engine->topology_path;
    if (chosen.empty()) {
        return 0;
    }

    std::ifstream input;
    open_ifstream_path(input, chosen);
    if (!input) {
        moe_pc_engine_set_topology_path(engine, chosen.c_str());
        return 1;
    }

    std::unordered_map<uint64_t, std::unordered_map<uint64_t, uint64_t>> loaded;
    std::string line;
    if (!std::getline(input, line)) {
        moe_pc_engine_set_topology_path(engine, chosen.c_str());
        return 1;
    }
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> row = split_csv_line(line);
        if (row.size() < 5) {
            continue;
        }
        const int src_layer = std::atoi(row[0].c_str());
        const int src_expert = std::atoi(row[1].c_str());
        const int dst_layer = std::atoi(row[2].c_str());
        const int dst_expert = std::atoi(row[3].c_str());
        const uint64_t count = parse_u64(row[4]);
        if (src_layer < 0 || src_expert < 0 || dst_layer < 0 || dst_expert < 0 || count == 0) {
            continue;
        }
        const uint64_t src_key = topology_key(src_layer, src_expert);
        const uint64_t dst_key = topology_key(dst_layer, dst_expert);
        if (src_key == UINT64_MAX || dst_key == UINT64_MAX) {
            continue;
        }
        loaded[src_key][dst_key] += count;
    }
    for (auto& src : loaded) {
        while (src.second.size() > engine->io_config.topology_max_edges_per_source) {
            uint64_t min_count = UINT64_MAX;
            uint64_t min_key = 0;
            for (const auto& e : src.second) {
                if (e.second < min_count) {
                    min_count = e.second;
                    min_key = e.first;
                }
            }
            src.second.erase(min_key);
        }
    }

    {
        std::unique_lock<std::shared_mutex> lock(engine->topology_mutex);
        engine->topology_path = chosen;
        engine->topology_edges = std::move(loaded);
    }
    return 1;
}
