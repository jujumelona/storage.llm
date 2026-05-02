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
        // BUGFIX 303: Check for integer overflow in atoi ★
        // Problem: atoi returns 0 on overflow, can't distinguish from valid 0
        // Negative values from overflow can pass validation
        // Solution: Use strtol with error checking
        char* end = nullptr;
        const long src_layer_l = std::strtol(row[0].c_str(), &end, 10);
        // BUGFIX 545: Check end pointer is not null before dereferencing ★
        // Problem: strtol can set end to null on failure
        // Solution: Check end != nullptr before comparing
        if (!end || end == row[0].c_str() || src_layer_l < 0 || src_layer_l > INT_MAX) {
            continue;
        }
        const long src_expert_l = std::strtol(row[1].c_str(), &end, 10);
        if (!end || end == row[1].c_str() || src_expert_l < 0 || src_expert_l > INT_MAX) {
            continue;
        }
        const long dst_layer_l = std::strtol(row[2].c_str(), &end, 10);
        if (!end || end == row[2].c_str() || dst_layer_l < 0 || dst_layer_l > INT_MAX) {
            continue;
        }
        const long dst_expert_l = std::strtol(row[3].c_str(), &end, 10);
        if (!end || end == row[3].c_str() || dst_expert_l < 0 || dst_expert_l > INT_MAX) {
            continue;
        }
        const int src_layer = (int)src_layer_l;
        const int src_expert = (int)src_expert_l;
        const int dst_layer = (int)dst_layer_l;
        const int dst_expert = (int)dst_expert_l;
        const uint64_t count = parse_u64(row[4]);
        if (count == 0) {
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
        // BUGFIX 304: Check for empty src.second before accessing ★
        // Problem: If src.second is empty, min_key remains 0 (uninitialized)
        // erase(0) can remove wrong entry or cause undefined behavior
        // Solution: Skip if empty
        if (src.second.empty()) {
            continue;
        }
        while (src.second.size() > engine->io_config.topology_max_edges_per_source) {
            uint64_t min_count = UINT64_MAX;
            uint64_t min_key = 0;
            for (const auto& e : src.second) {
                if (e.second < min_count) {
                    min_count = e.second;
                    min_key = e.first;
                }
            }
            // BUGFIX 305: Check if min_key was actually found ★
            // Problem: If all entries have count == UINT64_MAX, min_key stays 0
            // erase(0) removes wrong entry
            // Solution: Check min_count was updated
            if (min_count == UINT64_MAX) {
                break; // No valid minimum found
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
