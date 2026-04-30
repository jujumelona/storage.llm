int moe_pc_engine_save_topology(moe_pc_engine_t* engine, const char* path) {
    if (!engine) {
        return 0;
    }
    const std::string chosen = path && path[0] ? path : engine->topology_path;
    if (chosen.empty()) {
        return 0;
    }

    std::vector<std::tuple<int, int, int, int, uint64_t>> rows;
    {
        std::shared_lock<std::shared_mutex> lock(engine->topology_mutex);
        for (const auto& src : engine->topology_edges) {
            const int src_layer = topology_key_layer(src.first);
            const int src_expert = topology_key_expert(src.first);
            for (const auto& dst : src.second) {
                const int dst_layer = topology_key_layer(dst.first);
                const int dst_expert = topology_key_expert(dst.first);
                rows.emplace_back(src_layer, src_expert, dst_layer, dst_expert, dst.second);
            }
        }
    }
    {
        std::unique_lock<std::shared_mutex> lock(engine->topology_mutex);
        engine->topology_path = chosen;
    }

    std::sort(rows.begin(), rows.end());
    std::ofstream output;
    open_ofstream_path(output, chosen, std::ios::trunc);
    if (!output) {
        return 0;
    }
    output << "src_layer,src_expert,dst_layer,dst_expert,count\n";
    for (const auto& row : rows) {
        output << std::get<0>(row) << ','
               << std::get<1>(row) << ','
               << std::get<2>(row) << ','
               << std::get<3>(row) << ','
               << std::get<4>(row) << '\n';
    }
    return 1;
}
