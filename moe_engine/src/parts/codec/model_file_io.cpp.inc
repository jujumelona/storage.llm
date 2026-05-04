static int moe_read_model_root_text_file(
    const std::string& root,
    const char* file_name,
    uint64_t max_bytes,
    std::string* out
) {
    if (out) {
        out->clear();
    }
    if (!out || root.empty() || !file_name || !file_name[0]) {
        return 0;
    }
    const std::string path = storagellm::path_join(root, file_name);
    uint64_t bytes = 0;
    if (!file_size_bytes(path, &bytes) || bytes == 0 || bytes > max_bytes) {
        return 0;
    }
    std::ifstream input;
    open_ifstream_path(input, path, std::ios::in | std::ios::binary);
    if (!input) {
        return 0;
    }
    std::string data((size_t)bytes, '\0');
    input.read(&data[0], (std::streamsize)data.size());
    if ((uint64_t)input.gcount() != bytes) {
        return 0;
    }
    *out = std::move(data);
    return 1;
}

static int moe_read_model_config_json(
    const std::string& root,
    std::string* out
) {
    return moe_read_model_root_text_file(root, "config.json", 64ull * 1024ull * 1024ull, out);
}

static int moe_runtime_sidecar_name_interesting(const std::string& path) {
    const std::string lower = moe_lower_ascii_copy(path);
    if (lower.empty()) {
        return 0;
    }
    if (lower == "runtime_assets_manifest.json" ||
        lower == "storagellm_performance_metadata_manifest.json" ||
        lower == "generation_config.json" ||
        lower == "tokenizer_config.json" ||
        lower == "processor_config.json" ||
        lower == "preprocessor_config.json" ||
        lower == "image_processor_config.json" ||
        lower == "feature_extractor.json" ||
        lower == "audio_config.json" ||
        lower == "video_preprocessor_config.json") {
        return 1;
    }
    if (!moe_ends_with_ci(lower, ".json") &&
        !moe_ends_with_ci(lower, ".jsonl")) {
        return 0;
    }
    return lower.find("metadata") != std::string::npos ||
        lower.find("manifest") != std::string::npos ||
        lower.find("runtime") != std::string::npos ||
        lower.find("contract") != std::string::npos ||
        lower.find("graph") != std::string::npos ||
        lower.find("priority") != std::string::npos ||
        lower.find("prefetch") != std::string::npos ||
        lower.find("residency") != std::string::npos ||
        lower.find("qkv") != std::string::npos ||
        lower.find("offload") != std::string::npos ||
        lower.find("policy") != std::string::npos ||
        lower.find("scheduler") != std::string::npos ||
        lower.find("topology") != std::string::npos ||
        lower.find("profile") != std::string::npos ||
        lower.find("trace") != std::string::npos ||
        lower.find("stats") != std::string::npos ||
        lower.find("diagnostic") != std::string::npos ||
        lower.find("verify") != std::string::npos ||
        lower.find("tensor") != std::string::npos ||
        lower.find("index") != std::string::npos;
}

static std::string moe_json_escape_runtime_path(const std::string& value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (char c : value) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out.push_back(c); break;
        }
    }
    return out;
}

static void moe_collect_runtime_sidecar_files(
    const std::string& root,
    const std::string& rel_dir,
    int depth_left,
    std::vector<std::pair<std::string, std::string>>* out
) {
    if (!out || root.empty() || out->size() >= 512 || depth_left < 0) {
        return;
    }
    const std::string abs_dir = rel_dir.empty() ? root : storagellm::path_join(root, rel_dir);
    if (!moe_path_is_directory_local(abs_dir)) {
        return;
    }
    moe_dir_iterator_t* it = moe_open_directory(abs_dir);
    if (!it) {
        return;
    }
    std::string name;
    bool is_dir = false;
    while (moe_next_entry(it, &name, &is_dir)) {
        if (moe_skip_scan_dir_name(name)) {
            continue;
        }
        const std::string rel = rel_dir.empty() ? name : storagellm::path_join(rel_dir, name);
        const std::string abs = storagellm::path_join(root, rel);
        if (is_dir) {
            moe_collect_runtime_sidecar_files(root, rel, depth_left - 1, out);
        } else if (moe_runtime_sidecar_name_interesting(rel)) {
            out->push_back(std::make_pair(rel, abs));
        }
        if (out->size() >= 512) {
            break;
        }
    }
    moe_close_directory(it);
}

static void moe_append_runtime_sidecar_blob(
    moe_pc_engine* engine,
    const std::string& rel_path,
    const std::string& data
) {
    if (!engine || rel_path.empty() || data.empty()) {
        return;
    }
    if (!engine->offload_gguf_metadata_json.empty()) {
        engine->offload_gguf_metadata_json.push_back('\n');
    }
    engine->offload_gguf_metadata_json += "{\"__storagellm_runtime_sidecar_path__\":\"";
    engine->offload_gguf_metadata_json += moe_json_escape_runtime_path(rel_path);
    engine->offload_gguf_metadata_json += "\"}\n";
    engine->offload_gguf_metadata_json += data;

    const std::string lower = moe_lower_ascii_copy(rel_path);
    const bool graph_hint =
        lower.find("graph") != std::string::npos ||
        lower.find("priority") != std::string::npos ||
        lower.find("prefetch") != std::string::npos ||
        lower.find("residency") != std::string::npos ||
        lower.find("policy") != std::string::npos ||
        data.find("\"graph_ir\"") != std::string::npos ||
        data.find("\"priority_tables\"") != std::string::npos ||
        data.find("\"moe_offload_policy\"") != std::string::npos;
    if (graph_hint && engine->offload_graph_ir_json.size() < (4u << 20)) {
        engine->offload_graph_ir_json.push_back('\n');
        engine->offload_graph_ir_json += data.substr(0, std::min<size_t>(data.size(), 1u << 20));
    }
}

static void moe_append_model_root_runtime_sidecars(
    moe_pc_engine* engine,
    const std::string& root
) {
    if (!engine || root.empty() ||
        engine->offload_gguf_metadata_json.find("__storagellm_runtime_sidecar_path__") != std::string::npos) {
        return;
    }
    static const char* kRootSidecars[] = {
        "runtime_assets_manifest.json",
        "storagellm_performance_metadata_manifest.json",
        "generation_config.json",
        "tokenizer_config.json",
        "processor_config.json",
        "preprocessor_config.json",
        "image_processor_config.json",
        "feature_extractor.json",
        "audio_config.json",
        "video_preprocessor_config.json",
    };
    std::vector<std::pair<std::string, std::string>> sidecars;
    for (const char* rel : kRootSidecars) {
        const std::string path = storagellm::path_join(root, rel);
        uint64_t bytes = 0;
        if (file_size_bytes(path, &bytes) && bytes > 0) {
            sidecars.push_back(std::make_pair(std::string(rel), path));
        }
    }
    moe_collect_runtime_sidecar_files(root, "metadata", 4, &sidecars);
    moe_collect_runtime_sidecar_files(root, "verify", 3, &sidecars);
    moe_collect_runtime_sidecar_files(root, "tokenizer", 2, &sidecars);
    std::sort(sidecars.begin(), sidecars.end(),
        [](const std::pair<std::string, std::string>& a, const std::pair<std::string, std::string>& b) {
            return a.first < b.first;
        });
    sidecars.erase(std::unique(sidecars.begin(), sidecars.end(),
        [](const std::pair<std::string, std::string>& a, const std::pair<std::string, std::string>& b) {
            return a.first == b.first;
        }), sidecars.end());

    uint64_t total_loaded = 0;
    uint32_t loaded_count = 0;
    for (const auto& item : sidecars) {
        if (!moe_runtime_sidecar_name_interesting(item.first)) {
            continue;
        }
        uint64_t bytes = 0;
        if (!file_size_bytes(item.second, &bytes) || bytes == 0 ||
            bytes > (16ull << 20) || total_loaded + bytes > (64ull << 20)) {
            continue;
        }
        std::string data;
        if (!moe_read_model_root_text_file(root, item.first.c_str(), 16ull << 20, &data) || data.empty()) {
            continue;
        }
        moe_append_runtime_sidecar_blob(engine, item.first, data);
        total_loaded += bytes;
        ++loaded_count;
    }
    if (loaded_count) {
        if (!engine->offload_gguf_metadata_json.empty()) {
            engine->offload_gguf_metadata_json.push_back('\n');
        }
        engine->offload_gguf_metadata_json += "{\"__storagellm_runtime_sidecars_loaded__\":";
        engine->offload_gguf_metadata_json += std::to_string(loaded_count);
        engine->offload_gguf_metadata_json += ",\"__storagellm_runtime_sidecar_bytes__\":";
        engine->offload_gguf_metadata_json += std::to_string(total_loaded);
        engine->offload_gguf_metadata_json += "}";
        std::fprintf(stderr,
            "[storagellm] runtime sidecars loaded: files=%u bytes=%llu\n",
            loaded_count,
            (unsigned long long)total_loaded);
        std::fflush(stderr);
    }
}

static void moe_append_model_config_metadata(
    moe_pc_engine* engine,
    const std::string& root
) {
    if (!engine || root.empty() ||
        engine->offload_gguf_metadata_json.find("__storagellm_external_config_json__") != std::string::npos) {
        return;
    }
    std::string config_json;
    if (!moe_read_model_config_json(root, &config_json) || config_json.empty()) {
        return;
    }
    if (!engine->offload_gguf_metadata_json.empty()) {
        engine->offload_gguf_metadata_json.push_back('\n');
    }
    engine->offload_gguf_metadata_json += "{\"__storagellm_external_config_json__\":true}\n";
    engine->offload_gguf_metadata_json += config_json;
}

int moe_pc_engine_set_model_root(
    moe_pc_engine_t* engine,
    const char* model_root
) {
    if (!engine) {
        return 0;
    }
    // BUGFIX 52: Issue 23 - Stop IO workers before shard cleanup ?�★??    // Problem: wait_io only waits for queue to drain, but io_workers continue running.
    // After shard_mmaps.clear(), io_worker can call get_or_map_tensor_shard with stale
    // model_root, causing mmap of wrong files or crashes.
    // Solution: stop_io_workers() terminates all workers, then clear shards, then restart.
    // This ensures no worker accesses shards during cleanup.
    // BUGFIX 55: Hold io_start_mutex to serialize with ensure_io_ready
    {
        std::lock_guard<std::mutex> io_lock(engine->io_start_mutex);
        stop_io_workers_locked(engine);
    }

    {
        std::unique_lock<std::shared_mutex> lock(engine->shard_mutex);
        for (auto& item : engine->shard_mmaps) {
            unload_mapped_shard(item.second);
        }
        engine->shard_mmaps.clear();
        engine->shard_mmaps_by_id.clear();
        engine->shard_mmap_cvs.clear();
        engine->locked_mmaps.clear();
    }
    reset_source_mmap_slots(engine);
    {
        std::lock_guard<std::mutex> lock(engine->raw_file_size_cache_mutex);
        engine->raw_file_size_cache.clear();
    }
    {
        std::unique_lock<std::shared_mutex> lock(engine->raw_tensor_cache_mutex);
        engine->raw_tensor_cache.clear();
    }
    moe_engine_reset_qkv_cache(engine);
    moe_engine_clear_offload_gguf_contract(engine);
    engine->gguf_common_tensors.clear();
    engine->gguf_common_tensor_slots.clear();
    engine->model_root = model_root ? model_root : "";
    moe_model_root_check_t check{};
    int valid = 0;
    if (!engine->model_root.empty()) {
        if (moe_engine_load_offload_gguf_contract(engine, engine->model_root, &check)) {
            valid = check.valid;
            moe_append_model_config_metadata(engine, engine->model_root);
            moe_append_model_root_runtime_sidecars(engine, engine->model_root);
        } else {
            valid = moe_storage_validate_model_root(engine->model_root.c_str(), &check) && check.valid;
        }
    }
    engine->model_root_valid.store(valid ? 1 : 0, std::memory_order_relaxed);
    engine->tokenizer_valid.store(-1, std::memory_order_relaxed);
    engine->attention_projector_valid.store(-1, std::memory_order_relaxed);  // Fix 72: invalidate cache

    // Restart IO workers with new model_root
    {
        std::lock_guard<std::mutex> io_lock(engine->io_start_mutex);
        start_io_workers(engine);
    }

    return 1;
}
