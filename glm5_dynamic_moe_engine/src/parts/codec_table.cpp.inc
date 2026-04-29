int glm5_pc_engine_set_model_root(
    glm5_pc_engine_t* engine,
    const char* model_root
) {
    if (!engine) {
        return 0;
    }
    glm5_pc_engine_wait_io(engine);
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
    engine->model_root = model_root ? model_root : "";
    glm5_model_root_check_t check{};
    const int valid = !engine->model_root.empty() &&
        glm5_storage_validate_model_root(engine->model_root.c_str(), &check) &&
        check.valid;
    engine->model_root_valid.store(valid ? 1 : 0, std::memory_order_relaxed);
    engine->tokenizer_valid.store(-1, std::memory_order_relaxed);
    engine->attention_projector_valid.store(-1, std::memory_order_relaxed);  // Fix 72: invalidate cache
    return 1;
}

int glm5_pc_engine_load_codec_table(
    glm5_pc_engine_t* engine,
    const char* csv_path,
    const char* model_root,
    const char* scale4_path
) {
    if (!engine || !csv_path) {
        return 0;
    }

    glm5_pc_engine_wait_io(engine);
    device_free_pool(engine, GLM5_DEVICE_POOL_FIXED);
    device_free_pool(engine, GLM5_DEVICE_POOL_EXPERT);
    device_free_pool(engine, GLM5_DEVICE_POOL_ACTIVATION);
    device_free_pool(engine, GLM5_DEVICE_POOL_STREAM);
    engine->expert_states.assign(expert_slot_count(), glm5_block_state{});
    engine->ram_used_bytes.store(0);
    engine->db_used_bytes.store(0);
    apply_common_vram_reservation(engine);
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
    engine->tensors.clear();
    engine->tensor_paths.clear();
    engine->max_tensor_stream_bytes = 0;
    engine->common_raw_prefetched_bytes.store(0);
    engine->common_raw_prefetched_spans.store(0);
    engine->tensors.reserve(tensor_record_capacity());
    tensor_slots_reset(engine);
    engine->model_root = model_root ? model_root : "";
    glm5_model_root_check_t root_check{};
    const int root_valid = !engine->model_root.empty() &&
        glm5_storage_validate_model_root(engine->model_root.c_str(), &root_check) &&
        root_check.valid;
    engine->model_root_valid.store(root_valid ? 1 : 0, std::memory_order_relaxed);
    engine->tokenizer_valid.store(-1, std::memory_order_relaxed);
    engine->attention_projector_valid.store(-1, std::memory_order_relaxed);
    if (engine->scale4) {
        glm5_scale4_close(engine->scale4);
        engine->scale4 = nullptr;
    }
    if (scale4_path && scale4_path[0]) {
        std::fprintf(stderr,
            "[storagellm] scale4 startup attach deferred: %s\n",
            scale4_path);
        std::fflush(stderr);
    }

    std::ifstream input;
#ifdef _WIN32
    const std::wstring wide_csv_path = storagellm::wide_path_from_utf8(csv_path);
    if (!wide_csv_path.empty()) {
        input.open(wide_csv_path.c_str());
    }
#else
    input.open(csv_path);
#endif
    if (!input) {
        return 0;
    }
    std::string line;
    if (!std::getline(input, line) || !codec_header_supported(line)) {
        return 0;
    }

    std::unordered_map<std::string, uint16_t> path_index_map;
    std::unordered_map<std::string, std::string> resolved_path_cache;
    uint64_t parsed_rows = 0;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        glm5_tensor_record rec{};
        rec.info.part = parse_csv_u32_col_fast(line, GLM5_CODEC_COL_PART);
        rec.info.shard = parse_csv_u32_col_fast(line, GLM5_CODEC_COL_SHARD);
        rec.info.layer = parse_csv_u32_col_fast(line, GLM5_CODEC_COL_LAYER);
        rec.info.expert = parse_csv_u32_col_fast(line, GLM5_CODEC_COL_EXPERT);
        rec.info.proj = parse_projection(csv_string_col_fast(line, GLM5_CODEC_COL_PROJ));
        rec.info.rows = parse_csv_u32_col_fast(line, GLM5_CODEC_COL_ROWS);
        rec.info.cols = parse_csv_u32_col_fast(line, GLM5_CODEC_COL_COLS);
        rec.info.groups = parse_csv_u32_col_fast(line, GLM5_CODEC_COL_GROUPS);
        rec.info.group_size = parse_csv_u32_col_fast(line, GLM5_CODEC_COL_GROUP_SIZE);
        rec.info.weight_byte_offset = parse_csv_u64_col_fast(line, GLM5_CODEC_COL_WEIGHT_OFFSET);
        rec.info.weight_byte_length = parse_csv_u64_col_fast(line, GLM5_CODEC_COL_WEIGHT_LENGTH);
        rec.info.scale_byte_offset = parse_csv_u64_col_fast(line, GLM5_CODEC_COL_SCALE_OFFSET);
        rec.info.scale_byte_length = parse_csv_u64_col_fast(line, GLM5_CODEC_COL_SCALE_LENGTH);
        rec.info.scale2_byte_offset = parse_csv_u64_col_fast(line, GLM5_CODEC_COL_SCALE2_OFFSET);
        rec.info.scale2_byte_length = parse_csv_u64_col_fast(line, GLM5_CODEC_COL_SCALE2_LENGTH);
        std::string shard_file = csv_string_col_fast(line, GLM5_CODEC_COL_SHARD_FILE);
        std::string source_file = csv_string_col_fast(line, GLM5_CODEC_COL_SOURCE_FILE);
        const std::string path_cache_key = shard_file + "\n" + source_file;
        std::string resolved_path;
        auto resolved_it = resolved_path_cache.find(path_cache_key);
        if (resolved_it != resolved_path_cache.end()) {
            resolved_path = resolved_it->second;
        } else {
            resolved_path = resolve_model_file_path(engine->model_root, shard_file, source_file);
            if (resolved_path.empty()) {
                resolved_path = shard_file;
            }
            resolved_path_cache.emplace(path_cache_key, resolved_path);
        }

        uint16_t path_idx = 0xFFFF;
        auto it = path_index_map.find(resolved_path);
        if (it != path_index_map.end()) {
            path_idx = it->second;
        } else {
            path_idx = (uint16_t)engine->tensor_paths.size();
            engine->tensor_paths.push_back(resolved_path);
            path_index_map[resolved_path] = path_idx;
        }
        rec.resolved_path_index = path_idx;

        rec.scale4_entry = nullptr;
        rec.info.has_scale4 = 0;
        finalize_tensor_layout(&rec);
        const uint64_t stream_bytes = tensor_stream_bytes(rec);
        if (stream_bytes > engine->max_tensor_stream_bytes) {
            engine->max_tensor_stream_bytes = stream_bytes;
        }
        tensor_slots_set(engine, (int)rec.info.layer, (int)rec.info.expert, rec.info.proj, engine->tensors.size());
        expert_bytes_add(engine, (int)rec.info.layer, (int)rec.info.expert, stream_bytes);
        engine->tensors.push_back(std::move(rec));
        ++parsed_rows;
        if ((parsed_rows % 10000ull) == 0ull) {
            std::fprintf(stderr, "[storagellm] codec rows loaded: %llu paths=%zu\n",
                (unsigned long long)parsed_rows,
                engine->tensor_paths.size());
            std::fflush(stderr);
        }
    }
    std::fprintf(stderr, "[storagellm] codec rows loaded: %llu paths=%zu\n",
        (unsigned long long)parsed_rows,
        engine->tensor_paths.size());
    std::fflush(stderr);
    if (engine->staging && engine->staging_size > 0) {
        configure_staging_slots(engine);
        engine->staging_slot_cv.notify_all();
    } else {
        update_staging_slot_deficit(engine);
    }
    return engine->tensors.empty() ? 0 : 1;
}
