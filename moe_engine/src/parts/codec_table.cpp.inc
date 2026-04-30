int moe_pc_engine_set_model_root(
    moe_pc_engine_t* engine,
    const char* model_root
) {
    if (!engine) {
        return 0;
    }
    // BUGFIX 52: Issue 23 - Stop IO workers before shard cleanup ?…â˜…??    // Problem: wait_io only waits for queue to drain, but io_workers continue running.
    // After shard_mmaps.clear(), io_worker can call get_or_map_tensor_shard with stale
    // model_root, causing mmap of wrong files or crashes.
    // Solution: stop_io_workers() terminates all workers, then clear shards, then restart.
    // This ensures no worker accesses shards during cleanup.
    // BUGFIX 55: Hold io_start_mutex to serialize with ensure_io_ready
    {
        std::lock_guard<std::mutex> io_lock(engine->io_start_mutex);
        stop_io_workers(engine);
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
    engine->model_root = model_root ? model_root : "";
    moe_model_root_check_t check{};
    int valid = 0;
    if (!engine->model_root.empty()) {
        if (moe_engine_load_offload_gguf_contract(engine, engine->model_root, &check)) {
            valid = check.valid;
        } else {
            valid = moe_storage_validate_model_root(engine->model_root.c_str(), &check) && check.valid;
        }
    }
    engine->model_root_valid.store(valid ? 1 : 0, std::memory_order_relaxed);
    engine->tokenizer_valid.store(-1, std::memory_order_relaxed);
    engine->attention_projector_valid.store(-1, std::memory_order_relaxed);  // Fix 72: invalidate cache

    // Restart IO workers with new model_root
    start_io_workers(engine);

    return 1;
}

static int tensor_index_read_exact(std::ifstream& input, void* data, size_t bytes) {
    if (bytes == 0) {
        return 1;
    }
    input.read(static_cast<char*>(data), static_cast<std::streamsize>(bytes));
    return input.good() ? 1 : 0;
}

static int tensor_index_read_u16(std::ifstream& input, uint16_t* out) {
    return out && tensor_index_read_exact(input, out, sizeof(*out));
}

static int tensor_index_read_u32(std::ifstream& input, uint32_t* out) {
    return out && tensor_index_read_exact(input, out, sizeof(*out));
}

static int tensor_index_read_u64(std::ifstream& input, uint64_t* out) {
    return out && tensor_index_read_exact(input, out, sizeof(*out));
}

static void finish_tensor_index_load(moe_pc_engine* engine) {
    if (!engine) {
        return;
    }
    if (engine->staging && engine->staging_size > 0) {
        configure_staging_slots(engine);
        engine->staging_slot_cv.notify_all();
    } else {
        update_staging_slot_deficit(engine);
    }
}

static int load_binary_tensor_index(moe_pc_engine* engine, const char* index_path) {
    if (!engine || !index_path || !index_path[0]) {
        return 0;
    }
    std::ifstream input;
    open_ifstream_path(input, index_path, std::ios::in | std::ios::binary);
    if (!input) {
        return 0;
    }

    char magic[8]{};
    uint32_t version = 0;
    uint32_t path_count = 0;
    uint32_t tensor_count = 0;
    uint32_t reserved = 0;
    if (!tensor_index_read_exact(input, magic, sizeof(magic)) ||
        (std::memcmp(magic, "SLTIDX1", 7) != 0 && std::memcmp(magic, "SLTIDX2", 7) != 0) ||
        !tensor_index_read_u32(input, &version) ||
        !tensor_index_read_u32(input, &path_count) ||
        !tensor_index_read_u32(input, &tensor_count) ||
        !tensor_index_read_u32(input, &reserved) ||
        (version != 1 && version != 2) ||
        path_count == 0 ||
        path_count > 4096 ||
        tensor_count == 0 ||
        tensor_count > 1000000u) {
        return 0;
    }
    if (engine->scale4 && version < 2) {
        std::fprintf(stderr,
            "[storagellm] stale tensor index has no scale4 keys, falling back to tensors.csv: %s\n",
            index_path);
        std::fflush(stderr);
        return 0;
    }

    engine->tensor_paths.reserve(path_count);
    for (uint32_t i = 0; i < path_count; ++i) {
        uint16_t len = 0;
        if (!tensor_index_read_u16(input, &len) || len == 0) {
            return 0;
        }
        std::string raw_path(len, '\0');
        if (!tensor_index_read_exact(input, &raw_path[0], raw_path.size())) {
            return 0;
        }
        std::string resolved_path = resolve_model_file_path(engine->model_root, raw_path, raw_path);
        if (resolved_path.empty()) {
            resolved_path = raw_path;
        }
        engine->tensor_paths.push_back(std::move(resolved_path));
    }
    std::vector<std::string> scale4_keys;
    if (version >= 2) {
        const uint32_t key_count = reserved;
        if (key_count > tensor_count) {
            return 0;
        }
        scale4_keys.reserve(key_count);
        for (uint32_t i = 0; i < key_count; ++i) {
            uint16_t len = 0;
            if (!tensor_index_read_u16(input, &len) || len == 0) {
                return 0;
            }
            std::string key(len, '\0');
            if (!tensor_index_read_exact(input, &key[0], key.size())) {
                return 0;
            }
            scale4_keys.push_back(std::move(key));
        }
    }

    engine->tensors.reserve(tensor_count);
    for (uint32_t i = 0; i < tensor_count; ++i) {
        moe_tensor_record rec{};
        uint32_t proj = 0;
        uint16_t path_index = 0;
        uint16_t flags = 0;
        uint32_t rec_reserved = 0;
        if (!tensor_index_read_u32(input, &rec.info.part) ||
            !tensor_index_read_u32(input, &rec.info.shard) ||
            !tensor_index_read_u32(input, &rec.info.layer) ||
            !tensor_index_read_u32(input, &rec.info.expert) ||
            !tensor_index_read_u32(input, &proj) ||
            !tensor_index_read_u32(input, &rec.info.rows) ||
            !tensor_index_read_u32(input, &rec.info.cols) ||
            !tensor_index_read_u32(input, &rec.info.groups) ||
            !tensor_index_read_u32(input, &rec.info.group_size) ||
            !tensor_index_read_u64(input, &rec.info.weight_byte_offset) ||
            !tensor_index_read_u64(input, &rec.info.weight_byte_length) ||
            !tensor_index_read_u64(input, &rec.info.scale_byte_offset) ||
            !tensor_index_read_u64(input, &rec.info.scale_byte_length) ||
            !tensor_index_read_u64(input, &rec.info.scale2_byte_offset) ||
            !tensor_index_read_u64(input, &rec.info.scale2_byte_length) ||
            !tensor_index_read_u16(input, &path_index) ||
            !tensor_index_read_u16(input, &flags) ||
            !tensor_index_read_u32(input, &rec_reserved) ||
            proj > (uint32_t)moe_PROJ_DOWN ||
            path_index >= engine->tensor_paths.size()) {
            return 0;
        }
        rec.info.proj = static_cast<moe_projection_t>(proj);
        rec.resolved_path_index = path_index;
        rec.scale4_entry = nullptr;
        rec.info.has_scale4 = 0;
        if (engine->scale4 && version >= 2 && (flags & 1u) && rec_reserved < scale4_keys.size()) {
            rec.scale4_entry = moe_scale4_find(engine->scale4, scale4_keys[rec_reserved].c_str());
            rec.info.has_scale4 = rec.scale4_entry ? 1 : 0;
        }
        finalize_tensor_layout(&rec);
        const uint64_t stream_bytes = tensor_stream_bytes(rec);
        if (stream_bytes > engine->max_tensor_stream_bytes) {
            engine->max_tensor_stream_bytes = stream_bytes;
        }
        tensor_slots_set(engine, (int)rec.info.layer, (int)rec.info.expert, rec.info.proj, engine->tensors.size());
        expert_bytes_add(engine, (int)rec.info.layer, (int)rec.info.expert, stream_bytes);
        engine->tensors.push_back(std::move(rec));
    }

    if (!input.good() && !input.eof()) {
        return 0;
    }
    finish_tensor_index_load(engine);
    std::fprintf(stderr, "[storagellm] binary tensor index loaded: %u tensors paths=%u scale4=%s\n",
        tensor_count,
        path_count,
        engine->scale4 ? "on" : "off");
    std::fflush(stderr);
    return 1;
}

static int moe_parse_layer_from_gguf_name(const std::string& name, uint32_t* layer) {
    if (!layer) {
        return 0;
    }
    const char* needle = "blk.";
    const size_t pos = name.find(needle);
    if (pos == std::string::npos) {
        return 0;
    }
    size_t p = pos + std::strlen(needle);
    uint64_t value = 0;
    bool any = false;
    while (p < name.size() && name[p] >= '0' && name[p] <= '9') {
        value = value * 10ull + (uint64_t)(name[p] - '0');
        any = true;
        ++p;
    }
    if (!any || value > UINT32_MAX) {
        return 0;
    }
    *layer = (uint32_t)value;
    return 1;
}

static int moe_gguf_expert_projection(const std::string& name, moe_projection_t* proj) {
    if (!proj) {
        return 0;
    }
    const std::string value = moe_lower_ascii_copy(name);
    if (value.find("ffn_gate_exps.weight") != std::string::npos ||
        value.find("gate_proj.weight") != std::string::npos) {
        *proj = moe_PROJ_GATE;
        return 1;
    }
    if (value.find("ffn_up_exps.weight") != std::string::npos ||
        value.find("up_proj.weight") != std::string::npos) {
        *proj = moe_PROJ_UP;
        return 1;
    }
    if (value.find("ffn_down_exps.weight") != std::string::npos ||
        value.find("down_proj.weight") != std::string::npos) {
        *proj = moe_PROJ_DOWN;
        return 1;
    }
    return 0;
}

static uint8_t moe_weight_encoding_from_gguf_type(uint32_t type, const moe_offload_gguf_contract_t& contract) {
    if (type == 39u) {
        return moe_WEIGHT_ENCODING_MXFP4;
    }
    if (type == 16u || type == 17u || type == 22u) {
        return moe_WEIGHT_ENCODING_GGUF_IQ2;
    }
    if (type == 18u || type == 21u) {
        return moe_WEIGHT_ENCODING_GGUF_IQ3;
    }
    if (type == 20u || type == 23u || type == 12u) {
        return moe_WEIGHT_ENCODING_GGUF_IQ4;
    }
    if (contract.weight_encoding) {
        return (uint8_t)std::min<uint32_t>(contract.weight_encoding, 255u);
    }
    return moe_WEIGHT_ENCODING_UNKNOWN;
}

static int load_gguf_tensor_directory_as_tensor_index(
    moe_pc_engine* engine,
    const std::string& root,
    const moe_offload_gguf_contract_t& contract
) {
    if (!engine || root.empty()) {
        return 0;
    }
    std::vector<std::string> paths;
    moe_collect_gguf_files(root, 4, &paths);
    if (paths.empty()) {
        return 0;
    }
    std::sort(paths.begin(), paths.end());
    std::unordered_map<std::string, uint16_t> path_index_map;
    uint64_t records = 0;
    for (const std::string& path : paths) {
        moe_gguf_tensor_dir_t dir{};
        if (!moe_read_gguf_tensor_directory(path, &dir)) {
            continue;
        }
        uint16_t path_idx = 0xFFFFu;
        auto pit = path_index_map.find(path);
        if (pit != path_index_map.end()) {
            path_idx = pit->second;
        } else {
            if (engine->tensor_paths.size() >= 0xFFFFu) {
                continue;
            }
            path_idx = (uint16_t)engine->tensor_paths.size();
            engine->tensor_paths.push_back(path);
            path_index_map.emplace(path, path_idx);
        }
        const uint32_t part_id = dir.file_index ? dir.file_index : (uint32_t)path_idx + 1u;
        const uint32_t shard_id = part_id;
        for (const moe_gguf_tensor_dir_entry_t& entry : dir.tensors) {
            uint32_t layer = 0;
            moe_projection_t proj = moe_PROJ_GATE;
            if (!moe_parse_layer_from_gguf_name(entry.name, &layer) ||
                !moe_gguf_expert_projection(entry.name, &proj) ||
                entry.dims < 3 ||
                entry.shape[2] == 0 ||
                entry.bytes == 0) {
                continue;
            }
            const uint64_t expert_count64 = entry.shape[2];
            const uint64_t per_expert = expert_count64 ? entry.bytes / expert_count64 : 0;
            if (per_expert == 0 || expert_count64 > 4096ull) {
                continue;
            }
            const uint32_t rows = (uint32_t)std::min<uint64_t>(entry.shape[1], UINT32_MAX);
            const uint32_t cols = (uint32_t)std::min<uint64_t>(entry.shape[0], UINT32_MAX);
            for (uint64_t expert = 0; expert < expert_count64; ++expert) {
                moe_tensor_record rec{};
                rec.info.part = part_id;
                rec.info.shard = shard_id;
                rec.info.layer = layer;
                rec.info.expert = (uint32_t)expert;
                rec.info.proj = proj;
                rec.info.rows = rows;
                rec.info.cols = cols;
                rec.info.groups = cols >= 16u ? std::max<uint32_t>(1u, cols / 16u) : 1u;
                rec.info.group_size = cols >= 16u ? 16u : cols;
                rec.info.weight_byte_offset = entry.file_offset + expert * per_expert;
                rec.info.weight_byte_length = (expert + 1ull == expert_count64) ?
                    (entry.file_offset + entry.bytes - rec.info.weight_byte_offset) : per_expert;
                rec.info.scale_byte_offset = 0;
                rec.info.scale_byte_length = 0;
                rec.info.scale2_byte_offset = 0;
                rec.info.scale2_byte_length = 0;
                rec.resolved_path_index = path_idx;
                finalize_tensor_layout(&rec, moe_weight_encoding_from_gguf_type(entry.type, contract));
                const uint64_t stream_bytes = tensor_stream_bytes(rec);
                if (stream_bytes > engine->max_tensor_stream_bytes) {
                    engine->max_tensor_stream_bytes = stream_bytes;
                }
                tensor_slots_set(engine, (int)rec.info.layer, (int)rec.info.expert, rec.info.proj, engine->tensors.size());
                expert_bytes_add(engine, (int)rec.info.layer, (int)rec.info.expert, stream_bytes);
                engine->tensors.push_back(std::move(rec));
                ++records;
            }
        }
    }
    if (!records) {
        return 0;
    }
    finish_tensor_index_load(engine);
    std::fprintf(stderr,
        "[storagellm] GGUF tensor directory indexed: records=%llu paths=%zu max_stream=%llu\n",
        (unsigned long long)records,
        engine->tensor_paths.size(),
        (unsigned long long)engine->max_tensor_stream_bytes);
    std::fflush(stderr);
    return 1;
}

static int path_ends_with_csv(const char* path) {
    if (!path) {
        return 0;
    }
    const std::string value(path);
    return value.size() >= 4 && value.substr(value.size() - 4) == ".csv";
}

int moe_pc_engine_load_codec_table(
    moe_pc_engine_t* engine,
    const char* csv_path,
    const char* model_root,
    const char* scale4_path
) {
    const bool offload_gguf_header_only =
        !csv_path || !csv_path[0] || std::strcmp(csv_path, "__offload_gguf__") == 0;
    if (!engine) {
        return 0;
    }

    moe_pc_engine_wait_io(engine);
    device_free_pool(engine, moe_DEVICE_POOL_FIXED);
    device_free_pool(engine, moe_DEVICE_POOL_EXPERT);
    device_free_pool(engine, moe_DEVICE_POOL_ACTIVATION);
    device_free_pool(engine, moe_DEVICE_POOL_STREAM);
    engine->expert_states.assign(expert_slot_count(), moe_block_state{});
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
    moe_engine_reset_qkv_cache(engine);
    moe_engine_clear_offload_gguf_contract(engine);
    engine->tensors.clear();
    engine->tensor_paths.clear();
    engine->max_tensor_stream_bytes = 0;
    engine->common_raw_prefetched_bytes.store(0);
    engine->common_raw_prefetched_spans.store(0);
    engine->tensors.reserve(tensor_record_capacity());
    tensor_slots_reset(engine);
    engine->model_root = model_root ? model_root : "";
    moe_model_root_check_t root_check{};
    int root_valid = 0;
    if (!engine->model_root.empty()) {
        if (moe_engine_load_offload_gguf_contract(engine, engine->model_root, &root_check)) {
            root_valid = root_check.valid;
        } else {
            root_valid = moe_storage_validate_model_root(engine->model_root.c_str(), &root_check) &&
                root_check.valid;
        }
    }
    engine->model_root_valid.store(root_valid ? 1 : 0, std::memory_order_relaxed);
    engine->tokenizer_valid.store(-1, std::memory_order_relaxed);
    engine->attention_projector_valid.store(-1, std::memory_order_relaxed);
    if (engine->scale4) {
        moe_scale4_close(engine->scale4);
        engine->scale4 = nullptr;
    }
    if (scale4_path && scale4_path[0]) {
        std::fprintf(stderr,
            "[storagellm] scale4 attach: %s\n",
            scale4_path);
        std::fflush(stderr);
        engine->scale4 = moe_scale4_open(scale4_path);
        if (!engine->scale4 || moe_scale4_entry_count(engine->scale4) == 0) {
            std::fprintf(stderr, "[storagellm] failed to attach scale4: %s\n", scale4_path);
            std::fflush(stderr);
            if (engine->scale4) {
                moe_scale4_close(engine->scale4);
                engine->scale4 = nullptr;
            }
            return 0;
        }
        std::fprintf(stderr, "[storagellm] scale4 attached entries=%u\n",
            moe_scale4_entry_count(engine->scale4));
        std::fflush(stderr);
    }

    if (offload_gguf_header_only) {
        if (!root_valid || !engine->offload_gguf_valid.load(std::memory_order_relaxed)) {
            return 0;
        }
        moe_offload_gguf_contract_t tensor_contract{};
        tensor_contract.weight_bits = engine->offload_weight_quant_bits.load(std::memory_order_relaxed);
        tensor_contract.weight_encoding = engine->offload_weight_encoding.load(std::memory_order_relaxed);
        tensor_contract.weight_block_size = engine->offload_weight_block_size.load(std::memory_order_relaxed);
        tensor_contract.weight_quant_family = engine->offload_weight_quant_family;
        tensor_contract.weight_kernel_family = engine->offload_weight_kernel_family;
        if (!load_gguf_tensor_directory_as_tensor_index(
                engine,
                engine->model_root,
                tensor_contract)) {
            finish_tensor_index_load(engine);
        }
        std::fprintf(stderr,
            "[storagellm] offload GGUF index loaded: files=%u tensor_headers=%llu executable_tensors=%llu runtime_records=%zu\n",
            engine->offload_gguf_file_count.load(std::memory_order_relaxed),
            (unsigned long long)engine->offload_gguf_tensor_count.load(std::memory_order_relaxed),
            (unsigned long long)engine->offload_gguf_executable_tensor_count.load(std::memory_order_relaxed),
            engine->tensors.size());
        std::fflush(stderr);
        return 1;
    }

    if (load_binary_tensor_index(engine, csv_path)) {
        return 1;
    }
    std::string fallback_csv;
    const char* csv_load_path = csv_path;
    if (!path_ends_with_csv(csv_path)) {
        if (!engine->model_root.empty()) {
            fallback_csv = storagellm::path_join(engine->model_root, "tensors.csv");
            uint64_t ignored = 0;
            if (file_size_bytes(fallback_csv, &ignored)) {
                csv_load_path = fallback_csv.c_str();
                std::fprintf(stderr, "[storagellm] loading csv fallback: %s\n", csv_load_path);
                std::fflush(stderr);
            } else {
                return 0;
            }
        } else {
            return 0;
        }
    }

    std::ifstream input;
#ifdef _WIN32
    const std::wstring wide_csv_path = storagellm::wide_path_from_utf8(csv_load_path);
    if (!wide_csv_path.empty()) {
        input.open(wide_csv_path.c_str());
    }
#else
    input.open(csv_load_path);
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
        moe_tensor_record rec{};
        rec.info.part = parse_csv_u32_col_fast(line, moe_CODEC_COL_PART);
        rec.info.shard = parse_csv_u32_col_fast(line, moe_CODEC_COL_SHARD);
        rec.info.layer = parse_csv_u32_col_fast(line, moe_CODEC_COL_LAYER);
        rec.info.expert = parse_csv_u32_col_fast(line, moe_CODEC_COL_EXPERT);
        rec.info.proj = parse_projection(csv_string_col_fast(line, moe_CODEC_COL_PROJ));
        rec.info.rows = parse_csv_u32_col_fast(line, moe_CODEC_COL_ROWS);
        rec.info.cols = parse_csv_u32_col_fast(line, moe_CODEC_COL_COLS);
        rec.info.groups = parse_csv_u32_col_fast(line, moe_CODEC_COL_GROUPS);
        rec.info.group_size = parse_csv_u32_col_fast(line, moe_CODEC_COL_GROUP_SIZE);
        rec.info.weight_byte_offset = parse_csv_u64_col_fast(line, moe_CODEC_COL_WEIGHT_OFFSET);
        rec.info.weight_byte_length = parse_csv_u64_col_fast(line, moe_CODEC_COL_WEIGHT_LENGTH);
        rec.info.scale_byte_offset = parse_csv_u64_col_fast(line, moe_CODEC_COL_SCALE_OFFSET);
        rec.info.scale_byte_length = parse_csv_u64_col_fast(line, moe_CODEC_COL_SCALE_LENGTH);
        rec.info.scale2_byte_offset = parse_csv_u64_col_fast(line, moe_CODEC_COL_SCALE2_OFFSET);
        rec.info.scale2_byte_length = parse_csv_u64_col_fast(line, moe_CODEC_COL_SCALE2_LENGTH);
        const std::string weight_key = csv_string_col_fast(line, moe_CODEC_COL_WEIGHT_KEY);
        std::string shard_file = csv_string_col_fast(line, moe_CODEC_COL_SHARD_FILE);
        std::string source_file = csv_string_col_fast(line, moe_CODEC_COL_SOURCE_FILE);
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
        if (engine->scale4 && !weight_key.empty()) {
            rec.scale4_entry = moe_scale4_find(engine->scale4, weight_key.c_str());
        }
        rec.info.has_scale4 = rec.scale4_entry ? 1 : 0;
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
    finish_tensor_index_load(engine);
    return engine->tensors.empty() ? 0 : 1;
}
