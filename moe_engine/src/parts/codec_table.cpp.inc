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
    uint64_t expert_sum = 0;
    uint64_t expert_count = 0;
    for (uint64_t bytes : engine->expert_bytes) {
        if (bytes) {
            expert_sum += bytes;
            ++expert_count;
        }
    }
    engine->avg_expert_bytes = expert_count ? expert_sum / expert_count : engine->max_tensor_stream_bytes;
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

static void moe_dynamic_shape_note_tensor(
    moe_pc_engine* engine,
    const std::string& key,
    uint32_t dims,
    const uint64_t* shape
) {
    if (!engine || key.empty() || !shape) {
        return;
    }
    moe_model_shape_t next = engine->dynamic_shape;
    if (key == "token_embd.weight" && dims >= 2 && shape[0] && shape[1]) {
        next.hidden_size = (uint32_t)std::min<uint64_t>(shape[0], UINT32_MAX);
        next.vocab_size = (uint32_t)std::min<uint64_t>(shape[1], UINT32_MAX);
    }
    uint32_t layer = 0;
    if (moe_parse_layer_from_gguf_name(key, &layer)) {
        if (layer + 1u > next.num_hidden_layers) {
            next.num_hidden_layers = layer + 1u;
        }
        if (key.find("ffn_gate_up_exps.weight") != std::string::npos ||
            key.find("ffn_gate_exps.weight") != std::string::npos ||
            key.find("ffn_down_exps.weight") != std::string::npos) {
            if (next.first_moe_layer == 0 && next.last_moe_layer == 0) {
                next.first_moe_layer = layer;
            } else if (layer < next.first_moe_layer) {
                next.first_moe_layer = layer;
            }
            if (layer > next.last_moe_layer) {
                next.last_moe_layer = layer;
            }
            if (dims >= 3 && shape[2]) {
                next.experts_per_moe_layer =
                    std::max<uint32_t>(next.experts_per_moe_layer,
                        (uint32_t)std::min<uint64_t>(shape[2], UINT32_MAX));
            }
            if (dims >= 2 && shape[1]) {
                next.expert_intermediate_size =
                    std::max<uint32_t>(next.expert_intermediate_size,
                        (uint32_t)std::min<uint64_t>(shape[1], UINT32_MAX));
            }
        }
    }
    if (key.find("ffn_gate_up_exps.weight") != std::string::npos) {
        if (engine->dynamic_architecture.empty()) {
            engine->dynamic_architecture = "gemma";
        }
    }
    if (next.hidden_size && next.vocab_size && next.num_hidden_layers) {
        next.projection_count = 3;
        engine->dynamic_shape = next;
        engine->dynamic_shape_valid.store(1, std::memory_order_relaxed);
    } else {
        engine->dynamic_shape = next;
    }
}

static void moe_dynamic_shape_finalize_adapter(moe_pc_engine* engine) {
    if (!engine || !engine->dynamic_shape_valid.load(std::memory_order_relaxed)) {
        return;
    }
    const std::string arch = moe_lower_ascii_copy(engine->dynamic_architecture);
    uint32_t adapter = moe_FORWARD_ADAPTER_GGUF_GENERIC;
    int executable = 0;
    if (arch.find("glm") != std::string::npos) {
        adapter = moe_FORWARD_ADAPTER_GLM_RAW;
        executable = 1;
    } else if (arch.find("gemma") != std::string::npos ||
               engine->gguf_common_tensor_slots.find("blk.0.ffn_gate_up_exps.weight") != engine->gguf_common_tensor_slots.end()) {
        adapter = moe_FORWARD_ADAPTER_GGUF_GEMMA;
        executable = 0;
    }
    engine->forward_adapter.store(adapter, std::memory_order_relaxed);
    engine->forward_adapter_executable.store(executable, std::memory_order_relaxed);
    engine->attention_projector_valid.store(-1, std::memory_order_relaxed);
}


static void moe_register_gguf_common_tensor(
    moe_pc_engine* engine,
    const moe_gguf_tensor_dir_entry_t& entry,
    uint16_t path_idx
) {
    if (!engine || entry.name.empty() || entry.bytes == 0) {
        return;
    }
    const std::string key = moe_lower_ascii_copy(entry.name);
    if (engine->gguf_common_tensor_slots.find(key) != engine->gguf_common_tensor_slots.end()) {
        return;
    }
    moe_gguf_common_tensor_record rec{};
    rec.name = key;
    rec.resolved_path_index = path_idx;
    rec.dims = std::min<uint32_t>(entry.dims, 4u);
    for (uint32_t i = 0; i < rec.dims; ++i) {
        rec.shape[i] = entry.shape[i];
    }
    rec.type = entry.type;
    rec.file_offset = entry.file_offset;
    rec.bytes = entry.bytes;
    engine->gguf_common_tensor_slots.emplace(rec.name, engine->gguf_common_tensors.size());
    engine->gguf_common_tensors.push_back(std::move(rec));
    moe_dynamic_shape_note_tensor(engine, key, entry.dims, entry.shape);
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
            if (moe_gguf_tensor_name_executable(entry.name)) {
                moe_register_gguf_common_tensor(engine, entry, path_idx);
            }
            const std::string lower_name = moe_lower_ascii_copy(entry.name);
            const bool combined_gate_up =
                lower_name.find("ffn_gate_up_exps.weight") != std::string::npos;
            uint32_t layer = 0;
            moe_projection_t proj = moe_PROJ_GATE;
            if (!moe_parse_layer_from_gguf_name(entry.name, &layer) ||
                (!combined_gate_up && !moe_gguf_expert_projection(entry.name, &proj)) ||
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
            if (combined_gate_up && ((entry.shape[1] & 1ull) != 0 || per_expert < 2ull)) {
                continue;
            }
            const uint32_t rows_total = (uint32_t)std::min<uint64_t>(entry.shape[1], UINT32_MAX);
            const uint32_t cols = (uint32_t)std::min<uint64_t>(entry.shape[0], UINT32_MAX);
            for (uint64_t expert = 0; expert < expert_count64; ++expert) {
                const uint32_t split_count = combined_gate_up ? 2u : 1u;
                for (uint32_t split = 0; split < split_count; ++split) {
                    moe_tensor_record rec{};
                    rec.info.part = part_id;
                    rec.info.shard = shard_id;
                    rec.info.layer = layer;
                    rec.info.expert = (uint32_t)expert;
                    rec.info.proj = combined_gate_up ?
                        (split == 0 ? moe_PROJ_GATE : moe_PROJ_UP) : proj;
                    rec.info.rows = combined_gate_up ? (rows_total / 2u) : rows_total;
                    rec.info.cols = cols;
                    rec.info.groups = cols >= 16u ? std::max<uint32_t>(1u, cols / 16u) : 1u;
                    rec.info.group_size = cols >= 16u ? 16u : cols;
                    const uint64_t expert_offset = entry.file_offset + expert * per_expert;
                    const uint64_t split_bytes = combined_gate_up ? (per_expert / 2ull) : per_expert;
                    rec.info.weight_byte_offset = expert_offset + (uint64_t)split * split_bytes;
                    rec.info.weight_byte_length = combined_gate_up && split == 1u ?
                        (per_expert - split_bytes) : split_bytes;
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
    }
    if (!records) {
        return 0;
    }
    finish_tensor_index_load(engine);
    moe_dynamic_shape_finalize_adapter(engine);
    std::fprintf(stderr,
        "[storagellm] GGUF tensor directory indexed: records=%llu paths=%zu max_stream=%llu\n",
        (unsigned long long)records,
        engine->tensor_paths.size(),
        (unsigned long long)engine->max_tensor_stream_bytes);
    std::fflush(stderr);
    return 1;
}

static uint64_t moe_count_json_key_occurrences(const std::string& text, const char* key) {
    if (!key || !key[0]) {
        return 0;
    }
    uint64_t count = 0;
    size_t pos = 0;
    while ((pos = text.find(key, pos)) != std::string::npos) {
        ++count;
        pos += std::strlen(key);
    }
    return count;
}

static void moe_ingest_juju_graph_ir(moe_pc_engine* engine, const std::string& index_json) {
    if (!engine || index_json.empty()) {
        return;
    }
    uint64_t index_schema_version = 0;
    if (moe_json_get_u64_local(index_json, "schema_version", &index_schema_version)) {
        const uint32_t version = (uint32_t)std::min<uint64_t>(index_schema_version, UINT32_MAX);
        uint32_t current = engine->juju_idx_schema_version.load(std::memory_order_relaxed);
        while (version > current &&
               !engine->juju_idx_schema_version.compare_exchange_weak(
                   current, version, std::memory_order_relaxed)) {
        }
    }
    uint64_t split_count = 0;
    if (moe_json_get_u64_local(index_json, "split_count", &split_count) && split_count > 1) {
        engine->juju_split_group_count.fetch_add(1, std::memory_order_relaxed);
    }
    if (index_json.find("\"graph_ir\"") == std::string::npos ||
        index_json.find("JUJU_GRAPH_IR_V1") == std::string::npos) {
        return;
    }
    engine->offload_graph_ir_ready.store(1, std::memory_order_relaxed);
    if (index_json.find("\"graph_ir_required\":true") != std::string::npos ||
        index_json.find("\"graph_ir_required\": true") != std::string::npos) {
        engine->offload_graph_ir_required.store(1, std::memory_order_relaxed);
    }
    uint64_t schema_version = 1;
    size_t graph_pos = index_json.find("\"graph_ir\"");
    if (graph_pos != std::string::npos) {
        std::string graph_tail = index_json.substr(graph_pos);
        moe_json_get_u64_local(graph_tail, "schema_version", &schema_version);
        engine->offload_graph_ir_op_count.fetch_add(
            moe_count_json_key_occurrences(graph_tail, "\"op\""),
            std::memory_order_relaxed);
        engine->offload_graph_ir_layer_count.fetch_add(
            moe_count_json_key_occurrences(graph_tail, "\"layer\""),
            std::memory_order_relaxed);
        if (engine->offload_graph_ir_json.empty()) {
            engine->offload_graph_ir_json = graph_tail.substr(0, std::min<size_t>(graph_tail.size(), 1u << 20));
        }
    }
    engine->offload_graph_ir_version.store((uint32_t)std::min<uint64_t>(schema_version, UINT32_MAX), std::memory_order_relaxed);
}

static int load_juju_tensor_index_as_tensor_index(
    moe_pc_engine* engine,
    const std::string& root,
    const moe_offload_gguf_contract_t& contract
) {
    if (!engine || root.empty()) {
        return 0;
    }
    std::vector<std::string> paths;
    moe_collect_juju_files(root, 4, &paths);
    if (paths.empty()) {
        return 0;
    }
    std::sort(paths.begin(), paths.end());
    std::unordered_map<std::string, uint16_t> path_index_map;
    uint64_t records = 0;
    for (const std::string& path : paths) {
        moe_juju_header_t header{};
        std::vector<moe_juju_section_entry_t> sections;
        uint64_t file_bytes = 0;
        if (!moe_juju_read_table_local(path, &header, &sections, &file_bytes)) {
            continue;
        }
        std::string index_json;
        if (!moe_read_juju_tensor_index_json(path, sections, file_bytes, &index_json)) {
            continue;
        }
        moe_ingest_juju_graph_ir(engine, index_json);
        std::vector<moe_juju_tensor_json_entry_t> tensors;
        if (!moe_parse_juju_tensor_index_json(index_json, &tensors, nullptr, nullptr, nullptr) ||
            tensors.empty()) {
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
        const uint32_t part_id = (uint32_t)path_idx + 1u;
        const uint32_t shard_id = part_id;
        for (const moe_juju_tensor_json_entry_t& entry : tensors) {
            if (entry.juju_bytes == 0 ||
                entry.juju_offset > file_bytes ||
                entry.juju_bytes > file_bytes - entry.juju_offset) {
                continue;
            }
            if (entry.runtime_priority != 50 || entry.prefetch_priority != 50 ||
                entry.residency_hint != moe_JUJU_RESIDENCY_AUTO ||
                entry.graph_role != moe_JUJU_ROLE_UNKNOWN ||
                entry.prefetch_class != moe_JUJU_PREFETCH_AUTO) {
                engine->juju_runtime_hint_tensor_count.fetch_add(1, std::memory_order_relaxed);
            }
            if (entry.runtime_priority != 50 || entry.prefetch_priority != 50) {
                engine->juju_priority_tensor_count.fetch_add(1, std::memory_order_relaxed);
            }
            if (entry.residency_hint == moe_JUJU_RESIDENCY_FAST_MEM ||
                entry.residency_hint == moe_JUJU_RESIDENCY_FAST_MEM_STREAMABLE) {
                engine->juju_fastmem_tensor_count.fetch_add(1, std::memory_order_relaxed);
            } else if (entry.residency_hint == moe_JUJU_RESIDENCY_SLOW_MEM) {
                engine->juju_slowmem_tensor_count.fetch_add(1, std::memory_order_relaxed);
            }
            if (moe_gguf_tensor_name_executable(entry.name)) {
                moe_gguf_tensor_dir_entry_t common{};
                common.name = entry.name;
                common.dims = entry.dims;
                common.type = entry.gguf_type;
                common.file_offset = entry.juju_offset;
                common.bytes = entry.juju_bytes;
                for (uint32_t i = 0; i < std::min<uint32_t>(entry.dims, 4u); ++i) {
                    common.shape[i] = entry.shape[i];
                }
                moe_register_gguf_common_tensor(engine, common, path_idx);
                auto common_it = engine->gguf_common_tensor_slots.find(moe_lower_ascii_copy(entry.name));
                if (common_it != engine->gguf_common_tensor_slots.end() &&
                    common_it->second < engine->gguf_common_tensors.size()) {
                    moe_gguf_common_tensor_record& common_rec = engine->gguf_common_tensors[common_it->second];
                    common_rec.runtime_priority = entry.runtime_priority;
                    common_rec.prefetch_priority = entry.prefetch_priority;
                    common_rec.residency_hint = entry.residency_hint;
                    common_rec.graph_role = entry.graph_role;
                    common_rec.prefetch_class = entry.prefetch_class;
                }
            }
            const std::string lower_name = moe_lower_ascii_copy(entry.name);
            const bool combined_gate_up =
                lower_name.find("ffn_gate_up_exps.weight") != std::string::npos;
            uint32_t layer = 0;
            moe_projection_t proj = moe_PROJ_GATE;
            if (!moe_parse_layer_from_gguf_name(entry.name, &layer) ||
                (!combined_gate_up && !moe_gguf_expert_projection(entry.name, &proj)) ||
                entry.dims < 3 ||
                entry.shape[2] == 0) {
                continue;
            }
            const uint64_t expert_count64 = entry.shape[2];
            const uint64_t per_expert = expert_count64 ? entry.juju_bytes / expert_count64 : 0;
            if (per_expert == 0 || expert_count64 > 4096ull) {
                continue;
            }
            if (combined_gate_up && ((entry.shape[1] & 1ull) != 0 || per_expert < 2ull)) {
                continue;
            }
            const uint32_t rows_total = (uint32_t)std::min<uint64_t>(entry.shape[1], UINT32_MAX);
            const uint32_t cols = (uint32_t)std::min<uint64_t>(entry.shape[0], UINT32_MAX);
            for (uint64_t expert = 0; expert < expert_count64; ++expert) {
                const uint32_t split_count = combined_gate_up ? 2u : 1u;
                for (uint32_t split = 0; split < split_count; ++split) {
                    moe_tensor_record rec{};
                    rec.info.part = part_id;
                    rec.info.shard = shard_id;
                    rec.info.layer = layer;
                    rec.info.expert = (uint32_t)expert;
                    rec.info.proj = combined_gate_up ?
                        (split == 0 ? moe_PROJ_GATE : moe_PROJ_UP) : proj;
                    rec.info.rows = combined_gate_up ? (rows_total / 2u) : rows_total;
                    rec.info.cols = cols;
                    rec.info.groups = cols >= 16u ? std::max<uint32_t>(1u, cols / 16u) : 1u;
                    rec.info.group_size = cols >= 16u ? 16u : cols;
                    const uint64_t expert_offset = entry.juju_offset + expert * per_expert;
                    const uint64_t split_bytes = combined_gate_up ? (per_expert / 2ull) : per_expert;
                    rec.info.weight_byte_offset = expert_offset + (uint64_t)split * split_bytes;
                    rec.info.weight_byte_length = combined_gate_up && split == 1u ?
                        (per_expert - split_bytes) : split_bytes;
                    rec.info.scale_byte_offset = 0;
                    rec.info.scale_byte_length = 0;
                    rec.info.scale2_byte_offset = 0;
                    rec.info.scale2_byte_length = 0;
                    rec.resolved_path_index = path_idx;
                    rec.runtime_priority = entry.runtime_priority;
                    rec.prefetch_priority = entry.prefetch_priority;
                    rec.residency_hint = entry.residency_hint;
                    rec.graph_role = entry.graph_role;
                    rec.prefetch_class = entry.prefetch_class;
                    rec.juju_section_type = entry.juju_section_type;
                    finalize_tensor_layout(&rec, moe_weight_encoding_from_gguf_type(entry.gguf_type, contract));
                    const uint64_t stream_bytes = tensor_stream_bytes(rec);
                    if (stream_bytes > engine->max_tensor_stream_bytes) {
                        engine->max_tensor_stream_bytes = stream_bytes;
                    }
                    tensor_slots_set(engine, (int)rec.info.layer, (int)rec.info.expert, rec.info.proj, engine->tensors.size());
                    expert_bytes_add(engine, (int)rec.info.layer, (int)rec.info.expert, stream_bytes);
                    expert_runtime_hints_update(
                        engine,
                        (int)rec.info.layer,
                        (int)rec.info.expert,
                        rec.runtime_priority,
                        rec.prefetch_priority,
                        rec.residency_hint);
                    engine->tensors.push_back(std::move(rec));
                    ++records;
                }
            }
        }
    }
    if (!records) {
        return 0;
    }
    finish_tensor_index_load(engine);
    moe_dynamic_shape_finalize_adapter(engine);
    std::fprintf(stderr,
        "[storagellm] JUJU tensor index loaded: records=%llu paths=%zu max_stream=%llu\n",
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
    // BUGFIX 98: Issue 19 - Cache string size for bounds check ★
    // Problem: value.size() called twice (bounds check + substr offset)
    // Solution: Cache size to avoid repeated call
    const std::string value(path);
    const size_t val_size = value.size();
    return val_size >= 4 && value.substr(val_size - 4) == ".csv";
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
    engine->gguf_common_tensors.clear();
    engine->gguf_common_tensor_slots.clear();
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
        if (!load_juju_tensor_index_as_tensor_index(
                engine,
                engine->model_root,
                tensor_contract) &&
            !load_gguf_tensor_directory_as_tensor_index(
                engine,
                engine->model_root,
                tensor_contract)) {
            finish_tensor_index_load(engine);
        }
        std::fprintf(stderr,
            "[storagellm] offload JUJU/GGUF index loaded: files=%u tensor_headers=%llu executable_tensors=%llu runtime_records=%zu\n",
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
        // BUGFIX 94: Issue 15 - Avoid string concatenation in hot path ★★
        // Problem: path_cache_key = shard_file + "\n" + source_file creates temp strings
        // Called for every tensor row (100k+ times during model load)
        // Solution: Reserve capacity and use append to avoid reallocation
        std::string path_cache_key;
        path_cache_key.reserve(shard_file.size() + 1 + source_file.size());
        path_cache_key = shard_file;
        path_cache_key += '\n';
        path_cache_key += source_file;
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
