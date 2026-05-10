// JUJU main parsing logic
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
                    if (stream_bytes > engine->max_tensor_stream_bytes.load(std::memory_order_relaxed)) {
                        engine->max_tensor_stream_bytes.store(stream_bytes, std::memory_order_relaxed);
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
        (unsigned long long)engine->max_tensor_stream_bytes.load(std::memory_order_relaxed));
    std::fflush(stderr);
    return 1;
}

static uint64_t moe_count_json_key_occurrences(const std::string& text, const char* key, size_t start_pos = 0) {
    if (!key || !key[0]) {
        return 0;
    }
    uint64_t count = 0;
    size_t pos = start_pos;
    const size_t key_len = std::strlen(key);
    while ((pos = text.find(key, pos)) != std::string::npos) {
        ++count;
        pos += key_len;
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
    moe_ingest_juju_moe_layer_bitmask_json(engine, index_json);
    moe_ingest_juju_expert_tier_entries_json(engine, index_json);
    size_t graph_pos = index_json.find("\"graph_ir\"");
    if (graph_pos == std::string::npos ||
        index_json.find("JUJU_GRAPH_IR_V1", graph_pos) == std::string::npos) {
        return;
    }
    engine->offload_graph_ir_ready.store(1, std::memory_order_relaxed);
    if (index_json.find("\"graph_ir_required\":true") != std::string::npos ||
        index_json.find("\"graph_ir_required\": true") != std::string::npos) {
        engine->offload_graph_ir_required.store(1, std::memory_order_relaxed);
    }
    uint64_t schema_version = 1;
    moe_json_get_u64_slice(index_json, graph_pos, index_json.size(), "schema_version", &schema_version);
    engine->offload_graph_ir_op_count.fetch_add(
        moe_count_json_key_occurrences(index_json, "\"op\"", graph_pos),
        std::memory_order_relaxed);
    engine->offload_graph_ir_layer_count.fetch_add(
        moe_count_json_key_occurrences(index_json, "\"layer\"", graph_pos),
        std::memory_order_relaxed);
    if (index_json.find("\"runtime_execution_manifest\"", graph_pos) != std::string::npos &&
        index_json.find("JUJU_RUNTIME_EXECUTION_MANIFEST_V1", graph_pos) != std::string::npos) {
        engine->juju_runtime_execution_manifest_ready.store(1, std::memory_order_relaxed);
    }
    const int has_runtime_access_plan =
        index_json.find("\"runtime_access_plan\"", graph_pos) != std::string::npos &&
        index_json.find("JUJU_RUNTIME_ACCESS_PLAN_V1", graph_pos) != std::string::npos;
    if (has_runtime_access_plan) {
        engine->juju_runtime_access_plan_ready.store(1, std::memory_order_relaxed);
        uint64_t value = 0;
        if (moe_json_get_u64_slice(index_json, graph_pos, index_json.size(), "startup_hotset_count", &value)) {
            engine->juju_runtime_access_hotset_count.store(value, std::memory_order_relaxed);
        }
        if (moe_json_get_u64_slice(index_json, graph_pos, index_json.size(), "file_locality_group_count", &value)) {
            engine->juju_runtime_access_file_group_count.store(value, std::memory_order_relaxed);
        }
        if (moe_json_get_u64_slice(index_json, graph_pos, index_json.size(), "executor_tensor_count", &value)) {
            engine->juju_runtime_executor_tensor_count.store(value, std::memory_order_relaxed);
        }
        if (moe_json_get_u64_slice(index_json, graph_pos, index_json.size(), "layer_prefetch_plan_count", &value)) {
            engine->juju_runtime_layer_prefetch_plan_count.store(value, std::memory_order_relaxed);
        }
    }
    if (index_json.find("\"kv_layout_contract\"", graph_pos) != std::string::npos &&
        index_json.find("JUJU_KV_LAYOUT_CONTRACT_V1", graph_pos) != std::string::npos) {
        engine->juju_kv_layout_contract_ready.store(1, std::memory_order_relaxed);
        uint64_t value = 0;
        if (moe_json_get_u64_slice(index_json, graph_pos, index_json.size(), "max_entry_dim", &value)) {
            engine->juju_kv_layout_max_entry_dim.store(value, std::memory_order_relaxed);
        }
        if (moe_json_get_u64_slice(index_json, graph_pos, index_json.size(), "page_size_tokens", &value)) {
            engine->juju_kv_layout_page_size_tokens.store(value, std::memory_order_relaxed);
        }
    }
    if (engine->offload_graph_ir_json.empty()) {
        engine->offload_graph_ir_json = index_json.substr(graph_pos);
    }
    engine->offload_graph_ir_version.store((uint32_t)std::min<uint64_t>(schema_version, UINT32_MAX), std::memory_order_relaxed);
}

static int moe_juju_validate_required_bundle_native_experts(moe_pc_engine* engine) {
    if (!engine || !engine->io_config.require_bundle_native_experts) {
        return 1;
    }
    for (const moe_tensor_record& rec : engine->tensors) {
        if (rec.info.proj != moe_PROJ_GATE &&
            rec.info.proj != moe_PROJ_UP &&
            rec.info.proj != moe_PROJ_DOWN) {
            continue;
        }
        if (tensor_dense_slot(
                engine,
                static_cast<int>(rec.info.layer),
                static_cast<int>(rec.info.expert),
                rec.info.proj) == tensor_slot_invalid()) {
            continue;
        }
        if (rec.bundle_native) {
            continue;
        }
        std::fprintf(stderr,
            "[storagellm] JUJU MoE expert tensor is not bundle-native: layer=%u expert=%u proj=%u. "
            "Regenerate with bundle_native_required=true or disable require_bundle_native_experts for compatibility mode.\n",
            (unsigned)rec.info.layer,
            (unsigned)rec.info.expert,
            (unsigned)rec.info.proj);
        return 0;
    }
    return 1;
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
    uint64_t common_records = 0;
    moe_juju_runtime_probe_t probe{};
    std::string model_config_json;
    const int has_model_config_json = moe_read_model_config_json(root, &model_config_json);
    if (has_model_config_json) {
        moe_juju_probe_note_model_config_json(engine, &probe, model_config_json);
    }
    for (const std::string& path : paths) {
        moe_juju_header_t header{};
        std::vector<moe_juju_section_entry_t> sections;
        uint64_t file_bytes = 0;
        if (!moe_juju_read_table_local(path, &header, &sections, &file_bytes)) {
            continue;
        }
        moe_ingest_juju_runtime_metadata_sections(engine, path, sections, file_bytes);
        moe_juju_probe_note_header(&probe, header);
        std::string index_json;
        if (!moe_read_juju_tensor_index_json(path, sections, file_bytes, &index_json)) {
            continue;
        }
        if (!moe_juju_index_schema_supported(index_json)) {
            std::fprintf(stderr,
                "[storagellm] stale JUJU artifact rejected: %s. Regenerate .juju/.juju.idx with JUJU v2 bundle-native Colab notebooks.\n",
                path.c_str());
            return 0;
        }
        moe_ingest_juju_attention_contract(engine, index_json);
        moe_juju_probe_note_index_json(&probe, index_json);
        std::vector<moe_juju_tensor_json_entry_t> tensors;
        if (!moe_parse_juju_tensor_index_json(index_json, &tensors, nullptr, nullptr, nullptr)) {
            continue;
        }
        for (const moe_juju_tensor_json_entry_t& entry : tensors) {
            moe_juju_probe_note_tensor(&probe, entry);
        }
    }
    moe_juju_probe_apply(engine, &probe);
    engine->expert_states.assign(expert_slot_count(engine), moe_block_state{});
    tensor_slots_reset(engine);
    engine->tensors.reserve(tensor_record_capacity(engine));
    probe = moe_juju_runtime_probe_t{};
    if (has_model_config_json) {
        moe_juju_probe_note_model_config_json(engine, &probe, model_config_json);
    }
    for (const std::string& path : paths) {
        moe_juju_header_t header{};
        std::vector<moe_juju_section_entry_t> sections;
        uint64_t file_bytes = 0;
        if (!moe_juju_read_table_local(path, &header, &sections, &file_bytes)) {
            continue;
        }
        moe_ingest_juju_runtime_metadata_sections(engine, path, sections, file_bytes);
        moe_juju_probe_note_header(&probe, header);
        std::string index_json;
        if (!moe_read_juju_tensor_index_json(path, sections, file_bytes, &index_json)) {
            continue;
        }
        if (!moe_juju_index_schema_supported(index_json)) {
            std::fprintf(stderr,
                "[storagellm] stale JUJU artifact rejected: %s. Regenerate .juju/.juju.idx with JUJU v2 bundle-native Colab notebooks.\n",
                path.c_str());
            return 0;
        }
        moe_ingest_juju_attention_contract(engine, index_json);
        moe_juju_probe_note_index_json(&probe, index_json);
        moe_ingest_juju_graph_ir(engine, index_json);
        std::vector<moe_juju_tensor_json_entry_t> tensors;
        if (!moe_parse_juju_tensor_index_json(index_json, &tensors, nullptr, nullptr, nullptr) ||
            tensors.empty()) {
            continue;
        }
        for (const moe_juju_tensor_json_entry_t& entry : tensors) {
            moe_juju_probe_note_tensor(&probe, entry);
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
            if (!moe_juju_tensor_codec_contract_valid(entry, contract)) {
                std::fprintf(stderr,
                    "[storagellm] JUJU tensor codec contract rejected: name=%s gguf_type=%u codec_id=%s registry=%s weight_encoding=%u\n",
                    entry.name.c_str(),
                    (unsigned)entry.gguf_type,
                    entry.codec_id.c_str(),
                    entry.codec_registry_version.c_str(),
                    (unsigned)entry.weight_encoding);
                return 0;
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
            moe_projection_t bundle_projection_probe = moe_PROJ_GATE;
            if (entry.bundle_native &&
                !moe_juju_bundle_member_projection(entry, &bundle_projection_probe)) {
                continue;
            }
            if (moe_gguf_tensor_name_executable(entry.name)) {
                moe_gguf_tensor_dir_entry_t common{};
                common.name = entry.name;
                common.dims = entry.dims;
                common.type = entry.gguf_type;
                common.weight_encoding = moe_weight_encoding_from_juju_entry(entry, contract);
                common.file_offset = entry.juju_offset;
                common.bytes = moe_effective_juju_tensor_bytes(entry, contract);
                common.weight_row_bytes = entry.row_stride_bytes ? entry.row_stride_bytes : entry.row_bytes;
                for (uint32_t i = 0; i < std::min<uint32_t>(entry.dims, 4u); ++i) {
                    common.shape[i] = entry.shape[i];
                }
                moe_register_gguf_common_tensor(engine, common, path_idx);
                ++common_records;
                auto common_it = engine->gguf_common_tensor_slots.find(moe_lower_ascii_copy(entry.name));
                if (common_it != engine->gguf_common_tensor_slots.end() &&
                    common_it->second < engine->gguf_common_tensors.size()) {
                    moe_gguf_common_tensor_record& common_rec = engine->gguf_common_tensors[common_it->second];
                    common_rec.op_role = moe_lower_ascii_copy(!entry.op_role.empty() ? entry.op_role : entry.execution_op);
                    common_rec.codec_id = entry.codec_id;
                    common_rec.codec_registry_version = entry.codec_registry_version;
                    common_rec.storage_layout = entry.storage_layout;
                    common_rec.source_dims = std::min<uint32_t>(entry.source_dims, 4u);
                    for (uint32_t i = 0; i < common_rec.source_dims; ++i) {
                        common_rec.source_shape[i] = entry.source_shape[i];
                    }
                    common_rec.execution_layer = entry.execution_layer;
                    common_rec.math_rows = entry.math_out_features ?
                        (uint32_t)std::min<uint64_t>(entry.math_out_features, UINT32_MAX) : 0u;
                    common_rec.math_cols = entry.math_in_features ?
                        (uint32_t)std::min<uint64_t>(entry.math_in_features, UINT32_MAX) : 0u;
                    common_rec.math_transposed = entry.math_transposed;
                    common_rec.runtime_priority = entry.runtime_priority;
                    common_rec.prefetch_priority = entry.prefetch_priority;
                    common_rec.execution_order = entry.execution_order;
                    common_rec.hotset_rank = entry.hotset_rank;
                    common_rec.residency_hint = entry.residency_hint;
                    common_rec.graph_role = entry.graph_role;
                    common_rec.prefetch_class = entry.prefetch_class;
                }
            }
            const bool combined_gate_up = moe_juju_entry_is_combined_gate_up(entry) && !entry.bundle_native;
            uint32_t layer = 0;
            moe_projection_t proj = moe_PROJ_GATE;
            const int explicit_projection = moe_juju_entry_projection(entry, &proj);
            if (!moe_juju_entry_layer(entry, &layer) ||
                (!combined_gate_up && !explicit_projection)) {
                continue;
            }
            const uint64_t expert_count64 = moe_juju_entry_expert_count(entry);
            if (expert_count64 == 0) {
                continue;
            }
            const uint64_t effective_entry_bytes = moe_effective_juju_tensor_bytes(entry, contract);
            const uint64_t per_expert = entry.per_expert_bytes ? entry.per_expert_bytes :
                (expert_count64 ? effective_entry_bytes / expert_count64 : 0);
            const uint64_t expert_base_offset = entry.expert_base_offset ? entry.expert_base_offset : entry.juju_offset;
            if (per_expert == 0 || expert_count64 > 4096ull) {
                continue;
            }
            const uint64_t entry_end = entry.juju_offset + entry.juju_bytes;
            if (expert_base_offset < entry.juju_offset ||
                expert_base_offset > file_bytes ||
                expert_base_offset > entry_end ||
                expert_count64 > (UINT64_MAX - expert_base_offset) / per_expert) {
                std::fprintf(stderr,
                    "[storagellm] invalid JUJU expert layout range: name=%s base=%llu per_expert=%llu experts=%llu file=%llu\n",
                    entry.name.c_str(),
                    (unsigned long long)expert_base_offset,
                    (unsigned long long)per_expert,
                    (unsigned long long)expert_count64,
                    (unsigned long long)file_bytes);
                return 0;
            }
            const uint64_t expert_total_bytes = expert_count64 * per_expert;
            if (expert_base_offset + expert_total_bytes > file_bytes ||
                expert_base_offset + expert_total_bytes > entry_end) {
                std::fprintf(stderr,
                    "[storagellm] invalid JUJU expert layout range: name=%s base=%llu per_expert=%llu experts=%llu file=%llu\n",
                    entry.name.c_str(),
                    (unsigned long long)expert_base_offset,
                    (unsigned long long)per_expert,
                    (unsigned long long)expert_count64,
                    (unsigned long long)file_bytes);
                return 0;
            }
            if (combined_gate_up && !entry.combined_gate_up_split_enabled &&
                entry.bundle_member_role.empty()) {
                std::fprintf(stderr,
                    "[storagellm] JUJU combined gate/up tensor missing explicit split policy: name=%s\n",
                    entry.name.c_str());
                return 0;
            }
            if (combined_gate_up) {
                const std::string split_policy = moe_lower_ascii_copy(entry.split_policy);
                if (!split_policy.empty() && split_policy != "contiguous_rows") {
                    std::fprintf(stderr,
                        "[storagellm] JUJU combined gate/up tensor has unsupported split policy: name=%s policy=%s\n",
                        entry.name.c_str(),
                        entry.split_policy.c_str());
                    return 0;
                }
            }
            const uint64_t math_rows_total64 = moe_juju_entry_rows_total(entry);
            const uint64_t math_cols64 = moe_juju_entry_cols(entry);
            if (math_rows_total64 == 0 || math_cols64 == 0) {
                std::fprintf(stderr,
                    "[storagellm] JUJU expert tensor missing math shape: name=%s rows=%llu cols=%llu\n",
                    entry.name.c_str(),
                    (unsigned long long)math_rows_total64,
                    (unsigned long long)math_cols64);
                return 0;
            }
            if (combined_gate_up &&
                ((math_rows_total64 & 1ull) != 0 || per_expert < 2ull)) {
                std::fprintf(stderr,
                    "[storagellm] JUJU combined gate/up tensor has invalid split shape: name=%s rows=%llu per_expert=%llu\n",
                    entry.name.c_str(),
                    (unsigned long long)math_rows_total64,
                    (unsigned long long)per_expert);
                return 0;
            }
            const uint32_t rows_total = (uint32_t)std::min<uint64_t>(math_rows_total64, UINT32_MAX);
            const uint32_t cols = (uint32_t)std::min<uint64_t>(math_cols64, UINT32_MAX);
            for (uint64_t expert = 0; expert < expert_count64; ++expert) {
                const uint32_t split_count = combined_gate_up ? 2u : 1u;
                for (uint32_t split = 0; split < split_count; ++split) {
                    moe_tensor_record rec{};
                    rec.info.part = part_id;
                    rec.info.shard = shard_id;
                    rec.info.layer = layer;
                    rec.info.expert = entry.has_explicit_expert ?
                        entry.explicit_expert :
                        (uint32_t)expert;
                    rec.info.proj = combined_gate_up ?
                        (split == 0 ? moe_PROJ_GATE : moe_PROJ_UP) : proj;
                    rec.info.rows = combined_gate_up ? (rows_total / 2u) : rows_total;
                    rec.info.cols = cols;
                    rec.info.groups = cols >= 16u ? std::max<uint32_t>(1u, cols / 16u) : 1u;
                    rec.info.group_size = cols >= 16u ? 16u : cols;
                    const uint64_t expert_offset = expert_base_offset + expert * per_expert;
                    uint64_t split_rel_offset = 0;
                    uint64_t split_bytes = per_expert;
                    if (combined_gate_up) {
                        if (entry.combined_gate_up_split_enabled &&
                            entry.gate_bytes > 0 && entry.up_bytes > 0 &&
                            entry.gate_rel_offset <= per_expert &&
                            entry.up_rel_offset <= per_expert &&
                            entry.gate_bytes <= per_expert - entry.gate_rel_offset &&
                            entry.up_bytes <= per_expert - entry.up_rel_offset) {
                            split_rel_offset = split == 0u ? entry.gate_rel_offset : entry.up_rel_offset;
                            split_bytes = split == 0u ? entry.gate_bytes : entry.up_bytes;
                        } else {
                            std::fprintf(stderr,
                                "[storagellm] JUJU combined gate/up tensor missing explicit split offsets: name=%s\n",
                                entry.name.c_str());
                            return 0;
                        }
                    }
                    rec.info.weight_byte_offset = expert_offset + split_rel_offset;
                    rec.info.weight_byte_length = split_bytes;
                    rec.info.scale_byte_offset = 0;
                    rec.info.scale_byte_length = 0;
                    rec.info.scale2_byte_offset = 0;
                    rec.info.scale2_byte_length = 0;
                    rec.resolved_path_index = path_idx;
                    rec.runtime_priority = entry.runtime_priority;
                    rec.prefetch_priority = entry.prefetch_priority;
                    rec.execution_order = entry.execution_order;
                    rec.hotset_rank = entry.hotset_rank;
                    rec.residency_hint = entry.residency_hint;
                    rec.graph_role = entry.graph_role;
                    rec.prefetch_class = entry.prefetch_class;
                    rec.juju_section_type = entry.juju_section_type;
                    rec.bundle_native = entry.bundle_native;
                    rec.bundle_member_role = moe_juju_bundle_member_role_id(entry.bundle_member_role);
                    rec.bundle_id = entry.bundle_id;
                    rec.bundle_offset = entry.bundle_offset;
                    rec.bundle_size = entry.bundle_size;
                    rec.bundle_member_offset = entry.bundle_member_offset;
                    rec.bundle_member_size = entry.bundle_member_size;
                    rec.bundle_alignment = entry.bundle_alignment;
                    rec.expert_gpu_layout_kind = moe_juju_gpu_layout_kind_id(entry.expert_gpu_layout);
                    rec.expert_gpu_layout_offset = entry.expert_gpu_layout_offset;
                    rec.expert_gpu_layout_size = entry.expert_gpu_layout_size;
                    rec.expert_gpu_layout_row_bytes = entry.expert_gpu_layout_row_bytes;
                    const uint8_t rec_encoding = moe_weight_encoding_from_juju_entry(entry, contract);
                    finalize_tensor_layout(&rec, rec_encoding);
                    if (entry.row_stride_bytes >= rec.weight_row_bytes &&
                        rec.info.rows > 0 &&
                        entry.row_stride_bytes <= rec.info.weight_byte_length &&
                        entry.row_stride_bytes <= UINT64_MAX / (uint64_t)rec.info.rows &&
                        (uint64_t)rec.info.rows * entry.row_stride_bytes <= rec.info.weight_byte_length) {
                        rec.weight_row_bytes = entry.row_stride_bytes;
                    }
                    if (rec.weight_format == moe_WEIGHT_ENCODING_UNKNOWN ||
                        rec.weight_row_bytes == 0 ||
                        (!rec.cpu_executable && !rec.gpu_executable)) {
                        std::fprintf(stderr,
                            "[storagellm] unsupported JUJU tensor kernel: name=%s encoding=%u gguf_type=%u layer=%u expert=%u proj=%u bytes=%llu\n",
                            entry.name.c_str(),
                            (unsigned)rec_encoding,
                            (unsigned)entry.gguf_type,
                            (unsigned)rec.info.layer,
                            (unsigned)rec.info.expert,
                            (unsigned)rec.info.proj,
                            (unsigned long long)rec.info.weight_byte_length);
                        return 0;
                    }
                    const uint64_t stream_bytes = tensor_stream_bytes(rec);
                    if (stream_bytes > engine->max_tensor_stream_bytes.load(std::memory_order_relaxed)) {
                        engine->max_tensor_stream_bytes.store(stream_bytes, std::memory_order_relaxed);
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
                    expert_segment_hints_update(
                        engine,
                        (int)rec.info.layer,
                        (int)rec.info.expert,
                        entry.segment_policy,
                        entry.can_partial_exec,
                        entry.partial_accuracy_milli);
                    if (entry.can_partial_exec) {
                        engine->juju_partial_expert_tensor_count.fetch_add(1, std::memory_order_relaxed);
                    }
                    engine->tensors.push_back(std::move(rec));
                    ++records;
                }
            }
        }
    }
    if (!records && !common_records) {
        return 0;
    }
    if (!moe_juju_validate_required_bundle_native_experts(engine)) {
        return 0;
    }
    bool saw_bundle_native = false;
    bool all_bundle_4k_aligned = true;
    for (const moe_tensor_record& rec : engine->tensors) {
        if (!rec.bundle_native) {
            continue;
        }
        saw_bundle_native = true;
        if (rec.bundle_size == 0 ||
            ((rec.bundle_offset | rec.bundle_size) & 4095ull) != 0 ||
            (rec.bundle_alignment && rec.bundle_alignment != 4096ull)) {
            all_bundle_4k_aligned = false;
            break;
        }
    }
    engine->juju_bundle_4k_aligned.store(
        saw_bundle_native && all_bundle_4k_aligned,
        std::memory_order_release);
    moe_juju_probe_apply(engine, &probe);
    finish_tensor_index_load(engine);
    moe_dynamic_shape_finalize_adapter(engine);
    std::fprintf(stderr,
        "[storagellm] JUJU tensor index loaded: records=%llu common_records=%llu paths=%zu max_stream=%llu\n",
        (unsigned long long)records,
        (unsigned long long)common_records,
        engine->tensor_paths.size(),
        (unsigned long long)engine->max_tensor_stream_bytes.load(std::memory_order_relaxed));
    std::fflush(stderr);
    return 1;
}
static int path_ends_with_csv(const char* path) {
    if (!path) {
        return 0;
    }
    // BUGFIX 98: Issue 19 - Cache string size for bounds check ??
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

    // BUGFIX 133: Stop generation before clearing tensor records ?끸쁾
    // Problem: moe_pc_engine_wait_io() stops IO workers but not generation
    // Generation thread can still access engine->tensors during clear()
    // ??iterator invalidation, use-after-free, segfault
    // Solution: Set generation phase to IDLE before clearing tensors
    // Ensures no generation thread accesses tensors during model reload
    engine->generation_phase.store(moe_GEN_PHASE_IDLE, std::memory_order_release);

    moe_pc_engine_wait_io(engine);
    device_free_pool(engine, moe_DEVICE_POOL_FIXED);
    device_free_pool(engine, moe_DEVICE_POOL_EXPERT);
    device_free_pool(engine, moe_DEVICE_POOL_ACTIVATION);
    device_free_pool(engine, moe_DEVICE_POOL_STREAM);
    engine->expert_states.assign(expert_slot_count(engine), moe_block_state{});
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
    {
        std::unique_lock<std::shared_mutex> lock(engine->rmsnorm_unit_offset_cache_mutex);
        engine->rmsnorm_unit_offset_cache.clear();
    }
    moe_engine_reset_qkv_cache(engine);
    moe_engine_clear_offload_gguf_contract(engine);
    engine->tensors.clear();
    engine->gguf_common_tensors.clear();
    engine->gguf_common_tensor_slots.clear();
    engine->tensor_paths.clear();
    engine->max_tensor_stream_bytes.store(0, std::memory_order_relaxed);
    engine->common_raw_prefetched_bytes.store(0);
    engine->common_raw_prefetched_spans.store(0);
    engine->tensors.reserve(tensor_record_capacity(engine));
    tensor_slots_reset(engine);
    engine->model_root = model_root ? model_root : "";
    moe_model_root_check_t root_check{};
    int root_valid = 0;
    if (!engine->model_root.empty()) {
        if (moe_engine_load_offload_gguf_contract(engine, engine->model_root, &root_check)) {
            root_valid = root_check.valid;
            moe_append_model_config_metadata(engine, engine->model_root);
            moe_append_model_root_runtime_sidecars(engine, engine->model_root);
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
        std::vector<std::string> juju_probe_paths;
        moe_collect_juju_files(engine->model_root, 4, &juju_probe_paths);
        const bool has_juju_artifacts = !juju_probe_paths.empty();
        const int juju_loaded = load_juju_tensor_index_as_tensor_index(
            engine,
            engine->model_root,
            tensor_contract);
        if (!juju_loaded && has_juju_artifacts) {
            std::fprintf(stderr,
                "[storagellm] JUJU artifacts are present but tensor index loading failed; refusing GGUF fallback.\n");
            std::fflush(stderr);
            return 0;
        }
        if (!juju_loaded &&
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
        // BUGFIX 94: Issue 15 - Avoid string concatenation in hot path ?끸쁾
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
        finalize_tensor_layout(&rec, moe_WEIGHT_ENCODING_UNKNOWN);
        const uint64_t stream_bytes = tensor_stream_bytes(rec);
        if (stream_bytes > engine->max_tensor_stream_bytes.load(std::memory_order_relaxed)) {
            engine->max_tensor_stream_bytes.store(stream_bytes, std::memory_order_relaxed);
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
