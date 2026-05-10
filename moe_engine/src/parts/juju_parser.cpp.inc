// JUJU 포맷 파싱 유틸리티
// JUJU 텐서 인덱스, 메타데이터, 섹션 읽기

struct moe_juju_tensor_json_entry_t {
    std::string name;
    std::string bucket;
    std::string graph_role_text;
    std::string prefetch_class_text;
    std::string residency_hint_text;
    std::string codec_id;
    std::string codec_registry_version;
    std::string storage_layout;
    std::string op_role;
    std::string execution_op;
    std::string split_policy;
    uint64_t shape[4]{};
    uint32_t dims = 0;
    uint64_t source_shape[4]{};
    uint32_t source_dims = 0;
    uint64_t math_out_features = 0;
    uint64_t math_in_features = 0;
    uint8_t math_transposed = 0;
    int32_t execution_layer = INT32_MIN;
    uint32_t gguf_type = 0;
    uint8_t has_gguf_type = 0;
    uint8_t weight_encoding = moe_WEIGHT_ENCODING_UNKNOWN;
    uint64_t juju_offset = 0;
    uint64_t juju_bytes = 0;
    uint64_t source_bytes = 0;
    uint64_t logical_bytes = 0;
    uint64_t physical_bytes = 0;
    uint64_t logical_rows = 0;
    uint64_t logical_cols = 0;
    uint64_t row_bytes = 0;
    uint64_t row_stride_bytes = 0;
    uint64_t row_padding_bytes = 0;
    uint16_t runtime_priority = 50;
    uint16_t prefetch_priority = 50;
    uint16_t execution_order = 500;
    uint16_t hotset_rank = 1000;
    uint8_t residency_hint = moe_JUJU_RESIDENCY_AUTO;
    uint8_t graph_role = moe_JUJU_ROLE_UNKNOWN;
    uint8_t prefetch_class = moe_JUJU_PREFETCH_AUTO;
    uint32_t juju_section_type = 0;
    uint8_t segment_policy = moe_JUJU_SEGMENT_POLICY_AUTO;
    uint8_t can_partial_exec = 0;
    uint16_t partial_accuracy_milli = 0;
    uint8_t has_expert_layout = 0;
    uint32_t expert_count = 0;
    uint64_t expert_base_offset = 0;
    uint64_t per_expert_bytes = 0;
    uint8_t combined_gate_up_split_enabled = 0;
    uint64_t gate_rel_offset = 0;
    uint64_t gate_bytes = 0;
    uint64_t up_rel_offset = 0;
    uint64_t up_bytes = 0;
    uint8_t bundle_native = 0;
    uint32_t bundle_id = 0;
    uint32_t explicit_expert = 0;
    uint8_t has_explicit_expert = 0;
    std::string bundle_member_role;
    std::string expert_projection;
    uint64_t bundle_offset = 0;
    uint64_t bundle_size = 0;
    uint64_t bundle_member_offset = 0;
    uint64_t bundle_member_size = 0;
    uint64_t bundle_alignment = 0;
    std::string expert_gpu_layout;
    uint64_t expert_gpu_layout_offset = 0;
    uint64_t expert_gpu_layout_size = 0;
    uint64_t expert_gpu_layout_row_bytes = 0;
};

static uint8_t moe_juju_gpu_layout_kind_id(const std::string& text) {
    const std::string v = moe_lower_ascii_copy(text);
    if (v == "fp16_rowmajor" || v == "f16_rowmajor") return 1;
    if (v == "bf16_rowmajor") return 2;
    if (v == "fp32_rowmajor" || v == "f32_rowmajor") return 3;
    if (v == "fp16_col32" || v == "f16_col32" || v == "col32") return 4;
    if (v == "fp8_col32" || v == "e4m3_col32" || v == "e5m2_col32") return 5;
    return 0;
}

static int moe_juju_bundle_member_projection(
    const moe_juju_tensor_json_entry_t& entry,
    moe_projection_t* proj
) {
    if (!proj) {
        return 0;
    }
    const std::string role = moe_lower_ascii_copy(entry.bundle_member_role);
    if (role == "gate") {
        *proj = moe_PROJ_GATE;
        return 1;
    }
    if (role == "up") {
        *proj = moe_PROJ_UP;
        return 1;
    }
    if (role == "down") {
        *proj = moe_PROJ_DOWN;
        return 1;
    }
    const std::string explicit_proj = moe_lower_ascii_copy(entry.expert_projection);
    if (explicit_proj == "gate" && role.empty()) {
        *proj = moe_PROJ_GATE;
        return 1;
    }
    if (explicit_proj == "up" && role.empty()) {
        *proj = moe_PROJ_UP;
        return 1;
    }
    if (explicit_proj == "down" && role.empty()) {
        *proj = moe_PROJ_DOWN;
        return 1;
    }
    return 0;
}

static int moe_juju_entry_layer(
    const moe_juju_tensor_json_entry_t& entry,
    uint32_t* layer
) {
    if (!layer) {
        return 0;
    }
    if (entry.execution_layer >= 0) {
        *layer = (uint32_t)entry.execution_layer;
        return 1;
    }
    return moe_parse_layer_from_gguf_name(entry.name, layer);
}

static int moe_juju_projection_from_text(const std::string& text, moe_projection_t* proj) {
    if (!proj) {
        return 0;
    }
    const std::string v = moe_lower_ascii_copy(text);
    if (v.empty()) {
        return 0;
    }
    if (v == "gate" || v == "w1" || v == "ffn_gate" || v == "gate_proj") {
        *proj = moe_PROJ_GATE;
        return 1;
    }
    if (v == "up" || v == "w3" || v == "ffn_up" || v == "up_proj") {
        *proj = moe_PROJ_UP;
        return 1;
    }
    if (v == "down" || v == "w2" || v == "ffn_down" || v == "down_proj") {
        *proj = moe_PROJ_DOWN;
        return 1;
    }
    return 0;
}

static int moe_juju_entry_projection(
    const moe_juju_tensor_json_entry_t& entry,
    moe_projection_t* proj
) {
    if (!proj) {
        return 0;
    }
    if (moe_juju_bundle_member_projection(entry, proj)) {
        return 1;
    }
    if (moe_juju_projection_from_text(entry.expert_projection, proj)) {
        return 1;
    }
    return moe_gguf_expert_projection(entry.name, proj);
}

static int moe_juju_entry_is_combined_gate_up(const moe_juju_tensor_json_entry_t& entry) {
    const std::string explicit_proj = moe_lower_ascii_copy(entry.expert_projection);
    const std::string role = moe_lower_ascii_copy(entry.bundle_member_role);
    if (explicit_proj == "gate_up" || explicit_proj == "gate+up" || explicit_proj == "gate_up_fused") {
        return 1;
    }
    if (entry.combined_gate_up_split_enabled && entry.gate_bytes > 0 && entry.up_bytes > 0) {
        return 1;
    }
    if (role == "gate_up" || role == "gate+up") {
        return 1;
    }
    const std::string lower = moe_lower_ascii_copy(entry.name);
    return lower.find("ffn_gate_up_exps.weight") != std::string::npos ||
           lower.find("gate_up") != std::string::npos ||
           lower.find("gate-up") != std::string::npos ||
           lower.find("w1_w3") != std::string::npos ||
           lower.find("w13") != std::string::npos;
}

static uint64_t moe_juju_entry_expert_count(const moe_juju_tensor_json_entry_t& entry) {
    if (entry.bundle_native && entry.has_explicit_expert) {
        return 1ull;
    }
    if (entry.has_explicit_expert && entry.expert_count <= 1u && entry.dims <= 2u) {
        return 1ull;
    }
    if (entry.expert_count) {
        return (uint64_t)entry.expert_count;
    }
    if (entry.dims >= 3u && entry.shape[2]) {
        return entry.shape[2];
    }
    return 0ull;
}

static uint64_t moe_juju_entry_rows_total(const moe_juju_tensor_json_entry_t& entry) {
    if (entry.math_out_features) return entry.math_out_features;
    if (entry.logical_rows) return entry.logical_rows;
    if (entry.dims >= 2u) return entry.shape[1];
    return 0ull;
}

static uint64_t moe_juju_entry_cols(const moe_juju_tensor_json_entry_t& entry) {
    if (entry.math_in_features) return entry.math_in_features;
    if (entry.logical_cols) return entry.logical_cols;
    if (entry.dims >= 1u) return entry.shape[0];
    return 0ull;
}

static uint8_t moe_juju_bundle_member_role_id(const std::string& role_text) {
    const std::string role = moe_lower_ascii_copy(role_text);
    if (role == "gate") return 0;
    if (role == "gate_scale") return 1;
    if (role == "gate_scale2") return 2;
    if (role == "up") return 3;
    if (role == "up_scale") return 4;
    if (role == "up_scale2") return 5;
    if (role == "down") return 6;
    if (role == "down_scale") return 7;
    if (role == "down_scale2") return 8;
    return 255;
}

static uint8_t moe_juju_residency_hint_from_text(const std::string& text) {
    const std::string value = moe_lower_ascii_copy(text);
    if (value.find("fast_mem_streamable") != std::string::npos ||
        value.find("streamable") != std::string::npos) {
        return moe_JUJU_RESIDENCY_FAST_MEM_STREAMABLE;
    }
    if (value.find("fast_mem") != std::string::npos ||
        value.find("compute_mem") != std::string::npos ||
        value == "vram") {
        return moe_JUJU_RESIDENCY_FAST_MEM;
    }
    if (value.find("slow_mem") != std::string::npos ||
        value.find("storage") != std::string::npos ||
        value == "db") {
        return moe_JUJU_RESIDENCY_SLOW_MEM;
    }
    return moe_JUJU_RESIDENCY_AUTO;
}

static uint8_t moe_juju_graph_role_from_text(const std::string& text) {
    const std::string value = moe_lower_ascii_copy(text);
    if (value.find("shared_core") != std::string::npos) {
        return moe_JUJU_ROLE_SHARED_CORE;
    }
    if (value.find("attention") != std::string::npos) {
        return moe_JUJU_ROLE_ATTENTION;
    }
    if (value.find("router") != std::string::npos) {
        return moe_JUJU_ROLE_ROUTER;
    }
    if (value.find("expert") != std::string::npos) {
        return moe_JUJU_ROLE_EXPERT;
    }
    if (value.find("dense_ffn") != std::string::npos || value.find("dense") != std::string::npos) {
        return moe_JUJU_ROLE_DENSE_FFN;
    }
    if (value.find("norm") != std::string::npos) {
        return moe_JUJU_ROLE_NORM;
    }
    return moe_JUJU_ROLE_UNKNOWN;
}

static uint8_t moe_juju_prefetch_class_from_text(const std::string& text) {
    const std::string value = moe_lower_ascii_copy(text);
    if (value.find("startup_hot") != std::string::npos) {
        return moe_JUJU_PREFETCH_STARTUP_HOT;
    }
    if (value.find("router_hot") != std::string::npos) {
        return moe_JUJU_PREFETCH_ROUTER_HOT;
    }
    if (value.find("layer_hot") != std::string::npos) {
        return moe_JUJU_PREFETCH_LAYER_HOT;
    }
    if (value.find("layer_warm") != std::string::npos) {
        return moe_JUJU_PREFETCH_LAYER_WARM;
    }
    if (value.find("expert_stream") != std::string::npos) {
        return moe_JUJU_PREFETCH_EXPERT_STREAM;
    }
    if (value.find("stream") != std::string::npos) {
        return moe_JUJU_PREFETCH_STREAM;
    }
    return moe_JUJU_PREFETCH_AUTO;
}

static std::string moe_juju_path_basename_local(std::string path) {
    if (path.empty()) {
        return path;
    }
    while (!path.empty() && (path.back() == '/' || path.back() == '\\')) {
        path.pop_back();
    }
    const size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? path : path.substr(slash + 1u);
}

static int moe_juju_index_matches_weight_file(const std::string& index_json, const std::string& juju_path) {
    std::string weight_file;
    if (!moe_json_get_string_local(index_json, "weight_file", &weight_file) || weight_file.empty()) {
        return 0;
    }
    if (index_json.find("\"JUJU_IDX_JSON_V2\"") != std::string::npos) {
        std::string artifact_uid;
        if (!moe_json_get_string_local(index_json, "artifact_uid", &artifact_uid) || artifact_uid.empty()) {
            return 0;
        }
    }
    const std::string expected_weight = moe_juju_path_basename_local(juju_path);
    if (moe_juju_path_basename_local(weight_file) == expected_weight) {
        return 1;
    }
    // Some repair sidecars carry both weight_file and index_file.  Accept the
    // index basename as a secondary guard so path-prefixed metadata generated on
    // Colab/Linux still works on Windows/local engine paths.
    std::string index_file;
    if (moe_json_get_string_local(index_json, "index_file", &index_file) && !index_file.empty()) {
        return moe_juju_path_basename_local(index_file) == expected_weight + ".idx";
    }
    return 0;
}

static int moe_read_binary_range_to_string(
    const std::string& path,
    uint64_t offset,
    uint64_t bytes,
    uint64_t file_bytes,
    std::string* out
) {
    // BUGFIX 326: Check offset + bytes doesn't overflow before comparison ★
    // Problem: offset + bytes can overflow uint64_t if values are malformed
    // Causes offset > file_bytes check to pass incorrectly
    // Solution: Check bytes <= file_bytes - offset instead
    if (!out || path.empty() || bytes == 0 || bytes > 256ull * 1024ull * 1024ull ||
        offset > file_bytes || bytes > file_bytes - offset) {
        return 0;
    }
    std::ifstream input;
    open_ifstream_path(input, path, std::ios::in | std::ios::binary);
    if (!input) {
        return 0;
    }
    // BUGFIX 327: Check offset fits in streamoff before seekg ★
    // Problem: offset can exceed std::streamoff max value
    // seekg with out-of-range offset causes undefined behavior
    // Solution: Check offset <= streamoff max before seekg
    if (offset > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
        return 0;
    }
    input.seekg((std::streamoff)offset, std::ios::beg);
    if (!input) {
        return 0;
    }
    std::string data((size_t)bytes, '\0');
    input.read(&data[0], (std::streamsize)data.size());
    // BUGFIX 328: Check gcount() matches requested bytes ★
    // Problem: Partial read not detected, returns incomplete data
    // Solution: Check gcount() == bytes before returning success
    if ((uint64_t)input.gcount() != bytes) {
        return 0;
    }
    *out = std::move(data);
    return 1;
}

static int moe_juju_read_table_local(
    const std::string& path,
    moe_juju_header_t* header,
    std::vector<moe_juju_section_entry_t>* entries,
    uint64_t* file_bytes
) {
    if (!header || !entries || path.empty()) {
        return 0;
    }
    uint64_t actual_bytes = 0;
    if (!file_size_bytes(path, &actual_bytes) || actual_bytes < MOE_JUJU_HEADER_BYTES) {
        return 0;
    }
    if (!moe_juju_read_header(path.c_str(), header)) {
        return 0;
    }
    const uint64_t declared_bytes = header->file_size;
    const uint64_t table_bytes = (uint64_t)header->section_count * sizeof(moe_juju_section_entry_t);
    if (declared_bytes == 0 || declared_bytes > actual_bytes || header->section_count == 0 ||
        header->header_size != MOE_JUJU_HEADER_BYTES ||
        header->section_table_offset < MOE_JUJU_HEADER_BYTES ||
        header->section_table_offset > declared_bytes ||
        header->section_table_size != table_bytes ||
        header->section_table_size > declared_bytes - header->section_table_offset) {
        return 0;
    }
    entries->assign(header->section_count, moe_juju_section_entry_t{});
    uint32_t out_count = 0;
    if (!moe_juju_read_section_table(
            path.c_str(),
            header,
            entries->data(),
            (uint32_t)entries->size(),
            &out_count) ||
        out_count != header->section_count) {
        entries->clear();
        return 0;
    }
    for (const moe_juju_section_entry_t& entry : *entries) {
        if (entry.size == 0) {
            continue;
        }
        // BUGFIX 329: Check entry.offset + entry.size doesn't overflow ★
        // Problem: entry.offset + entry.size can overflow uint64_t
        // Solution: Check entry.size <= declared_bytes - entry.offset
        const uint64_t table_end = header->section_table_offset + header->section_table_size;
        if (entry.type == 0 ||
            entry.offset < table_end ||
            entry.offset > declared_bytes ||
            entry.size > declared_bytes - entry.offset) {
            entries->clear();
            return 0;
        }
    }
    if (file_bytes) {
        *file_bytes = declared_bytes;
    }
    return 1;
}

static int moe_read_juju_json_section(
    const std::string& path,
    const std::vector<moe_juju_section_entry_t>& entries,
    uint32_t section_type,
    uint64_t file_bytes,
    std::string* out
) {
    if (out) {
        out->clear();
    }
    if (!out) {
        return 0;
    }
    for (const moe_juju_section_entry_t& entry : entries) {
        if (entry.type != section_type || entry.size == 0) {
            continue;
        }
        return moe_read_binary_range_to_string(path, entry.offset, entry.size, file_bytes, out);
    }
    return 0;
}

static uint32_t moe_max_u32_local(uint32_t a, uint32_t b) {
    return a > b ? a : b;
}

static uint64_t moe_saturating_add_u64_local(uint64_t a, uint64_t b) {
    return b > UINT64_MAX - a ? UINT64_MAX : a + b;
}

static void moe_ingest_juju_section_table_hints(
    moe_pc_engine* engine,
    const std::vector<moe_juju_section_entry_t>& sections
) {
    if (!engine || sections.empty()) {
        return;
    }
    const uint32_t had_hints =
        engine->juju_io_section_hints_ready.load(std::memory_order_acquire);
    uint32_t shared_mmap = had_hints ? engine->juju_shared_mmap_friendly.load(std::memory_order_acquire) : 1u;
    uint32_t hot_mmap = had_hints ? engine->juju_hot_mmap_friendly.load(std::memory_order_acquire) : 1u;
    uint32_t warm_mmap = had_hints ? engine->juju_warm_mmap_friendly.load(std::memory_order_acquire) : 1u;
    uint32_t cold_mmap = had_hints ? engine->juju_cold_mmap_friendly.load(std::memory_order_acquire) : 1u;
    uint32_t shared_seq = had_hints ? engine->juju_shared_sequential_block_size.load(std::memory_order_acquire) : 0u;
    uint32_t hot_seq = had_hints ? engine->juju_hot_sequential_block_size.load(std::memory_order_acquire) : 0u;
    uint32_t warm_seq = had_hints ? engine->juju_warm_sequential_block_size.load(std::memory_order_acquire) : 0u;
    uint32_t cold_seq = had_hints ? engine->juju_cold_sequential_block_size.load(std::memory_order_acquire) : 0u;
    uint32_t max_prefetch_distance = had_hints ? engine->juju_max_prefetch_distance.load(std::memory_order_acquire) : 0u;
    uint64_t shared_bytes = had_hints ? engine->juju_shared_section_bytes.load(std::memory_order_acquire) : 0ull;
    uint64_t hot_bytes = had_hints ? engine->juju_hot_section_bytes.load(std::memory_order_acquire) : 0ull;
    uint64_t warm_bytes = had_hints ? engine->juju_warm_section_bytes.load(std::memory_order_acquire) : 0ull;
    uint64_t cold_bytes = had_hints ? engine->juju_cold_section_bytes.load(std::memory_order_acquire) : 0ull;
    int found = 0;

    for (const moe_juju_section_entry_t& entry : sections) {
        const uint32_t seq = entry.sequential_block_size;
        max_prefetch_distance = moe_max_u32_local(
            max_prefetch_distance,
            (uint32_t)entry.prefetch_distance);
        switch (entry.type) {
        case moe_JUJU_SECTION_SHARED_WEIGHTS:
            found = 1;
            shared_mmap = entry.mmap_friendly ? shared_mmap : 0u;
            shared_seq = moe_max_u32_local(shared_seq, seq);
            shared_bytes = moe_saturating_add_u64_local(shared_bytes, entry.size);
            break;
        case moe_JUJU_SECTION_HOT_EXPERTS:
            found = 1;
            hot_mmap = entry.mmap_friendly ? hot_mmap : 0u;
            hot_seq = moe_max_u32_local(hot_seq, seq);
            hot_bytes = moe_saturating_add_u64_local(hot_bytes, entry.size);
            break;
        case moe_JUJU_SECTION_WARM_EXPERTS:
            found = 1;
            warm_mmap = entry.mmap_friendly ? warm_mmap : 0u;
            warm_seq = moe_max_u32_local(warm_seq, seq);
            warm_bytes = moe_saturating_add_u64_local(warm_bytes, entry.size);
            break;
        case moe_JUJU_SECTION_COLD_EXPERTS:
            found = 1;
            cold_mmap = entry.mmap_friendly ? cold_mmap : 0u;
            cold_seq = moe_max_u32_local(cold_seq, seq);
            cold_bytes = moe_saturating_add_u64_local(cold_bytes, entry.size);
            break;
        default:
            break;
        }
    }
    if (!found) {
        return;
    }
    engine->juju_shared_mmap_friendly.store(shared_mmap, std::memory_order_release);
    engine->juju_hot_mmap_friendly.store(hot_mmap, std::memory_order_release);
    engine->juju_warm_mmap_friendly.store(warm_mmap, std::memory_order_release);
    engine->juju_cold_mmap_friendly.store(cold_mmap, std::memory_order_release);
    engine->juju_shared_sequential_block_size.store(shared_seq, std::memory_order_release);
    engine->juju_hot_sequential_block_size.store(hot_seq, std::memory_order_release);
    engine->juju_warm_sequential_block_size.store(warm_seq, std::memory_order_release);
    engine->juju_cold_sequential_block_size.store(cold_seq, std::memory_order_release);
    engine->juju_max_prefetch_distance.store(max_prefetch_distance, std::memory_order_release);
    engine->juju_shared_section_bytes.store(shared_bytes, std::memory_order_release);
    engine->juju_hot_section_bytes.store(hot_bytes, std::memory_order_release);
    engine->juju_warm_section_bytes.store(warm_bytes, std::memory_order_release);
    engine->juju_cold_section_bytes.store(cold_bytes, std::memory_order_release);
    engine->juju_io_section_hints_ready.store(1u, std::memory_order_release);
}

static int moe_read_text_file_local(const std::string& path, std::string* out) {
    if (!out) {
        return 0;
    }
    out->clear();
    uint64_t bytes = 0;
    // BUGFIX 330: Check file_size_bytes success before using bytes ★
    // Problem: If file_size_bytes fails, bytes remains 0 but function continues
    // Solution: Return 0 immediately if file_size_bytes fails
    if (!file_size_bytes(path, &bytes) || bytes == 0 || bytes > 256ull * 1024ull * 1024ull) {
        return 0;
    }
    return moe_read_binary_range_to_string(path, 0, bytes, bytes, out);
}

static int moe_read_juju_tensor_index_json(
    const std::string& path,
    const std::vector<moe_juju_section_entry_t>& entries,
    uint64_t file_bytes,
    std::string* out
) {
    std::string sidecar;
    const std::string sidecar_path = path + ".idx";
    uint64_t sidecar_bytes = 0;
    const int sidecar_present = file_size_bytes(sidecar_path, &sidecar_bytes);
    if (sidecar_present || moe_path_is_directory_local(sidecar_path)) {
        // If a sidecar exists it is authoritative.  Falling back to embedded
        // bootstrap metadata after an empty/unreadable/malformed sidecar silently
        // revives stale index contracts and makes sidecar repair workflows
        // ineffective.
        if (!sidecar_present || sidecar_bytes == 0 || sidecar_bytes > 256ull * 1024ull * 1024ull ||
            !moe_read_text_file_local(sidecar_path, &sidecar)) {
            std::fprintf(stderr,
                "[storagellm] JUJU sidecar idx rejected: unreadable authoritative sidecar idx=%s bytes=%llu\n",
                sidecar_path.c_str(),
                (unsigned long long)sidecar_bytes);
            return 0;
        }
        if (sidecar.find("\"JUJU_IDX_JSON_V2\"") == std::string::npos ||
            sidecar.find("\"tensors\"") == std::string::npos) {
            std::fprintf(stderr,
                "[storagellm] JUJU sidecar idx rejected: malformed authoritative sidecar idx=%s\n",
                sidecar_path.c_str());
            return 0;
        }
        if (!moe_juju_index_matches_weight_file(sidecar, path)) {
            std::fprintf(stderr,
                "[storagellm] JUJU sidecar idx rejected: weight_file does not match artifact path idx=%s weight=%s\n",
                sidecar_path.c_str(),
                moe_juju_path_basename_local(path).c_str());
            return 0;
        }
        // The sidecar .juju.idx is authoritative when it names this weight file.
        // Large JUJU payloads are immutable and expensive to rewrite; contract-only
        // repairs must be possible by replacing the sidecar index.  Do not reject a
        // valid sidecar merely because the embedded bootstrap index has an older or
        // different artifact_uid.
        if (out) {
            *out = std::move(sidecar);
        }
        return 1;
    }
    if (moe_read_juju_json_section(path, entries, moe_JUJU_SECTION_LAYER_ORDER_INDEX, file_bytes, out)) {
        return 1;
    }
    return 0;
}

static const char* const moe_juju_required_feature_contract[] = {
    "juju_idx_schema_v6",
    "juju_binary_wire_v2",
    "gguf_codec_registry_v1",
    "expert_bundle_table_v2",
    "bundle_native",
    "explicit_expert_layout",
    "graph_ir",
    "qkv_policy",
    "codec_registry_v1",
    "fail_closed_kernel_contract",
    "fail_closed_required_features",
    "exact_ppl_mode",
    "metadata_first_expert_resolution",
    "source_quant_rows_preserved",
    "deterministic_router_topk",
    "moe_router_topk_contract",
    "moe_router_topk_contract_aliases",
    "metadata_sidecar_config_extraction",
    "attention_scale_contract",
    "attention_head_dim_fallback_scale",
    "attention_query_pre_attn_scalar_contract",
    "storage_plan_contract",
    "runtime_tensor_index_contract",
    nullptr,
};

static int moe_juju_required_feature_supported(const std::string& feature) {
    const std::string f = moe_lower_ascii_copy(feature);
    if (f.empty()) return 0;
    for (const char* const* expected = moe_juju_required_feature_contract; *expected; ++expected) {
        if (f == *expected) {
            return 1;
        }
    }
    return 0;
}

static int moe_juju_required_feature_vector_has(
    const std::vector<std::string>& features,
    const char* feature
) {
    if (!feature || !*feature) {
        return 0;
    }
    const std::string expected = moe_lower_ascii_copy(feature);
    for (const std::string& value : features) {
        if (moe_lower_ascii_copy(value) == expected) {
            return 1;
        }
    }
    return 0;
}

static int moe_juju_required_feature_set_complete(const std::string& json) {
    std::vector<std::string> features;
    if (!moe_json_get_string_array_slice(json, 0, json.size(), "required_features", &features) ||
        features.empty()) {
        std::fprintf(stderr, "[storagellm] missing JUJU required_features\n");
        return 0;
    }
    for (const std::string& feature : features) {
        if (!moe_juju_required_feature_supported(feature)) {
            std::fprintf(stderr,
                "[storagellm] unsupported JUJU required feature: %s\n",
                feature.c_str());
            return 0;
        }
    }
    for (const char* const* expected = moe_juju_required_feature_contract; *expected; ++expected) {
        if (!moe_juju_required_feature_vector_has(features, *expected)) {
            std::fprintf(stderr,
                "[storagellm] missing JUJU required feature: %s\n",
                *expected);
            return 0;
        }
    }
    return 1;
}

static int moe_juju_required_feature_present(
    const std::string& json,
    const char* feature
) {
    if (json.empty() || !feature || !*feature) {
        return 0;
    }
    std::vector<std::string> features;
    if (!moe_json_get_string_array_slice(json, 0, json.size(), "required_features", &features)) {
        return 0;
    }
    const std::string expected = moe_lower_ascii_copy(feature);
    for (const std::string& value : features) {
        if (moe_lower_ascii_copy(value) == expected) {
            return 1;
        }
    }
    return 0;
}

static int moe_juju_extract_attention_contract(
    const std::string& json,
    double* out_query_pre_attn_scalar,
    double* out_attention_scale
) {
    if (out_query_pre_attn_scalar) {
        *out_query_pre_attn_scalar = 0.0;
    }
    if (out_attention_scale) {
        *out_attention_scale = 0.0;
    }
    if (json.empty()) {
        return 0;
    }
    double query_pre_attn_scalar = 0.0;
    const int has_query_pre_attn_scalar =
        (moe_json_get_double_local(json, "query_pre_attn_scalar", &query_pre_attn_scalar) ||
         moe_json_get_double_local(json, "attention_query_pre_attn_scalar", &query_pre_attn_scalar) ||
         moe_json_get_double_local(json, "attn_query_pre_attn_scalar", &query_pre_attn_scalar));
    if (!has_query_pre_attn_scalar ||
        !(query_pre_attn_scalar > 0.0) ||
        query_pre_attn_scalar > 65536.0 ||
        !std::isfinite(query_pre_attn_scalar)) {
        std::fprintf(stderr,
            "[storagellm] JUJU idx missing/invalid query_pre_attn_scalar: %.17g\n",
            query_pre_attn_scalar);
        return 0;
    }
    double attention_scale = 0.0;
    const int has_attention_scale =
        (moe_json_get_double_local(json, "attention_scale", &attention_scale) ||
         moe_json_get_double_local(json, "attention_score_scale", &attention_scale) ||
         moe_json_get_double_local(json, "attn_scale", &attention_scale) ||
         moe_json_get_double_local(json, "qk_scale", &attention_scale));
    if (!has_attention_scale ||
        !(attention_scale > 0.0) ||
        attention_scale > 1024.0 ||
        !std::isfinite(attention_scale)) {
        std::fprintf(stderr, "[storagellm] JUJU idx missing/invalid attention_scale: %.17g\n", attention_scale);
        return 0;
    }
    const double expected_attention_scale = 1.0 / std::sqrt(query_pre_attn_scalar);
    const double tolerance = std::max(1.0e-8, std::fabs(expected_attention_scale) * 1.0e-5);
    if (std::fabs(attention_scale - expected_attention_scale) > tolerance) {
        std::fprintf(stderr,
            "[storagellm] JUJU idx attention scale mismatch: scale=%.17g expected=%.17g qpre=%.17g\n",
            attention_scale, expected_attention_scale, query_pre_attn_scalar);
        return 0;
    }
    if (out_query_pre_attn_scalar) {
        *out_query_pre_attn_scalar = query_pre_attn_scalar;
    }
    if (out_attention_scale) {
        *out_attention_scale = attention_scale;
    }
    return 1;
}

static std::string moe_juju_attention_contract_compact_json(const std::string& json) {
    double query_pre_attn_scalar = 0.0;
    double attention_scale = 0.0;
    if (!moe_juju_extract_attention_contract(json, &query_pre_attn_scalar, &attention_scale)) {
        return std::string();
    }
    char buffer[512];
    std::snprintf(
        buffer,
        sizeof(buffer),
        "{\"format\":\"JUJU_RUNTIME_ATTENTION_CONTRACT_V1\",\"query_pre_attn_scalar\":%.17g,\"attention_query_pre_attn_scalar\":%.17g,\"attention_scale\":%.17g,\"attention_score_scale\":%.17g,\"attention_scale_source\":\"juju_idx_contract\"}",
        query_pre_attn_scalar,
        query_pre_attn_scalar,
        attention_scale,
        attention_scale);
    return std::string(buffer);
}

static void moe_ingest_juju_attention_contract(moe_pc_engine* engine, const std::string& json) {
    if (!engine) {
        return;
    }
    double query_pre_attn_scalar = 0.0;
    double attention_scale = 0.0;
    if (!moe_juju_extract_attention_contract(json, &query_pre_attn_scalar, &attention_scale)) {
        return;
    }
    engine->model_config_query_pre_attn_scalar = (float)query_pre_attn_scalar;
    engine->model_config_attention_scale = (float)attention_scale;
    engine->juju_attention_contract_ready.store(1, std::memory_order_release);
    engine->juju_exact_ppl_contract_ready.store(
        moe_juju_required_feature_present(json, "exact_ppl_mode") ? 1 : 0,
        std::memory_order_release);
    engine->juju_required_features_contract_ready.store(
        moe_juju_required_feature_set_complete(json) ? 1 : 0,
        std::memory_order_release);
    const std::string compact = moe_juju_attention_contract_compact_json(json);
    if (!compact.empty() && engine->offload_gguf_metadata_json.find("JUJU_RUNTIME_ATTENTION_CONTRACT_V1") == std::string::npos) {
        if (!engine->offload_gguf_metadata_json.empty()) {
            engine->offload_gguf_metadata_json.push_back('\n');
        }
        engine->offload_gguf_metadata_json += compact;
    }
}

static int moe_juju_index_schema_supported(const std::string& json) {
    if (json.empty() ||
        json.find("\"JUJU_IDX_JSON_V2\"") == std::string::npos) {
        return 0;
    }
    uint64_t schema_version = 0;
    if (!moe_json_get_u64_local(json, "schema_version", &schema_version) ||
        schema_version != 6u) {
        std::fprintf(stderr,
            "[storagellm] unsupported JUJU idx schema_version: %llu (expected 6)\n",
            (unsigned long long)schema_version);
        return 0;
    }
    std::string text;
    if (!moe_json_get_string_local(json, "binary_wire_id", &text) ||
        text != "JUJU_V2_HEADER4096_SECTION96_BUNDLE_NATIVE") {
        std::fprintf(stderr, "[storagellm] missing/wrong JUJU binary_wire_id\n");
        return 0;
    }
    if (!moe_json_get_string_local(json, "codec_registry_version", &text) ||
        text != "GGUF_CODEC_REGISTRY_V1") {
        std::fprintf(stderr, "[storagellm] missing/wrong JUJU codec_registry_version\n");
        return 0;
    }
    if (!moe_json_get_string_local(json, "expert_bundle_table_format", &text) ||
        text != "JUJU_EXPERT_BUNDLE_TABLE_V2") {
        std::fprintf(stderr, "[storagellm] missing/wrong JUJU expert_bundle_table_format\n");
        return 0;
    }
    if (!moe_json_get_string_local(json, "format_contract_source", &text) ||
        text != "juju_required_idx_contract_fields") {
        std::fprintf(stderr, "[storagellm] missing/wrong JUJU format_contract_source\n");
        return 0;
    }
    int mutable_runtime_index = 0;
    if (!moe_json_get_bool_local(json, "mutable_runtime_index", &mutable_runtime_index) ||
        !mutable_runtime_index) {
        std::fprintf(stderr, "[storagellm] missing/wrong JUJU mutable_runtime_index\n");
        return 0;
    }
    uint64_t format_contract_version = 0;
    if (!moe_json_get_u64_local(json, "format_contract_version", &format_contract_version) ||
        format_contract_version != 4u) {
        std::fprintf(stderr,
            "[storagellm] missing/wrong JUJU format_contract_version: %llu (expected 4)\n",
            (unsigned long long)format_contract_version);
        return 0;
    }
    if (!moe_juju_required_feature_set_complete(json)) {
        return 0;
    }
    double query_pre_attn_scalar = 0.0;
    double attention_scale = 0.0;
    if (!moe_juju_extract_attention_contract(json, &query_pre_attn_scalar, &attention_scale)) {
        std::fprintf(stderr,
            "[storagellm] JUJU idx missing/invalid attention contract: qpre=%.17g scale=%.17g\n",
            query_pre_attn_scalar, attention_scale);
        return 0;
    }
    return 1;
}

static uint16_t moe_juju_accuracy_to_milli(double value) {
    if (!std::isfinite(value) || value <= 0.0) {
        return 0;
    }
    if (value <= 1.0) {
        value *= 1000.0;
    } else if (value <= 100.0) {
        value *= 10.0;
    }
    if (value < 0.0) {
        value = 0.0;
    }
    if (value > 1000.0) {
        value = 1000.0;
    }
    return (uint16_t)std::llround(value);
}

static uint8_t moe_juju_segment_policy_u8(uint64_t value) {
    if (value > (uint64_t)moe_JUJU_SEGMENT_POLICY_ROW_COLUMN) {
        return moe_JUJU_SEGMENT_POLICY_AUTO;
    }
    return (uint8_t)value;
}

static int moe_json_get_u64_from_object_slice_local(
    const std::string& text,
    size_t begin,
    size_t end,
    const char* object_key,
    const char* key,
    uint64_t* out
) {
    if (!object_key || !key || !out || begin >= end || end > text.size() || begin >= text.size()) {
        return 0;
    }
    const std::string slice = text.substr(begin, end - begin);
    size_t obj_begin = 0;
    size_t obj_end = 0;
    return moe_json_find_object_local(slice, object_key, &obj_begin, &obj_end) &&
        moe_json_get_u64_slice(slice, obj_begin, obj_end, key, out);
}

static int moe_json_get_bool_from_object_slice_local(
    const std::string& text,
    size_t begin,
    size_t end,
    const char* object_key,
    const char* key,
    int* out
) {
    if (!object_key || !key || !out || begin >= end || end > text.size() || begin >= text.size()) {
        return 0;
    }
    const std::string slice = text.substr(begin, end - begin);
    size_t obj_begin = 0;
    size_t obj_end = 0;
    return moe_json_find_object_local(slice, object_key, &obj_begin, &obj_end) &&
        moe_json_get_bool_slice(slice, obj_begin, obj_end, key, out);
}

static int moe_json_get_bool_from_object_json_local(
    const std::string& text,
    const char* object_key,
    const char* key,
    int* out
) {
    if (!object_key || !key || !out || text.empty()) {
        return 0;
    }
    size_t obj_begin = 0;
    size_t obj_end = 0;
    return moe_json_find_object_local(text, object_key, &obj_begin, &obj_end) &&
        moe_json_get_bool_slice(text, obj_begin, obj_end, key, out);
}

static void moe_ingest_juju_runtime_contract_io_policy(moe_pc_engine* engine, const std::string& json) {
    if (!engine || json.empty()) {
        return;
    }
    int bool_value = 0;
    if (moe_json_get_bool_from_object_json_local(json, "io_policy", "mmap", &bool_value) ||
        moe_json_get_bool_from_object_json_local(json, "runtime_io_policy", "mmap", &bool_value) ||
        moe_json_get_bool_from_object_json_local(json, "runtime_policy", "mmap", &bool_value) ||
        moe_json_get_bool_local(json, "mmap", &bool_value)) {
        engine->juju_runtime_contract_mmap_allowed.store(
            bool_value != 0 ? 1u : 0u,
            std::memory_order_release);
        engine->juju_runtime_contract_io_policy_ready.store(1u, std::memory_order_release);
    }
}

static void moe_ingest_juju_expert_tier_entries_array(
    moe_pc_engine* engine,
    const std::string& json,
    const char* array_key
);
static void moe_ingest_juju_cross_layer_transition_prior(moe_pc_engine* engine, const std::string& json);

static uint32_t moe_count_bits_u64_local(uint64_t value) {
    uint32_t count = 0;
    while (value) {
        count += (uint32_t)(value & 1ull);
        value >>= 1u;
    }
    return count;
}

static void moe_ingest_juju_moe_layer_bitmask_json(moe_pc_engine* engine, const std::string& json) {
    if (!engine || json.empty()) {
        return;
    }
    size_t arr_begin = 0;
    size_t arr_end = 0;
    if (!moe_json_find_array_local(json, "moe_layer_bitmask_words", &arr_begin, &arr_end)) {
        return;
    }
    std::vector<uint64_t> words;
    words.reserve(16);
    uint64_t moe_layers = 0;
    size_t pos = arr_begin + 1;
    while (pos < arr_end && words.size() < 4096u) {
        pos = json.find_first_not_of(" \t\r\n,", pos);
        if (pos == std::string::npos || pos >= arr_end || json[pos] == ']') {
            break;
        }
        char* parsed_end = nullptr;
        const char* start = json.c_str() + pos;
        const uint64_t value = std::strtoull(start, &parsed_end, 10);
        if (!parsed_end || parsed_end == start) {
            break;
        }
        words.push_back(value);
        moe_layers += moe_count_bits_u64_local(value);
        pos = (size_t)(parsed_end - json.c_str());
    }
    if (words.empty()) {
        return;
    }
    engine->juju_moe_layer_bitmask_words = std::move(words);
    engine->juju_moe_layer_count.store(moe_layers, std::memory_order_relaxed);
    engine->juju_moe_layer_bitmask_ready.store(1u, std::memory_order_release);
}

static void moe_ingest_juju_predictor_section(moe_pc_engine* engine, const std::string& json) {
    if (!engine || json.empty()) {
        return;
    }
    std::string format;
    if (!moe_json_get_string_local(json, "format", &format) ||
        format.find("JUJU_PREDICTOR") == std::string::npos) {
        return;
    }
    uint64_t value = 0;
    if (moe_json_get_u64_local(json, "expert_layer_count", &value)) {
        engine->juju_predictor_layer_count.store(moe_u64_to_u32_clamped(value, 0), std::memory_order_relaxed);
    }
    if (moe_json_get_u64_local(json, "expert_tensor_count", &value)) {
        engine->juju_predictor_expert_tensor_count.store(value, std::memory_order_relaxed);
    }
    moe_ingest_juju_expert_tier_entries_array(engine, json, "expert_activation_priors");
    moe_ingest_juju_expert_tier_entries_array(engine, json, "expert_tier_entries");
    moe_ingest_juju_cross_layer_transition_prior(engine, json);
    engine->juju_predictor_metadata_ready.store(1, std::memory_order_release);
}

static void moe_ingest_juju_buddy_map_section(moe_pc_engine* engine, const std::string& json) {
    if (!engine || json.empty()) {
        return;
    }
    std::string format;
    if (!moe_json_get_string_local(json, "format", &format) ||
        format.find("JUJU_BUDDY_MAP") == std::string::npos) {
        return;
    }
    uint64_t value = 0;
    if (moe_json_get_u64_local(json, "unit_count", &value)) {
        engine->juju_buddy_unit_count.store(moe_u64_to_u32_clamped(value, 0), std::memory_order_relaxed);
    }
    if (moe_json_get_u64_local(json, "tensor_count", &value)) {
        engine->juju_buddy_tensor_count.store(value, std::memory_order_relaxed);
    }

    size_t arr_begin = 0;
    size_t arr_end = 0;
    if (moe_json_find_array_local(json, "expert_bundles", &arr_begin, &arr_end)) {
        size_t cursor = arr_begin + 1;
        size_t obj_begin = 0;
        size_t obj_end = 0;
        while (moe_json_next_object_in_array(json, &cursor, arr_end, &obj_begin, &obj_end)) {
            uint64_t layer64 = 0;
            uint64_t expert64 = 0;
            uint64_t buddies64[MOE_JUJU_MAX_BUDDIES]{};
            uint32_t buddy_count = 0;
            if (!moe_json_get_u64_slice(json, obj_begin, obj_end, "layer", &layer64) ||
                !moe_json_get_u64_slice(json, obj_begin, obj_end, "expert", &expert64) ||
                !moe_json_get_u64_array_slice(json, obj_begin, obj_end, "buddy_expert_ids",
                    buddies64, MOE_JUJU_MAX_BUDDIES, &buddy_count) ||
                layer64 > (uint64_t)std::numeric_limits<int>::max() ||
                expert64 > (uint64_t)std::numeric_limits<int>::max()) {
                continue;
            }
            std::vector<int> buddies;
            buddies.reserve(buddy_count);
            for (uint32_t i = 0; i < buddy_count; ++i) {
                if (buddies64[i] <= (uint64_t)std::numeric_limits<int>::max()) {
                    buddies.push_back((int)buddies64[i]);
                }
            }
            expert_buddy_hints_update(engine, (int)layer64, (int)expert64, buddies);
        }
    }
    engine->juju_buddy_map_ready.store(1, std::memory_order_release);
}

static void moe_ingest_juju_expert_tier_entries_array(
    moe_pc_engine* engine,
    const std::string& json,
    const char* array_key
) {
    if (!engine || json.empty() || !array_key || !array_key[0]) {
        return;
    }
    size_t arr_begin = 0;
    size_t arr_end = 0;
    if (!moe_json_find_array_local(json, array_key, &arr_begin, &arr_end)) {
        return;
    }
    size_t cursor = arr_begin + 1;
    size_t obj_begin = 0;
    size_t obj_end = 0;
    while (moe_json_next_object_in_array(json, &cursor, arr_end, &obj_begin, &obj_end)) {
        uint64_t layer64 = 0;
        uint64_t expert64 = 0;
        uint64_t runtime_priority64 = 0;
        uint64_t prefetch_priority64 = 0;
        uint64_t buddies64[MOE_JUJU_MAX_BUDDIES]{};
        uint32_t buddy_count = 0;
        std::string residency_text;
        if (!moe_json_get_u64_slice(json, obj_begin, obj_end, "layer", &layer64) ||
            !moe_json_get_u64_slice(json, obj_begin, obj_end, "expert", &expert64) ||
            layer64 > (uint64_t)std::numeric_limits<int>::max() ||
            expert64 > (uint64_t)std::numeric_limits<int>::max()) {
            continue;
        }
        if (!moe_json_get_u64_slice(json, obj_begin, obj_end, "runtime_priority", &runtime_priority64)) {
            runtime_priority64 = 65;
        }
        if (!moe_json_get_u64_slice(json, obj_begin, obj_end, "prefetch_priority", &prefetch_priority64)) {
            prefetch_priority64 = 80;
        }
        const uint8_t residency =
            moe_json_get_string_slice(json, obj_begin, obj_end, "residency_hint", &residency_text)
                ? moe_juju_residency_hint_from_text(residency_text)
                : moe_JUJU_RESIDENCY_AUTO;
        expert_runtime_hints_update(
            engine,
            (int)layer64,
            (int)expert64,
            (uint16_t)std::min<uint64_t>(runtime_priority64, 65535ull),
            (uint16_t)std::min<uint64_t>(prefetch_priority64, 65535ull),
            residency);
        if (moe_json_get_u64_array_slice(json, obj_begin, obj_end, "buddy_expert_ids",
                buddies64, MOE_JUJU_MAX_BUDDIES, &buddy_count)) {
            std::vector<int> buddies;
            buddies.reserve(buddy_count);
            for (uint32_t i = 0; i < buddy_count; ++i) {
                if (buddies64[i] <= (uint64_t)std::numeric_limits<int>::max()) {
                    buddies.push_back((int)buddies64[i]);
                }
            }
            expert_buddy_hints_update(engine, (int)layer64, (int)expert64, buddies);
        }
    }
}

static void moe_ingest_juju_expert_tier_entries_json(moe_pc_engine* engine, const std::string& json) {
    moe_ingest_juju_expert_tier_entries_array(engine, json, "expert_tier_entries");
    moe_ingest_juju_expert_tier_entries_array(engine, json, "expert_activation_priors");
}

static void moe_ingest_juju_cross_layer_transition_prior(moe_pc_engine* engine, const std::string& json) {
    if (!engine || json.empty()) {
        return;
    }
    size_t arr_begin = 0;
    size_t arr_end = 0;
    if (!moe_json_find_array_local(json, "cross_layer_transition_prior", &arr_begin, &arr_end)) {
        return;
    }
    std::unique_lock<std::shared_mutex> lock(engine->co_activation_mutex);
    size_t cursor = arr_begin + 1;
    size_t obj_begin = 0;
    size_t obj_end = 0;
    while (moe_json_next_object_in_array(json, &cursor, arr_end, &obj_begin, &obj_end)) {
        uint64_t from_layer64 = 0;
        uint64_t from_expert64 = 0;
        uint64_t to_layer64 = 0;
        uint64_t to_expert64 = 0;
        double probability = 0.0;
        if (!moe_json_get_u64_slice(json, obj_begin, obj_end, "from_layer", &from_layer64) ||
            !moe_json_get_u64_slice(json, obj_begin, obj_end, "from_expert", &from_expert64) ||
            !moe_json_get_u64_slice(json, obj_begin, obj_end, "to_layer", &to_layer64) ||
            !moe_json_get_u64_slice(json, obj_begin, obj_end, "to_expert", &to_expert64) ||
            from_layer64 > 9999ull ||
            to_layer64 > 9999ull ||
            from_expert64 > (uint64_t)std::numeric_limits<int>::max() ||
            to_expert64 > (uint64_t)std::numeric_limits<int>::max()) {
            continue;
        }
        moe_json_get_double_slice(json, obj_begin, obj_end, "probability", &probability);
        const float score = (std::isfinite(probability) && probability > 0.0)
            ? (float)std::max(0.001, probability)
            : 0.001f;
        if (engine->co_activation_stats.size() <= (size_t)to_layer64) {
            engine->co_activation_stats.resize((size_t)to_layer64 + 1u);
        }
        engine->co_activation_stats[(size_t)to_layer64][(int)from_expert64][(int)to_expert64] += score;
    }
}

static void moe_ingest_juju_tier_hint_section(moe_pc_engine* engine, const std::string& json) {
    if (!engine || json.empty()) {
        return;
    }
    std::string format;
    if (!moe_json_get_string_local(json, "format", &format) ||
        format.find("JUJU_TIER_HINT") == std::string::npos) {
        return;
    }
    uint64_t value = 0;
    if (moe_json_get_u64_local(json, "hot_expert_tensor_count", &value)) {
        engine->juju_tier_hot_expert_tensor_count.store(value, std::memory_order_relaxed);
    }
    if (moe_json_get_u64_local(json, "warm_expert_tensor_count", &value)) {
        engine->juju_tier_warm_expert_tensor_count.store(value, std::memory_order_relaxed);
    }
    if (moe_json_get_u64_local(json, "cold_expert_tensor_count", &value)) {
        engine->juju_tier_cold_expert_tensor_count.store(value, std::memory_order_relaxed);
    }
    moe_ingest_juju_moe_layer_bitmask_json(engine, json);
    moe_ingest_juju_expert_tier_entries_json(engine, json);
    engine->juju_tier_hint_ready.store(1, std::memory_order_release);
}

static void moe_ingest_juju_runtime_metadata_sections(
    moe_pc_engine* engine,
    const std::string& path,
    const std::vector<moe_juju_section_entry_t>& sections,
    uint64_t file_bytes
) {
    if (!engine || path.empty()) {
        return;
    }
    moe_ingest_juju_section_table_hints(engine, sections);
    std::string json;
    if (moe_read_juju_json_section(path, sections, moe_JUJU_SECTION_PREDICTOR, file_bytes, &json)) {
        moe_ingest_juju_predictor_section(engine, json);
    }
    if (moe_read_juju_json_section(path, sections, moe_JUJU_SECTION_BUDDY_MAP, file_bytes, &json)) {
        moe_ingest_juju_buddy_map_section(engine, json);
    }
    if (moe_read_juju_json_section(path, sections, moe_JUJU_SECTION_TIER_HINT, file_bytes, &json)) {
        moe_ingest_juju_tier_hint_section(engine, json);
    }
    if (moe_read_juju_json_section(path, sections, moe_JUJU_SECTION_RUNTIME_CONTRACT, file_bytes, &json)) {
        moe_ingest_juju_runtime_contract_io_policy(engine, json);
        if (!engine->offload_gguf_metadata_json.empty()) {
            engine->offload_gguf_metadata_json.push_back('\n');
        }
        engine->offload_gguf_metadata_json += json;
    }
}

static int moe_parse_juju_tensor_index_json(
    const std::string& json,
    std::vector<moe_juju_tensor_json_entry_t>* out_entries,
    uint64_t* out_tensor_count,
    uint64_t* out_executable_count,
    uint64_t* out_max_executable_bytes
) {
    if (out_entries) {
        out_entries->clear();
    }
    if (out_tensor_count) {
        *out_tensor_count = 0;
    }
    if (out_executable_count) {
        *out_executable_count = 0;
    }
    if (out_max_executable_bytes) {
        *out_max_executable_bytes = 0;
    }
    size_t arr_begin = 0;
    size_t arr_end = 0;
    if (json.empty() ||
        (!moe_json_find_top_level_array_local(json, "tensors", &arr_begin, &arr_end) &&
         !moe_json_find_array_local(json, "tensors", &arr_begin, &arr_end))) {
        return 0;
    }
    uint64_t tensor_count = 0;
    uint64_t executable_count = 0;
    uint64_t max_executable = 0;
    size_t cursor = arr_begin + 1;
    size_t obj_begin = 0;
    size_t obj_end = 0;
    while (moe_json_next_object_in_array(json, &cursor, arr_end, &obj_begin, &obj_end)) {
        moe_juju_tensor_json_entry_t rec{};
        uint64_t value = 0;
        uint64_t shape[4]{};
        uint32_t shape_count = 0;
        int bool_value = 0;
        if (!moe_json_get_string_slice(json, obj_begin, obj_end, "name", &rec.name) ||
            !moe_json_get_u64_slice(json, obj_begin, obj_end, "juju_offset", &rec.juju_offset) ||
            !moe_json_get_u64_slice(json, obj_begin, obj_end, "juju_bytes", &rec.juju_bytes)) {
            continue;
        }
        if (rec.juju_bytes == 0) {
            continue;
        }
        // BUGFIX 331: Check juju_bytes > 0 to prevent zero-size tensors ★
        // Problem: juju_bytes can be 0, causing issues in tensor loading
        // Solution: Skip tensors with juju_bytes == 0
        if (rec.juju_bytes == 0) {
            continue;
        }
        if (moe_json_get_u64_slice(json, obj_begin, obj_end, "gguf_type", &value)) {
            rec.gguf_type = moe_u64_to_u32_clamped(value, 0);
            rec.has_gguf_type = 1;
        }
        moe_json_get_string_slice(json, obj_begin, obj_end, "codec_id", &rec.codec_id);
        moe_json_get_string_slice(json, obj_begin, obj_end, "codec_registry_version", &rec.codec_registry_version);
        moe_json_get_string_slice(json, obj_begin, obj_end, "storage_layout", &rec.storage_layout);
        moe_json_get_string_slice(json, obj_begin, obj_end, "split_policy", &rec.split_policy);
        if (!moe_json_get_string_slice(json, obj_begin, obj_end, "op_role", &rec.op_role)) {
            moe_json_get_string_slice(json, obj_begin, obj_end, "execution_op", &rec.op_role);
        }
        moe_json_get_string_slice(json, obj_begin, obj_end, "execution_op", &rec.execution_op);
        if (moe_json_get_u64_slice(json, obj_begin, obj_end, "source_bytes", &value)) {
            rec.source_bytes = value;
        }
        if (moe_json_get_u64_slice(json, obj_begin, obj_end, "logical_bytes", &value)) {
            rec.logical_bytes = value;
        }
        if (moe_json_get_u64_slice(json, obj_begin, obj_end, "physical_bytes", &value)) {
            rec.physical_bytes = value;
        }
        if (moe_json_get_u64_slice(json, obj_begin, obj_end, "logical_rows", &value)) {
            rec.logical_rows = value;
        }
        if (moe_json_get_u64_slice(json, obj_begin, obj_end, "logical_cols", &value)) {
            rec.logical_cols = value;
        }
        if (moe_json_get_u64_slice(json, obj_begin, obj_end, "row_bytes", &value) ||
            moe_json_get_u64_slice(json, obj_begin, obj_end, "source_row_bytes", &value)) {
            rec.row_bytes = value;
        }
        if (moe_json_get_u64_slice(json, obj_begin, obj_end, "row_stride_bytes", &value)) {
            rec.row_stride_bytes = value;
        }
        if (moe_json_get_u64_slice(json, obj_begin, obj_end, "row_padding_bytes", &value)) {
            rec.row_padding_bytes = value;
        }
        uint64_t source_shape[4]{};
        uint32_t source_shape_count = 0;
        if (moe_json_get_u64_array_slice(json, obj_begin, obj_end, "source_shape", source_shape, 4, &source_shape_count)) {
            rec.source_dims = std::min<uint32_t>(source_shape_count, 4u);
            for (uint32_t i = 0; i < rec.source_dims; ++i) {
                rec.source_shape[i] = source_shape[i];
            }
        }
        if (moe_json_get_u64_from_object_slice_local(json, obj_begin, obj_end, "math_shape", "out_features", &value)) {
            rec.math_out_features = value;
        }
        if (moe_json_get_u64_from_object_slice_local(json, obj_begin, obj_end, "math_shape", "in_features", &value)) {
            rec.math_in_features = value;
        }
        if (moe_json_get_bool_from_object_slice_local(json, obj_begin, obj_end, "math_shape", "transposed", &bool_value)) {
            rec.math_transposed = bool_value ? 1 : 0;
        }
        if (moe_json_get_u64_slice(json, obj_begin, obj_end, "expert_count", &value) ||
            moe_json_get_u64_from_object_slice_local(json, obj_begin, obj_end, "expert_layout", "expert_count", &value)) {
            rec.expert_count = moe_u64_to_u32_clamped(value, 0);
            rec.has_expert_layout = 1;
        }
        if (moe_json_get_u64_slice(json, obj_begin, obj_end, "per_expert_bytes", &value) ||
            moe_json_get_u64_from_object_slice_local(json, obj_begin, obj_end, "expert_layout", "per_expert_bytes", &value)) {
            rec.per_expert_bytes = value;
            rec.has_expert_layout = 1;
        }
        if (moe_json_get_u64_from_object_slice_local(json, obj_begin, obj_end, "expert_layout", "base_offset", &value)) {
            rec.expert_base_offset = value;
            rec.has_expert_layout = 1;
        }
        if (moe_json_get_bool_from_object_slice_local(json, obj_begin, obj_end, "combined_gate_up_split", "enabled", &bool_value) ||
            moe_json_get_bool_slice(json, obj_begin, obj_end, "combined_gate_up_split_enabled", &bool_value)) {
            rec.combined_gate_up_split_enabled = bool_value ? 1 : 0;
        }
        if (moe_json_get_u64_from_object_slice_local(json, obj_begin, obj_end, "combined_gate_up_split", "gate_rel_offset", &value)) {
            rec.gate_rel_offset = value;
        }
        if (moe_json_get_u64_from_object_slice_local(json, obj_begin, obj_end, "combined_gate_up_split", "gate_bytes", &value)) {
            rec.gate_bytes = value;
        }
        if (moe_json_get_u64_from_object_slice_local(json, obj_begin, obj_end, "combined_gate_up_split", "up_rel_offset", &value)) {
            rec.up_rel_offset = value;
        }
        if (moe_json_get_u64_from_object_slice_local(json, obj_begin, obj_end, "combined_gate_up_split", "up_bytes", &value)) {
            rec.up_bytes = value;
        }
        if (moe_json_get_bool_slice(json, obj_begin, obj_end, "bundle_native", &bool_value)) {
            rec.bundle_native = bool_value ? 1 : 0;
        }
        if (moe_json_get_u64_slice(json, obj_begin, obj_end, "bundle_id", &value)) {
            rec.bundle_id = moe_u64_to_u32_clamped(value, 0);
        }
        if (moe_json_get_u64_slice(json, obj_begin, obj_end, "expert_id", &value)) {
            rec.explicit_expert = moe_u64_to_u32_clamped(value, 0);
            rec.has_explicit_expert = 1;
        }
        if (moe_json_get_string_slice(json, obj_begin, obj_end, "bundle_member_role", &rec.bundle_member_role)) {
            rec.bundle_native = 1;
        }
        moe_json_get_string_slice(json, obj_begin, obj_end, "expert_projection", &rec.expert_projection);
        if (moe_json_get_u64_slice(json, obj_begin, obj_end, "bundle_offset", &value) ||
            moe_json_get_u64_from_object_slice_local(json, obj_begin, obj_end, "expert_layout", "bundle_offset", &value)) {
            rec.bundle_offset = value;
        }
        if (moe_json_get_u64_slice(json, obj_begin, obj_end, "bundle_size", &value) ||
            moe_json_get_u64_from_object_slice_local(json, obj_begin, obj_end, "expert_layout", "bundle_size", &value)) {
            rec.bundle_size = value;
        }
        if (moe_json_get_u64_slice(json, obj_begin, obj_end, "bundle_member_offset", &value) ||
            moe_json_get_u64_from_object_slice_local(json, obj_begin, obj_end, "expert_layout", "bundle_member_offset", &value)) {
            rec.bundle_member_offset = value;
        }
        if (moe_json_get_u64_slice(json, obj_begin, obj_end, "bundle_member_size", &value) ||
            moe_json_get_u64_from_object_slice_local(json, obj_begin, obj_end, "expert_layout", "bundle_member_size", &value)) {
            rec.bundle_member_size = value;
        }
        if (moe_json_get_u64_slice(json, obj_begin, obj_end, "bundle_alignment", &value) ||
            moe_json_get_u64_from_object_slice_local(json, obj_begin, obj_end, "expert_layout", "bundle_alignment", &value)) {
            rec.bundle_alignment = value;
        }
        moe_json_get_string_slice(json, obj_begin, obj_end, "expert_gpu_layout", &rec.expert_gpu_layout);
        if (moe_json_get_u64_slice(json, obj_begin, obj_end, "expert_gpu_layout_offset", &value)) {
            rec.expert_gpu_layout_offset = value;
        }
        if (moe_json_get_u64_slice(json, obj_begin, obj_end, "expert_gpu_layout_size", &value)) {
            rec.expert_gpu_layout_size = value;
        }
        if (moe_json_get_u64_slice(json, obj_begin, obj_end, "expert_gpu_layout_row_bytes", &value)) {
            rec.expert_gpu_layout_row_bytes = value;
        }
        if (moe_json_get_u64_slice(json, obj_begin, obj_end, "weight_encoding", &value) ||
            moe_json_get_u64_slice(json, obj_begin, obj_end, "juju_weight_encoding", &value) ||
            moe_json_get_u64_slice(json, obj_begin, obj_end, "source_weight_encoding", &value)) {
            rec.weight_encoding = (uint8_t)std::min<uint64_t>(value, 255ull);
        }
        if (moe_json_get_u64_slice(json, obj_begin, obj_end, "runtime_priority", &value)) {
            rec.runtime_priority = (uint16_t)std::min<uint64_t>(value, 1000ull);
        }
        if (moe_json_get_u64_slice(json, obj_begin, obj_end, "prefetch_priority", &value)) {
            rec.prefetch_priority = (uint16_t)std::min<uint64_t>(value, 1000ull);
        }
        if (moe_json_get_u64_slice(json, obj_begin, obj_end, "execution_order", &value)) {
            rec.execution_order = (uint16_t)std::min<uint64_t>(value, 1000ull);
        }
        if (moe_json_get_u64_slice(json, obj_begin, obj_end, "hotset_rank", &value)) {
            rec.hotset_rank = (uint16_t)std::min<uint64_t>(value, 1000ull);
        }
        if (moe_json_get_string_slice(json, obj_begin, obj_end, "bucket", &rec.bucket)) {
            const std::string bucket = moe_lower_ascii_copy(rec.bucket);
            if (bucket.find("vision_projector") != std::string::npos || bucket.find("vision_proj") != std::string::npos ||
                bucket.find("image_projector") != std::string::npos || bucket.find("mm_projector") != std::string::npos) {
                rec.juju_section_type = (uint32_t)moe_JUJU_SECTION_VISION_PROJ;
            } else if (bucket.find("vision") != std::string::npos || bucket.find("image") != std::string::npos) {
                rec.juju_section_type = (uint32_t)moe_JUJU_SECTION_VISION_ENCODER;
            } else if (bucket.find("audio") != std::string::npos) {
                rec.juju_section_type = (uint32_t)moe_JUJU_SECTION_AUDIO_ENCODER;
            } else if (bucket.find("video") != std::string::npos) {
                rec.juju_section_type = (uint32_t)moe_JUJU_SECTION_VIDEO_ENCODER;
            } else if (bucket.find("document") != std::string::npos || bucket.find("pdf") != std::string::npos ||
                       bucket.find("ocr") != std::string::npos) {
                rec.juju_section_type = (uint32_t)moe_JUJU_SECTION_DOCUMENT_ENCODER;
            } else if (bucket.find("shared") != std::string::npos) {
                rec.juju_section_type = (uint32_t)moe_JUJU_SECTION_SHARED_WEIGHTS;
            } else if (bucket.find("hot") != std::string::npos) {
                rec.juju_section_type = (uint32_t)moe_JUJU_SECTION_HOT_EXPERTS;
            } else if (bucket.find("warm") != std::string::npos) {
                rec.juju_section_type = (uint32_t)moe_JUJU_SECTION_WARM_EXPERTS;
            } else if (bucket.find("cold") != std::string::npos) {
                rec.juju_section_type = (uint32_t)moe_JUJU_SECTION_COLD_EXPERTS;
            }
        }
        if (moe_json_get_string_slice(json, obj_begin, obj_end, "graph_role", &rec.graph_role_text)) {
            rec.graph_role = moe_juju_graph_role_from_text(rec.graph_role_text);
        }
        if (moe_json_get_string_slice(json, obj_begin, obj_end, "prefetch_class", &rec.prefetch_class_text)) {
            rec.prefetch_class = moe_juju_prefetch_class_from_text(rec.prefetch_class_text);
        }
        if (moe_json_get_string_slice(json, obj_begin, obj_end, "residency_hint", &rec.residency_hint_text)) {
            rec.residency_hint = moe_juju_residency_hint_from_text(rec.residency_hint_text);
        }
        if (moe_json_get_u64_slice(json, obj_begin, obj_end, "segment_policy", &value)) {
            rec.segment_policy = moe_juju_segment_policy_u8(value);
        }
        if (moe_json_get_u64_slice(json, obj_begin, obj_end, "can_partial_exec", &value)) {
            rec.can_partial_exec = value != 0 ? 1 : 0;
        } else {
            if (moe_json_get_bool_slice(json, obj_begin, obj_end, "can_partial_exec", &bool_value)) {
                rec.can_partial_exec = bool_value ? 1 : 0;
            }
        }
        double accuracy = 0.0;
        if (moe_json_get_double_slice(json, obj_begin, obj_end, "partial_accuracy", &accuracy)) {
            rec.partial_accuracy_milli = moe_juju_accuracy_to_milli(accuracy);
        }
        if (moe_json_get_u64_slice(json, obj_begin, obj_end, "dims", &value)) {
            rec.dims = moe_u64_to_u32_clamped(value, 0);
        }
        if (moe_json_get_u64_array_slice(json, obj_begin, obj_end, "shape", shape, 4, &shape_count)) {
            rec.dims = rec.dims ? std::min<uint32_t>(rec.dims, 4u) : shape_count;
            for (uint32_t i = 0; i < std::min<uint32_t>(shape_count, 4u); ++i) {
                rec.shape[i] = shape[i];
            }
        }
        if (moe_json_get_u64_slice(json, obj_begin, obj_end, "execution_layer", &value)) {
            rec.execution_layer = (int32_t)std::min<uint64_t>(value, (uint64_t)INT32_MAX);
        }
        if (rec.execution_layer == INT32_MIN) {
            uint32_t parsed_layer = 0;
            if (moe_parse_layer_from_gguf_name(rec.name, &parsed_layer)) {
                rec.execution_layer = (int32_t)std::min<uint32_t>(parsed_layer, (uint32_t)INT32_MAX);
            } else {
                const std::string role = moe_lower_ascii_copy(!rec.op_role.empty() ? rec.op_role : rec.execution_op);
                if (role == "token_embedding" || role == "lm_head" || role == "final_norm" ||
                    role == "rope" || role == "weight") {
                    rec.execution_layer = -1;
                }
            }
        }
        ++tensor_count;
        if (moe_gguf_tensor_name_executable(rec.name)) {
            ++executable_count;
            if (rec.juju_bytes > max_executable) {
                max_executable = rec.juju_bytes;
            }
        }
        if (out_entries) {
            out_entries->push_back(std::move(rec));
        }
    }
    if (out_tensor_count) {
        *out_tensor_count = tensor_count;
    }
    if (out_executable_count) {
        *out_executable_count = executable_count;
    }
    if (out_max_executable_bytes) {
        *out_max_executable_bytes = max_executable;
    }
    return tensor_count > 0 ? 1 : 0;
}
