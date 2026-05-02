// JUJU 포맷 파싱 유틸리티
// JUJU 텐서 인덱스, 메타데이터, 섹션 읽기

struct moe_juju_tensor_json_entry_t {
    std::string name;
    std::string bucket;
    std::string graph_role_text;
    std::string prefetch_class_text;
    std::string residency_hint_text;
    uint64_t shape[4]{};
    uint32_t dims = 0;
    uint32_t gguf_type = 0;
    uint8_t has_gguf_type = 0;
    uint8_t weight_encoding = moe_WEIGHT_ENCODING_UNKNOWN;
    uint64_t juju_offset = 0;
    uint64_t juju_bytes = 0;
    uint16_t runtime_priority = 50;
    uint16_t prefetch_priority = 50;
    uint8_t residency_hint = moe_JUJU_RESIDENCY_AUTO;
    uint8_t graph_role = moe_JUJU_ROLE_UNKNOWN;
    uint8_t prefetch_class = moe_JUJU_PREFETCH_AUTO;
    uint8_t juju_section_type = 0;
};

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
    if (moe_read_juju_json_section(path, entries, moe_JUJU_SECTION_LAYER_ORDER_INDEX, file_bytes, out)) {
        return 1;
    }
    return moe_read_text_file_local(path + ".idx", out);
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
    if (json.empty() || !moe_json_find_array_local(json, "tensors", &arr_begin, &arr_end)) {
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
        if (!moe_json_get_string_slice(json, obj_begin, obj_end, "name", &rec.name) ||
            !moe_json_get_u64_slice(json, obj_begin, obj_end, "juju_offset", &rec.juju_offset) ||
            !moe_json_get_u64_slice(json, obj_begin, obj_end, "juju_bytes", &rec.juju_bytes)) {
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
        if (moe_json_get_string_slice(json, obj_begin, obj_end, "bucket", &rec.bucket)) {
            const std::string bucket = moe_lower_ascii_copy(rec.bucket);
            if (bucket.find("vision_projector") != std::string::npos || bucket.find("vision_proj") != std::string::npos ||
                bucket.find("image_projector") != std::string::npos || bucket.find("mm_projector") != std::string::npos) {
                rec.juju_section_type = (uint8_t)moe_JUJU_SECTION_VISION_PROJ;
            } else if (bucket.find("vision") != std::string::npos || bucket.find("image") != std::string::npos) {
                rec.juju_section_type = (uint8_t)moe_JUJU_SECTION_VISION_ENCODER;
            } else if (bucket.find("audio") != std::string::npos) {
                rec.juju_section_type = (uint8_t)moe_JUJU_SECTION_AUDIO_ENCODER;
            } else if (bucket.find("video") != std::string::npos) {
                rec.juju_section_type = (uint8_t)moe_JUJU_SECTION_VIDEO_ENCODER;
            } else if (bucket.find("document") != std::string::npos || bucket.find("pdf") != std::string::npos ||
                       bucket.find("ocr") != std::string::npos) {
                rec.juju_section_type = (uint8_t)moe_JUJU_SECTION_DOCUMENT_ENCODER;
            } else if (bucket.find("shared") != std::string::npos) {
                rec.juju_section_type = (uint8_t)moe_JUJU_SECTION_SHARED_WEIGHTS;
            } else if (bucket.find("hot") != std::string::npos) {
                rec.juju_section_type = (uint8_t)moe_JUJU_SECTION_HOT_EXPERTS;
            } else if (bucket.find("warm") != std::string::npos) {
                rec.juju_section_type = (uint8_t)moe_JUJU_SECTION_WARM_EXPERTS;
            } else if (bucket.find("cold") != std::string::npos) {
                rec.juju_section_type = (uint8_t)moe_JUJU_SECTION_COLD_EXPERTS;
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
        if (moe_json_get_u64_slice(json, obj_begin, obj_end, "dims", &value)) {
            rec.dims = moe_u64_to_u32_clamped(value, 0);
        }
        if (moe_json_get_u64_array_slice(json, obj_begin, obj_end, "shape", shape, 4, &shape_count)) {
            rec.dims = rec.dims ? std::min<uint32_t>(rec.dims, 4u) : shape_count;
            for (uint32_t i = 0; i < std::min<uint32_t>(shape_count, 4u); ++i) {
                rec.shape[i] = shape[i];
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
