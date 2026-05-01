// GGUF tensor reading and directory parsing functions.
// Reads tensor metadata from GGUF files and builds tensor directories.

static int moe_gguf_tensor_name_executable(const std::string& name) {
    if (name.empty()) {
        return 0;
    }
    const std::string value = moe_lower_ascii_copy(name);
    if (value == "token_embd.weight" ||
        value == "output.weight" ||
        value == "output_norm.weight" ||
        value.find("blk.") == 0 ||
        value.find(".blk.") != std::string::npos ||
        value.find("model.layers.") != std::string::npos) {
        return 1;
    }
    return 0;
}

static int moe_read_gguf_tensor_infos(
    std::ifstream& input,
    uint64_t tensor_count,
    uint32_t alignment,
    uint64_t file_bytes,
    std::vector<moe_gguf_tensor_dir_entry_t>* tensors,
    uint64_t* data_start,
    uint64_t* executable_count,
    uint64_t* max_executable_bytes
) {
    if (!tensors || tensor_count > 10000000ull) {
        return 0;
    }
    tensors->clear();
    tensors->reserve((size_t)std::min<uint64_t>(tensor_count, 1000000ull));
    for (uint64_t i = 0; i < tensor_count; ++i) {
        moe_gguf_tensor_dir_entry_t entry{};
        if (!moe_gguf_read_string(input, &entry.name, 1024) ||
            !moe_gguf_read_u32(input, &entry.dims) ||
            entry.dims > 64) {
            return 0;
        }
        for (uint32_t d = 0; d < entry.dims; ++d) {
            uint64_t dim = 0;
            if (!moe_gguf_read_u64(input, &dim)) {
                return 0;
            }
            if (d < 4) {
                entry.shape[d] = dim;
            }
        }
        if (!moe_gguf_read_u32(input, &entry.type) ||
            !moe_gguf_read_u64(input, &entry.tensor_offset)) {
            return 0;
        }
        tensors->push_back(std::move(entry));
    }
    const uint64_t end = (uint64_t)input.tellg();
    const uint64_t base = moe_align_u64(end, alignment);
    if (data_start) {
        *data_start = base;
    }
    std::sort(tensors->begin(), tensors->end(), [](const moe_gguf_tensor_dir_entry_t& a, const moe_gguf_tensor_dir_entry_t& b) {
        return a.tensor_offset < b.tensor_offset;
    });
    uint64_t executable = 0;
    uint64_t max_exec = 0;
    const uint64_t payload_bytes = file_bytes > base ? file_bytes - base : 0;
    for (size_t i = 0; i < tensors->size(); ++i) {
        moe_gguf_tensor_dir_entry_t& entry = (*tensors)[i];
        const uint64_t next = (i + 1u < tensors->size()) ? (*tensors)[i + 1u].tensor_offset : payload_bytes;
        entry.bytes = next > entry.tensor_offset ? next - entry.tensor_offset : 0;
        entry.file_offset = base + entry.tensor_offset;
        if (moe_gguf_tensor_name_executable(entry.name)) {
            ++executable;
            if (entry.bytes > max_exec) {
                max_exec = entry.bytes;
            }
        }
    }
    if (executable_count) {
        *executable_count = executable;
    }
    if (max_executable_bytes) {
        *max_executable_bytes = max_exec;
    }
    return 1;
}

static int moe_read_gguf_tensor_directory(
    const std::string& path,
    moe_gguf_tensor_dir_t* out
) {
    if (!out || path.empty()) {
        return 0;
    }
    *out = moe_gguf_tensor_dir_t{};
    out->path = path;
    file_size_bytes(path, &out->file_bytes);
    std::ifstream input;
    open_ifstream_path(input, path, std::ios::in | std::ios::binary);
    if (!input) {
        return 0;
    }
    char magic[4]{};
    uint32_t version = 0;
    uint64_t tensor_count = 0;
    uint64_t kv_count = 0;
    if (!moe_gguf_read_exact(input, magic, sizeof(magic)) ||
        std::memcmp(magic, "GGUF", 4) != 0 ||
        !moe_gguf_read_u32(input, &version) ||
        !moe_gguf_read_u64(input, &tensor_count) ||
        !moe_gguf_read_u64(input, &kv_count) ||
        version == 0 ||
        kv_count > 1000000ull ||
        tensor_count > 10000000ull) {
        return 0;
    }
    for (uint64_t i = 0; i < kv_count; ++i) {
        std::string key;
        uint32_t type = 0;
        if (!moe_gguf_read_string(input, &key, 65536) ||
            !moe_gguf_read_u32(input, &type)) {
            return 0;
        }
        uint64_t u64_value = 0;
        int consumed = 0;
        if (key == "general.alignment" && moe_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            if (u64_value >= 8 && u64_value <= 65536) {
                out->alignment = (uint32_t)u64_value;
            }
        } else if (key == "offload.file_index" && moe_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->file_index = moe_u64_to_u32_clamped(u64_value, 0);
        } else if (key == "offload.file_count" && moe_gguf_read_uint_value(input, type, &u64_value)) {
            consumed = 1;
            out->file_count = moe_u64_to_u32_clamped(u64_value, 0);
        }
        if (!consumed && !moe_gguf_skip_value(input, type)) {
            return 0;
        }
    }
    uint64_t executable = 0;
    uint64_t max_executable = 0;
    return moe_read_gguf_tensor_infos(
        input,
        tensor_count,
        out->alignment,
        out->file_bytes,
        &out->tensors,
        &out->data_start,
        &executable,
        &max_executable);
}

static int moe_scan_gguf_tensor_infos(
    std::ifstream& input,
    uint64_t tensor_count,
    moe_offload_gguf_file_t* out
) {
    if (!out) {
        return 0;
    }
    std::vector<moe_gguf_tensor_dir_entry_t> tensors;
    uint64_t data_start = 0;
    uint64_t executable = 0;
    uint64_t max_executable = 0;
    if (!moe_read_gguf_tensor_infos(
            input,
            tensor_count,
            out->alignment,
            out->bytes,
            &tensors,
            &data_start,
            &executable,
            &max_executable)) {
        return 0;
    }
    out->tensor_count = tensor_count;
    out->data_start = data_start;
    out->executable_tensor_count = executable;
    out->max_executable_tensor_bytes = max_executable;
    return 1;
}
