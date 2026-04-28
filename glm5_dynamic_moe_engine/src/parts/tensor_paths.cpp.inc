static void open_ifstream_path(
    std::ifstream& input,
    const std::string& path,
    std::ios::openmode mode = std::ios::in
) {
#ifdef _WIN32
    const std::wstring wide = storagellm::wide_path_from_utf8(path);
    if (!wide.empty()) {
        input.open(wide.c_str(), mode);
    }
#else
    input.open(path, mode);
#endif
}

static void open_ofstream_path(
    std::ofstream& output,
    const std::string& path,
    std::ios::openmode mode = std::ios::out
) {
#ifdef _WIN32
    const std::wstring wide = storagellm::wide_path_from_utf8(path);
    if (!wide.empty()) {
        output.open(wide.c_str(), mode);
    }
#else
    output.open(path, mode);
#endif
}

static int file_size_bytes(const std::string& path, uint64_t* out_bytes) {
    if (out_bytes) {
        *out_bytes = 0;
    }
    if (path.empty()) {
        return 0;
    }
#ifdef _WIN32
    const std::wstring wide = storagellm::wide_path_from_utf8(path);
    if (wide.empty()) {
        return 0;
    }
    WIN32_FILE_ATTRIBUTE_DATA data{};
    if (!GetFileAttributesExW(wide.c_str(), GetFileExInfoStandard, &data)) {
        return 0;
    }
    if (data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
        return 0;
    }
    ULARGE_INTEGER value{};
    value.HighPart = data.nFileSizeHigh;
    value.LowPart = data.nFileSizeLow;
    if (out_bytes) {
        *out_bytes = value.QuadPart;
    }
    return 1;
#else
    struct stat st{};
    if (stat(path.c_str(), &st) != 0 || S_ISDIR(st.st_mode)) {
        return 0;
    }
    if (out_bytes) {
        *out_bytes = (uint64_t)st.st_size;
    }
    return 1;
#endif
}

static std::string resolve_model_file_path(
    const std::string& root,
    const std::string& file,
    const std::string& fallback
) {
    if (!root.empty() && !file.empty()) {
        const std::string rooted = storagellm::path_join(root, file);
        uint64_t ignored = 0;
        if (file_size_bytes(rooted, &ignored)) {
            return rooted;
        }
        const size_t slash = file.find_last_of("/\\");
        if (slash != std::string::npos && slash + 1 < file.size()) {
            const std::string flat = storagellm::path_join(root, file.substr(slash + 1));
            if (file_size_bytes(flat, &ignored)) {
                return flat;
            }
        }
        return rooted;
    }
    if (!fallback.empty()) {
        return fallback;
    }
    return file;
}

static size_t find_tensor_record_index(
    const glm5_pc_engine* engine,
    int layer,
    int expert,
    glm5_projection_t proj
) {
    if (!engine) {
        return tensor_slot_invalid();
    }
    const size_t slot_index = tensor_slots_get(engine, layer, expert, proj);
    if (slot_index != tensor_slot_invalid() && slot_index < engine->tensors.size()) {
        return slot_index;
    }
    return tensor_slot_invalid();
}


static const glm5_tensor_record* find_tensor_record(
    const glm5_pc_engine* engine,
    int layer,
    int expert,
    glm5_projection_t proj
) {
    const size_t index = find_tensor_record_index(engine, layer, expert, proj);
    if (index == tensor_slot_invalid()) {
        return nullptr;
    }
    return &engine->tensors[index];
}

static mmap_context_t* get_or_map_shard(glm5_pc_engine* engine, const std::string& path) {
    if (!engine || path.empty()) {
        return nullptr;
    }
    {
        // Fix 61: shard_mutex protects shard_mmaps (separate from expert_states)
        std::unique_lock<std::mutex> lock(engine->shard_mutex);
        while (true) {
            auto found = engine->shard_mmaps.find(path);
            if (found != engine->shard_mmaps.end()) {
                if (found->second == (mmap_context_t*)1) {
                    engine->map_cv.wait(lock);
                } else {
                    return found->second;
                }
            } else {
                engine->shard_mmaps[path] = (mmap_context_t*)1;
                break;
            }
        }
    }
    mmap_context_t* ctx = mmap_load(path.c_str());
    if (!ctx) {
        std::lock_guard<std::mutex> lock(engine->shard_mutex);
        engine->shard_mmaps.erase(path);
        engine->map_cv.notify_all();
        return nullptr;
    }
    {
        std::lock_guard<std::mutex> lock(engine->shard_mutex);
        engine->shard_mmaps[path] = ctx;
        engine->map_cv.notify_all();
    }
    return ctx;
}

static mmap_context_t* get_or_map_shard_partial(
    glm5_pc_engine* engine,
    const std::string& path,
    uint64_t offset,
    uint64_t bytes
) {
    if (!engine || path.empty() || bytes == 0 || offset > (uint64_t)SIZE_MAX || bytes > (uint64_t)SIZE_MAX) {
        return nullptr;
    }
    const std::string key = path + "#partial:" + std::to_string(offset) + ":" + std::to_string(bytes);
    {
        std::unique_lock<std::mutex> lock(engine->shard_mutex);
        while (true) {
            auto found = engine->shard_mmaps.find(key);
            if (found != engine->shard_mmaps.end()) {
                if (found->second == (mmap_context_t*)1) {
                    engine->map_cv.wait(lock);
                } else {
                    return found->second;
                }
            } else {
                engine->shard_mmaps[key] = (mmap_context_t*)1;
                break;
            }
        }
    }
    mmap_context_t* ctx = mmap_load_partial(path.c_str(), (size_t)offset, (size_t)bytes);
    if (!ctx) {
        std::lock_guard<std::mutex> lock(engine->shard_mutex);
        engine->shard_mmaps.erase(key);
        engine->map_cv.notify_all();
        return nullptr;
    }
    {
        std::lock_guard<std::mutex> lock(engine->shard_mutex);
        engine->shard_mmaps[key] = ctx;
        engine->map_cv.notify_all();
    }
    return ctx;
}

static uint64_t shard_numeric_key(uint32_t part, uint32_t shard) {
    return ((uint64_t)part << 32) | (uint64_t)shard;
}

static uint64_t shard_partial_numeric_key(uint32_t part, uint64_t offset, uint64_t bytes) {
    uint64_t x = ((uint64_t)part << 48) ^ (offset << 7) ^ (bytes >> 3);
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    return x;
}

static mmap_context_t* get_or_map_shard_partial_by_id(
    glm5_pc_engine* engine,
    uint32_t part,
    const std::string& path,
    uint64_t offset,
    uint64_t bytes
) {
    if (!engine || path.empty() || bytes == 0) {
        return nullptr;
    }
    const uint64_t id_key = shard_partial_numeric_key(part, offset, bytes);
    {
        std::lock_guard<std::mutex> lock(engine->shard_mutex);
        auto found = engine->shard_mmaps_by_id.find(id_key);
        if (found != engine->shard_mmaps_by_id.end() && found->second != (mmap_context_t*)1) {
            return found->second;
        }
    }
    mmap_context_t* mapped = get_or_map_shard_partial(engine, path, offset, bytes);
    if (mapped) {
        std::lock_guard<std::mutex> lock(engine->shard_mutex);
        engine->shard_mmaps_by_id[id_key] = mapped;
    }
    return mapped;
}

static void unload_mapped_shard(mmap_context_t* ctx) {
    if (ctx && ctx != (mmap_context_t*)1) {
        mmap_unlock(ctx);
        mmap_unload(ctx);
    }
}

static void lock_mapped_shard_once(glm5_pc_engine* engine, mmap_context_t* ctx) {
    if (!engine || !ctx || ctx == (mmap_context_t*)1) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(engine->shard_mutex);
        if (!engine->locked_mmaps.insert(ctx).second) {
            return;
        }
    }
    if (mmap_lock(ctx) != 0) {
        std::lock_guard<std::mutex> lock(engine->shard_mutex);
        engine->locked_mmaps.erase(ctx);
    }
}

static void reset_source_mmap_slots(glm5_pc_engine* engine) {
    if (!engine) {
        return;
    }
    for (auto& item : engine->source_mmaps) {
        item.store(nullptr);
    }
}

static mmap_context_t* get_or_map_tensor_shard(glm5_pc_engine* engine, const glm5_tensor_record& rec) {
    if (!engine) {
        return nullptr;
    }
    const uint32_t shard = rec.info.shard;
    if (shard < 85u) {
        mmap_context_t* cached = engine->source_mmaps[shard].load();
        if (cached) {
            return cached;
        }
    }
    const uint64_t id_key = shard_numeric_key(rec.info.part, rec.info.shard);
    {
        std::lock_guard<std::mutex> lock(engine->shard_mutex);
        auto found = engine->shard_mmaps_by_id.find(id_key);
        if (found != engine->shard_mmaps_by_id.end() && found->second != (mmap_context_t*)1) {
            return found->second;
        }
    }
    const std::string& path = (rec.resolved_path_index < engine->tensor_paths.size()) ?
        engine->tensor_paths[rec.resolved_path_index] : "";
    mmap_context_t* mapped = get_or_map_shard(engine, path);
    if (mapped && shard < 85u) {
        mmap_context_t* expected = nullptr;
        engine->source_mmaps[shard].compare_exchange_strong(expected, mapped);
    }
    if (mapped) {
        std::lock_guard<std::mutex> lock(engine->shard_mutex);
        engine->shard_mmaps_by_id[id_key] = mapped;
    }
    return mapped;
}
