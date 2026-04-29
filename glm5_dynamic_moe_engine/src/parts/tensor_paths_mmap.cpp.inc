// Memory-mapped shard management

static std::shared_ptr<std::condition_variable_any> shard_mmap_cv_locked(
    glm5_pc_engine* engine,
    const std::string& key
) {
    auto& cv = engine->shard_mmap_cvs[key];
    if (!cv) {
        cv = std::make_shared<std::condition_variable_any>();
    }
    return cv;
}

static std::shared_ptr<std::condition_variable_any> shard_mmap_take_cv_locked(
    glm5_pc_engine* engine,
    const std::string& key
) {
    auto found = engine->shard_mmap_cvs.find(key);
    if (found == engine->shard_mmap_cvs.end()) {
        return nullptr;
    }
    auto cv = found->second;
    engine->shard_mmap_cvs.erase(found);
    return cv;
}

static mmap_context_t* get_or_map_shard(glm5_pc_engine* engine, const std::string& path) {
    if (!engine || path.empty()) {
        return nullptr;
    }
    {
        std::shared_lock<std::shared_mutex> lock(engine->shard_mutex);
        auto found = engine->shard_mmaps.find(path);
        if (found != engine->shard_mmaps.end() && found->second != (mmap_context_t*)1) {
            return found->second;
        }
    }
    {
        std::unique_lock<std::shared_mutex> lock(engine->shard_mutex);
        while (true) {
            auto found = engine->shard_mmaps.find(path);
            if (found != engine->shard_mmaps.end()) {
                if (found->second == (mmap_context_t*)1) {
                    auto cv = shard_mmap_cv_locked(engine, path);
                    cv->wait(lock);
                } else {
                    auto cv_it = engine->shard_mmap_cvs.find(path);
                    if (cv_it != engine->shard_mmap_cvs.end() && cv_it->second) {
                        cv_it->second->notify_one();
                    }
                    return found->second;
                }
            } else {
                engine->shard_mmaps[path] = (mmap_context_t*)1;
                (void)shard_mmap_cv_locked(engine, path);
                break;
            }
        }
    }
    mmap_context_t* ctx = mmap_load(path.c_str());
    if (!ctx) {
        std::shared_ptr<std::condition_variable_any> cv;
        {
            std::unique_lock<std::shared_mutex> lock(engine->shard_mutex);
            engine->shard_mmaps.erase(path);
            cv = shard_mmap_take_cv_locked(engine, path);
        }
        if (cv) {
            cv->notify_all();
        }
        return nullptr;
    }
    std::shared_ptr<std::condition_variable_any> cv;
    {
        std::unique_lock<std::shared_mutex> lock(engine->shard_mutex);
        engine->shard_mmaps[path] = ctx;
        cv = shard_mmap_cv_locked(engine, path);
    }
    if (cv) {
        cv->notify_one();
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
        std::shared_lock<std::shared_mutex> lock(engine->shard_mutex);
        auto found = engine->shard_mmaps.find(key);
        if (found != engine->shard_mmaps.end() && found->second != (mmap_context_t*)1) {
            return found->second;
        }
    }
    {
        std::unique_lock<std::shared_mutex> lock(engine->shard_mutex);
        while (true) {
            auto found = engine->shard_mmaps.find(key);
            if (found != engine->shard_mmaps.end()) {
                if (found->second == (mmap_context_t*)1) {
                    auto cv = shard_mmap_cv_locked(engine, key);
                    cv->wait(lock);
                } else {
                    auto cv_it = engine->shard_mmap_cvs.find(key);
                    if (cv_it != engine->shard_mmap_cvs.end() && cv_it->second) {
                        cv_it->second->notify_one();
                    }
                    return found->second;
                }
            } else {
                engine->shard_mmaps[key] = (mmap_context_t*)1;
                (void)shard_mmap_cv_locked(engine, key);
                break;
            }
        }
    }
    mmap_context_t* ctx = mmap_load_partial(path.c_str(), (size_t)offset, (size_t)bytes);
    if (!ctx) {
        std::shared_ptr<std::condition_variable_any> cv;
        {
            std::unique_lock<std::shared_mutex> lock(engine->shard_mutex);
            engine->shard_mmaps.erase(key);
            cv = shard_mmap_take_cv_locked(engine, key);
        }
        if (cv) {
            cv->notify_all();
        }
        return nullptr;
    }
    std::shared_ptr<std::condition_variable_any> cv;
    {
        std::unique_lock<std::shared_mutex> lock(engine->shard_mutex);
        engine->shard_mmaps[key] = ctx;
        cv = shard_mmap_cv_locked(engine, key);
    }
    if (cv) {
        cv->notify_one();
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
        std::shared_lock<std::shared_mutex> lock(engine->shard_mutex);
        auto found = engine->shard_mmaps_by_id.find(id_key);
        if (found != engine->shard_mmaps_by_id.end() && found->second != (mmap_context_t*)1) {
            return found->second;
        }
    }
    mmap_context_t* mapped = get_or_map_shard_partial(engine, path, offset, bytes);
    if (mapped) {
        std::unique_lock<std::shared_mutex> lock(engine->shard_mutex);
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
        std::unique_lock<std::shared_mutex> lock(engine->shard_mutex);
        if (!engine->locked_mmaps.insert(ctx).second) {
            return;
        }
    }
    if (mmap_lock(ctx) != 0) {
        std::unique_lock<std::shared_mutex> lock(engine->shard_mutex);
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
        std::shared_lock<std::shared_mutex> lock(engine->shard_mutex);
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
        std::unique_lock<std::shared_mutex> lock(engine->shard_mutex);
        engine->shard_mmaps_by_id[id_key] = mapped;
    }
    return mapped;
}
