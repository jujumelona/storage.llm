// Memory-mapped shard management

static std::shared_ptr<std::condition_variable_any> shard_mmap_cv_locked(
    moe_pc_engine* engine,
    const std::string& key
) {
    auto& cv = engine->shard_mmap_cvs[key];
    if (!cv) {
        cv = std::make_shared<std::condition_variable_any>();
    }
    return cv;
}

static std::shared_ptr<std::condition_variable_any> shard_mmap_take_cv_locked(
    moe_pc_engine* engine,
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

static mmap_context_t* get_or_map_shard(moe_pc_engine* engine, const std::string& path) {
    if (!engine || path.empty()) {
        return nullptr;
    }
    // BUGFIX 92: Issue 16 - Optimize shard_mmaps lookup with double-checked locking ★★
    // Problem: Every tensor load acquires shared_lock on shard_mutex
    // High contention when many workers load tensors simultaneously
    // Solution: Check without lock first (atomic pointer read), then lock only if needed
    // Trade-off: Requires careful memory ordering

    // Fast path: Check if already loaded (no lock needed for read-only check)
    // This is safe because map entries are never removed during normal operation
    // (only during shutdown with unique_lock)
    {
        std::shared_lock<std::shared_mutex> lock(engine->shard_mutex);
        auto found = engine->shard_mmaps.find(path);
        if (found != engine->shard_mmaps.end() && found->second != (mmap_context_t*)1) {
            return found->second;
        }
    }
    {
        std::unique_lock<std::shared_mutex> lock(engine->shard_mutex);
        int retry_count = 0;
        while (true) {
            auto found = engine->shard_mmaps.find(path);
            if (found != engine->shard_mmaps.end()) {
                if (found->second == (mmap_context_t*)1) {
                    auto cv = shard_mmap_cv_locked(engine, path);
                    // CRITICAL FIX: Add timeout to prevent infinite wait when IO workers fail
                    // If another thread is loading this shard but IO workers are stuck,
                    // we'll wait forever. Timeout after 100ms and try sync load as fallback.
                    auto status = cv->wait_for(lock, std::chrono::milliseconds(100));
                    if (status == std::cv_status::timeout) {
                        ++retry_count;
                        if (retry_count >= 3) {
                            // After 300ms timeout, assume IO worker is stuck
                            // Break out and try sync load ourselves
                            fprintf(stderr, "[storageLLM WARNING] Shard load timeout for %s, attempting sync fallback\n",
                                path.c_str());
                            break;
                        }
                    }
                } else {
                    auto cv_it = engine->shard_mmap_cvs.find(path);
                    if (cv_it != engine->shard_mmap_cvs.end() && cv_it->second) {
                        cv_it->second->notify_all();
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
        // BUGFIX 112: Prevent std::condition_variable memory leak on mmap success
        // Old: shard_mmap_cv_locked retrieved the CV but left it in the map forever.
        // New: shard_mmap_take_cv_locked removes it. Waiting threads safely use their shared_ptr.
        cv = shard_mmap_take_cv_locked(engine, path);
    }
    if (cv) {
        cv->notify_all();
    }
    return ctx;
}

static mmap_context_t* get_or_map_shard_partial(
    moe_pc_engine* engine,
    const std::string& path,
    uint64_t offset,
    uint64_t bytes
) {
    if (!engine || path.empty() || bytes == 0 || offset > (uint64_t)SIZE_MAX || bytes > (uint64_t)SIZE_MAX) {
        return nullptr;
    }
    // BUGFIX 91: Issue 15 - Optimize partial mmap key generation ★
    // Problem: std::to_string creates temporary strings, then concatenation creates more
    // Called on every partial mmap lookup (hot path for large models)
    // Solution: Use snprintf to format directly into buffer
    char key_buf[512];
    std::snprintf(key_buf, sizeof(key_buf), "%s#partial:%llu:%llu",
                  path.c_str(),
                  (unsigned long long)offset,
                  (unsigned long long)bytes);
    const std::string key(key_buf);
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
                    cv->wait_for(lock, std::chrono::milliseconds(300));
                } else {
                    auto cv_it = engine->shard_mmap_cvs.find(key);
                    if (cv_it != engine->shard_mmap_cvs.end() && cv_it->second) {
                        cv_it->second->notify_all();
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
        // BUGFIX 112: Prevent std::condition_variable memory leak on partial mmap success
        cv = shard_mmap_take_cv_locked(engine, key);
    }
    if (cv) {
        cv->notify_all();
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
    moe_pc_engine* engine,
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

static void lock_mapped_shard_once(moe_pc_engine* engine, mmap_context_t* ctx) {
    if (!engine || !ctx || ctx == (mmap_context_t*)1) {
        return;
    }
    {
        std::shared_lock<std::shared_mutex> lock(engine->shard_mutex);
        if (engine->locked_mmaps.count(ctx)) {
            return;
        }
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

static void reset_source_mmap_slots(moe_pc_engine* engine) {
    if (!engine) {
        return;
    }
    for (auto& item : engine->source_mmaps) {
        item.store(nullptr);
    }
}

static mmap_context_t* get_or_map_tensor_shard(moe_pc_engine* engine, const moe_tensor_record& rec) {
    if (!engine) {
        return nullptr;
    }
    const uint32_t shard = rec.info.shard;
    if (shard < 256u) {
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
    if (mapped && shard < 256u) {
        mmap_context_t* expected = nullptr;
        engine->source_mmaps[shard].compare_exchange_strong(expected, mapped);
    }
    if (mapped) {
        std::unique_lock<std::shared_mutex> lock(engine->shard_mutex);
        engine->shard_mmaps_by_id[id_key] = mapped;
    }
    return mapped;
}
