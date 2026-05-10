// Raw span lookup and mapping helpers

static const moe_storage_raw_span_spec_t* find_raw_role_span(int32_t layer, uint32_t role) {
    const uint32_t begin = moe_storage_raw_span_find_first(layer);
    const uint32_t end = moe_storage_raw_span_find_end(layer);
    for (uint32_t i = begin; i < end; ++i) {
        const moe_storage_raw_span_spec_t* s = moe_storage_raw_span_at(i);
        if (s && (s->role & role) != 0 && s->bytes > 0) {
            return s;
        }
    }
    return nullptr;
}

static mmap_context_t* map_raw_span_part(moe_pc_engine* engine, const moe_storage_raw_span_spec_t* span) {
    if (!engine || !span || engine->model_root.empty()) {
        return nullptr;
    }
    const moe_storage_part_spec_t* part = moe_storage_part_by_id(span->part);
    return part && part->path ? get_or_map_shard_partial_by_id(
        engine,
        span->part,
        resolve_model_file_path(engine->model_root, part->path, ""),
        span->offset,
        span->bytes
    ) : nullptr;
}

static int raw_span_bounds_ok(const moe_storage_raw_span_spec_t* span, const mmap_context_t* mapped) {
    return span && mapped && mmap_get_ptr(mapped, 0) &&
        span->bytes <= (uint64_t)mmap_get_size(mapped);
}

static const uint8_t* raw_span_ptr(
    const moe_storage_raw_span_spec_t* span,
    const mmap_context_t* mapped,
    uint64_t rel_offset,
    uint64_t bytes
) {
    if (!raw_span_bounds_ok(span, mapped)) {
        return nullptr;
    }
    const uint64_t mapped_size = (uint64_t)mmap_get_size(mapped);
    if (rel_offset > mapped_size || bytes > mapped_size - rel_offset || rel_offset > (uint64_t)SIZE_MAX) {
        return nullptr;
    }
    return static_cast<const uint8_t*>(mmap_get_ptr(mapped, (size_t)rel_offset));
}

static const moe_storage_raw_span_spec_t* find_raw_role_span_bytes(
    int32_t layer,
    uint32_t role,
    uint64_t bytes,
    uint32_t occurrence
) {
    const uint32_t begin = moe_storage_raw_span_find_first(layer);
    const uint32_t end = moe_storage_raw_span_find_end(layer);
    uint32_t seen = 0;
    for (uint32_t i = begin; i < end; ++i) {
        const moe_storage_raw_span_spec_t* s = moe_storage_raw_span_at(i);
        if (s && (s->role & role) != 0 && s->bytes == bytes) {
            if (seen == occurrence) {
                return s;
            }
            ++seen;
        }
    }
    return nullptr;
}

static int raw_span_backing_file_ready(
    const moe_pc_engine* engine,
    const moe_storage_raw_span_spec_t* span
) {
    if (!engine || engine->model_root.empty() || !span) {
        return 0;
    }
    const moe_storage_part_spec_t* part = moe_storage_part_by_id(span->part);
    if (!part || !part->path) {
        return 0;
    }
    uint64_t file_bytes = 0;
    {
        std::lock_guard<std::mutex> lock(engine->raw_file_size_cache_mutex);
        auto it = engine->raw_file_size_cache.find(span->part);
        if (it != engine->raw_file_size_cache.end()) {
            file_bytes = it->second;
        } else {
            const std::string path = resolve_model_file_path(engine->model_root, part->path, "");
            if (!file_size_bytes(path, &file_bytes)) {
                return 0;
            }
            engine->raw_file_size_cache[span->part] = file_bytes;
        }
    }
    return span->offset <= file_bytes && span->bytes <= file_bytes - span->offset ? 1 : 0;
}
