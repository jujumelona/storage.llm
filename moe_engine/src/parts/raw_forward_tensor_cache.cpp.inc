// Raw forward - Tensor cache management

static uint64_t raw_tensor_cache_key(int32_t layer, moe_raw_tensor_kind_t kind) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(layer)) << 32) |
        static_cast<uint32_t>(kind);
}

static int get_cached_raw_tensor(
    moe_pc_engine* engine,
    int32_t layer,
    moe_raw_tensor_kind_t kind,
    moe_raw_tensor_cache_entry* out_entry
) {
    if (!engine || !out_entry) {
        return 0;
    }
    const uint64_t key = raw_tensor_cache_key(layer, kind);
    {
        std::shared_lock<std::shared_mutex> lock(engine->raw_tensor_cache_mutex);
        auto found = engine->raw_tensor_cache.find(key);
        if (found != engine->raw_tensor_cache.end() && found->second.base) {
            *out_entry = found->second;
            return 1;
        }
    }
    moe_raw_tensor_desc desc{};
    if (const moe_gguf_common_tensor_record* rec =
            find_common_raw_tensor_record(engine, layer, kind, &desc)) {
        moe_raw_tensor_cache_entry entry{};
        if (!map_common_tensor_record(engine, *rec, desc, &entry)) {
            return 0;
        }
        {
            std::unique_lock<std::shared_mutex> lock(engine->raw_tensor_cache_mutex);
            auto found = engine->raw_tensor_cache.find(key);
            if (found != engine->raw_tensor_cache.end() && found->second.base) {
                *out_entry = found->second;
                return 1;
            }
            engine->raw_tensor_cache[key] = entry;
            *out_entry = entry;
        }
        return 1;
    }
    if (engine->offload_gguf_valid.load(std::memory_order_relaxed)) {
        moe_trace_raw_tensor_static_fallback_blocked(layer, kind);
        return 0;
    }
    const moe_storage_raw_span_spec_t* span = nullptr;
    if (!find_raw_tensor_desc_and_span(layer, kind, &desc, &span)) {
        return 0;
    }
    mmap_context_t* mapped = map_raw_span_part(engine, span);
    if (!raw_span_bounds_ok(span, mapped)) {
        return 0;
    }
    const uint8_t* base = raw_span_ptr(span, mapped, desc.rel_offset, desc.bytes);
    if (!base) {
        return 0;
    }
    moe_raw_tensor_cache_entry entry{};
    entry.span = span;
    entry.mapped = mapped;
    entry.base = base;
    entry.rel_offset = desc.rel_offset;
    entry.bytes = desc.bytes;
    entry.rows = desc.rows;
    entry.cols = desc.cols;
    entry.scalar_encoding = moe_RAW_SCALAR_BF16;
    entry.weight_format = moe_WEIGHT_ENCODING_RAW_FP16;
    entry.weight_row_bytes = desc.cols ? (uint64_t)desc.cols * 2ull : (uint64_t)desc.rows * 2ull;
    {
        std::unique_lock<std::shared_mutex> lock(engine->raw_tensor_cache_mutex);
        auto found = engine->raw_tensor_cache.find(key);
        if (found != engine->raw_tensor_cache.end() && found->second.base) {
            *out_entry = found->second;
            return 1;
        }
        engine->raw_tensor_cache[key] = entry;
        *out_entry = entry;
    }
    return 1;
}
