static void glm5_enqueue_attention_prefetch(glm5_pc_engine* engine, int32_t layer) {
    if (!engine || layer < 0) {
        return;
    }
    const glm5_raw_tensor_kind_t kinds[] = {
        GLM5_RAW_TENSOR_Q_A_PROJ,
        GLM5_RAW_TENSOR_Q_B_PROJ,
        GLM5_RAW_TENSOR_KV_A_PROJ_WITH_MQA,
        GLM5_RAW_TENSOR_KV_B_PROJ,
        GLM5_RAW_TENSOR_O_PROJ
    };
    uint32_t queued_spans[sizeof(kinds) / sizeof(kinds[0]) + 1u] = {};
    uint32_t queued_count = 0;

    const glm5_storage_constants_t* c = glm5_storage_constants();
    if (c && layer >= c->first_moe_layer && layer <= c->last_moe_layer) {
        const glm5_storage_raw_span_spec_t* router_span = find_raw_role_span(layer, GLM5_RAW_ROLE_ROUTER);
        if (router_span) {
            const uint32_t begin = glm5_storage_raw_span_find_first(layer);
            const uint32_t end = glm5_storage_raw_span_find_end(layer);
            for (uint32_t si = begin; si < end; ++si) {
                if (glm5_storage_raw_span_at(si) == router_span) {
                    queued_spans[queued_count++] = si;
                    enqueue_raw_span_prefetch(engine, si, GLM5_DEVICE_POOL_ACTIVATION);
                    break;
                }
            }
        }
    }

    for (glm5_raw_tensor_kind_t kind : kinds) {
        glm5_raw_tensor_desc desc{};
        const glm5_storage_raw_span_spec_t* span = nullptr;
        if (!find_raw_tensor_desc_and_span(layer, kind, &desc, &span) || !span) {
            continue;
        }
        const uint32_t begin = glm5_storage_raw_span_find_first(layer);
        const uint32_t end = glm5_storage_raw_span_find_end(layer);
        for (uint32_t si = begin; si < end; ++si) {
            if (glm5_storage_raw_span_at(si) != span) {
                continue;
            }
            bool seen = false;
            for (uint32_t qi = 0; qi < queued_count; ++qi) {
                if (queued_spans[qi] == si) {
                    seen = true;
                    break;
                }
            }
            if (!seen && queued_count < sizeof(queued_spans) / sizeof(queued_spans[0])) {
                queued_spans[queued_count++] = si;
                enqueue_raw_span_prefetch(engine, si, GLM5_DEVICE_POOL_ACTIVATION);
            }
            break;
        }
    }
}

static uint64_t glm5_attention_prefetch_bytes(int32_t layer) {
    if (layer < 0) {
        return 0;
    }
    const glm5_raw_tensor_kind_t kinds[] = {
        GLM5_RAW_TENSOR_Q_A_PROJ,
        GLM5_RAW_TENSOR_Q_B_PROJ,
        GLM5_RAW_TENSOR_KV_A_PROJ_WITH_MQA,
        GLM5_RAW_TENSOR_KV_B_PROJ,
        GLM5_RAW_TENSOR_O_PROJ
    };
    const glm5_storage_raw_span_spec_t* spans[sizeof(kinds) / sizeof(kinds[0])] = {};
    uint32_t span_count = 0;
    uint64_t bytes = 0;
    for (glm5_raw_tensor_kind_t kind : kinds) {
        glm5_raw_tensor_desc desc{};
        const glm5_storage_raw_span_spec_t* span = nullptr;
        if (!find_raw_tensor_desc_and_span(layer, kind, &desc, &span) || !span) {
            continue;
        }
        bool seen = false;
        for (uint32_t i = 0; i < span_count; ++i) {
            if (spans[i] == span) {
                seen = true;
                break;
            }
        }
        if (!seen && span_count < sizeof(spans) / sizeof(spans[0])) {
            spans[span_count++] = span;
            bytes += span->bytes;
        }
    }
    return bytes;
}

static int glm5_can_enqueue_attention_lookahead(glm5_pc_engine* engine, int32_t layer) {
    if (!engine || layer < 0 || engine->config.vram_budget_bytes == 0) {
        return 1;
    }
    const uint64_t bytes = glm5_attention_prefetch_bytes(layer);
    if (bytes == 0) {
        return 1;
    }
    const uint64_t used = std::max(
        tier_used(engine, GLM5_TIER_VRAM),
        engine->device_allocated_bytes.load(std::memory_order_relaxed));
    const uint64_t budget = engine->config.vram_budget_bytes;
    // PIPELINE FIX 19: Raise attention lookahead threshold from 83% to 85% ★★
    // Old: threshold = budget * 5/6 ≈ 83% → dead zone between 83%-90%
    // New: threshold = budget * 85/100 = 85% → aligns better with proactive eviction (90%)
    // Prevents attention prefetch cutoff before expert eviction starts
    // Dead zone reduced from 7% to 5%, improves attention prefetch opportunities
    const uint64_t threshold = (budget * 85ull) / 100ull;
    return used < threshold && bytes <= threshold - used;
}
