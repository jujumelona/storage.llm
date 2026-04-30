static void moe_enqueue_attention_prefetch(moe_pc_engine* engine, int32_t layer) {
    if (!engine || layer < 0) {
        return;
    }

    // BUGFIX 30: Use O(1) attention span index table instead of O(N) linear search ?�★??    // Old: Loop through all spans (begin..end) for each kind, pointer comparison ??O(N) per kind
    // New: Direct table lookup moe_storage_attention_span_index ??O(1) per kind
    // Eliminates 5 kinds × 79 layers × O(N) pointer comparisons per token generation
    // Hot path optimization per ARCHITECTURE.md: "integer id and offset, no strings"

    // kind_index mapping: 0=Q_A_PROJ, 1=Q_B_PROJ, 2=KV_A_PROJ_WITH_MQA, 3=KV_B_PROJ, 4=O_PROJ
    const moe_raw_tensor_kind_t kinds[] = {
        moe_RAW_TENSOR_Q_A_PROJ,
        moe_RAW_TENSOR_Q_B_PROJ,
        moe_RAW_TENSOR_KV_A_PROJ_WITH_MQA,
        moe_RAW_TENSOR_KV_B_PROJ,
        moe_RAW_TENSOR_O_PROJ
    };
    uint32_t queued_spans[sizeof(kinds) / sizeof(kinds[0]) + 1u] = {};
    uint32_t queued_count = 0;

    const moe_storage_constants_t* c = moe_storage_constants();
    if (c && layer >= c->first_moe_layer && layer <= c->last_moe_layer) {
        const uint32_t router_si = moe_storage_router_span_index(layer);
        if (router_si != 0xFFFFFFFFu) {
            queued_spans[queued_count++] = router_si;
            enqueue_raw_span_prefetch(engine, router_si, moe_DEVICE_POOL_ACTIVATION);
        }
    }

    for (uint32_t kind_idx = 0; kind_idx < sizeof(kinds) / sizeof(kinds[0]); ++kind_idx) {
        const uint32_t si = moe_storage_attention_span_index(layer, kind_idx);
        if (si == 0xFFFFFFFFu) {
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
            enqueue_raw_span_prefetch(engine, si, moe_DEVICE_POOL_ACTIVATION);
        }
    }
}

static uint64_t moe_attention_prefetch_bytes(int32_t layer) {
    if (layer < 0) {
        return 0;
    }

    // BUGFIX 30: Use O(1) attention span index table ?�★??    // Same optimization as moe_enqueue_attention_prefetch
    uint64_t bytes = 0;
    uint32_t seen_spans[5] = {};
    uint32_t seen_count = 0;

    for (uint32_t kind_idx = 0; kind_idx < 5; ++kind_idx) {
        const uint32_t si = moe_storage_attention_span_index(layer, kind_idx);
        if (si == 0xFFFFFFFFu) {
            continue;
        }
        bool seen = false;
        for (uint32_t i = 0; i < seen_count; ++i) {
            if (seen_spans[i] == si) {
                seen = true;
                break;
            }
        }
        if (!seen && seen_count < 5) {
            seen_spans[seen_count++] = si;
            const moe_storage_raw_span_spec_t* span = moe_storage_raw_span_at(si);
            if (span) {
                bytes += span->bytes;
            }
        }
    }
    return bytes;
}

static int moe_can_enqueue_attention_lookahead(moe_pc_engine* engine, int32_t layer) {
    if (!engine || layer < 0 || engine->config.vram_budget_bytes == 0) {
        return 1;
    }
    const uint64_t bytes = moe_attention_prefetch_bytes(layer);
    if (bytes == 0) {
        return 1;
    }
    // BUGFIX 23: Use expert VRAM only, not total device allocation ?�★??    // Old: device_allocated_bytes includes fixed (KV cache, embeddings) ??premature cutoff
    // New: tier_used(VRAM) only counts expert weights ??accurate threshold
    // Fixed allocations don't compete with attention prefetch, only experts do
    const uint64_t used = tier_used(engine, moe_TIER_VRAM);
    const uint64_t budget = engine->config.vram_budget_bytes;
    // PIPELINE FIX 19: Raise attention lookahead threshold from 83% to 85% ?�★
    // Old: threshold = budget * 5/6 ??83% ??dead zone between 83%-90%
    // New: threshold = budget * 85/100 = 85% ??aligns better with proactive eviction (90%)
    // Prevents attention prefetch cutoff before expert eviction starts
    // Dead zone reduced from 7% to 5%, improves attention prefetch opportunities
    // BUGFIX 99: Precompute percentage threshold
    const uint64_t threshold = (budget * 87ull) / 100ull;
    return used < threshold && bytes <= threshold - used;
}
