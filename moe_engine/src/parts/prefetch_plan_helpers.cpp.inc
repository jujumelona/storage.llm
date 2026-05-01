static uint64_t prefetch_plan_item_key(uint32_t layer, uint32_t expert) {
    return ((uint64_t)layer << 32) | (uint64_t)expert;
}

// PIPELINE FIX 9: Replace unordered_set::clear() with generation counter ?�★
// BUGFIX 78: Issue 12 - Add size-based threshold for map cleanup ?�★
// Old: Clear every 10000 generations only (~100s at 100 tokens/sec)
// Problem: Map can grow to 350k entries (19456 experts × 6 window × 3 projs)
// Long-running inference ??multi-MB map growth
// New: Also clear when map size > 50000 entries
// Impact: Prevents unbounded memory growth, occasional duplicate prefetch acceptable
static void prefetch_plan_clear_seen(moe_pc_engine* engine) {
    if (!engine) {
        return;
    }
    ++engine->prefetch_plan_generation;
    // Periodic cleanup: every 10000 generations OR when map exceeds 50k entries
    // At 100 tokens/sec, generation cleanup triggers every ~100 seconds
    // Size cleanup prevents multi-MB growth in long-running inference
    const bool should_clear =
        (engine->prefetch_plan_generation % 10000 == 0) ||
        (engine->prefetch_plan_seen_gen.size() > 50000);

    if (should_clear) {
        engine->prefetch_plan_seen_gen.clear();
    }
}

static bool prefetch_plan_has_item(moe_pc_engine* engine, int layer, int expert) {
    if (!engine || layer < 0 || expert < 0) {
        return false;
    }
    const uint64_t key = prefetch_plan_item_key((uint32_t)layer, (uint32_t)expert);
    auto it = engine->prefetch_plan_seen_gen.find(key);
    return it != engine->prefetch_plan_seen_gen.end() &&
           it->second == engine->prefetch_plan_generation;
}

static void prefetch_plan_mark_seen(moe_pc_engine* engine, int layer, int expert) {
    if (!engine || layer < 0 || expert < 0) {
        return;
    }
    engine->prefetch_plan_seen_gen[prefetch_plan_item_key((uint32_t)layer, (uint32_t)expert)] =
        engine->prefetch_plan_generation;
}

static moe_tier_t juju_hint_target_tier(
    const moe_pc_engine* engine,
    int layer,
    int expert,
    moe_tier_t requested,
    moe_prefetch_reason_t reason
) {
    const uint8_t hint = expert_residency_hint(engine, layer, expert);
    if (hint == moe_JUJU_RESIDENCY_AUTO) {
        return requested;
    }
    const bool gpu_ready = engine && caps_gpu_backend_available(engine->backend_caps) &&
        vram_expert_budget_bytes(engine) > 0;
    if (hint == moe_JUJU_RESIDENCY_FAST_MEM) {
        return gpu_ready ? moe_TIER_VRAM : moe_TIER_RAM;
    }
    if (hint == moe_JUJU_RESIDENCY_FAST_MEM_STREAMABLE) {
        if (reason == moe_PREFETCH_REASON_ROUTER_SELECTED && gpu_ready) {
            return moe_TIER_VRAM;
        }
        return moe_TIER_RAM;
    }
    if (hint == moe_JUJU_RESIDENCY_SLOW_MEM) {
        if (reason == moe_PREFETCH_REASON_ROUTER_SELECTED) {
            return gpu_ready ? requested : moe_TIER_RAM;
        }
        const uint16_t priority = expert_prefetch_priority_hint(engine, layer, expert);
        return priority >= 80 ? moe_TIER_RAM : moe_TIER_DB;
    }
    return requested;
}

static float juju_hint_priority(
    const moe_pc_engine* engine,
    int layer,
    int expert,
    moe_prefetch_reason_t reason,
    float base_priority
) {
    const uint16_t hint = expert_prefetch_priority_hint(engine, layer, expert);
    if (hint == 0) {
        return base_priority;
    }
    if (reason == moe_PREFETCH_REASON_ROUTER_SELECTED) {
        return base_priority + (float)hint;
    }
    const float multiplier = 0.50f + (float)std::min<uint16_t>(hint, 100u) / 100.0f;
    return base_priority * multiplier;
}

static void prefetch_plan_add_item(
    moe_pc_engine* engine,
    std::vector<moe_expert_prefetch_item_t>& items,
    int layer,
    int expert,
    moe_tier_t tier,
    moe_prefetch_reason_t reason,
    float priority
) {
    if (layer < 0 || expert < 0 || prefetch_plan_has_item(engine, layer, expert)) {
        return;
    }
    const uint64_t bytes = expert_triplet_or_hardcoded_bytes(engine, layer, expert);
    if (bytes == 0) {
        return;
    }
    tier = juju_hint_target_tier(engine, layer, expert, tier, reason);
    priority = juju_hint_priority(engine, layer, expert, reason, priority);
    if (tier == moe_TIER_DB && reason != moe_PREFETCH_REASON_ROUTER_SELECTED) {
        return;
    }
    moe_expert_prefetch_item_t item{};
    item.layer = (uint32_t)layer;
    item.expert = (uint32_t)expert;
    item.bytes = bytes;
    item.target_tier = tier;
    item.reason = reason;
    item.priority = priority;
    items.push_back(item);
    prefetch_plan_mark_seen(engine, layer, expert);
}

static moe_tier_t prefetch_plan_selected_tier(const moe_pc_engine* engine) {
    return caps_gpu_backend_available(engine->backend_caps) && vram_expert_budget_bytes(engine) > 0 ?
        moe_TIER_VRAM : moe_TIER_RAM;
}

static void prefetch_plan_add_selected(
    moe_pc_engine* engine,
    std::vector<moe_expert_prefetch_item_t>& items,
    int layer,
    const int* experts,
    uint32_t count
) {
    if (!experts) {
        return;
    }
    const moe_tier_t tier = prefetch_plan_selected_tier(engine);
    for (uint32_t i = 0; i < count; ++i) {
        prefetch_plan_add_item(engine, items, layer, experts[i], tier, moe_PREFETCH_REASON_ROUTER_SELECTED, 1000.0f);
    }
}

static void prefetch_plan_add_lookahead(
    moe_pc_engine* engine,
    std::vector<moe_expert_prefetch_item_t>& items,
    int next_layer,
    const int* experts,
    uint32_t count
) {
    if (!engine->io_config.enable_lookahead_prefetch || !experts || count == 0) {
        return;
    }
    if (engine->generation_phase.load(std::memory_order_relaxed) == moe_GEN_PHASE_PREFILL) {
        return;
    }
    if (count <= 2u) {
        const moe_tier_t tier = prefetch_plan_selected_tier(engine);
        for (uint32_t i = 0; i < count; ++i) {
            prefetch_plan_add_item(engine, items, next_layer, experts[i], tier, moe_PREFETCH_REASON_LOOKAHEAD, 100.0f);
        }
        return;
    }
    const uint32_t keep_count = std::max(1u, (count + 1u) / 2u);
    const uint32_t vram_count = std::max(1u, (count * 30u + 99u) / 100u);
    for (uint32_t i = 0; i < keep_count; ++i) {
        const moe_tier_t tier = i < vram_count ? moe_TIER_VRAM : moe_TIER_RAM;
        const float priority = 100.0f * (float)(count - i) / (float)count;
        prefetch_plan_add_item(engine, items, next_layer, experts[i], tier, moe_PREFETCH_REASON_LOOKAHEAD, priority);
    }
}

// Optimization 4: Score-based Prefetch Filter
static void prefetch_plan_add_lookahead_scored(
    moe_pc_engine* engine,
    std::vector<moe_expert_prefetch_item_t>& items,
    int layer,
    const int* candidates,
    const float* scores,
    uint32_t count
) {
    if (!candidates || !scores || count == 0) return;

    std::vector<float> sorted_scores(scores, scores + count);
    std::sort(sorted_scores.begin(), sorted_scores.end());
    const float p70 = sorted_scores[(size_t)(count * 0.70f)];
    const float p50 = sorted_scores[(size_t)(count * 0.50f)];

    for (uint32_t i = 0; i < count; ++i) {
        moe_tier_t target_tier;
        if (scores[i] >= p70) target_tier = moe_TIER_VRAM;
        else if (scores[i] >= p50) target_tier = moe_TIER_RAM;
        else continue; // skip lower 50%

        prefetch_plan_add_item(engine, items, layer, candidates[i], target_tier, moe_PREFETCH_REASON_LOOKAHEAD, scores[i]);
    }
}

static void prefetch_plan_sort_items(std::vector<moe_expert_prefetch_item_t>& items) {
    std::sort(items.begin(), items.end(), [](const auto& a, const auto& b) {
        if (a.priority != b.priority) {
            return a.priority > b.priority;
        }
        if (a.layer != b.layer) {
            return a.layer < b.layer;
        }
        return a.expert < b.expert;
    });
}
