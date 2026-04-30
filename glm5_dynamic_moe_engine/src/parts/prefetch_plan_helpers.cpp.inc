static uint64_t prefetch_plan_item_key(uint32_t layer, uint32_t expert) {
    return ((uint64_t)layer << 32) | (uint64_t)expert;
}

// PIPELINE FIX 9: Replace unordered_set::clear() with generation counter ★★
// Old: O(bucket_count) clear every orchestrate call (79 layers × 256 experts)
// New: O(1) generation increment, map grows bounded to ~20k entries max
// Periodic cleanup when generation wraps to prevent unbounded growth
static void prefetch_plan_clear_seen(glm5_pc_engine* engine) {
    if (!engine) {
        return;
    }
    ++engine->prefetch_plan_generation;
    // Periodic cleanup every 10000 generations to prevent map growth
    // At 100 tokens/sec, this triggers every ~100 seconds
    if (engine->prefetch_plan_generation % 10000 == 0) {
        engine->prefetch_plan_seen_gen.clear();
    }
}

static bool prefetch_plan_has_item(glm5_pc_engine* engine, int layer, int expert) {
    if (!engine || layer < 0 || expert < 0) {
        return false;
    }
    const uint64_t key = prefetch_plan_item_key((uint32_t)layer, (uint32_t)expert);
    auto it = engine->prefetch_plan_seen_gen.find(key);
    return it != engine->prefetch_plan_seen_gen.end() &&
           it->second == engine->prefetch_plan_generation;
}

static void prefetch_plan_mark_seen(glm5_pc_engine* engine, int layer, int expert) {
    if (!engine || layer < 0 || expert < 0) {
        return;
    }
    engine->prefetch_plan_seen_gen[prefetch_plan_item_key((uint32_t)layer, (uint32_t)expert)] =
        engine->prefetch_plan_generation;
}

static void prefetch_plan_add_item(
    glm5_pc_engine* engine,
    std::vector<glm5_expert_prefetch_item_t>& items,
    int layer,
    int expert,
    glm5_tier_t tier,
    glm5_prefetch_reason_t reason,
    float priority
) {
    if (layer < 0 || expert < 0 || prefetch_plan_has_item(engine, layer, expert)) {
        return;
    }
    const uint64_t bytes = expert_triplet_or_hardcoded_bytes(engine, layer, expert);
    if (bytes == 0) {
        return;
    }
    glm5_expert_prefetch_item_t item{};
    item.layer = (uint32_t)layer;
    item.expert = (uint32_t)expert;
    item.bytes = bytes;
    item.target_tier = tier;
    item.reason = reason;
    item.priority = priority;
    items.push_back(item);
    prefetch_plan_mark_seen(engine, layer, expert);
}

static glm5_tier_t prefetch_plan_selected_tier(const glm5_pc_engine* engine) {
    return caps_gpu_backend_available(engine->backend_caps) && vram_expert_budget_bytes(engine) > 0 ?
        GLM5_TIER_VRAM : GLM5_TIER_RAM;
}

static void prefetch_plan_add_selected(
    glm5_pc_engine* engine,
    std::vector<glm5_expert_prefetch_item_t>& items,
    int layer,
    const int* experts,
    uint32_t count
) {
    if (!experts) {
        return;
    }
    const glm5_tier_t tier = prefetch_plan_selected_tier(engine);
    for (uint32_t i = 0; i < count; ++i) {
        prefetch_plan_add_item(engine, items, layer, experts[i], tier, GLM5_PREFETCH_REASON_ROUTER_SELECTED, 1000.0f);
    }
}

static void prefetch_plan_add_lookahead(
    glm5_pc_engine* engine,
    std::vector<glm5_expert_prefetch_item_t>& items,
    int next_layer,
    const int* experts,
    uint32_t count
) {
    if (!engine->io_config.enable_lookahead_prefetch || !experts) {
        return;
    }
    const glm5_tier_t tier = prefetch_plan_selected_tier(engine);
    for (uint32_t i = 0; i < count; ++i) {
        prefetch_plan_add_item(engine, items, next_layer, experts[i], tier, GLM5_PREFETCH_REASON_LOOKAHEAD, 100.0f);
    }
}

static void prefetch_plan_sort_items(std::vector<glm5_expert_prefetch_item_t>& items) {
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
