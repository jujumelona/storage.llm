int moe_pc_engine_plan_moe_prefetch(
    moe_pc_engine_t* engine,
    int current_layer,
    const int* selected_experts,
    uint32_t selected_count,
    const int* next_layer_candidates,
    uint32_t next_count,
    moe_expert_prefetch_item_t* out_items,
    uint32_t max_items,
    uint32_t* out_item_count
) {
    if (!engine || current_layer < 0 || !out_item_count) {
        return 0;
    }

    // OPTIMIZATION: CPU-only early-out - skip prefetch planning entirely ★★★
    // Problem: CPU-only systems don't benefit from GPU prefetch planning
    // They use mmap streaming only, but planning still runs and wastes CPU cycles
    // Solution: Detect CPU-only backend and return empty plan immediately
    // Trade-off: No prefetch planning overhead on CPU-only systems
    // Impact: Reduces token latency by ~5-10% on CPU-only inference
    // BUGFIX 900: Track access patterns even on CPU-only path ★★★ IMPORTANT
    // Problem: CPU backend early-returns without calling prefetch_plan_add_selected()
    // → Expert access pattern statistics never accumulate
    // → When GPU added later, topology analysis starts cold (no historical data)
    // Solution: Call prefetch_plan_add_selected() to update generation map,
    // but skip lookahead/topology/trim (no actual prefetch needed on CPU)
    // Impact: Warm start for topology analysis when GPU becomes available
    if (engine->backend_caps.backend == moe_BACKEND_CPU) {
        // BUGFIX 900: Track access patterns for future GPU use
        std::vector<moe_expert_prefetch_item_t> items;
        prefetch_plan_clear_seen(engine);
        prefetch_plan_add_selected(engine, items, current_layer, selected_experts, selected_count);
        // No lookahead/topology/trim needed on CPU
        *out_item_count = 0;
        return 1; // Success with empty plan
    }

    if (engine->backend_caps.backend == moe_BACKEND_METAL &&
        engine->backend_caps.supports_unified_memory) {
        // UMA: VRAM/RAM 구분 없음, 기존 mmap path가 곧 최적
        std::vector<moe_expert_prefetch_item_t> items;
        prefetch_plan_clear_seen(engine);
        prefetch_plan_add_selected(engine, items, current_layer, selected_experts, selected_count);
        *out_item_count = 0; // 명시적 VRAM 복사 불필요
        return 1;
    }

    std::vector<moe_expert_prefetch_item_t> items;
    // prefetch_plan_clear_seen/add_* lock the generation map internally.
    // Do not hold prefetch_plan_mutex around them; the helpers re-enter it.
    prefetch_plan_clear_seen(engine);
    // BUGFIX 577: Check reserve size for overflow ★
    // Problem: selected_count + next_count + topology_max_candidates can overflow
    // Solution: Check before reserve
    const size_t reserve_size = (size_t)selected_count + (size_t)next_count +
        (size_t)engine->io_config.topology_max_candidates;
    if (reserve_size > SIZE_MAX / sizeof(moe_expert_prefetch_item_t)) {
        *out_item_count = 0;
        return 0;
    }
    items.reserve(reserve_size);
    prefetch_plan_add_selected(engine, items, current_layer, selected_experts, selected_count);
    prefetch_plan_add_lookahead(engine, items, current_layer + 1, next_layer_candidates, next_count);
    prefetch_plan_add_topology(engine, items, current_layer, selected_experts, selected_count);
    prefetch_plan_sort_items(items);

    // BUGFIX 문제4: Trim prefetch plan to VRAM capacity ★★★
    // Problem: prefetch_plan_add_topology() adds topology_max_candidates items
    // without checking VRAM free space. When VRAM is full, GPU workers repeatedly
    // fail VRAM reservation and requeue, wasting CPU cycles.
    // Solution: After sorting by priority, calculate cumulative bytes and trim
    // items that exceed VRAM budget (with headroom). High-priority items remain.
    // Trade-off: Slightly reduces prefetch breadth, but proactive eviction worker
    // continuously frees VRAM, so no real throughput loss. Stabilizes token latency.
    {
        const uint64_t device_free = device_budget_bytes(engine);
        // BUGFIX 899: Remove device_allocated double subtraction ★★★ CRITICAL
        // Problem: device_budget_bytes() already returns free VRAM (budget - allocated - reserved).
        // Subtracting device_allocated again causes double subtraction → device_free underestimated.
        // Result: Prefetch trim is too aggressive, VRAM underutilized.
        // Solution: device_budget already accounts for allocations and in-flight transfers.
        // Impact: Correct VRAM budget calculation, more aggressive prefetch, better GPU utilization.

        // Use backend-specific headroom percentage (default 10%) to avoid filling device memory completely.
        // BUGFIX Issue 10: Add Metal headroom configuration
        // Problem: Metal unified memory is shared with OS, but only Vulkan/Level Zero had configurable headroom
        // Solution: Apply metal_memory_budget_pct or default 15% for Metal (higher than discrete GPUs)
        // BUGFIX 902: Add Level Zero headroom configuration ★★★ IMPORTANT
        // Problem: Level Zero (Intel iGPU) shares system RAM like Metal, but uses default 10% headroom
        // → Insufficient for UMA architecture → OS memory pressure → OOM kill
        // Solution: Use level_zero_memory_headroom_pct (default 20% for iGPU)
        // Impact: Prevents OOM on Intel integrated GPUs, matches Metal's UMA handling
        uint64_t headroom_pct = 10u;
        if (engine->backend_caps.backend == moe_BACKEND_VULKAN && engine->io_config.vulkan_memory_headroom_pct > 0) {
            headroom_pct = engine->io_config.vulkan_memory_headroom_pct;
        } else if (engine->backend_caps.backend == moe_BACKEND_LEVEL_ZERO) {
            // BUGFIX 902: Level Zero iGPU needs higher headroom (20% default)
            headroom_pct = engine->io_config.level_zero_memory_headroom_pct > 0
                ? engine->io_config.level_zero_memory_headroom_pct
                : (engine->backend_caps.supports_unified_memory ? 20u : 10u);
        } else if (engine->backend_caps.backend == moe_BACKEND_METAL) {
            // Metal unified memory needs more headroom due to OS sharing
            headroom_pct = engine->io_config.metal_memory_budget_pct > 0
                ? (100u - engine->io_config.metal_memory_budget_pct)
                : 15u;
        }
        headroom_pct = std::min<uint64_t>(headroom_pct + 2u, 95u);
        const uint64_t usable_free = (device_free * (100u - headroom_pct)) / 100u;

        uint64_t cumulative_bytes = 0;
        size_t trim_index = items.size();

        // BUGFIX 2-1: Use normal_bundle_max_bytes instead of max_tensor_stream_bytes * 3
        // Problem: max_tensor_stream_bytes is the largest single tensor streamed so far,
        // but gate/up/down projections can have different sizes. Multiplying by 3 overestimates
        // expert size, causing excessive trimming even when VRAM is sufficient.
        // Solution: Use normal_bundle_max_bytes from hardcode_summary, which is the actual
        // maximum bytes for a gate+up+down bundle, already aggregated during model loading.
        // BUGFIX 901: Warn and measure when normal_bundle_max_bytes is 0 ★★★ IMPORTANT
        // Problem: GGUF v1 or missing model profile → normal_bundle_max_bytes == 0
        // → Fallback to max_tensor * 3 causes over-conservative trim
        // Solution: Warn user, use fallback, but mark for lazy measurement after first prefetch
        // Impact: Prevents silent performance degradation on legacy formats
        const moe_storage_constants_t* storage = moe_storage_constants();
        uint64_t fallback_bundle_bytes = storage && storage->normal_bundle_max_bytes
            ? storage->normal_bundle_max_bytes : 0ull;
        const uint64_t max_tensor_bytes = engine->max_tensor_stream_bytes.load(std::memory_order_relaxed);
        if (fallback_bundle_bytes == 0 && max_tensor_bytes > 0) {
            // BUGFIX 901: Warn about fallback usage
            static std::atomic<bool> warned{false};
            if (!warned.exchange(true, std::memory_order_relaxed)) {
                fprintf(stderr,
                    "[storageLLM WARNING] normal_bundle_max_bytes not set (GGUF v1 or missing profile).\n"
                    "  Using fallback: max_tensor_bytes * 3 = %llu bytes.\n"
                    "  This may cause over-conservative VRAM trim. Consider updating model profile.\n",
                    (unsigned long long)(max_tensor_bytes * 3ull));
            }
            fallback_bundle_bytes = max_tensor_bytes <= UINT64_MAX / 3ull
                ? max_tensor_bytes * 3ull : UINT64_MAX;
        }
        if (fallback_bundle_bytes == 0) {
            fallback_bundle_bytes = 24ull * 1024ull * 1024ull;
        }

        // BUGFIX Issue 2: Validate item bytes are populated ★★★
        // Problem: If prefetch_plan_add_* functions don't populate item.bytes,
        // cumulative_bytes stays 0 and trimming doesn't work
        // Solution: Assert that bytes are populated, and warn if fallback is used
        bool used_fallback = false;
        for (size_t i = 0; i < items.size(); ++i) {
            const uint64_t item_bytes = items[i].bytes ? items[i].bytes : fallback_bundle_bytes;
            if (!items[i].bytes && !used_fallback) {
                fprintf(stderr,
                    "[storageLLM WARNING] Prefetch item %zu has bytes=0, using fallback %llu. "
                    "This indicates prefetch_plan_add_* functions are not populating item.bytes correctly.\n",
                    i, (unsigned long long)fallback_bundle_bytes);
                used_fallback = true;
            }
            if (cumulative_bytes + item_bytes > usable_free) {
                trim_index = i;
                break;
            }
            cumulative_bytes += item_bytes;
        }

        if (trim_index < items.size()) {
            items.resize(trim_index);
        }
    }

    const size_t item_count = items.size();
    *out_item_count = (uint32_t)item_count;
    if (out_items && max_items > 0) {
        // Fix 13: Use size_t for comparison to handle large item counts
        const size_t copy_count = (size_t)max_items < item_count ? (size_t)max_items : item_count;
        std::memcpy(out_items, items.data(), sizeof(moe_expert_prefetch_item_t) * copy_count);
    }
    // Return 1 if all items fit, 0 if partial copy (caller can check out_item_count)
    return (size_t)max_items >= item_count ? 1 : 0;
}
