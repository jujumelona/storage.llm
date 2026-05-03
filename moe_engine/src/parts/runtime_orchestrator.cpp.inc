static float policy_cpu_target(moe_execution_policy_t policy) {
    return policy == moe_EXECUTION_PERFORMANCE ? 92.0f : policy == moe_EXECUTION_LOW_IMPACT ? 45.0f : 70.0f;
}

static float policy_gpu_target(moe_execution_policy_t policy) {
    return policy == moe_EXECUTION_PERFORMANCE ? 96.0f : policy == moe_EXECUTION_LOW_IMPACT ? 60.0f : 85.0f;
}

static void refresh_external_gpu_memory_budget_if_due(moe_pc_engine* engine) {
    if (!engine) {
        return;
    }
    if (engine->backend_caps.backend == moe_BACKEND_VULKAN ||
        engine->backend_caps.backend == moe_BACKEND_LEVEL_ZERO ||
        engine->backend_caps.backend == moe_BACKEND_SYCL) {
        device_refresh_mem_info(engine, 0);
    }
}

static void fill_orchestrator_plan(moe_pc_engine* engine, const moe_runtime_orchestrator_state_t* state,
    moe_runtime_orchestrator_plan_t* plan) {
    const moe_execution_policy_t policy = state ? state->policy : moe_EXECUTION_BALANCED;
    static const uint32_t hw = std::max<uint32_t>(moe_cpu_thread_count_hint(), 4u);
    const uint32_t perf_disk_workers = moe_default_disk_worker_threads(hw);
    // Fix 3: Allow up to 3 GPU workers in PERFORMANCE mode to pipeline
    // blocking cuStreamSynchronize calls and improve PCIe utilization.
    const uint32_t perf_gpu_workers = engine && engine->backend_caps.copy_streams > 1
        ? std::min(8u, (uint32_t)engine->backend_caps.copy_streams) : 1u;
    std::memset(plan, 0, sizeof(*plan));
    plan->policy = policy;
    plan->disk_workers = policy == moe_EXECUTION_PERFORMANCE ? perf_disk_workers : policy == moe_EXECUTION_LOW_IMPACT ? 1u : 2u;
    plan->pinned_workers = policy == moe_EXECUTION_PERFORMANCE ? std::max(2u, perf_gpu_workers) : 1u;
    plan->gpu_workers = policy == moe_EXECUTION_PERFORMANCE ? perf_gpu_workers : 1u;
    const int juju_active = engine && engine->juju_strategy_valid.load(std::memory_order_relaxed);
    const moe_juju_pipeline_config_t* juju_cfg = juju_active ? &engine->juju_pipeline_config : nullptr;
    if (juju_cfg && juju_cfg->io_threads > 0) {
        const uint32_t disk_cap = disk_worker_cap();
        const uint32_t pinned_cap = pinned_worker_cap(engine);
        const uint32_t gpu_cap = gpu_worker_cap(engine);
        plan->disk_workers = clamp_to_cap(juju_cfg->io_threads, disk_cap);
        plan->pinned_workers = pinned_cap ?
            clamp_to_cap(std::max(1u, (juju_cfg->io_threads + 1u) / 2u), pinned_cap) : 0u;
        plan->gpu_workers = gpu_cap ?
            clamp_to_cap(std::min(8u, std::max(1u, engine->backend_caps.copy_streams)), gpu_cap) : 0u;
    }

    // BUGFIX 59: Issue 5 - dropped_requests feedback loop ?�★??
    // Problem: local_requeues full ??tasks dropped ??dropped_requests++
    // But orchestrator never reads dropped_requests ??window never shrinks
    // Drops continue indefinitely, prefetch effectiveness collapses
    // Solution: Check recent drops, immediately reduce window (asymmetric hysteresis)
    // Upside changes need 5-streak, downside (drops) react immediately
    //
    // BUGFIX 74: Issue 7 - Faster prefetch window recovery ?�★
    // Problem: After drops stop, recovery waits for 5-streak (can take dozens of tokens)
    // Orchestrator call interval is long ??window=1 persists too long
    // Solution: Reduce recovery streak from 5 to 3 for faster adaptation
    // Asymmetric: drops ??instant (window/2), recovery ??3-streak
    uint32_t base_window = policy == moe_EXECUTION_PERFORMANCE ? 6u : policy == moe_EXECUTION_LOW_IMPACT ? 1u : 3u;
    if (juju_cfg && juju_cfg->prefetch_depth > 0) {
        base_window = juju_cfg->prefetch_depth;
    }
    if (juju_cfg && engine->generation_token.load(std::memory_order_relaxed) < 128u) {
        base_window = std::max(base_window, 4u);
    }
    const uint32_t max_window = juju_cfg ?
        std::max(8u, juju_cfg->prefetch_depth ? juju_cfg->prefetch_depth : 8u) : 8u;

    if (engine) {
        const uint64_t dropped_now = engine->io_atomic.dropped_requests.load(std::memory_order_relaxed);
        const uint64_t dropped_delta = dropped_now > engine->last_dropped_requests_snapshot
            ? dropped_now - engine->last_dropped_requests_snapshot : 0;
        engine->last_dropped_requests_snapshot = dropped_now;

        const uint64_t dropped_expert_equiv = (dropped_delta + 2ull) / 3ull;
        if (dropped_expert_equiv > 0) {
            base_window = dropped_expert_equiv >= 4ull
                ? std::max(1u, base_window / 4u)
                : std::max(1u, base_window / 2u);
            // Reset growth streak on drop detection
            engine->prefetch_window_streak = 0;
        }
    }
    if (engine && policy == moe_EXECUTION_PERFORMANCE) {
        const uint32_t queued_depth = engine->total_queue_depth.load(std::memory_order_relaxed);
        const uint32_t inflight =
            engine->disk_stage.inflight.load(std::memory_order_relaxed) +
            engine->pinned_stage.inflight.load(std::memory_order_relaxed) +
            engine->gpu_stage.inflight.load(std::memory_order_relaxed);
        const uint32_t live_pressure = queued_depth + inflight;
        const uint32_t total_workers = plan->disk_workers + plan->pinned_workers + plan->gpu_workers;
        if (live_pressure > total_workers * 4u) {
            base_window = std::max(2u, base_window / 2u);
        } else if (live_pressure < total_workers) {
            base_window = std::min(max_window, base_window + 1u);
        }
    }
    // BUG 3 FIX: Apply 5-streak hysteresis to prevent window oscillation ?�★
    // BUGFIX 74: Issue 7 - Reduce recovery streak from 5 to 3 ?�★
    // Problem: 5-streak too slow for recovery after drops stop
    // Long orchestrator intervals ??window=1 persists for dozens of tokens
    // Solution: 3-streak for faster adaptation while still preventing oscillation
    // Without hysteresis, window toggles 2?? every orchestrate call when live_pressure
    // hovers around total_workers*4, causing stalls (window=2) and waste (window=8).
    if (engine) {
        uint32_t current_window = plan->prefetch_window_layers;
        if (current_window == 0) {
            current_window = engine->runtime_prefetch_window_layers.load(std::memory_order_relaxed);
        }
        if (current_window == 0) {
            current_window = base_window;
        }
        plan->prefetch_window_layers = current_window;
        if (base_window != current_window) {
            if (engine->prefetch_window_pending == base_window) {
                ++engine->prefetch_window_streak;
            } else {
                engine->prefetch_window_pending = base_window;
                engine->prefetch_window_streak = 1;
            }
            // Reduced from 5 to 3 for faster recovery
            if (engine->prefetch_window_streak >= 3u) {
                plan->prefetch_window_layers = base_window;
                engine->prefetch_window_streak = 0;
            } else {
                // Keep current window until 3 consecutive same signals
                plan->prefetch_window_layers = current_window;
            }
        } else {
            engine->prefetch_window_streak = 0;
        }
    } else {
        plan->prefetch_window_layers = base_window;
    }
    float cpu = state ? state->cpu_load_pct : 0.0f;
    float gpu = state ? state->gpu_load_pct : 0.0f;
    if (!state && engine) {
        // BUGFIX 41: Use cpu_row_active for accurate CPU load estimation ?�★??
        // Old: cpu_active = disk_stage.inflight + pinned_stage.inflight
        //      ??Counts workers blocked on GPU/queue as "CPU busy"
        // New: Use cpu_row_active (actual matmul workers) + disk_stage.inflight
        // Rationale: inflight counts "task popped from queue", not "CPU doing work"
        //            pinned_worker blocked on staging slot = inflight but CPU idle
        //            Only cpu_row_active and disk_stage.inflight do actual CPU work
        // Impact: More accurate route_sparse_to_cpu decisions
        const uint32_t cpu_workers = std::max(
            1u,
            runtime_disk_workers(engine) + runtime_pinned_workers(engine));
        const uint32_t gpu_workers = std::max(1u, runtime_gpu_workers(engine));

        // Actual CPU work: disk I/O + CPU matmul (not pinned staging, which waits on GPU)
        const uint32_t disk_active = engine->disk_stage.inflight.load(std::memory_order_relaxed);
        const uint32_t cpu_matmul_active = engine->cpu_row_active.load(std::memory_order_relaxed);
        const uint32_t cpu_active = disk_active + cpu_matmul_active;

        const uint32_t gpu_active = engine->gpu_stage.inflight.load(std::memory_order_relaxed);
        cpu = std::min(100.0f, 100.0f * (float)cpu_active / (float)cpu_workers);
        gpu = std::min(100.0f, 100.0f * (float)gpu_active / (float)gpu_workers);
    }
    const uint32_t os_tasks = state ? state->active_os_tasks : 0u;
    const uint32_t busy_task_threshold = hw;  // Already guaranteed > 0
    const bool os_busy = os_tasks >= busy_task_threshold;
    plan->allow_gpu_dense = gpu <= policy_gpu_target(policy);
    plan->route_sparse_to_cpu = cpu <= policy_cpu_target(policy);
    if (os_busy && policy != moe_EXECUTION_PERFORMANCE) {
        plan->disk_workers = plan->disk_workers > 1u ? plan->disk_workers / 2u : 1u;
        plan->pinned_workers = 1u;
        plan->gpu_workers = 1u;
        plan->prefetch_window_layers = plan->prefetch_window_layers > 1u ? plan->prefetch_window_layers - 1u : 1u;
    }
    plan->throttle_speculative = !plan->allow_gpu_dense || !plan->route_sparse_to_cpu || os_busy;
    plan->drop_speculative = policy == moe_EXECUTION_LOW_IMPACT && plan->throttle_speculative;
    const bool gpu_cache_ready = engine_has_device_upload_workers(engine) &&
        vram_expert_budget_bytes(engine) > 0 && plan->allow_gpu_dense;
    plan->selected_tier = gpu_cache_ready ? moe_TIER_VRAM : moe_TIER_RAM;
    plan->speculative_tier = plan->drop_speculative ? moe_TIER_DB :
        (gpu_cache_ready ? moe_TIER_VRAM : moe_TIER_RAM);
}

static int worker_resize_direction(
    uint32_t disk_current,
    uint32_t pinned_current,
    uint32_t gpu_current,
    uint32_t disk_target,
    uint32_t pinned_target,
    uint32_t gpu_target
) {
    const bool any_up =
        disk_target > disk_current ||
        pinned_target > pinned_current ||
        gpu_target > gpu_current;
    const bool any_down =
        disk_target < disk_current ||
        pinned_target < pinned_current ||
        gpu_target < gpu_current;
    if (any_up == any_down) {
        return 0;
    }
    return any_up ? 1 : -1;
}

static int worker_resize_hysteresis_ready(
    moe_pc_engine* engine,
    const moe_runtime_orchestrator_plan_t* plan
) {
    if (!engine || !plan) {
        return 0;
    }
    const uint32_t disk = clamp_to_cap(plan->disk_workers, disk_worker_cap());
    const uint32_t pinned = clamp_to_cap(plan->pinned_workers, pinned_worker_cap(engine));
    const uint32_t gpu = clamp_to_cap(plan->gpu_workers, gpu_worker_cap(engine));
    const uint32_t cur_disk = runtime_disk_workers(engine);
    const uint32_t cur_pinned = runtime_pinned_workers(engine);
    const uint32_t cur_gpu = runtime_gpu_workers(engine);
    if (cur_disk == 0 || cur_pinned == 0 || cur_gpu == 0) {
        engine->worker_resize_streak = 0;
        engine->worker_resize_pending_direction = 0;
        return 1;
    }
    if (disk == cur_disk && pinned == cur_pinned && gpu == cur_gpu) {
        engine->worker_resize_streak = 0;
        engine->worker_resize_pending_direction = 0;
        return 0;
    }
    const int direction = worker_resize_direction(cur_disk, cur_pinned, cur_gpu, disk, pinned, gpu);
    if (direction == 0) {
        engine->worker_resize_streak = 0;
        engine->worker_resize_pending_direction = 0;
        return 0;
    }
    if (engine->worker_resize_pending_disk == disk &&
        engine->worker_resize_pending_pinned == pinned &&
        engine->worker_resize_pending_gpu == gpu &&
        engine->worker_resize_pending_direction == direction) {
        ++engine->worker_resize_streak;
    } else {
        engine->worker_resize_pending_disk = disk;
        engine->worker_resize_pending_pinned = pinned;
        engine->worker_resize_pending_gpu = gpu;
        engine->worker_resize_pending_direction = direction;
        engine->worker_resize_streak = 1;
    }
    if (engine->worker_resize_streak < 3u) {
        return 0;
    }
    engine->worker_resize_streak = 0;
    engine->worker_resize_pending_direction = 0;
    return 1;
}

int moe_pc_engine_orchestrate_moe_step(moe_pc_engine_t* engine, int current_layer,
    const int* selected_experts, uint32_t selected_count, const int* next_layer_candidates,
    uint32_t next_count, const moe_runtime_orchestrator_state_t* state,
    moe_runtime_orchestrator_plan_t* out_plan, moe_expert_prefetch_item_t* out_items,
    uint32_t max_items, uint32_t* out_item_count) {
    if (!engine || !out_plan) {
        return 0;
    }
    refresh_external_gpu_memory_budget_if_due(engine);

    // BUGFIX 문제8: Async merge map rebuild ★★★
    // Check if rebuild is pending and trigger it
    // This enables MC-MoE / EvoMoE cluster prefetching which was completely disabled
    if (engine->merge_map_rebuild_pending.load(std::memory_order_acquire)) {
        engine->merge_map_rebuild_pending.store(false, std::memory_order_relaxed);

        // Rebuild in background (doesn't block inference)
        // moe_build_expert_merge_map() already has its own mutex (expert_merge_map.mtx)
        moe_build_expert_merge_map(engine);
    }

    fill_orchestrator_plan(engine, state, out_plan);

    // BUGFIX 29: Release plan_lock before calling execute_prefetch_plan ★★
    // Old: plan_lock held during execute_prefetch_plan → enqueue_stage_or_retry → disk_stage.mtx
    //      Creates lock order: prefetch_plan_mutex → disk_stage.mtx
    //      io_worker_loop holds disk_stage.mtx → get_or_map_tensor_shard (potential deadlock)
    // New: Release plan_lock after filling out_items, before execute_prefetch_plan
    //      out_items is local snapshot, consistency maintained
    //      prefetch_plan_clear_seen is idempotent, safe for concurrent access
    // BUGFIX 85: Issue 7 - Move prefetch_plan_clear_seen outside lock ★★
    // Problem: clear_seen can iterate 50k map entries under lock
    // Solution: Call before acquiring lock (generation increment is atomic-safe)
    prefetch_plan_clear_seen(engine);

    uint32_t kept = 0;
    // The prefetch-plan helper functions already protect their generation map.
    // Holding prefetch_plan_mutex here deadlocks when append helpers re-enter it.
    engine->runtime_prefetch_window_layers.store(out_plan->prefetch_window_layers, std::memory_order_relaxed);
    if (runtime_disk_workers(engine) != out_plan->disk_workers ||
        runtime_pinned_workers(engine) != out_plan->pinned_workers ||
        runtime_gpu_workers(engine) != out_plan->gpu_workers) {
        if (worker_resize_hysteresis_ready(engine, out_plan)) {
            orchestrator_resize_workers(engine, out_plan);
        }
    }
    for (uint32_t i = 0; selected_experts && i < selected_count; ++i) {
        out_plan->selected_expert_bytes += orchestrator_expert_bytes(engine, current_layer, selected_experts[i]);
    }
    if (!out_items || max_items == 0) {
        if (out_item_count) *out_item_count = 0;
        prefetch_moe_activation_raw(engine, current_layer);
        prefetch_moe_shared_compute_raw(engine, current_layer);
        if (!out_plan->drop_speculative) {
            prefetch_moe_activation_raw(engine, current_layer + 1);
        }
        return 1;
    }
    for (uint32_t i = 0; selected_experts && i < selected_count; ++i) {
        orchestrator_append_item(engine, out_items, max_items, &kept, current_layer, selected_experts[i],
            out_plan->selected_tier, moe_PREFETCH_REASON_ROUTER_SELECTED, 1000.0f);
    }
    if (!out_plan->drop_speculative) {
        for (uint32_t i = 0; next_layer_candidates && i < next_count; ++i) {
            orchestrator_append_item(engine, out_items, max_items, &kept, current_layer + 1, next_layer_candidates[i],
                out_plan->speculative_tier, moe_PREFETCH_REASON_LOOKAHEAD, 100.0f);
        }
        if (!out_plan->throttle_speculative) {
            orchestrator_append_topology(engine, current_layer, selected_experts, selected_count,
                out_plan->speculative_tier, out_items, max_items, &kept);

            std::vector<int> dejavu_predicted = moe_juju_predict_co_activation(engine, current_layer + 2, 4);
            std::vector<int> dejavu_domain;
            moe_expand_merged_experts(engine, current_layer + 2, dejavu_predicted, dejavu_domain);
            for (int e : dejavu_domain) {
                orchestrator_append_item(engine, out_items, max_items, &kept, current_layer + 2, e,
                    moe_TIER_RAM, moe_PREFETCH_REASON_LOOKAHEAD, 80.0f);
            }
        }
    }
    out_plan->planned_prefetch_count = kept;
    if (out_item_count) {
        *out_item_count = kept;
    }

    // Bug #8: These two activation prefetch paths are mutually exclusive (early return).
    if (out_plan->planned_prefetch_count == 0) {
        prefetch_moe_activation_raw(engine, current_layer);
        prefetch_moe_shared_compute_raw(engine, current_layer);
        if (!out_plan->drop_speculative) {
            prefetch_moe_activation_raw(engine, current_layer + 1);
        }
        moe_pc_engine_apply_topology_updates(engine);
        return 1;
    }
    const int ok = moe_pc_engine_execute_prefetch_plan(engine, out_items, out_plan->planned_prefetch_count);
    prefetch_moe_activation_raw(engine, current_layer);
    prefetch_moe_shared_compute_raw(engine, current_layer);
    // Bug #12: activation prefetch unconditional
    if (!out_plan->drop_speculative) {
        prefetch_moe_activation_raw(engine, current_layer + 1);
    }
    moe_pc_engine_apply_topology_updates(engine);
    return ok;
}


// Additional 2: PreScope Global Optimization Cross-Layer Scheduling
#define MAX_PREFETCH_DEPTH 8
struct prescope_plan_t {
    float bw_budget_per_layer[MAX_PREFETCH_DEPTH];
    uint32_t vram_slots[MAX_PREFETCH_DEPTH];
    uint32_t int4_slots[MAX_PREFETCH_DEPTH];
};

static void prescope_compute_global_plan(
    moe_pc_engine* engine,
    int start_layer, uint32_t depth,
    const std::vector<std::vector<int>>& predicted_experts,
    prescope_plan_t* plan
) {
    if (!engine || !plan || depth == 0) return;

    // BUGFIX 266: Clamp depth to predicted_experts size to prevent out-of-bounds ★★
    // Problem: Loop checks d < depth && d < MAX_PREFETCH_DEPTH, but if depth > predicted_experts.size()
    // then predicted_experts[d] access is out-of-bounds before the break check
    // Solution: Clamp depth to predicted_experts.size() before loop
    depth = std::min(depth, (uint32_t)predicted_experts.size());
    depth = std::min(depth, (uint32_t)MAX_PREFETCH_DEPTH);

    // BUGFIX 281: Additional safety check - return early if predicted_experts is empty ★
    // Problem: Even after clamping, if predicted_experts is empty, depth becomes 0
    // but loop condition d < depth would still be checked with uninitialized data
    // Solution: Early return if no predictions available
    if (predicted_experts.empty() || depth == 0) {
        return;
    }

    // Stub: compute the maximum experts that can be loaded in full precision within compute time
    // total_bw is DMA bandwidth GB/s (e.g., PCIe Gen4 x16 ~ 25GB/s)
    const float total_bw = engine->backend_caps.dma_bandwidth_gbps;

    // Fallback to 1ms if not profiled
    const float compute_ms_per_layer = 1.0f; // engine->profile.avg_layer_compute_ms;

    for (uint32_t d = 0; d < depth; ++d) {
        // BUGFIX 281: Double-check bounds before access (defense in depth) ★
        if (d >= predicted_experts.size()) {
            break;
        }
        const uint32_t n_experts = (uint32_t)predicted_experts[d].size();

        // Fix 16: Prevent divide-by-zero UB by ensuring avg_expert_bytes has a fallback
        const uint64_t max_tensor_bytes = engine->max_tensor_stream_bytes.load(std::memory_order_relaxed);
        const uint64_t safe_avg_bytes = engine->avg_expert_bytes > 0 ? engine->avg_expert_bytes :
            (max_tensor_bytes > 0 ? max_tensor_bytes : 20ull * 1024ull * 1024ull);
        const float expert_gb = safe_avg_bytes / 1e9f;
        // BUGFIX 639: Check total_bw > 0 before division ★
        // Problem: total_bw can be 0 causing division by zero
        // Solution: Check total_bw > 0 before division
        if (total_bw <= 0.0f) {
            continue;
        }
        const float full_transfer_ms = (n_experts * expert_gb) / total_bw * 1000.0f;

        if (full_transfer_ms <= compute_ms_per_layer) {
            plan->vram_slots[d] = n_experts;
            plan->int4_slots[d] = 0;
        } else {
            const float allowed_gb = (compute_ms_per_layer / 1000.0f) * total_bw;
            // BUGFIX 148: Check expert_gb > 0 before division to prevent division by zero
            // Very small experts (< 1MB) can result in expert_gb = 0.0f
            uint32_t full_capacity = 0;
            if (expert_gb > 1e-6f) {
                full_capacity = (uint32_t)(allowed_gb / expert_gb);
            } else {
                // Very small experts: assume we can fit all
                full_capacity = n_experts;
            }
            plan->vram_slots[d] = std::min(n_experts, full_capacity);
            plan->int4_slots[d] = n_experts - plan->vram_slots[d];
        }
    }
}

