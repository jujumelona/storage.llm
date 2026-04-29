static float policy_cpu_target(glm5_execution_policy_t policy) {
    return policy == GLM5_EXECUTION_PERFORMANCE ? 92.0f : policy == GLM5_EXECUTION_LOW_IMPACT ? 45.0f : 70.0f;
}

static float policy_gpu_target(glm5_execution_policy_t policy) {
    return policy == GLM5_EXECUTION_PERFORMANCE ? 96.0f : policy == GLM5_EXECUTION_LOW_IMPACT ? 60.0f : 85.0f;
}

static void fill_orchestrator_plan(glm5_pc_engine* engine, const glm5_runtime_orchestrator_state_t* state,
    glm5_runtime_orchestrator_plan_t* plan) {
    const glm5_execution_policy_t policy = state ? state->policy : GLM5_EXECUTION_BALANCED;
    // Fix 14: Handle hardware_concurrency() == 0 gracefully
    static const unsigned int hw_raw = std::thread::hardware_concurrency();
    static const unsigned int hw = hw_raw > 0 ? hw_raw : 4u;  // Fallback to 4 if undefined
    const uint32_t perf_disk_workers = hw >= 16 ? 3u : 2u;
    // Fix 3: Allow up to 3 GPU workers in PERFORMANCE mode to pipeline
    // blocking cuStreamSynchronize calls and improve PCIe utilization.
    const uint32_t perf_gpu_workers = engine && engine->backend_caps.copy_streams > 1
        ? std::min(3u, (uint32_t)engine->backend_caps.copy_streams) : 1u;
    std::memset(plan, 0, sizeof(*plan));
    plan->policy = policy;
    plan->disk_workers = policy == GLM5_EXECUTION_PERFORMANCE ? perf_disk_workers : policy == GLM5_EXECUTION_LOW_IMPACT ? 1u : 2u;
    plan->pinned_workers = policy == GLM5_EXECUTION_PERFORMANCE ? std::max(2u, perf_gpu_workers) : 1u;
    plan->gpu_workers = policy == GLM5_EXECUTION_PERFORMANCE ? perf_gpu_workers : 1u;
    // Dynamic prefetch window based on live IO pressure.
    uint32_t base_window = policy == GLM5_EXECUTION_PERFORMANCE ? 6u : policy == GLM5_EXECUTION_LOW_IMPACT ? 1u : 3u;
    if (engine && policy == GLM5_EXECUTION_PERFORMANCE) {
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
            base_window = std::min(8u, base_window + 1u);
        }
    }
    plan->prefetch_window_layers = base_window;
    float cpu = state ? state->cpu_load_pct : 0.0f;
    float gpu = state ? state->gpu_load_pct : 0.0f;
    if (!state && engine) {
        const uint32_t cpu_workers = std::max(
            1u,
            runtime_disk_workers(engine) + runtime_pinned_workers(engine));
        const uint32_t gpu_workers = std::max(1u, runtime_gpu_workers(engine));
        const uint32_t cpu_active =
            engine->disk_stage.inflight.load(std::memory_order_relaxed) +
            engine->pinned_stage.inflight.load(std::memory_order_relaxed);
        const uint32_t gpu_active = engine->gpu_stage.inflight.load(std::memory_order_relaxed);
        cpu = std::min(100.0f, 100.0f * (float)cpu_active / (float)cpu_workers);
        gpu = std::min(100.0f, 100.0f * (float)gpu_active / (float)gpu_workers);
    }
    const uint32_t os_tasks = state ? state->active_os_tasks : 0u;
    const uint32_t busy_task_threshold = hw;  // Already guaranteed > 0
    const bool os_busy = os_tasks >= busy_task_threshold;
    plan->allow_gpu_dense = gpu <= policy_gpu_target(policy);
    plan->route_sparse_to_cpu = cpu <= policy_cpu_target(policy);
    if (os_busy && policy != GLM5_EXECUTION_PERFORMANCE) {
        plan->disk_workers = plan->disk_workers > 1u ? plan->disk_workers / 2u : 1u;
        plan->pinned_workers = 1u;
        plan->gpu_workers = 1u;
        plan->prefetch_window_layers = plan->prefetch_window_layers > 1u ? plan->prefetch_window_layers - 1u : 1u;
    }
    plan->throttle_speculative = !plan->allow_gpu_dense || !plan->route_sparse_to_cpu || os_busy;
    plan->drop_speculative = policy == GLM5_EXECUTION_LOW_IMPACT && plan->throttle_speculative;
    const bool gpu_cache_ready = caps_gpu_backend_available(engine->backend_caps) &&
        vram_expert_budget_bytes(engine) > 0 && plan->allow_gpu_dense;
    plan->selected_tier = gpu_cache_ready ? GLM5_TIER_VRAM : GLM5_TIER_RAM;
    plan->speculative_tier = plan->drop_speculative ? GLM5_TIER_DB :
        (gpu_cache_ready ? GLM5_TIER_VRAM : GLM5_TIER_RAM);
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
    glm5_pc_engine* engine,
    const glm5_runtime_orchestrator_plan_t* plan
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
    if (engine->worker_resize_streak < 5u) {
        return 0;
    }
    engine->worker_resize_streak = 0;
    engine->worker_resize_pending_direction = 0;
    return 1;
}

int glm5_pc_engine_orchestrate_moe_step(glm5_pc_engine_t* engine, int current_layer,
    const int* selected_experts, uint32_t selected_count, const int* next_layer_candidates,
    uint32_t next_count, const glm5_runtime_orchestrator_state_t* state,
    glm5_runtime_orchestrator_plan_t* out_plan, glm5_expert_prefetch_item_t* out_items,
    uint32_t max_items, uint32_t* out_item_count) {
    if (!engine || !out_plan) {
        return 0;
    }
    fill_orchestrator_plan(engine, state, out_plan);
    {
        std::lock_guard<std::mutex> plan_lock(engine->prefetch_plan_mutex);
        prefetch_plan_clear_seen(engine);
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
        uint32_t kept = 0;
        for (uint32_t i = 0; selected_experts && i < selected_count; ++i) {
            orchestrator_append_item(engine, out_items, max_items, &kept, current_layer, selected_experts[i],
                out_plan->selected_tier, GLM5_PREFETCH_REASON_ROUTER_SELECTED, 1000.0f);
        }
        if (!out_plan->drop_speculative) {
            for (uint32_t i = 0; next_layer_candidates && i < next_count; ++i) {
                orchestrator_append_item(engine, out_items, max_items, &kept, current_layer + 1, next_layer_candidates[i],
                    out_plan->speculative_tier, GLM5_PREFETCH_REASON_LOOKAHEAD, 100.0f);
            }
            if (!out_plan->throttle_speculative) {
                orchestrator_append_topology(engine, current_layer, selected_experts, selected_count,
                    out_plan->speculative_tier, out_items, max_items, &kept);
            }
        }
        out_plan->planned_prefetch_count = kept;
        if (out_item_count) {
            *out_item_count = kept;
        }
    }
    // Bug #8: These two activation prefetch paths are mutually exclusive (early return).
    if (out_plan->planned_prefetch_count == 0) {
        prefetch_moe_activation_raw(engine, current_layer);
        prefetch_moe_shared_compute_raw(engine, current_layer);
        if (!out_plan->drop_speculative) {
            prefetch_moe_activation_raw(engine, current_layer + 1);
        }
        glm5_pc_engine_apply_topology_updates(engine);
        return 1;
    }
    const int ok = glm5_pc_engine_execute_prefetch_plan(engine, out_items, out_plan->planned_prefetch_count);
    prefetch_moe_activation_raw(engine, current_layer);
    prefetch_moe_shared_compute_raw(engine, current_layer);
    // Bug #12: activation prefetch unconditional
    if (!out_plan->drop_speculative) {
        prefetch_moe_activation_raw(engine, current_layer + 1);
    }
    glm5_pc_engine_apply_topology_updates(engine);
    return ok;
}
