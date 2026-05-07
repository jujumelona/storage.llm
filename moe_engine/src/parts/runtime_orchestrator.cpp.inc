static float policy_cpu_target(moe_execution_policy_t policy) {
    return policy == moe_EXECUTION_PERFORMANCE ? 92.0f : policy == moe_EXECUTION_LOW_IMPACT ? 45.0f : 70.0f;
}

static float policy_gpu_target(moe_execution_policy_t policy) {
    return policy == moe_EXECUTION_PERFORMANCE ? 96.0f : policy == moe_EXECUTION_LOW_IMPACT ? 60.0f : 85.0f;
}

static uint64_t moe_delta_u64(uint64_t now, std::atomic<uint64_t>& prev) {
    const uint64_t old = prev.exchange(now, std::memory_order_acq_rel);
    return now >= old ? now - old : 0;
}

static moe_runtime_perf_window collect_perf_window(moe_pc_engine* engine, uint64_t interval_ns) {
    moe_runtime_perf_window m{};
    if (!engine) {
        return m;
    }
    m.tokens = moe_delta_u64(
        engine->generation_completed.load(std::memory_order_relaxed),
        engine->autotune_prev_tokens);
    m.expert_hits = moe_delta_u64(
        engine->expert_cache_hit_count.load(std::memory_order_relaxed),
        engine->autotune_prev_hits);
    m.expert_misses = moe_delta_u64(
        engine->expert_cache_miss_count.load(std::memory_order_relaxed),
        engine->autotune_prev_misses);
    m.prefetch_used = moe_delta_u64(
        engine->speculative_prefetch_planned_count.load(std::memory_order_relaxed),
        engine->autotune_prev_prefetch_planned);
    m.prefetch_wasted = moe_delta_u64(
        engine->io_atomic.dropped_requests.load(std::memory_order_relaxed) +
            engine->io_atomic.urgent_dropped_requests.load(std::memory_order_relaxed),
        engine->autotune_prev_prefetch_wasted);
    m.sync_h2d_fallbacks = moe_delta_u64(
        engine->io_atomic.bytes_sync_h2d_fallback.load(std::memory_order_relaxed),
        engine->autotune_prev_sync_h2d);
    const uint64_t disk_bytes = moe_delta_u64(
        engine->io_atomic.bytes_disk_to_ram.load(std::memory_order_relaxed) +
            engine->io_atomic.bytes_directstorage_completed.load(std::memory_order_relaxed),
        engine->autotune_prev_directstorage_bytes);
    const uint64_t h2d_bytes = moe_delta_u64(
        engine->io_atomic.bytes_pinned_to_gpu.load(std::memory_order_relaxed),
        engine->autotune_prev_h2d_bytes);
    const uint64_t staging_waits = moe_delta_u64(
        engine->io_atomic.staging_slot_waits.load(std::memory_order_relaxed) +
            engine->io_atomic.staging_slot_contention.load(std::memory_order_relaxed),
        engine->autotune_prev_staging_waits);
    m.directstorage_queue_wait_ns = moe_delta_u64(
        engine->io_atomic.directstorage_queue_full_wait_ns.load(std::memory_order_relaxed),
        engine->autotune_prev_directstorage_queue_wait);
    const uint64_t storage_bw = engine->backend_caps.measured_storage_bw_bytes_per_sec
        ? engine->backend_caps.measured_storage_bw_bytes_per_sec
        : (3ull << 30);
    uint64_t h2d_bw = engine->backend_caps.measured_h2d_bw_bytes_per_sec;
    if (h2d_bw == 0 && engine->backend_caps.dma_bandwidth_gbps > 0.0) {
        h2d_bw = (uint64_t)(engine->backend_caps.dma_bandwidth_gbps * 1000000000.0);
    }
    if (h2d_bw == 0) {
        h2d_bw = 12ull * 1000ull * 1000ull * 1000ull;
    }
    m.disk_read_ns = storage_bw ? (disk_bytes * 1000000000ull) / storage_bw : 0;
    m.h2d_copy_ns = h2d_bw ? (h2d_bytes * 1000000000ull) / h2d_bw : 0;
    m.gpu_idle_ns = (interval_ns * engine->gpu_copy_idle_milli.load(std::memory_order_relaxed)) / 1000ull;
    m.staging_wait_ns = staging_waits * 1000000ull;
    return m;
}

static uint32_t expert_hit_rate_milli(const moe_runtime_perf_window& m) {
    const uint64_t total = m.expert_hits + m.expert_misses;
    return total ? (uint32_t)std::min<uint64_t>(1000ull, (m.expert_hits * 1000ull) / total) : 1000u;
}

static uint32_t prefetch_waste_milli(const moe_runtime_perf_window& m) {
    return m.prefetch_used ?
        (uint32_t)std::min<uint64_t>(1000ull, (m.prefetch_wasted * 1000ull) / m.prefetch_used) : 0u;
}

static uint64_t io_path_avg_wait_ns(const moe_io_path_perf& p) {
    const uint64_t completed = p.completed.load(std::memory_order_relaxed);
    const uint64_t wait = p.wait_ns.load(std::memory_order_relaxed);
    return completed ? wait / completed : 0;
}

static uint32_t io_path_fail_rate_milli(const moe_io_path_perf& p) {
    const uint64_t submitted = p.submitted.load(std::memory_order_relaxed);
    const uint64_t failed = p.fail_count.load(std::memory_order_relaxed);
    return submitted ? (uint32_t)std::min<uint64_t>(1000ull, (failed * 1000ull) / submitted) : 0u;
}

static void autotune_resize_io_workers(
    moe_pc_engine* engine,
    uint32_t disk,
    uint32_t pinned,
    uint32_t gpu
) {
    if (!engine || !engine->io_config.enable_compute_io_overlap) {
        return;
    }
    moe_runtime_orchestrator_plan_t plan{};
    plan.disk_workers = clamp_to_cap(disk, disk_worker_cap());
    plan.pinned_workers = clamp_to_cap(pinned, pinned_worker_cap(engine));
    plan.gpu_workers = clamp_to_cap(gpu, gpu_worker_cap(engine));
    plan.prefetch_window_layers = engine->runtime_prefetch_window_layers.load(std::memory_order_relaxed);
    orchestrator_resize_workers(engine, &plan);
}

static void moe_runtime_autotune_tick(moe_pc_engine* engine) {
    if (!engine) {
        return;
    }
    device_drain_directstorage_device_copies(engine);
    const uint64_t now_ms = moe_profile_now_us() / 1000ull;
    uint64_t last_ms = engine->autotune_last_tick_ms.load(std::memory_order_relaxed);
    if (last_ms != 0 && now_ms < last_ms + 1000ull) {
        return;
    }
    if (!engine->autotune_last_tick_ms.compare_exchange_strong(
            last_ms,
            now_ms,
            std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        return;
    }
    const uint64_t interval_ns = last_ms == 0 ? 1000000000ull : (now_ms - last_ms) * 1000000ull;
    moe_runtime_perf_window m = collect_perf_window(engine, interval_ns);
    uint32_t window = engine->runtime_prefetch_window_layers.load(std::memory_order_relaxed);
    if (window == 0) {
        window = engine->io_config.prefetch_window_layers ? engine->io_config.prefetch_window_layers : 1u;
    }
    const uint32_t waste = prefetch_waste_milli(m);
    const uint32_t hit = expert_hit_rate_milli(m);
    const uint64_t wait_limit = interval_ns / 20ull;
    const uint64_t idle_limit = interval_ns / 5ull;
    const uint64_t disk_limit = interval_ns / 3ull;
    const uint64_t pinned_wait = io_path_avg_wait_ns(engine->pinned_perf);
    const uint64_t gds_wait = io_path_avg_wait_ns(engine->gds_perf);
    const uint64_t ds_gpu_wait = io_path_avg_wait_ns(engine->ds_gpu_perf);
    const uint64_t ds_staging_wait = io_path_avg_wait_ns(engine->ds_staging_perf);
    const uint32_t gds_fail = io_path_fail_rate_milli(engine->gds_perf);
    const uint32_t ds_gpu_fail = io_path_fail_rate_milli(engine->ds_gpu_perf);

    if (m.sync_h2d_fallbacks > 0 || m.staging_wait_ns > wait_limit) {
        const uint64_t current = engine->staging_size ? engine->staging_size : engine->io_config.pinned_staging_bytes;
        const uint64_t target = current > 0 ? current + current / 4ull : moe_recommended_staging_bytes(engine);
        if (target > current) {
            moe_ensure_staging_capacity(engine, moe_rt_align_4k(target));
        }
        window = std::max(1u, window > 1u ? window - 1u : 1u);
        autotune_resize_io_workers(
            engine,
            runtime_disk_workers(engine),
            std::min<uint32_t>(pinned_worker_cap(engine), runtime_pinned_workers(engine) + 1u),
            runtime_gpu_workers(engine));
    }
    if (m.directstorage_queue_wait_ns > wait_limit) {
        window = std::max(1u, window > 1u ? window - 1u : 1u);
        const uint32_t staging_cap = std::min<uint32_t>(
            8192u,
            std::max<uint32_t>(engine->io_config.directstorage_staging_queue_capacity, 1024u));
        const uint32_t gpu_cap = std::min<uint32_t>(
            8192u,
            std::max<uint32_t>(engine->io_config.directstorage_gpu_queue_capacity, 1024u));
        engine->io_config.directstorage_staging_queue_capacity = std::min<uint32_t>(
            8192u,
            staging_cap + (staging_cap < 8192u ? 256u : 0u));
        engine->io_config.directstorage_gpu_queue_capacity = std::min<uint32_t>(
            8192u,
            gpu_cap + (gpu_cap < 8192u ? 256u : 0u));
        engine->directstorage_recreate_requested.store(true, std::memory_order_release);
    }
    if (hit < 850u && waste < 200u) {
        window = std::min<uint32_t>(16u, window + 1u);
    }
    if (waste > 350u) {
        window = std::max(1u, window / 2u);
        uint32_t depth = engine->adaptive_prefetch_depth.load(std::memory_order_relaxed);
        if (depth > 1u) {
            engine->adaptive_prefetch_depth.store(depth - 1u, std::memory_order_relaxed);
        }
        engine->merge_map_rebuild_pending.store(true, std::memory_order_release);
    }
    if (m.gpu_idle_ns > idle_limit && m.disk_read_ns < disk_limit && waste < 250u) {
        window = std::min<uint32_t>(16u, window + 1u);
        uint32_t depth = engine->adaptive_prefetch_depth.load(std::memory_order_relaxed);
        engine->adaptive_prefetch_depth.store(std::min<uint32_t>(8u, depth + 1u), std::memory_order_relaxed);
        autotune_resize_io_workers(
            engine,
            std::min<uint32_t>(disk_worker_cap(), runtime_disk_workers(engine) + 1u),
            runtime_pinned_workers(engine),
            runtime_gpu_workers(engine));
    }
    if (m.disk_read_ns > disk_limit && m.prefetch_wasted > 0) {
        window = std::max(1u, window / 2u);
        const uint32_t disk = runtime_disk_workers(engine);
        if (disk > 1u) {
            autotune_resize_io_workers(
                engine,
                disk - 1u,
                runtime_pinned_workers(engine),
                runtime_gpu_workers(engine));
        }
    }
    if (engine->runtime_gpudirect_ready.load(std::memory_order_acquire) &&
        engine->gds_perf.submitted.load(std::memory_order_relaxed) >= 4u &&
        (gds_fail > 50u ||
         (pinned_wait > 0 && gds_wait > (pinned_wait * 12ull) / 10ull))) {
        engine->runtime_gpudirect_ready.store(false, std::memory_order_release);
    }
    if (engine->runtime_directstorage_gpu_ready.load(std::memory_order_acquire) &&
        engine->ds_gpu_perf.submitted.load(std::memory_order_relaxed) >= 4u &&
        (engine->backend_caps.backend != moe_BACKEND_DIRECTML ||
         ds_gpu_fail > 50u ||
         (ds_staging_wait > 0 && ds_gpu_wait > ds_staging_wait))) {
        engine->runtime_directstorage_gpu_ready.store(false, std::memory_order_release);
    }
    engine->runtime_prefetch_window_layers.store(window, std::memory_order_relaxed);
}

static void refresh_external_gpu_memory_budget_if_due(moe_pc_engine* engine) {
    if (!engine) {
        return;
    }
    // BUGFIX 881: Add CUDA/HIP to external memory refresh ★★★ CRITICAL
    // Problem: device_refresh_mem_info only called for Vulkan/Level Zero/SYCL.
    // CUDA backend uses internal counter device_allocated_bytes only.
    // After hundreds of tokens, CUDA driver internal fragmentation and stream buffer
    // overhead accumulate, causing 100s of MB difference between counter and actual free VRAM.
    // Engine thinks VRAM is available, loads experts, hits OOM.
    // Solution: Call device_refresh_mem_info for CUDA/HIP with conditional refresh.
    //
    // BUGFIX 889: Improve CUDA/HIP refresh trigger conditions ★★★ CRITICAL
    // Problem: 10s absolute cooldown is too long for large batches - OOM can occur within 10s.
    // Vulkan/Level Zero refresh every orchestration cycle with no cooldown.
    // Solution: Conditional refresh based on VRAM pressure:
    // - If device_allocated_bytes >= 85% of budget, refresh immediately (no cooldown)
    // - Otherwise, use 10s cooldown to avoid overhead
    // - Also refresh on dropped_requests increase (indicates memory pressure)
    // Impact: Prevents OOM while limiting cuMemGetInfo overhead to high-pressure situations.
    // BUGFIX Problem 10: Lower threshold from 85% to 80% for earlier OOM prevention ★★★ CRITICAL
    // Problem: 85% threshold too late - driver fragmentation can cause OOM before threshold
    // → Expert load fails with OOM before cuMemGetInfo is called
    // → Timing window between 80-85% where OOM occurs without budget refresh
    // Solution: Lower threshold to 80% for earlier detection
    // → Adds reactive path: expert allocation failure triggers immediate refresh
    // Impact: Prevents OOM in 80-85% window, catches fragmentation earlier
    if (engine->backend_caps.backend == moe_BACKEND_CUDA ||
        engine->backend_caps.backend == moe_BACKEND_HIP) {
        const uint64_t now_ms = moe_profile_now_us() / 1000ull;
        const int64_t last_ms = engine->last_mem_query_ms.load(std::memory_order_relaxed);
        const uint64_t elapsed_ms = now_ms > (uint64_t)last_ms ? now_ms - (uint64_t)last_ms : 0;

        // Check if we're under memory pressure. device_budget_bytes() is already
        // "available after used bytes", so compare used bytes against the raw budget.
        const uint64_t raw_budget = device_raw_budget_bytes(engine);
        const uint64_t used =
            engine->vram_resident_bytes.load(std::memory_order_relaxed) +
            engine->vram_inflight_bytes.load(std::memory_order_relaxed) +
            engine->vram_kv_bytes.load(std::memory_order_relaxed) +
            engine->vram_common_bytes.load(std::memory_order_relaxed);
        const bool high_pressure =
            raw_budget > 0 && used >= (raw_budget * 80ull) / 100ull;

        // Check if dropped_requests increased (indicates memory pressure)
        const uint64_t dropped_now = engine->io_atomic.dropped_requests.load(std::memory_order_relaxed);
        const bool drops_increased = dropped_now > engine->last_dropped_requests_snapshot.load(std::memory_order_relaxed);

        // Refresh immediately if high pressure or drops, otherwise use 10s cooldown
        const bool should_refresh = high_pressure || drops_increased || elapsed_ms >= 10000ull;

        if (should_refresh) {
            const uint64_t snapshot_before = dropped_now;
            device_refresh_mem_info(engine, 0);
            engine->last_mem_query_ms.store((int64_t)now_ms, std::memory_order_relaxed);
            engine->last_dropped_requests_snapshot.store(snapshot_before, std::memory_order_relaxed);
        }
    } else if (engine->backend_caps.backend == moe_BACKEND_VULKAN ||
               engine->backend_caps.backend == moe_BACKEND_LEVEL_ZERO ||
               engine->backend_caps.backend == moe_BACKEND_SYCL) {
        device_refresh_mem_info(engine, 0);
    }
}

static uint64_t orchestrator_transfer_us(uint64_t bytes, uint64_t bytes_per_second) {
    if (bytes == 0 || bytes_per_second == 0) {
        return 0;
    }
    return (bytes * 1000000ull + bytes_per_second - 1ull) / bytes_per_second;
}

static uint64_t orchestrator_estimated_expert_miss_latency_us(
    const moe_pc_engine* engine,
    uint64_t bytes,
    moe_tier_t current_tier,
    moe_tier_t target_tier
) {
    if (!engine || bytes == 0) {
        return 0;
    }
    const uint64_t storage_bw =
        engine->backend_caps.measured_storage_bw_bytes_per_sec ?
        engine->backend_caps.measured_storage_bw_bytes_per_sec : (3ull << 30);
    uint64_t h2d_bw = engine->backend_caps.measured_h2d_bw_bytes_per_sec;
    if (h2d_bw == 0 && engine->backend_caps.dma_bandwidth_gbps > 0.0) {
        h2d_bw = (uint64_t)(engine->backend_caps.dma_bandwidth_gbps * 1000000000.0);
    }
    if (h2d_bw == 0) {
        h2d_bw = 12ull * 1000ull * 1000ull * 1000ull;
    }
    uint64_t us = 0;
    if (current_tier == moe_TIER_DB || current_tier == moe_TIER_STREAM) {
        us += orchestrator_transfer_us(bytes, storage_bw);
    }
    if (target_tier == moe_TIER_VRAM && current_tier != moe_TIER_VRAM) {
        us += orchestrator_transfer_us(bytes, h2d_bw);
    }
    return us;
}

static moe_tier_t orchestrator_expert_tier_snapshot(
    moe_pc_engine* engine,
    int layer,
    int expert,
    uint64_t* out_bytes
) {
    if (out_bytes) {
        *out_bytes = 0;
    }
    if (!engine || layer < 0 || expert < 0) {
        return moe_TIER_DB;
    }
    const size_t slot = expert_dense_slot(engine, layer, expert);
    if (slot == tensor_slot_invalid() || slot >= engine->expert_states.size()) {
        return moe_TIER_DB;
    }
    std::lock_guard<std::mutex> lock(engine->map_mutex);
    const moe_block_state& state = engine->expert_states[slot];
    if (!state.active) {
        return moe_TIER_DB;
    }
    if (out_bytes) {
        *out_bytes = state.bytes;
    }
    return state.tier;
}

static void orchestrator_record_selected_expert_metrics(
    moe_pc_engine* engine,
    int layer,
    const int* selected_experts,
    uint32_t selected_count,
    moe_tier_t target_tier
) {
    if (!engine || !selected_experts || selected_count == 0) {
        return;
    }
    for (uint32_t i = 0; i < selected_count; ++i) {
        const int expert = selected_experts[i];
        const uint64_t bytes = orchestrator_expert_bytes(engine, layer, expert);
        uint64_t resident_bytes = 0;
        const moe_tier_t current = orchestrator_expert_tier_snapshot(engine, layer, expert, &resident_bytes);
        engine->expert_cache_request_count.fetch_add(1, std::memory_order_relaxed);
        if (current == moe_TIER_VRAM || current == moe_TIER_RAM) {
            engine->expert_cache_hit_count.fetch_add(1, std::memory_order_relaxed);
            if (current == moe_TIER_VRAM) {
                engine->expert_cache_vram_hit_count.fetch_add(1, std::memory_order_relaxed);
            } else {
                engine->expert_cache_ram_hit_count.fetch_add(1, std::memory_order_relaxed);
            }
            continue;
        }
        engine->expert_cache_miss_count.fetch_add(1, std::memory_order_relaxed);
        engine->expert_cache_miss_bytes.fetch_add(bytes, std::memory_order_relaxed);
        engine->expert_miss_latency_us_total.fetch_add(
            orchestrator_estimated_expert_miss_latency_us(engine, bytes ? bytes : resident_bytes, current, target_tier),
            std::memory_order_relaxed);
    }
}

static void orchestrator_update_feedback_metrics(moe_pc_engine* engine, uint32_t* committed_window, uint32_t max_window) {
    if (!engine || !committed_window || *committed_window == 0) {
        return;
    }
    const uint64_t requests = engine->expert_cache_request_count.load(std::memory_order_relaxed);
    const uint64_t hits = engine->expert_cache_hit_count.load(std::memory_order_relaxed);
    const uint64_t misses = engine->expert_cache_miss_count.load(std::memory_order_relaxed);
    const uint32_t hit_milli = requests ? (uint32_t)std::min<uint64_t>(1000ull, (hits * 1000ull) / requests) : 0u;
    const uint32_t miss_milli = requests ? (uint32_t)std::min<uint64_t>(1000ull, (misses * 1000ull) / requests) : 0u;
    engine->expert_hit_rate_milli.store(hit_milli, std::memory_order_relaxed);

    const uint64_t speculative = engine->speculative_prefetch_planned_count.load(std::memory_order_relaxed);
    const uint64_t dropped =
        engine->io_atomic.dropped_requests.load(std::memory_order_relaxed) +
        engine->io_atomic.urgent_dropped_requests.load(std::memory_order_relaxed);
    const uint32_t waste_milli = speculative ?
        (uint32_t)std::min<uint64_t>(1000ull, (dropped * 1000ull) / speculative) : 0u;
    engine->prefetch_waste_ratio_milli.store(waste_milli, std::memory_order_relaxed);

    const uint32_t gpu_workers = std::max(1u, runtime_gpu_workers(engine));
    const uint32_t gpu_active = engine->gpu_stage.inflight.load(std::memory_order_relaxed);
    const uint32_t gpu_busy_milli = (uint32_t)std::min<uint64_t>(
        1000ull,
        ((uint64_t)gpu_active * 1000ull) / gpu_workers);
    const uint32_t gpu_idle_milli = gpu_busy_milli >= 1000u ? 0u : 1000u - gpu_busy_milli;
    engine->gpu_copy_idle_milli.store(gpu_idle_milli, std::memory_order_relaxed);

    if (waste_milli >= 350u || engine->io_atomic.staging_slot_deficit.load(std::memory_order_relaxed) > 0u) {
        *committed_window = std::max(1u, *committed_window / 2u);
    } else if (gpu_idle_milli >= 300u && miss_milli >= 120u && waste_milli <= 200u) {
        *committed_window = std::min(max_window, *committed_window + 1u);
    }
    engine->adaptive_prefetch_window_last.store(*committed_window, std::memory_order_relaxed);
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
    //
    // BUGFIX 880: Separate dropped_requests and live_pressure adjustments ★★★ CRITICAL
    // Problem: Two independent mechanisms modify base_window sequentially:
    // 1. dropped_requests check reduces window (e.g., 6→3)
    // 2. live_pressure check modifies again (e.g., 3→1)
    // Hysteresis 3-streak sees final value which changes every call due to both variables.
    // Result: pending value oscillates, streak never accumulates, window vibrates 1-6.
    // Solution: Separate into two paths:
    // - dropped_requests → hysteresis → committed_window
    // - live_pressure → immediate cap (no hysteresis) on committed_window
    // Impact: Stable window convergence, eliminates 1-6 oscillation.
    uint32_t base_window = policy == moe_EXECUTION_PERFORMANCE ? 6u : policy == moe_EXECUTION_LOW_IMPACT ? 1u : 3u;
    // BUGFIX 949: Use prefetch_depth as max_window, not base_window override ★★★ CRITICAL
    // Problem: Setting base_window = prefetch_depth BEFORE dropped_requests path
    //   causes drops to reduce from prefetch_depth every cycle, but hysteresis
    //   streak never accumulates because base_window resets to prefetch_depth
    //   → window oscillates between prefetch_depth and prefetch_depth/2
    // Solution: Apply prefetch_depth as initial suggestion only if no pressure,
    //   then let dropped_requests independently reduce base_window
    // Impact: Dropped_requests hysteresis works correctly with juju strategy active
    // BUGFIX 995: Don't override base_window with prefetch_depth ★★★ CRITICAL
    // Problem: Setting base_window = min(prefetch_depth, max_window) before the
    //   dropped_requests path causes hysteresis streak to reset every cycle when
    //   juju strategy router recalculates prefetch_depth. The pending check at line 159
    //   (prefetch_window_pending == base_window) fails because base_window keeps
    //   changing → streak never reaches 3 → window oscillates between 1 and 6.
    // Solution: juju depth sets max_window only (upper bound). base_window is policy-only.
    //   prefetch_depth influence is applied through the max_window clamp at commit time.
    // Impact: Stable window convergence; juju depth still limits maximum depth.
    const uint32_t max_window = juju_cfg ?
        std::max(8u, juju_cfg->prefetch_depth ? juju_cfg->prefetch_depth : 8u) : 8u;
    // base_window stays at policy value — juju depth only limits via max_window clamp
    if (juju_cfg && engine->generation_token.load(std::memory_order_relaxed) < 128u) {
        base_window = std::max(base_window, 4u);
    }

    // BUGFIX 880: Step 1 - dropped_requests path with hysteresis
    if (engine) {
        const uint64_t dropped_now = engine->io_atomic.dropped_requests.load(std::memory_order_relaxed);
        const uint64_t dropped_delta = dropped_now > engine->last_dropped_requests_snapshot.load(std::memory_order_relaxed)
            ? dropped_now - engine->last_dropped_requests_snapshot.load(std::memory_order_relaxed) : 0;
        engine->last_dropped_requests_snapshot.store(dropped_now, std::memory_order_relaxed);

        const uint64_t dropped_expert_equiv = (dropped_delta + 2ull) / 3ull;
        if (dropped_expert_equiv > 0) {
            base_window = dropped_expert_equiv >= 4ull
                ? std::max(1u, base_window / 4u)
                : std::max(1u, base_window / 2u);
            // Reset growth streak on drop detection
            engine->prefetch_window_streak.store(0, std::memory_order_relaxed);
        }
    }

    // BUGFIX 880: Apply hysteresis to dropped_requests path only
    uint32_t committed_window = base_window;
    if (engine) {
        uint32_t current_window = engine->runtime_prefetch_window_layers.load(std::memory_order_relaxed);
        if (current_window == 0) {
            current_window = base_window;
        }
        if (base_window != current_window) {
            // BUGFIX 1000: Use atomic load/store for window hysteresis fields
            if (engine->prefetch_window_pending.load(std::memory_order_relaxed) == base_window) {
                engine->prefetch_window_streak.fetch_add(1, std::memory_order_relaxed);
            } else {
                engine->prefetch_window_pending.store(base_window, std::memory_order_relaxed);
                engine->prefetch_window_streak.store(1, std::memory_order_relaxed);
            }
            // Reduced from 5 to 3 for faster recovery
            if (engine->prefetch_window_streak.load(std::memory_order_relaxed) >= 3u) {
                committed_window = base_window;
                engine->prefetch_window_streak.store(0, std::memory_order_relaxed);
            } else {
                // Keep current window until 3 consecutive same signals
                committed_window = current_window;
            }
        } else {
            committed_window = base_window;
            engine->prefetch_window_streak.store(0, std::memory_order_relaxed);
        }
    }

    // BUGFIX 880: Step 2 - live_pressure immediate cap (no hysteresis)
    // This allows fast pipeline throttle without interfering with dropped_requests hysteresis
    if (engine && policy == moe_EXECUTION_PERFORMANCE) {
        const uint32_t queued_depth = engine->total_queue_depth.load(std::memory_order_relaxed);
        const uint32_t inflight =
            engine->disk_stage.inflight.load(std::memory_order_relaxed) +
            engine->pinned_stage.inflight.load(std::memory_order_relaxed) +
            engine->gpu_stage.inflight.load(std::memory_order_relaxed);
        const uint32_t live_pressure = queued_depth + inflight;
        const uint32_t total_workers = plan->disk_workers + plan->pinned_workers + plan->gpu_workers;
        if (live_pressure > total_workers * 4u) {
            // Immediate cap without hysteresis
            committed_window = std::max(2u, committed_window / 2u);
        } else if (live_pressure < total_workers) {
            // Immediate increase without hysteresis
            committed_window = std::min(max_window, committed_window + 1u);
        }
    }

    orchestrator_update_feedback_metrics(engine, &committed_window, max_window);
    plan->prefetch_window_layers = committed_window;
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
    // BUGFIX Problem B: Renamed route_sparse_to_cpu → allow_speculative_prefetch ★★ FIXED
    // Problem: Original name implied CPU offload routing, but feature was never implemented
    // Flag only affects throttle_speculative calculation (speculative prefetch control)
    // Solution: Rename to clarify actual purpose - controls whether speculative prefetch is allowed
    // Impact: Code now matches actual behavior, no functional change
    plan->allow_speculative_prefetch = cpu <= policy_cpu_target(policy);
    if (os_busy && policy != moe_EXECUTION_PERFORMANCE) {
        plan->disk_workers = plan->disk_workers > 1u ? plan->disk_workers / 2u : 1u;
        plan->pinned_workers = 1u;
        plan->gpu_workers = 1u;
        plan->prefetch_window_layers = plan->prefetch_window_layers > 1u ? plan->prefetch_window_layers - 1u : 1u;
    }
    plan->throttle_speculative = !plan->allow_gpu_dense || !plan->allow_speculative_prefetch || os_busy;
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
    moe_runtime_autotune_tick(engine);

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
    orchestrator_record_selected_expert_metrics(
        engine,
        current_layer,
        selected_experts,
        selected_count,
        out_plan->selected_tier);

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
