static int prefetch_one_raw_span(
    moe_pc_engine* e,
    const moe_storage_raw_span_spec_t& s,
    uint8_t pool,
    int allow_sync_fallback = 1
);

// BUGFIX 62: Forward declaration for dynamic batch_max function ?占썩쁾
static uint32_t moe_gpu_stage_batch_max(const moe_pc_engine* engine);

static int io_worker_requeue_disk_task(
    moe_pc_engine* engine,
    const moe_prefetch_task& task,
    moe_transfer_stage_t stage = moe_TRANSFER_STAGE_DISK_TO_RAM,
    uint32_t attempts = 1,
    uint32_t backoff_ms = 0
) {
    if (!engine) {
        return 0;
    }
    if (attempts == 0) {
        attempts = 1;
    }
    for (uint32_t i = 0; i < attempts && !engine->io_stop.load(); ++i) {
        moe_prefetch_task retry = task;
        retry.stage = stage;
        retry.urgent = (retry.requeue_count >= 3u || retry.stage == moe_TRANSFER_STAGE_RAM_TO_PINNED) ? 1 : 0;
        if (retry.requeue_count < UINT8_MAX) {
            ++retry.requeue_count;
        }
        // BUGFIX 711: Respect disk_stage_depth_cap in requeue path ?끸쁾??        if (engine->disk_stage_depth_cap > 0 &&
            engine->disk_stage.depth.load(std::memory_order_relaxed) >= engine->disk_stage_depth_cap) {
            // Queue full - wait
            std::this_thread::yield();
            continue;
        }
        const int ok = enqueue_stage_or_retry(engine, engine->disk_stage, retry) ? 1 : 0;
        if (ok) {
            engine->disk_stage.cv.notify_one();
            return 1;
        }
        if (backoff_ms > 0 && i + 1u < attempts) {
            std::unique_lock<std::mutex> lock(engine->disk_stage.mtx);
            engine->disk_stage.pop_cv.wait_for(lock, std::chrono::milliseconds(backoff_ms));
        }
    }
    return 0;
}

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((noinline))
#endif
static uint64_t io_worker_device_allocated_bytes(const moe_pc_engine* engine) {
    return engine ? engine->device_allocated_bytes.load(std::memory_order_relaxed) : 0;
}

static void proactive_eviction_worker_loop(moe_pc_engine* engine) {
    if (!engine) {
        return;
    }
    // BUGFIX 95: Issue 16 - Cache io_stop load outside loop ??    // Problem: Atomic load on every iteration adds overhead
    // Solution: Cache load, re-check only after CV wakeup
    const auto should_stop = [&]() { return engine->io_stop.load(std::memory_order_relaxed); };
    while (!should_stop()) {
        // BUGFIX 81: Problem 6 - Replace 5ms polling with event-driven CV ?끸쁾
        // Old: 5ms sleep caused CPU overhead and potential oscillation.
        // New: Wait on vram_eviction_cv. Triggered instantly when allocation > 85%.
        // Fallback to 50ms wait to periodically check RAM eviction.
        bool vram_eviction_flag = false;
        bool ram_eviction_flag = false;
        {
            std::unique_lock<std::mutex> lock(engine->vram_eviction_mutex);
            engine->vram_eviction_cv.wait_for(lock, std::chrono::milliseconds(25), [&]() {
                return engine->vram_eviction_needed.load(std::memory_order_relaxed) ||
                       engine->ram_eviction_needed.load(std::memory_order_relaxed) ||
                       engine->io_stop.load(std::memory_order_relaxed);
            });
            // BUGFIX 臾몄젣2: proactive_eviction_worker_loop?먯꽌 ?뚮옒洹몃? eviction ?꾩뿉 clear ?끸쁾??            // Problem: CV 源⑥뼱??吏곹썑 ?뚮옒洹몃? false濡??명똿 ??eviction ?ㅽ뙣 ???ъ떆???놁쓬
            // Solution: ?뚮옒洹몃? ?쎌뼱??濡쒖뺄 蹂?섏뿉 ??ν븯怨? eviction ?꾨즺 ?꾩뿉留?clear
            vram_eviction_flag = engine->vram_eviction_needed.load(std::memory_order_relaxed);
            ram_eviction_flag = engine->ram_eviction_needed.load(std::memory_order_relaxed);
        }
        if (engine->io_stop.load(std::memory_order_relaxed)) {
            break;
        }
        const uint64_t vram_bud = tier_budget(engine, moe_TIER_VRAM);
        const uint64_t ram_bud = tier_budget(engine, moe_TIER_RAM);
        const uint64_t ram_used = tier_budget(engine, moe_TIER_RAM) > 0 ? tier_used(engine, moe_TIER_RAM) : 0;
        const uint64_t vram_used = tier_used(engine, moe_TIER_VRAM);

        // BUGFIX 57: Issue 2 - Lower proactive eviction trigger to 85% ?끸쁾
        // Problem: Attention prefetch cutoff at 87%, proactive eviction at 90%
        // ??% dead zone where attention blocked but eviction not started
        // ?묬PU fallback or mmap streaming for attention weights
        // Solution: Lower eviction trigger to 85%, target to 75%
        // Ensures eviction starts BEFORE attention prefetch cutoff
        // Trade-off: More frequent eviction, but prevents attention stalls
        // BUGFIX 99: Precompute percentage thresholds
        const uint64_t vram_target_pct  = engine->io_config.vram_eviction_threshold_pct ? engine->io_config.vram_eviction_threshold_pct : 75u;
        const uint64_t vram_trigger_pct  = vram_target_pct + 10u <= 100u ? vram_target_pct + 10u : 100u;
        const uint64_t vram_target = (vram_bud * vram_target_pct) / 100ull;
        const uint64_t vram_threshold_85 = (vram_bud * vram_trigger_pct) / 100ull;
        bool vram_eviction_succeeded = false;
        if (vram_bud > 0 && vram_used > vram_threshold_85 && vram_used > vram_target) {
            std::vector<std::pair<int, int>> freed;
            demote_cold_until(engine, moe_TIER_VRAM, vram_bud - vram_target, freed);
            if (!freed.empty() && !engine->io_stop.load(std::memory_order_relaxed)) {
                // BUGFIX 36: Reduce batch_size from 8 to 4 to lower mutex hold time ????
                // Old: 8 횞 cuMemFree (100쨉s each) = 800쨉s device_mutex hold
                // New: 4 횞 cuMemFree = 400쨉s device_mutex hold
                // All device_copy_tensor calls wait on device_mutex ??halving hold time improves throughput
                // Trade-off: More batches needed for large evictions, but each batch blocks less
                const size_t free_batch_size =
                    (engine->backend_caps.copy_streams >= 8u ||
                     engine->backend_caps.dma_bandwidth_gbps >= 100.0) ? 8u : 4u;
                device_free_experts_batched(engine, freed, free_batch_size);
                vram_eviction_succeeded = true;
            }
        }

        // BUGFIX 臾몄젣2: eviction ?꾨즺 ???뚮옒洹?clear (?깃났 ?쒖뿉留?
        // ?ㅽ뙣??寃쎌슦 ?뚮옒洹몃? ?좎??섏뿬 ?ㅼ쓬 ?ъ씠?댁뿉???ъ떆??        bool retry_vram_eviction = false;
        if (vram_eviction_flag) {
            const uint64_t vram_used_after = tier_used(engine, moe_TIER_VRAM);
            // eviction???깃났?덇굅???ъ슜瑜좎씠 紐⑺몴 ?댄븯濡??⑥뼱吏?寃쎌슦?먮쭔 ?뚮옒洹?clear
            if (vram_used_after > vram_threshold_85 && vram_used_after > vram_target) {
                if (vram_eviction_succeeded) {
                    engine->vram_eviction_needed.store(true, std::memory_order_relaxed);
                    retry_vram_eviction = true;
                } else {
                    engine->vram_eviction_needed.store(false, std::memory_order_relaxed);
                }
            } else {
                engine->vram_eviction_needed.store(false, std::memory_order_relaxed);
            }
        }

        // BUGFIX 124: Issue 29 - Skip RAM eviction for Metal unified memory ?끸쁾
        // Metal uses unified memory budget tracked in VRAM only (ram_budget_bytes = 0).
        // RAM eviction would interfere with OS's native unified memory pressure management.
        // Metal automatically pages out to backing store when needed.
        const bool skip_ram_eviction = (engine->backend_caps.backend == moe_BACKEND_METAL &&
                                        engine->backend_caps.supports_unified_memory);

        bool ram_eviction_succeeded = false;
        if (!skip_ram_eviction) {
            const uint64_t ram_target_pct  = engine->io_config.ram_eviction_threshold_pct  ? engine->io_config.ram_eviction_threshold_pct  : 80u;
            const uint64_t ram_trigger_pct  = ram_target_pct + 5u <= 100u ? ram_target_pct + 5u : 100u;
            const uint64_t ram_target = (ram_bud * ram_target_pct) / 100ull;
            const uint64_t ram_threshold_85 = (ram_bud * ram_trigger_pct) / 100ull;
            if (ram_bud > 0 && ram_used > ram_threshold_85 && ram_used > ram_target) {
                std::vector<std::pair<int, int>> ignored;
                demote_cold_until(engine, moe_TIER_RAM, ram_bud - ram_target, ignored);
                ram_eviction_succeeded = true;
            }
        }

        // BUGFIX 臾몄젣2: RAM eviction ?꾨즺 ???뚮옒洹?clear (?깃났 ?쒖뿉留?
        bool retry_ram_eviction = false;
        if (ram_eviction_flag) {
            const uint64_t ram_used_after = tier_used(engine, moe_TIER_RAM);
            const uint64_t ram_target_pct  = engine->io_config.ram_eviction_threshold_pct  ? engine->io_config.ram_eviction_threshold_pct  : 80u;
            const uint64_t ram_target = (ram_bud * ram_target_pct) / 100ull;
            const uint64_t ram_trigger_pct = ram_target_pct + 5u <= 100u ? ram_target_pct + 5u : 100u;
            const uint64_t ram_trigger = (ram_bud * ram_trigger_pct) / 100ull;
            if (!skip_ram_eviction && ram_used_after > ram_trigger && ram_used_after > ram_target) {
                engine->ram_eviction_needed.store(true, std::memory_order_relaxed);
                retry_ram_eviction = true;
            } else {
                engine->ram_eviction_needed.store(false, std::memory_order_relaxed);
            }
        }

        if ((retry_vram_eviction || retry_ram_eviction) &&
            !engine->io_stop.load(std::memory_order_relaxed)) {
            std::unique_lock<std::mutex> lock(engine->vram_eviction_mutex);
            engine->vram_eviction_cv.wait_for(lock, std::chrono::milliseconds(25), [&]() {
                return engine->io_stop.load(std::memory_order_relaxed);
            });
        }

        // BUGFIX 235: Check and compact eviction heaps if needed
        // Problem: pop_valid_evict_candidate sets evict_heap_compact_needed flag when stale ratio high,
        // but proactive_eviction_worker_loop never checks this flag to actually compact the heap
        // Solution: Check flag and compact all heaps periodically
        // Impact: Prevents heap from accumulating stale entries
        if (engine->evict_heap_compact_needed.load(std::memory_order_relaxed)) {
            engine->evict_heap_compact_needed.store(false, std::memory_order_relaxed);
            for (int idx = 0; idx < 3; ++idx) {
                compact_evict_heap(engine, idx);
            }
        }

        // BUGFIX 臾몄젣B: Adaptive queue depth adjustment ?끸쁾
        // Problem: Static queue depth caps don't adapt to runtime bottlenecks
        // pinned_stage saturation ??disk_stage should reduce cap
        // gpu_stage idle ??pinned_stage can increase cap
        // Solution: Periodically adjust caps based on queue utilization
        // Trade-off: Increased pinned cap uses more staging memory
        {
            const uint32_t disk_depth = engine->disk_stage.depth.load(std::memory_order_relaxed);
            const uint32_t pin_depth = engine->pinned_stage.depth.load(std::memory_order_relaxed);
            const uint32_t gpu_depth = engine->gpu_stage.depth.load(std::memory_order_relaxed);
            const uint32_t base = engine->io_config.max_prefetch_queue;

            // pinned_stage saturated ??reduce disk_stage cap to prevent overflow
            if (pin_depth > engine->pinned_stage_depth_cap * 9 / 10) {
                const uint32_t new_disk_cap = std::max(
                    engine->disk_stage_depth_cap * 3 / 4,
                    std::max(512u, base / 4)  // Never go below 512 or base/4
                );
                if (new_disk_cap != engine->disk_stage_depth_cap) {
                    engine->disk_stage_depth_cap = new_disk_cap;
                }
            }

            // gpu_stage idle AND pinned_stage underutilized ??increase pinned cap
            if (gpu_depth < 4 && pin_depth < engine->pinned_stage_depth_cap * 3 / 10) {
                const uint32_t max_pin_cap = std::min({base * 2,
                    (uint32_t)engine->staging_slots.size(),    // never exceed actual slot count
                    engine->disk_stage_depth_cap * 2});        // disk supply can't feed more
                const uint32_t new_pin_cap = std::min(
                    engine->pinned_stage_depth_cap * 5 / 4,
                    max_pin_cap
                );
                if (new_pin_cap != engine->pinned_stage_depth_cap &&
                    new_pin_cap <= max_pin_cap) {
                    engine->pinned_stage_depth_cap = new_pin_cap;
                }
            }
        }

    }
}

static void io_worker_loop(moe_pc_engine* engine, uint32_t worker_index) {
    if (!engine) {
        return;
    }

    // BUGFIX 27: Add StreamGuard to prevent CUDA stream leak ?占썩쁾
    // Old: device_stream_destroy_current only on normal exit paths
    // New: RAII guard ensures cleanup on exception/abnormal termination
    // Matches gpu_worker_loop pattern for consistency
    struct StreamGuard {
        moe_pc_engine* e;
        ~StreamGuard() {
            if (e) {
                device_stream_destroy_current(e);
            }
