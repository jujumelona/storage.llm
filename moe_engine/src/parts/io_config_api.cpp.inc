int moe_pc_engine_configure_io(moe_pc_engine_t* engine, const moe_io_config_t* config) {
    if (!engine) {
        return 0;
    }
    stop_io_workers(engine);
    reset_staging_slots(engine);
    free_staging(engine->staging, engine->staging_size, engine->staging_pinned, engine->staging_numa_allocated);
    engine->staging = nullptr;
    engine->staging_size = 0;
    engine->staging_pinned = false;
    engine->pending_ram_reservation.store(0, std::memory_order_relaxed);
    engine->staging_numa_allocated = false;

    const int using_default_io_config = config ? 0 : 1;
    engine->io_config = config ? *config : moe_pc_default_io_config();
    const int staging_bypass = engine_bypass_staging_pipeline(engine);
    const int dma_gpu_backend =
        engine->backend_caps.backend == moe_BACKEND_CUDA ||
        engine->backend_caps.backend == moe_BACKEND_HIP ||
        engine->backend_caps.backend == moe_BACKEND_LEVEL_ZERO ||
        engine->backend_caps.backend == moe_BACKEND_SYCL;
    if (!staging_bypass && dma_gpu_backend && engine->backend_caps.supports_pinned_host) {
        engine->io_config.enable_multistage_pipeline = 1;
        engine->io_config.enable_pinned_host = 1;
    }
    if (using_default_io_config || engine->io_config.prefetch_window_layers == 0) {
        const uint64_t vram = engine->backend_caps.device_memory_bytes ?
            engine->backend_caps.device_memory_bytes : engine->backend_caps.recommended_vram_cache_bytes;
        const uint64_t gib = 1024ull * 1024ull * 1024ull;
        engine->io_config.prefetch_window_layers =
            !engine->config.prefer_gpu || engine->backend_caps.backend == moe_BACKEND_CPU ? 4u :
            vram > 24ull * gib ? 8u :
            vram >= 12ull * gib ? 4u : 2u;
    }
    if (engine->io_config.vram_eviction_threshold_pct == 0 && engine->config.prefer_gpu) {
        const uint64_t vram = engine->backend_caps.device_memory_bytes ?
            engine->backend_caps.device_memory_bytes : engine->backend_caps.recommended_vram_cache_bytes;
        const uint64_t gib = 1024ull * 1024ull * 1024ull;
        engine->io_config.vram_eviction_threshold_pct =
            (vram > 0 && vram < 12ull * gib) || !engine->backend_caps.supports_gpu_direct_file_io ? 65u : 75u;
    }
    if (!staging_bypass && engine->backend_caps.supports_pinned_host) {
        const uint32_t streams = std::max<uint32_t>(engine->backend_caps.copy_streams, 1u);
        uint64_t tensor_bytes = engine->max_tensor_stream_bytes ?
            engine->max_tensor_stream_bytes : normal_bundle_max_or(32ull * 1024ull * 1024ull);
        uint64_t staging_target = tensor_bytes <= UINT64_MAX / streams ? tensor_bytes * streams : UINT64_MAX;
        staging_target = staging_target <= UINT64_MAX / 4ull ? staging_target * 4ull : UINT64_MAX;
        staging_target = std::max<uint64_t>(staging_target, engine->backend_caps.recommended_pinned_bytes);
        staging_target = std::max<uint64_t>(staging_target, (uint64_t)streams * 64ull * 1024ull * 1024ull);
        if (engine->config.ram_budget_bytes > 0) {
            staging_target = std::min<uint64_t>(staging_target, engine->config.ram_budget_bytes / 2ull);
        }
        staging_target = moe_rt_align_4k(staging_target);
        if (staging_target > 0 &&
            (using_default_io_config || engine->io_config.pinned_staging_bytes < staging_target)) {
            engine->io_config.pinned_staging_bytes = staging_target;
        }
    }
    if (engine->io_config.io_worker_threads > disk_worker_cap()) {
        engine->io_config.io_worker_threads = disk_worker_cap();
    }
    if (engine->io_config.max_prefetch_queue == 0) {
        engine->io_config.max_prefetch_queue = 4096;
    } else if (engine->io_config.max_prefetch_queue < 4096) {
        engine->io_config.max_prefetch_queue = 4096;
    }
    if (engine->io_config.metal_memory_budget_pct != 0 &&
        (engine->io_config.metal_memory_budget_pct < 50 ||
         engine->io_config.metal_memory_budget_pct > 90)) {
        fprintf(stderr,
            "[storageLLM WARNING] metal_memory_budget_pct=%u out of safe range [50,90], using default 75\n",
            (unsigned)engine->io_config.metal_memory_budget_pct);
        engine->io_config.metal_memory_budget_pct = 75;
    }
    if (engine->io_config.vulkan_memory_headroom_pct > 90) {
        fprintf(stderr,
            "[storageLLM WARNING] vulkan_memory_headroom_pct=%u out of range [0,90], using default 10\n",
            (unsigned)engine->io_config.vulkan_memory_headroom_pct);
        engine->io_config.vulkan_memory_headroom_pct = 10;
    }
    if (engine->io_config.level_zero_memory_headroom_pct > 90) {
        fprintf(stderr,
            "[storageLLM WARNING] level_zero_memory_headroom_pct=%u out of range [0,90], using default 10\n",
            (unsigned)engine->io_config.level_zero_memory_headroom_pct);
        engine->io_config.level_zero_memory_headroom_pct = 10;
    }

    // BUGFIX 645: 큐 깊이 불균형 수정 - 단계별 독립 큐 깊이 설정 ★★★
    // Problem: disk_stage, pinned_stage, gpu_stage가 동일한 max_prefetch_queue 공유
    // disk는 mmap으로 빠르게 채워지고, pinned는 memcpy로 느린데 같은 cap 사용
    // pinned 포화 시 disk도 함께 막힘
    // Solution: 각 단계별 독립 큐 깊이 설정
    // - disk_stage: 절반 (빠른 mmap, 선행 거리 짧아도 OK)
    // - pinned_stage: 기본값 (병목 단계)
    // - gpu_stage: 1.5배 (DMA 병렬성 활용)
    const uint32_t base_queue = engine->io_config.max_prefetch_queue;
    engine->disk_stage_depth_cap = base_queue / 2;
    if (engine->disk_stage_depth_cap < 2048) {
        engine->disk_stage_depth_cap = 2048;  // 최소값 보장
    }
    engine->pinned_stage_depth_cap = base_queue;
    engine->gpu_stage_depth_cap = (base_queue * 3) / 2;
    if (engine->gpu_stage_depth_cap > 16384) {
        engine->gpu_stage_depth_cap = 16384;  // 메모리 과다 사용 방지
    }
    if (engine->io_config.topology_max_candidates == 0) {
        engine->io_config.topology_max_candidates = 1;
    }
    if (engine->io_config.topology_max_edges_per_source == 0) {
        engine->io_config.topology_max_edges_per_source = 1;
    }
    if (engine->io_config.disk_worker_threads == 0) {
        engine->io_config.disk_worker_threads = moe_default_disk_worker_threads(moe_cpu_thread_count_hint());
    } else if (using_default_io_config &&
        engine->io_config.disk_worker_threads < moe_default_disk_worker_threads(moe_cpu_thread_count_hint())) {
        engine->io_config.disk_worker_threads = moe_default_disk_worker_threads(moe_cpu_thread_count_hint());
    }
    engine->io_config.disk_worker_threads =
        clamp_to_cap(engine->io_config.disk_worker_threads, disk_worker_cap());
    if (engine->io_config.pinned_worker_threads == 0) {
        engine->io_config.pinned_worker_threads = 1;
    } else if (using_default_io_config && engine->backend_caps.copy_streams > 2u) {
        engine->io_config.pinned_worker_threads =
            std::max<uint32_t>(engine->io_config.pinned_worker_threads,
                std::min<uint32_t>(engine->backend_caps.copy_streams, 8u));
    }
    engine->io_config.pinned_worker_threads =
        clamp_to_cap(engine->io_config.pinned_worker_threads, pinned_worker_cap(engine));
    if (engine->io_config.gpu_worker_threads == 0) {
        engine->io_config.gpu_worker_threads = engine->backend_caps.copy_streams ?
            engine->backend_caps.copy_streams : 1u;
    } else if (using_default_io_config && engine->backend_caps.copy_streams > engine->io_config.gpu_worker_threads) {
        engine->io_config.gpu_worker_threads =
            std::min<uint32_t>(engine->backend_caps.copy_streams, 8u);
    }
    engine->io_config.gpu_worker_threads =
        clamp_to_cap(engine->io_config.gpu_worker_threads, gpu_worker_cap(engine));
    engine->runtime_prefetch_window_layers.store(
        engine->io_config.prefetch_window_layers, std::memory_order_relaxed);
    runtime_store_io_workers(
        engine,
        engine->io_config.disk_worker_threads,
        engine->io_config.pinned_worker_threads,
        engine->io_config.gpu_worker_threads);
    if (engine->io_config.direct_upload_min_bytes == 0) {
        const moe_storage_constants_t* storage = moe_storage_constants();
        engine->io_config.direct_upload_min_bytes =
            storage ? storage->normal_bundle_min_bytes : 1;
    }
    if (engine_bypass_staging_pipeline(engine)) {
        engine->io_config.enable_directstorage = 0;
        engine->io_config.enable_gpudirect_storage = 0;
        engine->io_config.enable_bam_nvme = 0;
        engine->io_config.enable_pinned_host = 0;
        engine->io_config.enable_multistage_pipeline = 0;
        engine->io_config.pinned_staging_bytes = 0;
        engine->io_config.pinned_worker_threads = 0;
        engine->io_config.gpu_worker_threads = 0;
    }
    if (engine->io_config.enable_pinned_host && engine->io_config.pinned_staging_bytes > 0) {
        const uint64_t requested_staging_bytes = engine->io_config.pinned_staging_bytes;
        engine->staging = allocate_best_staging(engine,
            requested_staging_bytes,
            &engine->staging_size,
            &engine->staging_pinned,
            &engine->staging_numa_allocated
        );
        if (!engine->staging) {
            engine->io_config.enable_pinned_host = 0;
            engine->io_config.pinned_staging_bytes = 0;
            engine->io_config.enable_multistage_pipeline = 0;
            fprintf(stderr,
                "[storagellm] pinned staging unavailable; continuing without pinned staging "
                "(requested=%llu bytes)\n",
                (unsigned long long)requested_staging_bytes);
        }
        if (engine->staging && !engine->staging_pinned) {
            fprintf(stderr,
                "[storagellm] staging allocated but not pinned; "
                "keeping multistage pipeline with explicit synchronous DMA fallback. "
                "Raise RLIMIT_MEMLOCK/ulimit -l for async pinned DMA.\n");
        }
        if (engine->staging) {
            std::memset(engine->staging, 0, static_cast<size_t>(std::min<uint64_t>(engine->staging_size, 4096)));
            if (engine->max_tensor_stream_bytes > 0) {
                configure_staging_slots(engine);
            } else {
                update_staging_slot_deficit(engine);
            }
        }
    }

    reset_io_stats_for_config(engine);
    start_io_workers(engine);
    return 1;
}

int moe_pc_engine_get_io_stats(const moe_pc_engine_t* engine, moe_io_stats_t* out_stats) {
    if (!engine || !out_stats) {
        return 0;
    }
    snapshot_io_stats(engine, out_stats);
    return 1;
}

int moe_pc_engine_wait_io(moe_pc_engine_t* engine) {
    if (!engine) {
        return 0;
    }
    std::unique_lock<std::mutex> lock(engine->done_mutex);
    engine->io_done_cv.wait(lock, [&] {
        return stage_idle(engine->disk_stage) &&
               stage_idle(engine->pinned_stage) &&
               stage_idle(engine->gpu_stage);
    });
    return 1;
}
