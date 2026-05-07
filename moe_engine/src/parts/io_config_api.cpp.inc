static uint64_t zero_copy_mmap_checksum_cpu(const void* ptr, uint64_t bytes) {
    if (!ptr || bytes == 0) {
        return 0;
    }
    const uint8_t* p = static_cast<const uint8_t*>(ptr);
    uint64_t v = 1469598103934665603ull;
    const uint64_t stride = bytes >= 64ull ? bytes / 64ull : 1ull;
    for (uint64_t i = 0; i < 64ull; ++i) {
        const uint64_t off = (i * stride + i * 1315423911ull) % bytes;
        v ^= static_cast<uint64_t>(p[off]);
        v *= 1099511628211ull;
    }
    return v;
}

static int validate_true_zero_copy_mmap_hip_kernel(
    moe_pc_engine* engine,
    void* device_ptr,
    void* mapped_file_ptr,
    size_t bytes
) {
    if (!engine ||
        !device_ptr ||
        !mapped_file_ptr ||
        bytes == 0 ||
        !engine->hip.hiprtcCreateProgram ||
        !engine->hip.hiprtcCompileProgram ||
        !engine->hip.hiprtcGetCodeSize ||
        !engine->hip.hiprtcGetCode ||
        !engine->hip.hiprtcDestroyProgram ||
        !engine->hip.hipModuleLoadData ||
        !engine->hip.hipModuleGetFunction ||
        !engine->hip.hipModuleLaunchKernel ||
        !engine->hip.hipModuleUnload ||
        !engine->hip.hipMalloc ||
        !engine->hip.hipFree ||
        !engine->hip.hipDeviceSynchronize ||
        !engine->hip.hipMemcpyDtoH) {
        return 0;
    }

    static const char source[] = R"MOE_HIP(
extern "C" __global__ void storagellm_mmap_checksum(
    const unsigned char* p,
    unsigned long long n,
    unsigned long long* out
) {
    unsigned long long v = 1469598103934665603ULL;
    unsigned long long stride = n >= 64ULL ? n / 64ULL : 1ULL;
    for (unsigned long long i = 0; i < 64ULL; ++i) {
        unsigned long long off = (i * stride + i * 1315423911ULL) % n;
        v ^= (unsigned long long)p[off];
        v *= 1099511628211ULL;
    }
    *out = v;
}
)MOE_HIP";

    moe_hiprtc_program prog = nullptr;
    void* module = nullptr;
    void* kernel = nullptr;
    uint64_t out_dev = 0;
    int ok = 0;

    if (engine->hip.hiprtcCreateProgram(
            &prog,
            source,
            "storagellm_mmap_checksum.hip",
            0,
            nullptr,
            nullptr) == 0 && prog &&
        engine->hip.hiprtcCompileProgram(prog, 0, nullptr) == 0) {
        size_t code_size = 0;
        if (engine->hip.hiprtcGetCodeSize(prog, &code_size) == 0 && code_size > 0) {
            std::vector<char> code(code_size);
            if (engine->hip.hiprtcGetCode(prog, code.data()) == 0 &&
                engine->hip.hipModuleLoadData(&module, code.data()) == 0 &&
                module &&
                engine->hip.hipModuleGetFunction(
                    &kernel,
                    module,
                    "storagellm_mmap_checksum") == 0 &&
                kernel &&
                engine->hip.hipMalloc(&out_dev, sizeof(uint64_t)) == 0 &&
                out_dev != 0) {
                void* device_arg = device_ptr;
                uint64_t n_arg = static_cast<uint64_t>(bytes);
                void* out_arg = reinterpret_cast<void*>(static_cast<uintptr_t>(out_dev));
                void* args[] = {&device_arg, &n_arg, &out_arg};
                if (engine->hip.hipModuleLaunchKernel(
                        kernel,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        0,
                        nullptr,
                        args,
                        nullptr) == 0 &&
                    engine->hip.hipDeviceSynchronize() == 0) {
                    uint64_t gpu_sum = 0;
                    if (engine->hip.hipMemcpyDtoH(&gpu_sum, out_dev, sizeof(gpu_sum)) == 0) {
                        const uint64_t cpu_sum =
                            zero_copy_mmap_checksum_cpu(mapped_file_ptr, static_cast<uint64_t>(bytes));
                        ok = (gpu_sum == cpu_sum) ? 1 : 0;
                    }
                }
            }
        }
    }

    if (out_dev != 0) {
        engine->hip.hipFree(out_dev);
    }
    if (module) {
        engine->hip.hipModuleUnload(module);
    }
    if (prog) {
        engine->hip.hiprtcDestroyProgram(&prog);
    }
    return ok;
}

static int validate_true_zero_copy_mmap_hip(
    moe_pc_engine* engine,
    void* mapped_file_ptr,
    size_t bytes
) {
    static constexpr unsigned int hipHostRegisterMappedFlag = 2u;
    if (!engine ||
        engine->backend_caps.backend != moe_BACKEND_HIP ||
        !mapped_file_ptr ||
        bytes < 4096 ||
        !engine->hip.hipHostRegister ||
        !engine->hip.hipHostGetDevicePointer ||
        !engine->hip.hipHostUnregister ||
        !device_set_current(engine)) {
        return 0;
    }
    if (engine->hip.hipHostRegister(mapped_file_ptr, bytes, hipHostRegisterMappedFlag) != 0) {
        return 0;
    }
    void* device_ptr = nullptr;
    const int got_device_ptr =
        engine->hip.hipHostGetDevicePointer(&device_ptr, mapped_file_ptr, 0) == 0 && device_ptr;
    const int ok = got_device_ptr ?
        validate_true_zero_copy_mmap_hip_kernel(engine, device_ptr, mapped_file_ptr, bytes) : 0;
    engine->hip.hipHostUnregister(mapped_file_ptr);
    return ok ? 1 : 0;
}

static int validate_true_zero_copy_mmap_level_zero(
    moe_pc_engine* engine,
    void* mapped_file_ptr,
    size_t bytes
) {
    (void)engine;
    (void)mapped_file_ptr;
    (void)bytes;
    // A copy engine round-trip is not proof that shader/compute code can
    // directly load mmap-backed pages. Keep Level Zero/SYCL fail-closed until
    // a backend kernel checksum path is available.
    return 0;
}

static int validate_true_zero_copy_mmap(moe_pc_engine* engine, void* mapped_file_ptr, size_t bytes) {
    if (!engine) {
        return 0;
    }
    if (engine->backend_caps.backend == moe_BACKEND_METAL &&
        engine->backend_caps.supports_unified_memory) {
        return 1;
    }
    if (!mapped_file_ptr || bytes < 4096 ||
        !engine->backend_caps.supports_unified_memory ||
        !engine->backend_caps.supports_zero_copy_mmap) {
        return 0;
    }
    if (engine->backend_caps.backend == moe_BACKEND_HIP) {
        return validate_true_zero_copy_mmap_hip(engine, mapped_file_ptr, bytes);
    }
    if (engine->backend_caps.backend == moe_BACKEND_LEVEL_ZERO ||
        engine->backend_caps.backend == moe_BACKEND_SYCL) {
        return validate_true_zero_copy_mmap_level_zero(engine, mapped_file_ptr, bytes);
    }
    return 0;
}

int moe_pc_engine_configure_io(moe_pc_engine_t* engine, const moe_io_config_t* config) {
    if (!engine) {
        return 0;
    }
    stop_io_workers(engine);
    reset_staging_slots(engine);
    free_staging_for_engine(engine, engine->staging, engine->staging_size, engine->staging_pinned,
                            engine->staging_numa_allocated, engine->staging_device_host_allocated);
    engine->staging = nullptr;
    engine->staging_size = 0;
    engine->staging_pinned = false;
    engine->pending_ram_reservation.store(0, std::memory_order_relaxed);
    engine->staging_numa_allocated = false;
    engine->staging_device_host_allocated = false;

    const int using_default_io_config = config ? 0 : 1;
    engine->io_config = config ? *config : moe_pc_default_io_config();
    if (engine->io_config.directstorage_staging_queue_capacity == 0) {
        engine->io_config.directstorage_staging_queue_capacity = 1024;
    }
    if (engine->io_config.directstorage_gpu_queue_capacity == 0) {
        engine->io_config.directstorage_gpu_queue_capacity = 1024;
    }
    if (using_default_io_config) {
        engine->io_config.allow_required_sync_fallback = 1;
        engine->io_config.require_bundle_native_experts = 1;
    }
    if (engine->backend_caps.backend == moe_BACKEND_METAL) {
        engine->io_config.enable_pinned_host = 0;
    }
    engine->true_zero_copy_validated.store(
        validate_true_zero_copy_mmap(engine, nullptr, 0) != 0,
        std::memory_order_release);
#if defined(_WIN32)
    if (engine->io_config.enable_directstorage &&
        engine->backend_caps.supports_directstorage &&
        compiled_directstorage_available()) {
        directstorage_queue_ensure(engine);
    } else {
        engine->runtime_directstorage_staging_ready.store(false, std::memory_order_release);
        engine->runtime_directstorage_gpu_ready.store(false, std::memory_order_release);
    }
#else
    device_gpudirect_storage_runtime_probe(engine);
#endif
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
        // BUGFIX 962: Bandwidth-aware prefetch_window_layers ★★★ CRITICAL
        // Problem: VRAM-only tiers (2/4/8) ignore storage speed. 24GiB GPU with
        // slow HDD gets window=8 → 8x prefetch concurrency overwhelms I/O → stalls.
        // Solution: Compute window from overlap_ms/compute_ms ratio when
        // measured_storage_bw_bytes_per_sec is available. Fall back to VRAM tiers.
        // moe_rt_compute_tflops_hint is in juju_runtime_strategy.cpp.inc (included earlier).
        // Impact: Window automatically adapts to actual I/O throughput.
        uint32_t vram_cap =
            !engine->config.prefer_gpu || engine->backend_caps.backend == moe_BACKEND_CPU ? 4u :
            vram > 24ull * gib ? 8u :
            vram >= 12ull * gib ? 4u : 2u;

        uint32_t computed_window = vram_cap; // default to VRAM-tier if no measurement
        const uint64_t storage_bw = engine->backend_caps.measured_storage_bw_bytes_per_sec;
        if (storage_bw > 0 && engine->config.prefer_gpu) {
            const double tflops = moe_rt_compute_tflops_hint(engine->backend_caps);
            const uint64_t expert_bytes = engine->max_tensor_stream_bytes.load(std::memory_order_relaxed);
            const uint64_t safe_expert = expert_bytes > 0 ? expert_bytes : 20ull * 1024ull * 1024ull;
            // overlap_ms: time to transfer one expert set from storage
            const double overlap_ms = ((double)safe_expert / (double)storage_bw) * 1000.0;
            // compute_ms: time for GPU to process one expert (rough TFLOP-based estimate)
            // Conservative: 1 GFLOP per expert as baseline compute
            const double compute_ms = tflops > 0.0 ? (1.0 / tflops) * 1000.0 : 1.0;
            if (compute_ms > 0.001) {
                const uint32_t bw_window = (uint32_t)std::ceil(overlap_ms / compute_ms) + 1u;
                computed_window = std::max(1u, std::min(bw_window, vram_cap));
            }
        }
        engine->io_config.prefetch_window_layers = computed_window;
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
        const uint64_t max_tensor_bytes = engine->max_tensor_stream_bytes.load(std::memory_order_relaxed);
        uint64_t tensor_bytes = max_tensor_bytes ?
            max_tensor_bytes : normal_bundle_max_or(32ull * 1024ull * 1024ull);
        uint64_t staging_target = tensor_bytes <= UINT64_MAX / streams ? tensor_bytes * streams : UINT64_MAX;
        staging_target = staging_target <= UINT64_MAX / 4ull ? staging_target * 4ull : UINT64_MAX;
        staging_target = std::max<uint64_t>(staging_target, engine->backend_caps.recommended_pinned_bytes);
        staging_target = std::max<uint64_t>(staging_target, (uint64_t)streams * 64ull * 1024ull * 1024ull);
        // BUGFIX 964: Relax staging cap and add stream-based floor ★★ PERFORMANCE
        // Problem: ram_budget/4 (15% system RAM) too tight for large triplet models
        // (512MiB/triplet × 4 GPU workers = 2GiB needed). 4.8GiB cap on 32GiB system
        // works for typical experts but chokes on 478B-class bundles.
        // Solution: Raise cap to ram_budget/3, add floor = streams * max_tensor * 6
        // to ensure enough staging for full pipeline depth. Absolute cap at 25% system RAM.
        // Impact: ~5% more RAM for staging on large models; auto-suppressed on ≤16GiB.
        if (max_tensor_bytes > 0) {
            const uint64_t stream_floor = (uint64_t)streams * max_tensor_bytes * 6ull;
            staging_target = std::max<uint64_t>(staging_target, stream_floor);
        }
        // BUGFIX 1004: GDS/DirectStorage active → reduce staging to fallback minimum ★★ IMPORTANT
        // Problem: enable_directstorage=1, enable_gpudirect_storage=1 by default.
        // GDS path bypasses staging buffers entirely, but slot_count = io_workers × gpu_workers × 2
        // → 32-core machine: 8×4×2 = 64 slots × 32MB = 2GB pinned RAM wasted.
        // Solution: If GDS is enabled AND backend supports it, reduce staging to normal_max × 4
        // (minimum fallback for non-GDS paths). Only apply after GDS capability confirmed.
        // Impact: Saves 1-2GB RAM on high-core-count systems with GDS hardware.
        const bool direct_io_runtime_active =
            engine->runtime_gpudirect_ready.load(std::memory_order_acquire) ||
            engine->runtime_directstorage_gpu_ready.load(std::memory_order_acquire);
        const bool staging_still_needed =
            !direct_io_runtime_active ||
            engine->runtime_directstorage_staging_ready.load(std::memory_order_acquire);
        if (direct_io_runtime_active && !staging_still_needed) {
            const moe_storage_constants_t* storage = moe_storage_constants();
            const uint64_t normal_max = storage ? storage->normal_bundle_max_bytes : 32ull * 1024ull * 1024ull;
            const uint64_t minimal_control_staging = normal_max * 2ull;
            staging_target = std::max<uint64_t>(
                minimal_control_staging,
                std::min<uint64_t>(staging_target, minimal_control_staging));
        }
        if (engine->config.ram_budget_bytes > 0) {
            staging_target = std::min<uint64_t>(staging_target, engine->config.ram_budget_bytes / 3ull);
        }
        // Absolute cap: 25% of total system RAM to avoid starving KV cache
        const uint64_t total_ram = storagellm::total_ram_bytes();
        if (total_ram > 0) {
            staging_target = std::min<uint64_t>(staging_target, total_ram / 4ull);
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
            "[storageLLM WARNING] level_zero_memory_headroom_pct=%u out of range [0,90], using default 20\n",
            (unsigned)engine->io_config.level_zero_memory_headroom_pct);
        // BUGFIX 902: Level Zero iGPU needs higher headroom
        engine->io_config.level_zero_memory_headroom_pct = 20;
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
    const uint32_t hw_threads = moe_cpu_thread_count_hint();
    const uint32_t balanced_disk_workers = moe_balanced_disk_worker_threads(
        hw_threads,
        engine->io_config.io_worker_threads,
        engine->io_config.gpu_worker_threads);
    if (engine->io_config.disk_worker_threads == 0) {
        engine->io_config.disk_worker_threads = balanced_disk_workers;
    } else if (using_default_io_config &&
        engine->io_config.disk_worker_threads > balanced_disk_workers) {
        engine->io_config.disk_worker_threads = balanced_disk_workers;
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
    if (using_default_io_config) {
        const uint32_t final_balanced_disk_workers = moe_balanced_disk_worker_threads(
            hw_threads,
            engine->io_config.io_worker_threads,
            engine->io_config.gpu_worker_threads);
        if (engine->io_config.disk_worker_threads > final_balanced_disk_workers) {
            engine->io_config.disk_worker_threads = final_balanced_disk_workers;
        }
    }
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

    // BUGFIX 907: Windows mmap default path - auto-enable prefer_overlapped_io ★★★ HIGH
    // Problem: DirectStorage disabled (most Windows users) → VirtualLock path used
    // → VirtualLock doesn't guarantee page residency → OS can evict pages before copy
    // → io_worker blocks on page fault during copy_mapped_range_to_staging
    // Solution: Auto-enable prefer_overlapped_io when DirectStorage is disabled
    // → Uses OVERLAPPED ReadFile (io_iocp_adapter.cpp.inc) instead of mmap
    // → Eliminates blocking page faults, async I/O completion via IOCP
    // Impact: Safe async I/O on Windows without DirectStorage, no page fault stalls
#ifdef _WIN32
    if (!engine->io_config.enable_directstorage && !engine->io_config.prefer_overlapped_io) {
        engine->io_config.prefer_overlapped_io = 1;
        uint32_t current_gen = engine->prefer_overlapped_io_generation.load(std::memory_order_relaxed);
        if ((current_gen & 1) == 0) {
            engine->prefer_overlapped_io_generation.fetch_add(1, std::memory_order_release);
        }
    }
#endif
    if (engine->io_config.enable_pinned_host && engine->io_config.pinned_staging_bytes > 0) {
        const uint64_t requested_staging_bytes = engine->io_config.pinned_staging_bytes;
        engine->staging = allocate_best_staging(engine,
            requested_staging_bytes,
            &engine->staging_size,
            &engine->staging_pinned,
            &engine->staging_numa_allocated,
            &engine->staging_device_host_allocated
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
            if (engine->max_tensor_stream_bytes.load(std::memory_order_relaxed) > 0) {
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
               stage_idle(engine->gpu_stage) &&
               engine->gds_blocking_pending.load(std::memory_order_acquire) == 0;
    });
    return 1;
}
