// Device allocator initialization and shutdown

static std::atomic<uint64_t>* device_pool_counter(moe_pc_engine* e, uint8_t pool) {
    if (!e) return nullptr;
    // BUGFIX 590: Check pool value range ★
    // Problem: pool can be out of range
    // Solution: Validate pool before use
    if (pool == moe_DEVICE_POOL_FIXED) return &e->device_fixed_bytes;
    if (pool == moe_DEVICE_POOL_ACTIVATION) return &e->device_activation_bytes;
    if (pool == moe_DEVICE_POOL_EXPERT) return &e->device_expert_bytes;
    // BUGFIX 591: Return nullptr for invalid pool instead of stream_bytes ★
    // Problem: Invalid pool returns stream_bytes, causing wrong accounting
    // Solution: Return nullptr for invalid pool
    if (pool == moe_DEVICE_POOL_STREAM) return &e->device_stream_bytes;
    // BUGFIX 문제4: Add KV cache pool counter ★★★
    if (pool == moe_DEVICE_POOL_KV_CACHE) return &e->device_kv_cache_bytes;
    return nullptr;
}

static void device_stream_pool_init(moe_pc_engine* e);

static int device_allocator_init(moe_pc_engine* e) {
    if (!e || e->device_allocator_ready.load() != 0) return e && e->device_allocator_ready.load() > 0;
    if (!e->config.prefer_gpu) return 0;
    if (e->backend_caps.backend != moe_BACKEND_CUDA && e->backend_caps.backend != moe_BACKEND_HIP && e->backend_caps.backend != moe_BACKEND_METAL && e->backend_caps.backend != moe_BACKEND_VULKAN && e->backend_caps.backend != moe_BACKEND_LEVEL_ZERO && e->backend_caps.backend != moe_BACKEND_SYCL) return 0;
    std::lock_guard<std::mutex> lock(e->device_mutex);
    if (e->device_allocator_ready.load() != 0) return e->device_allocator_ready.load() > 0;

    // QKV State (~0.69 GiB) is now tracked via add_tier_usage dynamically,
    // so we no longer pre-reserve it here to avoid double-counting.

    if (e->backend_caps.backend == moe_BACKEND_CUDA) {
        if (!cuda_driver_load(&e->cuda) || e->cuda.cuInit(0) != 0 ||
            e->cuda.cuDeviceGet(&e->cuda.device, (int)e->backend_caps.device_id) != 0 ||
            e->cuda.cuDevicePrimaryCtxRetain(&e->cuda.context, e->cuda.device) != 0 ||
            e->cuda.cuCtxSetCurrent(e->cuda.context) != 0) {
            moe_close_shared_library(e->cuda.library);
            e->cuda = moe_cuda_driver_api{};
            e->device_allocator_ready.store(-1);
            return 0;
        }
        size_t free_b = 0, total_b = 0;
        if (e->cuda.cuMemGetInfo(&free_b, &total_b) == 0) {
            e->device_free_bytes.store((uint64_t)free_b);
            e->device_total_bytes.store((uint64_t)total_b);
            if (e->backend_caps.device_memory_bytes == 0) e->backend_caps.device_memory_bytes = (uint64_t)free_b;
        }
    } else if (e->backend_caps.backend == moe_BACKEND_HIP) {
        if (!hip_driver_load(&e->hip) || e->hip.hipInit(0) != 0 ||
            e->hip.hipDeviceGet(&e->hip.device, (int)e->backend_caps.device_id) != 0 ||
            e->hip.hipDevicePrimaryCtxRetain(&e->hip.context, e->hip.device) != 0 ||
            e->hip.hipCtxSetCurrent(e->hip.context) != 0) {
            moe_close_shared_library(e->hip.library);
            e->hip = moe_hip_driver_api{};
            e->device_allocator_ready.store(-1);
            return 0;
        }
        size_t free_b = 0, total_b = 0;
        if (e->hip.hipMemGetInfo(&free_b, &total_b) == 0) {
            const uint64_t safe_free_b = ((uint64_t)free_b * 90ull) / 100ull;
            e->device_free_bytes.store(safe_free_b);
            e->device_total_bytes.store((uint64_t)total_b);
            if (e->backend_caps.device_memory_bytes == 0) e->backend_caps.device_memory_bytes = safe_free_b;
        }
    // BUGFIX 80: Issue 2 - Fix Metal budget calculation and remove dead code.
    // Problem: Metal uses full system_memory_bytes as VRAM budget
    // 32GB Mac -> 32GB VRAM budget -> OS pressure and excessive expert caching.
    // Also: if (false) block is dead code from refactoring
    // Solution: Reserve 25% for OS, remove dead code
    // BUGFIX 124: Issue 29 - Metal unified memory budget integration ★★
    // Set RAM budget to 0 for Metal, all tracking in VRAM budget (unified memory).
    } else if (e->backend_caps.backend == moe_BACKEND_METAL) {
        // Apple Silicon unified memory: reserve 25% for OS
        const uint64_t sys = e->backend_caps.system_memory_bytes > 0
            ? e->backend_caps.system_memory_bytes : 16ull * 1024 * 1024 * 1024;
        if (e->io_config.metal_memory_budget_pct == 0) {
            const uint64_t sys_gb = sys / (1024ull * 1024ull * 1024ull);
            uint64_t auto_pct = 75u;
            if (sys_gb >= 64ull) {
                auto_pct = 80u;
            } else if (sys_gb >= 32ull) {
                auto_pct = 77u;
            }
            e->io_config.metal_memory_budget_pct = (uint8_t)auto_pct;
            fprintf(stderr,
                "[storageLLM INFO] Auto-detected Metal unified-memory budget: %u%% for %llu GB system memory\n",
                (unsigned)e->io_config.metal_memory_budget_pct,
                (unsigned long long)sys_gb);
        }
        const uint64_t usable = (sys * (uint64_t)e->io_config.metal_memory_budget_pct) / 100ull;
        if (e->backend_caps.device_memory_bytes == 0) {
            e->backend_caps.device_memory_bytes = usable;
        }
        e->device_free_bytes.store(e->backend_caps.device_memory_bytes);
        e->device_total_bytes.store(e->backend_caps.device_memory_bytes);
        // BUGFIX 124: Disable RAM budget for Metal - unified memory tracked in VRAM budget only
        e->config.ram_budget_bytes = 0;
    } else if (e->backend_caps.backend == moe_BACKEND_LEVEL_ZERO) {
        if (e->backend_caps.device_memory_bytes == 0) {
            e->backend_caps.device_memory_bytes = e->backend_caps.system_memory_bytes > 0
                ? e->backend_caps.system_memory_bytes : 16ull * 1024ull * 1024ull * 1024ull;
        }
        uint64_t free_mem = 0;
        uint64_t total_mem = e->backend_caps.device_memory_bytes;
        if (e->ze.zeDeviceGetMemoryProperties && e->ze.device) {
            struct ze_device_memory_properties_t_local {
                uint32_t stype;
                void* pNext;
                uint32_t flags;
                uint32_t maxClockRate;
                uint32_t maxBusWidth;
                uint64_t totalSize;
                char name[256];
            };
            struct ze_device_memory_ext_properties_t_local {
                uint32_t stype;
                void* pNext;
                uint32_t type;
                uint64_t physicalSize;
                uint64_t allocatedSize;
            } ext_props{};
            ext_props.stype = 0x0001000B;
            ze_device_memory_properties_t_local mem_props{};
            mem_props.stype = 0x1;
            mem_props.pNext = &ext_props;
            uint32_t count = 1;
            if (e->ze.zeDeviceGetMemoryProperties(e->ze.device, &count, &mem_props) == 0 &&
                mem_props.totalSize > 0) {
                total_mem = mem_props.totalSize;
                if (ext_props.allocatedSize > 0) {
                    const uint64_t avail = mem_props.totalSize > ext_props.allocatedSize
                        ? mem_props.totalSize - ext_props.allocatedSize : mem_props.totalSize;
                    free_mem = (avail * 90ull) / 100ull;
                } else {
                    free_mem = (mem_props.totalSize * 85ull) / 100ull;
                }
            }
        }
        if (free_mem == 0) {
            free_mem = (e->backend_caps.device_memory_bytes * 85ull) / 100ull;
        }
        e->device_free_bytes.store(free_mem);
        e->device_total_bytes.store(total_mem);
    } else if (e->backend_caps.backend == moe_BACKEND_SYCL) {
        if (e->backend_caps.device_memory_bytes == 0) {
            e->backend_caps.device_memory_bytes = e->backend_caps.system_memory_bytes > 0
                ? e->backend_caps.system_memory_bytes : 16ull * 1024ull * 1024ull * 1024ull;
        }
        uint64_t free_mem = 0;
        uint64_t total_mem = e->backend_caps.device_memory_bytes;
        if (e->ze.zeDeviceGetMemoryProperties && e->ze.device) {
            struct ze_device_memory_properties_t_local {
                uint32_t stype;
                void* pNext;
                uint32_t flags;
                uint32_t maxClockRate;
                uint32_t maxBusWidth;
                uint64_t totalSize;
                char name[256];
            };
            struct ze_device_memory_ext_properties_t_local {
                uint32_t stype;
                void* pNext;
                uint32_t type;
                uint64_t physicalSize;
                uint64_t allocatedSize;
            } ext_props{};
            ext_props.stype = 0x0001000B;
            ze_device_memory_properties_t_local mem_props{};
            mem_props.stype = 0x1;
            mem_props.pNext = &ext_props;
            uint32_t count = 1;
            if (e->ze.zeDeviceGetMemoryProperties(e->ze.device, &count, &mem_props) == 0 &&
                mem_props.totalSize > 0) {
                total_mem = mem_props.totalSize;
                if (ext_props.allocatedSize > 0) {
                    const uint64_t avail = mem_props.totalSize > ext_props.allocatedSize
                        ? mem_props.totalSize - ext_props.allocatedSize : mem_props.totalSize;
                    free_mem = (avail * 90ull) / 100ull;
                }
            }
        }
        if (free_mem == 0) {
            free_mem = (e->backend_caps.device_memory_bytes * 85ull) / 100ull;
        }
        e->device_free_bytes.store(free_mem);
        e->device_total_bytes.store(total_mem);
    } else if (e->backend_caps.backend == moe_BACKEND_VULKAN) {
        // Vulkan: use reported device memory or fallback
        if (e->backend_caps.device_memory_bytes == 0) {
            e->backend_caps.device_memory_bytes = e->backend_caps.system_memory_bytes > 0
                ? e->backend_caps.system_memory_bytes : 16ull * 1024 * 1024 * 1024;
        }

        uint64_t free_mem = 0;
        uint64_t total_mem = e->backend_caps.device_memory_bytes;
        int exact_queried = 0;

        // BUGFIX 200: Implement actual VRAM query for Vulkan and Level Zero ★
        // Problem: Vulkan/Level Zero use 85% estimate, never query actual VRAM
        // device_refresh_mem_info has query code but only runs on failure (hot path excluded)
        // Solution: Query at init time to set accurate device_free_bytes baseline
        if (e->backend_caps.backend == moe_BACKEND_VULKAN && e->vk.vkGetPhysicalDeviceMemoryProperties2 && e->vk.physicalDevice) {
            // Vulkan: Query memory budget using VK_EXT_memory_budget
            struct VkPhysicalDeviceMemoryBudgetPropertiesEXT_local {
                uint32_t sType;
                void* pNext;
                uint64_t heapUsage[16];
                uint64_t heapBudget[16];
            } budget{};
            budget.sType = 1000237000; // VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT
            struct VkMemoryHeap_local {
                uint64_t size;
                uint32_t flags;
            };
            struct VkPhysicalDeviceMemoryProperties_local {
                uint32_t memoryTypeCount;
                struct { uint32_t propertyFlags; uint32_t heapIndex; } memoryTypes[32];
                uint32_t memoryHeapCount;
                VkMemoryHeap_local memoryHeaps[16];
            };
            struct VkPhysicalDeviceMemoryProperties2_local {
                uint32_t sType;
                void* pNext;
                VkPhysicalDeviceMemoryProperties_local memoryProperties;
            } props{};
            props.sType = 1000059006; // VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2
            props.pNext = &budget;
            e->vk.vkGetPhysicalDeviceMemoryProperties2(e->vk.physicalDevice, &props);
            uint64_t calc_free = 0;
            uint64_t calc_total = 0;
            // BUGFIX 466: memoryHeapCount 범위 체크 (최대 16개)
            uint32_t heap_count = props.memoryProperties.memoryHeapCount;
            if (heap_count > 16) heap_count = 16;
            for (uint32_t i = 0; i < heap_count; ++i) {
                if (props.memoryProperties.memoryHeaps[i].flags & 1) { // VK_MEMORY_HEAP_DEVICE_LOCAL_BIT
                    calc_total += props.memoryProperties.memoryHeaps[i].size;
                    if (budget.heapBudget[i] > 0) {
                        uint64_t avail = (budget.heapBudget[i] > budget.heapUsage[i]) ? (budget.heapBudget[i] - budget.heapUsage[i]) : 0;
                        calc_free += (avail * 90ull) / 100ull;
                    } else {
                        // Fallback to 85% if VK_EXT_memory_budget is missing
                        calc_free += (props.memoryProperties.memoryHeaps[i].size * 85ull) / 100ull;
                    }
                }
            }
            if (calc_total > 0) {
                free_mem = calc_free;
                total_mem = calc_total;
                exact_queried = 1;
            }
        } else if (e->backend_caps.backend == moe_BACKEND_LEVEL_ZERO && e->ze.zeDeviceGetMemoryProperties && e->ze.device) {
            // Level Zero: Query memory properties
            struct ze_device_memory_properties_t_local {
                uint32_t stype;
                void* pNext;
                uint32_t flags;
                uint32_t maxClockRate;
                uint32_t maxBusWidth;
                uint64_t totalSize;
                char name[256];
            };
            struct ze_device_memory_ext_properties_t_local {
                uint32_t stype;
                void* pNext;
                uint32_t type;
                uint64_t physicalSize;
                uint64_t allocatedSize;
            } ext_props{};
            ext_props.stype = 0x0001000B; // ZE_STRUCTURE_TYPE_DEVICE_MEMORY_EXT_PROPERTIES
            ze_device_memory_properties_t_local mem_props{};
            mem_props.stype = 0x1; // ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES
            mem_props.pNext = &ext_props;
            uint32_t count = 1;
            if (e->ze.zeDeviceGetMemoryProperties(e->ze.device, &count, &mem_props) == 0) {
                total_mem = mem_props.totalSize;
                if (ext_props.allocatedSize > 0) {
                    uint64_t avail = (mem_props.totalSize > ext_props.allocatedSize) ? (mem_props.totalSize - ext_props.allocatedSize) : mem_props.totalSize;
                    free_mem = (avail * 90ull) / 100ull;
                } else {
                    // Fallback to 85% if allocatedSize not populated
                    free_mem = (mem_props.totalSize * 85ull) / 100ull;
                }
                exact_queried = 1;
            }
        }
#ifdef HAVE_SYCL
        else if (e->backend_caps.backend == moe_BACKEND_SYCL) {
            // SYCL: Level Zero 백엔드를 통해 쿼리하거나 OpenCL 메모리 쿼리 사용
            // Level Zero가 SYCL 아래에 있는 경우 기존 LZ 쿼리 재사용 가능
            // 없으면 sycl::device::get_info<sycl::info::device::global_mem_size>() 사용
        }
#endif
        
        if (!exact_queried) {
            // Use 85% as a conservative free memory estimate to avoid OOM
            free_mem = (e->backend_caps.device_memory_bytes * 85ull) / 100ull;
        }
        e->device_free_bytes.store(free_mem);
        e->device_total_bytes.store(total_mem);
    }
    device_refresh_mem_info(e);

    // Fix 27: Run a quick microbenchmark to measure actual TFLOPS
    // The hardcoded fallbacks (22/35 TFLOPS) are wildly inaccurate for many GPUs.
    if (e->backend_caps.tflops_hint <= 0.0 &&
        (e->backend_caps.backend == moe_BACKEND_CUDA || e->backend_caps.backend == moe_BACKEND_HIP)) {
        const size_t bench_bytes = 512 * 512 * sizeof(float);
        CUdeviceptr d_a = 0, d_b = 0;
        int alloc_ok = 0;
        if (e->backend_caps.backend == moe_BACKEND_CUDA && e->cuda.cuMemAlloc) {
            alloc_ok = (e->cuda.cuMemAlloc(&d_a, bench_bytes) == 0 &&
                        e->cuda.cuMemAlloc(&d_b, bench_bytes) == 0) ? 1 : 0;
        } else if (e->backend_caps.backend == moe_BACKEND_HIP && e->hip.hipMalloc) {
            alloc_ok = (e->hip.hipMalloc(&d_a, bench_bytes) == 0 &&
                        e->hip.hipMalloc(&d_b, bench_bytes) == 0) ? 1 : 0;
        }
        if (alloc_ok && d_a && d_b) {
            auto t0 = std::chrono::steady_clock::now();
            const int iters = 20;
            for (int i = 0; i < iters; ++i) {
                if (e->backend_caps.backend == moe_BACKEND_CUDA && e->cuda.cuMemcpyDtoD)
                    e->cuda.cuMemcpyDtoD(d_b, d_a, bench_bytes);
                else if (e->backend_caps.backend == moe_BACKEND_HIP && e->hip.hipMemcpyDtoD)
                    e->hip.hipMemcpyDtoD(d_b, d_a, bench_bytes);
            }
            if (e->backend_caps.backend == moe_BACKEND_CUDA && e->cuda.cuCtxSynchronize)
                e->cuda.cuCtxSynchronize();
            else if (e->backend_caps.backend == moe_BACKEND_HIP && e->hip.hipDeviceSynchronize)
                e->hip.hipDeviceSynchronize();
            auto t1 = std::chrono::steady_clock::now();
            double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
            if (elapsed_s > 1e-6) {
                double bw_gbs = (double)(bench_bytes * iters * 2) / elapsed_s / 1e9;
                double est_tflops = bw_gbs * 0.05;
                if (est_tflops > 1.0 && est_tflops < 500.0) {
                    e->backend_caps.tflops_hint = est_tflops;
                }
            }
        }
        if (d_a) {
            if (e->backend_caps.backend == moe_BACKEND_CUDA) e->cuda.cuMemFree(d_a);
            else if (e->backend_caps.backend == moe_BACKEND_HIP) e->hip.hipFree(d_a);
        }
        if (d_b) {
            if (e->backend_caps.backend == moe_BACKEND_CUDA) e->cuda.cuMemFree(d_b);
            else if (e->backend_caps.backend == moe_BACKEND_HIP) e->hip.hipFree(d_b);
        }
    }

    e->device_allocator_ready.store(1);
    // Stream pool init moved to engine start
    return 1;
}

static int device_set_current(moe_pc_engine* e) {
    static thread_local CUcontext tls_ctx = nullptr;
    if (!e) return 0;
    if (e->backend_caps.backend == moe_BACKEND_CUDA) {
        if (!e->cuda.cuCtxSetCurrent) return 0;
        if (tls_ctx == e->cuda.context) return 1;
        if (e->cuda.cuCtxSetCurrent(e->cuda.context) != 0) return 0;
        tls_ctx = e->cuda.context;
        return 1;
    } else if (e->backend_caps.backend == moe_BACKEND_HIP) {
        if (!e->hip.hipCtxSetCurrent) return 0;
        if (tls_ctx == e->hip.context) return 1;
        if (e->hip.hipCtxSetCurrent(e->hip.context) != 0) return 0;
        tls_ctx = e->hip.context;
        return 1;
    }
    return 0;
}

static void device_allocator_shutdown(moe_pc_engine* e) {
    if (!e) return;
    std::vector<std::pair<CUdeviceptr, uint64_t>> deferred_free_ptrs;
    std::vector<uint64_t> keys;
    {
        std::lock_guard<std::mutex> lock(e->device_mutex);
        for (const auto& it : e->device_allocations) keys.push_back(it.first);
        for (uint64_t key : keys) device_free_locked(e, key, &deferred_free_ptrs);
        decltype(e->activation_evict_heap) empty;
        e->activation_evict_heap.swap(empty);
    }
    device_free_deferred_ptrs(e, deferred_free_ptrs);
    if (e->backend_caps.backend == moe_BACKEND_CUDA) {
        if (e->cuda.cuDevicePrimaryCtxRelease && e->cuda.context) e->cuda.cuDevicePrimaryCtxRelease(e->cuda.device);
        moe_close_shared_library(e->cuda.library);
        e->cuda = moe_cuda_driver_api{};
    } else if (e->backend_caps.backend == moe_BACKEND_HIP) {
        if (e->hip.hipDevicePrimaryCtxRelease && e->hip.context) e->hip.hipDevicePrimaryCtxRelease(e->hip.device);
        moe_close_shared_library(e->hip.library);
        e->hip = moe_hip_driver_api{};
    }
    e->device_allocator_ready.store(0);
}
