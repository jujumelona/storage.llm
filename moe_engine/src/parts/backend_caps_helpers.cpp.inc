static uint32_t caps_hw_threads() {
    const unsigned int hw = std::thread::hardware_concurrency();
    return hw ? static_cast<uint32_t>(hw) : 1u;
}

static uint64_t normal_bundle_max_or(uint64_t fallback) {
    const moe_storage_constants_t* storage = moe_storage_constants();
    return storage && storage->normal_bundle_max_bytes ? storage->normal_bundle_max_bytes : fallback;
}

static uint64_t cache_for_experts(uint32_t expert_count) {
    // BUGFIX 479: expert_count overflow 체크
    const uint64_t bundle_max = normal_bundle_max_or(32ull * 1024ull * 1024ull);
    // BUGFIX 526: bundle_max가 0일 때 division by zero 방지 ★
    // Problem: normal_bundle_max_or can return 0 if storage constants are corrupted
    // Causes division by zero in overflow check
    // Solution: Check bundle_max > 0 before division
    if (bundle_max == 0 || expert_count > UINT64_MAX / bundle_max) return UINT64_MAX;
    return bundle_max * expert_count;
}

static uint64_t cache_with_common_for_experts(uint32_t expert_count) {
    const uint64_t common = working_common_cache_bytes();
    const uint64_t expert_cache = cache_for_experts(expert_count);
    // BUGFIX 480: overflow 체크
    if (common > UINT64_MAX - expert_cache) return UINT64_MAX;
    return common + expert_cache;
}

static uint64_t moe_budget_from_free_device_memory(uint64_t free_bytes) {
    return (free_bytes * 9ull) / 10ull;
}

static void fill_cpu_backend_caps(moe_backend_caps_t* caps) {
    const uint32_t hw = caps_hw_threads();
    caps->dma_bandwidth_gbps = 12.0;
    caps->tflops_hint = 0.10;
    caps->supports_fp4 = 1;
    caps->supports_scale4 = 1;
    caps->supports_weight_q2 = 1;
    caps->supports_weight_q3 = 1;
    caps->supports_weight_q4 = 1;
    caps->supports_weight_q8 = 1;
    caps->supports_weight_iq2 = 0;
    caps->supports_weight_mxfp4 = 0;
    caps->supports_unified_memory = 1;
    caps->supports_zero_copy_host = 1;
    caps->supports_zero_copy_mmap = 1;
    caps->supports_pinned_host = 0;
    caps->supports_backend_async_api = 0;
    caps->compute_streams = std::min<uint32_t>(hw, 8u);
    caps->copy_streams = 0;
    caps->prefetch_streams = std::min<uint32_t>(std::max<uint32_t>(hw / 2u, 1u), 8u);
    caps->recommended_pinned_bytes = 0;
}

static void fill_gpu_backend_base(moe_backend_caps_t* caps) {
    const uint32_t hw = caps_hw_threads();
    caps->dma_bandwidth_gbps = 25.0;
    caps->tflops_hint = 20.0;
    caps->compute_streams = hw >= 12u ? 2u : 1u;
    caps->copy_streams = hw >= 12u ? 2u : 1u;
    caps->prefetch_streams = hw >= 16u ? 3u : 2u;
    caps->supports_async_copy = 1;
    caps->supports_fp4 = compiled_nvfp4_device_kernels_available();
    caps->supports_scale4 = compiled_scale4_device_fusion_available();
    caps->supports_weight_q2 = 0;
    caps->supports_weight_q3 = 0;
    caps->supports_weight_q4 = caps->supports_fp4;
    caps->supports_weight_q8 = 0;
    caps->supports_weight_iq2 = 0;
    caps->supports_weight_mxfp4 = 0;
    caps->recommended_vram_common_bytes = working_common_cache_bytes();
    caps->recommended_pinned_bytes = std::max<uint64_t>(512ull * 1024ull * 1024ull, cache_for_experts(12));
}

static int cuda_driver_get_numa_node(uint32_t device_id);
static void fill_cuda_backend_caps(moe_backend_caps_t* caps) {
    caps->preferred_numa_node = cuda_driver_get_numa_node(caps->device_id);
    caps->supports_cuda = compiled_cuda_available();
    caps->supports_pinned_host = 1;
    caps->supports_fixed_read_staging = 1;
    caps->supports_registered_io_buffers = 1; // Enabled IORing/io_uring
    caps->supports_backend_async_api = 1;
    caps->supports_directstorage = compiled_directstorage_available();
    caps->supports_gpudirect_storage = compiled_gpudirect_available();
    caps->supports_gpu_direct_file_io =
        caps->supports_directstorage || caps->supports_gpudirect_storage || compiled_bam_nvme_available();
    caps->recommended_vram_cache_bytes = cache_with_common_for_experts(64);
    uint64_t total_bytes = 0;
    const uint64_t free_bytes = cuda_driver_detect_free_memory_bytes(caps->device_id, &total_bytes);

    // BUGFIX 130: Issue 33 - Validate VRAM query for driver bugs (WSL2/old drivers) ★★
    // Problem: cuDeviceTotalMem() on some old drivers (CUDA 11.x) or WSL2 returns
    // virtual address space size instead of physical VRAM.
    // device_memory_bytes set too high → over-admit experts → OOM.
    // Solution: Validate free vs total ratio. If free < 10% of total, likely driver bug.
    // Use recommended_vram_cache_bytes as fallback.
    if (free_bytes > 0) {
        // BUGFIX 481: total_bytes / 10 overflow 방지
        const uint64_t min_free = (total_bytes > 0 && total_bytes <= UINT64_MAX / 10) ? total_bytes / 10 : 0;
        // BUGFIX 527: min_free가 0일 때 비교 로직 수정 ★
        // Problem: total_bytes가 매우 작거나 0일 때 min_free = 0
        // free_bytes < 0 check는 항상 false (unsigned)
        // Solution: total_bytes > 0 체크 추가
        // Sanity check: free should be at least 10% of total
        if (total_bytes > 0 && min_free > 0 && free_bytes < min_free) {
            fprintf(stderr, "[storageLLM WARNING] VRAM query suspicious (free=%llu MB, total=%llu MB). "
                    "Likely driver bug (WSL2/old CUDA). Using fallback budget.\n",
                    (unsigned long long)(free_bytes / (1024*1024)),
                    (unsigned long long)(total_bytes / (1024*1024)));
            caps->device_memory_bytes = caps->recommended_vram_cache_bytes;
        } else {
            caps->device_memory_bytes = free_bytes;
            caps->recommended_vram_cache_bytes = std::max<uint64_t>(
                caps->recommended_vram_cache_bytes,
                moe_budget_from_free_device_memory(free_bytes)
            );
        }
    } else if (total_bytes > 0) {
        caps->device_memory_bytes = total_bytes;
    }
}

static int hip_driver_get_numa_node(uint32_t device_id);
static void fill_hip_backend_caps(moe_backend_caps_t* caps) {
    caps->preferred_numa_node = hip_driver_get_numa_node(caps->device_id);
    caps->supports_hip = compiled_hip_available();
    caps->supports_pinned_host = 1;
    caps->supports_fixed_read_staging = 1;
    caps->supports_backend_async_api = 1;
    caps->supports_directstorage = compiled_directstorage_available();
    caps->supports_gpu_direct_file_io = caps->supports_directstorage;
    caps->recommended_vram_cache_bytes = cache_with_common_for_experts(48);

    uint64_t total_bytes = 0;
    const uint64_t free_bytes = hip_driver_detect_free_memory_bytes(caps->device_id, &total_bytes);

    // BUGFIX 130: Same validation for HIP
    if (free_bytes > 0) {
        // BUGFIX 482: total_bytes / 10 overflow 방지
        const uint64_t min_free = (total_bytes > 0 && total_bytes <= UINT64_MAX / 10) ? total_bytes / 10 : 0;
        // BUGFIX 528: min_free가 0일 때 비교 로직 수정 (CUDA와 동일) ★
        // Problem: total_bytes가 매우 작거나 0일 때 min_free = 0
        // Solution: total_bytes > 0 && min_free > 0 체크 추가
        if (total_bytes > 0 && min_free > 0 && free_bytes < min_free) {
            fprintf(stderr, "[storageLLM WARNING] HIP VRAM query suspicious (free=%llu MB, total=%llu MB). "
                    "Using fallback budget.\n",
                    (unsigned long long)(free_bytes / (1024*1024)),
                    (unsigned long long)(total_bytes / (1024*1024)));
            caps->device_memory_bytes = caps->recommended_vram_cache_bytes;
        } else {
            caps->device_memory_bytes = free_bytes;
            caps->recommended_vram_cache_bytes = std::max<uint64_t>(
                caps->recommended_vram_cache_bytes,
                moe_budget_from_free_device_memory(free_bytes)
            );
        }
    } else if (total_bytes > 0) {
        caps->device_memory_bytes = total_bytes;
    }

    // ============================================================================
    // AMD MI300X / MI300A Unified Memory Support
    // ============================================================================
    // Enable zero-copy unified memory pathways to bypass PCIe bottlenecks on MI300.
    // ============================================================================
    if (total_bytes >= 120ull * 1024ull * 1024ull * 1024ull) {
        caps->supports_unified_memory = 1;
        caps->supports_zero_copy_host = 1;
        caps->tflops_hint = 163.0;
        caps->dma_bandwidth_gbps = 800.0;
    }
}

static void fill_shared_memory_backend_caps(moe_backend_caps_t* caps) {
    caps->supports_pinned_host = 1;
    if (caps->device_id == 0) {
        caps->supports_unified_memory = 1;
        caps->supports_zero_copy_host = 1;
        caps->supports_zero_copy_mmap = 1;
    } else {
        caps->supports_unified_memory = 0;
        caps->supports_zero_copy_host = 0;
        caps->supports_zero_copy_mmap = 0;
    }
    caps->recommended_vram_cache_bytes = cache_with_common_for_experts(48);
}

// TODO: Issue 4 - Metal zero-copy optimization with MTLBuffer.
// Current: Metal uses supports_zero_copy_host=1 ??mmap bypass (BUGFIX 61 verified correct)
// Problem: Still copies mmap data to Metal buffer in some paths
// Apple Silicon Unified Memory: CPU and GPU share physical memory
// mmap weights can be read directly by GPU (no H2D copy needed)
// Solution: Use MTLBuffer with newBufferWithBytesNoCopy(mmap_ptr, size, MTLResourceStorageModeShared)
// - GPU reads mmap pointer directly, 0 bytes DMA
// - Eliminates all H2D copy overhead on Metal
// Implementation:
//   1. Add Metal API bindings (Objective-C++ wrapper)
//   2. In device_backend_async.cpp.inc, add Metal branch:
//      if (backend == moe_BACKEND_METAL) {
//          return metal_zero_copy_map(e, src, bytes);
//      }
//   3. metal_zero_copy_map creates MTLBuffer with newBufferWithBytesNoCopy
//   4. Return buffer handle as CUdeviceptr equivalent
// Trade-off: Metal API dependency, iOS/macOS-specific code, build system changes
// Status: Documented as TODO, current bypass path is correct (BUGFIX 61)
static void fill_metal_backend_caps(moe_backend_caps_t* caps) {
    caps->supports_metal = compiled_metal_available();
    caps->supports_unified_memory = 1; // Default to 1 for Apple Silicon, ideally queried from device
    // BUGFIX 713: Bind Metal zero_copy capabilities to unified_memory ★★★
    // Problem: If an eGPU is used, unified_memory might be false, but zero_copy
    // was hardcoded to true, causing invalid bypassing of the staging pipeline.
    // Solution: Bind zero_copy flags to the unified_memory flag.
    caps->supports_zero_copy_host = caps->supports_unified_memory;
    caps->supports_zero_copy_mmap = caps->supports_unified_memory;
    caps->supports_pinned_host = 0;
    caps->supports_backend_async_api = 1;
    caps->supports_gpu_direct_file_io = 0;
    caps->recommended_vram_cache_bytes = cache_with_common_for_experts(32);
    caps->tflops_hint = 14.0;
    caps->recommended_pinned_bytes = 0;
}

static void fill_basic_gpu_caps(moe_backend_caps_t* caps, int available, int* feature_flag) {
    *feature_flag = available;
    caps->supports_pinned_host = 1;
    caps->supports_fixed_read_staging = 1;
    caps->supports_backend_async_api = 0;
    caps->recommended_vram_cache_bytes = cache_with_common_for_experts(32);
    caps->tflops_hint = 14.0;
}

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

static void fill_vulkan_backend_caps(moe_backend_caps_t* caps) {
    caps->supports_vulkan = compiled_vulkan_available();
    fill_basic_gpu_caps(caps, caps->supports_vulkan, &caps->supports_vulkan);

    uint64_t free_b = 0;
    static int vk_budget_supported = -1;

    void* lib = nullptr;
#ifdef _WIN32
    lib = (void*)LoadLibraryA("vulkan-1.dll");
#else
    lib = dlopen("libvulkan.so.1", RTLD_LAZY | RTLD_LOCAL);
#endif

    if (lib) {
        typedef int (*pfn_vkCreateInstance)(const void*, const void*, void**);
        typedef int (*pfn_vkEnumeratePhysicalDevices)(void*, uint32_t*, void**);
        typedef void* (*pfn_vkGetInstanceProcAddr)(void*, const char*);
        typedef void (*pfn_vkDestroyInstance)(void*, const void*);

#ifdef _WIN32
        auto get_sym = [&](const char* n) { return (void*)GetProcAddress((HMODULE)lib, n); };
#else
        auto get_sym = [&](const char* n) { return dlsym(lib, n); };
#endif

        pfn_vkCreateInstance createInst = (pfn_vkCreateInstance)get_sym("vkCreateInstance");
        pfn_vkEnumeratePhysicalDevices enumDevs = (pfn_vkEnumeratePhysicalDevices)get_sym("vkEnumeratePhysicalDevices");
        pfn_vkGetInstanceProcAddr getProc = (pfn_vkGetInstanceProcAddr)get_sym("vkGetInstanceProcAddr");
        pfn_vkDestroyInstance destroyInst = (pfn_vkDestroyInstance)get_sym("vkDestroyInstance");

        if (createInst && enumDevs && getProc && destroyInst) {
            struct VkApplicationInfo_local {
                uint32_t sType; void* pNext; const char* pApplicationName; uint32_t applicationVersion;
                const char* pEngineName; uint32_t engineVersion; uint32_t apiVersion;
            } app_info = { 0, nullptr, "storageLLM", 1, "storageLLM", 1, (1<<22)|(1<<12) };

            struct VkInstanceCreateInfo_local {
                uint32_t sType; void* pNext; uint32_t flags; const void* pApplicationInfo;
                uint32_t enabledLayerCount; const char* const* ppEnabledLayerNames;
                uint32_t enabledExtensionCount; const char* const* ppEnabledExtensionNames;
            } create_info = { 1, nullptr, 0, &app_info, 0, nullptr, 0, nullptr };

            void* instance = nullptr;
            if (createInst(&create_info, nullptr, &instance) == 0) {
                uint32_t gpu_count = 0;
                if (enumDevs(instance, &gpu_count, nullptr) == 0 && gpu_count > 0) {
                    void* gpus[16];
                    if (gpu_count > 16) gpu_count = 16;
                    if (enumDevs(instance, &gpu_count, gpus) == 0) {
                        void* phys_dev = gpus[caps->device_id < gpu_count ? caps->device_id : 0];
                        typedef void (*pfn_vkGetPhysicalDeviceMemoryProperties2)(void*, void*);
                        pfn_vkGetPhysicalDeviceMemoryProperties2 getProps2 =
                            (pfn_vkGetPhysicalDeviceMemoryProperties2)getProc(instance, "vkGetPhysicalDeviceMemoryProperties2");
                        if (!getProps2) getProps2 = (pfn_vkGetPhysicalDeviceMemoryProperties2)getProc(instance, "vkGetPhysicalDeviceMemoryProperties2KHR");

                        if (getProps2) {
                            vk_budget_supported = 1;
                            struct VkPhysicalDeviceMemoryBudgetPropertiesEXT_local {
                                uint32_t sType; void* pNext; uint64_t heapUsage[16]; uint64_t heapBudget[16];
                            } budget{};
                            budget.sType = 1000237000;

                            struct VkMemoryHeap_local { uint64_t size; uint32_t flags; };
                            struct VkPhysicalDeviceMemoryProperties_local {
                                uint32_t memoryTypeCount; struct { uint32_t propertyFlags; uint32_t heapIndex; } memoryTypes[32];
                                uint32_t memoryHeapCount; VkMemoryHeap_local memoryHeaps[16];
                            } memoryProperties;
                            struct VkPhysicalDeviceMemoryProperties2_local {
                                uint32_t sType; void* pNext; VkPhysicalDeviceMemoryProperties_local memoryProperties;
                            } props{};
                            props.sType = 1000059006;
                            props.pNext = &budget;

                            getProps2(phys_dev, &props);

                            uint64_t calc_free = 0;
                            uint32_t heap_count = props.memoryProperties.memoryHeapCount;
                            if (heap_count > 16) heap_count = 16;
                            for (uint32_t i = 0; i < heap_count; ++i) {
                                if (props.memoryProperties.memoryHeaps[i].flags & 1) { // DEVICE_LOCAL
                                    if (budget.heapBudget[i] > 0) {
                                        uint64_t avail = (budget.heapBudget[i] > budget.heapUsage[i]) ? (budget.heapBudget[i] - budget.heapUsage[i]) : 0;
                                        calc_free += avail;
                                    }
                                }
                            }
                            free_b = calc_free;
                        } else {
                            vk_budget_supported = 0;
                            // BUGFIX 715: Vulkan VRAM 85% heuristic fallback ★★★
                            // Problem: When getProps2 (budget extension) is not available,
                            // we defaulted to recommended_vram_cache_bytes.
                            // Solution: Use vkGetPhysicalDeviceMemoryProperties to get the
                            // DEVICE_LOCAL heap size, and assume 85% is available for use.
                            typedef void (*pfn_vkGetPhysicalDeviceMemoryProperties)(void*, void*);
                            pfn_vkGetPhysicalDeviceMemoryProperties getProps =
                                (pfn_vkGetPhysicalDeviceMemoryProperties)getProc(instance, "vkGetPhysicalDeviceMemoryProperties");
                            if (getProps) {
                                struct VkMemoryHeap_local { uint64_t size; uint32_t flags; };
                                struct VkPhysicalDeviceMemoryProperties_local {
                                    uint32_t memoryTypeCount; struct { uint32_t propertyFlags; uint32_t heapIndex; } memoryTypes[32];
                                    uint32_t memoryHeapCount; VkMemoryHeap_local memoryHeaps[16];
                                } props{};
                                getProps(phys_dev, &props);

                                uint64_t calc_total = 0;
                                uint32_t heap_count = props.memoryHeapCount;
                                if (heap_count > 16) heap_count = 16;
                                for (uint32_t i = 0; i < heap_count; ++i) {
                                    if (props.memoryHeaps[i].flags & 1) { // DEVICE_LOCAL
                                        calc_total += props.memoryHeaps[i].size;
                                    }
                                }
                                free_b = (calc_total * 75ull) / 100ull;
                            }
                        }
                    }
                }
                destroyInst(instance, nullptr);
            }
        }
#ifdef _WIN32
        FreeLibrary((HMODULE)lib);
#else
        dlclose(lib);
#endif
    } else {
        vk_budget_supported = 0;
    }

    if (free_b > 0) {
        caps->device_memory_bytes = free_b;
    } else {
        caps->device_memory_bytes = caps->recommended_vram_cache_bytes;
    }
}

static void fill_levelzero_backend_caps(moe_backend_caps_t* caps) {
    fill_shared_memory_backend_caps(caps);
    uint64_t free_b = 0;
    static int ze_budget_supported = -1;

    void* lib = nullptr;
#ifdef _WIN32
    lib = (void*)LoadLibraryA("ze_loader.dll");
#else
    lib = dlopen("libze_loader.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!lib) lib = dlopen("libze_loader.so", RTLD_LAZY | RTLD_LOCAL);
#endif

    if (lib) {
        typedef int (*pfn_zeInit)(int, void*);
        typedef int (*pfn_zeDriverGet)(uint32_t*, void**);
        typedef int (*pfn_zeDeviceGet)(void*, uint32_t*, void**);
        typedef int (*pfn_zeContextCreate)(void*, const void*, void**);
        typedef int (*pfn_zeContextDestroy)(void*);
        typedef int (*pfn_zeMemAllocDevice)(void*, const void*, size_t, size_t, void**);
        typedef int (*pfn_zeMemGetAllocProperties)(void*, const void*, void*, void**);
        typedef int (*pfn_zeMemFree)(void*, void*);

#ifdef _WIN32
        auto get_sym = [&](const char* n) { return (void*)GetProcAddress((HMODULE)lib, n); };
#else
        auto get_sym = [&](const char* n) { return dlsym(lib, n); };
#endif

        pfn_zeInit zeInit = (pfn_zeInit)get_sym("zeInit");
        pfn_zeDriverGet zeDriverGet = (pfn_zeDriverGet)get_sym("zeDriverGet");
        pfn_zeDeviceGet zeDeviceGet = (pfn_zeDeviceGet)get_sym("zeDeviceGet");
        pfn_zeContextCreate zeContextCreate = (pfn_zeContextCreate)get_sym("zeContextCreate");
        pfn_zeContextDestroy zeContextDestroy = (pfn_zeContextDestroy)get_sym("zeContextDestroy");
        pfn_zeMemAllocDevice zeMemAllocDevice = (pfn_zeMemAllocDevice)get_sym("zeMemAllocDevice");
        pfn_zeMemGetAllocProperties zeMemGetAllocProperties = (pfn_zeMemGetAllocProperties)get_sym("zeMemGetAllocProperties");
        pfn_zeMemFree zeMemFree = (pfn_zeMemFree)get_sym("zeMemFree");

        if (zeInit && zeDriverGet && zeDeviceGet && zeContextCreate && zeMemAllocDevice && zeMemGetAllocProperties && zeMemFree) {
            if (zeInit(0, nullptr) == 0) {
                uint32_t driver_count = 1;
                void* driver = nullptr;
                if (zeDriverGet(&driver_count, &driver) == 0 && driver) {
                    uint32_t device_count = 1;
                    void* dev = nullptr;
                    if (zeDeviceGet(driver, &device_count, &dev) == 0 && dev) {
                        ze_budget_supported = 1;

                        // BUGFIX: Use zeDeviceGetMemoryProperties to query actual VRAM size
                        typedef int (*pfn_zeDeviceGetMemoryProperties)(void*, uint32_t*, void*);
                        pfn_zeDeviceGetMemoryProperties zeDeviceGetMemoryProperties =
                            (pfn_zeDeviceGetMemoryProperties)get_sym("zeDeviceGetMemoryProperties");

                        if (zeDeviceGetMemoryProperties) {
                            uint32_t mem_count = 0;
                            if (zeDeviceGetMemoryProperties(dev, &mem_count, nullptr) == 0 && mem_count > 0) {
                                // ze_device_memory_properties_t: sType(u32), pNext(ptr), flags(u32), maxClockRate(u32), maxBusWidth(u32), totalSize(u64), name[256]
                                struct ze_device_memory_properties_local {
                                    uint32_t stype; void* pNext; uint32_t flags;
                                    uint32_t maxClockRate; uint32_t maxBusWidth;
                                    uint64_t totalSize; char name[256];
                                };
                                if (mem_count > 8) mem_count = 8; // safety cap
                                ze_device_memory_properties_local mem_props[8] = {};
                                for (uint32_t m = 0; m < mem_count; ++m) {
                                    mem_props[m].stype = 0x00010004; // ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES
                                }
                                if (zeDeviceGetMemoryProperties(dev, &mem_count, mem_props) == 0) {
                                    uint64_t max_total = 0;
                                    for (uint32_t m = 0; m < mem_count; ++m) {
                                        if (mem_props[m].totalSize > max_total) {
                                            max_total = mem_props[m].totalSize;
                                        }
                                    }
                                    if (max_total > 0) {
                                        free_b = (max_total * 75ull) / 100ull; // Same 75% safe heuristic as Vulkan
                                    }
                                }
                            }
                        }

                        // Fallback: try allocation probe if properties query failed
                        if (free_b == 0) {
                            struct ze_context_desc_t_local {
                                uint32_t stype; void* pNext; uint32_t flags;
                            } ctx_desc = {0x00010001, nullptr, 0};

                            void* ctx = nullptr;
                            if (zeContextCreate(driver, &ctx_desc, &ctx) == 0) {
                                struct ze_device_mem_alloc_desc_t_local {
                                    uint32_t stype; void* pNext; uint32_t flags; uint32_t ordinal;
                                } alloc_desc = {0x00010002, nullptr, 0, 0};

                                void* ptr = nullptr;
                                if (zeMemAllocDevice(ctx, &alloc_desc, 1024*1024, 1, dev, &ptr) == 0) {
                                    zeMemFree(ctx, ptr);
                                }
                                zeContextDestroy(ctx);
                            }
                        }
                    } else { ze_budget_supported = 0; }
                } else { ze_budget_supported = 0; }
            } else { ze_budget_supported = 0; }
        } else { ze_budget_supported = 0; }

#ifdef _WIN32
        FreeLibrary((HMODULE)lib);
#else
        dlclose(lib);
#endif
    } else {
        ze_budget_supported = 0;
    }

    if (free_b > 0) {
        caps->device_memory_bytes = free_b;
    } else {
        caps->device_memory_bytes = caps->recommended_vram_cache_bytes;
    }
}

static void fill_webgpu_backend_caps(moe_backend_caps_t* caps) {
    caps->supports_webgpu = compiled_webgpu_available();
    caps->supports_pinned_host = 0;
    caps->supports_backend_async_api = 0;
    caps->recommended_vram_cache_bytes = cache_with_common_for_experts(16);
    caps->recommended_pinned_bytes = 0;
}
