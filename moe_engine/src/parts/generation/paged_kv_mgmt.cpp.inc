        }
        if (res != 0) {
            moe_free_kv_vram_allocations(engine, vram_data_ptr, vram_scales_ptr, vram_zeros_ptr,
                device_accounted ? total_size : 0);
            device_atomic_sub_saturating(engine->vram_kv_bytes, total_size);
            return 0;
        }
    }

    if (host_scales && vram_scales_ptr) {
        if (engine->backend_caps.backend == moe_BACKEND_HIP) {
            res = engine->hip.hipMemcpyHtoD(vram_scales_ptr, host_scales, scales_size);
        } else if (engine->backend_caps.backend == moe_BACKEND_CUDA) {
            res = engine->cuda.cuMemcpyHtoD(vram_scales_ptr, host_scales, scales_size);
        }
        if (res != 0) {
            moe_free_kv_vram_allocations(engine, vram_data_ptr, vram_scales_ptr, vram_zeros_ptr,
                device_accounted ? total_size : 0);
            device_atomic_sub_saturating(engine->vram_kv_bytes, total_size);
            return 0;
        }
    }

    if (host_zeros && vram_zeros_ptr) {
        if (engine->backend_caps.backend == moe_BACKEND_HIP) {
            res = engine->hip.hipMemcpyHtoD(vram_zeros_ptr, host_zeros, zeros_size);
        } else if (engine->backend_caps.backend == moe_BACKEND_CUDA) {
            res = engine->cuda.cuMemcpyHtoD(vram_zeros_ptr, host_zeros, zeros_size);
        }
        if (res != 0) {
            moe_free_kv_vram_allocations(engine, vram_data_ptr, vram_scales_ptr, vram_zeros_ptr,
                device_accounted ? total_size : 0);
            device_atomic_sub_saturating(engine->vram_kv_bytes, total_size);
            return 0;
        }
    }

    // Update block state
    {
        std::lock_guard<std::mutex> lock(manager->mtx);
        if (block_id >= manager->max_blocks || manager->is_free[block_id]) {
            moe_free_kv_vram_allocations(engine, vram_data_ptr, vram_scales_ptr, vram_zeros_ptr,
                device_accounted ? total_size : 0);
            device_atomic_sub_saturating(engine->vram_kv_bytes, total_size);
            return 0;
        }
        moe_paged_kv_block_t& block = manager->blocks[block_id];
        if (block.tier == moe_TIER_VRAM && block.vram_data) {
            moe_free_kv_vram_allocations(engine, vram_data_ptr, vram_scales_ptr, vram_zeros_ptr,
                device_accounted ? total_size : 0);
            device_atomic_sub_saturating(engine->vram_kv_bytes, total_size);
            return 1;
        }
        if (block.compressed_data != host_data ||
            block.scales != host_scales ||
            block.zeros != host_zeros) {
            moe_free_kv_vram_allocations(engine, vram_data_ptr, vram_scales_ptr, vram_zeros_ptr,
                device_accounted ? total_size : 0);
            device_atomic_sub_saturating(engine->vram_kv_bytes, total_size);
            return 0;
        }
        block.vram_data = (void*)vram_data_ptr;
        block.vram_scales = (void*)vram_scales_ptr;
        block.vram_zeros = (void*)vram_zeros_ptr;
        block.tier = moe_TIER_VRAM;
        block.tier = moe_TIER_VRAM;
    }

    engine->device_kv_cache_bytes.fetch_add(total_size, std::memory_order_relaxed);
    engine->kv_actually_in_vram.store(true, std::memory_order_release);

    return 1;
}

static void touch_kv_block_heat(
    moe_pc_engine* engine,
    moe_paged_kv_block_t* block
) {
    if (!engine || !block) {
        return;
    }

    const uint64_t now = engine->expert_touch_epoch.fetch_add(1, std::memory_order_relaxed) + 1;
    // BUGFIX 827: Overflow check for epoch
    if (now == 0) return;
    block->last_access_epoch = now;
    block->access_count = (block->access_count < UINT32_MAX) ?
        block->access_count + 1 : UINT32_MAX;
}

// BUGFIX 臾몄젣4: Evict cold KV blocks from VRAM to RAM ?끸쁾??// Uses LRU + frequency scoring similar to expert eviction
// BUGFIX 臾몄젣8: Added manager->mtx locking to prevent UAF ?끸쁾??// Without locking, concurrent attention threads can read block.vram_data
// after cuMemFree completes but before nullptr assignment ??Use-After-Free.
// Strategy: collect candidates under lock ??clear ptrs under lock ??free VRAM outside lock.
static uint32_t evict_cold_kv_blocks(
    moe_pc_engine* engine,
    moe_paged_kv_manager_t* manager,
    uint64_t bytes_needed
) {
    if (!engine || !manager || bytes_needed == 0) {
        return 0;
    }

    const uint64_t now = engine->expert_touch_epoch.load(std::memory_order_relaxed);
    const uint32_t layers = moe_paged_kv_layer_count(engine);
    // BUGFIX 828: Layers validation
    if (layers == 0 || layers > 1000) return 0;
    // BUGFIX 829: Block size validation
    if (manager->block_size == 0 || manager->block_size > 1000000) return 0;

    const uint64_t block_size = (uint64_t)manager->block_size * layers * (512 + 2 * sizeof(float));
    // BUGFIX 830: Block size overflow check
    if (block_size == 0 || block_size > UINT64_MAX / 2) return 0;

    // Phase 1: Collect eviction candidates and clear block pointers under lock.
    // Attention threads check block.tier/vram_data under the same mtx,
    // so clearing here prevents them from seeing freed pointers.
    struct evict_victim_t {
        uint32_t block_id;
        void* vram_data;
        void* vram_scales;
        void* vram_zeros;
    };
    std::vector<evict_victim_t> victims;

    {
        std::lock_guard<std::mutex> lock(manager->mtx);
        std::vector<std::pair<uint32_t, float>> candidates;
        candidates.reserve(manager->max_blocks);

        for (uint32_t i = 0; i < manager->max_blocks; ++i) {
            moe_paged_kv_block_t& block = manager->blocks[i];
            if (block.tier != moe_TIER_VRAM || !block.vram_data) continue;
            if (block.pinned) continue;

            const uint64_t age = now > block.last_access_epoch ? now - block.last_access_epoch : 0;
            const float frequency = (float)block.access_count;
            const float score = frequency / (1.0f + (float)age * 0.001f);
            // BUGFIX 831: Score NaN check
            if (!std::isfinite(score)) continue;
            candidates.push_back({i, score});
        }

        std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });

        uint64_t planned = 0;
        for (const auto& [block_id, score] : candidates) {
            if (planned >= bytes_needed) break;
            moe_paged_kv_block_t& block = manager->blocks[block_id];
            evict_victim_t v{};
            v.block_id = block_id;
            v.vram_data = block.vram_data;
            v.vram_scales = block.vram_scales;
            v.vram_zeros = block.vram_zeros;
            // Atomically clear under lock so attention sees nullptr
            block.vram_data = nullptr;
            block.vram_scales = nullptr;
            block.vram_zeros = nullptr;
            block.tier = moe_TIER_RAM;
            victims.push_back(v);
            planned += block_size;
        }
    }

    // Phase 2: Free VRAM outside lock (cuMemFree can block)
    uint64_t freed = 0;
    uint32_t evicted_count = 0;

    for (const auto& v : victims) {
        if (v.vram_data) {
            if (engine->backend_caps.backend == moe_BACKEND_HIP)
                engine->hip.hipFree((CUdeviceptr)v.vram_data);
            else if (engine->backend_caps.backend == moe_BACKEND_CUDA)
                engine->cuda.cuMemFree((CUdeviceptr)v.vram_data);
        }
        if (v.vram_scales) {
            if (engine->backend_caps.backend == moe_BACKEND_HIP)
                engine->hip.hipFree((CUdeviceptr)v.vram_scales);
            else if (engine->backend_caps.backend == moe_BACKEND_CUDA)
                engine->cuda.cuMemFree((CUdeviceptr)v.vram_scales);
        }
        if (v.vram_zeros) {
            if (engine->backend_caps.backend == moe_BACKEND_HIP)
                engine->hip.hipFree((CUdeviceptr)v.vram_zeros);
            else if (engine->backend_caps.backend == moe_BACKEND_CUDA)
                engine->cuda.cuMemFree((CUdeviceptr)v.vram_zeros);
        }
        device_account_bytes_freed(engine, block_size);
        freed += block_size;
        evicted_count++;
        device_atomic_sub_saturating(engine->device_kv_cache_bytes, block_size);
        device_atomic_sub_saturating(engine->vram_kv_bytes, block_size);
        if (engine->vram_kv_bytes.load(std::memory_order_acquire) == 0) {
            engine->kv_actually_in_vram.store(false, std::memory_order_release);
        }
    }

    if (evicted_count > 0) {
        fprintf(stderr, "[storageLLM INFO] Evicted %u cold KV blocks from VRAM, freed %llu MB\n",
            evicted_count, (unsigned long long)(freed / (1024*1024)));


    }

    return evicted_count;
}

static int enable_kv_pinned_memory(
    moe_pc_engine* engine,
    moe_paged_kv_manager_t* manager
) {
    if (!engine || !manager || !engine->backend_caps.supports_pinned_host) {
        return 0;
    }

    // Reallocate KV blocks with pinned memory for faster PCIe transfer
    // This uses mlock() on Linux or VirtualLock() on Windows

    uint32_t pinned_count = 0;
    const uint32_t layers = moe_paged_kv_layer_count(engine);
    // BUGFIX 832: Layers validation
    if (layers == 0 || layers > 1000) return 0;
    // BUGFIX 833: Block size validation
    if (manager->block_size == 0 || manager->block_size > 1000000) return 0;

    const size_t data_size = (size_t)manager->block_size * layers * 512;
    const size_t scales_size = (size_t)manager->block_size * layers * sizeof(float);
    const size_t zeros_size = (size_t)manager->block_size * layers * sizeof(float);

    // BUGFIX 834: Size overflow checks
    if (data_size == 0 || data_size > SIZE_MAX / 2) return 0;
    if (scales_size == 0 || scales_size > SIZE_MAX / 2) return 0;
    if (zeros_size == 0 || zeros_size > SIZE_MAX / 2) return 0;

    for (uint32_t i = 0; i < manager->max_blocks; ++i) {
        moe_paged_kv_block_t& block = manager->blocks[i];

        // Only pin RAM-tier blocks
        if (block.tier != moe_TIER_RAM || !block.compressed_data) {
            continue;
        }

#ifdef _WIN32
        // Windows: Use VirtualLock
        if (VirtualLock(block.compressed_data, data_size) &&
            VirtualLock(block.scales, scales_size) &&
            VirtualLock(block.zeros, zeros_size)) {
            pinned_count++;
        }
#else
        // Linux: Use mlock
        if (mlock(block.compressed_data, data_size) == 0 &&
            mlock(block.scales, scales_size) == 0 &&
            mlock(block.zeros, zeros_size) == 0) {
            pinned_count++;
        }
#endif
    }

    if (pinned_count > 0) {
        fprintf(stderr, "[storageLLM INFO] Pinned %u KV blocks in RAM for faster PCIe transfer\n",
            pinned_count);
    }

    return pinned_count > 0 ? 1 : 0;
}

#endif // MOE_GENERATION_PAGED_KV_TYPES_ONLY
