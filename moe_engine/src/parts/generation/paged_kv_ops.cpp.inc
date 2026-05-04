        }
        return -1;

        // Find oldest non-pinned block with tokens
        int evict_candidate = -1;
        uint32_t min_access_time = UINT32_MAX; // In real implementation, track access time per block

        for (uint32_t i = 0; i < max_blocks; ++i) {
            // BUGFIX 128: Skip pinned blocks (contain sink tokens)
            if (blocks[i].pinned) {
                continue;
            }

            // BUGFIX 臾몄젣7: O(1) check with bitmap instead of O(n) loop
            if (is_free[i]) {
                continue;
            }

            // In real implementation, use LRU tracking
            // For now, just pick first non-pinned, non-free block
            if (blocks[i].num_tokens > 0) {
                evict_candidate = i;
                break;
            }
        }

        if (evict_candidate >= 0) {
            // BUGFIX 302 + 臾몄젣7: O(1) check with bitmap ??            if (!is_free[evict_candidate]) {
                // Clear the block and add to free list
                blocks[evict_candidate].num_tokens = 0;
                is_free[evict_candidate] = true;
                free_blocks.push_back(evict_candidate);
            }
        }

        return evict_candidate;
    }
};

#else

struct moe_paged_kv_block_t {};

struct moe_paged_kv_manager_t {
    uint32_t block_size = 0;
    uint32_t max_blocks = 0;
    std::atomic<uint32_t> free_count{0};

    void init(uint32_t, uint32_t, uint32_t, uint32_t = 4) {}
    void cleanup() {}
    int allocate_block() { return -1; }
    void free_block(uint32_t) {}
    int evict_lru_block() { return -1; }
    uint32_t available_free_blocks() { return 0; }
};

#endif

#endif // MOE_GENERATION_PAGED_KV_TYPES_DEFINED

#if MOE_ENABLE_PAGED_KV && !defined(MOE_GENERATION_PAGED_KV_TYPES_ONLY)

static void device_account_bytes_freed(moe_pc_engine* e, uint64_t bytes);
static void device_account_free_bytes_sub(moe_pc_engine* e, uint64_t bytes);

static uint32_t evict_cold_kv_blocks(
    moe_pc_engine* engine,
    moe_paged_kv_manager_t* manager,
    uint64_t bytes_needed
);

static uint32_t moe_paged_kv_layer_count(const moe_pc_engine* engine) {
    const moe_model_shape_t shape = moe_engine_model_shape_or_static(engine);
    return shape.num_hidden_layers ? shape.num_hidden_layers : 80u;
}

static void moe_free_kv_vram_allocations(
    moe_pc_engine* engine,
    CUdeviceptr data,
    CUdeviceptr scales,
    CUdeviceptr zeros,
    uint64_t accounted_bytes
) {
    if (!engine) {
        return;
    }
    if (engine->backend_caps.backend == moe_BACKEND_HIP && engine->hip.hipFree) {
        if (data) engine->hip.hipFree(data);
        if (scales) engine->hip.hipFree(scales);
        if (zeros) engine->hip.hipFree(zeros);
    } else if (engine->backend_caps.backend == moe_BACKEND_CUDA && engine->cuda.cuMemFree) {
        if (data) engine->cuda.cuMemFree(data);
        if (scales) engine->cuda.cuMemFree(scales);
        if (zeros) engine->cuda.cuMemFree(zeros);
    }
    if (accounted_bytes) {
        device_account_bytes_freed(engine, accounted_bytes);
    }
}


// BUGFIX 臾몄젣4: KV block VRAM promotion/eviction ?끸쁾??// Promotes hot KV blocks to VRAM for faster attention computation
// Uses LRU + frequency tracking similar to expert residency

static int promote_kv_block_to_vram(
    moe_pc_engine* engine,
    moe_paged_kv_manager_t* manager,
    uint32_t block_id
) {
    if (!engine || !manager || block_id >= manager->max_blocks) {
        return 0;
    }
#if !defined(MOE_ENABLE_DEVICE_KV_ATTENTION) || !MOE_ENABLE_DEVICE_KV_ATTENTION
    return 0;
#endif

    uint8_t* host_data = nullptr;
    float* host_scales = nullptr;
    float* host_zeros = nullptr;
    {
        std::lock_guard<std::mutex> lock(manager->mtx);
        if (block_id >= manager->max_blocks || manager->is_free[block_id]) {
            return 0;
        }
        moe_paged_kv_block_t& block = manager->blocks[block_id];
        if (block.tier == moe_TIER_VRAM && block.vram_data) {
            return 1;
        }
        host_data = block.compressed_data;
        host_scales = block.scales;
        host_zeros = block.zeros;
    }

    // Calculate block size (assuming layers from engine config)
    const uint32_t layers = moe_paged_kv_layer_count(engine);
    const uint64_t data_size = (uint64_t)manager->block_size * layers * 512;
    const uint64_t scales_size = (uint64_t)manager->block_size * layers * sizeof(float);
    const uint64_t zeros_size = (uint64_t)manager->block_size * layers * sizeof(float);
    const uint64_t total_size = data_size + scales_size + zeros_size;

    // Check VRAM KV budget
    const uint64_t kv_budget = engine->vram_kv_budget_bytes.load(std::memory_order_relaxed);
    const uint64_t kv_used = engine->vram_kv_used_bytes.load(std::memory_order_relaxed);

    if (kv_budget == 0 || total_size > kv_budget) {
        return 0;  // No KV budget or block too large
    }

    // Try to allocate VRAM budget
    uint64_t current_used = kv_used;
    while (current_used + total_size <= kv_budget) {
        if (engine->vram_kv_used_bytes.compare_exchange_weak(
                current_used, current_used + total_size,
                std::memory_order_acq_rel, std::memory_order_acquire)) {
            break;
        }
    }

    if (current_used + total_size > kv_budget) {
        // Budget exceeded - evict cold blocks first
        const uint32_t evicted = evict_cold_kv_blocks(engine, manager, total_size);
        if (evicted == 0) {
            return 0;  // Could not free enough space
        }

        // Retry budget allocation after eviction
        current_used = engine->vram_kv_used_bytes.load(std::memory_order_relaxed);
        if (current_used + total_size > kv_budget) {
            return 0;
        }
        if (!engine->vram_kv_used_bytes.compare_exchange_strong(
                current_used, current_used + total_size,
                std::memory_order_acq_rel, std::memory_order_acquire)) {
            return 0;
        }
    }

    // Allocate VRAM using CUDA/HIP API directly
    CUdeviceptr vram_data_ptr = 0;
    CUdeviceptr vram_scales_ptr = 0;
    CUdeviceptr vram_zeros_ptr = 0;
    int alloc_ok = 0;

    // Allocate data buffer
    if (data_size > 0) {
        if (engine->backend_caps.backend == moe_BACKEND_HIP && engine->hip.hipMalloc) {
            alloc_ok = (engine->hip.hipMalloc(&vram_data_ptr, (size_t)data_size) == 0) ? 1 : 0;
        } else if (engine->backend_caps.backend == moe_BACKEND_CUDA && engine->cuda.cuMemAlloc) {
            alloc_ok = (engine->cuda.cuMemAlloc(&vram_data_ptr, (size_t)data_size) == 0) ? 1 : 0;
        }
        if (!alloc_ok) {
            engine->vram_kv_used_bytes.fetch_sub(total_size, std::memory_order_relaxed);
            return 0;
        }
    }

    // Allocate scales buffer
    if (scales_size > 0) {
        alloc_ok = 0;
        if (engine->backend_caps.backend == moe_BACKEND_HIP && engine->hip.hipMalloc) {
            alloc_ok = (engine->hip.hipMalloc(&vram_scales_ptr, (size_t)scales_size) == 0) ? 1 : 0;
        } else if (engine->backend_caps.backend == moe_BACKEND_CUDA && engine->cuda.cuMemAlloc) {
            alloc_ok = (engine->cuda.cuMemAlloc(&vram_scales_ptr, (size_t)scales_size) == 0) ? 1 : 0;
        }
        if (!alloc_ok) {
            if (vram_data_ptr) {
                if (engine->backend_caps.backend == moe_BACKEND_HIP) engine->hip.hipFree(vram_data_ptr);
                else if (engine->backend_caps.backend == moe_BACKEND_CUDA) engine->cuda.cuMemFree(vram_data_ptr);
            }
            engine->vram_kv_used_bytes.fetch_sub(total_size, std::memory_order_relaxed);
            return 0;
        }
    }

    // Allocate zeros buffer
    if (zeros_size > 0) {
        alloc_ok = 0;
        if (engine->backend_caps.backend == moe_BACKEND_HIP && engine->hip.hipMalloc) {
            alloc_ok = (engine->hip.hipMalloc(&vram_zeros_ptr, (size_t)zeros_size) == 0) ? 1 : 0;
        } else if (engine->backend_caps.backend == moe_BACKEND_CUDA && engine->cuda.cuMemAlloc) {
            alloc_ok = (engine->cuda.cuMemAlloc(&vram_zeros_ptr, (size_t)zeros_size) == 0) ? 1 : 0;
        }
        if (!alloc_ok) {
            if (vram_data_ptr) {
                if (engine->backend_caps.backend == moe_BACKEND_HIP) engine->hip.hipFree(vram_data_ptr);
                else if (engine->backend_caps.backend == moe_BACKEND_CUDA) engine->cuda.cuMemFree(vram_data_ptr);
            }
            if (vram_scales_ptr) {
                if (engine->backend_caps.backend == moe_BACKEND_HIP) engine->hip.hipFree(vram_scales_ptr);
                else if (engine->backend_caps.backend == moe_BACKEND_CUDA) engine->cuda.cuMemFree(vram_scales_ptr);
            }
            engine->vram_kv_used_bytes.fetch_sub(total_size, std::memory_order_relaxed);
            return 0;
        }
    }

    // BUGFIX Problem 14: Fix device_accounted initialization ★★★
    // Problem: device_accounted = 1 after device_account_free_bytes_sub
    // → Error paths before sub() don't restore VRAM budget → budget leak
    // Solution: Initialize to 0, set to 1 after sub() call
    // Impact: Correct budget restoration on allocation failure
    int device_accounted = 0;
    int res = 0;

    // Copy data to VRAM
    if (host_data && vram_data_ptr) {
        if (engine->backend_caps.backend == moe_BACKEND_HIP) {
            res = engine->hip.hipMemcpyHtoD(vram_data_ptr, host_data, data_size);
        } else if (engine->backend_caps.backend == moe_BACKEND_CUDA) {
            res = engine->cuda.cuMemcpyHtoD(vram_data_ptr, host_data, data_size);
        }
        if (res != 0) {
            moe_free_kv_vram_allocations(engine, vram_data_ptr, vram_scales_ptr, vram_zeros_ptr, device_accounted ? total_size : 0);
            engine->vram_kv_used_bytes.fetch_sub(total_size, std::memory_order_relaxed);
            return 0;
        }
    }

    device_account_free_bytes_sub(engine, total_size);
    device_accounted = 1;
