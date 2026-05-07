        }
        return -1;

        // BUGFIX 944: Replace first-match stub with actual LRU eviction ★★★ CRITICAL
        // Problem: Original stub picked first non-pinned block → random eviction order
        //   → hot KV blocks evicted before cold → VRAM KV hit rate ~random
        // Solution: Min-scan over last_access_epoch to find coldest block
        // Impact: Correct LRU eviction for long-context VRAM KV retention
        // Find coldest non-pinned block with tokens (lowest last_access_epoch)
        int evict_candidate = -1;
        uint64_t min_score = UINT64_MAX;

        for (uint32_t i = 0; i < max_blocks; ++i) {
            if (blocks[i].pinned) {
                continue;
            }
            if (is_free[i]) {
                continue;
            }
            if (blocks[i].num_tokens > 0) {
                // Score: prioritize low access_epoch, break ties by low access_count
                const uint64_t score = blocks[i].last_access_epoch;
                if (score < min_score) {
                    min_score = score;
                    evict_candidate = i;
                }
            }
        }

        if (evict_candidate >= 0) {
            if (!is_free[evict_candidate]) {
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
static void device_atomic_sub_saturating(std::atomic<uint64_t>& counter, uint64_t bytes);

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


// BUGFIX ?얜챷??: KV block VRAM promotion/eviction ??몄겲??// Promotes hot KV blocks to VRAM for faster attention computation
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
    if (layers == 0) {
        return 0;
    }
    const uint64_t data_size = (uint64_t)manager->block_size * layers * 512;
    const uint64_t scales_size = (uint64_t)manager->block_size * layers * sizeof(float);
    const uint64_t zeros_size = (uint64_t)manager->block_size * layers * sizeof(float);
    if (data_size > UINT64_MAX / 2 || scales_size > UINT64_MAX / 2 || zeros_size > UINT64_MAX / 2) {
        return 0;
    }
    const uint64_t total_size = data_size + scales_size + zeros_size;
    if (total_size < data_size || total_size < scales_size || total_size < zeros_size) {
        return 0;
    }

    // Check VRAM KV budget
    const uint64_t kv_budget = engine->vram_kv_budget_bytes.load(std::memory_order_relaxed);
    const uint64_t kv_used = engine->vram_kv_bytes.load(std::memory_order_relaxed);

    if (kv_budget == 0 || total_size > kv_budget) {
        return 0;  // No KV budget or block too large
    }

    // Try to allocate VRAM budget
    uint64_t current_used = kv_used;
    if (current_used > UINT64_MAX - total_size) {
        return 0;
    }
    while (current_used + total_size <= kv_budget) {
        if (engine->vram_kv_bytes.compare_exchange_weak(
                current_used, current_used + total_size,
                std::memory_order_acq_rel, std::memory_order_acquire)) {
            break;
        }
        // BUGFIX 754: Re-check overflow after CAS failure ?끸쁾
        if (current_used > UINT64_MAX - total_size) {
            return 0;
        }
    }

    if (current_used + total_size > kv_budget) {
        // Budget exceeded - evict cold blocks first
        const uint32_t evicted = evict_cold_kv_blocks(engine, manager, total_size);
        if (evicted == 0) {
            return 0;  // Could not free enough space
        }

        // Retry budget allocation after eviction
        current_used = engine->vram_kv_bytes.load(std::memory_order_relaxed);
        if (current_used + total_size > kv_budget) {
            return 0;
        }
        if (!engine->vram_kv_bytes.compare_exchange_strong(
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
        if (data_size > 10ull * 1024 * 1024 * 1024 || data_size > SIZE_MAX) {
            device_atomic_sub_saturating(engine->vram_kv_bytes, total_size);
            return 0;
        }
        if (engine->backend_caps.backend == moe_BACKEND_HIP && engine->hip.hipMalloc) {
            alloc_ok = (engine->hip.hipMalloc(&vram_data_ptr, (size_t)data_size) == 0) ? 1 : 0;
        } else if (engine->backend_caps.backend == moe_BACKEND_CUDA && engine->cuda.cuMemAlloc) {
            alloc_ok = (engine->cuda.cuMemAlloc(&vram_data_ptr, (size_t)data_size) == 0) ? 1 : 0;
        }
        if (!alloc_ok) {
            device_atomic_sub_saturating(engine->vram_kv_bytes, total_size);
            return 0;
        }
    }

    // Allocate scales buffer
    if (scales_size > 0) {
        if (scales_size > 1ull * 1024 * 1024 * 1024 || scales_size > SIZE_MAX) {
            if (vram_data_ptr) {
                if (engine->backend_caps.backend == moe_BACKEND_HIP) engine->hip.hipFree(vram_data_ptr);
                else if (engine->backend_caps.backend == moe_BACKEND_CUDA) engine->cuda.cuMemFree(vram_data_ptr);
            }
            device_atomic_sub_saturating(engine->vram_kv_bytes, total_size);
            return 0;
        }
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
            device_atomic_sub_saturating(engine->vram_kv_bytes, total_size);
            return 0;
        }
    }

    // Allocate zeros buffer
    if (zeros_size > 0) {
        if (zeros_size > 1ull * 1024 * 1024 * 1024 || zeros_size > SIZE_MAX) {
            if (vram_data_ptr) {
                if (engine->backend_caps.backend == moe_BACKEND_HIP) engine->hip.hipFree(vram_data_ptr);
                else if (engine->backend_caps.backend == moe_BACKEND_CUDA) engine->cuda.cuMemFree(vram_data_ptr);
            }
            if (vram_scales_ptr) {
                if (engine->backend_caps.backend == moe_BACKEND_HIP) engine->hip.hipFree(vram_scales_ptr);
                else if (engine->backend_caps.backend == moe_BACKEND_CUDA) engine->cuda.cuMemFree(vram_scales_ptr);
            }
            device_atomic_sub_saturating(engine->vram_kv_bytes, total_size);
            return 0;
        }
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
            device_atomic_sub_saturating(engine->vram_kv_bytes, total_size);
            return 0;
        }
    }

    // Keep device accounting ownership aligned after the pending reservation becomes a real allocation.
    // ??Error paths before sub() don't restore VRAM budget ??budget leak
    // Solution: Initialize to 0, set to 1 after sub() call
    // Impact: Correct budget restoration on allocation failure
    int device_accounted = 0;
    int res = 0;

    // BUGFIX 945: Use async H2D copy for KV block promotion ★★★ PERFORMANCE
    // Problem: cuMemcpyHtoD/hipMemcpyHtoD blocks generation thread during copy
    //   → long-context prefill stalls for several ms per KV block
    // Solution: Use device_backend_copy_h2d_async via engine's GPU worker stream
    //   with sync fallback for backends without async support
    // Impact: Eliminates generation thread blocking during KV VRAM promotion
    if (host_data && vram_data_ptr) {
        if (data_size > SIZE_MAX) {
            moe_free_kv_vram_allocations(engine, vram_data_ptr, vram_scales_ptr, vram_zeros_ptr, device_accounted ? total_size : 0);
            device_atomic_sub_saturating(engine->vram_kv_bytes, total_size);
            return 0;
        }
        // Try async path first, fall back to sync
        CUstream stream = device_tensor_stream_current(engine);
        if (stream && device_backend_async_h2d_available(engine, stream, 1)) {
            res = device_backend_copy_h2d_async(engine, vram_data_ptr, host_data, data_size, stream) ? 0 : -1;
            if (res == 0) {
                // Sync to ensure data is available before attention uses it
                device_backend_stream_sync(engine, stream);
            }
        } else {
            // Sync fallback
            if (engine->backend_caps.backend == moe_BACKEND_HIP) {
                res = engine->hip.hipMemcpyHtoD(vram_data_ptr, host_data, data_size);
            } else if (engine->backend_caps.backend == moe_BACKEND_CUDA) {
                res = engine->cuda.cuMemcpyHtoD(vram_data_ptr, host_data, data_size);
            }
        }
        if (res != 0) {
            moe_free_kv_vram_allocations(engine, vram_data_ptr, vram_scales_ptr, vram_zeros_ptr, device_accounted ? total_size : 0);
            device_atomic_sub_saturating(engine->vram_kv_bytes, total_size);
            return 0;
        }
    }

    device_account_free_bytes_sub(engine, total_size);
    device_accounted = 1;
