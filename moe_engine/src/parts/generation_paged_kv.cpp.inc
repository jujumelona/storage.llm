
// ============================================================================
// PagedAttention KV Cache Manager (vLLM Style Continuous Batching)
// ============================================================================
// Implements a page-based memory allocator for KV blocks to eliminate memory
// fragmentation and allow dynamic request scheduling without pre-allocation.

#include <vector>
#include <mutex>
#include <cstdint>

#ifndef MOE_GENERATION_PAGED_KV_TYPES_DEFINED
#define MOE_GENERATION_PAGED_KV_TYPES_DEFINED

struct moe_paged_kv_block_t {
    uint32_t block_id;
    uint32_t num_tokens;
    // Data layout: [layer][token_in_block][dim]
    uint8_t* compressed_data; // INT8
    float* scales;
    float* zeros;
    // BUGFIX 128: Protect sink tokens from eviction
    bool pinned; // true if this block contains sink tokens (initial tokens that must be preserved)

    // BUGFIX 문제4: Add VRAM tier support for hot KV blocks ★★★
    moe_tier_t tier = moe_TIER_RAM;  // RAM, VRAM, or PINNED
    void* vram_data = nullptr;       // VRAM allocation if tier == VRAM
    void* vram_scales = nullptr;
    void* vram_zeros = nullptr;
    uint64_t last_access_epoch = 0;  // For LRU eviction
    uint32_t access_count = 0;       // For frequency tracking
};

struct moe_paged_kv_manager_t {
    uint32_t block_size; // e.g., 16 tokens per block
    uint32_t max_blocks;
    uint32_t sink_token_count; // BUGFIX 128: Number of initial tokens to preserve (typically 4-8)
    std::vector<uint32_t> free_blocks;
    std::vector<bool> is_free;  // BUGFIX 문제7: O(1) duplicate check bitmap
    std::vector<moe_paged_kv_block_t> blocks;
    std::mutex mtx;

    void init(uint32_t capacity_tokens, uint32_t b_size, uint32_t layers, uint32_t sink_tokens = 4) {
        // BUGFIX 514: Check b_size > 0 to prevent division by zero ★
        // Problem: max_blocks = capacity_tokens / block_size crashes if b_size == 0
        // Solution: Check b_size > 0 before division
        if (b_size == 0) {
            return;
        }
        // BUGFIX 515: Check layers upper bound to prevent overflow ★
        // Problem: block_size * layers * 512 can overflow in malloc
        // Solution: Add reasonable upper bound check (e.g., 1000 layers)
        if (layers > 1000) {
            return;
        }
        block_size = b_size;
        max_blocks = capacity_tokens / block_size;
        sink_token_count = sink_tokens; // BUGFIX 128: Store sink token count
        blocks.resize(max_blocks);
        is_free.resize(max_blocks, false);  // BUGFIX 문제7: Initialize bitmap
        free_blocks.reserve(max_blocks);

        // BUGFIX 128: Calculate how many blocks contain sink tokens
        const uint32_t sink_blocks = (sink_token_count + block_size - 1) / block_size;

        for (uint32_t i = 0; i < max_blocks; ++i) {
            blocks[i].block_id = i;
            blocks[i].num_tokens = 0;
            // BUGFIX 128: Mark first N blocks as pinned (contain sink tokens)
            blocks[i].pinned = (i < sink_blocks);

            // BUGFIX 171: Check malloc failure before using allocated memory
            blocks[i].compressed_data = (uint8_t*)std::malloc(block_size * layers * 512);
            blocks[i].scales = (float*)std::malloc(block_size * layers * sizeof(float));
            blocks[i].zeros = (float*)std::malloc(block_size * layers * sizeof(float));

            // Check all allocations succeeded
            if (!blocks[i].compressed_data || !blocks[i].scales || !blocks[i].zeros) {
                // Cleanup partial allocations
                if (blocks[i].compressed_data) std::free(blocks[i].compressed_data);
                if (blocks[i].scales) std::free(blocks[i].scales);
                if (blocks[i].zeros) std::free(blocks[i].zeros);

                // Mark block as invalid
                blocks[i].compressed_data = nullptr;
                blocks[i].scales = nullptr;
                blocks[i].zeros = nullptr;
                blocks[i].pinned = false;

                // Don't add to free_blocks - this block is unusable
                continue;
            }

            free_blocks.push_back(i);
            is_free[i] = true;  // BUGFIX 문제7: Mark as free in bitmap
        }
    }

    // BUGFIX 228: Add cleanup method to free allocated memory ★★
    // Problem: malloc'd memory in init() never freed → memory leak on manager destruction
    // Solution: Add cleanup() method to free all allocated blocks
    // Impact: Prevents memory leak (block_size * layers * 512 + 2*sizeof(float)) per block
    void cleanup() {
        for (auto& block : blocks) {
            if (block.compressed_data) {
                std::free(block.compressed_data);
                block.compressed_data = nullptr;
            }
            if (block.scales) {
                std::free(block.scales);
                block.scales = nullptr;
            }
            if (block.zeros) {
                std::free(block.zeros);
                block.zeros = nullptr;
            }
        }
        blocks.clear();
        free_blocks.clear();
        is_free.clear();  // BUGFIX 문제7: Clear bitmap
    }

    ~moe_paged_kv_manager_t() {
        cleanup();
    }

    int allocate_block() {
        std::lock_guard<std::mutex> lock(mtx);
        if (free_blocks.empty()) return -1;
        uint32_t id = free_blocks.back();
        free_blocks.pop_back();
        // BUGFIX 300: Check id bounds before accessing blocks array ★
        // Problem: free_blocks can contain invalid id due to corruption
        // Accessing blocks[id] with invalid id causes out-of-bounds access
        // Solution: Validate id before use
        if (id >= max_blocks) {
            return -1;
        }
        blocks[id].num_tokens = 0;
        is_free[id] = false;  // BUGFIX 문제7: Mark as allocated
        return id;
    }

    void free_block(uint32_t id) {
        std::lock_guard<std::mutex> lock(mtx);
        // BUGFIX 301 + 문제7: Prevent double-free with O(1) bitmap check ★★
        // Problem: Caller can call free_block(id) twice
        // Old solution: O(n) linear search through free_blocks
        // New solution: O(1) bitmap check
        if (id >= max_blocks) return;

        // BUGFIX 문제7: O(1) duplicate check with bitmap
        if (is_free[id]) {
            return;  // Already free
        }

        is_free[id] = true;
        free_blocks.push_back(id);
    }

    // BUGFIX 128: Evict oldest non-pinned block when cache is full
    // Returns -1 if no evictable blocks available (all pinned or in use)
    int evict_lru_block() {
        std::lock_guard<std::mutex> lock(mtx);

        // Find oldest non-pinned block with tokens
        int evict_candidate = -1;
        uint32_t min_access_time = UINT32_MAX; // In real implementation, track access time per block

        for (uint32_t i = 0; i < max_blocks; ++i) {
            // BUGFIX 128: Skip pinned blocks (contain sink tokens)
            if (blocks[i].pinned) {
                continue;
            }

            // BUGFIX 문제7: O(1) check with bitmap instead of O(n) loop
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
            // BUGFIX 302 + 문제7: O(1) check with bitmap ★
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

#endif // MOE_GENERATION_PAGED_KV_TYPES_DEFINED

#ifndef MOE_GENERATION_PAGED_KV_TYPES_ONLY

static uint32_t evict_cold_kv_blocks(
    moe_pc_engine* engine,
    moe_paged_kv_manager_t* manager,
    uint64_t bytes_needed
);

static uint32_t moe_paged_kv_layer_count(const moe_pc_engine* engine) {
    const moe_model_shape_t shape = moe_engine_model_shape_or_static(engine);
    return shape.num_hidden_layers ? shape.num_hidden_layers : 80u;
}


// BUGFIX 문제4: KV block VRAM promotion/eviction ★★★
// Promotes hot KV blocks to VRAM for faster attention computation
// Uses LRU + frequency tracking similar to expert residency

static int promote_kv_block_to_vram(
    moe_pc_engine* engine,
    moe_paged_kv_manager_t* manager,
    uint32_t block_id
) {
    if (!engine || !manager || block_id >= manager->max_blocks) {
        return 0;
    }

    moe_paged_kv_block_t& block = manager->blocks[block_id];

    // Already in VRAM
    if (block.tier == moe_TIER_VRAM && block.vram_data) {
        return 1;
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

    // Copy data to VRAM
    int res = 0;
    if (block.compressed_data && vram_data_ptr) {
        if (engine->backend_caps.backend == moe_BACKEND_HIP) {
            res = engine->hip.hipMemcpyHtoD(vram_data_ptr, block.compressed_data, data_size);
        } else if (engine->backend_caps.backend == moe_BACKEND_CUDA) {
            res = engine->cuda.cuMemcpyHtoD(vram_data_ptr, block.compressed_data, data_size);
        }
        if (res != 0) {
            if (engine->backend_caps.backend == moe_BACKEND_HIP) {
                engine->hip.hipFree(vram_data_ptr);
                if (vram_scales_ptr) engine->hip.hipFree(vram_scales_ptr);
                if (vram_zeros_ptr) engine->hip.hipFree(vram_zeros_ptr);
            } else if (engine->backend_caps.backend == moe_BACKEND_CUDA) {
                engine->cuda.cuMemFree(vram_data_ptr);
                if (vram_scales_ptr) engine->cuda.cuMemFree(vram_scales_ptr);
                if (vram_zeros_ptr) engine->cuda.cuMemFree(vram_zeros_ptr);
            }
            engine->vram_kv_used_bytes.fetch_sub(total_size, std::memory_order_relaxed);
            return 0;
        }
    }

    if (block.scales && vram_scales_ptr) {
        if (engine->backend_caps.backend == moe_BACKEND_HIP) {
            res = engine->hip.hipMemcpyHtoD(vram_scales_ptr, block.scales, scales_size);
        } else if (engine->backend_caps.backend == moe_BACKEND_CUDA) {
            res = engine->cuda.cuMemcpyHtoD(vram_scales_ptr, block.scales, scales_size);
        }
        if (res != 0) {
            if (engine->backend_caps.backend == moe_BACKEND_HIP) {
                engine->hip.hipFree(vram_data_ptr);
                engine->hip.hipFree(vram_scales_ptr);
                if (vram_zeros_ptr) engine->hip.hipFree(vram_zeros_ptr);
            } else if (engine->backend_caps.backend == moe_BACKEND_CUDA) {
                engine->cuda.cuMemFree(vram_data_ptr);
                engine->cuda.cuMemFree(vram_scales_ptr);
                if (vram_zeros_ptr) engine->cuda.cuMemFree(vram_zeros_ptr);
            }
            engine->vram_kv_used_bytes.fetch_sub(total_size, std::memory_order_relaxed);
            return 0;
        }
    }

    if (block.zeros && vram_zeros_ptr) {
        if (engine->backend_caps.backend == moe_BACKEND_HIP) {
            res = engine->hip.hipMemcpyHtoD(vram_zeros_ptr, block.zeros, zeros_size);
        } else if (engine->backend_caps.backend == moe_BACKEND_CUDA) {
            res = engine->cuda.cuMemcpyHtoD(vram_zeros_ptr, block.zeros, zeros_size);
        }
        if (res != 0) {
            if (engine->backend_caps.backend == moe_BACKEND_HIP) {
                engine->hip.hipFree(vram_data_ptr);
                engine->hip.hipFree(vram_scales_ptr);
                engine->hip.hipFree(vram_zeros_ptr);
            } else if (engine->backend_caps.backend == moe_BACKEND_CUDA) {
                engine->cuda.cuMemFree(vram_data_ptr);
                engine->cuda.cuMemFree(vram_scales_ptr);
                engine->cuda.cuMemFree(vram_zeros_ptr);
            }
            engine->vram_kv_used_bytes.fetch_sub(total_size, std::memory_order_relaxed);
            return 0;
        }
    }

    // Update block state
    block.vram_data = (void*)vram_data_ptr;
    block.vram_scales = (void*)vram_scales_ptr;
    block.vram_zeros = (void*)vram_zeros_ptr;
    block.tier = moe_TIER_VRAM;

    // Update device pool counter
    engine->device_kv_cache_bytes.fetch_add(total_size, std::memory_order_relaxed);

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
    block->last_access_epoch = now;
    block->access_count = (block->access_count < UINT32_MAX) ?
        block->access_count + 1 : UINT32_MAX;
}

// BUGFIX 문제4: Evict cold KV blocks from VRAM to RAM ★★★
// Uses LRU + frequency scoring similar to expert eviction
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
    const uint64_t block_size = (uint64_t)manager->block_size * layers * (512 + 2 * sizeof(float));

    std::vector<std::pair<uint32_t, float>> candidates;
    candidates.reserve(manager->max_blocks);

    // Collect VRAM blocks with scores
    for (uint32_t i = 0; i < manager->max_blocks; ++i) {
        moe_paged_kv_block_t& block = manager->blocks[i];

        if (block.tier != moe_TIER_VRAM || !block.vram_data) {
            continue;
        }

        // Skip pinned blocks (sink tokens)
        if (block.pinned) {
            continue;
        }

        // Calculate eviction score (lower = evict first)
        const uint64_t age = now > block.last_access_epoch ? now - block.last_access_epoch : 0;
        const float frequency = (float)block.access_count;
        const float score = frequency / (1.0f + (float)age * 0.001f);

        candidates.push_back({i, score});
    }

    // Sort by score (ascending - coldest first)
    std::sort(candidates.begin(), candidates.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });

    uint64_t freed = 0;
    uint32_t evicted_count = 0;

    for (const auto& [block_id, score] : candidates) {
        if (freed >= bytes_needed) {
            break;
        }

        moe_paged_kv_block_t& block = manager->blocks[block_id];

        // Free VRAM allocations
        if (block.vram_data) {
            if (engine->backend_caps.backend == moe_BACKEND_HIP) {
                engine->hip.hipFree((CUdeviceptr)block.vram_data);
            } else if (engine->backend_caps.backend == moe_BACKEND_CUDA) {
                engine->cuda.cuMemFree((CUdeviceptr)block.vram_data);
            }
            block.vram_data = nullptr;
        }
        if (block.vram_scales) {
            if (engine->backend_caps.backend == moe_BACKEND_HIP) {
                engine->hip.hipFree((CUdeviceptr)block.vram_scales);
            } else if (engine->backend_caps.backend == moe_BACKEND_CUDA) {
                engine->cuda.cuMemFree((CUdeviceptr)block.vram_scales);
            }
            block.vram_scales = nullptr;
        }
        if (block.vram_zeros) {
            if (engine->backend_caps.backend == moe_BACKEND_HIP) {
                engine->hip.hipFree((CUdeviceptr)block.vram_zeros);
            } else if (engine->backend_caps.backend == moe_BACKEND_CUDA) {
                engine->cuda.cuMemFree((CUdeviceptr)block.vram_zeros);
            }
            block.vram_zeros = nullptr;
        }

        // Demote to RAM
        block.tier = moe_TIER_RAM;

        freed += block_size;
        evicted_count++;

        // Update budgets
        engine->vram_kv_used_bytes.fetch_sub(block_size, std::memory_order_relaxed);
        engine->device_kv_cache_bytes.fetch_sub(block_size, std::memory_order_relaxed);
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
    const size_t data_size = (size_t)manager->block_size * layers * 512;
    const size_t scales_size = (size_t)manager->block_size * layers * sizeof(float);
    const size_t zeros_size = (size_t)manager->block_size * layers * sizeof(float);

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
