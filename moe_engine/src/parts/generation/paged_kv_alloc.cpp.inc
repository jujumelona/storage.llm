
// ============================================================================
// PagedAttention KV Cache Manager (vLLM Style Continuous Batching)
// ============================================================================
// Implements a page-based memory allocator for KV blocks to eliminate memory
// fragmentation and allow dynamic request scheduling without pre-allocation.

#include <vector>
#include <mutex>
#include <cstdint>
#include <atomic>
#include <functional>
#include <queue>

#ifndef MOE_ENABLE_PAGED_KV
#define MOE_ENABLE_PAGED_KV 0
#endif

// Fix 29: Define MOE_PAGED_KV_VRAM_TIER to enable VRAM tier for KV blocks
#define MOE_PAGED_KV_VRAM_TIER

#ifndef MOE_GENERATION_PAGED_KV_TYPES_DEFINED
#define MOE_GENERATION_PAGED_KV_TYPES_DEFINED

#if MOE_ENABLE_PAGED_KV

struct moe_paged_kv_block_t {
    uint32_t block_id;
    uint32_t num_tokens;
    // Data layout: [layer][token_in_block][dim]
    uint8_t* compressed_data; // INT8
    float* scales;
    float* zeros;
    // BUGFIX 128: Protect sink tokens from eviction
    bool pinned; // true if this block contains sink tokens (initial tokens that must be preserved)

    // BUGFIX 臾몄젣4: Add VRAM tier support for hot KV blocks ?끸쁾??    moe_tier_t tier = moe_TIER_RAM;  // RAM, VRAM, or PINNED
#ifdef MOE_PAGED_KV_VRAM_TIER
    // Fix 29: Guarded behind MOE_PAGED_KV_VRAM_TIER to save 36 bytes per block
    // when VRAM tier is not compiled in. With 4096 blocks, this saves 148KB
    // and improves cache-line density during eviction sweeps.
    void* vram_data = nullptr;       // VRAM allocation if tier == VRAM
    void* vram_scales = nullptr;
    void* vram_zeros = nullptr;
#endif
    uint64_t last_access_epoch = 0;  // For LRU eviction
    uint32_t access_count = 0;       // For frequency tracking
};

struct moe_paged_kv_manager_t {
    uint32_t block_size; // e.g., 16 tokens per block
    uint32_t max_blocks;
    uint32_t sink_token_count; // BUGFIX 128: Number of initial tokens to preserve (typically 4-8)
    std::vector<uint32_t> free_blocks;
    std::vector<bool> is_free;  // BUGFIX 臾몄젣7: O(1) duplicate check bitmap
    std::vector<moe_paged_kv_block_t> blocks;
    std::atomic<uint32_t> free_count{0};
    // Fix 25: Replaced std::priority_queue with vector + lazy deletion
    // priority_queue cannot remove arbitrary elements, causing stale entry buildup
    std::vector<std::pair<uint64_t, uint32_t>> lru_heap;
    bool lru_heap_compact_needed = false;
    std::mutex mtx;

    void init(uint32_t capacity_tokens, uint32_t b_size, uint32_t layers, uint32_t sink_tokens = 4) {
        // BUGFIX 514: Check b_size > 0 to prevent division by zero ??        // Problem: max_blocks = capacity_tokens / block_size crashes if b_size == 0
        // Solution: Check b_size > 0 before division
        if (b_size == 0) {
            return;
        }
        // BUGFIX 515: Check layers upper bound to prevent overflow ??        // Problem: block_size * layers * 512 can overflow in malloc
        // Solution: Add reasonable upper bound check (e.g., 1000 layers)
        if (layers > 1000) {
            return;
        }
        if (b_size > 65536u) {
            return;
        }
        const uint64_t data_bytes64 = (uint64_t)b_size * (uint64_t)layers * 512ull;
        const uint64_t aux_bytes64 = (uint64_t)b_size * (uint64_t)layers * (uint64_t)sizeof(float);
        if (data_bytes64 > (uint64_t)SIZE_MAX || aux_bytes64 > (uint64_t)SIZE_MAX) {
            return;
        }
        const size_t data_bytes = (size_t)data_bytes64;
        const size_t aux_bytes = (size_t)aux_bytes64;
        block_size = b_size;
        max_blocks = capacity_tokens / block_size;
        sink_token_count = sink_tokens; // BUGFIX 128: Store sink token count
        blocks.resize(max_blocks);
        is_free.resize(max_blocks, false);  // BUGFIX 臾몄젣7: Initialize bitmap
        free_blocks.reserve(max_blocks);
        free_count.store(0, std::memory_order_relaxed);
        lru_heap.clear(); lru_heap_compact_needed = false;

        // BUGFIX 128: Calculate how many blocks contain sink tokens
        const uint32_t sink_blocks = (sink_token_count + block_size - 1) / block_size;

        for (uint32_t i = 0; i < max_blocks; ++i) {
            blocks[i].block_id = i;
            blocks[i].num_tokens = 0;
            // BUGFIX 128: Mark first N blocks as pinned (contain sink tokens)
            blocks[i].pinned = (i < sink_blocks);

            // BUGFIX 171: Check malloc failure before using allocated memory
            blocks[i].compressed_data = (uint8_t*)std::malloc(data_bytes);
            blocks[i].scales = (float*)std::malloc(aux_bytes);
            blocks[i].zeros = (float*)std::malloc(aux_bytes);

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
            free_count.fetch_add(1, std::memory_order_relaxed);
            is_free[i] = true;  // BUGFIX 臾몄젣7: Mark as free in bitmap
        }
    }

    // BUGFIX 228: Add cleanup method to free allocated memory ?끸쁾
    // Problem: malloc'd memory in init() never freed ??memory leak on manager destruction
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
        free_count.store(0, std::memory_order_relaxed);
        lru_heap.clear(); lru_heap_compact_needed = false;
        is_free.clear();  // BUGFIX 臾몄젣7: Clear bitmap
    }

    ~moe_paged_kv_manager_t() {
        cleanup();
    }

    int allocate_block() {
        std::lock_guard<std::mutex> lock(mtx);
        while (!free_blocks.empty()) {
            uint32_t id = free_blocks.back();
            free_blocks.pop_back();
            if (id >= max_blocks || !is_free[id]) {
                continue;
            }
            blocks[id].num_tokens = 0;
            is_free[id] = false;
            uint32_t current = free_count.load(std::memory_order_relaxed);
            while (current > 0 &&
                   !free_count.compare_exchange_weak(
                       current, current - 1u,
                       std::memory_order_relaxed,
                       std::memory_order_relaxed)) {}
            lru_heap.push_back({blocks[id].last_access_epoch, id}); std::push_heap(lru_heap.begin(), lru_heap.end(), std::greater<std::pair<uint64_t, uint32_t>>());
            return id;
        }
        free_count.store(0, std::memory_order_relaxed);
        return -1;
    }

    void free_block(uint32_t id) {
        std::lock_guard<std::mutex> lock(mtx);
        // BUGFIX 301 + 臾몄젣7: Prevent double-free with O(1) bitmap check ?끸쁾
        // Problem: Caller can call free_block(id) twice
        // Old solution: O(n) linear search through free_blocks
        // New solution: O(1) bitmap check
        if (id >= max_blocks) return;

        // BUGFIX 臾몄젣7: O(1) duplicate check with bitmap
        if (is_free[id]) {
            return;  // Already free
        }

        is_free[id] = true;
        free_blocks.push_back(id);
        free_count.fetch_add(1, std::memory_order_relaxed);
    }

    uint32_t available_free_blocks() const {
        return free_count.load(std::memory_order_relaxed);
    }

    // BUGFIX 128: Evict oldest non-pinned block when cache is full
    // Returns -1 if no evictable blocks available (all pinned or in use)
    int evict_lru_block() {
        std::lock_guard<std::mutex> lock(mtx);
        // Fix 25: Compact heap if too many stale entries
        if (lru_heap_compact_needed && lru_heap.size() > max_blocks * 2) {
            std::vector<std::pair<uint64_t, uint32_t>> fresh;
            fresh.reserve(max_blocks);
            for (auto& e : lru_heap) {
                if (e.second < max_blocks && !is_free[e.second] &&
                    blocks[e.second].last_access_epoch == e.first) {
                    fresh.push_back(e);
                }
            }
            lru_heap = std::move(fresh);
            std::make_heap(lru_heap.begin(), lru_heap.end(), std::greater<std::pair<uint64_t, uint32_t>>());
            lru_heap_compact_needed = false;
        }
        while (!lru_heap.empty()) {
            std::pop_heap(lru_heap.begin(), lru_heap.end(), std::greater<std::pair<uint64_t, uint32_t>>());
            const auto entry = lru_heap.back();
            lru_heap.pop_back();
            const uint32_t id = entry.second;
            if (id >= max_blocks || is_free[id] || blocks[id].pinned || blocks[id].num_tokens == 0) {
                continue;
            }
            if (blocks[id].last_access_epoch != entry.first) {
                lru_heap.push_back({blocks[id].last_access_epoch, id});
                std::push_heap(lru_heap.begin(), lru_heap.end(), std::greater<std::pair<uint64_t, uint32_t>>());
                lru_heap_compact_needed = true;
                continue;
            }
            blocks[id].num_tokens = 0;
            is_free[id] = true;
            free_blocks.push_back(id);
            free_count.fetch_add(1, std::memory_order_relaxed);
            return (int)id;
        }
        return -1;
    }
};
