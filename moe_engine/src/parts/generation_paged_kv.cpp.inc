
// ============================================================================
// PagedAttention KV Cache Manager (vLLM Style Continuous Batching)
// ============================================================================
// Implements a page-based memory allocator for KV blocks to eliminate memory
// fragmentation and allow dynamic request scheduling without pre-allocation.

#include <vector>
#include <mutex>
#include <cstdint>

struct moe_paged_kv_block_t {
    uint32_t block_id;
    uint32_t num_tokens;
    // Data layout: [layer][token_in_block][dim]
    uint8_t* compressed_data; // INT8
    float* scales;
    float* zeros;
    // BUGFIX 128: Protect sink tokens from eviction
    bool pinned; // true if this block contains sink tokens (initial tokens that must be preserved)
};

struct moe_paged_kv_manager_t {
    uint32_t block_size; // e.g., 16 tokens per block
    uint32_t max_blocks;
    uint32_t sink_token_count; // BUGFIX 128: Number of initial tokens to preserve (typically 4-8)
    std::vector<uint32_t> free_blocks;
    std::vector<moe_paged_kv_block_t> blocks;
    std::mutex mtx;

    void init(uint32_t capacity_tokens, uint32_t b_size, uint32_t layers, uint32_t sink_tokens = 4) {
        block_size = b_size;
        max_blocks = capacity_tokens / block_size;
        sink_token_count = sink_tokens; // BUGFIX 128: Store sink token count
        blocks.resize(max_blocks);
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
    }

    ~moe_paged_kv_manager_t() {
        cleanup();
    }

    int allocate_block() {
        std::lock_guard<std::mutex> lock(mtx);
        if (free_blocks.empty()) return -1;
        uint32_t id = free_blocks.back();
        free_blocks.pop_back();
        blocks[id].num_tokens = 0;
        return id;
    }

    void free_block(uint32_t id) {
        std::lock_guard<std::mutex> lock(mtx);
        if (id < max_blocks) free_blocks.push_back(id);
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
            
            // Skip blocks that are already free
            bool is_free = false;
            for (uint32_t free_id : free_blocks) {
                if (free_id == i) {
                    is_free = true;
                    break;
                }
            }
            if (is_free) {
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
            // Clear the block and add to free list
            blocks[evict_candidate].num_tokens = 0;
            free_blocks.push_back(evict_candidate);
        }
        
        return evict_candidate;
    }
};
