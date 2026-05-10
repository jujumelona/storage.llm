// MODULARIZATION: Paged KV cache split by functionality
// Original 729 lines split into 3 files

#ifndef MOE_ENABLE_PAGED_KV
#define MOE_ENABLE_PAGED_KV 0
#endif

#if MOE_ENABLE_PAGED_KV
#include "generation/paged_kv_alloc.cpp.inc"
#include "generation/paged_kv_ops.cpp.inc"
#include "generation/paged_kv_mgmt.cpp.inc"
#else
#ifndef MOE_GENERATION_PAGED_KV_TYPES_DEFINED
#define MOE_GENERATION_PAGED_KV_TYPES_DEFINED
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
#endif
