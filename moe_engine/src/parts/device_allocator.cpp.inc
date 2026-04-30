// Device allocator - Main entry point
// Modularized: Sections extracted to separate files

static void device_free_deferred_ptrs(moe_pc_engine* e, std::vector<std::pair<CUdeviceptr, uint64_t>>& ptrs);
static void device_free_locked(
    moe_pc_engine* e,
    uint64_t key,
    std::vector<std::pair<CUdeviceptr, uint64_t>>* deferred_free_ptrs
);

#include "device_allocator_init.cpp.inc"
#include "device_allocator_memory.cpp.inc"
#include "device_allocator_eviction.cpp.inc"

static void device_free_expert_locked(
    moe_pc_engine* e,
    int layer,
    int expert,
    std::vector<std::pair<CUdeviceptr, uint64_t>>* deferred_free_ptrs
);
