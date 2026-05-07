// Raw forward CPU - Main entry point
// Modularized: Sections extracted to separate files

#include "raw_forward_utils.cpp.inc"
#include "raw_forward_tensor_desc.cpp.inc"
#include "raw_forward_tensor_cache.cpp.inc"
#include "raw_forward_ready_checks.cpp.inc"
#include "raw_forward_ops.cpp.inc"
#include "raw_forward_embedding.cpp.inc"
#include "raw_forward_qkv.cpp.inc"
#include "raw_forward_lm_head.cpp.inc"

// Additional 5: CPU Fallback redesigned as "Hot Miss" handling
// This determines if CPU computation is faster than PCIe transfer for small batches
static bool should_use_cpu_compute(const moe_pc_engine* engine, int layer, int expert, uint32_t token_count) {
    if (!engine) return false;
    
    // Check if expert is in RAM (not VRAM, not DB)
    const size_t slot = expert_dense_slot(engine, layer, expert);
    if (slot >= engine->expert_states.size()) return false;
    const moe_block_state& state = engine->expert_states[slot];
    if (state.tier != moe_TIER_RAM) return false;
    
    if (token_count <= 4) {
        const uint64_t bytes = expert_triplet_bytes(engine, layer, expert);
        const float dma_gbps = engine->backend_caps.dma_bandwidth_gbps > 0.0f ?
            engine->backend_caps.dma_bandwidth_gbps : 12.0f;
        if (dma_gbps <= 0.0f) return false;
        const float pcie_ms = bytes / (dma_gbps * 1e9f / 1000.f);
        
        // Estimate CPU AMX / AVX-512 ms (rough heuristic)
        const uint32_t hw = std::max<uint32_t>(1u, moe_cpu_thread_count_hint());
        const float cpu_gflops =
            (engine->backend_caps.backend == moe_BACKEND_CPU && engine->backend_caps.tflops_hint > 0.0) ?
            std::max(100.0f, (float)(engine->backend_caps.tflops_hint * 1000.0)) :
            std::min(4000.0f, std::max(250.0f, (float)hw * 150.0f));
        const float compute_gflops = (bytes / 2.0f * 2.0f * token_count) / 1e9f;
        const float cpu_ms = (compute_gflops / cpu_gflops) * 1000.0f;
        
        return cpu_ms < pcie_ms;
    }
    return false;
}
