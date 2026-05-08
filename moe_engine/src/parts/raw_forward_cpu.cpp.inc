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

static double moe_observed_h2d_ms_per_byte(const moe_pc_engine* engine) {
    if (!engine) {
        return 0.0;
    }
    if (engine->backend_caps.measured_h2d_bw_bytes_per_sec > 0) {
        return 1000.0 / (double)engine->backend_caps.measured_h2d_bw_bytes_per_sec;
    }
    const double dma_gbps = engine->backend_caps.dma_bandwidth_gbps > 0.0
        ? engine->backend_caps.dma_bandwidth_gbps : 12.0;
    return dma_gbps > 0.0 ? 1000.0 / (dma_gbps * 1000000000.0) : 0.0;
}

static double moe_observed_cpu_expert_ms(
    const moe_pc_engine* engine,
    int layer,
    int expert,
    uint32_t token_count
) {
    if (!engine || token_count == 0) {
        return 1.0e300;
    }
    const uint64_t bytes = expert_triplet_bytes(engine, layer, expert);
    if (bytes == 0) {
        return 1.0e300;
    }
    const uint32_t hw = std::max<uint32_t>(1u, moe_cpu_thread_count_hint());
    const double cpu_gflops = std::min(3000.0, std::max(200.0, (double)hw * 100.0));
    const double compute_gflops =
        ((double)bytes * (double)std::max<uint32_t>(1u, token_count)) / 1000000000.0;
    return (compute_gflops / cpu_gflops) * 1000.0;
}

// CPU fallback is a narrow hot-miss escape hatch. Prefer GPU promotion unless
// the expert is RAM-resident, VRAM is tight, and CPU is clearly faster.
static bool should_use_cpu_compute(const moe_pc_engine* engine, int layer, int expert, uint32_t token_count) {
    if (!engine) return false;
    if (!engine->io_config.enable_cold_expert_cpu_fallback) return false;

    const size_t slot = expert_dense_slot(engine, layer, expert);
    if (slot >= engine->expert_states.size()) return false;
    const moe_block_state& state = engine->expert_states[slot];
    if (state.tier != moe_TIER_RAM) return false;
    if (token_count == 0 || token_count > engine->io_config.cpu_fallback_max_tokens) return false;

    const uint64_t bytes = expert_triplet_bytes(engine, layer, expert);
    if (bytes == 0) return false;

    const uint64_t available = device_budget_bytes(const_cast<moe_pc_engine*>(engine));
    const uint64_t safety = engine->io_config.vram_safety_margin_bytes;
    if (available > bytes + safety) {
        return false;
    }

    const double observed_h2d_ms = moe_observed_h2d_ms_per_byte(engine) * (double)bytes;
    const double observed_cpu_ms = moe_observed_cpu_expert_ms(engine, layer, expert, token_count);
    if (!(observed_h2d_ms > 0.0) || !(observed_cpu_ms > 0.0)) {
        return false;
    }
    return observed_cpu_ms * 1.25 < observed_h2d_ms;
}
