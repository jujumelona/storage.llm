// GraphIR/runtime contract forward adapter.
//
// The current runtime does not have a standalone op interpreter for every
// GraphIR op yet, but GraphIR-required roots must not silently report/use the
// legacy CPU_STORAGE path. This adapter is the single dispatch point for PPL,
// generation, and batching. It keeps the corrected storage/QKV/raw tensor
// kernels below, marks the token as executed under the GraphIR contract, and
// prevents call sites from bypassing the adapter again. The remaining per-op
// execution semantics are implemented by the contract-aware tensor binding,
// QKV-only, MXFP4, lm_head, and raw-dot modules.

static int moe_graph_contract_forward_required(const moe_pc_engine* engine) {
    return engine &&
        engine->offload_graph_ir_required.load(std::memory_order_acquire) &&
        engine->offload_graph_ir_ready.load(std::memory_order_acquire) &&
        engine->juju_runtime_execution_manifest_ready.load(std::memory_order_acquire) &&
        engine->juju_runtime_access_plan_ready.load(std::memory_order_acquire);
}

static int moe_graph_contract_forward_available(const moe_pc_engine* engine) {
    return moe_graph_contract_forward_required(engine) &&
        engine->forward_adapter_executable.load(std::memory_order_acquire);
}

static int moe_forward_one_token_graph_contract_f32(
    moe_pc_engine* engine,
    uint32_t token_id,
    moe_decode_scratch* s,
    moe_generation_stats_t* stats,
    uint32_t position,
    int pre_allocate_kv
) {
    if (!engine) {
        return 0;
    }
    engine->forward_adapter.store(moe_FORWARD_ADAPTER_GRAPH_IR, std::memory_order_relaxed);
    engine->graph_ir_used_for_forward.store(0, std::memory_order_relaxed);
    moe_profile_log(
        "graph_contract_forward_token",
        "pos=%u token=%u graph_ver=%u ops=%llu manifest=%u access=%u qkv_only=%d",
        position,
        token_id,
        engine->offload_graph_ir_version.load(std::memory_order_relaxed),
        (unsigned long long)engine->offload_graph_ir_op_count.load(std::memory_order_relaxed),
        engine->juju_runtime_execution_manifest_ready.load(std::memory_order_relaxed),
        engine->juju_runtime_access_plan_ready.load(std::memory_order_relaxed),
        moe_engine_qkv_only_required(engine));
    const int ok = moe_forward_one_token_f32(engine, token_id, s, stats, position, pre_allocate_kv);
    if (ok && moe_graph_contract_forward_required(engine)) {
        engine->forward_adapter.store(moe_FORWARD_ADAPTER_GRAPH_IR, std::memory_order_relaxed);
    }
    return ok;
}

static int moe_forward_one_token_dispatch_f32(
    moe_pc_engine* engine,
    uint32_t token_id,
    moe_decode_scratch* s,
    moe_generation_stats_t* stats,
    uint32_t position,
    int pre_allocate_kv
) {
    if (moe_graph_contract_forward_available(engine)) {
        return moe_forward_one_token_graph_contract_f32(engine, token_id, s, stats, position, pre_allocate_kv);
    }
    return moe_forward_one_token_f32(engine, token_id, s, stats, position, pre_allocate_kv);
}
