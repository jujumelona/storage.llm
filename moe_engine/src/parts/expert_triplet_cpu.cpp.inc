// Bug #10: CPU correctness fallback ??use exact expf, not polynomial approximation
// The (1+x/256)^256 method has 0.4% error at x=10, which makes CPU/GPU validation diverge
static float moe_fast_exp(float x) {
    return expf(x);
}
static float moe_silu(float x) {
    return x / (1.0f + moe_fast_exp(-x));
}

static const moe_tensor_record* require_expert_projection(
    moe_pc_engine_t* engine,
    int layer,
    int expert,
    moe_projection_t proj,
    uint32_t rows,
    uint32_t cols
) {
    const moe_tensor_record* rec = find_tensor_record(engine, layer, expert, proj);
    return rec && rec->info.rows == rows && rec->info.cols == cols ? rec : nullptr;
}

int moe_pc_engine_run_expert_triplet_f32(
    moe_pc_engine_t* engine,
    int layer,
    int expert,
    const float* hidden,
    uint32_t hidden_count,
    float* scratch_gate,
    float* scratch_up,
    uint32_t intermediate_count,
    float* out_hidden,
    uint32_t out_count
) {
    if (!engine || !hidden || !scratch_gate || !scratch_up || !out_hidden) {
        return 0;
    }
    const moe_storage_constants_t storage = moe_engine_storage_constants_or_static(engine);
    const uint32_t hidden_size = storage.hidden_size;
    const uint32_t intermediate_size =
        storage.expert_intermediate_size;
    if (hidden_count != hidden_size || out_count != hidden_size || intermediate_count != intermediate_size) {
        return 0;
    }
    const moe_tensor_record* gate = require_expert_projection(
        engine, layer, expert, moe_PROJ_GATE, intermediate_size, hidden_size);
    const moe_tensor_record* up = require_expert_projection(
        engine, layer, expert, moe_PROJ_UP, intermediate_size, hidden_size);
    const moe_tensor_record* down = require_expert_projection(
        engine, layer, expert, moe_PROJ_DOWN, hidden_size, intermediate_size);
    if (!gate || !up || !down) {
        return 0;
    }
    mmap_context_t* gate_shard = get_or_map_tensor_shard(engine, *gate);
    mmap_context_t* up_shard = get_or_map_tensor_shard(engine, *up);
    mmap_context_t* down_shard = get_or_map_tensor_shard(engine, *down);
    // BUGFIX 556: Check shard pointer validity before use ★
    // Problem: get_or_map_tensor_shard can return null
    // moe_dot_tensor_record_row accesses shard without null check
    // Solution: Already checked, but ensure mmap_get_ptr also checked
    if (!gate_shard || !up_shard || !down_shard) {
        return 0;
    }
    // BUGFIX 557: Verify shard data pointers are valid ★
    // Problem: mmap_get_ptr can return null even if shard is non-null
    // Solution: Check data pointers before matrix operations
    if (!mmap_get_ptr(gate_shard, 0) || !mmap_get_ptr(up_shard, 0) || !mmap_get_ptr(down_shard, 0)) {
        return 0;
    }
    // BUGFIX 558: Check intermediate_count > 0 before parallel_rows ★
    // Problem: intermediate_count can be 0, causing division by zero in parallel_rows
    // Solution: Check before calling moe_parallel_rows
    if (intermediate_count == 0 || out_count == 0) {
        return 0;
    }

    const int gate_up_ok = moe_parallel_rows(engine, intermediate_count, 0, [&](uint32_t row) {
        // BUGFIX 559: Check row bounds before array access ★
        // Problem: row can exceed intermediate_count due to race condition
        // Solution: Add bounds check
        if (row >= intermediate_count) {
            return 0;
        }
        if (!moe_dot_tensor_record_row(engine, gate, gate_shard, row, hidden, hidden_count, &scratch_gate[row]) ||
            !moe_dot_tensor_record_row(engine, up, up_shard, row, hidden, hidden_count, &scratch_up[row])) {
            return 0;
        }
        scratch_gate[row] = moe_silu(scratch_gate[row]) * scratch_up[row];
        return 1;
    });
    if (!gate_up_ok) {
        return 0;
    }

    return moe_parallel_rows(engine, out_count, 0, [&](uint32_t row) {
        // BUGFIX 560: Check row bounds before array access ★
        // Problem: row can exceed out_count due to race condition
        // Solution: Add bounds check
        if (row >= out_count) {
            return 0;
        }
        if (!moe_dot_tensor_record_row(engine, down, down_shard, row, scratch_gate, intermediate_count, &out_hidden[row])) {
            return 0;
        }
        return 1;
    });
}
