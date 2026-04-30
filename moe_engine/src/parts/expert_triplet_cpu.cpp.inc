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
    const moe_storage_constants_t* storage = moe_storage_constants();
    const uint32_t hidden_size = storage && storage->hidden_size ? storage->hidden_size : 6144u;
    const uint32_t intermediate_size =
        storage && storage->expert_intermediate_size ? storage->expert_intermediate_size : 2048u;
    if (hidden_count != hidden_size || out_count != hidden_size || intermediate_count != intermediate_size) {
        return 0;
    }
    const moe_projection_shape_spec_t* gate_shape = moe_storage_projection_shape(moe_PROJ_GATE);
    const moe_projection_shape_spec_t* up_shape = moe_storage_projection_shape(moe_PROJ_UP);
    const moe_projection_shape_spec_t* down_shape = moe_storage_projection_shape(moe_PROJ_DOWN);
    if (!gate_shape || !up_shape || !down_shape) {
        return 0;
    }
    const moe_tensor_record* gate = require_expert_projection(
        engine, layer, expert, moe_PROJ_GATE, gate_shape->rows, gate_shape->cols);
    const moe_tensor_record* up = require_expert_projection(
        engine, layer, expert, moe_PROJ_UP, up_shape->rows, up_shape->cols);
    const moe_tensor_record* down = require_expert_projection(
        engine, layer, expert, moe_PROJ_DOWN, down_shape->rows, down_shape->cols);
    if (!gate || !up || !down) {
        return 0;
    }
    mmap_context_t* gate_shard = get_or_map_tensor_shard(engine, *gate);
    mmap_context_t* up_shard = get_or_map_tensor_shard(engine, *up);
    mmap_context_t* down_shard = get_or_map_tensor_shard(engine, *down);
    if (!gate_shard || !up_shard || !down_shard) {
        return 0;
    }
    const int gate_up_ok = moe_parallel_rows(engine, intermediate_count, 0, [&](uint32_t row) {
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
        if (!moe_dot_tensor_record_row(engine, down, down_shard, row, scratch_gate, intermediate_count, &out_hidden[row])) {
            return 0;
        }
        return 1;
    });
}
