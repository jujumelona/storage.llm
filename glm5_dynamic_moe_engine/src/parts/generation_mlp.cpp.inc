static int glm5_layer_dense_mlp_f32(
    glm5_pc_engine* engine,
    int32_t layer,
    glm5_decode_scratch* s,
    float* hidden,
    uint32_t hidden_size
) {
    const uint32_t dense_intermediate = 12288u;
    const uint64_t bytes =
        2ull * (uint64_t)dense_intermediate * (uint64_t)hidden_size * 2ull +
        (uint64_t)hidden_size * (uint64_t)dense_intermediate * 2ull;
    // Bug #5 Fix: Dense MLP should use DENSE_MLP role, not ROUTER.
    // Fall back to ROUTER for backward compatibility with old manifests.
    const glm5_storage_raw_span_spec_t* span = find_raw_role_span_bytes(layer, GLM5_RAW_ROLE_DENSE_MLP, bytes, 0);
    if (!span) {
        span = find_raw_role_span_bytes(layer, GLM5_RAW_ROLE_ROUTER, bytes, 0);
    }
    if (!span) {
        return 0;
    }
    if (!glm5_raw_mlp_span_f32(
            engine, span, s->norm.data(), hidden_size, dense_intermediate,
            s->gate.data(), s->up.data(), s->mlp.data())) {
        return 0;
    }
    for (uint32_t i = 0; i < hidden_size; ++i) {
        hidden[i] += s->mlp[i];
    }
    return 1;
}

static int glm5_layer_moe_mlp_f32(
    glm5_pc_engine* engine,
    int32_t layer,
    glm5_decode_scratch* s,
    float* hidden,
    const glm5_storage_constants_t* c
) {
    const uint32_t hidden_size = c->hidden_size;
    const uint32_t expert_intermediate = c->expert_intermediate_size;
    std::fill(s->mlp.begin(), s->mlp.end(), 0.0f);

    const uint64_t shared_bytes =
        2ull * (uint64_t)expert_intermediate * (uint64_t)hidden_size * 2ull +
        (uint64_t)hidden_size * (uint64_t)expert_intermediate * 2ull;
    const glm5_storage_raw_span_spec_t* shared = find_raw_role_span_bytes(layer, GLM5_RAW_ROLE_SHARED, shared_bytes, 0);
    if (shared &&
        glm5_raw_mlp_span_f32(
            engine, shared, s->norm.data(), hidden_size, expert_intermediate,
            s->gate.data(), s->up.data(), s->shared.data())) {
        for (uint32_t i = 0; i < hidden_size; ++i) {
            s->mlp[i] += s->shared[i];
        }
    }

    if (!glm5_router_topk_f32(
            engine,
            layer,
            s->norm.data(),
            hidden_size,
            c->experts_per_moe_layer,
            GLM5_MOE_TOP_EXPERTS,
            &s->selected_experts,
            &s->selected_weights,
            &s->router_logits,
            &s->router_order)) {
        return 0;
    }

    // Fix 1: Expert-level parallelism.
    // Phase 1: Batch-request all experts upfront so prefetch can overlap.
    const glm5_tier_t target_tier = prefetch_plan_selected_tier(engine);
    const size_t selected_count = std::min(s->selected_experts.size(), (size_t)GLM5_MOE_TOP_EXPERTS);
    uint8_t stream_inflight[GLM5_MOE_TOP_EXPERTS] = {};
    {
        std::lock_guard<std::mutex> lock(engine->map_mutex);
        for (size_t e = 0; e < selected_count; ++e) {
            const int expert_id = s->selected_experts[e];
            const size_t eslot = expert_dense_slot(layer, expert_id);
            if (eslot < engine->expert_states.size()) {
                const glm5_block_state& state = engine->expert_states[eslot];
                stream_inflight[e] = state.active && state.tier == GLM5_TIER_STREAM ? 1u : 0u;
            }
        }
    }
    for (size_t e = 0; e < selected_count; ++e) {
        const int expert_id = s->selected_experts[e];
        if (stream_inflight[e]) {
            continue;
        }
        glm5_pc_engine_request_expert(engine, layer, expert_id,
            expert_triplet_bytes(engine, layer, expert_id), target_tier);
        enqueue_tensor_prefetch(engine, layer, expert_id, GLM5_PROJ_GATE, target_tier, 1);
        enqueue_tensor_prefetch(engine, layer, expert_id, GLM5_PROJ_UP, target_tier, 1);
        enqueue_tensor_prefetch(engine, layer, expert_id, GLM5_PROJ_DOWN, target_tier, 1);
    }
    if (engine->io_config.enable_compute_io_overlap) {
        engine->disk_stage.cv.notify_all();
    }

    // Phase 2: Run experts in parallel with per-expert scratch buffers.
    // Each expert gets its own gate/up/out buffers to prevent data races.
    const size_t num_selected = s->selected_experts.size();
    if (num_selected > s->expert_work.size()) {
        return 0;
    }
    glm5_expert_work_t* work = s->expert_work.data();
    for (size_t e = 0; e < num_selected; ++e) {
        work[e].gate = s->expert_gate_pool.data() + e * (size_t)expert_intermediate;
        work[e].up = s->expert_up_pool.data() + e * (size_t)expert_intermediate;
        work[e].out = s->expert_out_pool.data() + e * (size_t)hidden_size;
        std::fill(work[e].out, work[e].out + hidden_size, 0.0f);
        work[e].weight = s->selected_weights[e];
        work[e].expert_id = s->selected_experts[e];
        work[e].ok = 0;
    }

    // Fix 2: Flat row-parallel expert dispatch.
    // Instead of 8 sequential run_expert_triplet_f32 calls (each taking the full
    // pool via cpu_row_active), we pre-map all shards, then dispatch ALL experts'
    // rows in one parallel_rows call. This eliminates cpu_row_active serialization.
    const glm5_projection_shape_spec_t* gate_shape = glm5_storage_projection_shape(GLM5_PROJ_GATE);
    const glm5_projection_shape_spec_t* up_shape = glm5_storage_projection_shape(GLM5_PROJ_UP);
    const glm5_projection_shape_spec_t* down_shape = glm5_storage_projection_shape(GLM5_PROJ_DOWN);
    if (!gate_shape || !up_shape || !down_shape) return 0;

    // Pre-map all expert weight shards (these are cached, so fast for hot experts)
    struct expert_shards_t {
        const glm5_tensor_record* gate_rec = nullptr;
        const glm5_tensor_record* up_rec = nullptr;
        const glm5_tensor_record* down_rec = nullptr;
        mmap_context_t* gate_shard = nullptr;
        mmap_context_t* up_shard = nullptr;
        mmap_context_t* down_shard = nullptr;
    };
    expert_shards_t shards[GLM5_MOE_TOP_EXPERTS];
    std::atomic<int> expert_ok[GLM5_MOE_TOP_EXPERTS];
    for (uint32_t e = 0; e < GLM5_MOE_TOP_EXPERTS; ++e) {
        expert_ok[e].store(0, std::memory_order_relaxed);
    }
    for (size_t e = 0; e < num_selected; ++e) {
        shards[e].gate_rec = find_tensor_record(engine, layer, work[e].expert_id, GLM5_PROJ_GATE);
        shards[e].up_rec = find_tensor_record(engine, layer, work[e].expert_id, GLM5_PROJ_UP);
        shards[e].down_rec = find_tensor_record(engine, layer, work[e].expert_id, GLM5_PROJ_DOWN);
        if (!shards[e].gate_rec || !shards[e].up_rec || !shards[e].down_rec) {
            continue;
        }
        if (shards[e].gate_rec->info.rows != gate_shape->rows || shards[e].gate_rec->info.cols != gate_shape->cols ||
            shards[e].up_rec->info.rows != up_shape->rows || shards[e].up_rec->info.cols != up_shape->cols ||
            shards[e].down_rec->info.rows != down_shape->rows || shards[e].down_rec->info.cols != down_shape->cols) {
            continue;
        }
        shards[e].gate_shard = get_or_map_tensor_shard(engine, *shards[e].gate_rec);
        shards[e].up_shard = get_or_map_tensor_shard(engine, *shards[e].up_rec);
        shards[e].down_shard = get_or_map_tensor_shard(engine, *shards[e].down_rec);
        if (!shards[e].gate_shard || !shards[e].up_shard || !shards[e].down_shard) {
            continue;
        }
        expert_ok[e].store(1, std::memory_order_release);
    }

    // Phase 2a: Flat dispatch: gate+up+SiLU for all experts simultaneously.
    // Total rows = num_selected * intermediate_size, row i maps to expert e and row r.
    const uint32_t total_gate_rows = (uint32_t)num_selected * expert_intermediate;
    const int gate_up_ok = glm5_parallel_rows(engine, total_gate_rows, 0, [&](uint32_t flat_row) {
        const uint32_t e = flat_row / expert_intermediate;
        const uint32_t r = flat_row % expert_intermediate;
        if (!expert_ok[e].load(std::memory_order_acquire)) return 1;
        if (!glm5_dot_tensor_record_row(engine, shards[e].gate_rec, shards[e].gate_shard,
                r, s->norm.data(), hidden_size, &work[e].gate[r]) ||
            !glm5_dot_tensor_record_row(engine, shards[e].up_rec, shards[e].up_shard,
                r, s->norm.data(), hidden_size, &work[e].up[r])) {
            expert_ok[e].store(0, std::memory_order_release);
            return 1;
        }
        work[e].gate[r] = glm5_generation_silu(work[e].gate[r]) * work[e].up[r];
        return 1;
    });
    if (!gate_up_ok) return 0;

    // Phase 2b: Flat dispatch: down_proj for all experts simultaneously.
    const uint32_t total_down_rows = (uint32_t)num_selected * hidden_size;
    const int down_ok = glm5_parallel_rows(engine, total_down_rows, 0, [&](uint32_t flat_row) {
        const uint32_t e = flat_row / hidden_size;
        const uint32_t r = flat_row % hidden_size;
        if (!expert_ok[e].load(std::memory_order_acquire)) return 1;
        if (!glm5_dot_tensor_record_row(engine, shards[e].down_rec, shards[e].down_shard,
                r, work[e].gate, expert_intermediate, &work[e].out[r])) {
            expert_ok[e].store(0, std::memory_order_release);
        }
        return 1;
    });
    if (!down_ok) return 0;
    for (size_t e = 0; e < num_selected; ++e) {
        work[e].ok = expert_ok[e].load(std::memory_order_acquire);
    }

    // Phase 3: Accumulate weighted expert outputs into mlp.
    for (size_t e = 0; e < num_selected; ++e) {
        if (!work[e].ok) continue;
        glm5_axpy_f32(s->mlp.data(), work[e].out, work[e].weight, hidden_size);
    }

    for (uint32_t i = 0; i < hidden_size; ++i) {
        hidden[i] += s->mlp[i];
    }
    return 1;
}
