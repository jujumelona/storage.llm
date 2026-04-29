static constexpr uint32_t GLM5_RAW_ROLE_ROUTER = 1u;
static constexpr uint32_t GLM5_RAW_ROLE_ATTENTION = 2u;
static constexpr uint32_t GLM5_RAW_ROLE_NORM = 4u;
static constexpr uint32_t GLM5_RAW_ROLE_SHARED = 8u;
static constexpr uint32_t GLM5_RAW_ROLE_EMBED = 16u;
static constexpr uint32_t GLM5_RAW_ROLE_LM_HEAD = 32u;
static constexpr uint32_t GLM5_RAW_ROLE_DENSE_MLP = 64u;

static uint64_t raw_phase_budget_left(const glm5_pc_engine* e) {
    if (!e || e->config.vram_budget_bytes == 0) {
        return UINT64_MAX;
    }
    const uint64_t used = e->device_allocated_bytes.load();
    return e->config.vram_budget_bytes > used ? e->config.vram_budget_bytes - used : 0;
}

static int prefetch_one_raw_span(glm5_pc_engine* e, const glm5_storage_raw_span_spec_t& s, uint8_t pool) {
    if (!e || e->model_root.empty() || s.bytes == 0) {
        return 0;
    }
    if (device_copy_raw_span(e, s, pool)) {
        e->io_atomic.bytes_prefetched.fetch_add(s.bytes);
        e->io_atomic.bytes_pinned_to_gpu.fetch_add(s.bytes);
        e->io_atomic.gpu_stage_done.fetch_add(1);
        return 1;
    }
    const glm5_storage_part_spec_t* part = glm5_storage_part_by_id(s.part);
        mmap_context_t* mapped = part ? get_or_map_shard_partial_by_id(
            e,
            s.part,
            resolve_model_file_path(e->model_root, part->path ? part->path : "", ""),
            s.offset,
            s.bytes
        ) : nullptr;
    const uint64_t file_size = mapped ? (uint64_t)mmap_get_size(mapped) : 0;
    if (!mapped || !mmap_get_ptr(mapped, 0) || s.bytes > file_size) {
        return 0;
    }
    bool hint_issued = false;
    if (e->io_config.enable_mmap_prefetch) {
        mmap_prefetch(mapped, 0, (size_t)s.bytes);
        hint_issued = true;
    }
    const uint8_t* src = static_cast<const uint8_t*>(mmap_get_ptr(mapped, 0));
    if (!src) {
        return 0;
    }
    touch_mapped_range(src, s.bytes);
    if (!hint_issued) {
        e->io_atomic.bytes_touch_fallback.fetch_add(s.bytes);
    }
    e->io_atomic.bytes_prefetched.fetch_add(s.bytes);
    e->io_atomic.bytes_disk_to_ram.fetch_add(s.bytes);
    e->io_atomic.disk_stage_done.fetch_add(1);
    return 1;
}

static int enqueue_raw_span_prefetch(glm5_pc_engine* e, uint32_t span_index, uint8_t pool) {
    if (!e) {
        return 0;
    }
    if (!ensure_io_ready(e)) {
        return 0;
    }
    glm5_prefetch_task task{};
    task.raw_span_index = (int32_t)span_index;
    task.raw_pool = pool;
    task.urgent = 1;
    if (!enqueue_stage_or_retry(e, e->disk_stage, task)) {
        return 0;
    }
    e->io_atomic.queued_requests.fetch_add(1);
    e->disk_stage.cv.notify_one();
    return 1;
}

static uint64_t prefetch_raw_layer_roles(
    glm5_pc_engine* e,
    int32_t layer,
    uint32_t roles,
    uint8_t pool,
    uint64_t budget,
    int enqueue_only = 0
) {
    uint64_t bytes = 0;
    uint64_t spans = 0;
    const uint32_t begin = glm5_storage_raw_span_find_first(layer);
    const uint32_t end = glm5_storage_raw_span_find_end(layer);
    for (uint32_t i = begin; i < end; ++i) {
        const glm5_storage_raw_span_spec_t* s = glm5_storage_raw_span_at(i);
        if (!s || (s->role & roles) == 0 || bytes >= budget || s->bytes > budget - bytes ||
            s->bytes > raw_phase_budget_left(e)) {
            continue;
        }
        const int ok = enqueue_only ?
            enqueue_raw_span_prefetch(e, i, pool) :
            prefetch_one_raw_span(e, *s, pool);
        if (ok) {
            bytes += s->bytes;
            ++spans;
        }
    }
    if (bytes) {
        e->common_raw_prefetched_bytes.fetch_add(bytes);
        e->common_raw_prefetched_spans.fetch_add(spans);
    }
    return bytes;
}

static int prefetch_fixed_core_raw(glm5_pc_engine* e) {
    uint64_t budget = e ? e->common_vram_reserved_bytes.load() : 0;
    uint64_t bytes = 0;
    if (!e || !e->io_config.enable_common_raw_vram_pin || budget == 0) {
        if (e) {
            e->common_vram_upload_complete.store(true, std::memory_order_release);
        }
        return 0;
    }
    const glm5_storage_constants_t* storage = glm5_storage_constants();
    const int32_t norm_layer = storage ? (int32_t)storage->num_hidden_layers : 79;
    const int32_t lm_head_layer = norm_layer + 1;
    std::fprintf(stderr, "[storagellm] common raw device pin start: budget=%llu\n",
        (unsigned long long)budget);
    std::fflush(stderr);
    bytes += prefetch_raw_layer_roles(e, norm_layer, GLM5_RAW_ROLE_NORM, GLM5_DEVICE_POOL_FIXED, budget - bytes);
    bytes += prefetch_raw_layer_roles(e, lm_head_layer, GLM5_RAW_ROLE_LM_HEAD, GLM5_DEVICE_POOL_FIXED, budget - bytes);
    if (bytes < budget) {
        bytes += prefetch_raw_layer_roles(e, -1, GLM5_RAW_ROLE_EMBED, GLM5_DEVICE_POOL_FIXED, budget - bytes);
    }
    std::fprintf(stderr, "[storagellm] common raw device pin done: bytes=%llu reserve=%llu device=%llu allocs=%llu failures=%llu\n",
        (unsigned long long)bytes,
        (unsigned long long)budget,
        (unsigned long long)e->device_allocated_bytes.load(std::memory_order_relaxed),
        (unsigned long long)e->device_allocation_count.load(std::memory_order_relaxed),
        (unsigned long long)e->io_atomic.device_alloc_failures.load(std::memory_order_relaxed));
    std::fflush(stderr);
    return bytes > 0 ? 1 : 0;
}

static int prefetch_common_raw_storage(glm5_pc_engine* e) {
    int ok = prefetch_fixed_core_raw(e);
    if (e && ok) {
        const bool was_incomplete = !e->common_vram_upload_complete.exchange(true);
        if (was_incomplete) {
            e->vram_used_bytes.fetch_add(e->common_vram_reserved_bytes.load());
            // Bug 19 Fix: Re-evaluate RAM-staged experts for VRAM promotion
            // When common_vram_upload_complete flips to true, hot experts that were
            // demoted to RAM can now be promoted to VRAM
            const uint64_t vram_budget = tier_budget(e, GLM5_TIER_VRAM);
            const uint64_t vram_used = tier_used(e, GLM5_TIER_VRAM);
            if (vram_budget > vram_used) {
                const uint64_t vram_room = vram_budget - vram_used;
                const uint64_t avg_expert_bytes = e->max_tensor_stream_bytes > 0
                    ? e->max_tensor_stream_bytes : 20ull * 1024ull * 1024ull;
                const uint32_t max_promote = vram_room > 0 && avg_expert_bytes > 0
                    ? std::min<uint32_t>(32u, (uint32_t)(vram_room / avg_expert_bytes)) : 0u;
                if (max_promote > 0) {
                    std::vector<std::pair<size_t, double>> ram_experts;
                    {
                        std::lock_guard<std::mutex> map_lock(e->map_mutex);
                        for (size_t slot = 0; slot < e->expert_states.size(); ++slot) {
                            const glm5_block_state& state = e->expert_states[slot];
                            if (state.active && state.tier == GLM5_TIER_RAM && !state.pending_evict) {
                                ram_experts.push_back({slot, state.hot_score});
                            }
                        }
                    }
                    std::sort(ram_experts.begin(), ram_experts.end(),
                        [](const auto& a, const auto& b) { return a.second > b.second; });
                    const uint32_t promote_count = std::min<uint32_t>(max_promote, (uint32_t)ram_experts.size());
                    for (uint32_t i = 0; i < promote_count; ++i) {
                        int layer = 0, expert = 0;
                        expert_dense_slot_decode(ram_experts[i].first, &layer, &expert);
                        const uint64_t bytes = expert_triplet_bytes(e, layer, expert);
                        if (bytes > 0) {
                            glm5_pc_engine_request_expert(e, layer, expert, bytes, GLM5_TIER_VRAM);
                        }
                    }
                }
            }
        }
    }
    return ok;
}

// static thread_local int tls_last_activation_layer = -1; // Removed in favor of engine member

static int activation_key_layer(uint64_t key, int* out_layer) {
    if (!out_layer) {
        return 0;
    }
    const uint32_t key_type = (uint32_t)((key >> 60) & 0x0f);
    if (key_type == 3u) {
        *out_layer = (int)((key >> 48) & 0x0fffu);
        return 1;
    }
    if (key_type == 4u) {
        const uint32_t encoded = (uint32_t)((key >> 40) & 0xffffu);
        *out_layer = encoded == 0xffffu ? -1 : (int)encoded;
        return 1;
    }
    return 0;
}

static uint32_t runtime_prefetch_window(const glm5_pc_engine* engine) {
    if (!engine) {
        return 1u;
    }
    uint32_t window = engine->runtime_prefetch_window_layers.load(std::memory_order_relaxed);
    if (window == 0) {
        window = engine->io_config.prefetch_window_layers;
    }
    if (window == 0) {
        window = 1u;
    }
    return std::min<uint32_t>(window, 8u);
}

static int prefetch_moe_activation_raw(glm5_pc_engine* e, int layer) {
    if (!e || layer < 0) {
        return 0;
    }
    int previous_layer = e->last_activation_layer.load(std::memory_order_acquire);
    while (previous_layer != layer) {
        if (e->last_activation_layer.compare_exchange_weak(
                previous_layer,
                layer,
                std::memory_order_acq_rel,
                std::memory_order_acquire)) {
            break;
        }
    }
    if (previous_layer == layer) {
        return 1;
    }
    const glm5_storage_constants_t* c = glm5_storage_constants();
    const int is_moe = c && layer >= (int)c->first_moe_layer && layer <= (int)c->last_moe_layer;
    const uint32_t roles = GLM5_RAW_ROLE_NORM | GLM5_RAW_ROLE_ATTENTION;
    const int ok = prefetch_raw_layer_roles(
        e,
        layer,
        roles,
        GLM5_DEVICE_POOL_ACTIVATION,
        raw_phase_budget_left(e),
        1
    ) > 0 ? 1 : 0;
    if (ok) {
        // Bug #6: Single lock pass - activation_keys is small.
        {
            std::vector<std::pair<CUdeviceptr, uint64_t>> deferred_free_ptrs;
            {
                std::lock_guard<std::mutex> lock(e->device_mutex);
                const int keep_begin = layer > 0 ? layer - 1 : layer;
                const int keep_end = layer + (int)runtime_prefetch_window(e);
                std::vector<uint64_t> to_free_keys;
                for (uint64_t k : e->activation_keys) {
                    bool should_free = false;
                    int key_layer = 0;
                    if (activation_key_layer(k, &key_layer)) {
                        should_free = key_layer < keep_begin || key_layer > keep_end;
                    } else {
                        auto it = e->device_allocations.find(k);
                        should_free = (it != e->device_allocations.end() && it->second.inflight_copies == 0);
                    }
                    if (should_free) {
                        to_free_keys.push_back(k);
                    }
                }
                for (uint64_t k : to_free_keys) {
                    device_free_locked(e, k, &deferred_free_ptrs);
                }
            }
            device_free_deferred_ptrs(e, deferred_free_ptrs);
        }
    } else {
        int claimed_layer = layer;
        e->last_activation_layer.compare_exchange_strong(
            claimed_layer,
            -1,
            std::memory_order_acq_rel,
            std::memory_order_acquire);
    }
    return ok;
}

static void release_moe_activation_raw(glm5_pc_engine* e) {
    e->last_activation_layer.store(-1);
    device_free_pool(e, GLM5_DEVICE_POOL_ACTIVATION);
    device_free_pool(e, GLM5_DEVICE_POOL_STREAM);
}

static void prefetch_moe_shared_compute_raw(glm5_pc_engine* e, int layer) {
    prefetch_raw_layer_roles(
        e,
        layer,
        GLM5_RAW_ROLE_SHARED,
        GLM5_DEVICE_POOL_STREAM,
        raw_phase_budget_left(e),
        1
    );
}
