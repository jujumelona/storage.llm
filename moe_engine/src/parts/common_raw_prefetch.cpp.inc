static constexpr uint32_t moe_RAW_ROLE_ROUTER = 1u;
static constexpr uint32_t moe_RAW_ROLE_ATTENTION = 2u;
static constexpr uint32_t moe_RAW_ROLE_NORM = 4u;
static constexpr uint32_t moe_RAW_ROLE_SHARED = 8u;
static constexpr uint32_t moe_RAW_ROLE_EMBED = 16u;
static constexpr uint32_t moe_RAW_ROLE_LM_HEAD = 32u;
static constexpr uint32_t moe_RAW_ROLE_DENSE_MLP = 64u;

static uint64_t raw_phase_budget_left(const moe_pc_engine* e) {
    if (!e || e->config.vram_budget_bytes == 0) {
        return UINT64_MAX;
    }
    const uint64_t used = e->device_allocated_bytes.load();
    return e->config.vram_budget_bytes > used ? e->config.vram_budget_bytes - used : 0;
}

static int prefetch_one_raw_span(moe_pc_engine* e, const moe_storage_raw_span_spec_t& s, uint8_t pool, int allow_sync_fallback) {
    if (!e || e->model_root.empty() || s.bytes == 0) {
        return 0;
    }
    const int copy_ok = device_copy_raw_span(e, s, pool, allow_sync_fallback);
    if (copy_ok == moe_DEVICE_COPY_BACKPRESSURE) {
        return moe_DEVICE_COPY_BACKPRESSURE;
    }
    if (copy_ok) {
        e->io_atomic.bytes_prefetched.fetch_add(s.bytes);
        e->io_atomic.bytes_pinned_to_gpu.fetch_add(s.bytes);
        e->io_atomic.gpu_stage_done.fetch_add(1);
        return 1;
    }
    const moe_storage_part_spec_t* part = moe_storage_part_by_id(s.part);
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
    touch_mapped_range(e, src, s.bytes);
    if (!hint_issued) {
        e->io_atomic.bytes_touch_fallback.fetch_add(s.bytes);
    }
    e->io_atomic.bytes_prefetched.fetch_add(s.bytes);
    e->io_atomic.bytes_disk_to_ram.fetch_add(s.bytes);
    e->io_atomic.disk_stage_done.fetch_add(1);
    return 1;
}

static int enqueue_raw_span_prefetch(moe_pc_engine* e, uint32_t span_index, uint8_t pool) {
    if (!e) {
        return 0;
    }
    const moe_storage_raw_span_spec_t* s = moe_storage_raw_span_at(span_index);
    if (!s) return 0;
    const uint64_t layer = (uint64_t)((uint32_t)s->layer & 0xffffu);
    const uint64_t part = (uint64_t)((uint32_t)s->part & 0xffffu);
    const uint64_t key = (4ull << 60) | ((uint64_t)(pool & 0x0fu) << 56) | (part << 40) | (layer << 24) | ((s->offset >> 12) & 0x00ffffffull);
    {
        std::lock_guard<std::mutex> lock(e->device_mutex);
        auto it = e->device_allocations.find(key);
        if (it != e->device_allocations.end() && !it->second.pending_free) {
            return 1; // BUGFIX: Already resident, skip enqueue!
        }
    }
    if (!ensure_io_ready(e)) {
        return 0;
    }
    moe_prefetch_task task{};
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
    moe_pc_engine* e,
    int32_t layer,
    uint32_t roles,
    uint8_t pool,
    uint64_t budget,
    int enqueue_only = 0
) {
    uint64_t bytes = 0;
    uint64_t spans = 0;
    const uint32_t begin = moe_storage_raw_span_find_first(layer);
    const uint32_t end = moe_storage_raw_span_find_end(layer);
    // BUGFIX 475: begin/end 범위 체크
    if (begin > end) return 0;
    for (uint32_t i = begin; i < end; ++i) {
        const moe_storage_raw_span_spec_t* s = moe_storage_raw_span_at(i);
        if (!s || (s->role & roles) == 0 || bytes >= budget || s->bytes > budget - bytes ||
            s->bytes > raw_phase_budget_left(e)) {
            continue;
        }
        const int ok = enqueue_only ?
            enqueue_raw_span_prefetch(e, i, pool) :
            prefetch_one_raw_span(e, *s, pool, 1);
        if (ok) {
            // BUGFIX 476: bytes overflow 체크
            if (bytes > UINT64_MAX - s->bytes) break;
            bytes += s->bytes;
            ++spans;
        }
    }
    // BUGFIX 33: Separate queued vs transferred byte counters ?�★??    // Old: Both enqueue_only and actual transfer incremented common_raw_prefetched_bytes
    // New: enqueue_only ??common_raw_queued_bytes, actual transfer ??common_raw_prefetched_bytes
    // Prevents stats confusion: "prefetched" now means actually in VRAM, "queued" means in pipeline
    if (bytes) {
        if (enqueue_only) {
            e->common_raw_queued_bytes.fetch_add(bytes);
            e->common_raw_queued_spans.fetch_add(spans);
        } else {
            e->common_raw_prefetched_bytes.fetch_add(bytes);
            e->common_raw_prefetched_spans.fetch_add(spans);
        }
    }
    return bytes;
}

static int common_tensor_is_startup_core(const moe_gguf_common_tensor_record& rec) {
    if (rec.prefetch_class == moe_JUJU_PREFETCH_STARTUP_HOT ||
        rec.graph_role == moe_JUJU_ROLE_NORM) {
        return 1;
    }
    const std::string name = moe_lower_ascii_copy(rec.name);
    return name == "output.weight" ||
        name == "lm_head.weight" ||
        name == "token_embd.weight" ||
        name == "model.embed_tokens.weight" ||
        name == "embed_tokens.weight" ||
        name == "output_norm.weight" ||
        name == "final_norm.weight" ||
        name == "model.norm.weight" ||
        name == "rope_freqs.weight";
}

static int common_tensor_seen(
    const std::vector<std::pair<uint16_t, uint64_t>>& seen,
    const moe_gguf_common_tensor_record& rec
) {
    for (const auto& item : seen) {
        if (item.first == rec.resolved_path_index && item.second == rec.file_offset) {
            return 1;
        }
    }
    return 0;
}

static uint64_t prefetch_common_tensor_records_to_ram(moe_pc_engine* e, uint64_t budget) {
    if (!e || budget == 0 || e->gguf_common_tensors.empty()) {
        return 0;
    }
    uint64_t bytes = 0;
    uint64_t spans = 0;
    std::vector<std::pair<uint16_t, uint64_t>> seen;
    for (const moe_gguf_common_tensor_record& rec : e->gguf_common_tensors) {
        if (!common_tensor_is_startup_core(rec) ||
            rec.bytes == 0 ||
            bytes >= budget ||
            rec.bytes > budget - bytes ||
            rec.resolved_path_index >= e->tensor_paths.size() ||
            common_tensor_seen(seen, rec)) {
            continue;
        }
        const std::string& path = e->tensor_paths[rec.resolved_path_index];
        mmap_context_t* mapped = get_or_map_shard_partial(e, path, rec.file_offset, rec.bytes);
        const uint64_t mapped_size = mapped ? static_cast<uint64_t>(mmap_get_size(mapped)) : 0;
        if (!mapped || !mmap_get_ptr(mapped, 0) ||
            rec.bytes > mapped_size) {
            continue;
        }
        bool hint_issued = false;
        if (e->io_config.enable_mmap_prefetch) {
            mmap_prefetch(mapped, 0, (size_t)rec.bytes);
            hint_issued = true;
        }
        const uint8_t* src = static_cast<const uint8_t*>(mmap_get_ptr(mapped, 0));
        if (!src) {
            continue;
        }
        touch_mapped_range(e, src, rec.bytes);
        if (!hint_issued) {
            e->io_atomic.bytes_touch_fallback.fetch_add(rec.bytes);
        }
        e->io_atomic.bytes_prefetched.fetch_add(rec.bytes);
        e->io_atomic.bytes_disk_to_ram.fetch_add(rec.bytes);
        e->io_atomic.disk_stage_done.fetch_add(1);
        seen.push_back({rec.resolved_path_index, rec.file_offset});
        bytes += rec.bytes;
        ++spans;
    }
    if (bytes) {
        e->common_raw_prefetched_bytes.fetch_add(bytes);
        e->common_raw_prefetched_spans.fetch_add(spans);
    }
    return bytes;
}

static void trigger_ram_expert_promotion(moe_pc_engine* e);
static int prefetch_fixed_core_raw(moe_pc_engine* e) {
    uint64_t budget = e ? e->common_vram_reserved_bytes.load() : 0;
    uint64_t bytes = 0;
    if (!e) {
        return 0;
    }
    const moe_storage_constants_t runtime_storage = moe_engine_storage_constants_or_static(e);
    const moe_storage_constants_t* storage = &runtime_storage;
    const int32_t norm_layer = storage ? (int32_t)storage->num_hidden_layers : 79;
    const int32_t lm_head_layer = norm_layer + 1;
    if (!e->io_config.enable_common_raw_vram_pin || budget == 0) {
        const uint64_t ram_budget = tier_budget(e, moe_TIER_RAM);
        const uint64_t ram_used = tier_used(e, moe_TIER_RAM);
        uint64_t ram_room = ram_budget > ram_used ? ram_budget - ram_used : 0;
        if (ram_room > 0) {
            const uint64_t available = storagellm::available_ram_bytes();
            const uint64_t reserve = 512ull * 1024ull * 1024ull;
            if (available > reserve) {
                ram_room = std::min<uint64_t>(ram_room, available - reserve);
            }
        }
        if (ram_room > 0) {
            std::fprintf(stderr, "[storagellm] common raw RAM warm start: budget=%llu\n",
                (unsigned long long)ram_room);
            std::fflush(stderr);
            bytes += prefetch_raw_layer_roles(e, norm_layer, moe_RAW_ROLE_NORM, moe_DEVICE_POOL_FIXED, ram_room - bytes);
            if (bytes < ram_room) {
                bytes += prefetch_raw_layer_roles(e, lm_head_layer, moe_RAW_ROLE_LM_HEAD, moe_DEVICE_POOL_FIXED, ram_room - bytes);
            }
            if (bytes == 0 && ram_room > 0) {
                bytes += prefetch_raw_layer_roles(e, -1, moe_RAW_ROLE_EMBED, moe_DEVICE_POOL_FIXED, ram_room);
            }
            if (bytes < ram_room) {
                bytes += prefetch_common_tensor_records_to_ram(e, ram_room - bytes);
            }
            std::fprintf(stderr, "[storagellm] common raw RAM warm done: bytes=%llu ram_used=%llu budget=%llu\n",
                (unsigned long long)bytes,
                (unsigned long long)tier_used(e, moe_TIER_RAM),
                (unsigned long long)ram_budget);
            std::fflush(stderr);
        }
        if (e) {
            trigger_ram_expert_promotion(e);
        }
        return bytes > 0 ? 1 : 0;
    }
    std::fprintf(stderr, "[storagellm] common raw device pin start: budget=%llu\n",
        (unsigned long long)budget);
    std::fflush(stderr);
    bytes += prefetch_raw_layer_roles(e, norm_layer, moe_RAW_ROLE_NORM, moe_DEVICE_POOL_FIXED, budget - bytes);
    bytes += prefetch_raw_layer_roles(e, lm_head_layer, moe_RAW_ROLE_LM_HEAD, moe_DEVICE_POOL_FIXED, budget - bytes);
    if (bytes < budget) {
        bytes += prefetch_raw_layer_roles(e, -1, moe_RAW_ROLE_EMBED, moe_DEVICE_POOL_FIXED, budget - bytes);
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

static void trigger_ram_expert_promotion(moe_pc_engine* e) {
    if (!e) return;
    const bool was_incomplete = !e->common_vram_upload_complete.exchange(true);
    if (!was_incomplete) return;

    e->vram_used_bytes.fetch_add(e->common_vram_reserved_bytes.load());

    const uint64_t vram_budget = tier_budget(e, moe_TIER_VRAM);
    const uint64_t vram_used = tier_used(e, moe_TIER_VRAM);
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
                    const moe_block_state& state = e->expert_states[slot];
                    if (state.active && state.tier == moe_TIER_RAM && !state.pending_evict) {
                        ram_experts.push_back({slot, state.hot_score});
                    }
                }
            }
            std::sort(ram_experts.begin(), ram_experts.end(),
                [](const auto& a, const auto& b) { return a.second > b.second; });
            const uint32_t promote_count = std::min<uint32_t>(max_promote, (uint32_t)ram_experts.size());
            for (uint32_t i = 0; i < promote_count; ++i) {
                int layer = 0, expert = 0;
                expert_dense_slot_decode(e, ram_experts[i].first, &layer, &expert);
                const uint64_t bytes = expert_triplet_bytes(e, layer, expert);
                if (bytes > 0) {
                    moe_pc_engine_request_expert(e, layer, expert, bytes, moe_TIER_VRAM);
                }
            }
        }
    }
}

static int prefetch_common_raw_storage(moe_pc_engine* e) {
    int ok = prefetch_fixed_core_raw(e);
    if (!e) {
        return ok;
    }
    if (!ok) {
        e->common_vram_reserved_bytes.store(0, std::memory_order_release);
    }
    trigger_ram_expert_promotion(e);
    return ok;
}

int moe_pc_engine_prefetch_common_raw(moe_pc_engine_t* engine) {
    return prefetch_common_raw_storage(engine);
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

static uint32_t runtime_prefetch_window(const moe_pc_engine* engine) {
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

static int prefetch_moe_activation_raw(moe_pc_engine* e, int layer) {
    if (!e || layer < 0) {
        return 0;
    }
    // BUGFIX 89: Issue 13 - Optimize last_activation_layer CAS loop ★
    // Problem: CAS loop when multiple workers call prefetch_moe_activation_raw
    // Contention when layer changes frequently (every token in some models)
    // Solution: Early exit if layer already set (no CAS needed)
    // Only CAS when layer actually needs to change
    int previous_layer = e->last_activation_layer.load(std::memory_order_acquire);

    // Fast path: layer already set, no CAS needed
    if (previous_layer == layer) {
        return 1;
    }

    // Slow path: CAS to update layer
    while (previous_layer != layer) {
        if (e->last_activation_layer.compare_exchange_weak(
                previous_layer,
                layer,
                std::memory_order_acq_rel,
                std::memory_order_acquire)) {
            break;
        }
        // After CAS failure, check if another thread already set it
        if (previous_layer == layer) {
            return 1;
        }
    }
    const moe_storage_constants_t* c = moe_storage_constants();
    const int is_moe = c && layer >= (int)c->first_moe_layer && layer <= (int)c->last_moe_layer;
    const uint32_t roles = moe_RAW_ROLE_NORM | moe_RAW_ROLE_ATTENTION;
    const int ok = prefetch_raw_layer_roles(
        e,
        layer,
        roles,
        moe_DEVICE_POOL_ACTIVATION,
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

static void release_moe_activation_raw(moe_pc_engine* e) {
    e->last_activation_layer.store(-1);
    device_free_pool(e, moe_DEVICE_POOL_ACTIVATION);
    device_free_pool(e, moe_DEVICE_POOL_STREAM);
}

static void prefetch_moe_shared_compute_raw(moe_pc_engine* e, int layer) {
    prefetch_raw_layer_roles(
        e,
        layer,
        moe_RAW_ROLE_SHARED,
        moe_DEVICE_POOL_STREAM,
        raw_phase_budget_left(e),
        1
    );
}
