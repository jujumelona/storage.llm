// Device tensor copy operations
#include <cassert>

static uint64_t device_key_tensor(const moe_tensor_record& r) {
    assert((r.info.layer & ~0xFFF) == 0 && "layer exceeds 12-bit range");
    assert((r.info.expert & ~0xFFF) == 0 && "expert exceeds 12-bit range");
    assert((r.info.part & ~0xFF) == 0 && "part exceeds 8-bit range");
    assert((r.info.shard & ~0xFFFFFF) == 0 && "shard exceeds 24-bit range");
    return (3ull << 60) |
        (((uint64_t)r.info.layer & 0xFFFull) << 48) |
        (((uint64_t)r.info.expert & 0xFFFull) << 36) |
        (((uint64_t)r.info.proj & 0xFull) << 32) |
        (((uint64_t)r.info.part & 0xFFull) << 24) |
        ((uint64_t)r.info.shard & 0xFFFFFFull);
}

static uint64_t device_key_expert_bundle(const moe_tensor_record* const recs[3]) {
    if (!recs || !recs[0]) {
        return 0;
    }
    const moe_tensor_record& r = *recs[0];
    return (4ull << 60) |
        (((uint64_t)r.info.layer & 0xFFFull) << 48) |
        (((uint64_t)r.info.expert & 0xFFFull) << 36) |
        (((uint64_t)r.resolved_path_index & 0xFFFFull) << 20) |
        ((uint64_t)r.bundle_id & 0xFFFFFull);
}

struct device_vram_token_guard {
    moe_pc_engine* engine = nullptr;
    vram_reservation_token token{};
    bool active = false;

    device_vram_token_guard(moe_pc_engine* e, const vram_reservation_token& t)
        : engine(e), token(t), active(t.valid) {}

    ~device_vram_token_guard() {
        if (active) {
            device_vram_release_reservation_token(engine, token);
        }
    }

    void disarm() {
        active = false;
    }

    void complete(uint64_t key) {
        if (active) {
            device_vram_complete_reservation_token(engine, token, key);
            active = false;
        }
    }
};

static int device_resolve_vram_token(
    moe_pc_engine* e,
    uint8_t pool,
    uint64_t bytes,
    const vram_reservation_token& provided,
    vram_reservation_token* out
) {
    if (!out) {
        return 0;
    }
    *out = provided;
    if (pool != moe_DEVICE_POOL_EXPERT) {
        return 1;
    }
    if (provided.valid && provided.pool == pool && provided.bytes >= bytes) {
        return 1;
    }
    if (provided.valid) {
        device_vram_release_reservation_token(e, provided);
        *out = vram_reservation_token{};
    }
    return 0;
}

static int device_resolve_bundle_vram_token(
    moe_pc_engine* e,
    uint8_t pool,
    uint64_t bytes,
    const vram_reservation_token& provided,
    vram_reservation_token* out
) {
    if (!out || !e || bytes == 0 || pool != moe_DEVICE_POOL_EXPERT) {
        return 0;
    }
    *out = vram_reservation_token{};
    if (provided.valid) {
        if (provided.pool != pool) {
            device_vram_release_reservation_token(e, provided);
            return 0;
        }
        if (provided.bytes >= bytes) {
            *out = provided;
            out->bytes = bytes;
            if (provided.bytes > bytes) {
                vram_reservation_token rest = provided;
                rest.bytes -= bytes;
                device_vram_release_reservation_token(e, rest);
            }
            return 1;
        }
        vram_reservation_token extra{};
        if (!device_vram_try_reserve_token(e, bytes - provided.bytes, pool, &extra)) {
            device_vram_release_reservation_token(e, provided);
            return 0;
        }
        *out = provided;
        out->bytes = bytes;
        return 1;
    }
    return device_vram_try_reserve_token(e, bytes, pool, out);
}

static void device_finish_prepared_copy_success(
    moe_pc_engine* e,
    uint64_t key,
    device_vram_token_guard& token_guard,
    const vram_reservation_token& effective_token
) {
    device_complete_pending_copy_key(e, key);
    if (effective_token.valid) {
        token_guard.complete(key);
    } else {
        device_vram_mark_allocation_resident(e, key);
    }
}

static void device_finish_prepared_copy_failure(moe_pc_engine* e, uint64_t key) {
    device_cancel_pending_copy(e, key);
    device_free_key(e, key);
}

static size_t device_drain_directstorage_device_copies(moe_pc_engine* e) {
    if (!e) {
        return 0;
    }
    size_t drained = 0;
    std::lock_guard<std::mutex> lock(e->directstorage_pending_mutex);
    auto& pending = e->directstorage_pending_device_copies;
    for (size_t i = 0; i < pending.size();) {
        moe_directstorage_device_pending& copy = pending[i];
        const int state = directstorage_poll_pending_copy(e, &copy.io);
        if (state == 0) {
            ++i;
            continue;
        }
        for (uint8_t j = 0; j < copy.count; ++j) {
            if (state > 0) {
                device_complete_pending_copy_key(e, copy.keys[j]);
                if (copy.tokens[j].valid) {
                    device_vram_complete_reservation_token(e, copy.tokens[j], copy.keys[j]);
                    copy.tokens[j].valid = false;
                } else if (copy.mark_resident[j]) {
                    device_vram_mark_allocation_resident(e, copy.keys[j]);
                }
                device_touch_allocation(e, copy.keys[j]);
            } else {
                device_finish_prepared_copy_failure(e, copy.keys[j]);
                if (copy.tokens[j].valid) {
                    device_vram_release_reservation_token(e, copy.tokens[j]);
                    copy.tokens[j].valid = false;
                }
            }
        }
        pending[i] = pending.back();
        pending.pop_back();
        e->directstorage_pending_count.fetch_sub(1, std::memory_order_acq_rel);
        ++drained;
    }
    return drained;
}

static bool device_range_4k_aligned(uint64_t offset, uint64_t size) {
    return ((offset | size) & 4095ull) == 0;
}

static void gds_blocking_copy_finish_failure(moe_pc_engine* e, const moe_gds_blocking_copy& copy) {
    if (!e) {
        return;
    }
    if (copy.kind == moe_GDS_BLOCKING_BUNDLE) {
        for (uint32_t i = 0; i < copy.view_count && i < 3u; ++i) {
            device_finish_prepared_copy_failure(e, copy.view_keys[i]);
        }
    }
    device_finish_prepared_copy_failure(e, copy.key);
    if (copy.token.valid) {
        device_vram_release_reservation_token(e, copy.token);
    }
}

static void gds_blocking_copy_finish_success(moe_pc_engine* e, const moe_gds_blocking_copy& copy) {
    if (!e) {
        return;
    }
    device_complete_pending_copy_key(e, copy.key);
    if (copy.token.valid) {
        device_vram_complete_reservation_token(e, copy.token, copy.key);
    } else {
        device_vram_mark_allocation_resident(e, copy.key);
    }
    device_touch_allocation(e, copy.key);
    if (copy.kind == moe_GDS_BLOCKING_BUNDLE) {
        for (uint32_t i = 0; i < copy.view_count && i < 3u; ++i) {
            device_complete_pending_copy_key(e, copy.view_keys[i]);
            device_touch_allocation(e, copy.view_keys[i]);
        }
    }
    e->io_atomic.bytes_direct_to_gpu.fetch_add(copy.bytes, std::memory_order_relaxed);
}

static void gds_blocking_worker_loop(moe_pc_engine* e) {
    while (e) {
        moe_gds_blocking_copy copy{};
        {
            std::unique_lock<std::mutex> lock(e->gds_blocking_mutex);
            e->gds_blocking_cv.wait(lock, [&] {
                return e->io_stop.load(std::memory_order_acquire) ||
                       !e->gds_blocking_queue.empty();
            });
            if (e->io_stop.load(std::memory_order_acquire) &&
                e->gds_blocking_queue.empty()) {
                break;
            }
            copy = e->gds_blocking_queue.front();
            e->gds_blocking_queue.pop_front();
        }

        int ok = 1;
        uint64_t dst_offset = 0;
        const uint64_t read_start_us = moe_profile_now_us();
        void* dst_base = reinterpret_cast<void*>(static_cast<uintptr_t>(copy.dst));
        for (uint32_t i = 0; i < copy.range_count && i < 3u; ++i) {
            const uint64_t file_offset = copy.ranges[i][0];
            const uint64_t length = copy.ranges[i][1];
            if (length == 0) {
                continue;
            }
            const long long got = e->cuda.cuFileRead(
                copy.file_handle,
                dst_base,
                static_cast<size_t>(length),
                static_cast<int64_t>(file_offset),
                static_cast<int64_t>(dst_offset));
            if (got != static_cast<long long>(length)) {
                ok = 0;
                break;
            }
            dst_offset += length;
        }
        const uint64_t read_end_us = moe_profile_now_us();
        const uint64_t read_wait_ns = read_end_us > read_start_us ?
            (read_end_us - read_start_us) * 1000ull : 0ull;
        if (ok && dst_offset == copy.bytes) {
            moe_io_path_perf_complete(e->gds_perf, read_wait_ns);
            gds_blocking_copy_finish_success(e, copy);
        } else {
            e->runtime_gpudirect_ready.store(false, std::memory_order_release);
            e->io_atomic.device_copy_failures.fetch_add(1, std::memory_order_relaxed);
            moe_io_path_perf_fail(e->gds_perf, read_wait_ns);
            gds_blocking_copy_finish_failure(e, copy);
        }
        e->gds_blocking_pending.fetch_sub(1, std::memory_order_acq_rel);
        e->io_done_cv.notify_all();
    }
}

static int enqueue_gds_blocking_copy(moe_pc_engine* e, const moe_gds_blocking_copy& copy) {
    if (!e || !copy.file_handle || copy.dst == 0 || copy.bytes == 0 || copy.range_count == 0) {
        return 0;
    }
    {
        std::lock_guard<std::mutex> lock(e->gds_blocking_mutex);
        e->gds_blocking_queue.push_back(copy);
        e->gds_blocking_pending.fetch_add(1, std::memory_order_release);
        e->gds_blocking_fallback_count.fetch_add(1, std::memory_order_relaxed);
    }
    e->gds_blocking_cv.notify_one();
    return 1;
}

static int expert_bundle_is_contiguous_4k_aligned(
    const moe_tensor_record* const recs[3],
    uint64_t* out_bundle_offset,
    uint64_t* out_bundle_size
) {
    if (!recs || !recs[0] || !recs[1] || !recs[2]) {
        return 0;
    }
    const moe_tensor_record& a = *recs[0];
    if (!a.bundle_native ||
        a.bundle_size == 0 ||
        !device_range_4k_aligned(a.bundle_offset, a.bundle_size)) {
        return 0;
    }
    if (a.bundle_alignment && a.bundle_alignment != 4096ull) {
        return 0;
    }
    for (uint32_t i = 1; i < 3u; ++i) {
        const moe_tensor_record& r = *recs[i];
        if (!r.bundle_native ||
            r.resolved_path_index != a.resolved_path_index ||
            r.bundle_id != a.bundle_id ||
            r.bundle_offset != a.bundle_offset ||
            r.bundle_size != a.bundle_size ||
            (r.bundle_alignment && r.bundle_alignment != 4096ull)) {
            return 0;
        }
    }
    for (uint32_t i = 0; i < 3u; ++i) {
        const uint64_t bytes = tensor_stream_bytes(*recs[i]);
        if (bytes == 0 ||
            recs[i]->bundle_member_offset > a.bundle_size ||
            bytes > a.bundle_size - recs[i]->bundle_member_offset) {
            return 0;
        }
    }
    if (out_bundle_offset) {
        *out_bundle_offset = a.bundle_offset;
    }
    if (out_bundle_size) {
        *out_bundle_size = a.bundle_size;
    }
    return 1;
}

static void device_release_bundle_views_locked(
    moe_pc_engine* e,
    const uint64_t keys[3],
    uint64_t owner_key,
    std::vector<std::pair<CUdeviceptr, uint64_t>>* deferred_free_ptrs
) {
    if (!e) {
        return;
    }
    for (uint32_t i = 0; i < 3u; ++i) {
        auto it = e->device_allocations.find(keys[i]);
        if (it != e->device_allocations.end() &&
            it->second.is_view &&
            it->second.owner_key == owner_key) {
            device_release_allocation_locked(e, it, deferred_free_ptrs);
        }
    }
}

static int device_prepare_expert_bundle_allocation(
    moe_pc_engine* e,
    const moe_tensor_record* const recs[3],
    uint8_t pool,
    const vram_reservation_token& reservation,
    uint64_t* out_owner_key,
    CUdeviceptr* out_owner_ptr,
    uint64_t* out_bundle_size,
    vram_reservation_token* out_owner_token
) {
    if (out_owner_key) *out_owner_key = 0;
    if (out_owner_ptr) *out_owner_ptr = 0;
    if (out_bundle_size) *out_bundle_size = 0;
    if (out_owner_token) *out_owner_token = vram_reservation_token{};
    if (!e || !out_owner_key || !out_owner_ptr || !out_bundle_size || !out_owner_token) {
        if (reservation.valid) {
            device_vram_release_reservation_token(e, reservation);
        }
        return 0;
    }
    uint64_t bundle_offset = 0;
    uint64_t bundle_size = 0;
    if (!expert_bundle_is_contiguous_4k_aligned(recs, &bundle_offset, &bundle_size)) {
        if (reservation.valid) {
            device_vram_release_reservation_token(e, reservation);
        }
        return 0;
    }
    (void)bundle_offset;
    vram_reservation_token owner_token{};
    if (!device_resolve_bundle_vram_token(e, pool, bundle_size, reservation, &owner_token)) {
        return 0;
    }

    const uint64_t owner_key = device_key_expert_bundle(recs);
    CUdeviceptr owner_ptr = 0;
    if (!device_prepare_tensor_allocation(e, owner_key, bundle_size, pool, 0, &owner_ptr, nullptr, owner_token)) {
        device_vram_release_reservation_token(e, owner_token);
        return 0;
    }

    uint64_t keys[3]{};
    std::vector<std::pair<CUdeviceptr, uint64_t>> deferred_free_ptrs;
    int ok = 1;
    {
        std::lock_guard<std::mutex> lock(e->device_mutex);
        auto owner_it = e->device_allocations.find(owner_key);
        if (owner_it == e->device_allocations.end() ||
            owner_it->second.pending_free ||
            owner_it->second.ptr == 0 ||
            owner_it->second.bytes < bundle_size ||
            owner_it->second.is_view) {
            ok = 0;
        } else {
            uint32_t new_view_count = 0;
            for (uint32_t i = 0; i < 3u; ++i) {
                const moe_tensor_record& r = *recs[i];
                const uint64_t key = device_key_tensor(r);
                const uint64_t bytes = tensor_stream_bytes(r);
                keys[i] = key;
                if (bytes == 0 ||
                    r.bundle_member_offset > bundle_size ||
                    bytes > bundle_size - r.bundle_member_offset) {
                    ok = 0;
                    break;
                }
                auto old = e->device_allocations.find(key);
                if (old != e->device_allocations.end()) {
                    if (old->second.inflight_copies != 0) {
                        ok = 0;
                        break;
                    }
                    if (old->second.is_view && old->second.owner_key == owner_key) {
                        old->second.bytes = bytes;
                        old->second.pool = pool;
                        old->second.ptr = owner_it->second.backend_kind == moe_DEVICE_ALLOC_D3D12_RESOURCE
                            ? owner_it->second.ptr
                            : owner_it->second.ptr + r.bundle_member_offset;
                        old->second.view_offset = r.bundle_member_offset;
                        old->second.inflight_copies = 1;
                        old->second.pending_free = false;
                        old->second.resident_accounted = false;
                        old->second.backend_kind = owner_it->second.backend_kind;
                        old->second.backend_aux = owner_key;
                        ++new_view_count;
                        continue;
                    } else {
                        device_release_allocation_locked(e, old, &deferred_free_ptrs);
                        owner_it = e->device_allocations.find(owner_key);
                        if (owner_it == e->device_allocations.end()) {
                            ok = 0;
                            break;
                        }
                    }
                }
                moe_device_allocation view{};
                view.key = key;
                view.bytes = bytes;
                view.pool = pool;
                view.ptr = owner_it->second.backend_kind == moe_DEVICE_ALLOC_D3D12_RESOURCE
                    ? owner_it->second.ptr
                    : owner_it->second.ptr + r.bundle_member_offset;
                view.owner_key = owner_key;
                view.view_offset = r.bundle_member_offset;
                view.inflight_copies = 1;
                view.pending_free = false;
                view.resident_accounted = false;
                view.is_view = true;
                view.backend_kind = owner_it->second.backend_kind;
                view.backend_aux = owner_key;
                e->device_allocations.emplace(key, view);
                ++new_view_count;
            }
            if (ok) {
                owner_it = e->device_allocations.find(owner_key);
                if (owner_it != e->device_allocations.end()) {
                    owner_it->second.bundle_view_ref_count = new_view_count;
                } else {
                    ok = 0;
                }
            }
        }
        if (!ok) {
            device_release_bundle_views_locked(e, keys, owner_key, &deferred_free_ptrs);
        }
    }
    device_free_deferred_ptrs(e, deferred_free_ptrs);
    if (!ok) {
        device_finish_prepared_copy_failure(e, owner_key);
        device_vram_release_reservation_token(e, owner_token);
        return 0;
    }

    *out_owner_key = owner_key;
    *out_owner_ptr = owner_ptr;
    *out_bundle_size = bundle_size;
    *out_owner_token = owner_token;
    return 1;
}

// Fix 49: _locked version ??caller already holds device_mutex
static void device_free_expert_locked(
    moe_pc_engine* e,
    int layer,
    int expert,
    std::vector<std::pair<CUdeviceptr, uint64_t>>* deferred_free_ptrs
) {
    if (!e || e->device_allocator_ready.load() <= 0) {
        return;
    }
    const moe_projection_t projs[3] = {moe_PROJ_GATE, moe_PROJ_UP, moe_PROJ_DOWN};
    for (moe_projection_t proj : projs) {
        const moe_tensor_record* r = find_tensor_record(e, layer, expert, proj);
        if (r) {
            device_free_locked(e, device_key_tensor(*r), deferred_free_ptrs);
        }
    }
}

// Fix 49: Public version ??acquires device_mutex then delegates
static void device_free_expert(moe_pc_engine* e, int layer, int expert) {
    if (!e || e->device_allocator_ready.load() <= 0) {
        return;
    }
    std::vector<std::pair<CUdeviceptr, uint64_t>> deferred_free_ptrs;
    {
        std::lock_guard<std::mutex> lock(e->device_mutex);
        device_free_expert_locked(e, layer, expert, &deferred_free_ptrs);
    }
    device_free_deferred_ptrs(e, deferred_free_ptrs);
}

static void device_free_experts_batched(
    moe_pc_engine* e,
    const std::vector<std::pair<int, int>>& experts,
    size_t batch_size = 8
) {
    if (!e || e->device_allocator_ready.load() <= 0 || experts.empty()) {
        return;
    }
    if (batch_size == 0) {
        batch_size = 1;
    }
    for (size_t i = 0; i < experts.size(); i += batch_size) {
        std::vector<std::pair<CUdeviceptr, uint64_t>> deferred_free_ptrs;
        {
            std::lock_guard<std::mutex> lock(e->device_mutex);
            const size_t end = std::min(i + batch_size, experts.size());
            for (size_t j = i; j < end; ++j) {
                device_free_expert_locked(e, experts[j].first, experts[j].second, &deferred_free_ptrs);
            }
        }
        device_free_deferred_ptrs(e, deferred_free_ptrs);
    }
}

static int device_copy_tensor_from_host(
    moe_pc_engine* e,
    const moe_tensor_record& r,
    uint8_t pool,
    const uint8_t* host,
    uint64_t host_bytes,
    uint32_t staging_slot = UINT32_MAX,
    int release_slot_on_submit = 1,
    const vram_reservation_token& reservation = vram_reservation_token{},
    int allow_sync_fallback = 0
) {
    const uint64_t bytes = tensor_stream_bytes(r);
    if (!e || !host || host_bytes < bytes || bytes == 0 ||
        !device_allocator_init(e) || !device_set_current(e)) {
        return 0;
    }
    // Check staging_slot_size > 0 if staging_slot is provided
    if (staging_slot != UINT32_MAX && !staging_slot_valid(e, staging_slot)) {
        fprintf(stderr,
            "[storageLLM ERROR] device_copy_tensor_from_host: staging_slot_size=0, "
            "slots are invalid after reset timeout. Rejecting copy.\n");
        return 0;
    }
    CUstream stream = device_tensor_stream_current(e);
    bool is_host_pinned = true;
    if (staging_slot != UINT32_MAX && staging_slot_valid(e, staging_slot)) {
        is_host_pinned = e->staging_slots[staging_slot].pinned;
    }
    if (!is_host_pinned) {
        e->io_atomic.nonpinned_fallback_count.fetch_add(1, std::memory_order_relaxed);
    }
    const int async_copy = is_host_pinned && device_backend_async_h2d_available(e, stream, 1);
    if (!async_copy && e->backend_caps.supports_backend_async_api) {
        e->io_atomic.backend_async_fallback.fetch_add(bytes, std::memory_order_relaxed);
    }
    if (!async_copy && moe_perf_gpu_profile_requires_pinned_host(e->backend_caps.backend)) {
        e->io_atomic.bytes_sync_h2d_fallback.fetch_add(bytes, std::memory_order_relaxed);
        e->io_atomic.device_copy_failures.fetch_add(1, std::memory_order_relaxed);
        std::fprintf(stderr,
            "[storageLLM STRICT ERROR] tensor H2D requires async pinned/device-direct copy; "
            "refusing sync fallback layer=%u expert=%u proj=%u bytes=%llu pinned=%d\n",
            r.info.layer,
            r.info.expert,
            (unsigned)r.info.proj,
            (unsigned long long)bytes,
            is_host_pinned ? 1 : 0);
        if (staging_slot != UINT32_MAX && release_slot_on_submit) {
            release_staging_slot(e, staging_slot);
        }
        if (reservation.valid) {
            device_vram_release_reservation_token(e, reservation);
        }
        return 0;
    }
    if (!async_copy && !allow_sync_fallback) {
        if (staging_slot != UINT32_MAX && release_slot_on_submit) {
            release_staging_slot(e, staging_slot);
        }
        if (reservation.valid) {
            device_vram_release_reservation_token(e, reservation);
        }
        return 0;
    }
    const uint64_t key = device_key_tensor(r);
    vram_reservation_token effective_token{};
    if (!device_resolve_vram_token(e, pool, bytes, reservation, &effective_token)) {
        if (staging_slot != UINT32_MAX && release_slot_on_submit) {
            release_staging_slot(e, staging_slot);
        }
        return 0;
    }
    device_vram_token_guard token_guard(e, effective_token);
    CUdeviceptr dst = 0;
    if (!device_prepare_tensor_allocation(e, key, bytes, pool, async_copy, &dst, host, effective_token)) {
        if (staging_slot != UINT32_MAX && release_slot_on_submit) {
            release_staging_slot(e, staging_slot);
        }
        return 0;
    }

    moe_io_path_perf_submit(e->pinned_perf, bytes);
    const uint64_t copy_start_us = moe_profile_now_us();
    const int ok = async_copy ?
        device_backend_copy_h2d_async(e, dst, host, bytes, stream) :
        device_backend_copy_h2d_sync(e, dst, host, bytes);
    const uint64_t copy_end_us = moe_profile_now_us();
    const uint64_t copy_submit_ns = copy_end_us > copy_start_us ?
        (copy_end_us - copy_start_us) * 1000ull : 0ull;
    if (!ok) {
        moe_io_path_perf_fail(e->pinned_perf, copy_submit_ns);
        e->io_atomic.device_copy_failures.fetch_add(1, std::memory_order_relaxed);
        if (async_copy) {
            device_backend_stream_sync(e, stream);
        }
        device_finish_prepared_copy_failure(e, key);
        if (staging_slot != UINT32_MAX && release_slot_on_submit) {
            release_staging_slot(e, staging_slot);
        }
        return 0;
    }

    // Record the slot event before release so the next acquirer can fence correctly.
    // Problem: "record event, release slot" ordering can race with a new acquirer.
    // Solution: Set event_recorded flag atomically AFTER successful record
    // Acquirer checks event_recorded before waiting.
    // Impact: Eliminates race condition, ensures DMA fence correctness
    bool event_ok = false;
    if (async_copy) {
        uint32_t slot_for_defer = UINT32_MAX;
        if (staging_slot != UINT32_MAX && staging_slot_valid(e, staging_slot)) {
            auto& slot = e->staging_slots[staging_slot];
            const bool uses_submit_fence =
                e->backend_caps.backend == moe_BACKEND_LEVEL_ZERO ||
                e->backend_caps.backend == moe_BACKEND_VULKAN;
            // Record fence event on the slot itself
            if (!uses_submit_fence && !slot.last_copy_event) {
                slot.last_copy_event = device_take_copy_event(e);
            }
            if (!uses_submit_fence && slot.last_copy_event) {
                event_ok = device_backend_event_record(e, slot.last_copy_event, stream) != 0;
                slot.last_copy_stream = event_ok ? stream : nullptr;
                slot.last_copy_event_recorded.store(event_ok, std::memory_order_release);
            }
            // Release immediately only when the slot has its own DMA fence.
            if (event_ok && release_slot_on_submit) {
                release_staging_slot(e, staging_slot);
            } else {
                slot_for_defer = staging_slot;
            }
        }
        if (device_defer_staging_copy(
                e,
                slot_for_defer,
                key,
                bytes,
                effective_token.valid ? bytes : 0,
                stream,
                effective_token)) {
            token_guard.disarm();
        } else {
            const int sync_ok = device_backend_stream_sync(e, stream);
            if (sync_ok) {
                moe_io_path_perf_complete(e->pinned_perf, copy_submit_ns);
                e->io_atomic.backend_async_completed.fetch_add(bytes, std::memory_order_relaxed);
                device_finish_prepared_copy_success(e, key, token_guard, effective_token);
                if (slot_for_defer != UINT32_MAX) {
                    release_staging_slot(e, slot_for_defer);
                }
            } else {
                moe_io_path_perf_fail(e->pinned_perf, copy_submit_ns);
                e->io_atomic.device_copy_failures.fetch_add(1, std::memory_order_relaxed);
                device_finish_prepared_copy_failure(e, key);
                if (slot_for_defer != UINT32_MAX) {
                    release_staging_slot(e, slot_for_defer);
                }
                return 0;
            }
        }
    } else if (staging_slot != UINT32_MAX && release_slot_on_submit) {
        release_staging_slot(e, staging_slot);
    }
    if (!async_copy) {
        moe_io_path_perf_complete(e->pinned_perf, copy_submit_ns);
        device_finish_prepared_copy_success(e, key, token_guard, effective_token);
    }
    device_touch_allocation(e, key);
    return 1;
}

// Process-global cuFile driver initialization
static std::once_flag g_cufile_open_once;
static std::atomic<int> g_cufile_open_ok{0};

static int device_gpudirect_storage_runtime_probe(moe_pc_engine* e) {
#if defined(__linux__)
    if (!e ||
        e->backend_caps.backend != moe_BACKEND_CUDA ||
        !e->io_config.enable_gpudirect_storage ||
        !compiled_gpudirect_available() ||
        !device_allocator_init(e) ||
        !device_set_current(e) ||
        !e->cuda.cufile_available ||
        !e->cuda.cuFileDriverOpen ||
        !e->cuda.cuFileHandleRegister ||
        !e->cuda.cuFileBufRegister ||
        !e->cuda.cuFileRead) {
        if (e) {
            e->runtime_gpudirect_ready.store(false, std::memory_order_release);
        }
        return 0;
    }
    std::call_once(g_cufile_open_once, [&]() {
        const moe_cufile_error_t st = e->cuda.cuFileDriverOpen();
        g_cufile_open_ok.store(st.err == 0 ? 1 : -1, std::memory_order_release);
    });
    const int ok = g_cufile_open_ok.load(std::memory_order_acquire) > 0;
    e->runtime_gpudirect_ready.store(ok != 0, std::memory_order_release);
    return ok ? 1 : 0;
#else
    if (e) {
        e->runtime_gpudirect_ready.store(false, std::memory_order_release);
    }
    return 0;
#endif
}

#if defined(__linux__)
static void* device_cufile_get_cached_file_handle(moe_pc_engine* e, mmap_context_t* shard) {
    if (!e || !shard || shard->fd < 0 || !e->cuda.cuFileHandleRegister) {
        return nullptr;
    }
#if defined(_WIN32)
    return nullptr;
#else
    struct stat st {};
    if (fstat(shard->fd, &st) != 0) {
        return nullptr;
    }
    const moe_cufile_handle_key cache_key{
        static_cast<uint64_t>(st.st_dev),
        static_cast<uint64_t>(st.st_ino)
    };
    std::lock_guard<std::mutex> lock(e->cufile_cache_mutex);
    auto cached = e->cufile_file_handle_cache.find(cache_key);
    if (cached != e->cufile_file_handle_cache.end()) {
        return cached->second;
    }

    struct moe_cufile_descr_local {
        int type;
        union {
            int fd;
            void* handle;
        } handle;
        const void* fs_ops;
    } desc{};
    desc.type = 1; // CU_FILE_HANDLE_TYPE_OPAQUE_FD
    desc.handle.fd = shard->fd;

    void* file_handle = nullptr;
    const moe_cufile_error_t hst = e->cuda.cuFileHandleRegister(&file_handle, &desc);
    if (hst.err != 0 || !file_handle) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true, std::memory_order_relaxed)) {
            fprintf(stderr,
                "[storageLLM WARNING] cuFileHandleRegister failed (err=%d); "
                "falling back to pinned staging. Check nvidia-fs, filesystem support, and O_DIRECT mode.\n",
                hst.err);
        }
        return nullptr;
    }
    e->cufile_file_handle_cache[cache_key] = file_handle;
    return file_handle;
#endif
}

static int device_cufile_ensure_buffer_registered(moe_pc_engine* e, void* ptr, uint64_t bytes) {
    if (!e || !ptr || bytes == 0 || !e->cuda.cuFileBufRegister) {
        return 0;
    }
    const uint64_t key = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr));
    std::lock_guard<std::mutex> lock(e->cufile_cache_mutex);
    auto cached = e->cufile_buffer_registration_cache.find(key);
    if (cached != e->cufile_buffer_registration_cache.end() && cached->second >= bytes) {
        return 1;
    }
    if (cached != e->cufile_buffer_registration_cache.end()) {
        if (e->cuda.cuFileBufDeregister) {
            e->cuda.cuFileBufDeregister(ptr);
        }
        e->cufile_buffer_registration_cache.erase(cached);
    }
    const moe_cufile_error_t bst = e->cuda.cuFileBufRegister(ptr, (size_t)bytes, 0);
    if (bst.err != 0) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true, std::memory_order_relaxed)) {
            fprintf(stderr,
                "[storageLLM WARNING] cuFileBufRegister failed (err=%d); falling back to pinned staging.\n",
                bst.err);
        }
        return 0;
    }
    e->cufile_buffer_registration_cache[key] = bytes;
    return 1;
}

static int ensure_gds_stream_registered(moe_pc_engine* e, CUstream stream) {
    if (!e || !stream || !e->cuda.cuFileStreamRegister) {
        return 1;
    }
    std::lock_guard<std::mutex> lock(e->gds_stream_cache_mutex);
    if (e->gds_registered_streams.find(stream) != e->gds_registered_streams.end()) {
        return 1;
    }
    const moe_cufile_error_t st = e->cuda.cuFileStreamRegister(stream, 0);
    if (st.err != 0) {
        return 0;
    }
    e->gds_registered_streams.insert(stream);
    return 1;
}

#endif

static int device_copy_tensor_gpudirect_storage(
    moe_pc_engine* e,
    const moe_tensor_record& r,
    uint8_t pool,
    const vram_reservation_token& reservation = vram_reservation_token{},
    int allow_sync_fallback = 0
) {
#if defined(__linux__)
    if (!e || e->backend_caps.backend != moe_BACKEND_CUDA ||
        !e->io_config.enable_gpudirect_storage ||
        !e->cuda.cufile_available ||
        !e->cuda.cuFileHandleRegister ||
        !e->cuda.cuFileHandleDeregister ||
        !e->cuda.cuFileBufDeregister ||
        !e->cuda.cuFileRead) {
        return 0;
    }
    const uint64_t bytes = tensor_stream_bytes(r);
    if (bytes == 0 || !device_allocator_init(e) || !device_set_current(e)) {
        return 0;
    }
    mmap_context_t* shard = get_or_map_tensor_shard(e, r);
    if (!shard || shard->fd < 0) {
        return 0;
    }
    const uint64_t ranges[3][2] = {
        {r.info.weight_byte_offset, r.info.weight_byte_length},
        {r.info.scale_byte_offset, r.info.scale_byte_length},
        {r.info.scale2_byte_offset, r.info.scale2_byte_length}
    };
    const uint64_t file_size = static_cast<uint64_t>(mmap_get_size(shard));
    for (const auto& range : ranges) {
        if (range[1] && (range[0] > file_size || range[1] > file_size - range[0])) {
            return 0;
        }
        if (range[1] && !device_range_4k_aligned(range[0], range[1])) {
            return 0;
        }
    }

    if (!device_gpudirect_storage_runtime_probe(e)) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true, std::memory_order_relaxed)) {
            fprintf(stderr,
                "[storageLLM WARNING] GPUDirect Storage runtime probe failed; falling back to staged H2D\n");
        }
        return 0;
    }

    const uint64_t key = device_key_tensor(r);
    vram_reservation_token effective_token{};
    if (!device_resolve_vram_token(e, pool, bytes, reservation, &effective_token)) {
        return 0;
    }
    device_vram_token_guard token_guard(e, effective_token);
    CUdeviceptr dst = 0;
    if (!device_prepare_tensor_allocation(e, key, bytes, pool, 0, &dst, nullptr, effective_token)) {
        return 0;
    }

    void* file_handle = device_cufile_get_cached_file_handle(e, shard);
    if (!file_handle) {
        e->runtime_gpudirect_ready.store(false, std::memory_order_release);
        device_cancel_pending_copy(e, key);
        device_free_key(e, key);
        return 0;
    }

    int ok = 1;
    uint64_t dst_offset = 0;
    void* dst_base = reinterpret_cast<void*>(static_cast<uintptr_t>(dst));
    if (!device_cufile_ensure_buffer_registered(e, dst_base, bytes)) {
        e->runtime_gpudirect_ready.store(false, std::memory_order_release);
        ok = 0;
    }

    CUstream stream = device_tensor_stream_current(e);
    if (ok && stream && e->cuda.cuFileReadAsync) {
        int stream_ready = ensure_gds_stream_registered(e, stream);
        if (stream_ready) {
            uint64_t async_dst_offset = 0;
            int async_ok = 1;
            moe_io_path_perf_submit(e->gds_perf, bytes);
            e->gds_async_attempted.fetch_add(1, std::memory_order_relaxed);
            for (const auto& range : ranges) {
                if (range[1] == 0) {
                    continue;
                }
                size_t sz = static_cast<size_t>(range[1]);
                int64_t file_off = static_cast<int64_t>(range[0]);
                int64_t buf_off = static_cast<int64_t>(async_dst_offset);
                const moe_cufile_error_t st = e->cuda.cuFileReadAsync(
                    file_handle,
                    dst_base,
                    &sz,
                    &file_off,
                    &buf_off,
                    stream);
                if (st.err != 0) {
                    async_ok = 0;
                    break;
                }
                async_dst_offset += range[1];
            }
            if (async_ok && async_dst_offset == bytes) {
                if (device_defer_staging_copy(
                        e,
                        UINT32_MAX,
                        key,
                        bytes,
                        effective_token.valid ? 0 : bytes,
                        stream,
                        effective_token,
                        moe_IO_PATH_GPUDIRECT_STORAGE)) {
                    e->gds_async_submitted.fetch_add(1, std::memory_order_relaxed);
                    token_guard.disarm();
                    return moe_DEVICE_COPY_PENDING_ASYNC;
                }
                if (!allow_sync_fallback) {
                    device_backend_stream_sync(e, stream);
                    moe_io_path_perf_fail(e->gds_perf, 0);
                    device_finish_prepared_copy_failure(e, key);
                    return 0;
                }
                if (device_backend_stream_sync(e, stream)) {
                    moe_io_path_perf_complete(e->gds_perf, 0);
                    device_finish_prepared_copy_success(e, key, token_guard, effective_token);
                    device_touch_allocation(e, key);
                    return 1;
                }
                moe_io_path_perf_fail(e->gds_perf, 0);
                device_finish_prepared_copy_failure(e, key);
                return 0;
            }
            if (async_dst_offset > 0) {
                device_backend_stream_sync(e, stream);
            }
            moe_io_path_perf_fail(e->gds_perf, 0);
        }
    }

    if (ok && !allow_sync_fallback) {
        moe_io_path_perf_fail(e->gds_perf, 0);
        device_finish_prepared_copy_failure(e, key);
        return 0;
    }

    if (ok) {
        moe_io_path_perf_submit(e->gds_perf, bytes);
        moe_gds_blocking_copy blocking{};
        blocking.kind = moe_GDS_BLOCKING_TENSOR;
        blocking.file_handle = file_handle;
        blocking.dst = dst;
        blocking.bytes = bytes;
        blocking.key = key;
        blocking.token = effective_token;
        for (const auto& range : ranges) {
            if (range[1] == 0) {
                continue;
            }
            if (blocking.range_count >= 3u) {
                ok = 0;
                break;
            }
            blocking.ranges[blocking.range_count][0] = range[0];
            blocking.ranges[blocking.range_count][1] = range[1];
            ++blocking.range_count;
            dst_offset += range[1];
        }
        if (ok && dst_offset == bytes && enqueue_gds_blocking_copy(e, blocking)) {
            token_guard.disarm();
            return moe_DEVICE_COPY_PENDING_ASYNC;
        }
        moe_io_path_perf_fail(e->gds_perf, 0);
        ok = 0;
    }

    if (!ok || dst_offset != bytes) {
        e->io_atomic.device_copy_failures.fetch_add(1, std::memory_order_relaxed);
        e->runtime_gpudirect_ready.store(false, std::memory_order_release);
        device_cancel_pending_copy(e, key);
        device_free_key(e, key);
        return 0;
    }
    e->runtime_gpudirect_ready.store(true, std::memory_order_release);
    device_finish_prepared_copy_success(e, key, token_guard, effective_token);
    device_touch_allocation(e, key);
    return 1;
#else
    (void)e;
    (void)r;
    (void)pool;
    (void)reservation;
    (void)allow_sync_fallback;
    return 0;
#endif
}

static int device_copy_expert_bundle_gpudirect_storage(
    moe_pc_engine* e,
    const moe_tensor_record* const recs[3],
    uint8_t pool,
    const vram_reservation_token& reservation = vram_reservation_token{},
    uint32_t worker_id = UINT32_MAX,
    int allow_sync_fallback = 0
) {
#if defined(__linux__)
    if (!e || e->backend_caps.backend != moe_BACKEND_CUDA ||
        !e->io_config.enable_gpudirect_storage ||
        !e->runtime_gpudirect_ready.load(std::memory_order_acquire) ||
        !e->cuda.cufile_available ||
        !e->cuda.cuFileHandleRegister ||
        !e->cuda.cuFileBufRegister ||
        !e->cuda.cuFileBufDeregister ||
        !e->cuda.cuFileRead ||
        !e->cuda.cuMemAlloc) {
        return 0;
    }
    uint64_t bundle_offset = 0;
    uint64_t bundle_size = 0;
    if (!expert_bundle_is_contiguous_4k_aligned(recs, &bundle_offset, &bundle_size) ||
        !device_allocator_init(e) || !device_set_current(e)) {
        return 0;
    }
    mmap_context_t* shard = get_or_map_tensor_shard(e, *recs[0]);
    if (!shard || shard->fd < 0) {
        return 0;
    }
    const uint64_t file_size = static_cast<uint64_t>(mmap_get_size(shard));
    if (bundle_offset > file_size || bundle_size > file_size - bundle_offset) {
        return 0;
    }

    if (!device_gpudirect_storage_runtime_probe(e)) {
        return 0;
    }

    void* file_handle = device_cufile_get_cached_file_handle(e, shard);
    if (!file_handle) {
        e->runtime_gpudirect_ready.store(false, std::memory_order_release);
        return 0;
    }

    (void)worker_id;
    uint64_t owner_key = 0;
    CUdeviceptr bundle_dst = 0;
    uint64_t prepared_bundle_size = 0;
    vram_reservation_token owner_token{};
    if (!device_prepare_expert_bundle_allocation(
            e,
            recs,
            pool,
            reservation,
            &owner_key,
            &bundle_dst,
            &prepared_bundle_size,
            &owner_token)) {
        return 0;
    }
    if (prepared_bundle_size != bundle_size || !bundle_dst) {
        device_finish_prepared_copy_failure(e, owner_key);
        device_vram_release_reservation_token(e, owner_token);
        return 0;
    }
    void* bundle_ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(bundle_dst));
    if (!device_cufile_ensure_buffer_registered(e, bundle_ptr, bundle_size)) {
        device_finish_prepared_copy_failure(e, owner_key);
        device_vram_release_reservation_token(e, owner_token);
        e->runtime_gpudirect_ready.store(false, std::memory_order_release);
        return 0;
    }

    CUstream stream = device_tensor_stream_current(e);
    if (stream && e->cuda.cuFileReadAsync) {
        int stream_ready = ensure_gds_stream_registered(e, stream);
        if (stream_ready) {
            size_t async_size = static_cast<size_t>(bundle_size);
            int64_t file_off = static_cast<int64_t>(bundle_offset);
            int64_t buf_off = 0;
            moe_io_path_perf_submit(e->gds_perf, bundle_size);
            e->gds_async_attempted.fetch_add(1, std::memory_order_relaxed);
            const moe_cufile_error_t ast = e->cuda.cuFileReadAsync(
                file_handle,
                bundle_ptr,
                &async_size,
                &file_off,
                &buf_off,
                stream);
            if (ast.err == 0) {
                uint64_t view_keys[3]{};
                for (uint32_t i = 0; i < 3u; ++i) {
                    view_keys[i] = device_key_tensor(*recs[i]);
                }
                int deferred_ok = device_defer_staging_copy(
                    e,
                    UINT32_MAX,
                    owner_key,
                    bundle_size,
                    0,
                    stream,
                    owner_token,
                    moe_IO_PATH_GPUDIRECT_STORAGE,
                    view_keys,
                    3u);
                if (deferred_ok) {
                    e->gds_async_submitted.fetch_add(1, std::memory_order_relaxed);
                    return moe_DEVICE_COPY_PENDING_ASYNC;
                }
                if (!allow_sync_fallback) {
                    device_backend_stream_sync(e, stream);
                    for (uint32_t i = 0; i < 3u; ++i) {
                        device_finish_prepared_copy_failure(e, device_key_tensor(*recs[i]));
                    }
                    device_finish_prepared_copy_failure(e, owner_key);
                    device_vram_release_reservation_token(e, owner_token);
                    moe_io_path_perf_fail(e->gds_perf, 0);
                    return moe_DEVICE_COPY_BACKPRESSURE;
                }
                if (device_backend_stream_sync(e, stream)) {
                    moe_io_path_perf_complete(e->gds_perf, 0);
                    device_complete_pending_copy_key(e, owner_key);
                    device_vram_complete_reservation_token(e, owner_token, owner_key);
                    device_touch_allocation(e, owner_key);
                    for (uint32_t i = 0; i < 3u; ++i) {
                        const uint64_t key = device_key_tensor(*recs[i]);
                        device_complete_pending_copy_key(e, key);
                        device_touch_allocation(e, key);
                    }
                    e->io_atomic.bytes_direct_to_gpu.fetch_add(bundle_size, std::memory_order_relaxed);
                    return 1;
                }
                for (uint32_t i = 0; i < 3u; ++i) {
                    device_finish_prepared_copy_failure(e, device_key_tensor(*recs[i]));
                }
                device_finish_prepared_copy_failure(e, owner_key);
                device_vram_release_reservation_token(e, owner_token);
                moe_io_path_perf_fail(e->gds_perf, 0);
                return 0;
            }
            moe_io_path_perf_fail(e->gds_perf, 0);
        }
    }

    if (!allow_sync_fallback) {
        moe_io_path_perf_fail(e->gds_perf, 0);
        for (uint32_t i = 0; i < 3u; ++i) {
            device_finish_prepared_copy_failure(e, device_key_tensor(*recs[i]));
        }
        device_finish_prepared_copy_failure(e, owner_key);
        device_vram_release_reservation_token(e, owner_token);
        return moe_DEVICE_COPY_BACKPRESSURE;
    }

    moe_io_path_perf_submit(e->gds_perf, bundle_size);
    moe_gds_blocking_copy blocking{};
    blocking.kind = moe_GDS_BLOCKING_BUNDLE;
    blocking.file_handle = file_handle;
    blocking.dst = bundle_dst;
    blocking.bytes = bundle_size;
    blocking.key = owner_key;
    blocking.token = owner_token;
    blocking.ranges[0][0] = bundle_offset;
    blocking.ranges[0][1] = bundle_size;
    blocking.range_count = 1;
    for (uint32_t i = 0; i < 3u; ++i) {
        blocking.view_keys[i] = device_key_tensor(*recs[i]);
        blocking.view_count = i + 1u;
    }
    if (enqueue_gds_blocking_copy(e, blocking)) {
        return moe_DEVICE_COPY_PENDING_ASYNC;
    }
    moe_io_path_perf_fail(e->gds_perf, 0);
    for (uint32_t i = 0; i < 3u; ++i) {
        device_finish_prepared_copy_failure(e, device_key_tensor(*recs[i]));
    }
    device_finish_prepared_copy_failure(e, owner_key);
    device_vram_release_reservation_token(e, owner_token);
    return moe_DEVICE_COPY_BACKPRESSURE;
#else
    (void)e;
    (void)recs;
    (void)pool;
    (void)reservation;
    (void)worker_id;
    (void)allow_sync_fallback;
    return 0;
#endif
}

static int device_copy_tensor_directstorage_gpu(
    moe_pc_engine* e,
    const moe_tensor_record& r,
    uint8_t pool,
    const vram_reservation_token& reservation = vram_reservation_token{}
) {
#if defined(_WIN32)
    if (!e ||
        e->backend_caps.backend != moe_BACKEND_DIRECTML ||
        choose_active_io_path(e) != moe_IO_PATH_DIRECTSTORAGE_GPU ||
        !e->runtime_directstorage_gpu_ready.load(std::memory_order_acquire) ||
        !e->io_config.enable_directstorage ||
        !directstorage_direct_gpu_available(e)) {
        return 0;
    }
    const uint64_t bytes = tensor_stream_bytes(r);
    if (bytes == 0 || !device_allocator_init(e)) {
        return 0;
    }
    mmap_context_t* shard = get_or_map_tensor_shard(e, r);
    if (!shard) {
        return 0;
    }
    const uint64_t file_size = static_cast<uint64_t>(mmap_get_size(shard));
    const uint64_t ranges[3][2] = {
        {r.info.weight_byte_offset, r.info.weight_byte_length},
        {r.info.scale_byte_offset, r.info.scale_byte_length},
        {r.info.scale2_byte_offset, r.info.scale2_byte_length}
    };
    for (const auto& range : ranges) {
        if (range[1] && (range[0] > file_size || range[1] > file_size - range[0])) {
            return 0;
        }
    }

    const uint64_t key = device_key_tensor(r);
    vram_reservation_token effective_token{};
    if (!device_resolve_vram_token(e, pool, bytes, reservation, &effective_token)) {
        return 0;
    }
    device_vram_token_guard token_guard(e, effective_token);
    CUdeviceptr dst = 0;
    if (!device_prepare_tensor_allocation(e, key, bytes, pool, 0, &dst, nullptr, effective_token)) {
        return 0;
    }

    void* d3d12_resource = nullptr;
    uint64_t d3d12_resource_offset = 0;
    {
        std::lock_guard<std::mutex> lock(e->device_mutex);
        auto it = e->device_allocations.find(key);
        if (it != e->device_allocations.end() &&
            it->second.backend_kind == moe_DEVICE_ALLOC_D3D12_RESOURCE) {
            d3d12_resource = reinterpret_cast<void*>(static_cast<uintptr_t>(it->second.ptr));
            d3d12_resource_offset = it->second.view_offset;
        }
    }
    if (!d3d12_resource) {
        device_cancel_pending_copy(e, key);
        device_free_key(e, key);
        return 0;
    }

    const std::string path = r.resolved_path_index < e->tensor_paths.size()
        ? e->tensor_paths[r.resolved_path_index] : std::string();
    const std::wstring wide_path = storagellm::wide_path_from_utf8(path);
    moe_dstorage_pending_copy pending{};
    const int ok = !wide_path.empty() &&
        directstorage_submit_ranges_to_gpu_buffer(
            e,
            wide_path.c_str(),
            ranges,
            sizeof(ranges) / sizeof(ranges[0]),
            d3d12_resource,
            d3d12_resource_offset,
            bytes,
            file_size,
            &pending);
    if (!ok) {
        device_finish_prepared_copy_failure(e, key);
        return 0;
    }
    moe_directstorage_device_pending device_pending{};
    device_pending.io = pending;
    device_pending.keys[0] = key;
    device_pending.bytes[0] = bytes;
    device_pending.tokens[0] = effective_token;
    device_pending.mark_resident[0] = !effective_token.valid;
    device_pending.count = 1;
    {
        std::lock_guard<std::mutex> lock(e->directstorage_pending_mutex);
        e->directstorage_pending_device_copies.push_back(device_pending);
        e->directstorage_pending_count.fetch_add(1, std::memory_order_release);
    }
    token_guard.disarm();
    return moe_DEVICE_COPY_PENDING_ASYNC;
#else
    (void)e;
    (void)r;
    (void)pool;
    (void)reservation;
    return 0;
#endif
}

static int device_copy_expert_bundle_directstorage_gpu(
    moe_pc_engine* e,
    const moe_tensor_record* const recs[3],
    uint8_t pool,
    const vram_reservation_token& reservation = vram_reservation_token{},
    uint32_t worker_id = UINT32_MAX
) {
#if defined(_WIN32)
    (void)worker_id;
    if (!e ||
        e->backend_caps.backend != moe_BACKEND_DIRECTML ||
        choose_active_io_path(e) != moe_IO_PATH_DIRECTSTORAGE_GPU ||
        !e->runtime_directstorage_gpu_ready.load(std::memory_order_acquire) ||
        !e->io_config.enable_directstorage ||
        !directstorage_direct_gpu_available(e)) {
        return 0;
    }
    uint64_t bundle_offset = 0;
    uint64_t bundle_size = 0;
    if (!expert_bundle_is_contiguous_4k_aligned(recs, &bundle_offset, &bundle_size) ||
        !device_allocator_init(e)) {
        return 0;
    }
    mmap_context_t* shard = get_or_map_tensor_shard(e, *recs[0]);
    if (!shard) {
        return 0;
    }
    const uint64_t file_size = static_cast<uint64_t>(mmap_get_size(shard));
    if (bundle_offset > file_size || bundle_size > file_size - bundle_offset) {
        return 0;
    }
    const std::string path = recs[0]->resolved_path_index < e->tensor_paths.size()
        ? e->tensor_paths[recs[0]->resolved_path_index] : std::string();
    const std::wstring wide_path = storagellm::wide_path_from_utf8(path);
    if (wide_path.empty()) {
        return 0;
    }

    uint64_t owner_key = 0;
    CUdeviceptr owner_ptr = 0;
    uint64_t prepared_bundle_size = 0;
    vram_reservation_token owner_token{};
    if (!device_prepare_expert_bundle_allocation(
            e,
            recs,
            pool,
            reservation,
            &owner_key,
            &owner_ptr,
            &prepared_bundle_size,
            &owner_token)) {
        return 0;
    }
    if (prepared_bundle_size != bundle_size || !owner_ptr) {
        device_finish_prepared_copy_failure(e, owner_key);
        device_vram_release_reservation_token(e, owner_token);
        return 0;
    }
    void* d3d12_resource = nullptr;
    {
        std::lock_guard<std::mutex> lock(e->device_mutex);
        auto it = e->device_allocations.find(owner_key);
        if (it != e->device_allocations.end() &&
            it->second.backend_kind == moe_DEVICE_ALLOC_D3D12_RESOURCE) {
            d3d12_resource = reinterpret_cast<void*>(static_cast<uintptr_t>(it->second.ptr));
        }
    }
    if (!d3d12_resource) {
        for (uint32_t i = 0; i < 3u; ++i) {
            device_finish_prepared_copy_failure(e, device_key_tensor(*recs[i]));
        }
        device_finish_prepared_copy_failure(e, owner_key);
        device_vram_release_reservation_token(e, owner_token);
        return 0;
    }

    const uint64_t ranges[1][2] = {{bundle_offset, bundle_size}};
    moe_dstorage_pending_copy pending{};
    if (!directstorage_submit_ranges_to_gpu_buffer(
            e,
            wide_path.c_str(),
            ranges,
            1u,
            d3d12_resource,
            0,
            bundle_size,
            file_size,
            &pending)) {
        for (uint32_t i = 0; i < 3u; ++i) {
            device_finish_prepared_copy_failure(e, device_key_tensor(*recs[i]));
        }
        device_finish_prepared_copy_failure(e, owner_key);
        device_vram_release_reservation_token(e, owner_token);
        return 0;
    }
    moe_directstorage_device_pending device_pending{};
    device_pending.io = pending;
    device_pending.keys[0] = owner_key;
    device_pending.bytes[0] = bundle_size;
    device_pending.tokens[0] = owner_token;
    device_pending.mark_resident[0] = false;
    device_pending.count = 1;
    for (uint32_t i = 0; i < 3u; ++i) {
        const uint8_t slot = static_cast<uint8_t>(i + 1u);
        device_pending.keys[slot] = device_key_tensor(*recs[i]);
        device_pending.bytes[slot] = tensor_stream_bytes(*recs[i]);
        device_pending.tokens[slot] = vram_reservation_token{};
        device_pending.mark_resident[slot] = false;
        device_pending.count = static_cast<uint8_t>(slot + 1u);
    }
    {
        std::lock_guard<std::mutex> lock(e->directstorage_pending_mutex);
        e->directstorage_pending_device_copies.push_back(device_pending);
        e->directstorage_pending_count.fetch_add(1, std::memory_order_release);
    }
    return moe_DEVICE_COPY_PENDING_ASYNC;
#else
    (void)e;
    (void)recs;
    (void)pool;
    (void)reservation;
    (void)worker_id;
    return 0;
#endif
}

static int device_copy_tensor(
    moe_pc_engine* e,
    const moe_tensor_record& r,
    uint8_t pool,
    int allow_sync_fallback = 0,
    const vram_reservation_token& reservation = vram_reservation_token{}
) {
    if (!e || tensor_stream_bytes(r) == 0 || !device_allocator_init(e) || !device_set_current(e)) {
        return 0;
    }
    mmap_context_t* shard = get_or_map_tensor_shard(e, r);
    if (!shard || !mmap_get_ptr(shard, 0)) {
        return 0;
    }
    const uint64_t key = device_key_tensor(r);
    const uint64_t bytes = tensor_stream_bytes(r);
    const uint64_t ranges[3][2] = {
        {r.info.weight_byte_offset, r.info.weight_byte_length},
        {r.info.scale_byte_offset, r.info.scale_byte_length},
        {r.info.scale2_byte_offset, r.info.scale2_byte_length}
    };
    const uint64_t file_size = static_cast<uint64_t>(mmap_get_size(shard));
    for (const auto& range : ranges) {
        if (range[1] && (range[0] > file_size || range[1] > file_size - range[0])) {
            return 0;
        }
    }

    vram_reservation_token effective_token{};
    if (!device_resolve_vram_token(e, pool, bytes, reservation, &effective_token)) {
        return 0;
    }
    device_vram_token_guard token_guard(e, effective_token);

    CUstream stream = device_tensor_stream_current(e);
    CUdeviceptr dst = 0;

    // Fix 55: Try staging fast-path ??consolidate 3 ranges into pinned buffer
    // for single true-async DMA instead of 3 false-async non-pinned copies
    // Bug #9: GPU direct uses second half of slot pool (for_pinned = false)
    // PIPELINE FIX 17: Use thread-local turn counter to reduce slot contention ??좎뜦??
    // Old: worker_index=0, turn=0 fixed ??all workers compete for slot 0
    // New: thread-local turn counter ??workers spread across slot pool
    static thread_local uint32_t tl_device_copy_turn = 0;
    // BUGFIX 985: Use staging_slot_valid()-consistent guard ??몄겲 DEFENSIVE
    // Problem: Direct staging_slots.size() check doesn't catch slot.buf==nullptr
    // after reset_staging_slots timeout (staging_slot_size=0 but vector not cleared).
    // staging_slot_valid() enforces correct 3-step check: staging_slot_size > 0,
    // slot_index < vector size, buf != nullptr.
    // Note: We still do the size check first to avoid acquire_staging_slot overhead.
    const bool staging_eligible = e->staging_slot_size > 0 &&
        bytes <= e->staging_slot_size;
    uint32_t slot_idx = staging_eligible
        ? acquire_staging_slot(e, 0u, tl_device_copy_turn++, bytes, false) : UINT32_MAX;

    // If slot acquisition fails, drain completed async copies once and retry.
    if (slot_idx == UINT32_MAX && staging_eligible) {
        device_drain_deferred_staging_copies(e, false);
        slot_idx = acquire_staging_slot(e, 0u, tl_device_copy_turn++, bytes, false);
    }

    if (slot_idx != UINT32_MAX) {
        // Staging fast-path: memcpy ranges ??pinned staging ??single H2D async
        moe_staging_slot& slot = e->staging_slots[slot_idx];
        if (!staging_slot_valid(e, slot_idx)) {
            release_staging_slot(e, slot_idx);
            return 0;
        }
        uint8_t* staging_buf = static_cast<uint8_t*>(slot.buf.load(std::memory_order_acquire));
        if (!staging_buf || bytes > slot.size) {
            release_staging_slot(e, slot_idx);
            return 0;
        }
        // PIPELINE FIX 16: Use AVX2 non-temporal copy instead of plain memcpy ??좎뜦???
        // Old: plain memcpy ??L1/L2 cache pollution, ~1.5-2x slower
        // New: copy_mapped_range_to_staging_linear ??AVX2 streaming stores
        // Matches io_worker performance for mmap??좏릪inned copies
        uint64_t offset = 0;
        int copy_ok = 1;
        for (const auto& range : ranges) {
            if (!range[1]) continue;
            const uint8_t* src = static_cast<const uint8_t*>(mmap_get_ptr(shard, static_cast<size_t>(range[0])));
            if (!src) {
                copy_ok = 0;
                break;
            }
            const uint64_t copied = copy_mapped_range_to_staging_linear(e, src, range[1], slot_idx, &offset);
            if (copied != range[1]) {
                copy_ok = 0;
                break;
            }
        }
        if (!copy_ok) {
            release_staging_slot(e, slot_idx);
            return 0;
        }
        const int async_copy = device_backend_async_h2d_available(
            e, stream, e->staging_slots[slot_idx].pinned ? 1 : 0);
        if (!async_copy && e->backend_caps.supports_backend_async_api) {
            e->io_atomic.backend_async_fallback.fetch_add(bytes, std::memory_order_relaxed);
        }
        if (!device_prepare_tensor_allocation(e, key, bytes, pool, async_copy, &dst, staging_buf, effective_token)) {
            release_staging_slot(e, slot_idx);
            return 0;
        }
        int ok = 0;
        if (async_copy) {
            ok = device_backend_copy_h2d_async(e, dst, staging_buf, bytes, stream);
            // Release slot immediately after async copy starts; completion is deferred.
            // Slot released right after DMA starts, not after completion
        } else {
            ok = device_backend_copy_h2d_sync(e, dst, staging_buf, bytes);
        }
        if (ok && async_copy) {
            // slot??defer????釉??뽱룖 event ?袁⑥┷ ????곸젫
            // BUGFIX ?얜챷??: Pass bytes for VRAM reservation release at completion
            if (device_defer_staging_copy(
                    e,
                    slot_idx,
                    key,
                    bytes,
                    effective_token.valid ? bytes : 0,
                    stream,
                    effective_token)) {
                token_guard.disarm();
            } else {
                const int sync_ok = device_backend_stream_sync(e, stream);
                if (sync_ok) {
                    e->io_atomic.backend_async_completed.fetch_add(bytes, std::memory_order_relaxed);
                    device_finish_prepared_copy_success(e, key, token_guard, effective_token);
                    release_staging_slot(e, slot_idx);
                } else {
                    e->io_atomic.device_copy_failures.fetch_add(1, std::memory_order_relaxed);
                    device_finish_prepared_copy_failure(e, key);
                    release_staging_slot(e, slot_idx);
                    return 0;
                }
            }
            // staging_slot????由????곸젫??? ??놁벉
        } else {
            release_staging_slot(e, slot_idx);
        }
        if (!ok) {
            e->io_atomic.device_copy_failures.fetch_add(1, std::memory_order_relaxed);
            device_finish_prepared_copy_failure(e, key);
            return 0;
        }
        if (async_copy) {
            // Staged async copies are completed by their per-copy event.
        } else {
            device_finish_prepared_copy_success(e, key, token_guard, effective_token);
        }
        device_touch_allocation(e, key);
        return 1;
    }

    // No staging: use explicit synchronous H2D copy.
    if (device_backend_is_metal(e)) {
        token_guard.disarm();
        return moe_DEVICE_COPY_BACKPRESSURE;
    }
    if (!allow_sync_fallback) {
        token_guard.disarm();
        return moe_DEVICE_COPY_BACKPRESSURE;
    }
    if (e) {
        e->io_atomic.bytes_sync_h2d_fallback.fetch_add(bytes);
        if (e->backend_caps.supports_backend_async_api) {
            e->io_atomic.backend_async_fallback.fetch_add(bytes, std::memory_order_relaxed);
        }
    }
    const void* host_ptr = shard ? mmap_get_ptr(shard, static_cast<size_t>(ranges[0][0])) : nullptr;
    if (!device_prepare_tensor_allocation(e, key, bytes, pool, 0 /*sync*/, &dst, host_ptr, effective_token)) {
        return 0;
    }
    CUdeviceptr cur = dst;
    for (const auto& range : ranges) {
        if (!range[1]) continue;
        const void* src = mmap_get_ptr(shard, static_cast<size_t>(range[0]));
        if (!src) {
            device_cancel_pending_copy(e, key);
            device_free_key(e, key);
            return 0;
        }
        // Check cur + range[1] before copy; malformed ranges can overflow CUdeviceptr.
        // Solution: Check that cur + range[1] stays within dst + bytes bounds
        if (range[1] > bytes || cur - dst > bytes - range[1]) {
            e->io_atomic.device_copy_failures.fetch_add(1, std::memory_order_relaxed);
            device_cancel_pending_copy(e, key);
            device_free_key(e, key);
            return 0;
        }
        if (!device_backend_copy_h2d_sync(e, cur, src, range[1])) {
            e->io_atomic.device_copy_failures.fetch_add(1, std::memory_order_relaxed);
            device_cancel_pending_copy(e, key);
            device_free_key(e, key);
            return 0;
        }
        cur += range[1];
    }
#ifdef __linux__
    for (const auto& range : ranges) {
        if (range[1] > 0) {
            void* ptr = mmap_get_ptr(shard, static_cast<size_t>(range[0]));
            if (ptr) madvise(ptr, static_cast<size_t>(range[1]), MADV_COLD);
        }
    }
#endif
    device_finish_prepared_copy_success(e, key, token_guard, effective_token);
    device_touch_allocation(e, key);
    return 1;
}
