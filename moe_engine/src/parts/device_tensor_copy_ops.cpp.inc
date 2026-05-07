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

static bool device_range_4k_aligned(uint64_t offset, uint64_t size) {
    return ((offset | size) & 4095ull) == 0;
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
    const vram_reservation_token& reservation = vram_reservation_token{}
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
    bool is_host_pinned = true; if (staging_slot != UINT32_MAX && staging_slot_valid(e, staging_slot)) { is_host_pinned = e->staging_slots[staging_slot].pinned; } if (!is_host_pinned) { e->io_atomic.nonpinned_fallback_count.fetch_add(1, std::memory_order_relaxed); } const int async_copy = is_host_pinned && device_backend_async_h2d_available(e, stream, 1);
    if (!async_copy && e->backend_caps.supports_backend_async_api) {
        e->io_atomic.backend_async_fallback.fetch_add(bytes, std::memory_order_relaxed);
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

    const int ok = async_copy ?
        device_backend_copy_h2d_async(e, dst, host, bytes, stream) :
        device_backend_copy_h2d_sync(e, dst, host, bytes);
    if (!ok) {
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
                e->io_atomic.backend_async_completed.fetch_add(bytes, std::memory_order_relaxed);
                device_finish_prepared_copy_success(e, key, token_guard, effective_token);
                if (slot_for_defer != UINT32_MAX) {
                    release_staging_slot(e, slot_for_defer);
                }
            } else {
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
        device_finish_prepared_copy_success(e, key, token_guard, effective_token);
    }
    device_touch_allocation(e, key);
    return 1;
}

// Process-global cuFile driver initialization
static std::once_flag g_cufile_open_once;
static std::atomic<int> g_cufile_open_ok{0};

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
#endif

static int device_copy_tensor_gpudirect_storage(
    moe_pc_engine* e,
    const moe_tensor_record& r,
    uint8_t pool,
    const vram_reservation_token& reservation = vram_reservation_token{}
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

    if (e->cuda.cuFileDriverOpen) {
        std::call_once(g_cufile_open_once, [&]() {
            const moe_cufile_error_t st = e->cuda.cuFileDriverOpen();
            g_cufile_open_ok.store(st.err == 0 ? 1 : -1, std::memory_order_release);
        });
        if (g_cufile_open_ok.load(std::memory_order_acquire) < 0) {
            static std::atomic<bool> warned{false};
            if (!warned.exchange(true, std::memory_order_relaxed)) {
                fprintf(stderr,
                    "[storageLLM WARNING] GPUDirect Storage driver open failed; falling back to staged H2D\n");
            }
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

    void* file_handle = device_cufile_get_cached_file_handle(e, shard);
    if (!file_handle) {
        device_cancel_pending_copy(e, key);
        device_free_key(e, key);
        return 0;
    }

    int ok = 1;
    uint64_t dst_offset = 0;
    void* dst_base = reinterpret_cast<void*>(static_cast<uintptr_t>(dst));
    if (!device_cufile_ensure_buffer_registered(e, dst_base, bytes)) {
        ok = 0;
    }

    if (ok) {
        for (const auto& range : ranges) {
            if (range[1] == 0) continue;
            const long long got = e->cuda.cuFileRead(
                file_handle,
                dst_base,
                static_cast<size_t>(range[1]),
                static_cast<int64_t>(range[0]),
                static_cast<int64_t>(dst_offset));
            if (got != static_cast<long long>(range[1])) {
                ok = 0;
                break;
            }
            dst_offset += range[1];
        }
    }

    if (!ok || dst_offset != bytes) {
        e->io_atomic.device_copy_failures.fetch_add(1, std::memory_order_relaxed);
        device_cancel_pending_copy(e, key);
        device_free_key(e, key);
        return 0;
    }
    device_finish_prepared_copy_success(e, key, token_guard, effective_token);
    device_touch_allocation(e, key);
    return 1;
#else
    (void)e;
    (void)r;
    (void)pool;
    return 0;
#endif
}

static int device_copy_tensor(
    moe_pc_engine* e,
    const moe_tensor_record& r,
    uint8_t pool,
    int allow_sync_fallback = 1,
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
