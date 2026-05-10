// Backend-neutral async copy facade.
// CUDA/HIP/Metal keep their native paths. Level Zero uses queue-submit fences.
// Vulkan uses VkBuffer-backed allocations, host-visible staging buffers, and
// queue-submit fences for true async H2D completion.

static int device_backend_copy_h2d_sync(
    moe_pc_engine* e,
    CUdeviceptr dst,
    const void* src,
    uint64_t bytes
);
static int device_backend_event_sync(moe_pc_engine* e, CUevent event);

static int device_backend_is_cuda(const moe_pc_engine* e) {
    return e && e->backend_caps.backend == moe_BACKEND_CUDA && e->cuda.library;
}

static int device_backend_is_hip(const moe_pc_engine* e) {
    return e && e->backend_caps.backend == moe_BACKEND_HIP && e->hip.library;
}

static int device_backend_is_metal(const moe_pc_engine* e) {
    return e && e->backend_caps.backend == moe_BACKEND_METAL;
}

static int device_backend_is_level_zero(const moe_pc_engine* e) {
    return e && e->backend_caps.backend == moe_BACKEND_LEVEL_ZERO && e->ze.library;
}

static int device_backend_is_vulkan(const moe_pc_engine* e) {
    return e && e->backend_caps.backend == moe_BACKEND_VULKAN && e->vk.library;
}

extern "C" {
    void* metal_stream_create(void* device_handle);
    void metal_stream_destroy(void* stream);
    void* metal_event_create();
    void metal_event_destroy(void* event);
    int metal_event_record(void* event, void* stream);
    int metal_event_query(void* event);
    int metal_event_sync(void* event);
    int metal_copy_h2d_async(void* dst_buffer, const void* src, uint64_t bytes, void* stream);
    int metal_copy_h2d_sync(void* dst_buffer, const void* src, uint64_t bytes);
    uint64_t metal_measure_h2d_bandwidth(void* device_handle);
    void* metal_get_default_device();
}

#if !(defined(__APPLE__) && defined(__MACH__))
extern "C" void* metal_stream_create(void*) { return nullptr; }
extern "C" void metal_stream_destroy(void*) {}
extern "C" void* metal_event_create() { return nullptr; }
extern "C" void metal_event_destroy(void*) {}
extern "C" int metal_event_record(void*, void*) { return 0; }
extern "C" int metal_event_query(void*) { return 0; }
extern "C" int metal_event_sync(void*) { return 0; }
extern "C" int metal_copy_h2d_async(void*, const void*, uint64_t, void*) { return 0; }
extern "C" int metal_copy_h2d_sync(void*, const void*, uint64_t) { return 0; }
extern "C" uint64_t metal_measure_h2d_bandwidth(void*) { return 0; }
extern "C" void* metal_get_default_device() { return nullptr; }
#endif

struct moe_level_zero_stream_state {
    std::vector<void*> idle_command_lists;
    void* pending_command_list = nullptr;
};

struct moe_level_zero_copy_event {
    void* fence = nullptr;
    void* command_list = nullptr;
    moe_level_zero_stream_state* owner = nullptr;
    bool submitted = false;
};

struct moe_vulkan_stream_state {
    std::vector<void*> idle_command_buffers;
    void* pending_command_buffer = nullptr;
    void* pending_staging_buffer = nullptr;
    void* pending_staging_memory = nullptr;
    uint64_t pending_bytes = 0;
};

struct moe_vulkan_copy_event {
    void* fence = nullptr;
    void* command_buffer = nullptr;
    void* staging_buffer = nullptr;
    void* staging_memory = nullptr;
    moe_vulkan_stream_state* owner = nullptr;
    bool submitted = false;
};

static int level_zero_async_runtime_ready(const moe_pc_engine* e) {
    return device_backend_is_level_zero(e) &&
        e->ze.context &&
        e->ze.device &&
        e->ze.command_queue &&
        e->ze.zeCommandListCreate &&
        e->ze.zeCommandListDestroy &&
        e->ze.zeCommandListClose &&
        e->ze.zeCommandListReset &&
        e->ze.zeCommandListAppendMemoryCopy &&
        e->ze.zeCommandQueueExecuteCommandLists &&
        e->ze.zeCommandQueueSynchronize &&
        e->ze.zeFenceCreate &&
        e->ze.zeFenceDestroy &&
        e->ze.zeFenceHostSynchronize &&
        e->ze.zeFenceQueryStatus;
}

static void* level_zero_create_command_list(moe_pc_engine* e) {
    if (!level_zero_async_runtime_ready(e)) {
        return nullptr;
    }
    struct ze_command_list_desc_t_local {
        uint32_t stype;
        void* pNext;
        uint32_t commandQueueGroupOrdinal;
        uint32_t flags;
    } desc = {0x00010007, nullptr, 0u, 0u}; // ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC

    void* cmd = nullptr;
    return e->ze.zeCommandListCreate(e->ze.context, e->ze.device, &desc, &cmd) == 0 ? cmd : nullptr;
}

static void level_zero_destroy_command_list(moe_pc_engine* e, void* cmd) {
    if (cmd && e && e->ze.zeCommandListDestroy) {
        e->ze.zeCommandListDestroy(cmd);
    }
}

static void level_zero_recycle_command_list(
    moe_pc_engine* e,
    moe_level_zero_stream_state* owner,
    void* cmd
) {
    if (!cmd) {
        return;
    }
    if (owner && e && e->ze.zeCommandListReset &&
        e->ze.zeCommandListReset(cmd) == 0) {
        owner->idle_command_lists.push_back(cmd);
        return;
    }
    level_zero_destroy_command_list(e, cmd);
}

static void level_zero_recycle_submitted_list(
    moe_pc_engine* e,
    moe_level_zero_copy_event* ev
) {
    if (!ev || !ev->submitted) {
        return;
    }
    void* cmd = ev->command_list;
    moe_level_zero_stream_state* owner = ev->owner;
    ev->command_list = nullptr;
    ev->owner = nullptr;
    ev->submitted = false;
    if (ev->fence && e && e->ze.zeFenceReset) {
        e->ze.zeFenceReset(ev->fence);
    }
    level_zero_recycle_command_list(e, owner, cmd);
}

static void* level_zero_take_command_list(
    moe_pc_engine* e,
    moe_level_zero_stream_state* state
) {
    if (!state || state->pending_command_list) {
        return nullptr;
    }
    if (!state->idle_command_lists.empty()) {
        void* cmd = state->idle_command_lists.back();
        state->idle_command_lists.pop_back();
        return cmd;
    }
    return level_zero_create_command_list(e);
}

static int level_zero_submit_pending_and_sync(
    moe_pc_engine* e,
    moe_level_zero_stream_state* state
) {
    if (!level_zero_async_runtime_ready(e) || !state) {
        return 0;
    }
    if (!state->pending_command_list) {
        e->io_atomic.backend_fence_waits.fetch_add(1, std::memory_order_relaxed);
        e->io_atomic.level_zero_fence_waits.fetch_add(1, std::memory_order_relaxed);
        return e->ze.zeCommandQueueSynchronize(e->ze.command_queue, UINT64_MAX) == 0;
    }

    void* cmd = state->pending_command_list;
    state->pending_command_list = nullptr;
    if (e->ze.zeCommandListClose(cmd) != 0) {
        level_zero_destroy_command_list(e, cmd);
        return 0;
    }
    if (e->ze.zeCommandQueueExecuteCommandLists(e->ze.command_queue, 1, &cmd, nullptr) != 0) {
        level_zero_destroy_command_list(e, cmd);
        return 0;
    }
    e->io_atomic.backend_fence_waits.fetch_add(1, std::memory_order_relaxed);
    e->io_atomic.level_zero_fence_waits.fetch_add(1, std::memory_order_relaxed);
    const int ok = e->ze.zeCommandQueueSynchronize(e->ze.command_queue, UINT64_MAX) == 0;
    level_zero_recycle_command_list(e, state, cmd);
    return ok;
}

static void vulkan_release_staging(
    moe_pc_engine* e,
    void*& buffer,
    void*& memory
) {
    if (buffer || memory) {
        moe_vulkan_destroy_buffer_allocation(e, buffer, memory);
        buffer = nullptr;
        memory = nullptr;
    }
}

static void* vulkan_take_command_buffer(
    moe_pc_engine* e,
    moe_vulkan_stream_state* state
) {
    if (!state || state->pending_command_buffer) {
        return nullptr;
    }
    if (!state->idle_command_buffers.empty()) {
        void* cmd = state->idle_command_buffers.back();
        state->idle_command_buffers.pop_back();
        return cmd;
    }
    return moe_vulkan_create_command_buffer(e);
}

static void vulkan_recycle_command_buffer(
    moe_pc_engine* e,
    moe_vulkan_stream_state* owner,
    void* cmd
) {
    if (!cmd) {
        return;
    }
    if (owner && moe_vulkan_reset_command_buffer(e, cmd)) {
        owner->idle_command_buffers.push_back(cmd);
        return;
    }
    moe_vulkan_destroy_command_buffer(e, cmd);
}

static void vulkan_recycle_submitted_copy(
    moe_pc_engine* e,
    moe_vulkan_copy_event* ev
) {
    if (!ev || !ev->submitted) {
        return;
    }
    void* cmd = ev->command_buffer;
    void* staging_buffer = ev->staging_buffer;
    void* staging_memory = ev->staging_memory;
    moe_vulkan_stream_state* owner = ev->owner;
    ev->command_buffer = nullptr;
    ev->staging_buffer = nullptr;
    ev->staging_memory = nullptr;
    ev->owner = nullptr;
    ev->submitted = false;
    vulkan_release_staging(e, staging_buffer, staging_memory);
    moe_vulkan_reset_fence(e, ev->fence);
    vulkan_recycle_command_buffer(e, owner, cmd);
}

static void vulkan_clear_pending_copy(
    moe_pc_engine* e,
    moe_vulkan_stream_state* state
) {
    if (!state) {
        return;
    }
    void* cmd = state->pending_command_buffer;
    void* staging_buffer = state->pending_staging_buffer;
    void* staging_memory = state->pending_staging_memory;
    state->pending_command_buffer = nullptr;
    state->pending_staging_buffer = nullptr;
    state->pending_staging_memory = nullptr;
    state->pending_bytes = 0;
    vulkan_release_staging(e, staging_buffer, staging_memory);
    vulkan_recycle_command_buffer(e, state, cmd);
}

static int vulkan_submit_pending_and_sync(
    moe_pc_engine* e,
    moe_vulkan_stream_state* state
) {
    if (!moe_vulkan_transfer_runtime_ready(e) || !state) {
        return 0;
    }
    if (!state->pending_command_buffer) {
        return 1;
    }
    void* fence = moe_vulkan_create_fence(e);
    if (!fence) {
        return 0;
    }
    void* cmd = state->pending_command_buffer;
    void* staging_buffer = state->pending_staging_buffer;
    void* staging_memory = state->pending_staging_memory;
    const uint64_t submitted_bytes = state->pending_bytes;
    if (!moe_vulkan_submit_command_buffer(e, cmd, fence)) {
        moe_vulkan_destroy_fence(e, fence);
        return 0;
    }
    state->pending_command_buffer = nullptr;
    state->pending_staging_buffer = nullptr;
    state->pending_staging_memory = nullptr;
    state->pending_bytes = 0;
    if (submitted_bytes > 0) {
        e->io_atomic.backend_async_submitted.fetch_add(
            submitted_bytes, std::memory_order_relaxed);
    }
    e->io_atomic.backend_fence_waits.fetch_add(1, std::memory_order_relaxed);
    e->io_atomic.vulkan_fence_waits.fetch_add(1, std::memory_order_relaxed);
    const int ok = moe_vulkan_wait_fence(e, fence);
    moe_vulkan_reset_fence(e, fence);
    moe_vulkan_destroy_fence(e, fence);
    vulkan_release_staging(e, staging_buffer, staging_memory);
    vulkan_recycle_command_buffer(e, state, cmd);
    return ok;
}

static int device_backend_stream_create(moe_pc_engine* e, CUstream* out_stream, unsigned int flags) {
    (void)flags;
    if (out_stream) {
        *out_stream = nullptr;
    }
    if (!out_stream) {
        return 0;
    }
    if (e && !e->backend_caps.supports_backend_async_api) {
        return 0;
    }
    if (device_backend_is_cuda(e) && e->cuda.cuStreamCreate) {
        return e->cuda.cuStreamCreate(out_stream, flags) == 0;
    }
    if (device_backend_is_hip(e) && e->hip.hipStreamCreate) {
        return e->hip.hipStreamCreate(out_stream, flags) == 0;
    }
    if (device_backend_is_metal(e)) {
        *out_stream = (CUstream)metal_stream_create(nullptr);
        return *out_stream != nullptr;
    }
    if (moe_vulkan_transfer_runtime_ready(e)) {
        auto* state = new moe_vulkan_stream_state();
        *out_stream = (CUstream)state;
        return 1;
    }
    if (level_zero_async_runtime_ready(e)) {
        auto* state = new moe_level_zero_stream_state();
        *out_stream = (CUstream)state;
        return 1;
    }
    return 0;
}

static int device_backend_stream_sync(moe_pc_engine* e, CUstream stream) {
    if (!stream) {
        return 1;
    }
    if (device_backend_is_cuda(e) && e->cuda.cuStreamSynchronize) {
        return e->cuda.cuStreamSynchronize(stream) == 0;
    }
    if (device_backend_is_hip(e) && e->hip.hipStreamSynchronize) {
        return e->hip.hipStreamSynchronize(stream) == 0;
    }
    if (device_backend_is_metal(e)) {
        return 1;
    }
    if (moe_vulkan_transfer_runtime_ready(e)) {
        return vulkan_submit_pending_and_sync(
            e,
            reinterpret_cast<moe_vulkan_stream_state*>(stream));
    }
    if (level_zero_async_runtime_ready(e)) {
        return level_zero_submit_pending_and_sync(
            e,
            reinterpret_cast<moe_level_zero_stream_state*>(stream));
    }
    return 0;
}

static void device_backend_stream_destroy(moe_pc_engine* e, CUstream stream) {
    if (!stream) {
        return;
    }
    if (device_backend_is_cuda(e) && e->cuda.cuStreamDestroy) {
        e->cuda.cuStreamDestroy(stream);
    } else if (device_backend_is_hip(e) && e->hip.hipStreamDestroy) {
        e->hip.hipStreamDestroy(stream);
    } else if (device_backend_is_metal(e)) {
        metal_stream_destroy((void*)stream);
    } else if (device_backend_is_vulkan(e)) {
        auto* state = reinterpret_cast<moe_vulkan_stream_state*>(stream);
        device_backend_stream_sync(e, stream);
        vulkan_clear_pending_copy(e, state);
        for (void* cmd : state->idle_command_buffers) {
            moe_vulkan_destroy_command_buffer(e, cmd);
        }
        delete state;
    } else if (device_backend_is_level_zero(e)) {
        auto* state = reinterpret_cast<moe_level_zero_stream_state*>(stream);
        device_backend_stream_sync(e, stream);
        level_zero_destroy_command_list(e, state->pending_command_list);
        state->pending_command_list = nullptr;
        for (void* cmd : state->idle_command_lists) {
            level_zero_destroy_command_list(e, cmd);
        }
        delete state;
    }
}

static int device_backend_event_create(moe_pc_engine* e, CUevent* out_event) {
    if (out_event) {
        *out_event = nullptr;
    }
    if (!out_event) {
        return 0;
    }
    constexpr unsigned int disable_timing = 2u;
    if (device_backend_is_cuda(e) && e->cuda.cuEventCreate) {
        return e->cuda.cuEventCreate(out_event, disable_timing) == 0;
    }
    if (device_backend_is_hip(e) && e->hip.hipEventCreate) {
        return e->hip.hipEventCreate(out_event, disable_timing) == 0;
    }
    if (device_backend_is_metal(e)) {
        *out_event = (CUevent)metal_event_create();
        return *out_event != nullptr;
    }
    if (moe_vulkan_transfer_runtime_ready(e)) {
        auto* ev = new moe_vulkan_copy_event();
        ev->fence = moe_vulkan_create_fence(e);
        if (ev->fence) {
            *out_event = (CUevent)ev;
            return 1;
        }
        delete ev;
    }
    if (level_zero_async_runtime_ready(e)) {
        auto* ev = new moe_level_zero_copy_event();
        struct ze_fence_desc_t_local {
            uint32_t stype;
            void* pNext;
            uint32_t flags;
        } desc = {0x0001000C, nullptr, 0u}; // ZE_STRUCTURE_TYPE_FENCE_DESC
        if (e->ze.zeFenceCreate(e->ze.command_queue, &desc, &ev->fence) == 0 && ev->fence) {
            *out_event = (CUevent)ev;
            return 1;
        }
        delete ev;
    }
    return 0;
}

static int device_backend_event_record(moe_pc_engine* e, CUevent event, CUstream stream) {
    if (!event || !stream) {
        return 0;
    }
    if (device_backend_is_cuda(e) && e->cuda.cuEventRecord) {
        return e->cuda.cuEventRecord(event, stream) == 0;
    }
    if (device_backend_is_hip(e) && e->hip.hipEventRecord) {
        return e->hip.hipEventRecord(event, stream) == 0;
    }
    if (device_backend_is_metal(e)) {
        return metal_event_record((void*)event, (void*)stream);
    }
    if (moe_vulkan_transfer_runtime_ready(e)) {
        auto* ev = reinterpret_cast<moe_vulkan_copy_event*>(event);
        auto* state = reinterpret_cast<moe_vulkan_stream_state*>(stream);
        if (!ev || !state || !ev->fence || !state->pending_command_buffer) {
            return 0;
        }
        if (ev->submitted && !device_backend_event_sync(e, event)) {
            return 0;
        }
        if (!moe_vulkan_reset_fence(e, ev->fence)) {
            return 0;
        }
        void* cmd = state->pending_command_buffer;
        const uint64_t submitted_bytes = state->pending_bytes;
        if (!moe_vulkan_submit_command_buffer(e, cmd, ev->fence)) {
            return 0;
        }
        state->pending_command_buffer = nullptr;
        ev->command_buffer = cmd;
        ev->staging_buffer = state->pending_staging_buffer;
        ev->staging_memory = state->pending_staging_memory;
        ev->owner = state;
        ev->submitted = true;
        state->pending_staging_buffer = nullptr;
        state->pending_staging_memory = nullptr;
        state->pending_bytes = 0;
        if (submitted_bytes > 0) {
            e->io_atomic.backend_async_submitted.fetch_add(
                submitted_bytes, std::memory_order_relaxed);
        }
        return 1;
    }
    if (level_zero_async_runtime_ready(e)) {
        auto* ev = reinterpret_cast<moe_level_zero_copy_event*>(event);
        auto* state = reinterpret_cast<moe_level_zero_stream_state*>(stream);
        if (!ev || !state || !ev->fence || !state->pending_command_list) {
            return 0;
        }
        if (ev->submitted) {
            if (!device_backend_event_sync(e, event)) {
                return 0;
            }
        }
        if (e->ze.zeFenceReset) {
            e->ze.zeFenceReset(ev->fence);
        }
        void* cmd = state->pending_command_list;
        if (e->ze.zeCommandListClose(cmd) != 0) {
            state->pending_command_list = nullptr;
            level_zero_destroy_command_list(e, cmd);
            return 0;
        }
        if (e->ze.zeCommandQueueExecuteCommandLists(
                e->ze.command_queue, 1, &cmd, ev->fence) != 0) {
            state->pending_command_list = nullptr;
            level_zero_destroy_command_list(e, cmd);
            return 0;
        }
        state->pending_command_list = nullptr;
        ev->command_list = cmd;
        ev->owner = state;
        ev->submitted = true;
        return 1;
    }
    return 0;
}

static int device_backend_event_query(moe_pc_engine* e, CUevent event) {
    if (!event) {
        return 0;
    }
    if (device_backend_is_cuda(e) && e->cuda.cuEventQuery) {
        return e->cuda.cuEventQuery(event) == 0;
    }
    if (device_backend_is_hip(e) && e->hip.hipEventQuery) {
        return e->hip.hipEventQuery(event) == 0;
    }
    if (device_backend_is_metal(e)) {
        return metal_event_query((void*)event);
    }
    if (moe_vulkan_transfer_runtime_ready(e)) {
        auto* ev = reinterpret_cast<moe_vulkan_copy_event*>(event);
        if (!ev || !ev->submitted) {
            return 1;
        }
        if (moe_vulkan_fence_signaled(e, ev->fence)) {
            vulkan_recycle_submitted_copy(e, ev);
            return 1;
        }
        return 0;
    }
    if (level_zero_async_runtime_ready(e)) {
        auto* ev = reinterpret_cast<moe_level_zero_copy_event*>(event);
        if (!ev || !ev->submitted) {
            return 1;
        }
        if (e->ze.zeFenceQueryStatus(ev->fence) == 0) {
            level_zero_recycle_submitted_list(e, ev);
            return 1;
        }
        return 0;
    }
    return 0;
}

static int device_backend_event_sync(moe_pc_engine* e, CUevent event) {
    if (!event) {
        return 1;
    }
    if (device_backend_is_cuda(e)) {
        if (e->cuda.cuEventSynchronize) {
            return e->cuda.cuEventSynchronize(event) == 0;
        }
        if (!e->cuda.cuEventQuery) {
            return 0;
        }
        uint32_t spin_count = 0;
        while (true) {
            if (e->cuda.cuEventQuery(event) == 0) {
                return 1;
            }
            if (e->io_stop.load(std::memory_order_relaxed)) {
                break;
            }
            if (++spin_count < 20) {
                std::this_thread::yield();
            } else if (spin_count < 10000) {
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            } else {
                break;
            }
        }
        return e->cuda.cuEventQuery(event) == 0;
    }
    if (device_backend_is_hip(e)) {
        if (e->hip.hipEventSynchronize) {
            return e->hip.hipEventSynchronize(event) == 0;
        }
        if (!e->hip.hipEventQuery) {
            return 0;
        }
        uint32_t spin_count = 0;
        while (true) {
            if (e->hip.hipEventQuery(event) == 0) {
                return 1;
            }
            if (e->io_stop.load(std::memory_order_relaxed)) {
                break;
            }
            if (++spin_count < 20) {
                std::this_thread::yield();
            } else if (spin_count < 10000) {
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            } else {
                break;
            }
        }
        return e->hip.hipEventQuery(event) == 0;
    }
    if (device_backend_is_metal(e)) {
        return metal_event_sync((void*)event);
    }
    if (moe_vulkan_transfer_runtime_ready(e)) {
        auto* ev = reinterpret_cast<moe_vulkan_copy_event*>(event);
        if (!ev || !ev->submitted) {
            return 1;
        }
        e->io_atomic.backend_fence_waits.fetch_add(1, std::memory_order_relaxed);
        e->io_atomic.vulkan_fence_waits.fetch_add(1, std::memory_order_relaxed);
        if (moe_vulkan_wait_fence(e, ev->fence)) {
            vulkan_recycle_submitted_copy(e, ev);
            return 1;
        }
        return 0;
    }
    if (level_zero_async_runtime_ready(e)) {
        auto* ev = reinterpret_cast<moe_level_zero_copy_event*>(event);
        if (!ev || !ev->submitted) {
            return 1;
        }
        e->io_atomic.backend_fence_waits.fetch_add(1, std::memory_order_relaxed);
        e->io_atomic.level_zero_fence_waits.fetch_add(1, std::memory_order_relaxed);
        if (e->ze.zeFenceHostSynchronize(ev->fence, UINT64_MAX) == 0) {
            level_zero_recycle_submitted_list(e, ev);
            return 1;
        }
        return 0;
    }
    return 0;
}

static void device_backend_event_destroy(moe_pc_engine* e, CUevent event) {
    if (!event) {
        return;
    }
    if (device_backend_is_cuda(e) && e->cuda.cuEventDestroy) {
        e->cuda.cuEventDestroy(event);
    } else if (device_backend_is_hip(e) && e->hip.hipEventDestroy) {
        e->hip.hipEventDestroy(event);
    } else if (device_backend_is_metal(e)) {
        metal_event_destroy((void*)event);
    } else if (device_backend_is_vulkan(e)) {
        auto* ev = reinterpret_cast<moe_vulkan_copy_event*>(event);
        if (ev) {
            if (ev->submitted && !device_backend_event_sync(e, event) &&
                e && e->vk.vkQueueWaitIdle && e->vk.vkQueueWaitIdle(e->vk.queue) == 0) {
                vulkan_recycle_submitted_copy(e, ev);
            }
            if (!ev->submitted) {
                void* staging_buffer = ev->staging_buffer;
                void* staging_memory = ev->staging_memory;
                vulkan_release_staging(e, staging_buffer, staging_memory);
                if (ev->command_buffer) {
                    vulkan_recycle_command_buffer(e, ev->owner, ev->command_buffer);
                }
                if (ev->fence) {
                    moe_vulkan_destroy_fence(e, ev->fence);
                }
            }
            delete ev;
        }
    } else if (device_backend_is_level_zero(e)) {
        auto* ev = reinterpret_cast<moe_level_zero_copy_event*>(event);
        if (ev) {
            if (ev->submitted) {
                device_backend_event_sync(e, event);
            }
            if (ev->command_list) {
                level_zero_destroy_command_list(e, ev->command_list);
            }
            if (ev->fence && e->ze.zeFenceDestroy) {
                e->ze.zeFenceDestroy(ev->fence);
            }
            delete ev;
        }
    }
}

static int device_backend_async_h2d_available(moe_pc_engine* e, CUstream stream, int host_pinned) {
    if (!e || !stream || !e->backend_caps.supports_backend_async_api) {
        return 0;
    }
    if (moe_vulkan_transfer_runtime_ready(e)) {
        return 1;
    }
    if (!host_pinned) {
        return 0;
    }
    if (device_backend_is_cuda(e)) {
        return e->cuda.cuMemcpyHtoDAsync != nullptr;
    }
    if (device_backend_is_hip(e)) {
        return e->hip.hipMemcpyHtoDAsync != nullptr;
    }
    if (device_backend_is_metal(e)) {
        return 1;
    }
    if (level_zero_async_runtime_ready(e)) {
        return 1;
    }
    return 0;
}

static int device_backend_copy_h2d_async(
    moe_pc_engine* e,
    CUdeviceptr dst,
    const void* src,
    uint64_t bytes,
    CUstream stream
) {
    if (!e || !stream || !src || bytes == 0) {
        return 0;
    }
    if (device_backend_is_cuda(e) && e->cuda.cuMemcpyHtoDAsync) {
        const int ok = e->cuda.cuMemcpyHtoDAsync(dst, src, (size_t)bytes, stream) == 0;
        if (ok) {
            e->io_atomic.backend_async_submitted.fetch_add(bytes, std::memory_order_relaxed);
        }
        return ok;
    }
    if (device_backend_is_hip(e) && e->hip.hipMemcpyHtoDAsync) {
        const int ok = e->hip.hipMemcpyHtoDAsync(dst, src, (size_t)bytes, stream) == 0;
        if (ok) {
            e->io_atomic.backend_async_submitted.fetch_add(bytes, std::memory_order_relaxed);
        }
        return ok;
    }
    if (device_backend_is_metal(e)) {
        const bool measure = bytes >= 10ull * 1024ull * 1024ull;
        auto t0 = measure ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const int ok = metal_copy_h2d_async((void*)dst, src, bytes, (void*)stream);
        if (ok) {
            e->io_atomic.backend_async_submitted.fetch_add(bytes, std::memory_order_relaxed);
        }
        if (ok && measure) {
            auto t1 = std::chrono::steady_clock::now();
            const double elapsed_sec = std::chrono::duration<double>(t1 - t0).count();
            if (elapsed_sec > 0.0001) {
                const uint64_t current_bw = (uint64_t)((double)bytes / elapsed_sec);
                static std::mutex s_storage_bw_update_mutex;
                std::lock_guard<std::mutex> bw_lock(s_storage_bw_update_mutex);
                const uint64_t old_bw = e->backend_caps.measured_storage_bw_bytes_per_sec;
                const uint64_t new_bw = old_bw == 0 ? current_bw :
                    (uint64_t)(0.8 * (double)old_bw + 0.2 * (double)current_bw);
                e->backend_caps.measured_storage_bw_bytes_per_sec = new_bw;
            }
        }
        return ok;
    }
    if (moe_vulkan_transfer_runtime_ready(e)) {
        auto* state = reinterpret_cast<moe_vulkan_stream_state*>(stream);
        if (!state || state->pending_command_buffer || dst == 0) {
            e->io_atomic.backend_async_fallback.fetch_add(bytes, std::memory_order_relaxed);
            return 0;
        }
        void* staging_buffer = nullptr;
        void* staging_memory = nullptr;
        void* mapped = nullptr;
        if (!moe_vulkan_create_staging_buffer(
                e, bytes, &staging_buffer, &staging_memory, &mapped)) {
            e->io_atomic.backend_async_fallback.fetch_add(bytes, std::memory_order_relaxed);
            return 0;
        }
        std::memcpy(mapped, src, static_cast<size_t>(bytes));
        e->vk.vkUnmapMemory(e->vk.device, staging_memory);

        void* cmd = vulkan_take_command_buffer(e, state);
        if (!cmd) {
            vulkan_release_staging(e, staging_buffer, staging_memory);
            e->io_atomic.backend_async_fallback.fetch_add(bytes, std::memory_order_relaxed);
            return 0;
        }
        void* dst_buffer = reinterpret_cast<void*>(static_cast<uintptr_t>(dst));
        if (!moe_vulkan_record_copy_buffer(e, cmd, staging_buffer, dst_buffer, bytes)) {
            vulkan_release_staging(e, staging_buffer, staging_memory);
            vulkan_recycle_command_buffer(e, state, cmd);
            e->io_atomic.backend_async_fallback.fetch_add(bytes, std::memory_order_relaxed);
            return 0;
        }
        state->pending_command_buffer = cmd;
        state->pending_staging_buffer = staging_buffer;
        state->pending_staging_memory = staging_memory;
        state->pending_bytes = bytes;
        return 1;
    }
    if (level_zero_async_runtime_ready(e)) {
        auto* state = reinterpret_cast<moe_level_zero_stream_state*>(stream);
        void* cmd = level_zero_take_command_list(e, state);
        if (!cmd) {
            e->io_atomic.backend_async_fallback.fetch_add(bytes, std::memory_order_relaxed);
            return 0;
        }
        if (e->ze.zeCommandListAppendMemoryCopy(
                cmd,
                reinterpret_cast<void*>(static_cast<uintptr_t>(dst)),
                src,
                static_cast<size_t>(bytes),
                nullptr,
                0,
                nullptr) != 0) {
            level_zero_destroy_command_list(e, cmd);
            e->io_atomic.backend_async_fallback.fetch_add(bytes, std::memory_order_relaxed);
            return 0;
        }
        state->pending_command_list = cmd;
        e->io_atomic.backend_async_submitted.fetch_add(bytes, std::memory_order_relaxed);
        return 1;
    }
    if (e->backend_caps.backend == moe_BACKEND_VULKAN) {
        return 0;
    }
    return 0;
}

static int device_backend_copy_h2d_sync(
    moe_pc_engine* e,
    CUdeviceptr dst,
    const void* src,
    uint64_t bytes
) {
    if (!e || !src || bytes == 0) {
        return 0;
    }
    if (device_backend_is_cuda(e) && e->cuda.cuMemcpyHtoD) {
        return e->cuda.cuMemcpyHtoD(dst, src, (size_t)bytes) == 0;
    }
    if (device_backend_is_hip(e) && e->hip.hipMemcpyHtoD) {
        return e->hip.hipMemcpyHtoD(dst, src, (size_t)bytes) == 0;
    }
    if (device_backend_is_metal(e)) {
        return metal_copy_h2d_sync((void*)dst, src, bytes);
    }
    if (moe_vulkan_transfer_runtime_ready(e)) {
        if (dst == 0) {
            return 0;
        }
        void* staging_buffer = nullptr;
        void* staging_memory = nullptr;
        void* mapped = nullptr;
        if (!moe_vulkan_create_staging_buffer(
                e, bytes, &staging_buffer, &staging_memory, &mapped)) {
            return 0;
        }
        std::memcpy(mapped, src, static_cast<size_t>(bytes));
        e->vk.vkUnmapMemory(e->vk.device, staging_memory);

        void* cmd = moe_vulkan_create_command_buffer(e);
        void* fence = moe_vulkan_create_fence(e);
        int ok = 0;
        if (cmd && fence) {
            void* dst_buffer = reinterpret_cast<void*>(static_cast<uintptr_t>(dst));
            ok = moe_vulkan_record_copy_buffer(e, cmd, staging_buffer, dst_buffer, bytes) &&
                 moe_vulkan_submit_command_buffer(e, cmd, fence);
            if (ok) {
                e->io_atomic.backend_fence_waits.fetch_add(1, std::memory_order_relaxed);
                e->io_atomic.vulkan_fence_waits.fetch_add(1, std::memory_order_relaxed);
                ok = moe_vulkan_wait_fence(e, fence);
            }
        }
        if (fence) {
            moe_vulkan_reset_fence(e, fence);
            moe_vulkan_destroy_fence(e, fence);
        }
        if (cmd) {
            moe_vulkan_destroy_command_buffer(e, cmd);
        }
        vulkan_release_staging(e, staging_buffer, staging_memory);
        return ok;
    }
    if (level_zero_async_runtime_ready(e)) {
        moe_level_zero_stream_state state{};
        void* cmd = level_zero_take_command_list(e, &state);
        if (!cmd) {
            return 0;
        }
        if (e->ze.zeCommandListAppendMemoryCopy(
                cmd,
                reinterpret_cast<void*>(static_cast<uintptr_t>(dst)),
                src,
                static_cast<size_t>(bytes),
                nullptr,
                0,
                nullptr) != 0) {
            level_zero_destroy_command_list(e, cmd);
            return 0;
        }
        state.pending_command_list = cmd;
        const int ok = level_zero_submit_pending_and_sync(e, &state);
        for (void* idle : state.idle_command_lists) {
            level_zero_destroy_command_list(e, idle);
        }
        return ok;
    }
    return 0;
}
