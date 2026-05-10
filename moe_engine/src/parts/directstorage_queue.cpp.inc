// Windows DirectStorage queue facade.
// The compiled path is enabled only when the SDK header is present and the build
// opts in. All other platforms keep the existing IOCP/IORing/mmap fallbacks.

static uint64_t moe_profile_now_us();
static size_t drain_directstorage_staging_copies(moe_pc_engine* engine);
static size_t device_drain_directstorage_device_copies(moe_pc_engine* e);

#if defined(_WIN32) && (defined(moe_ENABLE_DIRECTSTORAGE) || defined(HAVE_DIRECTSTORAGE)) && defined(__has_include)
#if __has_include(<dstorage.h>) && __has_include(<d3d12.h>)
#define MOE_DIRECTSTORAGE_QUEUE_ENABLED 1
#endif
#endif

#ifndef MOE_MAYBE_UNUSED
#if defined(__has_cpp_attribute)
#if __has_cpp_attribute(maybe_unused)
#define MOE_MAYBE_UNUSED [[maybe_unused]]
#else
#define MOE_MAYBE_UNUSED
#endif
#else
#define MOE_MAYBE_UNUSED
#endif
#endif

#if defined(MOE_DIRECTSTORAGE_QUEUE_ENABLED)
#include <d3d12.h>
#include <dstorage.h>
#include <unknwn.h>

static uint32_t directstorage_clamp_queue_capacity(uint32_t requested) {
    return std::max<uint32_t>(
        moe_DSTORAGE_MIN_STATUS_SLOTS,
        std::min<uint32_t>(
            requested ? requested : moe_DSTORAGE_DEFAULT_STATUS_SLOTS,
            moe_DSTORAGE_MAX_STATUS_SLOTS));
}

static uint32_t directstorage_status_slot_count(const std::vector<moe_dstorage_status_slot>& slots) {
    return static_cast<uint32_t>(
        std::min<size_t>(slots.size(), static_cast<size_t>(UINT32_MAX)));
}
using moe_dstorage_get_factory_fn = HRESULT (WINAPI*)(REFIID, void**);

static void CALLBACK directstorage_completion_event_callback(PVOID ctx, BOOLEAN) {
    auto* engine = static_cast<moe_pc_engine*>(ctx);
    if (!engine) {
        return;
    }
    engine->disk_stage.cv.notify_one();
    engine->gpu_stage.cv.notify_one();
}

static void directstorage_release_unknown(void*& ptr) {
    if (ptr) {
        static_cast<IUnknown*>(ptr)->Release();
        ptr = nullptr;
    }
}

static int directstorage_queue_supports_completion_event(
    moe_pc_engine* engine,
    IDStorageQueue* queue);

static moe_dstorage_get_factory_fn directstorage_get_factory_proc() {
    static std::mutex s_load_mutex;
    static HMODULE s_library = nullptr;
    static moe_dstorage_get_factory_fn s_proc = nullptr;
    if (s_proc) {
        return s_proc;
    }
    std::lock_guard<std::mutex> lock(s_load_mutex);
    if (s_proc) {
        return s_proc;
    }
    s_library = LoadLibraryA("dstorage.dll");
    if (!s_library) {
        return nullptr;
    }
    s_proc = reinterpret_cast<moe_dstorage_get_factory_fn>(
        GetProcAddress(s_library, "DStorageGetFactory"));
    if (!s_proc) {
        FreeLibrary(s_library);
        s_library = nullptr;
    }
    return s_proc;
}

static int directstorage_runtime_available() {
    return directstorage_get_factory_proc() ? 1 : 0;
}

static void directstorage_status_slots_bind(
    std::vector<moe_dstorage_status_slot>& slots,
    void* status_array,
    uint32_t capacity
) {
    if (!status_array || capacity == 0) {
        slots.clear();
        return;
    }
    capacity = directstorage_clamp_queue_capacity(capacity);
    slots.clear();
    slots.resize(capacity);
    for (uint32_t i = 0; i < capacity; ++i) {
        slots[i].in_use.store(false, std::memory_order_release);
        slots[i].array = status_array;
        slots[i].index = i;
        ++slots[i].generation;
    }
}

static int acquire_dstorage_status_slot(
    moe_pc_engine* engine,
    std::vector<moe_dstorage_status_slot>& slots,
    bool nonblocking = true
) {
    if (!engine || slots.empty()) {
        return -1;
    }
    const uint32_t capacity = directstorage_status_slot_count(slots);
    const uint32_t retries = nonblocking ? 2u : 64u;
    for (uint32_t retry = 0; retry < retries; ++retry) {
        for (uint32_t i = 0; i < capacity; ++i) {
            bool expected = false;
            if (slots[i].in_use.compare_exchange_strong(
                    expected,
                    true,
                    std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                return static_cast<int>(i);
            }
        }
        if (engine->io_stop.load(std::memory_order_relaxed)) {
            return -1;
        }
        if (nonblocking) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    return -1;
}

static void release_dstorage_status_slot(moe_dstorage_status_slot* slots, uint32_t capacity, int i) {
    if (!slots || i < 0 || static_cast<uint32_t>(i) >= capacity) {
        return;
    }
    ++slots[i].generation;
    slots[i].in_use.store(false, std::memory_order_release);
}

struct moe_dstorage_status_guard {
    moe_dstorage_status_slot* slots = nullptr;
    uint32_t capacity = 0;
    int slot = -1;
    bool release_on_destroy = true;

    moe_dstorage_status_guard(moe_dstorage_status_slot* s, uint32_t c, int i)
        : slots(s), capacity(c), slot(i) {}

    ~moe_dstorage_status_guard() {
        if (release_on_destroy) {
            release_dstorage_status_slot(slots, capacity, slot);
        }
    }

    uint32_t index() const {
        return slot >= 0 && slots ? slots[slot].index : 0u;
    }

    void keep_reserved() {
        release_on_destroy = false;
    }
};

static void directstorage_release_pending_copy_resources(moe_dstorage_pending_copy* pending) {
    if (!pending) {
        return;
    }
    if (pending->resource) {
        static_cast<IUnknown*>(pending->resource)->Release();
        pending->resource = nullptr;
    }
    if (pending->file) {
        static_cast<IUnknown*>(pending->file)->Release();
        pending->file = nullptr;
    }
    release_dstorage_status_slot(pending->slots, pending->slot_capacity, pending->slot_index);
    pending->status_array = nullptr;
    pending->slot_index = -1;
    pending->slot_capacity = 0;
    pending->slots = nullptr;
}

static int directstorage_queue_ensure(moe_pc_engine* engine) {
    if (!engine || !compiled_directstorage_available() || !directstorage_runtime_available()) {
        if (engine) {
            engine->runtime_directstorage_staging_ready.store(false, std::memory_order_release);
        }
        return 0;
    }
    if (engine->directstorage.initialized) {
        engine->runtime_directstorage_staging_ready.store(
            engine->directstorage.available,
            std::memory_order_release);
        return engine->directstorage.available ? 1 : 0;
    }

    static std::mutex s_init_mutex;
    std::lock_guard<std::mutex> lock(s_init_mutex);
    if (engine->directstorage.initialized) {
        engine->runtime_directstorage_staging_ready.store(
            engine->directstorage.available,
            std::memory_order_release);
        return engine->directstorage.available ? 1 : 0;
    }

    engine->directstorage.initialized = true;
    engine->directstorage.available = false;

    auto get_factory = directstorage_get_factory_proc();
    if (!get_factory) {
        return 0;
    }

    IDStorageFactory* factory = nullptr;
    HRESULT hr = get_factory(__uuidof(IDStorageFactory), reinterpret_cast<void**>(&factory));
    if (FAILED(hr) || !factory) {
        engine->runtime_directstorage_staging_ready.store(false, std::memory_order_release);
        return 0;
    }

    DSTORAGE_QUEUE_DESC queue_desc{};
    queue_desc.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
    const uint32_t queue_capacity = directstorage_clamp_queue_capacity(
        engine->io_config.directstorage_staging_queue_capacity);
    queue_desc.Capacity = queue_capacity;
    queue_desc.Priority = DSTORAGE_PRIORITY_NORMAL;
    queue_desc.Device = nullptr;
    queue_desc.Name = "storageLLM staging queue";

    IDStorageQueue* queue = nullptr;
    hr = factory->CreateQueue(&queue_desc, __uuidof(IDStorageQueue), reinterpret_cast<void**>(&queue));
    if (FAILED(hr) || !queue) {
        factory->Release();
        engine->runtime_directstorage_staging_ready.store(false, std::memory_order_release);
        return 0;
    }

    IDStorageStatusArray* status_array = nullptr;
    hr = factory->CreateStatusArray(
        queue_capacity,
        "storageLLM staging statuses",
        __uuidof(IDStorageStatusArray),
        reinterpret_cast<void**>(&status_array));
    if (FAILED(hr) || !status_array) {
        queue->Release();
        factory->Release();
        engine->runtime_directstorage_staging_ready.store(false, std::memory_order_release);
        return 0;
    }

    engine->directstorage.factory = factory;
    engine->directstorage.queue = queue;
    engine->directstorage.status_array = status_array;
    engine->directstorage.status_index = 0;
    if (!engine->directstorage.event_handle) {
        engine->directstorage.event_handle = CreateEventA(nullptr, FALSE, FALSE, nullptr);
    }
    if (engine->directstorage.event_handle && !engine->directstorage.completion_wait_handle) {
        HANDLE wait_handle = nullptr;
        if (RegisterWaitForSingleObject(
                &wait_handle,
                static_cast<HANDLE>(engine->directstorage.event_handle),
                directstorage_completion_event_callback,
                engine,
                INFINITE,
                WT_EXECUTEDEFAULT)) {
            engine->directstorage.completion_wait_handle = wait_handle;
        }
    }
    if (engine->io_config.directstorage_require_completion_event &&
        !directstorage_queue_supports_completion_event(engine, queue)) {
        status_array->Release();
        queue->Release();
        factory->Release();
        if (engine->directstorage.completion_wait_handle) {
            UnregisterWaitEx(
                static_cast<HANDLE>(engine->directstorage.completion_wait_handle),
                INVALID_HANDLE_VALUE);
            engine->directstorage.completion_wait_handle = nullptr;
        }
        if (engine->directstorage.event_handle) {
            CloseHandle(static_cast<HANDLE>(engine->directstorage.event_handle));
            engine->directstorage.event_handle = nullptr;
        }
        engine->directstorage.initialized = false;
        engine->runtime_directstorage_staging_ready.store(false, std::memory_order_release);
        return 0;
    }
    directstorage_status_slots_bind(engine->directstorage.status_slots, status_array, queue_capacity);
    engine->directstorage.available = true;
    engine->runtime_directstorage_staging_ready.store(true, std::memory_order_release);
    return 1;
}

static void directstorage_queue_cleanup(moe_pc_engine* engine) {
    if (!engine) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(engine->directstorage_pending_mutex);
        for (moe_directstorage_device_pending& pending : engine->directstorage_pending_device_copies) {
            directstorage_release_pending_copy_resources(&pending.io);
        }
        engine->directstorage_pending_device_copies.clear();
    }
    {
        std::lock_guard<std::mutex> lock(engine->directstorage_staging_pending_mutex);
        for (moe_directstorage_staging_pending& pending : engine->directstorage_pending_staging_copies) {
            directstorage_release_pending_copy_resources(&pending.io);
        }
        engine->directstorage_pending_staging_copies.clear();
    }
    engine->directstorage_pending_count.store(0, std::memory_order_release);
    directstorage_release_unknown(engine->directstorage.gpu_status_array);
    directstorage_release_unknown(engine->directstorage.gpu_queue);
    directstorage_release_unknown(engine->directstorage.status_array);
    directstorage_release_unknown(engine->directstorage.queue);
    directstorage_release_unknown(engine->directstorage.d3d12_device);
    directstorage_release_unknown(engine->directstorage.factory);
    if (engine->directstorage.completion_wait_handle) {
        UnregisterWaitEx(
            static_cast<HANDLE>(engine->directstorage.completion_wait_handle),
            INVALID_HANDLE_VALUE);
        engine->directstorage.completion_wait_handle = nullptr;
    }
    if (engine->directstorage.event_handle) {
        CloseHandle(static_cast<HANDLE>(engine->directstorage.event_handle));
        engine->directstorage.event_handle = nullptr;
    }
    engine->directstorage.initialized = false;
    engine->directstorage.available = false;
    engine->directstorage.gpu_available = false;
    engine->directstorage_submission_paused.store(false, std::memory_order_release);
    engine->directstorage_completion_event_supported.store(false, std::memory_order_release);
    engine->directstorage.status_index = 0;
    engine->directstorage.gpu_status_index = 0;
    directstorage_status_slots_bind(engine->directstorage.status_slots, nullptr, 0);
    directstorage_status_slots_bind(engine->directstorage.gpu_status_slots, nullptr, 0);
    engine->runtime_directstorage_staging_ready.store(false, std::memory_order_release);
    engine->runtime_directstorage_gpu_ready.store(false, std::memory_order_release);
}

static int directstorage_gpu_queue_ensure(moe_pc_engine* engine);

MOE_MAYBE_UNUSED static int directstorage_bind_d3d12_device(moe_pc_engine* engine, void* d3d12_device) {
    if (!engine || !d3d12_device || !compiled_directstorage_available()) {
        return 0;
    }
    static std::mutex s_bind_mutex;
    std::lock_guard<std::mutex> lock(s_bind_mutex);
    if (engine->directstorage.d3d12_device == d3d12_device) {
        return 1;
    }
    directstorage_release_unknown(engine->directstorage.gpu_status_array);
    directstorage_release_unknown(engine->directstorage.gpu_queue);
    directstorage_release_unknown(engine->directstorage.d3d12_device);
    static_cast<IUnknown*>(d3d12_device)->AddRef();
    engine->directstorage.d3d12_device = d3d12_device;
    engine->directstorage.gpu_available = false;
    engine->directstorage.gpu_status_index = 0;
    directstorage_status_slots_bind(engine->directstorage.gpu_status_slots, nullptr, 0);
    engine->runtime_directstorage_gpu_ready.store(false, std::memory_order_release);
    if (engine->backend_caps.backend == moe_BACKEND_DIRECTML) {
        directstorage_gpu_queue_ensure(engine);
    }
    return 1;
}

static int directstorage_gpu_queue_ensure(moe_pc_engine* engine) {
    if (!engine || !engine->directstorage.d3d12_device ||
        !directstorage_queue_ensure(engine)) {
        if (engine) {
            engine->runtime_directstorage_gpu_ready.store(false, std::memory_order_release);
        }
        return 0;
    }
    if (engine->directstorage.gpu_queue &&
        engine->directstorage.gpu_status_array &&
        engine->directstorage.gpu_available) {
        engine->runtime_directstorage_gpu_ready.store(
            engine->backend_caps.backend == moe_BACKEND_DIRECTML,
            std::memory_order_release);
        return 1;
    }

    static std::mutex s_gpu_init_mutex;
    std::lock_guard<std::mutex> lock(s_gpu_init_mutex);
    if (engine->directstorage.gpu_queue &&
        engine->directstorage.gpu_status_array &&
        engine->directstorage.gpu_available) {
        engine->runtime_directstorage_gpu_ready.store(
            engine->backend_caps.backend == moe_BACKEND_DIRECTML,
            std::memory_order_release);
        return 1;
    }

    IDStorageFactory* factory = static_cast<IDStorageFactory*>(engine->directstorage.factory);
    if (!factory) {
        engine->runtime_directstorage_gpu_ready.store(false, std::memory_order_release);
        return 0;
    }

    DSTORAGE_QUEUE_DESC queue_desc{};
    queue_desc.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
    const uint32_t queue_capacity = directstorage_clamp_queue_capacity(
        engine->io_config.directstorage_gpu_queue_capacity);
    queue_desc.Capacity = queue_capacity;
    queue_desc.Priority = DSTORAGE_PRIORITY_NORMAL;
    queue_desc.Device = static_cast<ID3D12Device*>(engine->directstorage.d3d12_device);
    queue_desc.Name = "storageLLM GPU resource queue";

    IDStorageQueue* queue = nullptr;
    HRESULT hr = factory->CreateQueue(&queue_desc, __uuidof(IDStorageQueue), reinterpret_cast<void**>(&queue));
    if (FAILED(hr) || !queue) {
        engine->directstorage.gpu_available = false;
        engine->runtime_directstorage_gpu_ready.store(false, std::memory_order_release);
        return 0;
    }

    IDStorageStatusArray* status_array = nullptr;
    hr = factory->CreateStatusArray(
        queue_capacity,
        "storageLLM GPU resource statuses",
        __uuidof(IDStorageStatusArray),
        reinterpret_cast<void**>(&status_array));
    if (FAILED(hr) || !status_array) {
        queue->Release();
        engine->directstorage.gpu_available = false;
        engine->runtime_directstorage_gpu_ready.store(false, std::memory_order_release);
        return 0;
    }
    if (engine->io_config.directstorage_require_completion_event &&
        !directstorage_queue_supports_completion_event(engine, queue)) {
        status_array->Release();
        queue->Release();
        engine->directstorage.gpu_available = false;
        engine->runtime_directstorage_gpu_ready.store(false, std::memory_order_release);
        return 0;
    }

    engine->directstorage.gpu_queue = queue;
    engine->directstorage.gpu_status_array = status_array;
    engine->directstorage.gpu_status_index = 0;
    directstorage_status_slots_bind(engine->directstorage.gpu_status_slots, status_array, queue_capacity);
    engine->directstorage.gpu_available = true;
    engine->runtime_directstorage_gpu_ready.store(
        engine->backend_caps.backend == moe_BACKEND_DIRECTML,
        std::memory_order_release);
    return 1;
}

MOE_MAYBE_UNUSED static int directstorage_direct_gpu_available(moe_pc_engine* engine) {
    return directstorage_gpu_queue_ensure(engine);
}

static void directstorage_maybe_recreate_queues(moe_pc_engine* engine) {
    if (!engine ||
        !engine->directstorage_recreate_requested.load(std::memory_order_acquire)) {
        return;
    }
    engine->directstorage_submission_paused.store(true, std::memory_order_release);
    if (engine->directstorage_pending_count.load(std::memory_order_acquire) != 0) {
        engine->runtime_prefetch_window_layers.store(1u, std::memory_order_relaxed);
        engine->adaptive_prefetch_window_last.store(1u, std::memory_order_relaxed);
        return;
    }
    if (!engine->directstorage_recreate_requested.exchange(false, std::memory_order_acq_rel)) {
        engine->directstorage_submission_paused.store(false, std::memory_order_release);
        return;
    }
    void* d3d12_device = engine->directstorage.d3d12_device;
    if (d3d12_device) {
        static_cast<IUnknown*>(d3d12_device)->AddRef();
    }
    directstorage_queue_cleanup(engine);
    directstorage_queue_ensure(engine);
    if (d3d12_device) {
        directstorage_bind_d3d12_device(engine, d3d12_device);
        static_cast<IUnknown*>(d3d12_device)->Release();
    }
    engine->directstorage_submission_paused.store(false, std::memory_order_release);
}

MOE_MAYBE_UNUSED static int directstorage_create_d3d12_buffer_resource(
    moe_pc_engine* engine,
    uint64_t bytes,
    void** out_resource
) {
    if (out_resource) {
        *out_resource = nullptr;
    }
    if (!engine || !out_resource || bytes == 0 || bytes > UINT32_MAX ||
        !directstorage_gpu_queue_ensure(engine) ||
        engine->backend_caps.backend != moe_BACKEND_DIRECTML) {
        return 0;
    }
    auto* device = static_cast<ID3D12Device*>(engine->directstorage.d3d12_device);
    if (!device) {
        return 0;
    }
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    heap.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heap.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heap.CreationNodeMask = 1;
    heap.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Alignment = 0;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    ID3D12Resource* resource = nullptr;
    const HRESULT hr = device->CreateCommittedResource(
        &heap,
        D3D12_HEAP_FLAG_NONE,
        &desc,
        D3D12_RESOURCE_STATE_COMMON,
        nullptr,
        __uuidof(ID3D12Resource),
        reinterpret_cast<void**>(&resource));
    if (FAILED(hr) || !resource) {
        return 0;
    }
    *out_resource = resource;
    return 1;
}

MOE_MAYBE_UNUSED static void directstorage_release_d3d12_resource(void* resource) {
    if (resource) {
        static_cast<IUnknown*>(resource)->Release();
    }
}

static int directstorage_queue_supports_completion_event(
    moe_pc_engine* engine,
    IDStorageQueue* queue
) {
    if (!engine || !queue || !engine->directstorage.event_handle) {
        return 0;
    }
#if defined(DSTORAGE_ENQUEUE_REQUEST_FLAG_NONE) && defined(__IDStorageQueue3_INTERFACE_DEFINED__)
    IDStorageQueue3* queue3 = nullptr;
    if (SUCCEEDED(queue->QueryInterface(__uuidof(IDStorageQueue3), reinterpret_cast<void**>(&queue3))) &&
        queue3) {
        queue3->Release();
        return 1;
    }
#endif
    return 0;
}

static void directstorage_enqueue_requests_status_and_event(
    moe_pc_engine* engine,
    IDStorageQueue* queue,
    const std::vector<DSTORAGE_REQUEST>& requests,
    IDStorageStatusArray* status_array,
    uint32_t status_index
) {
    if (!queue || !status_array) {
        return;
    }
#if defined(DSTORAGE_ENQUEUE_REQUEST_FLAG_NONE) && defined(__IDStorageQueue3_INTERFACE_DEFINED__)
    IDStorageQueue3* queue3 = nullptr;
    if (SUCCEEDED(queue->QueryInterface(__uuidof(IDStorageQueue3), reinterpret_cast<void**>(&queue3))) &&
        queue3) {
        if (!requests.empty()) {
            queue3->EnqueueRequests(
                static_cast<UINT>(requests.size()),
                requests.data(),
                nullptr,
                0,
                DSTORAGE_ENQUEUE_REQUEST_FLAG_NONE);
        }
        queue->EnqueueStatus(status_array, status_index);
        bool completion_event_supported = false;
        if (engine && engine->directstorage.event_handle) {
            queue3->EnqueueSetEvent(static_cast<HANDLE>(engine->directstorage.event_handle));
            completion_event_supported = true;
        }
        if (engine) {
            engine->directstorage_completion_event_supported.store(
                completion_event_supported,
                std::memory_order_release);
        }
        queue3->Release();
        return;
    }
#endif
    if (engine) {
        engine->directstorage_completion_event_supported.store(false, std::memory_order_release);
    }
    for (const auto& request : requests) {
        queue->EnqueueRequest(&request);
    }
    queue->EnqueueStatus(status_array, status_index);
}

MOE_MAYBE_UNUSED static int directstorage_submit_ranges_to_gpu_buffer(
    moe_pc_engine* engine,
    const wchar_t* path,
    const uint64_t ranges[][2],
    size_t range_count,
    void* d3d12_resource,
    uint64_t resource_offset,
    uint64_t total_bytes,
    uint64_t file_size,
    moe_dstorage_pending_copy* out
) {
    if (out) {
        *out = moe_dstorage_pending_copy{};
    }
    if (!engine || !path || !path[0] || !ranges || !d3d12_resource || !out ||
        total_bytes == 0 ||
        engine->directstorage_submission_paused.load(std::memory_order_acquire) ||
        !engine->io_config.enable_directstorage ||
        !engine->backend_caps.supports_directstorage ||
        !directstorage_gpu_queue_ensure(engine)) {
        return 0;
    }

    IDStorageFactory* factory = static_cast<IDStorageFactory*>(engine->directstorage.factory);
    IDStorageQueue* queue = static_cast<IDStorageQueue*>(engine->directstorage.gpu_queue);
    IDStorageStatusArray* status_array =
        static_cast<IDStorageStatusArray*>(engine->directstorage.gpu_status_array);
    auto* resource = static_cast<ID3D12Resource*>(d3d12_resource);
    if (!factory || !queue || !status_array || !resource) {
        return 0;
    }

    if (engine->io_config.directstorage_require_completion_event &&
        !directstorage_queue_supports_completion_event(engine, queue)) {
        engine->runtime_directstorage_gpu_ready.store(false, std::memory_order_release);
        moe_io_path_perf_fail(engine->ds_gpu_perf, 0);
        return 0;
    }

    IDStorageFile* file = nullptr;
    HRESULT hr = factory->OpenFile(path, __uuidof(IDStorageFile), reinterpret_cast<void**>(&file));
    if (FAILED(hr) || !file) {
        engine->io_atomic.bytes_directstorage_failed.fetch_add(total_bytes, std::memory_order_relaxed);
        moe_io_path_perf_fail(engine->ds_gpu_perf, 0);
        return 0;
    }
    resource->AddRef();

    const int acquired_status_slot = acquire_dstorage_status_slot(
        engine,
        engine->directstorage.gpu_status_slots,
        true);
    if (acquired_status_slot < 0) {
        resource->Release();
        file->Release();
        engine->io_atomic.directstorage_queue_full_wait_ns.fetch_add(
            1000000ull,
            std::memory_order_relaxed);
        engine->io_atomic.bytes_directstorage_failed.fetch_add(total_bytes, std::memory_order_relaxed);
        moe_io_path_perf_fail(engine->ds_gpu_perf, 0);
        return 0;
    }
    moe_dstorage_status_slot* gpu_status_slots = engine->directstorage.gpu_status_slots.data();
    const uint32_t gpu_status_capacity = directstorage_status_slot_count(
        engine->directstorage.gpu_status_slots);
    moe_dstorage_status_guard status_guard(
        gpu_status_slots,
        gpu_status_capacity,
        acquired_status_slot);
    const uint32_t status_index = status_guard.index();
    uint64_t checked_gpu_offset = 0;
    for (size_t i = 0; i < range_count; ++i) {
        const uint64_t file_offset = ranges[i][0];
        const uint64_t length = ranges[i][1];
        if (length == 0) {
            continue;
        }
        if (length > UINT32_MAX ||
            file_offset > file_size ||
            length > file_size - file_offset ||
            checked_gpu_offset > total_bytes ||
            length > total_bytes - checked_gpu_offset ||
            resource_offset > UINT64_MAX - checked_gpu_offset ||
            length > UINT64_MAX - (resource_offset + checked_gpu_offset)) {
            resource->Release();
            file->Release();
            engine->io_atomic.bytes_directstorage_failed.fetch_add(total_bytes, std::memory_order_relaxed);
            moe_io_path_perf_fail(engine->ds_gpu_perf, 0);
            return 0;
        }
        checked_gpu_offset += length;
    }
    const uint64_t submit_start_us = moe_profile_now_us();
    {
        std::lock_guard<std::mutex> submit_lock(engine->directstorage_gpu_submit_mutex);

        std::vector<DSTORAGE_REQUEST> requests;
        requests.reserve(range_count);
        uint64_t gpu_offset = 0;
        for (size_t i = 0; i < range_count; ++i) {
            const uint64_t file_offset = ranges[i][0];
            const uint64_t length = ranges[i][1];
            if (length == 0) {
                continue;
            }
            if (length > UINT32_MAX ||
                file_offset > file_size ||
                length > file_size - file_offset ||
                gpu_offset > total_bytes ||
                length > total_bytes - gpu_offset ||
                resource_offset > UINT64_MAX - gpu_offset) {
                resource->Release();
                file->Release();
                engine->io_atomic.bytes_directstorage_failed.fetch_add(total_bytes, std::memory_order_relaxed);
                moe_io_path_perf_fail(engine->ds_gpu_perf, 0);
                return 0;
            }
            const uint64_t dst_offset = resource_offset + gpu_offset;
            if (length > UINT64_MAX - dst_offset) {
                resource->Release();
                file->Release();
                engine->io_atomic.bytes_directstorage_failed.fetch_add(total_bytes, std::memory_order_relaxed);
                moe_io_path_perf_fail(engine->ds_gpu_perf, 0);
                return 0;
            }

            DSTORAGE_REQUEST request{};
            request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
            request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_BUFFER;
            request.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT_NONE;
            request.Source.File.Source = file;
            request.Source.File.Offset = file_offset;
            request.Source.File.Size = static_cast<uint32_t>(length);
            request.Destination.Buffer.Resource = resource;
            request.Destination.Buffer.Offset = dst_offset;
            request.Destination.Buffer.Size = static_cast<uint32_t>(length);
            request.UncompressedSize = static_cast<uint32_t>(length);
            request.CancellationTag = 0;
            request.Name = "storageLLM tensor GPU range";
            requests.push_back(request);
            gpu_offset += length;
        }

        directstorage_enqueue_requests_status_and_event(
            engine,
            queue,
            requests,
            status_array,
            status_index);
        queue->Submit();
    }
    const uint64_t submit_done_us = moe_profile_now_us();
    if (submit_done_us > submit_start_us) {
        const uint64_t wait_ns = (submit_done_us - submit_start_us) * 1000ull;
        if (wait_ns > 100000ull) {
            engine->io_atomic.directstorage_queue_full_wait_ns.fetch_add(
                wait_ns,
                std::memory_order_relaxed);
        }
    }

    engine->io_atomic.bytes_directstorage_submitted.fetch_add(total_bytes, std::memory_order_relaxed);
    moe_io_path_perf_submit(engine->ds_gpu_perf, total_bytes);
    status_guard.keep_reserved();
    out->status_array = status_array;
    out->status_index = status_index;
    out->slots = gpu_status_slots;
    out->slot_capacity = gpu_status_capacity;
    out->slot_index = acquired_status_slot;
    out->generation = gpu_status_slots[acquired_status_slot].generation;
    out->bytes = total_bytes;
    out->resource = resource;
    out->file = file;
    out->submitted_us = moe_profile_now_us();
    out->direct_to_gpu = true;
    return 1;
}

static int directstorage_poll_pending_copy(
    moe_pc_engine* engine,
    moe_dstorage_pending_copy* pending
) {
    if (!engine || !pending || !pending->status_array) {
        return -1;
    }
    IDStorageStatusArray* status_array = static_cast<IDStorageStatusArray*>(pending->status_array);
    if (!status_array->IsComplete(pending->status_index)) {
        return 0;
    }
    const HRESULT hr = status_array->GetHResult(pending->status_index);
    directstorage_release_pending_copy_resources(pending);
    const uint64_t now_us = moe_profile_now_us();
    const uint64_t wait_ns = now_us > pending->submitted_us ?
        (now_us - pending->submitted_us) * 1000ull : 0ull;
    if (FAILED(hr)) {
        engine->io_atomic.bytes_directstorage_failed.fetch_add(pending->bytes, std::memory_order_relaxed);
        if (pending->direct_to_gpu) {
            moe_io_path_perf_fail(engine->ds_gpu_perf, wait_ns);
        } else {
            moe_io_path_perf_fail(engine->ds_staging_perf, wait_ns);
        }
        return -1;
    }
    engine->io_atomic.bytes_directstorage_completed.fetch_add(pending->bytes, std::memory_order_relaxed);
    if (pending->direct_to_gpu) {
        engine->io_atomic.bytes_direct_to_gpu.fetch_add(pending->bytes, std::memory_order_relaxed);
        moe_io_path_perf_complete(engine->ds_gpu_perf, wait_ns);
    } else {
        moe_io_path_perf_complete(engine->ds_staging_perf, wait_ns);
    }
    return 1;
}

static void directstorage_completion_worker_loop(moe_pc_engine* engine) {
    while (engine && !engine->io_stop.load(std::memory_order_acquire)) {
        const uint32_t pending = engine->directstorage_pending_count.load(std::memory_order_acquire);
        if (pending == 0) {
#ifdef _WIN32
            if (engine->directstorage.event_handle) {
                WaitForSingleObject(static_cast<HANDLE>(engine->directstorage.event_handle), 1);
            } else
#endif
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            directstorage_maybe_recreate_queues(engine);
            continue;
        }
        const bool completion_event_supported =
            engine->directstorage_completion_event_supported.load(std::memory_order_acquire);
#ifdef _WIN32
        if (completion_event_supported && engine->directstorage.event_handle) {
            WaitForSingleObject(static_cast<HANDLE>(engine->directstorage.event_handle), 1);
        } else
#endif
        {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
        drain_directstorage_staging_copies(engine);
        device_drain_directstorage_device_copies(engine);
        directstorage_maybe_recreate_queues(engine);
    }
}

static int directstorage_submit_ranges_to_staging(
    moe_pc_engine* engine,
    const wchar_t* path,
    const uint64_t ranges[][2],
    size_t range_count,
    uint8_t* dst,
    uint64_t total_bytes,
    uint64_t file_size,
    moe_dstorage_pending_copy* out
) {
    if (out) {
        *out = moe_dstorage_pending_copy{};
    }
    if (!engine || !path || !path[0] || !ranges || !dst || !out || total_bytes == 0 ||
        engine->directstorage_submission_paused.load(std::memory_order_acquire) ||
        !engine->io_config.enable_directstorage ||
        !engine->backend_caps.supports_directstorage ||
        !directstorage_queue_ensure(engine)) {
        return 0;
    }

    IDStorageFactory* factory = static_cast<IDStorageFactory*>(engine->directstorage.factory);
    IDStorageQueue* queue = static_cast<IDStorageQueue*>(engine->directstorage.queue);
    IDStorageStatusArray* status_array =
        static_cast<IDStorageStatusArray*>(engine->directstorage.status_array);
    if (!factory || !queue || !status_array) {
        return 0;
    }

    if (engine->io_config.directstorage_require_completion_event &&
        !directstorage_queue_supports_completion_event(engine, queue)) {
        engine->runtime_directstorage_staging_ready.store(false, std::memory_order_release);
        moe_io_path_perf_fail(engine->ds_staging_perf, 0);
        return 0;
    }

    IDStorageFile* file = nullptr;
    HRESULT hr = factory->OpenFile(path, __uuidof(IDStorageFile), reinterpret_cast<void**>(&file));
    if (FAILED(hr) || !file) {
        engine->io_atomic.bytes_directstorage_failed.fetch_add(total_bytes, std::memory_order_relaxed);
        moe_io_path_perf_fail(engine->ds_staging_perf, 0);
        return 0;
    }

    const int acquired_status_slot = acquire_dstorage_status_slot(
        engine,
        engine->directstorage.status_slots,
        true);
    if (acquired_status_slot < 0) {
        file->Release();
        engine->io_atomic.directstorage_queue_full_wait_ns.fetch_add(
            1000000ull,
            std::memory_order_relaxed);
        engine->io_atomic.bytes_directstorage_failed.fetch_add(total_bytes, std::memory_order_relaxed);
        moe_io_path_perf_fail(engine->ds_staging_perf, 0);
        return 0;
    }
    moe_dstorage_status_slot* status_slots = engine->directstorage.status_slots.data();
    const uint32_t status_capacity = directstorage_status_slot_count(
        engine->directstorage.status_slots);
    moe_dstorage_status_guard status_guard(
        status_slots,
        status_capacity,
        acquired_status_slot);
    const uint32_t status_index = status_guard.index();
    uint64_t checked_staging_offset = 0;
    for (size_t i = 0; i < range_count; ++i) {
        const uint64_t file_offset = ranges[i][0];
        const uint64_t length = ranges[i][1];
        if (length == 0) {
            continue;
        }
        if (length > UINT32_MAX ||
            file_offset > file_size ||
            length > file_size - file_offset ||
            checked_staging_offset > total_bytes ||
            length > total_bytes - checked_staging_offset) {
            file->Release();
            engine->io_atomic.bytes_directstorage_failed.fetch_add(total_bytes, std::memory_order_relaxed);
            moe_io_path_perf_fail(engine->ds_staging_perf, 0);
            return 0;
        }
        checked_staging_offset += length;
    }
    const uint64_t submit_start_us = moe_profile_now_us();
    {
        std::lock_guard<std::mutex> submit_lock(engine->directstorage_staging_submit_mutex);

        std::vector<DSTORAGE_REQUEST> requests;
        requests.reserve(range_count);
        uint64_t staging_offset = 0;
        for (size_t i = 0; i < range_count; ++i) {
            const uint64_t file_offset = ranges[i][0];
            const uint64_t length = ranges[i][1];
            if (length == 0) {
                continue;
            }
            if (length > UINT32_MAX ||
                file_offset > file_size ||
                length > file_size - file_offset ||
                staging_offset > total_bytes ||
                length > total_bytes - staging_offset) {
                file->Release();
                engine->io_atomic.bytes_directstorage_failed.fetch_add(total_bytes, std::memory_order_relaxed);
                moe_io_path_perf_fail(engine->ds_staging_perf, 0);
                return 0;
            }

            DSTORAGE_REQUEST request{};
            request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
            request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_MEMORY;
            request.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT_NONE;
            request.Source.File.Source = file;
            request.Source.File.Offset = file_offset;
            request.Source.File.Size = static_cast<uint32_t>(length);
            request.Destination.Memory.Buffer = dst + staging_offset;
            request.Destination.Memory.Size = static_cast<uint32_t>(length);
            request.UncompressedSize = static_cast<uint32_t>(length);
            request.CancellationTag = 0;
            request.Name = "storageLLM tensor range";
            requests.push_back(request);
            staging_offset += length;
        }

        directstorage_enqueue_requests_status_and_event(
            engine,
            queue,
            requests,
            status_array,
            status_index);
        queue->Submit();
    }
    const uint64_t submit_done_us = moe_profile_now_us();
    if (submit_done_us > submit_start_us) {
        const uint64_t wait_ns = (submit_done_us - submit_start_us) * 1000ull;
        if (wait_ns > 100000ull) {
            engine->io_atomic.directstorage_queue_full_wait_ns.fetch_add(
                wait_ns,
                std::memory_order_relaxed);
        }
    }

    engine->io_atomic.bytes_directstorage_submitted.fetch_add(total_bytes, std::memory_order_relaxed);
    moe_io_path_perf_submit(engine->ds_staging_perf, total_bytes);
    status_guard.keep_reserved();
    out->status_array = status_array;
    out->status_index = status_index;
    out->slots = status_slots;
    out->slot_capacity = status_capacity;
    out->slot_index = acquired_status_slot;
    out->generation = status_slots[acquired_status_slot].generation;
    out->bytes = total_bytes;
    out->file = file;
    out->submitted_us = moe_profile_now_us();
    out->direct_to_gpu = false;
    return 1;
}

#else

static int directstorage_runtime_available() {
    return 0;
}

static int directstorage_queue_ensure(moe_pc_engine*) {
    return 0;
}

static void directstorage_queue_cleanup(moe_pc_engine*) {}

static void directstorage_maybe_recreate_queues(moe_pc_engine*) {}

static void directstorage_completion_worker_loop(moe_pc_engine*) {}

static void directstorage_release_pending_copy_resources(moe_dstorage_pending_copy* pending) {
    if (pending) {
        pending->status_array = nullptr;
        pending->slot_index = -1;
        pending->slot_capacity = 0;
        pending->slots = nullptr;
        pending->resource = nullptr;
        pending->file = nullptr;
    }
}

MOE_MAYBE_UNUSED static int directstorage_bind_d3d12_device(moe_pc_engine*, void*) {
    return 0;
}

MOE_MAYBE_UNUSED static int directstorage_direct_gpu_available(moe_pc_engine*) {
    return 0;
}

MOE_MAYBE_UNUSED static int directstorage_create_d3d12_buffer_resource(
    moe_pc_engine*,
    uint64_t,
    void**
) {
    return 0;
}

MOE_MAYBE_UNUSED static void directstorage_release_d3d12_resource(void*) {}

MOE_MAYBE_UNUSED static int directstorage_submit_ranges_to_gpu_buffer(
    moe_pc_engine*,
    const wchar_t*,
    const uint64_t[][2],
    size_t,
    void*,
    uint64_t,
    uint64_t,
    uint64_t,
    moe_dstorage_pending_copy*
) {
    return 0;
}

static int directstorage_poll_pending_copy(moe_pc_engine*, moe_dstorage_pending_copy*) {
    return -1;
}

static int directstorage_submit_ranges_to_staging(
    moe_pc_engine*,
    const wchar_t*,
    const uint64_t[][2],
    size_t,
    uint8_t*,
    uint64_t,
    uint64_t,
    moe_dstorage_pending_copy*
) {
    return 0;
}

#endif
