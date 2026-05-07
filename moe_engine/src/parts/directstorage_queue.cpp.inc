// Windows DirectStorage queue facade.
// The compiled path is enabled only when the SDK header is present and the build
// opts in. All other platforms keep the existing IOCP/IORing/mmap fallbacks.

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

static constexpr uint32_t moe_DIRECTSTORAGE_STATUS_CAPACITY = 1024u;
using moe_dstorage_get_factory_fn = HRESULT (WINAPI*)(REFIID, void**);

static void directstorage_release_unknown(void*& ptr) {
    if (ptr) {
        static_cast<IUnknown*>(ptr)->Release();
        ptr = nullptr;
    }
}

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

static int directstorage_queue_ensure(moe_pc_engine* engine) {
    if (!engine || !compiled_directstorage_available() || !directstorage_runtime_available()) {
        return 0;
    }
    if (engine->directstorage.initialized) {
        return engine->directstorage.available ? 1 : 0;
    }

    static std::mutex s_init_mutex;
    std::lock_guard<std::mutex> lock(s_init_mutex);
    if (engine->directstorage.initialized) {
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
        return 0;
    }

    DSTORAGE_QUEUE_DESC queue_desc{};
    queue_desc.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
    queue_desc.Capacity = 256;
    queue_desc.Priority = DSTORAGE_PRIORITY_NORMAL;
    queue_desc.Device = nullptr;
    queue_desc.Name = "storageLLM staging queue";

    IDStorageQueue* queue = nullptr;
    hr = factory->CreateQueue(&queue_desc, __uuidof(IDStorageQueue), reinterpret_cast<void**>(&queue));
    if (FAILED(hr) || !queue) {
        factory->Release();
        return 0;
    }

    IDStorageStatusArray* status_array = nullptr;
    hr = factory->CreateStatusArray(
        moe_DIRECTSTORAGE_STATUS_CAPACITY,
        "storageLLM staging statuses",
        __uuidof(IDStorageStatusArray),
        reinterpret_cast<void**>(&status_array));
    if (FAILED(hr) || !status_array) {
        queue->Release();
        factory->Release();
        return 0;
    }

    engine->directstorage.factory = factory;
    engine->directstorage.queue = queue;
    engine->directstorage.status_array = status_array;
    engine->directstorage.status_index = 0;
    engine->directstorage.available = true;
    return 1;
}

static void directstorage_queue_cleanup(moe_pc_engine* engine) {
    if (!engine) {
        return;
    }
    directstorage_release_unknown(engine->directstorage.gpu_status_array);
    directstorage_release_unknown(engine->directstorage.gpu_queue);
    directstorage_release_unknown(engine->directstorage.status_array);
    directstorage_release_unknown(engine->directstorage.queue);
    directstorage_release_unknown(engine->directstorage.d3d12_device);
    directstorage_release_unknown(engine->directstorage.factory);
    if (engine->directstorage.event_handle) {
        CloseHandle(static_cast<HANDLE>(engine->directstorage.event_handle));
        engine->directstorage.event_handle = nullptr;
    }
    engine->directstorage.initialized = false;
    engine->directstorage.available = false;
    engine->directstorage.gpu_available = false;
    engine->directstorage.status_index = 0;
    engine->directstorage.gpu_status_index = 0;
}

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
    return 1;
}

static int directstorage_gpu_queue_ensure(moe_pc_engine* engine) {
    if (!engine || !engine->directstorage.d3d12_device ||
        !directstorage_queue_ensure(engine)) {
        return 0;
    }
    if (engine->directstorage.gpu_queue &&
        engine->directstorage.gpu_status_array &&
        engine->directstorage.gpu_available) {
        return 1;
    }

    static std::mutex s_gpu_init_mutex;
    std::lock_guard<std::mutex> lock(s_gpu_init_mutex);
    if (engine->directstorage.gpu_queue &&
        engine->directstorage.gpu_status_array &&
        engine->directstorage.gpu_available) {
        return 1;
    }

    IDStorageFactory* factory = static_cast<IDStorageFactory*>(engine->directstorage.factory);
    if (!factory) {
        return 0;
    }

    DSTORAGE_QUEUE_DESC queue_desc{};
    queue_desc.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
    queue_desc.Capacity = 256;
    queue_desc.Priority = DSTORAGE_PRIORITY_NORMAL;
    queue_desc.Device = static_cast<ID3D12Device*>(engine->directstorage.d3d12_device);
    queue_desc.Name = "storageLLM GPU resource queue";

    IDStorageQueue* queue = nullptr;
    HRESULT hr = factory->CreateQueue(&queue_desc, __uuidof(IDStorageQueue), reinterpret_cast<void**>(&queue));
    if (FAILED(hr) || !queue) {
        engine->directstorage.gpu_available = false;
        return 0;
    }

    IDStorageStatusArray* status_array = nullptr;
    hr = factory->CreateStatusArray(
        moe_DIRECTSTORAGE_STATUS_CAPACITY,
        "storageLLM GPU resource statuses",
        __uuidof(IDStorageStatusArray),
        reinterpret_cast<void**>(&status_array));
    if (FAILED(hr) || !status_array) {
        queue->Release();
        engine->directstorage.gpu_available = false;
        return 0;
    }

    engine->directstorage.gpu_queue = queue;
    engine->directstorage.gpu_status_array = status_array;
    engine->directstorage.gpu_status_index = 0;
    engine->directstorage.gpu_available = true;
    return 1;
}

MOE_MAYBE_UNUSED static int directstorage_direct_gpu_available(moe_pc_engine* engine) {
    return directstorage_gpu_queue_ensure(engine);
}

MOE_MAYBE_UNUSED static int directstorage_read_ranges_to_gpu_buffer(
    moe_pc_engine* engine,
    const wchar_t* path,
    const uint64_t ranges[][2],
    size_t range_count,
    void* d3d12_resource,
    uint64_t resource_offset,
    uint64_t total_bytes,
    uint64_t file_size
) {
    if (!engine || !path || !path[0] || !ranges || !d3d12_resource ||
        total_bytes == 0 ||
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

    IDStorageFile* file = nullptr;
    HRESULT hr = factory->OpenFile(path, __uuidof(IDStorageFile), reinterpret_cast<void**>(&file));
    if (FAILED(hr) || !file) {
        engine->io_atomic.bytes_directstorage_failed.fetch_add(total_bytes, std::memory_order_relaxed);
        return 0;
    }
    resource->AddRef();

    uint32_t status_index = 0;
    {
        static std::mutex s_gpu_submit_mutex;
        std::lock_guard<std::mutex> submit_lock(s_gpu_submit_mutex);
        status_index = engine->directstorage.gpu_status_index++ % moe_DIRECTSTORAGE_STATUS_CAPACITY;

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
                return 0;
            }
            const uint64_t dst_offset = resource_offset + gpu_offset;
            if (length > UINT64_MAX - dst_offset) {
                resource->Release();
                file->Release();
                engine->io_atomic.bytes_directstorage_failed.fetch_add(total_bytes, std::memory_order_relaxed);
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
            queue->EnqueueRequest(&request);
            gpu_offset += length;
        }

        queue->EnqueueStatus(status_array, status_index);
        queue->Submit();
    }

    engine->io_atomic.bytes_directstorage_submitted.fetch_add(total_bytes, std::memory_order_relaxed);

    uint32_t spins = 0;
    while (!status_array->IsComplete(status_index)) {
        if (engine->io_stop.load(std::memory_order_relaxed)) {
            resource->Release();
            file->Release();
            engine->io_atomic.bytes_directstorage_failed.fetch_add(total_bytes, std::memory_order_relaxed);
            return 0;
        }
        if (++spins < 64) {
            std::this_thread::yield();
        } else {
            Sleep(1);
        }
    }

    hr = status_array->GetHResult(status_index);
    resource->Release();
    file->Release();
    if (FAILED(hr)) {
        engine->io_atomic.bytes_directstorage_failed.fetch_add(total_bytes, std::memory_order_relaxed);
        return 0;
    }

    engine->io_atomic.bytes_directstorage_completed.fetch_add(total_bytes, std::memory_order_relaxed);
    engine->io_atomic.bytes_direct_to_gpu.fetch_add(total_bytes, std::memory_order_relaxed);
    return 1;
}

static int directstorage_read_ranges_to_staging(
    moe_pc_engine* engine,
    const wchar_t* path,
    const uint64_t ranges[][2],
    size_t range_count,
    uint8_t* dst,
    uint64_t total_bytes,
    uint64_t file_size
) {
    if (!engine || !path || !path[0] || !ranges || !dst || total_bytes == 0 ||
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

    IDStorageFile* file = nullptr;
    HRESULT hr = factory->OpenFile(path, __uuidof(IDStorageFile), reinterpret_cast<void**>(&file));
    if (FAILED(hr) || !file) {
        engine->io_atomic.bytes_directstorage_failed.fetch_add(total_bytes, std::memory_order_relaxed);
        return 0;
    }

    uint32_t status_index = 0;
    {
        static std::mutex s_submit_mutex;
        std::lock_guard<std::mutex> submit_lock(s_submit_mutex);
        status_index = engine->directstorage.status_index++ % moe_DIRECTSTORAGE_STATUS_CAPACITY;

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
            queue->EnqueueRequest(&request);
            staging_offset += length;
        }

        queue->EnqueueStatus(status_array, status_index);
        queue->Submit();
    }

    engine->io_atomic.bytes_directstorage_submitted.fetch_add(total_bytes, std::memory_order_relaxed);

    uint32_t spins = 0;
    while (!status_array->IsComplete(status_index)) {
        if (engine->io_stop.load(std::memory_order_relaxed)) {
            file->Release();
            engine->io_atomic.bytes_directstorage_failed.fetch_add(total_bytes, std::memory_order_relaxed);
            return 0;
        }
        if (++spins < 64) {
            std::this_thread::yield();
        } else {
            Sleep(1);
        }
    }

    hr = status_array->GetHResult(status_index);
    file->Release();
    if (FAILED(hr)) {
        engine->io_atomic.bytes_directstorage_failed.fetch_add(total_bytes, std::memory_order_relaxed);
        return 0;
    }

    engine->io_atomic.bytes_directstorage_completed.fetch_add(total_bytes, std::memory_order_relaxed);
    return 1;
}

#else

static int directstorage_runtime_available() {
    return 0;
}

static void directstorage_queue_cleanup(moe_pc_engine*) {}

MOE_MAYBE_UNUSED static int directstorage_bind_d3d12_device(moe_pc_engine*, void*) {
    return 0;
}

MOE_MAYBE_UNUSED static int directstorage_direct_gpu_available(moe_pc_engine*) {
    return 0;
}

MOE_MAYBE_UNUSED static int directstorage_read_ranges_to_gpu_buffer(
    moe_pc_engine*,
    const wchar_t*,
    const uint64_t[][2],
    size_t,
    void*,
    uint64_t,
    uint64_t,
    uint64_t
) {
    return 0;
}

static int directstorage_read_ranges_to_staging(
    moe_pc_engine*,
    const wchar_t*,
    const uint64_t[][2],
    size_t,
    uint8_t*,
    uint64_t,
    uint64_t
) {
    return 0;
}

#endif
