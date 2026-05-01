#pragma once

#ifdef __linux__
#if defined(__has_include)
#if __has_include(<liburing.h>)
#define MOE_HAVE_LIBURING 1
#include <liburing.h>
#else
#define MOE_HAVE_LIBURING 0
#endif
#else
#define MOE_HAVE_LIBURING 1
#include <liburing.h>
#endif
#include <unistd.h>
#include <sys/mman.h>
#elif defined(_WIN32)
#include <windows.h>
#include <ioringapi.h>
#endif

#include <vector>

#if defined(_WIN32)
struct moe_win_ioring_api_t {
    using create_fn = HRESULT (WINAPI*)(
        IORING_VERSION,
        IORING_CREATE_FLAGS,
        UINT32,
        UINT32,
        HIORING*
    );
    using register_buffers_fn = HRESULT (WINAPI*)(
        HIORING,
        UINT32,
        IORING_BUFFER_INFO const*,
        UINT_PTR
    );
    using read_file_fn = HRESULT (WINAPI*)(
        HIORING,
        IORING_HANDLE_REF,
        IORING_BUFFER_REF,
        UINT32,
        UINT64,
        UINT_PTR,
        IORING_SQE_FLAGS
    );
    using submit_fn = HRESULT (WINAPI*)(HIORING, UINT32, UINT32, UINT32*);
    using pop_completion_fn = HRESULT (WINAPI*)(HIORING, IORING_CQE*);
    using close_fn = HRESULT (WINAPI*)(HIORING);

    create_fn create = nullptr;
    register_buffers_fn register_buffers = nullptr;
    read_file_fn read_file = nullptr;
    submit_fn submit = nullptr;
    pop_completion_fn pop_completion = nullptr;
    close_fn close = nullptr;
    bool available = false;
};

static const moe_win_ioring_api_t* moe_win_ioring_api() {
    static moe_win_ioring_api_t api{};
    static std::once_flag once;
    std::call_once(once, []() {
        HMODULE kernel = GetModuleHandleW(L"kernel32.dll");
        if (!kernel) {
            kernel = LoadLibraryW(L"kernel32.dll");
        }
        if (!kernel) {
            return;
        }
        api.create = reinterpret_cast<moe_win_ioring_api_t::create_fn>(
            GetProcAddress(kernel, "CreateIoRing"));
        api.register_buffers = reinterpret_cast<moe_win_ioring_api_t::register_buffers_fn>(
            GetProcAddress(kernel, "BuildIoRingRegisterBuffers"));
        api.read_file = reinterpret_cast<moe_win_ioring_api_t::read_file_fn>(
            GetProcAddress(kernel, "BuildIoRingReadFile"));
        api.submit = reinterpret_cast<moe_win_ioring_api_t::submit_fn>(
            GetProcAddress(kernel, "SubmitIoRing"));
        api.pop_completion = reinterpret_cast<moe_win_ioring_api_t::pop_completion_fn>(
            GetProcAddress(kernel, "PopIoRingCompletion"));
        api.close = reinterpret_cast<moe_win_ioring_api_t::close_fn>(
            GetProcAddress(kernel, "CloseIoRing"));
        api.available = api.create && api.register_buffers && api.read_file &&
            api.submit && api.pop_completion && api.close;
    });
    return api.available ? &api : nullptr;
}
#endif

struct moe_io_ring_adapter {
#if defined(__linux__) && MOE_HAVE_LIBURING
    struct io_uring ring;
    bool initialized = false;
#elif defined(_WIN32)
    HIORING ring = nullptr;
    bool initialized = false;
#else
    bool initialized = false;
#endif

    int init(uint32_t queue_depth) {
#if defined(__linux__) && MOE_HAVE_LIBURING
        if (io_uring_queue_init(queue_depth, &ring, 0) == 0) {
            initialized = true;
            return 1;
        }
#elif defined(_WIN32)
        const moe_win_ioring_api_t* api = moe_win_ioring_api();
        if (!api) {
            return 0;
        }
        IORING_CREATE_FLAGS flags{};
        flags.Required = IORING_CREATE_REQUIRED_FLAGS_NONE;
        flags.Advisory = IORING_CREATE_ADVISORY_FLAGS_NONE;
        if (api->create(IORING_VERSION_3, flags, queue_depth, queue_depth, &ring) == S_OK) {
            initialized = true;
            return 1;
        }
#endif
        return 0;
    }

    int register_buffers(void* buffers[], size_t lengths[], uint32_t count) {
        if (!initialized) return 0;
#if defined(__linux__) && MOE_HAVE_LIBURING
        std::vector<struct iovec> iovecs(count);
        for (uint32_t i = 0; i < count; ++i) {
            iovecs[i].iov_base = buffers[i];
            iovecs[i].iov_len = lengths[i];
        }
        return io_uring_register_buffers(&ring, iovecs.data(), count) == 0 ? 1 : 0;
#elif defined(_WIN32)
        const moe_win_ioring_api_t* api = moe_win_ioring_api();
        if (!api) {
            return 0;
        }
        std::vector<IORING_BUFFER_INFO> win_bufs(count);
        for (uint32_t i = 0; i < count; ++i) {
            win_bufs[i].Address = buffers[i];
            win_bufs[i].Length = static_cast<UINT32>(lengths[i]);
        }
        return api->register_buffers(ring, count, win_bufs.data(), 0) == S_OK ? 1 : 0;
#endif
        return 0;
    }

    // Fix 6: Added buf_offset to prevent Windows IORing data corruption
    int prep_read_fixed(int fd, void* file_handle, void* buf, size_t len, uint64_t offset, uint32_t buf_index, void* user_data, uint64_t buf_offset = 0) {
        if (!initialized) return 0;
#if defined(__linux__) && MOE_HAVE_LIBURING
        struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
        if (!sqe) return 0;
        io_uring_prep_read_fixed(sqe, fd, buf, len, offset, buf_index);
        io_uring_sqe_set_data(sqe, user_data);
        return 1;
#elif defined(_WIN32)
        const moe_win_ioring_api_t* api = moe_win_ioring_api();
        if (!api) {
            return 0;
        }
        IORING_HANDLE_REF fileRef = IoRingHandleRefFromHandle((HANDLE)file_handle);
        IORING_BUFFER_REF dataRef = IoRingBufferRefFromIndexAndOffset(buf_index, static_cast<UINT32>(buf_offset));
        if (api->read_file(ring, fileRef, dataRef, static_cast<UINT32>(len), offset, reinterpret_cast<UINT_PTR>(user_data), IOSQE_FLAGS_NONE) == S_OK) {
            return 1;
        }
#endif
        return 0;
    }

    int submit() {
        if (!initialized) return 0;
#if defined(__linux__) && MOE_HAVE_LIBURING
        return io_uring_submit(&ring);
#elif defined(_WIN32)
        const moe_win_ioring_api_t* api = moe_win_ioring_api();
        if (!api) {
            return 0;
        }
        UINT32 submitted = 0;
        api->submit(ring, 0, 0, &submitted);
        return submitted;
#endif
        return 0;
    }

    // Fix 7: CQE error code missing check
    void* peek_completion() {
        if (!initialized) return nullptr;
#if defined(__linux__) && MOE_HAVE_LIBURING
        struct io_uring_cqe* cqe;
        if (io_uring_peek_cqe(&ring, &cqe) == 0 && cqe) {
            void* user_data = io_uring_cqe_get_data(cqe);
            const int res = cqe->res;
            io_uring_cqe_seen(&ring, cqe);
            return res >= 0 ? user_data : nullptr;
        }
#elif defined(_WIN32)
        const moe_win_ioring_api_t* api = moe_win_ioring_api();
        if (!api) {
            return nullptr;
        }
        IORING_CQE cqe;
        if (api->pop_completion(ring, &cqe) == S_OK) {
            if (static_cast<INT32>(cqe.ResultCode) < 0) return nullptr;
            return reinterpret_cast<void*>(cqe.UserData);
        }
#endif
        return nullptr;
    }

    void cleanup() {
        if (!initialized) return;
#if defined(__linux__) && MOE_HAVE_LIBURING
        io_uring_queue_exit(&ring);
#elif defined(_WIN32)
        const moe_win_ioring_api_t* api = moe_win_ioring_api();
        if (api) {
            api->close(ring);
        }
        ring = nullptr;
#endif
        initialized = false;
    }
};


