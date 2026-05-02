// io_iocp_adapter.cpp.inc — IOCP-based async I/O for Windows pre-Win11 22H2
// Provides the same read interface as io_ring_adapter but uses
// CreateFile + ReadFile + IOCP instead of IORing.
// This eliminates the synchronous ifstream fallback that blocks io_worker threads.

#ifdef _WIN32

#include <windows.h>

struct moe_iocp_adapter_t {
    HANDLE iocp = INVALID_HANDLE_VALUE;
    bool initialized = false;

    // Per-request context for overlapped I/O
    struct iocp_request_t {
        OVERLAPPED overlapped;
        void* user_data;
        HANDLE file_handle;
        uint8_t* dst_buf;
        uint32_t bytes_requested;
        bool completed;
    };

    static constexpr uint32_t MAX_CONCURRENT = 64;
    iocp_request_t requests[MAX_CONCURRENT];
    uint32_t request_head = 0;

    bool init(uint32_t /*queue_depth*/) {
        if (initialized) return true;
        iocp = CreateIoCompletionPort(INVALID_HANDLE_VALUE, nullptr, 0, 0);
        if (iocp == nullptr || iocp == INVALID_HANDLE_VALUE) {
            iocp = INVALID_HANDLE_VALUE;
            return false;
        }
        memset(requests, 0, sizeof(requests));
        request_head = 0;
        initialized = true;
        return true;
    }

    // Associate a file handle with the IOCP
    bool associate_file(HANDLE hFile) {
        if (!initialized || hFile == INVALID_HANDLE_VALUE) return false;
        HANDLE result = CreateIoCompletionPort(hFile, iocp, (ULONG_PTR)hFile, 0);
        return result != nullptr;
    }

    // Submit an async read using OVERLAPPED I/O
    bool submit_read(HANDLE hFile, uint8_t* dst, uint64_t file_offset, uint32_t bytes, void* user_data) {
        if (!initialized || !dst || bytes == 0) return false;

        uint32_t slot = request_head % MAX_CONCURRENT;
        request_head++;

        iocp_request_t& req = requests[slot];
        memset(&req.overlapped, 0, sizeof(OVERLAPPED));
        req.overlapped.Offset = (DWORD)(file_offset & 0xFFFFFFFF);
        req.overlapped.OffsetHigh = (DWORD)(file_offset >> 32);
        req.user_data = user_data;
        req.file_handle = hFile;
        req.dst_buf = dst;
        req.bytes_requested = bytes;
        req.completed = false;

        BOOL ok = ReadFile(hFile, dst, bytes, nullptr, &req.overlapped);
        if (!ok) {
            DWORD err = GetLastError();
            if (err != ERROR_IO_PENDING) {
                // Synchronous completion or actual error
                return err == ERROR_SUCCESS;
            }
        }
        return true;
    }

    // Wait for one completion, returns user_data or nullptr on timeout
    void* wait_completion(uint32_t timeout_ms = 100) {
        if (!initialized) return nullptr;

        DWORD bytes_transferred = 0;
        ULONG_PTR completion_key = 0;
        OVERLAPPED* pOverlapped = nullptr;

        BOOL ok = GetQueuedCompletionStatus(iocp, &bytes_transferred, &completion_key, &pOverlapped, timeout_ms);
        if (!ok || !pOverlapped) {
            return nullptr;
        }

        // Find the request that owns this OVERLAPPED
        for (uint32_t i = 0; i < MAX_CONCURRENT; ++i) {
            if (&requests[i].overlapped == pOverlapped) {
                requests[i].completed = true;
                return requests[i].user_data;
            }
        }
        return (void*)1; // Unknown completion, treat as success
    }

    // Non-blocking peek
    void* peek_completion() {
        return wait_completion(0);
    }

    void cleanup() {
        if (initialized && iocp != INVALID_HANDLE_VALUE) {
            CloseHandle(iocp);
            iocp = INVALID_HANDLE_VALUE;
        }
        initialized = false;
    }
};

// Helper: Open a file with OVERLAPPED flag for async I/O
static HANDLE iocp_open_file_async(const char* path) {
    if (!path) return INVALID_HANDLE_VALUE;
    HANDLE h = CreateFileA(
        path,
        GENERIC_READ,
        FILE_SHARE_READ,
        nullptr,
        OPEN_EXISTING,
        FILE_FLAG_OVERLAPPED | FILE_FLAG_SEQUENTIAL_SCAN,
        nullptr
    );
    return h;
}

#endif // _WIN32
