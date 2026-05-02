static int moe_profile_enabled() {
    static const int enabled = []() -> int {
        const char* value = std::getenv("STORAGELLM_TRACE");
        if (!value || !value[0]) {
            return 0;
        }
        const unsigned char c = static_cast<unsigned char>(value[0]);
        return (c == '1' || c == 'y' || c == 'Y' || c == 't' || c == 'T');
    }();
    return enabled;
}

static uint64_t moe_profile_now_us() {
    using clock = std::chrono::steady_clock;
    return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
        clock::now().time_since_epoch()).count();
}

static double moe_profile_elapsed_ms(uint64_t start_us) {
    const uint64_t now_us = moe_profile_now_us();
    return now_us >= start_us ? (double)(now_us - start_us) / 1000.0 : 0.0;
}

static void moe_profile_log(const char* tag, const char* fmt, ...) {
    if (!moe_profile_enabled()) {
        return;
    }
    std::fprintf(stderr, "[storagellm trace] %s ", tag ? tag : "event");
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
    std::fprintf(stderr, "\n");
    std::fflush(stderr);
}

static std::atomic<uint64_t> g_moe_profile_generation_id{0};

static uint64_t moe_profile_next_generation_id() {
    return g_moe_profile_generation_id.fetch_add(1, std::memory_order_relaxed) + 1u;
}

static thread_local const char* g_moe_profile_parallel_label = nullptr;

static const char* moe_profile_parallel_label() {
    return g_moe_profile_parallel_label ? g_moe_profile_parallel_label : "unlabeled";
}

struct moe_profile_parallel_label_scope {
    const char* previous = nullptr;
    explicit moe_profile_parallel_label_scope(const char* label) {
        previous = g_moe_profile_parallel_label;
        g_moe_profile_parallel_label = label ? label : "unlabeled";
    }
    ~moe_profile_parallel_label_scope() {
        g_moe_profile_parallel_label = previous;
    }
};
