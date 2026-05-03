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

static void moe_profile_resource_snapshot(
    uint64_t now_us,
    double* out_proc_cpu_core_pct,
    double* out_proc_cpu_host_pct,
    uint64_t* out_rss_bytes,
    uint64_t* out_avail_ram_bytes
) {
    if (out_proc_cpu_core_pct) {
        *out_proc_cpu_core_pct = 0.0;
    }
    if (out_proc_cpu_host_pct) {
        *out_proc_cpu_host_pct = 0.0;
    }
    if (out_rss_bytes) {
        *out_rss_bytes = storagellm::current_process_rss_bytes();
    }
    if (out_avail_ram_bytes) {
        *out_avail_ram_bytes = storagellm::available_ram_bytes();
    }
    const uint64_t cpu_us = storagellm::current_process_cpu_time_us();
    static std::atomic<uint64_t> last_wall_us{0};
    static std::atomic<uint64_t> last_cpu_us{0};
    const uint64_t prev_wall = last_wall_us.exchange(now_us, std::memory_order_relaxed);
    const uint64_t prev_cpu = last_cpu_us.exchange(cpu_us, std::memory_order_relaxed);
    if (!prev_wall || !prev_cpu || now_us <= prev_wall || cpu_us < prev_cpu) {
        return;
    }
    const double wall_delta = (double)(now_us - prev_wall);
    const double cpu_delta = (double)(cpu_us - prev_cpu);
    const double core_pct = wall_delta > 0.0 ? (100.0 * cpu_delta / wall_delta) : 0.0;
    const uint32_t hw = std::max<uint32_t>(1u, std::thread::hardware_concurrency());
    if (out_proc_cpu_core_pct) {
        *out_proc_cpu_core_pct = core_pct;
    }
    if (out_proc_cpu_host_pct) {
        *out_proc_cpu_host_pct = core_pct / (double)hw;
    }
}

static void moe_profile_log(const char* tag, const char* fmt, ...) {
    if (!moe_profile_enabled()) {
        return;
    }
    const uint64_t now_us = moe_profile_now_us();
    double proc_cpu_core_pct = 0.0;
    double proc_cpu_host_pct = 0.0;
    uint64_t rss_bytes = 0;
    uint64_t avail_ram_bytes = 0;
    moe_profile_resource_snapshot(
        now_us,
        &proc_cpu_core_pct,
        &proc_cpu_host_pct,
        &rss_bytes,
        &avail_ram_bytes);
    std::fprintf(stderr, "[storagellm trace] %s ", tag ? tag : "event");
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
    std::fprintf(stderr,
        " proc_cpu_core_pct=%.1f proc_cpu_host_pct=%.1f rss_bytes=%llu avail_ram_bytes=%llu\n",
        proc_cpu_core_pct,
        proc_cpu_host_pct,
        (unsigned long long)rss_bytes,
        (unsigned long long)avail_ram_bytes);
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
