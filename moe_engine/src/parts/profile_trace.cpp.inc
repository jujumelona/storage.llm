static int moe_profile_level() {
    static std::atomic<int> cached{-1};
    int value = cached.load(std::memory_order_acquire);
    if (value >= 0) {
        return value;
    }
    int parsed = 0;
    const char* env = std::getenv("STORAGELLM_TRACE_LEVEL");
    if (!env || !env[0]) {
        env = std::getenv("STORAGELLM_TRACE");
    }
    if (env && env[0]) {
        if (std::strcmp(env, "0") == 0 ||
            std::strcmp(env, "off") == 0 ||
            std::strcmp(env, "false") == 0 ||
            std::strcmp(env, "none") == 0) {
            parsed = 0;
        } else if (std::strcmp(env, "1") == 0) {
            parsed = 1;
        } else if (std::strcmp(env, "2") == 0) {
            parsed = 2;
        } else if (std::strcmp(env, "summary") == 0 ||
                   std::strcmp(env, "basic") == 0) {
            parsed = 1;
        } else {
            parsed = 2;
        }
    }
    cached.store(parsed, std::memory_order_release);
    return parsed;
}

static int moe_profile_enabled() {
    return moe_profile_level() > 0;
}

static int moe_profile_tag_enabled(const char* tag) {
    const int level = moe_profile_level();
    if (level <= 0) {
        return 0;
    }
    if (level >= 2 || !tag) {
        return 1;
    }
    static const char* const summary_tags[] = {
        "generation_contract",
        "generation_begin",
        "generation_end",
        "forward_embed",
        "prefill_batch_embed",
        "prefill_layer_attention_token",
        "prefill_layer_token",
        "prefill_layer_mlp",
        "prefill_layer",
        "prefill_mlp_batch_prepare",
        "prefill_mlp_batch_experts",
        "prefill_mlp_batch_finish",
        "prefill_end",
        "decode_forward",
        "decode_sample",
        "forward_token_end",
        "forward_layer",
        "ple_prepare",
        "embedding_scale_contract",
        "activation_mode",
        "rmsnorm_unit_offset",
        "router_scale_apply",
        "router_score_contract",
        "router_topk",
        "mlp_moe_begin",
        "mlp_moe_norm",
        "mlp_moe_dense_branch",
        "mlp_moe_shared",
        "mlp_moe_map",
        "mlp_moe_router",
        "mlp_moe_gate_up",
        "mlp_moe_fused_down_accum_end",
        "mlp_moe_end",
        "attn_standard_cfg",
        "attn_standard_map",
        "attn_standard_input_norm",
        "attn_standard_qkv",
        "attn_standard_qkv_norm",
        "attn_standard_rope",
        "attn_standard_kv",
        "attn_standard_qkv_append",
        "attn_standard_heads",
        "attn_standard_o_proj",
        "attn_standard_total",
        "cpu_ram",
        "kv_cache",
        "io_pipeline",
        "qkv_config",
        "raw_tensor_static_fallback_blocked",
        "lm_head_top1_begin",
        "lm_head_top1_end",
        "lm_head_logprob_time",
        "lm_head_logprob_end",
        "eval_input_ids",
        "eval_prepare_begin",
        "eval_prepare_end",
        "eval_token_logprob",
        "eval_done",
        "eval_end",
    };
    for (const char* allowed : summary_tags) {
        if (std::strcmp(tag, allowed) == 0) {
            return 1;
        }
    }
    return 0;
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
    static std::atomic<uint64_t> last_sample_us{0};
    static std::atomic<uint64_t> last_wall_us{0};
    static std::atomic<uint64_t> last_cpu_us{0};
    static std::atomic<uint64_t> cached_core_milli{0};
    static std::atomic<uint64_t> cached_host_milli{0};
    static std::atomic<uint64_t> cached_rss_bytes{0};
    static std::atomic<uint64_t> cached_avail_ram_bytes{0};

    const auto store_pct = [](double value) -> uint64_t {
        if (!std::isfinite(value) || value <= 0.0) {
            return 0;
        }
        const double capped = std::min<double>(value, 1000000.0);
        return (uint64_t)(capped * 1000.0 + 0.5);
    };
    const auto emit_cached = [&]() {
        if (out_proc_cpu_core_pct) {
            *out_proc_cpu_core_pct = (double)cached_core_milli.load(std::memory_order_relaxed) / 1000.0;
        }
        if (out_proc_cpu_host_pct) {
            *out_proc_cpu_host_pct = (double)cached_host_milli.load(std::memory_order_relaxed) / 1000.0;
        }
        if (out_rss_bytes) {
            *out_rss_bytes = cached_rss_bytes.load(std::memory_order_relaxed);
        }
        if (out_avail_ram_bytes) {
            *out_avail_ram_bytes = cached_avail_ram_bytes.load(std::memory_order_relaxed);
        }
    };

    const uint64_t sample_us = last_sample_us.load(std::memory_order_relaxed);
    if (sample_us && now_us >= sample_us && now_us - sample_us < 10000ull) {
        emit_cached();
        return;
    }
    uint64_t expected = sample_us;
    if (!last_sample_us.compare_exchange_strong(
            expected,
            now_us,
            std::memory_order_relaxed,
            std::memory_order_relaxed)) {
        emit_cached();
        return;
    }

    const uint64_t rss_bytes = storagellm::current_process_rss_bytes();
    const uint64_t avail_ram_bytes = storagellm::available_ram_bytes();
    const uint64_t cpu_us = storagellm::current_process_cpu_time_us();
    const uint64_t prev_wall = last_wall_us.exchange(now_us, std::memory_order_relaxed);
    const uint64_t prev_cpu = last_cpu_us.exchange(cpu_us, std::memory_order_relaxed);
    double core_pct = 0.0;
    double host_pct = 0.0;
    if (!prev_wall || !prev_cpu || now_us <= prev_wall || cpu_us < prev_cpu) {
        cached_core_milli.store(0, std::memory_order_relaxed);
        cached_host_milli.store(0, std::memory_order_relaxed);
        cached_rss_bytes.store(rss_bytes, std::memory_order_relaxed);
        cached_avail_ram_bytes.store(avail_ram_bytes, std::memory_order_relaxed);
        emit_cached();
        return;
    }
    const double wall_delta = (double)(now_us - prev_wall);
    const double cpu_delta = (double)(cpu_us - prev_cpu);
    core_pct = wall_delta > 0.0 ? (100.0 * cpu_delta / wall_delta) : 0.0;
    const uint32_t hw = std::max<uint32_t>(1u, std::thread::hardware_concurrency());
    host_pct = core_pct / (double)hw;
    cached_core_milli.store(store_pct(core_pct), std::memory_order_relaxed);
    cached_host_milli.store(store_pct(host_pct), std::memory_order_relaxed);
    cached_rss_bytes.store(rss_bytes, std::memory_order_relaxed);
    cached_avail_ram_bytes.store(avail_ram_bytes, std::memory_order_relaxed);
    emit_cached();
}

static void moe_profile_log(const char* tag, const char* fmt, ...) {
    if (!moe_profile_tag_enabled(tag)) {
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
    static std::atomic<uint64_t> last_flush_us{0};
    static std::atomic<int> cached_flush_ms{-1};
    int flush_ms = cached_flush_ms.load(std::memory_order_acquire);
    if (flush_ms < 0) {
        flush_ms = 1000;
        const char* env = std::getenv("STORAGELLM_TRACE_FLUSH_MS");
        if (env && env[0]) {
            char* end = nullptr;
            const long parsed = std::strtol(env, &end, 10);
            if (end && *end == '\0' && parsed >= 0 && parsed <= 60000) {
                flush_ms = (int)parsed;
            }
        }
        cached_flush_ms.store(flush_ms, std::memory_order_release);
    }
    const int force_flush =
        std::strcmp(tag ? tag : "", "generation_end") == 0 ||
        std::strcmp(tag ? tag : "", "generation_engine_end") == 0 ||
        std::strcmp(tag ? tag : "", "eval_done") == 0 ||
        std::strcmp(tag ? tag : "", "eval_end") == 0;
    const uint64_t flush_interval_us = (uint64_t)flush_ms * 1000ull;
    const uint64_t prev_flush = last_flush_us.load(std::memory_order_relaxed);
    if (force_flush ||
        flush_ms == 0 ||
        !prev_flush ||
        (now_us >= prev_flush && now_us - prev_flush >= flush_interval_us)) {
        last_flush_us.store(now_us, std::memory_order_relaxed);
        std::fflush(stderr);
    }
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
