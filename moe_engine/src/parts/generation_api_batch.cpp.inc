// ============================================================================
// Real continuous batching executor (safe, per-request scratch, incremental)
// ============================================================================
// This path is deliberately not a fake loop around moe_pc_engine_generate_token_ids().
// Requests are admitted into an always-built scheduler and each request owns an
// independent decode scratch/KV state that persists across moe_pc_engine_step_batch()
// calls.  Each batch step processes a bounded number of prompt/decode tokens across
// active requests, so long prompts and long decodes do not monopolize the engine.
//
// This path no longer takes the legacy generation_api_mutex. Shared IO state
// that is still engine-global is guarded by shared_forward_io_mutex inside each
// step while request scratch/KV remains per request.

#include <vector>
#include <memory>
#include <atomic>
#include <unordered_map>
#include <mutex>
#include <cstdint>
#include <algorithm>

static void moe_batch_erase_runtime_state_by_request_id(uint64_t request_id);
static uint64_t moe_batch_runtime_state_count();

static uint32_t moe_batch_env_u32_local(const char* name, uint32_t fallback, uint32_t lo, uint32_t hi) {
    const char* v = name ? std::getenv(name) : nullptr;
    if (!v || !*v) return fallback;
    char* end = nullptr;
    unsigned long parsed = std::strtoul(v, &end, 10);
    if (end == v) return fallback;
    if (parsed < lo) parsed = lo;
    if (parsed > hi) parsed = hi;
    return static_cast<uint32_t>(parsed);
}

static int moe_batch_env_truthy_local(const char* name, int fallback = 0) {
    const char* v = name ? std::getenv(name) : nullptr;
    if (!v || !*v) return fallback;
    return !(std::strcmp(v, "0") == 0 || std::strcmp(v, "false") == 0 ||
             std::strcmp(v, "FALSE") == 0 || std::strcmp(v, "off") == 0 ||
             std::strcmp(v, "OFF") == 0 || std::strcmp(v, "no") == 0 ||
             std::strcmp(v, "NO") == 0);
}

struct moe_batch_qkv_mode_scope_t {
    bool restore = false;
    bool previous = false;
    explicit moe_batch_qkv_mode_scope_t(bool allow_global_qkv) {
        // Real continuous batching owns per-request float KV in moe_decode_scratch.
        // The legacy QKV path keeps layer/head state on the engine object, keyed only
        // by layer/head/context epoch, so it is not request-isolated yet.  Default to
        // plain per-request KV for correctness and to avoid resetting the global QKV
        // cache on every batch step. Operators can opt back in while developing a
        // request-scoped QKV allocator with STORAGELLM_BATCH_ALLOW_GLOBAL_QKV=1.
        // QKV-only format roots must never be forced down to plain KV by batching.
        if (!allow_global_qkv && !qkv_is_enabled()) {
            previous = qkv_is_enabled();
        }
    }
    ~moe_batch_qkv_mode_scope_t() {
        if (restore) qkv_set_mode(previous);
    }
    moe_batch_qkv_mode_scope_t(const moe_batch_qkv_mode_scope_t&) = delete;
    moe_batch_qkv_mode_scope_t& operator=(const moe_batch_qkv_mode_scope_t&) = delete;
};

int moe_pc_engine_submit_request_ex(
    moe_pc_engine_t* engine,
    uint64_t request_id,
    const int32_t* prompt_tokens,
    uint32_t token_count,
    uint32_t max_new_tokens,
    uint32_t eos_token_id,
    uint32_t stop_on_eos
) {
    if (!engine || !prompt_tokens || token_count == 0 || max_new_tokens == 0) return 0;
    moe_batch_request_t* req = new (std::nothrow) moe_batch_request_t();
    if (!req) return 0;
    req->request_id = request_id;
    req->prompt_tokens.assign(prompt_tokens, prompt_tokens + token_count);
    req->phase = moe_REQ_WAITING;
    req->max_new_tokens = std::min<uint32_t>(max_new_tokens, 16384u);
    req->eos_token_id = eos_token_id;
    req->stop_on_eos = stop_on_eos ? 1u : 0u;
    req->generated_tokens.reserve(req->max_new_tokens);

    engine->batch_scheduler.add_request(req);
    std::fprintf(stderr,
        "[storageLLM batch] request %llu queued tokens=%u max_new=%u eos=%u stop=%u\n",
        (unsigned long long)request_id,
        token_count,
        req->max_new_tokens,
        req->eos_token_id,
        req->stop_on_eos);
    return 1;
}

int moe_pc_engine_submit_request(
    moe_pc_engine_t* engine,
    uint64_t request_id,
    const int32_t* prompt_tokens,
    uint32_t token_count
) {
    return moe_pc_engine_submit_request_ex(
        engine,
        request_id,
        prompt_tokens,
        token_count,
        moe_batch_env_u32_local("STORAGELLM_BATCH_MAX_NEW_TOKENS", 128u, 1u, 16384u),
        moe_batch_env_u32_local("STORAGELLM_BATCH_EOS_TOKEN_ID", 0u, 0u, UINT32_MAX),
        moe_batch_env_truthy_local("STORAGELLM_BATCH_STOP_ON_EOS", 1) ? 1u : 0u);
}

int moe_pc_engine_fetch_request_result(
    moe_pc_engine_t* engine,
    uint64_t request_id,
    int32_t* out_token_ids,
    uint32_t out_token_capacity,
    uint32_t* out_token_count,
    uint32_t* out_prompt_token_count,
    int* out_failed,
    int remove_after_fetch
) {
    if (!engine) return 0;
    return moe_batch_scheduler_fetch_result(
        &engine->batch_scheduler,
        request_id,
        out_token_ids,
        out_token_capacity,
        out_token_count,
        out_prompt_token_count,
        out_failed,
        remove_after_fetch);
}

int moe_pc_engine_cancel_request(
    moe_pc_engine_t* engine,
    uint64_t request_id
) {
    if (!engine) return 0;
    const int cancelled = moe_batch_scheduler_cancel_request(&engine->batch_scheduler, request_id);
    // A cancelled running request may never be visited by the executor again, so
    // do not leave its per-request KV/scratch state in the static runtime map.
    if (cancelled) {
        moe_batch_erase_runtime_state_by_request_id(request_id);
    }
    return cancelled;
}

int moe_pc_engine_wait_request_result(
    moe_pc_engine_t* engine,
    uint64_t request_id,
    uint32_t timeout_ms
) {
    if (!engine) return 0;
    return moe_batch_scheduler_wait_for_result(&engine->batch_scheduler, request_id, timeout_ms);
}

struct moe_batch_runtime_state_t {
    moe_pc_engine_t* engine = nullptr;
    moe_batch_request_t* req = nullptr;
    moe_decode_scratch scratch;
    moe_generation_stats_t stats{};
    int pre_allocate_kv = 0;
    uint32_t prefill_count = 0;
    uint32_t hidden_size = 0;
    uint32_t vocab_size = 0;
    int initialized = 0;
    int failed = 0;
    float last_logit = 0.0f;

    moe_batch_runtime_state_t(moe_pc_engine_t* e, moe_batch_request_t* r)
        : engine(e), req(r) {}

    ~moe_batch_runtime_state_t() {
        if (engine) {
            moe_kv_cache_release_ram(engine, &scratch.kv_cache);
            moe_kv_decode_cache_release_ram(engine, &scratch);
        }
    }
};

static std::mutex& moe_batch_state_mutex() {
    static std::mutex m;
    return m;
}

static std::unordered_map<uintptr_t, std::unique_ptr<moe_batch_runtime_state_t>>& moe_batch_state_map() {
    static std::unordered_map<uintptr_t, std::unique_ptr<moe_batch_runtime_state_t>> states;
    return states;
}

static uintptr_t moe_batch_state_key(const moe_batch_request_t* req) {
    return reinterpret_cast<uintptr_t>(req);
}

static uint64_t moe_batch_runtime_state_count() {
    std::lock_guard<std::mutex> lock(moe_batch_state_mutex());
    return (uint64_t)moe_batch_state_map().size();
}

static void moe_batch_erase_runtime_state_by_request_id(uint64_t request_id) {
    std::lock_guard<std::mutex> lock(moe_batch_state_mutex());
    auto& map = moe_batch_state_map();
    for (auto it = map.begin(); it != map.end(); ) {
        const moe_batch_runtime_state_t* st = it->second.get();
        if (st && st->req && st->req->request_id == request_id) {
            it = map.erase(it);
        } else {
            ++it;
        }
    }
}

static int moe_batch_init_runtime_state(
    moe_batch_runtime_state_t* st,
    const moe_storage_constants_t* c,
    const moe_attn_config_t* attn_cfg
) {
    if (!st || !st->engine || !st->req || !c || !attn_cfg) return 0;
    if (st->initialized) return 1;
    if (st->req->prompt_tokens.empty()) return 0;
    st->hidden_size = c->hidden_size;
    st->vocab_size = c->vocab_size;
    st->prefill_count = (uint32_t)st->req->prompt_tokens.size() - 1u;
    st->pre_allocate_kv = moe_generation_should_preallocate_kv_cache(
        st->engine, c->num_hidden_layers, moe_attn_kv_entry_dim(*attn_cfg), attn_cfg->max_seq_len);

    if (!moe_decode_scratch_resize(&st->scratch, c, attn_cfg, st->pre_allocate_kv, st->engine)) {
        moe_generation_set_error(&st->stats, "batch decode scratch allocation failed");
        return 0;
    }
    if (!moe_engine_qkv_only_required(st->engine)) {
        moe_kv_cache_track_ram(st->engine, &st->scratch.kv_cache);
    }
    st->scratch.position = st->req->processed_tokens;
    st->scratch.session_router_entropy_milli = st->req->smoothed_router_entropy_milli;
    st->scratch.session_adaptive_prefetch_depth = st->req->smoothed_adaptive_prefetch_depth;

    int32_t current = st->req->prompt_tokens.back();
    if (current < 0 || (uint32_t)current >= c->vocab_size) {
        current = (int32_t)st->req->eos_token_id;
    }
    st->req->current_token = current;
    st->initialized = 1;
    return 1;
}

static moe_batch_runtime_state_t* moe_batch_get_runtime_state(
    moe_pc_engine_t* engine,
    moe_batch_request_t* req,
    const moe_storage_constants_t* c,
    const moe_attn_config_t* attn_cfg
) {
    if (!engine || !req || !c || !attn_cfg) return nullptr;
    std::lock_guard<std::mutex> lock(moe_batch_state_mutex());
    auto& map = moe_batch_state_map();
    const uintptr_t key = moe_batch_state_key(req);
    auto it = map.find(key);
    if (it == map.end()) {
        std::unique_ptr<moe_batch_runtime_state_t> fresh(new (std::nothrow) moe_batch_runtime_state_t(engine, req));
        if (!fresh) return nullptr;
        auto inserted = map.emplace(key, std::move(fresh));
        it = inserted.first;
    }
    moe_batch_runtime_state_t* st = it->second.get();
    if (!moe_batch_init_runtime_state(st, c, attn_cfg)) {
        st->failed = 1;
        return nullptr;
    }
    return st;
}

static void moe_batch_erase_runtime_state(moe_batch_request_t* req) {
    if (!req) return;
    std::lock_guard<std::mutex> lock(moe_batch_state_mutex());
    moe_batch_state_map().erase(moe_batch_state_key(req));
}

static uint32_t moe_batch_safe_prompt_token(
    const moe_batch_runtime_state_t* st,
    const moe_storage_constants_t* c,
    uint32_t index
) {
    if (!st || !st->req || !c || index >= st->req->prompt_tokens.size()) return 0;
    int32_t token = st->req->prompt_tokens[index];
    if (token < 0 || (uint32_t)token >= c->vocab_size) return st->req->eos_token_id;
    return (uint32_t)token;
}

static void moe_batch_save_session_state(moe_batch_runtime_state_t* st) {
    if (!st || !st->req) return;
    st->req->smoothed_router_entropy_milli = st->scratch.session_router_entropy_milli;
    st->req->smoothed_adaptive_prefetch_depth = st->scratch.session_adaptive_prefetch_depth;
}

static void moe_batch_finish_request(moe_batch_runtime_state_t* st, moe_request_phase_t phase) {
    if (!st || !st->req) return;
    st->req->phase = phase;
    if (phase == moe_REQ_FAILED) {
        st->req->failed = 1;
        st->failed = 1;
    }
    moe_batch_save_session_state(st);
    moe_batch_erase_runtime_state(st->req);
}

struct moe_batch_step_scope_t {
    moe_pc_engine_t* engine = nullptr;
    uint64_t id = 0;
    uint64_t start_us = 0;
    int progressed = 0;
    explicit moe_batch_step_scope_t(moe_pc_engine_t* e, bool reset_global_qkv, bool reset_staging) : engine(e) {
        if (!engine) return;
        id = moe_profile_next_generation_id();
        start_us = moe_profile_now_us();
        engine->generation_started.fetch_add(1, std::memory_order_relaxed);
        engine->generation_active.fetch_add(1, std::memory_order_relaxed);
        engine->generation_phase.store(moe_GEN_PHASE_INIT, std::memory_order_relaxed);
        engine->generation_layer.store(0xffffffffu, std::memory_order_relaxed);
        engine->generation_token.store(0, std::memory_order_relaxed);
        // Do not reset global caches on every batch tick by default.  That was
        // correct for a single monolithic generation call, but it destroys reuse in
        // incremental continuous batching. QKV is disabled by default in this path
        // until it is moved into per-request scratch; staging slots are protected by
        // their own slot mutex/CV and are reset only when explicitly requested.
        if (reset_global_qkv) moe_engine_reset_qkv_cache(engine);
        if (reset_staging) reset_staging_slots(engine);
    }
    ~moe_batch_step_scope_t() {
        if (!engine) return;
        if (progressed) {
            engine->generation_completed.fetch_add(1, std::memory_order_relaxed);
            engine->generation_phase.store(moe_GEN_PHASE_DONE, std::memory_order_relaxed);
        } else {
            engine->generation_phase.store(moe_GEN_PHASE_IDLE, std::memory_order_relaxed);
        }
        uint32_t active = engine->generation_active.load(std::memory_order_relaxed);
        while (active > 0 && !engine->generation_active.compare_exchange_weak(
                active, active - 1u, std::memory_order_release, std::memory_order_relaxed)) {}
        engine->generation_layer.store(0xffffffffu, std::memory_order_relaxed);
        moe_profile_log(
            "batch_step_end",
            "batch=%llu progressed=%d active=%u ms=%.3f",
            (unsigned long long)id,
            progressed,
            engine->generation_active.load(std::memory_order_relaxed),
            moe_profile_elapsed_ms(start_us));
    }
};

static int moe_batch_process_one_prefill_token(
    moe_pc_engine_t* engine,
    moe_batch_runtime_state_t* st,
    const moe_storage_constants_t* c
) {
    if (!engine || !st || !st->req || !c) return 0;
    if (st->req->processed_tokens >= st->prefill_count) {
        st->req->phase = moe_REQ_DECODE;
        return 0;
    }
    const uint32_t pos = st->req->processed_tokens;
    const uint32_t token = moe_batch_safe_prompt_token(st, c, pos);
    engine->generation_phase.store(moe_GEN_PHASE_PREFILL, std::memory_order_relaxed);
    engine->generation_token.store(pos, std::memory_order_relaxed);
    if (!moe_forward_one_token_dispatch_f32(
            engine, token, &st->scratch, &st->stats,
            st->scratch.position, st->pre_allocate_kv)) {
        moe_generation_set_error(&st->stats, "batch prefill token failed");
        moe_batch_finish_request(st, moe_REQ_FAILED);
        return 0;
    }
    ++st->scratch.position;
    ++st->req->processed_tokens;
    moe_batch_save_session_state(st);
    if (st->req->processed_tokens >= st->prefill_count) {
        st->req->phase = moe_REQ_DECODE;
    }
    return 1;
}

static int moe_batch_process_one_decode_token(
    moe_pc_engine_t* engine,
    moe_batch_runtime_state_t* st,
    const moe_storage_constants_t* c
) {
    if (!engine || !st || !st->req || !c) return 0;
    if (st->req->produced_tokens >= st->req->max_new_tokens) {
        moe_batch_finish_request(st, moe_REQ_FINISHED);
        return 0;
    }
    engine->generation_phase.store(moe_GEN_PHASE_ATTENTION, std::memory_order_relaxed);
    engine->generation_token.store(st->prefill_count + st->req->produced_tokens, std::memory_order_relaxed);
    if (!moe_forward_one_token_dispatch_f32(
            engine, (uint32_t)st->req->current_token, &st->scratch, &st->stats,
            st->scratch.position, st->pre_allocate_kv)) {
        moe_generation_set_error(&st->stats, "batch decode token failed");
        moe_batch_finish_request(st, moe_REQ_FAILED);
        return 0;
    }
    engine->generation_phase.store(moe_GEN_PHASE_LM_HEAD, std::memory_order_relaxed);
    uint32_t next_token = 0;
    float next_logit = 0.0f;
    if (!moe_pc_engine_lm_head_top1_f32(
            engine, st->scratch.hidden.data(), c->hidden_size,
            &next_token, &next_logit)) {
        moe_generation_set_error(&st->stats, "batch lm_head top1 failed");
        moe_batch_finish_request(st, moe_REQ_FAILED);
        return 0;
    }
    st->req->generated_tokens.push_back((int32_t)next_token);
    st->req->current_token = (int32_t)next_token;
    st->last_logit = next_logit;
    ++st->req->produced_tokens;
    ++st->scratch.position;
    moe_batch_save_session_state(st);
    moe_profile_log(
        "batch_decode_token",
        "request=%llu produced=%u next=%u",
        (unsigned long long)st->req->request_id,
        st->req->produced_tokens,
        next_token);
    if ((st->req->stop_on_eos && next_token == st->req->eos_token_id) ||
        st->req->produced_tokens >= st->req->max_new_tokens) {
        moe_batch_finish_request(st, moe_REQ_FINISHED);
    }
    return 1;
}

int moe_pc_engine_step_batch(moe_pc_engine_t* engine) {
    if (!engine) return 0;
    if (!moe_perf_real_continuous_batch_exec_requested()) {
        static std::atomic<uint32_t> warned{0};
        if (warned.fetch_add(1, std::memory_order_relaxed) == 0) {
            std::fprintf(stderr,
                "[storageLLM batch] real continuous batch executor is linked but disabled. "
                "Set STORAGELLM_ENABLE_REAL_CONTINUOUS_BATCH_EXEC=1 to run it.\n");
        }
        return 0;
    }

    std::vector<moe_batch_request_t*> prefill_reqs;
    std::vector<moe_batch_request_t*> decode_reqs;
    engine->batch_scheduler.schedule_next_iteration(prefill_reqs, decode_reqs);
    if (prefill_reqs.empty() && decode_reqs.empty()) {
        return 0;
    }

    // Keep a narrow correctness lock for shared IO/forward resources only.
    // Request-local scratch/KV stays outside the legacy generation API mutex.
    std::unique_lock<std::mutex> shared_io_lock(engine->shared_forward_io_mutex);

    const bool allow_global_qkv = moe_batch_env_truthy_local("STORAGELLM_BATCH_ALLOW_GLOBAL_QKV", 0) != 0;
    const bool reset_global_qkv = allow_global_qkv &&
        moe_batch_env_truthy_local("STORAGELLM_BATCH_RESET_GLOBAL_QKV_EVERY_STEP", 0) != 0;
    const bool reset_staging = moe_batch_env_truthy_local("STORAGELLM_BATCH_RESET_STAGING_EVERY_STEP", 0) != 0;
    moe_batch_qkv_mode_scope_t qkv_scope(allow_global_qkv);

    const moe_storage_constants_t runtime_c = moe_engine_storage_constants_or_static(engine);
    const moe_storage_constants_t* c = &runtime_c;
    const moe_attn_config_t attn_cfg = moe_generation_attn_config(engine);
    if (!c || c->hidden_size == 0 || c->vocab_size == 0 || c->num_hidden_layers == 0) {
        return 0;
    }

    moe_batch_step_scope_t scope(engine, reset_global_qkv, reset_staging);
    uint32_t progressed = 0;
    uint32_t failed = 0;

    for (auto* req : prefill_reqs) {
        if (!req || req->phase != moe_REQ_PREFILL) continue;
        moe_batch_runtime_state_t* st = moe_batch_get_runtime_state(engine, req, c, &attn_cfg);
        if (!st) {
            req->phase = moe_REQ_FAILED;
            req->failed = 1;
            ++failed;
            continue;
        }
        progressed += (uint32_t)moe_batch_process_one_prefill_token(engine, st, c);
        if (req->phase == moe_REQ_FAILED) ++failed;
    }

    for (auto* req : decode_reqs) {
        if (!req || req->phase != moe_REQ_DECODE) continue;
        moe_batch_runtime_state_t* st = moe_batch_get_runtime_state(engine, req, c, &attn_cfg);
        if (!st) {
            req->phase = moe_REQ_FAILED;
            req->failed = 1;
            ++failed;
            continue;
        }
        progressed += (uint32_t)moe_batch_process_one_decode_token(engine, st, c);
        if (req->phase == moe_REQ_FAILED) ++failed;
    }

    if (failed > 0) {
        engine->generation_failed.fetch_add(failed, std::memory_order_relaxed);
    }
    scope.progressed = progressed ? 1 : 0;
    engine->batch_scheduler.cleanup_finished();
    return progressed > 0 ? 1 : 0;
}
