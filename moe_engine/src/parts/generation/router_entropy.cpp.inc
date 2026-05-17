static void moe_save_gate_input_snapshot(
    moe_pc_engine* engine,
    int32_t layer,
    const float* hidden,
    uint32_t hidden_size
) {
    if (!engine || !hidden || layer < 0 || hidden_size == 0) {
        return;
    }
    const size_t slot = (size_t)layer;
    if (slot > moe_GATE_SNAPSHOT_MAX_LAYER) return;
    if (hidden_size > 100000) return;

    const size_t req_size = (slot + 1) * 2;
    {
        std::lock_guard<std::mutex> lock(engine->gate_input_snapshot_mutex);
        if (engine->gate_input_snapshots.size() < req_size) {
            engine->gate_input_snapshots.resize(req_size);
        }

        const size_t slot0 = slot * 2;
        const size_t slot1 = slot * 2 + 1;

        if (slot0 >= moe_GATE_SNAPSHOT_EPOCH_SLOTS || slot1 >= moe_GATE_SNAPSHOT_EPOCH_SLOTS) {
            return;
        }
        std::atomic<uint64_t>& epoch0_ref = engine->gate_input_snapshot_epochs[slot0];
        std::atomic<uint64_t>& epoch1_ref = engine->gate_input_snapshot_epochs[slot1];

        uint64_t e0 = epoch0_ref.load(std::memory_order_acquire);
        uint64_t e1 = epoch1_ref.load(std::memory_order_acquire);

        const int write_idx = (e0 >= e1) ? 1 : 0;
        const size_t write_slot = slot * 2 + write_idx;

        std::vector<float>& dst = engine->gate_input_snapshots[write_slot];
        dst.assign(hidden, hidden + hidden_size);

        uint64_t new_epoch = std::max(e0, e1) + 1;
        if (write_idx == 0) {
            epoch0_ref.store(new_epoch, std::memory_order_release);
        } else {
            epoch1_ref.store(new_epoch, std::memory_order_release);
        }
    }
}

// Expects sigmoid-applied scores, not raw logits or a normalized probability
// distribution. It measures relative routing concentration for telemetry.
static inline float fast_log2f(float x) {
    if (!std::isfinite(x) || x <= 0.0f) {
        x = 1.0e-10f;
    }
    int e; float m = std::frexp(x, &e);
    return (float)(e - 1) + (m - 1.0f) * (1.44269504f - (m - 1.0f) * 0.5f);
}

static float moe_router_normalized_entropy(const std::vector<float>& sigmoid_scores) {
    if (sigmoid_scores.size() <= 1u) {
        return 0.0f;
    }
#ifndef NDEBUG
    for (float p : sigmoid_scores) {
        if (p < 0.0f || p > 1.0f) { return 0.5f; }
    }
#else
    if (!sigmoid_scores.empty()) {
        const float first = sigmoid_scores.front();
        const float last = sigmoid_scores.back();
        if (first < 0.0f || first > 1.0f || last < 0.0f || last > 1.0f) return 0.5f;
    }
#endif
    double sum = 0.0;
    for (float p : sigmoid_scores) {
        if (p > 0.0f && std::isfinite(p)) {
            sum += (double)p;
        }
    }
    if (sum <= 0.0) {
        return 0.0f;
    }
    double entropy = 0.0;
    for (float p : sigmoid_scores) {
        if (p <= 0.0f || !std::isfinite(p)) {
            continue;
        }
        const double q = (double)p / sum;
        entropy -= q * (double)fast_log2f((float)q);
    }
    const double denom = (double)fast_log2f((float)sigmoid_scores.size());
    if (denom <= 0.0) {
        return 0.0f;
    }
    const double normalized = entropy / denom;
    return (float)std::max(0.0, std::min(1.0, normalized));
}

static uint32_t moe_router_adaptive_top_k(
    moe_pc_engine* engine,
    uint32_t requested_top_k,
    uint32_t experts,
    float entropy,
    moe_decode_scratch* session = nullptr
) {
    uint32_t top_k = requested_top_k;
    if (entropy < 0.30f) {
        top_k = std::max(1u, requested_top_k / 2u);
    } else if (entropy >= 0.70f) {
        top_k = std::min<uint32_t>(experts, 12u);
    }
    top_k = std::min(top_k, experts);
    const uint32_t entropy_milli = (uint32_t)(entropy * 1000.0f);
    if (session) {
        session->session_router_entropy_milli = entropy_milli;
    }
    if (engine) {
        engine->last_router_entropy_milli.store(entropy_milli, std::memory_order_relaxed);
    }
    return top_k;
}

static uint32_t moe_router_fnv1a_hash_string(const char* str) {
    uint32_t hash = 2166136261u;
    while (*str) {
        hash ^= (uint8_t)*str++;
        hash *= 16777619u;
    }
    return hash;
}
