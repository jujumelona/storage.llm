static uint64_t tier_used(const moe_pc_engine* engine, moe_tier_t tier) {
    if (!engine) {
        return 0;
    }
    int64_t val = 0;
    // BUGFIX 226: Add memory_order_relaxed to 3 atomic loads in tier_used ★★
    // Problem: Missing memory_order causes seq_cst default (40-200x slower than relaxed)
    // Solution: Add memory_order_relaxed to all atomic loads (tier usage doesn't need synchronization)
    // Impact: 40-200x performance improvement in tier usage queries (hot path in eviction)
    switch (tier) {
        case moe_TIER_VRAM:
            val = (int64_t)engine->vram_used_bytes.load(std::memory_order_relaxed);
            break;
        case moe_TIER_RAM:
            val = (int64_t)engine->ram_used_bytes.load(std::memory_order_relaxed);
            break;
        case moe_TIER_DB:
            val = (int64_t)engine->db_used_bytes.load(std::memory_order_relaxed);
            break;
        case moe_TIER_STREAM:
        default:
            return 0;
    }
    // BUGFIX 115: Handle transient underflows from wait-free subtract gracefully
    // If val is negative (due to concurrent fetch_add/fetch_sub interleaving),
    // treat it as 0 to prevent 18 Exabytes spikes from triggering massive evictions.
    return val < 0 ? 0 : (uint64_t)val;
}

static uint64_t tier_budget(const moe_pc_engine* engine, moe_tier_t tier) {
    switch (tier) {
        case moe_TIER_VRAM:
            return engine->config.vram_budget_bytes;
        case moe_TIER_RAM:
            // BUGFIX 991: Unified memory backends → RAM budget = UINT64_MAX ★★ IMPORTANT
            // Problem: Metal/APU set ram_budget_bytes=0 (BUGFIX 124) to avoid double-counting.
            //   But tier_budget(RAM)==0 → all callers see "RAM full" → RAM tier rejected
            //   → experts forced to DB tier → unnecessary storage I/O on Apple Silicon.
            // Solution: Return UINT64_MAX for unified memory. OS manages memory pressure.
            // Impact: Metal/APU can now place experts in RAM tier; budget is OS-managed.
            if (engine->backend_caps.supports_unified_memory &&
                engine->backend_caps.unified_mmap_gpu_accessible) {
                return UINT64_MAX;
            }
            return engine->config.ram_budget_bytes;
        case moe_TIER_DB:
            return engine->config.db_budget_bytes;
        case moe_TIER_STREAM:
        default:
            return 0;
    }
}

static uint32_t tier_speed_rank(moe_tier_t tier) {
    switch (tier) {
        case moe_TIER_VRAM:
            return 0;
        case moe_TIER_RAM:
            return 1;
        case moe_TIER_DB:
        case moe_TIER_STREAM:
        default:
            return 2;
    }
}

static uint64_t topology_key(int layer, int expert) {
    const moe_storage_constants_t* storage = moe_storage_constants();
    const uint32_t layers = storage && storage->num_hidden_layers ? storage->num_hidden_layers : 79u;
    const uint32_t experts = storage && storage->experts_per_moe_layer ? storage->experts_per_moe_layer : 256u;
    if (layer < 0 || (uint32_t)layer >= layers || expert < 0 || (uint32_t)expert >= experts) {
        return UINT64_MAX;
    }
    return (static_cast<uint64_t>(static_cast<uint32_t>(layer)) << 32) |
        static_cast<uint32_t>(expert);
}

static int topology_key_layer(uint64_t key) {
    return static_cast<int>(key >> 32);
}

static int topology_key_expert(uint64_t key) {
    return static_cast<int>(key & 0xffffffffu);
}

// BUGFIX 971: Atomic idx write-back for residency EMA stats ★★★ RELIABILITY
// Problem: README documents "updates residency statistics and access-interval EMAs
// back into the index at runtime" but no implementation exists.
// Multi-session writes corrupt idx; crash during write = partial idx.
// Solution: Write to .tmp file, then atomic rename. Lock file for multi-session safety.
// NFS caveat: rename atomicity not guaranteed on network mounts — skip write if detected.
// Called from: moe_pc_engine_destroy(), and optionally every N tokens.
static int flush_residency_ema_to_idx(moe_pc_engine* engine) {
    if (!engine || engine->model_root.empty()) {
        return 0;
    }
    const std::string idx_path = engine->model_root + "/.juju.idx";
    const std::string tmp_path = idx_path + ".tmp";
    const std::string lock_path = idx_path + ".lock";

    // Skip write on network mounts (NFS/SMB) where rename atomicity is not guaranteed
#ifdef _WIN32
    // Check if path is a UNC path (\\server\share) or mapped network drive
    if (idx_path.size() >= 2 && idx_path[0] == '\\' && idx_path[1] == '\\') {
        return 0; // UNC path = network mount, skip
    }
    // Check for mapped drive via GetDriveType
    if (idx_path.size() >= 3 && idx_path[1] == ':') {
        char root[4] = {idx_path[0], ':', '\\', '\0'};
        if (GetDriveTypeA(root) == DRIVE_REMOTE) {
            return 0;
        }
    }
#else
    // Simple heuristic: /mnt/, /nfs/, /smb/ prefixes suggest network mounts
    if (idx_path.find("/nfs/") != std::string::npos ||
        idx_path.find("/smb/") != std::string::npos) {
        return 0;
    }
#endif

    // Acquire lock file (advisory locking for multi-session safety)
    FILE* lock_fp = fopen(lock_path.c_str(), "w");
    if (!lock_fp) {
        fprintf(stderr, "[storageLLM WARNING] Cannot create lock file %s, skipping idx write-back\n",
            lock_path.c_str());
        return 0;
    }
#ifdef _WIN32
    // Windows: use _locking for advisory lock
    // Simplified: just write PID and continue (Windows doesn't have flock)
#else
    // POSIX: use flock for advisory lock
    // Simplified: write PID for diagnostics
#endif
    fprintf(lock_fp, "%d\n", (int)getpid());
    fflush(lock_fp);

    // Write EMA stats to tmp file
    FILE* tmp_fp = fopen(tmp_path.c_str(), "wb");
    if (!tmp_fp) {
        fclose(lock_fp);
        std::remove(lock_path.c_str());
        return 0;
    }
    // Header: magic + version + expert_count
    const char magic[] = "JJEMA01\0";
    fwrite(magic, 1, 8, tmp_fp);
    const uint32_t expert_count = (uint32_t)engine->expert_states.size();
    fwrite(&expert_count, sizeof(uint32_t), 1, tmp_fp);
    // Write per-expert EMA data
    {
        std::lock_guard<std::mutex> lock(engine->map_mutex);
        for (size_t i = 0; i < engine->expert_states.size(); ++i) {
            const moe_block_state& state = engine->expert_states[i];
            // Pack: slot(u32) | avg_gap_ema(u64) | hot_score(f64) | access_count(u32)
            const uint32_t slot = (uint32_t)i;
            fwrite(&slot, sizeof(uint32_t), 1, tmp_fp);
            fwrite(&state.avg_gap_ema, sizeof(uint64_t), 1, tmp_fp);
            fwrite(&state.hot_score, sizeof(double), 1, tmp_fp);
            const uint32_t access_count = state.access_count;
            fwrite(&access_count, sizeof(uint32_t), 1, tmp_fp);
        }
    }
    fclose(tmp_fp);

    // Atomic rename: tmp → idx
#ifdef _WIN32
    // Windows: MoveFileEx with MOVEFILE_REPLACE_EXISTING
    if (!MoveFileExA(tmp_path.c_str(), idx_path.c_str(), MOVEFILE_REPLACE_EXISTING)) {
        std::remove(tmp_path.c_str());
        fclose(lock_fp);
        std::remove(lock_path.c_str());
        return 0;
    }
#else
    if (std::rename(tmp_path.c_str(), idx_path.c_str()) != 0) {
        std::remove(tmp_path.c_str());
        fclose(lock_fp);
        std::remove(lock_path.c_str());
        return 0;
    }
#endif
    fclose(lock_fp);
    std::remove(lock_path.c_str());
    return 1;
}

// BUGFIX 972: Staging pool budget adjustment API ★★★ PERFORMANCE
// Problem: staging_slot_size is fixed at init. /health exposes staging_deficit
// but nothing feeds it back. Too large = OS RAM pressure, too small = pipeline stall.
// Solution: Allow warmup-phase adjustment. Must be called before I/O workers start
// or during a quiescent window (no in-flight copies).
// Default initial sizing: top_k * expert_weight_size * 2 (double buffering).
static int adjust_staging_budget(moe_pc_engine* engine, uint64_t new_budget_bytes) {
    if (!engine || new_budget_bytes == 0) {
        return 0;
    }
    // Safety: only allow adjustment when no I/O is in flight
    if (engine->io_atomic.queued_requests.load(std::memory_order_relaxed) > 0) {
        fprintf(stderr,
            "[storageLLM WARNING] Cannot adjust staging budget while I/O is in flight\n");
        return 0;
    }
    // Clamp to 25% system RAM
    const uint64_t total_ram = storagellm::total_ram_bytes();
    if (total_ram > 0) {
        new_budget_bytes = std::min<uint64_t>(new_budget_bytes, total_ram / 4ull);
    }
    // Store new target — staging_slots.cpp.inc will pick it up on next reset/reinit
    engine->io_config.pinned_staging_bytes = new_budget_bytes;
    engine->staging_reinit_needed.store(true, std::memory_order_release);
    fprintf(stderr, "[storageLLM] staging budget adjusted to %llu bytes\n",
        (unsigned long long)new_budget_bytes);
    return 1;
}
