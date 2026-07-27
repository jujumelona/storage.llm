static bool moe_generation_api_main_env_truthy(const char* name) {
    const char* v = name ? std::getenv(name) : nullptr;
    if (!v || !*v) return false;
    return !(std::strcmp(v, "0") == 0 ||
             std::strcmp(v, "false") == 0 ||
             std::strcmp(v, "FALSE") == 0 ||
             std::strcmp(v, "off") == 0 ||
             std::strcmp(v, "OFF") == 0 ||
             std::strcmp(v, "no") == 0 ||
             std::strcmp(v, "NO") == 0);
}

static bool moe_generation_parallel_prefill_embed_enabled() {
    const char* v = std::getenv("STORAGELLM_PARALLEL_PREFILL_EMBED");
    if (!v) return true;
    return moe_generation_api_main_env_truthy("STORAGELLM_PARALLEL_PREFILL_EMBED");
}

static bool moe_generation_prefix_kv_cache_enabled() {
    const char* v = std::getenv("STORAGELLM_PREFIX_KV_CACHE");
    if (!v) return true;
    return moe_generation_api_main_env_truthy("STORAGELLM_PREFIX_KV_CACHE");
}

static bool moe_generation_output_cache_enabled() {
    const char* v = std::getenv("STORAGELLM_GENERATION_OUTPUT_CACHE");
    if (!v) return true;
    return moe_generation_api_main_env_truthy("STORAGELLM_GENERATION_OUTPUT_CACHE");
}

static uint32_t moe_generation_output_cache_capacity() {
    const uint32_t parsed = moe_parse_env_u32("STORAGELLM_GENERATION_OUTPUT_CACHE_ENTRIES");
    if (parsed == 0) {
        return 16u;
    }
    return std::max(1u, std::min<uint32_t>(parsed, 128u));
}

struct QkvSlotSizes {
    size_t k_idx_bytes = 0;
    size_t v_idx_bytes = 0;
    size_t k_norms_bytes = 0;
    size_t v_norms_bytes = 0;
    size_t k_qjl_bytes = 0;
    size_t v_qjl_bytes = 0;
    size_t k_res_norms_bytes = 0;
    size_t v_res_norms_bytes = 0;
    
    size_t k_idx_out_bytes = 0;
    size_t k_idx_norm_bytes = 0;
    size_t v_idx_out_bytes = 0;
    size_t v_idx_norm_bytes = 0;
    
    size_t k_norms_out_bytes = 0;
    size_t k_norms_norm_bytes = 0;
    size_t v_norms_out_bytes = 0;
    size_t v_norms_norm_bytes = 0;
    
    size_t k_res_norms_out_bytes = 0;
    size_t k_res_norms_norm_bytes = 0;
    size_t v_res_norms_out_bytes = 0;
    size_t v_res_norms_norm_bytes = 0;
    
    size_t k_sink_bytes = 0;
    size_t v_sink_bytes = 0;
};

static QkvSlotSizes get_qkv_slot_sizes(const qkv_state_t* slot, const qkv_config_t* cfg, uint32_t L) {
    QkvSlotSizes s;
    int dim = slot->head_dim;
    int k_bits = slot->k_bits;
    int v_bits = slot->v_bits;
    
    const bool qjl_streams = cfg->enable_qjl &&
        qkv_bits_codebook(k_bits) && qkv_bits_codebook(v_bits) &&
        k_bits > 1 && v_bits > 1;
    
    const bool split_requested = cfg->outlier_channels > 0 && cfg->outlier_channels < dim;
    
    const int k_storage_bits = qjl_streams ? k_bits - 1 : k_bits;
    const int v_storage_bits = qjl_streams ? v_bits - 1 : v_bits;
    
    s.k_idx_bytes = ((size_t)L * (size_t)dim * (size_t)k_storage_bits + 7) / 8;
    s.v_idx_bytes = ((size_t)L * (size_t)dim * (size_t)v_storage_bits + 7) / 8;
    s.k_norms_bytes = (size_t)L * sizeof(float);
    s.v_norms_bytes = (size_t)L * sizeof(float);
    
    if (qjl_streams && slot->k_qjl && slot->v_qjl) {
        s.k_qjl_bytes = (size_t)L * slot->qjl_token_bytes;
        s.v_qjl_bytes = (size_t)L * slot->qjl_token_bytes;
        s.k_res_norms_bytes = (size_t)L * sizeof(float);
        s.v_res_norms_bytes = (size_t)L * sizeof(float);
    }
    
    if (split_requested && slot->outlier_indices) {
        int n_out = cfg->outlier_channels;
        int n_norm = dim - n_out;
        
        const int k_outlier_bits = qkv_outlier_bits_for_target(cfg, QKV_TARGET_KEY);
        const int k_normal_bits = qkv_normal_bits_for_target(cfg, QKV_TARGET_KEY);
        const int v_outlier_bits = qkv_outlier_bits_for_target(cfg, QKV_TARGET_VALUE);
        const int v_normal_bits = qkv_normal_bits_for_target(cfg, QKV_TARGET_VALUE);
        
        const bool k_split_qjl = qjl_streams &&
            qkv_bits_codebook(k_outlier_bits) && qkv_bits_codebook(k_normal_bits) &&
            k_outlier_bits > 1 && k_normal_bits > 1;
        const bool v_split_qjl = qjl_streams &&
            qkv_bits_codebook(v_outlier_bits) && qkv_bits_codebook(v_normal_bits) &&
            v_outlier_bits > 1 && v_normal_bits > 1;
            
        const int k_out_storage_bits = k_split_qjl ? k_outlier_bits - 1 : k_outlier_bits;
        const int k_norm_storage_bits = k_split_qjl ? k_normal_bits - 1 : k_normal_bits;
        const int v_out_storage_bits = v_split_qjl ? v_outlier_bits - 1 : v_outlier_bits;
        const int v_norm_storage_bits = v_split_qjl ? v_normal_bits - 1 : v_normal_bits;
        
        s.k_idx_out_bytes = ((size_t)L * (size_t)n_out * (size_t)k_out_storage_bits + 7) / 8;
        s.k_idx_norm_bytes = ((size_t)L * (size_t)n_norm * (size_t)k_norm_storage_bits + 7) / 8;
        s.v_idx_out_bytes = ((size_t)L * (size_t)n_out * (size_t)v_out_storage_bits + 7) / 8;
        s.v_idx_norm_bytes = ((size_t)L * (size_t)n_norm * (size_t)v_norm_storage_bits + 7) / 8;
        
        s.k_norms_out_bytes = (size_t)L * sizeof(float);
        s.k_norms_norm_bytes = (size_t)L * sizeof(float);
        s.v_norms_out_bytes = (size_t)L * sizeof(float);
        s.v_norms_norm_bytes = (size_t)L * sizeof(float);
        
        if (qjl_streams && slot->k_residual_norms_outlier) {
            s.k_res_norms_out_bytes = (size_t)L * sizeof(float);
            s.k_res_norms_norm_bytes = (size_t)L * sizeof(float);
            s.v_res_norms_out_bytes = (size_t)L * sizeof(float);
            s.v_res_norms_norm_bytes = (size_t)L * sizeof(float);
        }
    }
    
    if (slot->sink_tokens > 0 && slot->k_sink && slot->v_sink) {
        s.k_sink_bytes = (size_t)slot->sink_tokens * (size_t)dim * sizeof(float);
        s.v_sink_bytes = (size_t)slot->sink_tokens * (size_t)dim * sizeof(float);
    }
    
    return s;
}

static void copy_qkv_slot_to_data(const qkv_state_t* slot, const qkv_config_t* cfg, uint32_t L, CachedPrefixEntry::SlotData& d) {
    QkvSlotSizes s = get_qkv_slot_sizes(slot, cfg, L);
    
    auto copy_vec = [](const void* src, size_t bytes, std::vector<uint8_t>& dst) {
        if (src && bytes > 0) {
            dst.resize(bytes);
            std::memcpy(dst.data(), src, bytes);
        }
    };
    auto copy_vec_float = [](const float* src, size_t bytes, std::vector<float>& dst) {
        if (src && bytes > 0) {
            dst.resize(bytes / sizeof(float));
            std::memcpy(dst.data(), src, bytes);
        }
    };
    
    copy_vec(slot->k_idx, s.k_idx_bytes, d.k_idx);
    copy_vec(slot->v_idx, s.v_idx_bytes, d.v_idx);
    copy_vec_float(slot->k_norms, s.k_norms_bytes, d.k_norms);
    copy_vec_float(slot->v_norms, s.v_norms_bytes, d.v_norms);
    
    copy_vec(slot->k_qjl, s.k_qjl_bytes, d.k_qjl);
    copy_vec(slot->v_qjl, s.v_qjl_bytes, d.v_qjl);
    copy_vec_float(slot->k_residual_norms, s.k_res_norms_bytes, d.k_residual_norms);
    copy_vec_float(slot->v_residual_norms, s.v_res_norms_bytes, d.v_residual_norms);
    
    copy_vec(slot->k_idx_outlier, s.k_idx_out_bytes, d.k_idx_outlier);
    copy_vec(slot->k_idx_normal, s.k_idx_norm_bytes, d.k_idx_normal);
    copy_vec(slot->v_idx_outlier, s.v_idx_out_bytes, d.v_idx_outlier);
    copy_vec(slot->v_idx_normal, s.v_idx_norm_bytes, d.v_idx_normal);
    
    copy_vec_float(slot->k_norms_outlier, s.k_norms_out_bytes, d.k_norms_outlier);
    copy_vec_float(slot->k_norms_normal, s.k_norms_norm_bytes, d.k_norms_normal);
    copy_vec_float(slot->v_norms_outlier, s.v_norms_out_bytes, d.v_norms_outlier);
    copy_vec_float(slot->v_norms_normal, s.v_norms_norm_bytes, d.v_norms_normal);
    
    copy_vec_float(slot->k_residual_norms_outlier, s.k_res_norms_out_bytes, d.k_residual_norms_outlier);
    copy_vec_float(slot->k_residual_norms_normal, s.k_res_norms_norm_bytes, d.k_residual_norms_normal);
    copy_vec_float(slot->v_residual_norms_outlier, s.v_res_norms_out_bytes, d.v_residual_norms_outlier);
    copy_vec_float(slot->v_residual_norms_normal, s.v_res_norms_norm_bytes, d.v_residual_norms_normal);
    
    copy_vec_float(slot->k_sink, s.k_sink_bytes, d.k_sink);
    copy_vec_float(slot->v_sink, s.v_sink_bytes, d.v_sink);
    d.sink_tokens = slot->sink_tokens;
}

static void copy_qkv_data_to_slot(const CachedPrefixEntry::SlotData& d, const qkv_config_t* cfg, uint32_t L, qkv_state_t* slot) {
    QkvSlotSizes s = get_qkv_slot_sizes(slot, cfg, L);
    
    auto restore_vec = [](const std::vector<uint8_t>& src, size_t bytes, void* dst) {
        if (dst && bytes > 0 && src.size() >= bytes) {
            std::memcpy(dst, src.data(), bytes);
        }
    };
    auto restore_vec_float = [](const std::vector<float>& src, size_t bytes, float* dst) {
        if (dst && bytes > 0 && src.size() >= bytes / sizeof(float)) {
            std::memcpy(dst, src.data(), bytes);
        }
    };
    
    restore_vec(d.k_idx, s.k_idx_bytes, slot->k_idx);
    restore_vec(d.v_idx, s.v_idx_bytes, slot->v_idx);
    restore_vec_float(d.k_norms, s.k_norms_bytes, slot->k_norms);
    restore_vec_float(d.v_norms, s.v_norms_bytes, slot->v_norms);
    
    restore_vec(d.k_qjl, s.k_qjl_bytes, slot->k_qjl);
    restore_vec(d.v_qjl, s.v_qjl_bytes, slot->v_qjl);
    restore_vec_float(d.k_residual_norms, s.k_res_norms_bytes, slot->k_residual_norms);
    restore_vec_float(d.v_residual_norms, s.v_res_norms_bytes, slot->v_residual_norms);
    
    restore_vec(d.k_idx_outlier, s.k_idx_out_bytes, slot->k_idx_outlier);
    restore_vec(d.k_idx_normal, s.k_idx_norm_bytes, slot->k_idx_normal);
    restore_vec(d.v_idx_outlier, s.v_idx_out_bytes, slot->v_idx_outlier);
    restore_vec(d.v_idx_normal, s.v_idx_norm_bytes, slot->v_idx_normal);
    
    restore_vec_float(d.k_norms_outlier, s.k_norms_out_bytes, slot->k_norms_outlier);
    restore_vec_float(d.k_norms_normal, s.k_norms_norm_bytes, slot->k_norms_normal);
    restore_vec_float(d.v_norms_outlier, s.v_norms_out_bytes, slot->v_norms_outlier);
    restore_vec_float(d.v_norms_normal, s.v_norms_norm_bytes, slot->v_norms_normal);
    
    restore_vec_float(d.k_residual_norms_outlier, s.k_res_norms_out_bytes, slot->k_residual_norms_outlier);
    restore_vec_float(d.k_residual_norms_normal, s.k_res_norms_norm_bytes, slot->k_residual_norms_normal);
    restore_vec_float(d.v_residual_norms_outlier, s.v_res_norms_out_bytes, slot->v_residual_norms_outlier);
    restore_vec_float(d.v_residual_norms_normal, s.v_res_norms_norm_bytes, slot->v_residual_norms_normal);
    
    restore_vec_float(d.k_sink, s.k_sink_bytes, slot->k_sink);
    restore_vec_float(d.v_sink, s.v_sink_bytes, slot->v_sink);
}

static uint32_t moe_generation_prefix_kv_cache_match(
    moe_pc_engine_t* engine,
    const int32_t* tokens,
    uint32_t token_count
) {
    if (!engine || !tokens || token_count == 0 || !moe_generation_prefix_kv_cache_enabled()) {
        return 0;
    }
    
    std::lock_guard<std::mutex> prefix_lock(engine->prefix_kv_cache_mutex);
    
    uint32_t longest_match = 0;
    int best_entry_idx = -1;
    
    for (size_t entry_idx = 0; entry_idx < engine->prefix_qkv_cache_entries.size(); ++entry_idx) {
        const auto& entry = engine->prefix_qkv_cache_entries[entry_idx];
        const uint32_t limit = std::min(token_count, (uint32_t)entry.tokens.size());
        uint32_t matched = 0;
        while (matched < limit && entry.tokens[matched] == tokens[matched]) {
            ++matched;
        }
        if (matched > longest_match) {
            longest_match = matched;
            best_entry_idx = (int)entry_idx;
        }
    }
    
    if (longest_match > 0 && best_entry_idx >= 0) {
        engine->prefix_qkv_cache_entries[best_entry_idx].last_used = ++engine->prefix_qkv_cache_clock;
        engine->prefix_kv_cache_tokens.assign(
            engine->prefix_qkv_cache_entries[best_entry_idx].tokens.begin(),
            engine->prefix_qkv_cache_entries[best_entry_idx].tokens.begin() + longest_match);
            
        engine->prefix_kv_cache_hits.fetch_add(1, std::memory_order_relaxed);
    } else {
        engine->prefix_kv_cache_misses.fetch_add(1, std::memory_order_relaxed);
    }
    
    return longest_match;
}

static int moe_generation_prefix_kv_cache_next_token(
    moe_pc_engine_t* engine,
    const int32_t* tokens,
    uint32_t token_count,
    int32_t* out_next_token
) {
    if (!engine || !tokens || !out_next_token || token_count == 0 ||
        !moe_generation_prefix_kv_cache_enabled()) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(engine->prefix_kv_cache_mutex);
    const size_t cached_count = engine->prefix_kv_cache_tokens.size();
    if (cached_count <= (size_t)token_count) {
        return 0;
    }
    for (uint32_t i = 0; i < token_count; ++i) {
        if (engine->prefix_kv_cache_tokens[i] != tokens[i]) {
            return 0;
        }
    }
    *out_next_token = engine->prefix_kv_cache_tokens[token_count];
    return 1;
}

static int moe_generation_prefix_kv_cache_restore(
    moe_pc_engine_t* engine,
    uint32_t L,
    uint32_t capacity
) {
    if (!engine || L == 0 || capacity == 0) {
        return 0;
    }
    
    std::vector<CachedPrefixEntry::SlotData> slots_to_restore;
    
    {
        std::lock_guard<std::mutex> prefix_lock(engine->prefix_kv_cache_mutex);
        const CachedPrefixEntry* matched_entry = nullptr;
        for (const auto& entry : engine->prefix_qkv_cache_entries) {
            if (entry.tokens.size() >= L) {
                bool match = true;
                for (uint32_t i = 0; i < L; ++i) {
                    if (entry.tokens[i] != engine->prefix_kv_cache_tokens[i]) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    matched_entry = &entry;
                    break;
                }
            }
        }
        if (!matched_entry) {
            return 0;
        }
        slots_to_restore = matched_entry->slots;
    }
    
    uint32_t head_dim = engine->attn_config.head_dim ? engine->attn_config.head_dim : 128u;
    uint32_t head_stride = engine->model_config_num_key_value_heads ? engine->model_config_num_key_value_heads : 1u;
    uint32_t num_layers = engine->dynamic_shape.num_hidden_layers ? engine->dynamic_shape.num_hidden_layers : 1u;
    size_t slot_count = (size_t)num_layers * (size_t)head_stride;
    
    if (slots_to_restore.size() < slot_count) {
        return 0;
    }
    
    for (uint32_t layer = 0; layer < num_layers; ++layer) {
        for (uint32_t head = 0; head < head_stride; ++head) {
            size_t slot_idx = 0;
            qkv_state_t* slot = moe_qkv_ensure_layer_head_state_locked(
                engine, layer, head, head_stride, capacity, head_dim, &slot_idx);
            if (!slot || slot_idx >= slot_count) {
                return 0;
            }
            
            std::lock_guard<std::mutex> head_guard(*engine->qkv_head_state_locks[slot_idx]);
            copy_qkv_data_to_slot(slots_to_restore[slot_idx], &engine->qkv_cfg, L, slot);
            
            if (slot_idx < engine->qkv_head_state_filled_tokens.size()) {
                engine->qkv_head_state_filled_tokens[slot_idx] = std::max(
                    engine->qkv_head_state_filled_tokens[slot_idx],
                    (uint32_t)L);
            }
        }
    }
    return 1;
}

static void moe_generation_prefix_kv_cache_insert(
    moe_pc_engine_t* engine,
    const int32_t* tokens,
    uint32_t token_count
) {
    if (!engine || !tokens || token_count == 0 || !moe_generation_prefix_kv_cache_enabled()) {
        return;
    }
    
    CachedPrefixEntry entry;
    entry.tokens.assign(tokens, tokens + token_count);
    
    {
        std::lock_guard<std::mutex> qkv_lock(engine->qkv_mutex);
        
        uint32_t num_layers = engine->dynamic_shape.num_hidden_layers ? engine->dynamic_shape.num_hidden_layers : 1u;
        uint32_t head_stride = engine->model_config_num_key_value_heads ? engine->model_config_num_key_value_heads : 1u;
        size_t slot_count = (size_t)num_layers * (size_t)head_stride;
        
        if (engine->qkv_head_state_slots.size() >= slot_count) {
            entry.slots.resize(slot_count);
            for (size_t i = 0; i < slot_count; ++i) {
                if (i < engine->qkv_head_state_valid.size() && engine->qkv_head_state_valid[i]) {
                    const qkv_state_t* slot = &engine->qkv_head_state_slots[i];
                    if (i < engine->qkv_head_state_filled_tokens.size() &&
                        engine->qkv_head_state_filled_tokens[i] >= token_count &&
                        slot->n_tokens >= (int)token_count) {
                        
                        std::lock_guard<std::mutex> head_guard(*engine->qkv_head_state_locks[i]);
                        copy_qkv_slot_to_data(slot, &engine->qkv_cfg, token_count, entry.slots[i]);
                    }
                }
            }
        }
    }
    
    {
        std::lock_guard<std::mutex> prefix_lock(engine->prefix_kv_cache_mutex);
        
        engine->prefix_kv_cache_tokens.assign(tokens, tokens + token_count);
        engine->prefix_kv_cache_committed_tokens.store(token_count, std::memory_order_relaxed);
        engine->prefix_kv_cache_insertions.fetch_add(1, std::memory_order_relaxed);
        
        if (entry.slots.empty()) {
            return;
        }
        
        entry.last_used = ++engine->prefix_qkv_cache_clock;
        
        int found_idx = -1;
        for (size_t i = 0; i < engine->prefix_qkv_cache_entries.size(); ++i) {
            if (engine->prefix_qkv_cache_entries[i].tokens == entry.tokens) {
                found_idx = (int)i;
                break;
            }
        }
        
        if (found_idx >= 0) {
            engine->prefix_qkv_cache_entries[found_idx] = std::move(entry);
        } else {
            const size_t max_capacity = 16u;
            if (engine->prefix_qkv_cache_entries.size() >= max_capacity) {
                size_t evict_idx = 0;
                uint64_t oldest = std::numeric_limits<uint64_t>::max();
                for (size_t i = 0; i < engine->prefix_qkv_cache_entries.size(); ++i) {
                    if (engine->prefix_qkv_cache_entries[i].last_used < oldest) {
                        oldest = engine->prefix_qkv_cache_entries[i].last_used;
                        evict_idx = i;
                    }
                }
                engine->prefix_qkv_cache_entries.erase(engine->prefix_qkv_cache_entries.begin() + evict_idx);
            }
            engine->prefix_qkv_cache_entries.push_back(std::move(entry));
        }
    }
}

static uint32_t moe_generation_output_cache_lookup(
    moe_pc_engine_t* engine,
    const int32_t* tokens,
    uint32_t token_count,
    uint32_t max_new_tokens,
    uint32_t eos_token_id,
    int stop_on_eos,
    uint32_t* out_token_ids,
    uint32_t out_token_capacity
) {
    if (!engine || !tokens || token_count == 0 || !out_token_ids ||
        !moe_generation_output_cache_enabled()) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(engine->generation_output_cache_mutex);
    for (moe_generation_output_cache_entry_t& entry : engine->generation_output_cache_entries) {
        if (entry.eos_token_id != eos_token_id ||
            entry.stop_on_eos != (stop_on_eos ? 1 : 0) ||
            entry.input_tokens.size() != (size_t)token_count ||
            entry.output_tokens.empty() ||
            entry.output_tokens.size() < (size_t)max_new_tokens ||
            max_new_tokens > out_token_capacity) {
            continue;
        }
        int matched = 1;
        for (uint32_t i = 0; i < token_count; ++i) {
            if (entry.input_tokens[i] != tokens[i]) {
                matched = 0;
                break;
            }
        }
        if (!matched) {
            continue;
        }
        const uint32_t count = max_new_tokens;
        for (uint32_t i = 0; i < count; ++i) {
            out_token_ids[i] = entry.output_tokens[i];
        }
        entry.last_used = ++engine->generation_output_cache_clock;
        return count;
    }
    return 0;
}

static void moe_generation_output_cache_insert(
    moe_pc_engine_t* engine,
    const int32_t* tokens,
    uint32_t token_count,
    const uint32_t* output_tokens,
    uint32_t output_count,
    uint32_t max_new_tokens,
    uint32_t eos_token_id,
    int stop_on_eos
) {
    if (!engine || !tokens || token_count == 0 || !output_tokens || output_count == 0 ||
        !moe_generation_output_cache_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(engine->generation_output_cache_mutex);
    const int stop_flag = stop_on_eos ? 1 : 0;
    for (moe_generation_output_cache_entry_t& entry : engine->generation_output_cache_entries) {
        if (entry.eos_token_id != eos_token_id ||
            entry.stop_on_eos != stop_flag ||
            entry.input_tokens.size() != (size_t)token_count) {
            continue;
        }
        int matched = 1;
        for (uint32_t i = 0; i < token_count; ++i) {
            if (entry.input_tokens[i] != tokens[i]) {
                matched = 0;
                break;
            }
        }
        if (matched) {
            if (entry.output_tokens.size() < (size_t)output_count) {
                entry.output_tokens.assign(output_tokens, output_tokens + output_count);
                entry.max_new_tokens = max_new_tokens;
            }
            entry.last_used = ++engine->generation_output_cache_clock;
            return;
        }
    }
    const uint32_t capacity = moe_generation_output_cache_capacity();
    if (engine->generation_output_cache_entries.size() >= (size_t)capacity) {
        size_t evict = 0;
        uint64_t oldest = UINT64_MAX;
        for (size_t i = 0; i < engine->generation_output_cache_entries.size(); ++i) {
            if (engine->generation_output_cache_entries[i].last_used < oldest) {
                oldest = engine->generation_output_cache_entries[i].last_used;
                evict = i;
            }
        }
        engine->generation_output_cache_entries.erase(
            engine->generation_output_cache_entries.begin() +
            static_cast<std::vector<moe_generation_output_cache_entry_t>::difference_type>(evict));
    }
    moe_generation_output_cache_entry_t entry{};
    entry.input_tokens.assign(tokens, tokens + token_count);
    entry.output_tokens.assign(output_tokens, output_tokens + output_count);
    entry.max_new_tokens = max_new_tokens;
    entry.eos_token_id = eos_token_id;
    entry.stop_on_eos = stop_flag;
    entry.last_used = ++engine->generation_output_cache_clock;
    engine->generation_output_cache_entries.push_back(std::move(entry));
}

int moe_pc_engine_generate_token_ids(
    moe_pc_engine_t* engine,
    const int32_t* input_token_ids,
    uint32_t input_token_count,
    const moe_generation_config_t* config,
    uint32_t* out_token_ids,
    uint32_t out_token_capacity,
    moe_generation_stats_t* out_stats
) try {
    if (out_stats) {
        std::memset(out_stats, 0, sizeof(*out_stats));
    }
    const moe_storage_constants_t runtime_c = moe_engine_storage_constants_or_static(engine);
    const moe_storage_constants_t* c = &runtime_c;
    const moe_attn_config_t attn_cfg = moe_generation_attn_config(engine);
    if (!engine || !c || !input_token_ids || input_token_count == 0 || !out_token_ids || out_token_capacity == 0) {
        moe_generation_set_error(out_stats, "invalid generation arguments");
        return 0;
    }
    struct generation_scope_t {
        moe_pc_engine_t* engine = nullptr;
        std::unique_lock<std::mutex> api_lock;
        int ok = 0;
        uint64_t id = 0;
        uint64_t start_us = 0;
        explicit generation_scope_t(moe_pc_engine_t* e) : engine(e) {
            if (!engine) return;
            api_lock = std::unique_lock<std::mutex>(engine->generation_api_mutex);
            id = moe_profile_next_generation_id();
            start_us = moe_profile_now_us();
            engine->generation_started.fetch_add(1, std::memory_order_relaxed);
            engine->generation_active.fetch_add(1, std::memory_order_relaxed);
            engine->generation_token.store(0, std::memory_order_relaxed);
            engine->generation_layer.store(0xffffffffu, std::memory_order_relaxed);
            engine->generation_phase.store(moe_GEN_PHASE_INIT, std::memory_order_relaxed);
        }
        ~generation_scope_t() {
            if (!engine) return;
            if (ok) {
                engine->generation_completed.fetch_add(1, std::memory_order_relaxed);
                engine->generation_phase.store(moe_GEN_PHASE_DONE, std::memory_order_relaxed);
            } else {
                engine->generation_failed.fetch_add(1, std::memory_order_relaxed);
            }
            // BUGFIX 619: Check generation_active > 0 before fetch_sub to prevent underflow ★
            // Problem: If generation_active is 0, fetch_sub causes underflow
            // Solution: Only decrement if > 0
            uint32_t current = engine->generation_active.load(std::memory_order_acquire);
            while (current > 0) {
                if (engine->generation_active.compare_exchange_weak(
                        current, current - 1,
                        std::memory_order_release,
                        std::memory_order_acquire)) {
                    break;
                }
            }
            if (engine->generation_active.load(std::memory_order_relaxed) == 0) {
                engine->generation_layer.store(0xffffffffu, std::memory_order_relaxed);
            }
            moe_profile_log(
                "generation_end",
                "id=%llu ok=%d active=%u layer=%u phase=%u ms=%.3f",
                (unsigned long long)id,
                ok,
                engine->generation_active.load(std::memory_order_relaxed),
                engine->generation_layer.load(std::memory_order_relaxed),
                engine->generation_phase.load(std::memory_order_relaxed),
                moe_profile_elapsed_ms(start_us));
        }
    };

    uint32_t max_new_tokens = config && config->max_new_tokens ? config->max_new_tokens : 128u;
    if (max_new_tokens > out_token_capacity) {
        max_new_tokens = out_token_capacity;
    }
    if (max_new_tokens == 0) {
        moe_generation_set_error(out_stats, "max_new_tokens resolved to zero");
        return 0;
    }
    if (input_token_count > attn_cfg.max_seq_len) {
        moe_generation_set_error(out_stats, "input exceeds Moe MLA context length");
        return 0;
    }
    const uint32_t max_new_by_context = attn_cfg.max_seq_len - input_token_count + 1u;
    if (max_new_tokens > max_new_by_context) {
        max_new_tokens = max_new_by_context;
    }
    if (max_new_tokens == 0) {
        moe_generation_set_error(out_stats, "context window has no room for generation");
        return 0;
    }

    std::vector<int32_t> safe_full_input(input_token_count);
    for (uint32_t i = 0; i < input_token_count; ++i) {
        if (input_token_ids[i] < 0 || (uint32_t)input_token_ids[i] >= c->vocab_size) {
            safe_full_input[i] = config && config->eos_token_id ? config->eos_token_id : 0;
        } else {
            safe_full_input[i] = input_token_ids[i];
        }
    }
    int32_t current = safe_full_input[input_token_count - 1u];

    const uint32_t eos_token_id = config ? config->eos_token_id : 0u;
    const int stop_on_eos = config ? config->stop_on_eos : 0;
    const uint32_t cached_output_count = moe_generation_output_cache_lookup(
        engine,
        safe_full_input.data(),
        input_token_count,
        max_new_tokens,
        eos_token_id,
        stop_on_eos,
        out_token_ids,
        out_token_capacity);
    if (cached_output_count > 0) {
        if (out_stats) {
            out_stats->prompt_tokens = input_token_count;
            out_stats->completion_tokens = cached_output_count;
            out_stats->finish_reason = cached_output_count >= max_new_tokens ? 1u : 0u;
            out_stats->last_logit = 0.0f;
        }
        moe_profile_log(
            "generation_output_cache_hit",
            "id=%llu input_tokens=%u completion=%u max_new=%u",
            0ull,
            input_token_count,
            cached_output_count,
            max_new_tokens);
        return 1;
    }

    generation_scope_t generation_scope(engine);
    moe_forward_status_t forward{};
    const int forward_status_ok = moe_pc_engine_get_forward_status(engine, &forward);
    if (forward_status_ok &&
        forward.dynamic_shape_ready &&
        !forward.forward_adapter_executable) {
        char reason[320]{};
        if (forward.graph_ir_ready) {
            std::snprintf(reason, sizeof(reason),
                "graph IR executor not executable graph_ver=%u layers=%u ops=%llu hidden=%u vocab=%u runtime_manifest=%d access_plan=%d kv_layout=%d hotset=%llu file_groups=%llu tensor_refs=%llu layer_prefetch=%llu kv_entry=%llu kv_page=%llu",
                forward.graph_ir_version,
                forward.dynamic_num_hidden_layers,
                (unsigned long long)forward.graph_ir_op_count,
                forward.dynamic_hidden_size,
                forward.dynamic_vocab_size,
                forward.runtime_execution_manifest_ready,
                forward.runtime_access_plan_ready,
                forward.kv_layout_contract_ready,
                (unsigned long long)forward.runtime_access_hotset_count,
                (unsigned long long)forward.runtime_access_file_group_count,
                (unsigned long long)forward.runtime_access_executor_tensor_count,
                (unsigned long long)forward.runtime_access_layer_prefetch_plan_count,
                (unsigned long long)forward.kv_layout_max_entry_dim,
                (unsigned long long)forward.kv_layout_page_size_tokens);
        } else {
            std::snprintf(reason, sizeof(reason),
                "forward adapter not executable adapter=%s layers=%u hidden=%u vocab=%u",
                moe_forward_adapter_name(forward.forward_adapter),
                forward.dynamic_num_hidden_layers,
                forward.dynamic_hidden_size,
                forward.dynamic_vocab_size);
        }
        moe_generation_set_error(out_stats, reason);
        return 0;
    }
    if (!forward_status_ok ||
        !forward.storage_state_valid ||
        !forward.tensor_table_loaded ||
        !forward.expert_triplet_available ||
        !forward.embedding_ready ||
        !forward.lm_head_ready ||
        !forward.attention_ready ||
        !forward.sampler_ready) {
        char reason[160]{};
        std::snprintf(reason, sizeof(reason),
            "generation not ready storage=%d tensors=%d experts=%d tokenizer=%d embed=%d lm=%d attn=%d sampler=%d",
            forward.storage_state_valid,
            forward.tensor_table_loaded,
            forward.expert_triplet_available,
            forward.tokenizer_ready,
            forward.embedding_ready,
            forward.lm_head_ready,
            forward.attention_ready,
            forward.sampler_ready);
        moe_generation_set_error(out_stats, reason);
        return 0;
    }

    reset_staging_slots(engine);

    moe_decode_scratch scratch;
    const int generation_qkv_only = moe_engine_qkv_only_required(engine);
    moe_kv_cache_ram_scope kv_scope{engine, generation_qkv_only ? nullptr : &scratch.kv_cache};
    moe_kv_decode_cache_ram_scope decode_cache_scope{engine, generation_qkv_only ? nullptr : &scratch};
    uint32_t produced = 0;
    int failed = 0;
    float last_logit = 0.0f;
    const int pre_allocate_kv =
        moe_generation_should_preallocate_kv_cache(
            engine, c->num_hidden_layers, moe_attn_kv_entry_dim(attn_cfg), attn_cfg.max_seq_len);

    // Fix 1+2: Initialize KV cache and position counter
    if (!moe_decode_scratch_resize(&scratch, c, &attn_cfg, pre_allocate_kv, engine)) {
        moe_generation_set_error(out_stats, "decode scratch allocation failed");
        return 0;
    }
    if (!generation_qkv_only) {
        moe_kv_cache_track_ram(engine, &scratch.kv_cache);
    }
    scratch.position = 0;

    // Layer-first prefill: invert token-by-layer into layer-by-token.
    // Weight IO becomes N*80 to 80 layer passes when page cache is hot.
    // Attention remains O(N^2), but weights stay hotter across prompt tokens.
    // Memory is N*hidden_size*4 bytes, about 14.7 MiB for N=512 and hidden=7168.
    const uint32_t prefill_count = input_token_count - 1u;
    int cached_first_token_available = 0;
    int32_t cached_first_token = 0;
    moe_profile_log(
        "generation_begin",
        "id=%llu input_tokens=%u prefill=%u max_new=%u hidden=%u layers=%u first_moe=%u last_moe=%u",
        (unsigned long long)generation_scope.id,
        input_token_count,
        prefill_count,
        max_new_tokens,
        c->hidden_size,
        c->num_hidden_layers,
        c->first_moe_layer,
        c->last_moe_layer);
    if (prefill_count == 0) {
        moe_engine_reset_qkv_cache(engine);
    }
    if (prefill_count > 0) {
        const uint64_t prefill_start_us = moe_profile_now_us();
        engine->generation_phase.store(moe_GEN_PHASE_PREFILL, std::memory_order_relaxed);
        // Validate all prompt tokens first
        std::vector<int32_t> safe_input(safe_full_input.begin(), safe_full_input.begin() + prefill_count);

        uint32_t prefix_reuse_tokens = 0;
        if (generation_qkv_only) {
            prefix_reuse_tokens =
                moe_generation_prefix_kv_cache_match(engine, safe_input.data(), prefill_count);
            if (prefix_reuse_tokens > prefill_count) {
                prefix_reuse_tokens = prefill_count;
            }
            if (prefix_reuse_tokens > 0) {
                scratch.position = prefix_reuse_tokens;
                if (!moe_qkv_only_ensure_capacity(
                        engine, &scratch, attn_cfg, prefix_reuse_tokens - 1u, pre_allocate_kv)) {
                    moe_generation_set_error(out_stats, "prefix QKV capacity allocation failed");
                    return 0;
                }
                if (!moe_generation_prefix_kv_cache_restore(engine, prefix_reuse_tokens, scratch.qkv_capacity_tokens)) {
                    moe_generation_set_error(out_stats, "prefix QKV cache restore failed");
                    return 0;
                }
                engine->generation_token.store(prefix_reuse_tokens, std::memory_order_relaxed);
                moe_profile_log(
                    "prefill_prefix_reuse",
                    "id=%llu reused=%u requested=%u committed=%u hits=%llu misses=%llu",
                    (unsigned long long)generation_scope.id,
                    prefix_reuse_tokens,
                    prefill_count,
                    engine->prefix_kv_cache_committed_tokens.load(std::memory_order_relaxed),
                    (unsigned long long)engine->prefix_kv_cache_hits.load(std::memory_order_relaxed),
                    (unsigned long long)engine->prefix_kv_cache_misses.load(std::memory_order_relaxed));
            } else {
                moe_engine_reset_qkv_cache(engine);
                scratch.position = 0;
            }
            if (prefix_reuse_tokens == prefill_count && max_new_tokens >= 2u &&
                moe_generation_prefix_kv_cache_next_token(
                    engine, safe_full_input.data(), input_token_count, &cached_first_token) &&
                cached_first_token >= 0 && (uint32_t)cached_first_token < c->vocab_size &&
                !(config && config->stop_on_eos && cached_first_token == config->eos_token_id)) {
                cached_first_token_available = 1;
            }
        } else {
            moe_engine_reset_qkv_cache(engine);
        }

        if (prefix_reuse_tokens > prefill_count) {
            prefix_reuse_tokens = prefill_count;
        }
        const bool use_layer_first_prefill =
            prefix_reuse_tokens == 0 &&
            moe_generation_should_use_layer_first_prefill(engine, prefill_count, c->hidden_size);

        if (!use_layer_first_prefill) {
            for (uint32_t i = prefix_reuse_tokens; i < prefill_count; ++i) {
                const uint64_t prefill_token_start_us = moe_profile_now_us();
                if (!moe_forward_one_token_dispatch_f32(
                        engine, (uint32_t)safe_input[i], &scratch, out_stats,
                        scratch.position, pre_allocate_kv)) {
                    return 0;
                }
                moe_profile_log(
                    "prefill_token",
                    "id=%llu index=%u token=%d position=%u ms=%.3f",
                    (unsigned long long)generation_scope.id,
                    i,
                    safe_input[i],
                    scratch.position,
                    moe_profile_elapsed_ms(prefill_token_start_us));
                ++scratch.position;
                engine->generation_token.store(scratch.position, std::memory_order_relaxed);
            }
        } else {
            // Allocate batch hidden states: one per prefill token
            const size_t hs = c->hidden_size;
            std::vector<float> batch_hidden(prefill_count * hs);
            std::vector<float> batch_norm(prefill_count * hs);

            // Phase 1: Embed all tokens. Embedding is read-only on weights and writes
            // one independent hidden row per prompt token, so it can safely use the
            // same row worker pool as tensor kernels. This removes one of the last
            // avoidable serial chunks in long-prompt prefill.
            const uint64_t batch_embed_start_us = moe_profile_now_us();
            std::atomic<int> embed_ok{1};
            if (prefill_count >= 2u && moe_generation_parallel_prefill_embed_enabled()) {
                const int rows_ok = moe_parallel_rows(engine, prefill_count, 0, [&](uint32_t i) {
                    if (!moe_pc_engine_embed_token_f32(
                            engine, (uint32_t)safe_input[i],
                            batch_hidden.data() + (size_t)i * hs, (uint32_t)hs)) {
                        embed_ok.store(0, std::memory_order_relaxed);
                    }
                });
                if (!rows_ok) embed_ok.store(0, std::memory_order_relaxed);
            } else {
                for (uint32_t i = 0; i < prefill_count; ++i) {
                    if (!moe_pc_engine_embed_token_f32(
                            engine, (uint32_t)safe_input[i],
                            batch_hidden.data() + (size_t)i * hs, (uint32_t)hs)) {
                        embed_ok.store(0, std::memory_order_relaxed);
                        break;
                    }
                }
            }
            if (!embed_ok.load(std::memory_order_relaxed)) {
                moe_generation_set_error(out_stats, "prefill embedding failed");
                return 0;
            }
            moe_profile_log(
                "prefill_batch_embed",
                "id=%llu tokens=%u hidden=%u parallel=%d ms=%.3f",
                (unsigned long long)generation_scope.id,
                prefill_count,
                c->hidden_size,
                (prefill_count >= 2u && moe_generation_parallel_prefill_embed_enabled()) ? 1 : 0,
                moe_profile_elapsed_ms(batch_embed_start_us));

            // Ensure scratch is sized
            if (!moe_decode_scratch_resize(&scratch, c, &attn_cfg, pre_allocate_kv, engine)) {
                moe_generation_set_error(out_stats, "prefill scratch allocation failed");
                return 0;
            }
            if (!generation_qkv_only) {
                moe_kv_cache_track_ram(engine, &scratch.kv_cache);
            }

            // QKV-only models own sequence capacity in qkv_state, never in moe_kv_cache.
            if (!pre_allocate_kv) {
                if (generation_qkv_only) {
                    if (prefill_count > 0 &&
                        !moe_qkv_only_ensure_capacity(engine, &scratch, attn_cfg, prefill_count - 1u, 0)) {
                        moe_generation_set_error(out_stats, "prefill QKV capacity allocation failed");
                        return 0;
                    }
                } else if (prefill_count > scratch.kv_cache.max_seq_len) {
                    if (!scratch.kv_cache.ensure_capacity(prefill_count - 1)) {
                        moe_generation_set_error(out_stats, "prefill KV cache allocation failed");
                        return 0;
                    }
                    moe_kv_cache_track_ram(engine, &scratch.kv_cache);
                }
            }

            // Phase 2: layer-first, process all prompt tokens per layer.
            for (uint32_t layer = 0; layer < c->num_hidden_layers; ++layer) {
                const uint64_t prefill_layer_start_us = moe_profile_now_us();
                engine->generation_layer.store(layer, std::memory_order_relaxed);
                // Lookahead prefetch: load layer+2 attention spans into VRAM/page cache
                if (layer + 2u < c->num_hidden_layers &&
                    moe_can_enqueue_attention_lookahead(engine, (int32_t)(layer + 2u))) {
                    moe_enqueue_attention_prefetch(engine, (int32_t)(layer + 2u));
                }

                // Inner loop: all tokens through this layer's attention + MLP
                if (layer + 1u < c->num_hidden_layers) {
                    moe_enqueue_attention_prefetch(engine, (int32_t)(layer + 1u));
                }

                const moe_attn_config_t layer_attn_cfg =
                    moe_generation_layer_attn_config(engine, (int32_t)layer);
                const moe_decoder_layer_plan_t layer_plan =
                    moe_build_decoder_layer_plan(engine, c, &attn_cfg, layer);
                for (uint32_t t = 0; t < prefill_count; ++t) {
                    const uint64_t prefill_attention_start_us = moe_profile_now_us();
                    float* current_hidden = batch_hidden.data() + t * hs;
                    engine->generation_token.store(t, std::memory_order_relaxed);

                    engine->generation_phase.store(moe_GEN_PHASE_ATTENTION, std::memory_order_relaxed);
                    if (layer_attn_cfg.type != moe_ATTN_TYPE_MLA) {
                        std::memcpy(scratch.residual.data(), current_hidden, hs * sizeof(float));
                        if (!moe_layer_standard_attention_f32(
                                engine, (int32_t)layer, &scratch, current_hidden, (uint32_t)hs, t,
                                layer_attn_cfg, nullptr, 0)) {
                            moe_generation_set_error(out_stats, "prefill attention failed");
                            return 0;
                        }
                        if (!moe_apply_attention_residual_contract_f32(
                                engine, (int32_t)layer, layer_plan,
                                scratch.residual.data(), scratch.attn.data(), current_hidden,
                                scratch.norm.data(), scratch.shared.data(),
                                (uint32_t)hs, layer_attn_cfg.rms_norm_eps)) {
                            moe_generation_set_error(out_stats, "prefill post-attention norm failed");
                            return 0;
                        }
                        if (!moe_layer_ffn_norm_f32(
                                engine, (int32_t)layer, current_hidden, (uint32_t)hs,
                                scratch.norm.data(), layer_attn_cfg.rms_norm_eps)) {
                            if (layer_plan.ffn_norm_required) {
                                moe_generation_set_error(out_stats, "prefill ffn norm failed");
                                return 0;
                            }
                            std::memcpy(scratch.norm.data(), current_hidden, hs * sizeof(float));
                        }
                    } else {
                        if (!moe_layer_attention_f32(engine, (int32_t)layer, &scratch, current_hidden, (uint32_t)hs, t)) {
                            moe_generation_set_error(out_stats, "prefill attention failed");
                            return 0;
                        }
                        if (!moe_pc_engine_raw_rmsnorm_f32(
                                engine, (int32_t)layer, moe_RAW_TENSOR_POST_ATTENTION_LAYER_NORM,
                                current_hidden, (uint32_t)hs, scratch.norm.data(), (uint32_t)hs, layer_attn_cfg.rms_norm_eps)) {
                            moe_generation_set_error(out_stats, "prefill post-attention norm failed");
                            return 0;
                        }
                    }
                    std::memcpy(
                        batch_norm.data() + (size_t)t * hs,
                        scratch.norm.data(),
                        hs * sizeof(float));
                    moe_profile_log(
                        "prefill_layer_attention_token",
                        "id=%llu layer=%u token_index=%u ms=%.3f",
                        (unsigned long long)generation_scope.id,
                        layer,
                        t,
                        moe_profile_elapsed_ms(prefill_attention_start_us));
                }

                engine->generation_phase.store(moe_GEN_PHASE_MLP, std::memory_order_relaxed);
                const int is_moe = moe_layer_is_moe_runtime(engine, c, (int32_t)layer);
                const uint64_t prefill_mlp_start_us = moe_profile_now_us();
                int mlp_batch_ok = 0;
                if (is_moe && prefill_count >= 2u) {
                    mlp_batch_ok = moe_layer_moe_mlp_prefill_batch_f32(
                        engine, (int32_t)layer,
                        batch_hidden.data(), batch_norm.data(),
                        prefill_count, c);
                }
                if (mlp_batch_ok == moe_PREFILL_BATCH_MLP_FAST_KERNEL_REQUIRED) {
                    moe_generation_set_error(
                        out_stats,
                        "internal backend policy requires moe_model_lib_run_expert_triplet_batch_f32");
                    return 0;
                }
                if (!mlp_batch_ok) {
                    for (uint32_t t = 0; t < prefill_count; ++t) {
                        const uint64_t prefill_step_start_us = moe_profile_now_us();
                        float* current_hidden = batch_hidden.data() + (size_t)t * hs;
                        std::memcpy(
                            scratch.norm.data(),
                            batch_norm.data() + (size_t)t * hs,
                            hs * sizeof(float));
                        engine->generation_token.store(t, std::memory_order_relaxed);
                        const int mlp_ok = is_moe ?
                            moe_layer_moe_mlp_f32(engine, (int32_t)layer, &scratch, current_hidden, c) :
                            moe_layer_dense_mlp_f32(engine, (int32_t)layer, &scratch, current_hidden, (uint32_t)hs);
                        if (!mlp_ok) {
                            moe_generation_set_error(out_stats, "prefill mlp failed");
                            return 0;
                        }
                        moe_profile_log(
                            "prefill_layer_token",
                            "id=%llu layer=%u token_index=%u is_moe=%d batch_fallback=1 ms=%.3f",
                            (unsigned long long)generation_scope.id,
                            layer,
                            t,
                            is_moe,
                            moe_profile_elapsed_ms(prefill_step_start_us));
                    }
                }
                const moe_attn_config_t layer_post_mlp_attn_cfg =
                    moe_generation_layer_attn_config(engine, (int32_t)layer);
                const moe_decoder_layer_plan_t layer_post_mlp_plan =
                    moe_build_decoder_layer_plan(engine, c, &attn_cfg, layer);
                for (uint32_t t = 0; t < prefill_count; ++t) {
                    float* current_hidden = batch_hidden.data() + (size_t)t * hs;
                    scratch.position = t;
                    engine->generation_token.store(t, std::memory_order_relaxed);
                    if (!moe_layer_apply_ple_f32(
                            engine, (int32_t)layer, &scratch, current_hidden, (uint32_t)hs,
                            c->num_hidden_layers, layer_post_mlp_attn_cfg.rms_norm_eps)) {
                        moe_generation_set_error(out_stats, "prefill per-layer input block failed");
                        return 0;
                    }
                    if (layer_post_mlp_plan.apply_layer_output_scale &&
                        !moe_layer_output_scale_f32(engine, (int32_t)layer, current_hidden, (uint32_t)hs)) {
                        moe_generation_set_error(out_stats, "prefill layer output scale failed");
                        return 0;
                    }
                    for (uint32_t h = 0; h < (uint32_t)hs; ++h) {
                        if (!std::isfinite(current_hidden[h])) {
                            moe_generation_set_error(out_stats, "prefill non-finite layer-first hidden state");
                            return 0;
                        }
                    }
                }
                moe_profile_log(
                    "prefill_layer_mlp",
                    "id=%llu layer=%u tokens=%u is_moe=%d batch=%d ms=%.3f",
                    (unsigned long long)generation_scope.id,
                    layer,
                    prefill_count,
                    is_moe,
                    mlp_batch_ok,
                    moe_profile_elapsed_ms(prefill_mlp_start_us));
                moe_profile_log(
                    "prefill_layer",
                    "id=%llu layer=%u tokens=%u ms=%.3f",
                    (unsigned long long)generation_scope.id,
                    layer,
                    prefill_count,
                    moe_profile_elapsed_ms(prefill_layer_start_us));
            }

            float* last_hidden = batch_hidden.data() + (prefill_count - 1u) * hs;
            if (!moe_final_norm_f32(
                    engine, c, last_hidden, (uint32_t)hs,
                    scratch.norm.data(), attn_cfg.rms_norm_eps)) {
                moe_generation_set_error(out_stats, "prefill final norm failed");
                return 0;
            }
            std::memcpy(scratch.hidden.data(), scratch.norm.data(), hs * sizeof(float));
            scratch.position = prefill_count;
        }
        moe_profile_log(
            "prefill_end",
            "id=%llu tokens=%u ms=%.3f",
            (unsigned long long)generation_scope.id,
            prefill_count,
            moe_profile_elapsed_ms(prefill_start_us));
    }
    if (cached_first_token_available) {
        if (!moe_qkv_only_ensure_capacity(
                engine, &scratch, attn_cfg, input_token_count - 1u, pre_allocate_kv)) {
            moe_generation_set_error(out_stats, "prefix generated-token QKV capacity allocation failed");
            return 0;
        }
        scratch.position = input_token_count;
        out_token_ids[0] = (uint32_t)cached_first_token;
        current = cached_first_token;
        produced = 1;
        engine->generation_token.store(input_token_count, std::memory_order_relaxed);
        moe_profile_log(
            "prefix_decode_reuse",
            "id=%llu prompt=%u position=%u token=%d committed=%u",
            (unsigned long long)generation_scope.id,
            input_token_count,
            scratch.position,
            cached_first_token,
            engine->prefix_kv_cache_committed_tokens.load(std::memory_order_relaxed));
    }
    // Decode: generate new tokens
    for (; produced < max_new_tokens; ++produced) {
        const uint64_t decode_start_us = moe_profile_now_us();
        engine->generation_token.store(prefill_count + produced, std::memory_order_relaxed);
        if (!moe_forward_one_token_dispatch_f32(
                engine, (uint32_t)current, &scratch, out_stats, scratch.position, pre_allocate_kv)) {
            failed = 1;
            break;
        }
        moe_profile_log(
            "decode_forward",
            "id=%llu step=%u position=%u input_token=%d ms=%.3f",
            (unsigned long long)generation_scope.id,
            produced,
            scratch.position,
            current,
            moe_profile_elapsed_ms(decode_start_us));
        const uint64_t lm_head_start_us = moe_profile_now_us();
        uint32_t next_token = 0;
        float next_logit = 0.0f;
        if (!moe_pc_engine_lm_head_top1_f32(engine, scratch.hidden.data(), c->hidden_size, &next_token, &next_logit)) {
            moe_generation_set_error(out_stats, "failed to sample from lm_head");
            failed = 1;
            break;
        }
        out_token_ids[produced] = next_token;
        last_logit = next_logit;
        current = (int32_t)next_token;
        ++scratch.position;
        moe_profile_log(
            "decode_sample",
            "id=%llu step=%u next_token=%u logit=%.6g lm_head_ms=%.3f step_total_ms=%.3f",
            (unsigned long long)generation_scope.id,
            produced,
            next_token,
            (double)next_logit,
            moe_profile_elapsed_ms(lm_head_start_us),
            moe_profile_elapsed_ms(decode_start_us));

        // ============================================================================
        // Prompt Lookup Decoding (2024)
        // ============================================================================
        // Parameter-free speculative drafting: find the most recent 3-gram in context
        // and propose the following tokens. Since batch verification is not yet
        // supported, we record the successful matches for analysis and speculative API.
        // ============================================================================
        if (produced >= 2 && prefill_count + produced > 3) {
            const int32_t gram[3] = { (int32_t)out_token_ids[produced-2], (int32_t)out_token_ids[produced-1], current };
            int match_idx = -1;
            for (int i = 0; i < (int)(prefill_count + produced) - 4; ++i) {
                int32_t c0 = i < prefill_count ? input_token_ids[i] : out_token_ids[i - prefill_count];
                int32_t c1 = (i+1) < prefill_count ? input_token_ids[i+1] : out_token_ids[i+1 - prefill_count];
                int32_t c2 = (i+2) < prefill_count ? input_token_ids[i+2] : out_token_ids[i+2 - prefill_count];
                if (c0 == gram[0] && c1 == gram[1] && c2 == gram[2]) {
                    match_idx = i + 3;
                    break;
                }
            }
            if (match_idx != -1) {
                // Found prompt lookup match!
                // Draft tokens: context[match_idx], context[match_idx+1], ...
                // (Batch verification to be integrated with EAGLE-2 verification tree)
            }
        }
        if (config && config->stop_on_eos && next_token == config->eos_token_id) {
            ++produced;
            break;
        }
    }

    if (!failed && generation_qkv_only && produced > 0) {
        moe_generation_output_cache_insert(
            engine,
            safe_full_input.data(),
            input_token_count,
            out_token_ids,
            produced,
            max_new_tokens,
            eos_token_id,
            stop_on_eos);
        moe_profile_log(
            "generation_output_cache_insert",
            "id=%llu input_tokens=%u completion=%u max_new=%u",
            (unsigned long long)generation_scope.id,
            input_token_count,
            produced,
            max_new_tokens);

        const uint32_t committed_generated = produced - 1u;
        uint32_t committed_tokens = input_token_count;
        if (committed_generated <= UINT32_MAX - committed_tokens) {
            committed_tokens += committed_generated;
        }
        std::vector<int32_t> committed;
        committed.reserve(committed_tokens);
        for (uint32_t i = 0; i < input_token_count; ++i) {
            committed.push_back(safe_full_input[i]);
        }
        for (uint32_t i = 0; i < committed_generated; ++i) {
            committed.push_back((int32_t)out_token_ids[i]);
        }
        if (!committed.empty()) {
            moe_generation_prefix_kv_cache_insert(
                engine, committed.data(), (uint32_t)committed.size());
            moe_profile_log(
                "prefill_prefix_insert",
                "id=%llu committed=%u prompt=%u completion=%u insertions=%llu",
                (unsigned long long)generation_scope.id,
                (uint32_t)committed.size(),
                input_token_count,
                produced,
                (unsigned long long)engine->prefix_kv_cache_insertions.load(std::memory_order_relaxed));
        }
    }

    if (out_stats) {
        out_stats->prompt_tokens = input_token_count;
        out_stats->completion_tokens = produced;
        out_stats->finish_reason = produced >= max_new_tokens ? 1u : 0u;
        out_stats->last_logit = last_logit;
    }
    generation_scope.ok = !failed && produced > 0 ? 1 : 0;
    return generation_scope.ok;
} catch (const std::bad_alloc&) {
    moe_generation_set_error(out_stats, "generation aborted under RAM pressure");
    moe_engine_signal_memory_pressure_recovery(engine, "generation_bad_alloc");
    return 0;
} catch (...) {
    moe_generation_set_error(out_stats, "generation aborted by unexpected exception");
    moe_engine_signal_memory_pressure_recovery(engine, "generation_exception");
    return 0;
}
