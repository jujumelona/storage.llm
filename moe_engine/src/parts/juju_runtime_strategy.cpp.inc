static uint64_t moe_rt_align_4k(uint64_t value) {
    return moe_juju_align_up(value, 4096ull);
}

static uint64_t moe_rt_min_u64(uint64_t a, uint64_t b) {
    return a < b ? a : b;
}

static uint64_t moe_rt_max_u64(uint64_t a, uint64_t b) {
    return a > b ? a : b;
}

static uint64_t moe_rt_clamp_u64(uint64_t value, uint64_t lo, uint64_t hi) {
    if (value < lo) {
        return lo;
    }
    return value > hi ? hi : value;
}

static uint64_t moe_rt_expert_bytes_hint(
    const moe_pc_engine* engine,
    const moe_offload_gguf_scan_t& scan
) {
    uint64_t expert_bytes = normal_bundle_max_or(24ull * 1024ull * 1024ull);
    if (engine && engine->max_tensor_stream_bytes > expert_bytes) {
        expert_bytes = engine->max_tensor_stream_bytes;
    }
    if (scan.max_executable_tensor_bytes > expert_bytes) {
        expert_bytes = scan.max_executable_tensor_bytes;
    }
    if (scan.executable_tensor_count > 0 && scan.present_total_bytes > 0) {
        const uint64_t avg_tensor = scan.present_total_bytes / scan.executable_tensor_count;
        if (avg_tensor > expert_bytes) {
            expert_bytes = avg_tensor;
        }
    }
    return moe_rt_align_4k(moe_rt_clamp_u64(expert_bytes, 512ull * 1024ull, 512ull * 1024ull * 1024ull));
}

static uint64_t moe_rt_model_bytes_hint(const moe_offload_gguf_scan_t& scan) {
    if (scan.present_total_bytes > 0) {
        return scan.present_total_bytes;
    }
    const moe_storage_constants_t* storage = moe_storage_constants();
    return storage ? storage->file_bytes : 0;
}

static double moe_rt_compute_tflops_hint(const moe_backend_caps_t& caps) {
    if (caps.tflops_hint > 0.0) return caps.tflops_hint;
    switch (caps.backend) {
    case moe_BACKEND_CUDA:
        return caps.supports_fp4 || caps.supports_scale4 ? 35.0 : 22.0;
    case moe_BACKEND_HIP:
        return 20.0;
    case moe_BACKEND_METAL:
        return 8.0;
    case moe_BACKEND_VULKAN:
    case moe_BACKEND_DIRECTML:
    case moe_BACKEND_LEVEL_ZERO:
    case moe_BACKEND_SYCL:
        return 10.0;
    case moe_BACKEND_OPENCL:
    case moe_BACKEND_WEBGPU:
        return 5.0;
    default:
        return 0.45;
    }
}

static double moe_rt_compute_mem_bw_hint(const moe_backend_caps_t& caps) {
    switch (caps.backend) {
    case moe_BACKEND_CUDA:
    case moe_BACKEND_HIP:
        return 650.0;
    case moe_BACKEND_METAL:
        return caps.supports_unified_memory ? 120.0 : 80.0;
    case moe_BACKEND_VULKAN:
    case moe_BACKEND_DIRECTML:
    case moe_BACKEND_LEVEL_ZERO:
    case moe_BACKEND_SYCL:
        return 220.0;
    default:
        return 45.0;
    }
}

static double moe_rt_fast_mem_bw_hint(const moe_pc_engine* engine) {
    if (!engine) {
        return 12.0;
    }
    if (engine->backend_caps.supports_zero_copy_host ||
        engine->backend_caps.supports_unified_memory) {
        return 120.0;
    }
    if (engine->backend_caps.supports_pinned_host &&
        engine->backend_caps.supports_async_copy) {
        return 24.0;
    }
    return 11.0;
}

static double moe_rt_slow_mem_bw_hint(const moe_pc_engine* engine) {
    if (!engine) {
        return 1.2;
    }
    if (engine->backend_caps.supports_registered_io_buffers &&
        (engine->io_config.enable_gpudirect_storage ||
         engine->io_config.enable_directstorage ||
         engine->io_config.enable_bam_nvme)) {
        return 3.5;
    }
    if (engine->io_config.enable_multistage_pipeline ||
        engine->io_config.enable_mmap_prefetch) {
        return 2.2;
    }
    return 1.2;
}

static double moe_rt_effective_compute_ms(
    const moe_pc_engine* engine,
    uint64_t expert_bytes,
    double computed_ms
) {
    const double mb = (double)expert_bytes / (1024.0 * 1024.0);
    double floor_ms = 4.0;
    if (!engine || !caps_gpu_backend_available(engine->backend_caps)) {
        floor_ms = std::min(180.0, std::max(10.0, mb * 1.8));
    } else if (engine->backend_caps.backend == moe_BACKEND_CUDA ||
               engine->backend_caps.backend == moe_BACKEND_HIP) {
        floor_ms = std::min(45.0, std::max(5.0, mb * 0.40));
    } else if (engine->backend_caps.supports_unified_memory) {
        floor_ms = std::min(70.0, std::max(6.0, mb * 0.65));
    } else {
        floor_ms = std::min(90.0, std::max(7.0, mb * 0.85));
    }
    return computed_ms > floor_ms ? computed_ms : floor_ms;
}

static int moe_rt_contract_is_iq2(const moe_offload_gguf_contract_t& contract) {
    std::string family = contract.weight_quant_family;
    std::transform(family.begin(), family.end(), family.begin(), [](unsigned char c) {
        return (char)std::tolower(c);
    });
    return (contract.weight_bits > 0u && contract.weight_bits <= 2u) ||
        family.find("iq2") != std::string::npos;
}

static double moe_rt_overlap_budget_ms(
    const moe_offload_gguf_contract_t& contract,
    double compute_ms
) {
    if (compute_ms <= 0.0) {
        return 1.0;
    }
    if (moe_rt_contract_is_iq2(contract)) {
        return std::max(1.0, compute_ms * 0.60);
    }
    return compute_ms;
}

static void moe_rt_recompute_juju_depth(moe_juju_pipeline_config_t* cfg, double overlap_budget_ms) {
    if (!cfg || overlap_budget_ms <= 0.0) {
        return;
    }
    const uint64_t depth = cfg->tier2_to_tier1_ms > 0.0 ?
        (uint64_t)std::ceil(cfg->tier2_to_tier1_ms / overlap_budget_ms) + 1ull : 2ull;
    cfg->prefetch_depth = (uint32_t)moe_rt_clamp_u64(depth, 2ull, 16ull);
    cfg->io_threads = (uint32_t)moe_rt_clamp_u64((uint64_t)cfg->prefetch_depth * 2ull, 2ull, 16ull);
}

static void moe_rt_build_tiers(
    const moe_pc_engine* engine,
    uint64_t expert_bytes,
    moe_juju_tier_capability_t* tier0,
    moe_juju_tier_capability_t* tier1,
    moe_juju_tier_capability_t* tier2
) {
    std::memset(tier0, 0, sizeof(*tier0));
    std::memset(tier1, 0, sizeof(*tier1));
    std::memset(tier2, 0, sizeof(*tier2));
    const moe_backend_caps_t& caps = engine->backend_caps;
    const int gpu_available = engine->config.prefer_gpu && caps_gpu_backend_available(caps);
    const uint64_t ram_budget = engine->config.ram_budget_bytes ? engine->config.ram_budget_bytes :
        moe_rt_max_u64(caps.recommended_pinned_bytes, expert_bytes * 8ull);
    uint64_t vram_budget = gpu_available ? vram_expert_budget_bytes(engine) : 0;
    if (gpu_available && vram_budget == 0) {
        vram_budget = engine->config.vram_budget_bytes ? engine->config.vram_budget_bytes :
            caps.recommended_vram_cache_bytes;
    }
    if (gpu_available && engine->device_free_bytes.load(std::memory_order_relaxed) > 0) {
        vram_budget = moe_rt_min_u64(vram_budget, engine->device_free_bytes.load(std::memory_order_relaxed));
    }

    tier0->capacity_bytes = gpu_available ? vram_budget : ram_budget;
    tier0->bandwidth_gbps = gpu_available ? moe_rt_compute_mem_bw_hint(caps) : 45.0;
    tier0->latency_us = gpu_available ? 0.2 : 1.0;
    tier0->compute_tflops = moe_rt_compute_tflops_hint(caps);
    tier0->supports_async = gpu_available ? caps.supports_backend_async_api : 1;
    tier0->supports_dma = gpu_available ? caps.supports_async_copy : 0;
    tier0->is_unified_memory = caps.supports_unified_memory || caps.supports_zero_copy_host;

    tier1->capacity_bytes = caps.supports_zero_copy_host ? tier0->capacity_bytes : ram_budget;
    tier1->bandwidth_gbps = moe_rt_fast_mem_bw_hint(engine);
    tier1->latency_us = caps.supports_zero_copy_host ? tier0->latency_us : 2.0;
    tier1->compute_tflops = 0.0;
    tier1->supports_async = engine->io_config.enable_multistage_pipeline || caps.supports_pinned_host;
    tier1->supports_dma = caps.supports_pinned_host || caps.supports_zero_copy_host;
    tier1->is_unified_memory = caps.supports_unified_memory || caps.supports_zero_copy_host;

    tier2->capacity_bytes = engine->config.db_budget_bytes ? engine->config.db_budget_bytes : UINT64_MAX / 4ull;
    tier2->bandwidth_gbps = moe_rt_slow_mem_bw_hint(engine);
    tier2->latency_us = 80.0;
    tier2->compute_tflops = 0.0;
    tier2->supports_async = engine->io_config.enable_mmap_prefetch || engine->io_config.enable_multistage_pipeline;
    tier2->supports_dma = caps.supports_registered_io_buffers &&
        (caps.supports_gpudirect_storage || caps.supports_directstorage || caps.supports_gpu_direct_file_io);
    tier2->is_unified_memory = 0;
}

static void moe_rt_apply_juju_io_flags(
    moe_pc_engine* engine,
    moe_juju_execution_path_t path,
    const moe_juju_pipeline_config_t& cfg,
    moe_io_config_t* io
) {
    io->enable_compute_io_overlap = 1;
    io->enable_lookahead_prefetch = 1;
    io->enable_topology_prefetch = 1;
    io->enable_mmap_prefetch = 1;
    io->prefetch_window_layers = cfg.prefetch_depth ? cfg.prefetch_depth : io->prefetch_window_layers;
    io->io_worker_threads = clamp_to_cap(cfg.io_threads ? cfg.io_threads : io->io_worker_threads, disk_worker_cap());
    io->disk_worker_threads = io->io_worker_threads;

    const uint32_t pinned_cap = pinned_worker_cap(engine);
    const uint32_t gpu_cap = gpu_worker_cap(engine);
    io->pinned_worker_threads = pinned_cap ?
        clamp_to_cap(std::max(1u, (cfg.io_threads + 1u) / 2u), pinned_cap) : 0u;
    io->gpu_worker_threads = gpu_cap ?
        clamp_to_cap(std::min(3u, std::max(1u, engine->backend_caps.copy_streams)), gpu_cap) : 0u;

    const uint64_t segment = cfg.segment_bytes ? cfg.segment_bytes :
        normal_bundle_max_or(24ull * 1024ull * 1024ull);
    io->direct_upload_min_bytes = segment > 0xffffffffull ? 0xffffffffull : segment;
    if (io->max_prefetch_queue < 12288u) {
        io->max_prefetch_queue = 12288u;
    }
    if (io->topology_max_candidates < 4u) {
        io->topology_max_candidates = 4u;
    }
    if (io->topology_max_edges_per_source < 64u) {
        io->topology_max_edges_per_source = 64u;
    }

    const uint32_t active_workers = io->disk_worker_threads + io->pinned_worker_threads + io->gpu_worker_threads;
    uint64_t staging_target = segment * (uint64_t)std::max(8u, active_workers * 2u);
    staging_target = moe_rt_max_u64(staging_target, 512ull * 1024ull * 1024ull);
    if (engine->config.ram_budget_bytes > 0) {
        staging_target = moe_rt_min_u64(staging_target, engine->config.ram_budget_bytes / 2ull);
    }
    io->pinned_staging_bytes = moe_rt_align_4k(staging_target);

    switch (path) {
    case moe_JUJU_EXEC_DIRECT_STORAGE_TO_COMPUTE:
        io->enable_multistage_pipeline = 1;
        io->enable_pinned_host = engine->backend_caps.supports_pinned_host ? 1 : 0;
        // BUGFIX 157: Enable DirectStorage when backend supports it
        // DirectStorage 1.0/1.1 API is implemented in prefetch_bytes.cpp.inc
        // Only enable if backend explicitly reports support to avoid fallback overhead
        io->enable_directstorage = engine->backend_caps.supports_directstorage ? 1 : 0;
        io->enable_gpudirect_storage = engine->backend_caps.supports_gpudirect_storage ? 1 : 0;
        io->enable_bam_nvme = engine->backend_caps.supports_gpu_direct_file_io ? 1 : 0;
        break;
    case moe_JUJU_EXEC_THREE_TIER_PIPELINE:
    case moe_JUJU_EXEC_ROTATING_BUFFER:
        io->enable_multistage_pipeline = 1;
        io->enable_pinned_host = engine->backend_caps.supports_pinned_host ? 1 : 0;
        io->enable_directstorage = 0;
        io->enable_gpudirect_storage = 0;
        io->enable_bam_nvme = 0;
        break;
    case moe_JUJU_EXEC_FULL_COMPUTE_MEM:
    case moe_JUJU_EXEC_CPU_WITH_CACHE:
    case moe_JUJU_EXEC_STORAGE_STREAMING:
    default:
        io->enable_directstorage = 0;
        io->enable_gpudirect_storage = 0;
        io->enable_bam_nvme = 0;
        if (engine_bypass_staging_pipeline(engine) || path == moe_JUJU_EXEC_FULL_COMPUTE_MEM) {
            io->enable_pinned_host = 0;
            io->enable_multistage_pipeline = 0;
            io->pinned_staging_bytes = 0;
            io->pinned_worker_threads = 0;
            io->gpu_worker_threads = 0;
            // Extreme tuning for CPU-only or unified-memory zero-copy paths.
            if (path == moe_JUJU_EXEC_FULL_COMPUTE_MEM) {
                // Everything is in RAM, IO workers just waste CPU cycles context switching
                io->io_worker_threads = 1;
                io->disk_worker_threads = 1;
            } else if (!engine->config.prefer_gpu || engine->backend_caps.supports_unified_memory) {
                // CPU_WITH_CACHE or unified GPU. Limit IO to preserve compute cores.
                // 2 threads are enough to saturate NVMe page faults without thrashing AVX2.
                io->io_worker_threads = io->io_worker_threads > 2u ? 2u : io->io_worker_threads;
                io->disk_worker_threads = io->io_worker_threads;
            }
        } else {
            io->enable_multistage_pipeline = 1;
            io->enable_pinned_host = engine->backend_caps.supports_pinned_host ? 1 : 0;
        }
        break;
    }
}

static std::vector<int> moe_juju_predict_co_activation(moe_pc_engine* engine, int target_layer, int top_k) {
    std::vector<int> predictions;
    if (!engine || target_layer < 2) return predictions;
    // BUGFIX 520: Check top_k upper bound to prevent overflow ★
    // Problem: top_k can be arbitrarily large from caller
    // Loop can run billions of iterations
    // Solution: Add reasonable upper bound check (e.g., 1000)
    if (top_k > 1000) return predictions;

    int source_layer = target_layer - 2;
    if (source_layer >= engine->last_selected_experts.size() || engine->last_selected_experts[source_layer].empty()) {
        return predictions;
    }

    if (target_layer >= engine->co_activation_stats.size()) {
        return predictions;
    }

    const auto& stats = engine->co_activation_stats[target_layer];
    std::unordered_map<int, float> aggregated_scores;

    for (int src_expert : engine->last_selected_experts[source_layer]) {
        auto it = stats.find(src_expert);
        if (it != stats.end()) {
            for (const auto& pair : it->second) {
                aggregated_scores[pair.first] += pair.second;
            }
        }
    }

    std::vector<std::pair<int, float>> sorted_scores(aggregated_scores.begin(), aggregated_scores.end());
    std::sort(sorted_scores.begin(), sorted_scores.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    for (int i = 0; i < std::min((int)sorted_scores.size(), top_k); ++i) {
        predictions.push_back(sorted_scores[i].first);
    }

    return predictions;
}

static void moe_juju_update_co_activation(moe_pc_engine* engine, int current_layer, const std::vector<int>& selected) {
    if (!engine) return;

    // BUGFIX 505: current_layer 범위 체크
    // BUGFIX 635: current_layer 음수 체크 추가 ★
    // Problem: current_layer가 음수일 때 resize 호출 전 체크 없음
    // Solution: current_layer < 0 체크 추가
    if (current_layer < 0) return;
    if (current_layer >= engine->last_selected_experts.size() && current_layer < 10000) {
        engine->last_selected_experts.resize(current_layer + 1);
        engine->co_activation_stats.resize(current_layer + 1);
    }

    // BUGFIX 636: current_layer 범위 재확인 ★
    // Problem: resize 후에도 current_layer가 범위 내인지 확인 필요
    // Solution: 인덱스 접근 전 범위 체크
    if (current_layer >= engine->last_selected_experts.size()) return;

    // BUGFIX 636: current_layer 범위 재확인 ★
    // Problem: resize 후에도 current_layer가 범위 내인지 확인 필요
    // Solution: 인덱스 접근 전 범위 체크
    if (current_layer >= engine->last_selected_experts.size()) return;

    engine->last_selected_experts[current_layer] = selected;

    int source_layer = current_layer - 2;
    // BUGFIX 637: source_layer 범위 체크 강화 ★
    // Problem: source_layer가 음수일 때 >= 0 체크만으로 불충분
    // Solution: 명시적으로 음수 체크 및 범위 체크
    if (source_layer < 0 || source_layer >= engine->last_selected_experts.size()) return;
    if (engine->last_selected_experts[source_layer].empty()) return;

    // BUGFIX 638: co_activation_stats 범위 체크 ★
    // Problem: current_layer가 co_activation_stats 범위를 벗어날 수 있음
    // Solution: 접근 전 범위 체크
    if (current_layer >= engine->co_activation_stats.size()) return;

    auto& stats = engine->co_activation_stats[current_layer];
    for (int src_expert : engine->last_selected_experts[source_layer]) {
        for (int tgt_expert : selected) {
            stats[src_expert][tgt_expert]++;
        }
    }

    // BUGFIX 701: Sliding window eviction to prevent co_activation_stats memory leak ★★
    // Problem: co_activation_stats grows unboundedly — 256×256×60 layers = ~400MB at 3M tokens.
    // Solution: Every 100K token updates, clear all stats. The merge_map already captured
    // the clustering info from previous data, so raw counters are safely disposable.
    // Fresh data starts accumulating immediately for the next merge_map rebuild cycle.
    ++engine->co_activation_token_count;
    constexpr uint64_t kCoActivationEvictInterval = 100000;
    if (engine->co_activation_token_count - engine->last_merge_build_token_count >= kCoActivationEvictInterval) {
        for (auto& layer_stats : engine->co_activation_stats) {
            layer_stats.clear();
        }
        engine->last_merge_build_token_count = engine->co_activation_token_count;
    }
}

// ============================================================================
// Expert Co-activation Merging (MC-MoE 2024, EvoMoE)
// Union-find groups experts with >80% co-activation into logical clusters.
// Prefetch pipeline loads entire cluster when any member is selected.
// ============================================================================
static void moe_build_expert_merge_map(moe_pc_engine* engine) {
    if (!engine) return;
    std::unique_lock<std::shared_mutex> lock(engine->expert_merge_map.mtx);
    engine->expert_merge_map.valid = false;
    const moe_storage_constants_t runtime_c = moe_engine_storage_constants_or_static(engine);
    const moe_storage_constants_t* c = &runtime_c;
    if (c->experts_per_moe_layer == 0) return;
    const uint32_t nL = c->num_hidden_layers, nE = c->experts_per_moe_layer;
    engine->expert_merge_map.merge_groups.resize(nL);
    engine->expert_merge_map.group_members.resize(nL);
    for (uint32_t L = c->first_moe_layer; L <= c->last_moe_layer && L < nL; ++L) {
        auto& grp = engine->expert_merge_map.merge_groups[L];
        auto& mem = engine->expert_merge_map.group_members[L];
        grp.assign(nE, -1); mem.clear();
        if (L >= engine->co_activation_stats.size()) {
            for (uint32_t e = 0; e < nE; ++e) { grp[e]=(int)e; mem[(int)e]={(int)e}; }
            continue;
        }
        std::vector<int> par(nE);
        for (uint32_t i = 0; i < nE; ++i) par[i]=(int)i;
        std::function<int(int)> find=[&](int x)->int{while(par[x]!=x){par[x]=par[par[x]];x=par[x];}return x;};
        for (const auto& [src,tgts] : engine->co_activation_stats[L]) {
            float tot=0; for (const auto& [t,c2]:tgts) tot+=c2;
            if (tot<10.f) continue;
            for (const auto& [t,c2]:tgts) {
                if (src!=t && c2/tot>0.80f && src>=0 && t>=0 && (uint32_t)src<nE && (uint32_t)t<nE) {
                    int a=find(src),b=find(t); if(a!=b) par[a]=b;
                }
            }
        }
        std::unordered_map<int,int> r2g; int ng=0;
        for (uint32_t e=0;e<nE;++e){
            int r=find((int)e); auto it=r2g.find(r);
            if(it==r2g.end()){r2g[r]=ng;grp[e]=ng;mem[ng].push_back((int)e);++ng;}
            else{grp[e]=it->second;mem[it->second].push_back((int)e);}
        }
    }
    engine->expert_merge_map.valid = true;

    // BUGFIX 701: Clear stats after merge map rebuild ★★
    // merge_map now holds the clustered grouping info. Raw counters are no longer needed.
    // Clearing here ensures the next rebuild cycle uses only fresh post-rebuild data,
    // implementing a true sliding window that prevents unbounded memory growth.
    for (auto& layer_stats : engine->co_activation_stats) {
        layer_stats.clear();
    }
    engine->last_merge_build_token_count = engine->co_activation_token_count;
}

static void moe_expand_merged_experts(moe_pc_engine* engine, int layer, const std::vector<int>& sel, std::vector<int>& out) {
    out = sel;
    if (!engine) return;
    std::shared_lock<std::shared_mutex> lock(engine->expert_merge_map.mtx);
    // BUGFIX 277: Add comprehensive bounds checking ★★
    // Problem: layer index checked once, but merge_groups[layer] accessed again without check
    // If layer is valid but e is out of bounds, undefined behavior occurs
    // Solution: Check both layer and expert indices before every access
    if (!engine->expert_merge_map.valid || layer < 0 ||
        (size_t)layer >= engine->expert_merge_map.group_members.size() ||
        (size_t)layer >= engine->expert_merge_map.merge_groups.size()) {
        return;
    }
    const auto& mem = engine->expert_merge_map.group_members[layer];
    const auto& grp = engine->expert_merge_map.merge_groups[layer];
    std::unordered_set<int> seen(sel.begin(), sel.end());
    for (int e : sel) {
        // BUGFIX 277: Check expert index before accessing merge_groups
        if (e < 0 || (size_t)e >= grp.size()) continue;
        auto it = mem.find(grp[e]);
        if (it != mem.end()) {
            for (int m : it->second) {
                if (seen.insert(m).second) {
                    out.push_back(m);
                }
            }
        }
    }
}

static int moe_engine_apply_juju_strategy(
    moe_pc_engine* engine,
    const moe_offload_gguf_scan_t& scan,
    const moe_offload_gguf_contract_t& contract
) {
    if (!engine) {
        return 0;
    }
    const int has_strategy_contract =
        contract.has_juju_contracts || contract.optimization_catalog_count >= 5000u;
    if (!has_strategy_contract) {
        engine->juju_strategy_valid.store(0, std::memory_order_relaxed);
        return 0;
    }

    const uint64_t expert_bytes = moe_rt_expert_bytes_hint(engine, scan);
    const uint64_t model_bytes = moe_rt_model_bytes_hint(scan);
    moe_juju_tier_capability_t tier0{}, tier1{}, tier2{};
    moe_rt_build_tiers(engine, expert_bytes, &tier0, &tier1, &tier2);

    moe_juju_pipeline_config_t cfg{};
    if (!moe_juju_compute_pipeline_config(&tier0, &tier1, &tier2, expert_bytes, 0.0, &cfg)) {
        engine->juju_strategy_valid.store(0, std::memory_order_relaxed);
        return 0;
    }
    cfg.compute_ms = moe_rt_effective_compute_ms(engine, expert_bytes, cfg.compute_ms);
    if (contract.reference_moe_compute_ms > 0.0) {
        cfg.compute_ms = contract.reference_moe_compute_ms;
    }
    if (contract.reference_ram_to_gpu_ms > 0.0) {
        cfg.tier1_to_tier0_ms = std::max(cfg.tier1_to_tier0_ms, contract.reference_ram_to_gpu_ms);
    }
    if (contract.reference_nvme_to_ram_ms > 0.0) {
        cfg.tier2_to_tier1_ms = std::max(cfg.tier2_to_tier1_ms, contract.reference_nvme_to_ram_ms);
    }
    moe_rt_recompute_juju_depth(&cfg, moe_rt_overlap_budget_ms(contract, cfg.compute_ms));
    if (moe_rt_contract_is_iq2(contract) && cfg.prefetch_depth < 3u) {
        cfg.prefetch_depth = 3u;
        cfg.io_threads = std::max<uint32_t>(cfg.io_threads, 6u);
    }

    moe_juju_execution_path_t path = moe_juju_select_execution_path(
        engine->config.prefer_gpu && caps_gpu_backend_available(engine->backend_caps),
        model_bytes,
        expert_bytes,
        &tier0,
        &tier1,
        &tier2
    );

    moe_io_config_t io = engine->io_config;
    moe_rt_apply_juju_io_flags(engine, path, cfg, &io);

    engine->juju_execution_path = path;
    engine->juju_pipeline_config = cfg;
    engine->juju_strategy_valid.store(1, std::memory_order_relaxed);
    const int configured = moe_pc_engine_configure_io(engine, &io);
    if (!configured) {
        engine->io_config = io;
        engine->runtime_prefetch_window_layers.store(io.prefetch_window_layers, std::memory_order_relaxed);
        runtime_store_io_workers(engine, io.disk_worker_threads, io.pinned_worker_threads, io.gpu_worker_threads);
    }

    std::fprintf(
        stderr,
        "[storagellm] JUJU strategy: path=%s catalog=%u expert=%llu segment=%llu segments=%u "
        "prefetch=%u io_threads=%u block=%llu compute_ms=%.3f t1_t0_ms=%.3f t2_t1_ms=%.3f\n",
        moe_juju_execution_path_name(path),
        contract.optimization_catalog_count,
        (unsigned long long)expert_bytes,
        (unsigned long long)cfg.segment_bytes,
        cfg.segment_count,
        cfg.prefetch_depth,
        cfg.io_threads,
        (unsigned long long)cfg.block_size,
        cfg.compute_ms,
        cfg.tier1_to_tier0_ms,
        cfg.tier2_to_tier1_ms
    );
    std::fflush(stderr);
    return 1;
}
