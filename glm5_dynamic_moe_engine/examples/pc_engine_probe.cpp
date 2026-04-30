#include "glm5_pc_engine.h"
#include "../native/glm5_scale4.h"
#include "../../loader/weight_loader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

static void print_forward_status(const char* label, glm5_pc_engine_t* engine) {
    glm5_forward_status_t status;
    if (!glm5_pc_engine_get_forward_status(engine, &status)) {
        printf("forward[%s]=unavailable\n", label);
        return;
    }
    printf(
        "forward[%s] storage=%d tensors=%d expert_triplet=%d tokenizer=%d embed=%d lm_head=%d attn_proj=%d attn_decode=%d kv_cache=%d attention=%d sampler=%d decode_loop=%d chat=%d tensor_count=%llu\n",
        label,
        status.storage_state_valid,
        status.tensor_table_loaded,
        status.expert_triplet_available,
        status.tokenizer_ready,
        status.embedding_ready,
        status.lm_head_ready,
        status.attention_projector_ready,
        status.attention_decode_ready,
        status.kv_cache_ready,
        status.attention_ready,
        status.sampler_ready,
        status.decode_loop_ready,
        status.chat_graph_ready,
        (unsigned long long)status.tensor_count
    );
}

static void print_model_root_check(const char* model_root) {
    if (!model_root || !model_root[0]) {
        return;
    }
    glm5_model_root_check_t check{};
    if (!glm5_storage_validate_model_root(model_root, &check)) {
        printf("model_root path=%s check=unavailable\n", model_root);
        return;
    }
    printf(
        "model_root path=%s valid=%d expected_parts=%u present=%u missing=%u size_mismatch=%u expected_bytes=%llu present_bytes=%llu\n",
        model_root,
        check.valid,
        check.expected_part_count,
        check.present_part_count,
        check.missing_part_count,
        check.size_mismatch_count,
        (unsigned long long)check.expected_total_bytes,
        (unsigned long long)check.present_total_bytes
    );
    if (check.first_missing_part) {
        printf(
            "model_root first_missing part=%u path=%s\n",
            check.first_missing_part,
            check.first_missing_path
        );
    }
    if (check.first_size_mismatch_part) {
        printf(
            "model_root first_size_mismatch part=%u path=%s expected=%llu actual=%llu\n",
            check.first_size_mismatch_part,
            check.first_size_mismatch_path,
            (unsigned long long)check.first_expected_bytes,
            (unsigned long long)check.first_actual_bytes
        );
    }
}

static void print_scale4_probe(const char* scale4_path) {
    if (!scale4_path || !scale4_path[0]) {
        return;
    }
    glm5_scale4_file_t* scale4 = glm5_scale4_open(scale4_path);
    if (!scale4) {
        printf("scale4 path=%s open=0 entries=0\n", scale4_path);
        return;
    }
    const uint32_t count = glm5_scale4_entry_count(scale4);
    const glm5_scale4_entry_t* first = glm5_scale4_entry_at(scale4, 0);
    const char* first_key = glm5_scale4_entry_key(scale4, first);
    printf(
        "scale4 path=%s open=1 entries=%u first_key=%s\n",
        scale4_path,
        count,
        first_key ? first_key : ""
    );
    glm5_scale4_close(scale4);
}

static int probe_main(int argc, char** argv) {
    glm5_pc_engine_config_t cfg = glm5_pc_default_config();
    cfg.vram_budget_bytes = 128ull * 1024ull * 1024ull;
    cfg.ram_budget_bytes = 512ull * 1024ull * 1024ull;
    const char* table_path = NULL;
    const char* model_root = NULL;
    const char* scale4_path = NULL;
    const char* topology_path = NULL;
    const char* manifest_path = NULL;
    const char* footer_root = NULL;
    int prefetch_layer = -1;
    int prefetch_expert = -1;
    int learn_next_expert = -1;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--qkv") == 0) {
            cfg.kv_mode = GLM5_KV_MODE_QKV;
        } else if (strcmp(argv[i], "--table") == 0 && i + 1 < argc) {
            table_path = argv[++i];
        } else if (strcmp(argv[i], "--model-root") == 0 && i + 1 < argc) {
            model_root = argv[++i];
        } else if (strcmp(argv[i], "--scale4") == 0 && i + 1 < argc) {
            scale4_path = argv[++i];
        } else if (strcmp(argv[i], "--topology") == 0 && i + 1 < argc) {
            topology_path = argv[++i];
        } else if (strcmp(argv[i], "--manifest") == 0 && i + 1 < argc) {
            manifest_path = argv[++i];
        } else if (strcmp(argv[i], "--footer-root") == 0 && i + 1 < argc) {
            footer_root = argv[++i];
        } else if (strcmp(argv[i], "--prefetch-layer") == 0 && i + 1 < argc) {
            prefetch_layer = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--prefetch-expert") == 0 && i + 1 < argc) {
            prefetch_expert = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--learn-next-expert") == 0 && i + 1 < argc) {
            learn_next_expert = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            const char* name = argv[++i];
            if (strcmp(name, "cpu") == 0) {
                cfg.preferred_backend = GLM5_BACKEND_CPU;
            } else if (strcmp(name, "cuda") == 0) {
                cfg.preferred_backend = GLM5_BACKEND_CUDA;
            } else if (strcmp(name, "hip") == 0 || strcmp(name, "rocm") == 0 || strcmp(name, "amd") == 0) {
                cfg.preferred_backend = GLM5_BACKEND_HIP;
            } else if (strcmp(name, "metal") == 0) {
                cfg.preferred_backend = GLM5_BACKEND_METAL;
            } else if (strcmp(name, "vulkan") == 0) {
                cfg.preferred_backend = GLM5_BACKEND_VULKAN;
            } else if (strcmp(name, "directml") == 0) {
                cfg.preferred_backend = GLM5_BACKEND_DIRECTML;
            } else if (strcmp(name, "opencl") == 0) {
                cfg.preferred_backend = GLM5_BACKEND_OPENCL;
            } else if (strcmp(name, "levelzero") == 0 || strcmp(name, "level_zero") == 0 || strcmp(name, "intel") == 0) {
                cfg.preferred_backend = GLM5_BACKEND_LEVEL_ZERO;
            } else if (strcmp(name, "sycl") == 0 || strcmp(name, "oneapi") == 0) {
                cfg.preferred_backend = GLM5_BACKEND_SYCL;
            } else if (strcmp(name, "webgpu") == 0) {
                cfg.preferred_backend = GLM5_BACKEND_WEBGPU;
            } else {
                cfg.preferred_backend = GLM5_BACKEND_AUTO;
            }
        } else if (strcmp(argv[i], "--platform") == 0 && i + 1 < argc) {
            const char* name = argv[++i];
            if (strcmp(name, "windows") == 0) {
                cfg.platform = GLM5_PLATFORM_WINDOWS_PC;
            } else if (strcmp(name, "linux") == 0) {
                cfg.platform = GLM5_PLATFORM_LINUX_PC;
            } else if (strcmp(name, "mac") == 0 || strcmp(name, "apple") == 0) {
                cfg.platform = GLM5_PLATFORM_MACOS_APPLE;
            } else if (strcmp(name, "cpu") == 0) {
                cfg.platform = GLM5_PLATFORM_CPU_ONLY;
            } else {
                cfg.platform = GLM5_PLATFORM_AUTO;
            }
        }
    }

    if (manifest_path && model_root && prefetch_layer >= 0 && prefetch_expert >= 0) {
        storagellm::WeightLoader loader;
        const char* footer = footer_root ? footer_root : "";
        storagellm::ExpertWeights weights;
        const int opened = loader.open(model_root, manifest_path, footer) ? 1 : 0;
        const int loaded = opened && loader.load_expert(
            (uint32_t)prefetch_layer,
            (uint32_t)prefetch_expert,
            &weights
        ) ? 1 : 0;
        printf(
            "storage_loader manifest_open=%d expert_loaded=%d L%d.E%d gate=%llu up=%llu down=%llu scale_mode=%s\n",
            opened,
            loaded,
            prefetch_layer,
            prefetch_expert,
            (unsigned long long)weights.gate.weight.bytes,
            (unsigned long long)weights.up.weight.bytes,
            (unsigned long long)weights.down.weight.bytes,
            weights.gate.scale_mode.c_str()
        );
        loader.close();
    }

    glm5_pc_engine_t* engine = glm5_pc_engine_create(&cfg);
    if (!engine) {
        return 1;
    }
    if (model_root) {
        glm5_pc_engine_set_model_root(engine, model_root);
        print_model_root_check(model_root);
    }
    print_scale4_probe(scale4_path);
    if (topology_path) {
        glm5_pc_engine_load_topology(engine, topology_path);
    }

    glm5_backend_caps_t caps;
    if (glm5_pc_engine_get_backend_caps(engine, &caps)) {
        printf(
            "backend=%s platform=%s async_copy=%d backend_async=%d pinned=%d zero_copy=%d fixed_read=%d registered_io=%d unified=%d mmap=%d cuda=%d hip=%d level_zero=%d sycl=%d metal=%d vulkan=%d directml=%d opencl=%d webgpu=%d directstorage=%d gds=%d rec_vram=%llu common=%llu\n",
            glm5_backend_name(caps.backend),
            glm5_platform_name(caps.platform),
            caps.supports_async_copy,
            caps.supports_backend_async_api,
            caps.supports_pinned_host,
            caps.supports_zero_copy_host,
            caps.supports_fixed_read_staging,
            caps.supports_registered_io_buffers,
            caps.supports_unified_memory,
            caps.supports_mmap,
            caps.supports_cuda,
            caps.supports_hip,
            caps.supports_level_zero,
            caps.supports_sycl,
            caps.supports_metal,
            caps.supports_vulkan,
            caps.supports_directml,
            caps.supports_opencl,
            caps.supports_webgpu,
            caps.supports_directstorage,
            caps.supports_gpudirect_storage,
            (unsigned long long)caps.recommended_vram_cache_bytes,
            (unsigned long long)caps.recommended_vram_common_bytes
        );
    }
    glm5_model_shape_t shape = glm5_pc_glm51_model_shape();
    printf(
        "shape layers=%u moe=%u-%u experts=%u hidden=%u expert_hidden=%u vocab=%u\n",
        shape.num_hidden_layers,
        shape.first_moe_layer,
        shape.last_moe_layer,
        shape.experts_per_moe_layer,
        shape.hidden_size,
        shape.expert_intermediate_size,
        shape.vocab_size
    );
    const glm5_storage_model_state_t* model_state = glm5_storage_model_state_summary();
    printf(
        "storage parts=%u shards=%u-%u covered=%u missing=%u dup_primary=%u bytes=%llu blocks=%u experts=%u raw_experts=%u raw_tensors=%u raw_tensor_bytes=%llu valid=%d\n",
        model_state->part_count,
        model_state->shard_first,
        model_state->shard_last,
        model_state->shard_covered_count,
        model_state->missing_shard_count,
        model_state->duplicate_primary_shard_count,
        (unsigned long long)model_state->total_bytes,
        model_state->block_count,
        model_state->expert_bundle_count,
        model_state->raw_expert_bundle_count,
        model_state->raw_tensor_count,
        (unsigned long long)model_state->raw_tensor_bytes,
        glm5_storage_model_state_valid()
    );
    const glm5_storage_layer_spec_t* layer78 = glm5_storage_layer_at(78);
    if (layer78) {
        printf(
            "layer78 parts=%u-%u shards=%u-%u raw_expert_scales=%u bundle_bytes=%llu\n",
            layer78->part0,
            layer78->part1,
            layer78->source_shard0,
            layer78->source_shard1,
            layer78->raw_expert_scale_count,
            (unsigned long long)layer78->bundle_total_bytes
        );
    }
    glm5_raw_tensor_info_t raw_info;
    if (glm5_pc_engine_find_raw_tensor(engine, 0, GLM5_RAW_TENSOR_Q_A_PROJ, &raw_info)) {
        printf(
            "raw L0.q_a_proj part=%u off=%llu bytes=%llu rows=%u cols=%u bf16=%d\n",
            raw_info.part,
            (unsigned long long)raw_info.offset,
            (unsigned long long)raw_info.bytes,
            raw_info.rows,
            raw_info.cols,
            raw_info.bf16
        );
    }
    if (glm5_pc_engine_find_raw_tensor(engine, 0, GLM5_RAW_TENSOR_O_PROJ, &raw_info)) {
        printf(
            "raw L0.o_proj part=%u off=%llu bytes=%llu rows=%u cols=%u bf16=%d\n",
            raw_info.part,
            (unsigned long long)raw_info.offset,
            (unsigned long long)raw_info.bytes,
            raw_info.rows,
            raw_info.cols,
            raw_info.bf16
        );
    }
    const int selected_experts[2] = {0, 1};
    const int next_candidates[2] = {2, 3};
    glm5_runtime_orchestrator_state_t orch_state{};
    orch_state.policy = GLM5_EXECUTION_BALANCED;
    orch_state.cpu_load_pct = 50.0f;
    orch_state.gpu_load_pct = 50.0f;
    orch_state.active_tokens = 1;
    glm5_runtime_orchestrator_plan_t orch_plan{};
    glm5_expert_prefetch_item_t orch_items[16];
    uint32_t orch_count = 0;
    glm5_pc_engine_prepare_moe_activation(engine, 3);
    if (glm5_pc_engine_orchestrate_moe_step(
        engine, 3, selected_experts, 2, next_candidates, 2,
        &orch_state, &orch_plan, orch_items, 16, &orch_count)) {
        printf(
            "orchestrator policy=%u tier=%s spec_tier=%s prefetch=%u selected_bytes=%llu workers=%u/%u/%u throttle=%d drop=%d\n",
            (unsigned)orch_plan.policy,
            glm5_tier_name(orch_plan.selected_tier),
            glm5_tier_name(orch_plan.speculative_tier),
            orch_count,
            (unsigned long long)orch_plan.selected_expert_bytes,
            orch_plan.disk_workers,
            orch_plan.pinned_workers,
            orch_plan.gpu_workers,
            orch_plan.throttle_speculative,
            orch_plan.drop_speculative
        );
    }
    glm5_runtime_optimization_plan_t plan;
    if (glm5_pc_engine_get_optimization_plan(engine, GLM5_PHASE_BOTH, &plan)) {
        printf(
            "plan static_shape=%d static_pool=%d prefetch=%d async=%d pinned_cache=%d direct_io=%d fused_fp4=%d scale4_fusion=%d cuda_graph=%d metal_cmd=%d paged_kv=%d split=%d streams=%u/%u/%u common_vram=%llu expert_vram=%llu\n",
            plan.use_model_shape_constants,
            plan.use_static_memory_pool,
            plan.use_expert_prefetch,
            plan.use_async_copy_stream,
            plan.use_pinned_ram_cache,
            plan.use_direct_to_gpu_io,
            plan.use_fused_fp4_dequant_matmul,
            plan.use_scale4_decode_fusion,
            plan.use_cuda_graphs,
            plan.use_metal_command_buffers,
            plan.use_paged_kv_cache,
            plan.use_prefill_decode_split,
            plan.compute_streams,
            plan.copy_streams,
            plan.prefetch_streams,
            (unsigned long long)plan.recommended_vram_common_resident_bytes,
            (unsigned long long)plan.recommended_vram_expert_cache_bytes
        );
    }

    glm5_pc_engine_request_expert(engine, 10, 0, 80ull * 1024ull * 1024ull, GLM5_TIER_VRAM);
    glm5_pc_engine_request_expert(engine, 10, 1, 80ull * 1024ull * 1024ull, GLM5_TIER_VRAM);

    glm5_pc_engine_stats_t stats;
    if (!glm5_pc_engine_get_stats(engine, &stats)) {
        glm5_pc_engine_destroy(engine);
        return 2;
    }

    printf(
        "kv=%s offload_gguf=%d files=%u tensor_headers=%llu executable_tensors=%llu qkv_forced=%u qkv_bits=%u/%u qkv_group=%u qkv_page=%u qkv_sink=%u weight=%s/%ub enc=%u kernel=%s first_gguf=%s vram=%llu common=%llu expert_vram=%llu device=%llu allocs=%llu device_mem=%llu/%llu device_pools=%llu/%llu/%llu/%llu model_lib=%d/%d/%d path=%s common_prefetch=%llu/%llu ram=%llu db=%llu experts=%llu tiers=%llu/%llu/%llu\n",
        glm5_kv_mode_name(stats.kv_mode),
        stats.offload_gguf_valid,
        stats.offload_gguf_file_count,
        (unsigned long long)stats.offload_gguf_tensor_count,
        (unsigned long long)stats.offload_gguf_executable_tensor_count,
        stats.qkv_forced_by_format,
        stats.qkv_k_bits,
        stats.qkv_v_bits,
        stats.qkv_group_size,
        stats.qkv_page_size_tokens,
        stats.qkv_sink_tokens,
        stats.weight_quant_family,
        stats.weight_quant_bits,
        stats.weight_quant_encoding,
        stats.weight_kernel_family,
        stats.offload_gguf_first_file,
        (unsigned long long)stats.vram_used_bytes,
        (unsigned long long)stats.common_vram_reserved_bytes,
        (unsigned long long)stats.vram_expert_used_bytes,
        (unsigned long long)stats.device_allocated_bytes,
        (unsigned long long)stats.device_allocation_count,
        (unsigned long long)stats.device_free_bytes,
        (unsigned long long)stats.device_total_bytes,
        (unsigned long long)stats.device_fixed_bytes,
        (unsigned long long)stats.device_activation_bytes,
        (unsigned long long)stats.device_expert_bytes,
        (unsigned long long)stats.device_stream_bytes,
        stats.model_lib_loaded,
        stats.model_lib_generated,
        stats.model_lib_compile_succeeded,
        stats.model_lib_path,
        (unsigned long long)stats.common_raw_prefetched_bytes,
        (unsigned long long)stats.common_raw_prefetched_spans,
        (unsigned long long)stats.ram_used_bytes,
        (unsigned long long)stats.db_used_bytes,
        (unsigned long long)stats.hot_expert_count,
        (unsigned long long)stats.vram_expert_count,
        (unsigned long long)stats.ram_expert_count,
        (unsigned long long)stats.db_expert_count
    );
    print_forward_status("initial", engine);

    if (table_path) {
        if (!glm5_pc_engine_load_codec_table(engine, table_path, model_root, scale4_path)) {
            fprintf(stderr, "failed to load codec table: %s\n", table_path);
            glm5_pc_engine_destroy(engine);
            return 3;
        }
        printf("codec_tensors=%llu\n", (unsigned long long)glm5_pc_engine_tensor_count(engine));
        print_forward_status("loaded", engine);

        glm5_tensor_info_t info;
        if (glm5_pc_engine_find_tensor(engine, 3, 0, GLM5_PROJ_DOWN, &info)) {
            printf(
                "L3.E0.down rows=%u cols=%u shard=%u weight_off=%llu weight_len=%llu scale4=%d\n",
                info.rows,
                info.cols,
                info.shard,
                (unsigned long long)info.weight_byte_offset,
                (unsigned long long)info.weight_byte_length,
                info.has_scale4
            );
        } else {
            printf("L3.E0.down not present in this table\n");
        }

        if (prefetch_layer >= 0 && prefetch_expert >= 0) {
            const int current_experts[1] = {prefetch_expert};
            int next_experts[1] = {learn_next_expert >= 0 ? learn_next_expert : prefetch_expert};
            glm5_expert_prefetch_item_t items[32];
            uint32_t item_count = 0;
            if (!glm5_pc_engine_plan_moe_prefetch(
                    engine,
                    prefetch_layer,
                    current_experts,
                    1,
                    next_experts,
                    1,
                    items,
                    32,
                    &item_count) ||
                !glm5_pc_engine_execute_prefetch_plan(engine, items, item_count)) {
                fprintf(stderr, "failed to queue prefetch L%d.E%d\n", prefetch_layer, prefetch_expert);
                glm5_pc_engine_destroy(engine);
                return 4;
            }
            if (learn_next_expert >= 0) {
                glm5_pc_engine_record_expert_transition(
                    engine,
                    prefetch_layer,
                    prefetch_expert,
                    prefetch_layer + 1,
                    learn_next_expert
                );
                if (topology_path) {
                    glm5_pc_engine_save_topology(engine, topology_path);
                }
            }
            glm5_pc_engine_wait_io(engine);
            glm5_io_stats_t io_stats;
            if (glm5_pc_engine_get_io_stats(engine, &io_stats)) {
                printf(
                    "io path=%s queued=%llu done=%llu failed=%llu dropped=%llu hint=%llu disk=%llu pinned=%llu gpu=%llu direct=%llu q=%u/%u/%u topology=%llu pinned_active=%d staging_deficit=%u recommended_staging=%llu\n",
                    glm5_io_path_name(io_stats.active_path),
                    (unsigned long long)io_stats.queued_requests,
                    (unsigned long long)io_stats.completed_requests,
                    (unsigned long long)io_stats.failed_requests,
                    (unsigned long long)io_stats.dropped_requests,
                    (unsigned long long)io_stats.bytes_disk_hint_issued,
                    (unsigned long long)io_stats.bytes_disk_to_ram,
                    (unsigned long long)io_stats.bytes_ram_to_pinned,
                    (unsigned long long)io_stats.bytes_pinned_to_gpu,
                    (unsigned long long)io_stats.bytes_direct_to_gpu,
                    io_stats.disk_queue_depth,
                    io_stats.pinned_queue_depth,
                    io_stats.gpu_queue_depth,
                    (unsigned long long)io_stats.topology_predictions,
                    io_stats.pinned_host_active,
                    io_stats.staging_slot_deficit,
                    (unsigned long long)io_stats.recommended_staging_bytes
                );
            }
        }
    } else if (model_root) {
        if (glm5_pc_engine_load_codec_table(engine, nullptr, model_root, scale4_path)) {
            glm5_pc_engine_get_stats(engine, &stats);
            printf(
                "offload_gguf_header_tensors=%llu codec_tensors=%llu\n",
                (unsigned long long)stats.offload_gguf_tensor_count,
                (unsigned long long)glm5_pc_engine_tensor_count(engine)
            );
            print_forward_status("loaded", engine);
        }
    }

    glm5_pc_engine_destroy(engine);
    return 0;
}

#ifdef _WIN32
static std::string wide_to_utf8_arg(const wchar_t* value) {
    if (!value) {
        return std::string();
    }
    const int count = WideCharToMultiByte(CP_UTF8, 0, value, -1, nullptr, 0, nullptr, nullptr);
    if (count <= 0) {
        return std::string();
    }
    std::string out((size_t)count, '\0');
    if (!WideCharToMultiByte(CP_UTF8, 0, value, -1, &out[0], count, nullptr, nullptr)) {
        return std::string();
    }
    if (!out.empty() && out.back() == '\0') {
        out.pop_back();
    }
    return out;
}

int wmain(int argc, wchar_t** wargv) {
    std::vector<std::string> args;
    std::vector<char*> argv8;
    args.reserve((size_t)argc);
    argv8.reserve((size_t)argc + 1);
    for (int i = 0; i < argc; ++i) {
        args.push_back(wide_to_utf8_arg(wargv[i]));
    }
    for (std::string& arg : args) {
        argv8.push_back(arg.empty() ? const_cast<char*>("") : &arg[0]);
    }
    argv8.push_back(nullptr);
    return probe_main(argc, argv8.data());
}
#else
int main(int argc, char** argv) {
    return probe_main(argc, argv);
}
#endif

