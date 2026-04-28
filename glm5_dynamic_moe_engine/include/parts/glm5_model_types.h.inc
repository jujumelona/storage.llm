typedef struct {
    uint32_t num_hidden_layers;
    uint32_t first_moe_layer;
    uint32_t last_moe_layer;
    uint32_t experts_per_moe_layer;
    uint32_t hidden_size;
    uint32_t expert_intermediate_size;
    uint32_t vocab_size;
    uint32_t projection_count;
} glm5_model_shape_t;

typedef struct {
    glm5_backend_t backend;
    glm5_platform_t platform;
    glm5_inference_phase_t phase;
    uint32_t compute_streams;
    uint32_t copy_streams;
    uint32_t prefetch_streams;
    uint64_t recommended_vram_expert_cache_bytes;
    uint64_t recommended_vram_common_resident_bytes;
    uint64_t recommended_vram_total_cache_bytes;
    uint64_t recommended_pinned_staging_bytes;
    int use_model_shape_constants;
    int use_static_memory_pool;
    int use_expert_prefetch;
    int use_vram_hot_cache;
    int use_pinned_ram_cache;
    int use_storage_streaming;
    int use_async_copy_stream;
    int use_direct_to_gpu_io;
    int use_fused_fp4_dequant_matmul;
    int use_scale4_decode_fusion;
    int use_cuda_graphs;
    int use_metal_command_buffers;
    int use_paged_kv_cache;
    int use_prefill_decode_split;
} glm5_runtime_optimization_plan_t;

