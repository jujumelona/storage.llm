#pragma once
#include <stdint.h>

typedef enum moe_fast_backend_kind {
    moe_FAST_BACKEND_NONE = 0,
    moe_FAST_BACKEND_CUDA_CUTLASS = 1,
    moe_FAST_BACKEND_CUDA_CUBLASLT = 2,
    moe_FAST_BACKEND_ROCM_HIPBLASLT = 3,
    moe_FAST_BACKEND_ROCM_CK = 4,
    moe_FAST_BACKEND_METAL_MPS = 5,
    moe_FAST_BACKEND_INTEL_ONEDNN_SYCL = 6,
    moe_FAST_BACKEND_VULKAN_COOPMAT = 7,
    moe_FAST_BACKEND_OPENCL_CLBLAST = 8,
    moe_FAST_BACKEND_CPU_ONEDNN = 9,
    moe_FAST_BACKEND_TVM_CUDA = 10,
    moe_FAST_BACKEND_TVM_METAL = 11,
    moe_FAST_BACKEND_TVM_VULKAN = 12,
    moe_FAST_BACKEND_TVM_OPENCL = 13,
    moe_FAST_BACKEND_TVM_CPU = 14
} moe_fast_backend_kind_t;

typedef struct moe_grouped_expert_device_task {
    int32_t layer;
    int32_t expert;

    const void* gate_weight;
    const void* up_weight;
    const void* down_weight;

    const void* d_input;
    uint32_t input_stride;

    const uint32_t* d_token_indices;
    const float* d_token_weights;
    uint32_t assignment_offset;
    uint32_t assignment_count;

    void* d_accum;
    uint32_t accum_stride;

    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t activation_mode;
} moe_grouped_expert_device_task_t;


#define STORAGELLM_FAST_BACKEND_DISPATCH_ABI_V2 2u

typedef enum moe_fast_backend_context_kind {
    moe_FAST_BACKEND_CONTEXT_NONE = 0,
    moe_FAST_BACKEND_CONTEXT_CUDA = 1,
    moe_FAST_BACKEND_CONTEXT_HIP = 2,
    moe_FAST_BACKEND_CONTEXT_METAL = 3,
    moe_FAST_BACKEND_CONTEXT_VULKAN = 4,
    moe_FAST_BACKEND_CONTEXT_OPENCL = 5,
    moe_FAST_BACKEND_CONTEXT_SYCL = 6,
    moe_FAST_BACKEND_CONTEXT_CPU = 7
} moe_fast_backend_context_kind_t;

typedef struct moe_fast_backend_cuda_context {
    void* cuda_stream;
    void* cublas_handle;
    void* cublaslt_handle;
    void* workspace;
    uint64_t workspace_bytes;
    uint32_t sm_version;
    uint32_t flags;
} moe_fast_backend_cuda_context_t;

typedef struct moe_fast_backend_hip_context {
    void* hip_stream;
    void* hipblas_handle;
    void* hipblaslt_handle;
    void* workspace;
    uint64_t workspace_bytes;
    uint32_t gfx_version;
    uint32_t flags;
} moe_fast_backend_hip_context_t;

typedef struct moe_fast_backend_metal_context {
    void* command_queue;
    void* command_buffer;
    void* device;
    void* scratch_heap;
    uint64_t scratch_bytes;
    uint32_t family;
    uint32_t flags;
} moe_fast_backend_metal_context_t;

typedef struct moe_fast_backend_vulkan_context {
    void* instance;
    void* physical_device;
    void* device;
    void* queue;
    void* command_buffer;
    void* command_pool;
    void* descriptor_pool;
    void* pipeline_cache;
    uint32_t queue_family_index;
    uint32_t subgroup_size;
    uint32_t flags;
} moe_fast_backend_vulkan_context_t;

typedef struct moe_fast_backend_opencl_context {
    void* context;
    void* command_queue;
    void* device_id;
    void* program_cache;
    void* scratch_buffer;
    uint64_t scratch_bytes;
    uint32_t flags;
} moe_fast_backend_opencl_context_t;

typedef struct moe_fast_backend_sycl_context {
    void* queue;
    void* dnnl_engine;
    void* dnnl_stream;
    void* scratch_allocator;
    uint64_t scratch_bytes;
    uint32_t flags;
} moe_fast_backend_sycl_context_t;

typedef struct moe_fast_backend_cpu_context {
    void* thread_pool;
    void* numa_policy;
    uint32_t thread_count;
    uint32_t numa_node;
    uint32_t isa_flags;
    uint32_t flags;
} moe_fast_backend_cpu_context_t;

typedef struct moe_fast_backend_context {
    uint32_t abi_version;
    uint32_t context_kind;
    uint64_t global_flags;
    union {
        moe_fast_backend_cuda_context_t cuda;
        moe_fast_backend_hip_context_t hip;
        moe_fast_backend_metal_context_t metal;
        moe_fast_backend_vulkan_context_t vulkan;
        moe_fast_backend_opencl_context_t opencl;
        moe_fast_backend_sycl_context_t sycl;
        moe_fast_backend_cpu_context_t cpu;
    } u;
} moe_fast_backend_context_t;

typedef struct moe_fast_backend_dispatch_request {
    uint32_t abi_version;
    int32_t backend;
    int32_t fast_backend;
    const moe_grouped_expert_device_task_t* tasks;
    uint32_t task_count;
    const moe_fast_backend_context_t* context;
    void* legacy_stream_or_queue;
    uint64_t flags;
} moe_fast_backend_dispatch_request_t;

#ifdef __cplusplus
extern "C" {
#endif

uint32_t moe_model_lib_fast_backend_mask(void);

int moe_model_lib_run_grouped_expert_triplets_indexed_device_f32(
    uint32_t abi_version,
    int32_t backend,
    const moe_grouped_expert_device_task_t* tasks,
    uint32_t task_count,
    void* stream_or_queue);

int storagellm_fast_grouped_moe_indexed_device_f32(
    int32_t backend,
    const moe_grouped_expert_device_task_t* tasks,
    uint32_t task_count,
    void* stream_or_queue);

int storagellm_cutlass_grouped_moe_indexed_device_f32(
    int32_t backend,
    const moe_grouped_expert_device_task_t* tasks,
    uint32_t task_count,
    void* stream_or_queue);

int storagellm_cublaslt_grouped_moe_indexed_device_f32(
    int32_t backend,
    const moe_grouped_expert_device_task_t* tasks,
    uint32_t task_count,
    void* stream_or_queue);

int storagellm_hipblaslt_grouped_moe_indexed_device_f32(
    int32_t backend,
    const moe_grouped_expert_device_task_t* tasks,
    uint32_t task_count,
    void* stream_or_queue);

int storagellm_ck_grouped_moe_indexed_device_f32(
    int32_t backend,
    const moe_grouped_expert_device_task_t* tasks,
    uint32_t task_count,
    void* stream_or_queue);

int storagellm_metal_mps_grouped_moe_indexed_device_f32(
    int32_t backend,
    const moe_grouped_expert_device_task_t* tasks,
    uint32_t task_count,
    void* stream_or_queue);

int storagellm_onednn_sycl_grouped_moe_indexed_device_f32(
    int32_t backend,
    const moe_grouped_expert_device_task_t* tasks,
    uint32_t task_count,
    void* stream_or_queue);

int storagellm_vulkan_coopmat_grouped_moe_indexed_device_f32(
    int32_t backend,
    const moe_grouped_expert_device_task_t* tasks,
    uint32_t task_count,
    void* stream_or_queue);

int storagellm_clblast_grouped_moe_indexed_device_f32(
    int32_t backend,
    const moe_grouped_expert_device_task_t* tasks,
    uint32_t task_count,
    void* stream_or_queue);

int storagellm_onednn_cpu_grouped_moe_indexed_device_f32(
    int32_t backend,
    const moe_grouped_expert_device_task_t* tasks,
    uint32_t task_count,
    void* stream_or_queue);

int storagellm_tvm_codegen_available(int32_t backend);

int storagellm_tvm_codegen_grouped_moe_indexed_device_f32(
    int32_t backend,
    const moe_grouped_expert_device_task_t* tasks,
    uint32_t task_count,
    void* stream_or_queue);


int moe_model_lib_run_grouped_expert_triplets_indexed_device_f32_v2(
    const moe_fast_backend_dispatch_request_t* request);

int storagellm_cutlass_grouped_moe_indexed_device_f32_v2(
    const moe_fast_backend_dispatch_request_t* request);
int storagellm_cublaslt_grouped_moe_indexed_device_f32_v2(
    const moe_fast_backend_dispatch_request_t* request);
int storagellm_hipblaslt_grouped_moe_indexed_device_f32_v2(
    const moe_fast_backend_dispatch_request_t* request);
int storagellm_ck_grouped_moe_indexed_device_f32_v2(
    const moe_fast_backend_dispatch_request_t* request);
int storagellm_metal_mps_grouped_moe_indexed_device_f32_v2(
    const moe_fast_backend_dispatch_request_t* request);
int storagellm_onednn_sycl_grouped_moe_indexed_device_f32_v2(
    const moe_fast_backend_dispatch_request_t* request);
int storagellm_vulkan_coopmat_grouped_moe_indexed_device_f32_v2(
    const moe_fast_backend_dispatch_request_t* request);
int storagellm_clblast_grouped_moe_indexed_device_f32_v2(
    const moe_fast_backend_dispatch_request_t* request);
int storagellm_onednn_cpu_grouped_moe_indexed_device_f32_v2(
    const moe_fast_backend_dispatch_request_t* request);
int storagellm_tvm_codegen_grouped_moe_indexed_device_f32_v2(
    const moe_fast_backend_dispatch_request_t* request);

#ifdef __cplusplus
}
#endif
