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

#ifdef __cplusplus
}
#endif
