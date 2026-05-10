using moe_model_lib_fast_backend_mask_fn = uint32_t (*)();

static int moe_fast_backend_selected_name_is(const char* expected) {
    if (!expected || !expected[0]) {
        return 0;
    }
    const char* selected = std::getenv("STORAGELLM_SELECTED_FAST_BACKEND");
    return selected && std::strcmp(selected, expected) == 0;
}

static moe_fast_backend_kind_t moe_fast_backend_kind_from_selected_env(
    moe_backend_t backend,
    uint32_t mask
) {
    if (!mask) {
        return moe_FAST_BACKEND_NONE;
    }
    switch (backend) {
        case moe_BACKEND_CUDA:
            if (moe_fast_backend_selected_name_is("cuda_cutlass") &&
                (mask & (1u << moe_FAST_BACKEND_CUDA_CUTLASS))) {
                return moe_FAST_BACKEND_CUDA_CUTLASS;
            }
            if (moe_fast_backend_selected_name_is("cuda_cublaslt") &&
                (mask & (1u << moe_FAST_BACKEND_CUDA_CUBLASLT))) {
                return moe_FAST_BACKEND_CUDA_CUBLASLT;
            }
            if (moe_fast_backend_selected_name_is("tvm_cuda") &&
                (mask & (1u << moe_FAST_BACKEND_TVM_CUDA))) {
                return moe_FAST_BACKEND_TVM_CUDA;
            }
            break;
        case moe_BACKEND_HIP:
            if (moe_fast_backend_selected_name_is("rocm_hipblaslt") &&
                (mask & (1u << moe_FAST_BACKEND_ROCM_HIPBLASLT))) {
                return moe_FAST_BACKEND_ROCM_HIPBLASLT;
            }
            if (moe_fast_backend_selected_name_is("rocm_ck") &&
                (mask & (1u << moe_FAST_BACKEND_ROCM_CK))) {
                return moe_FAST_BACKEND_ROCM_CK;
            }
            if (moe_fast_backend_selected_name_is("tvm_rocm") &&
                (mask & (1u << moe_FAST_BACKEND_TVM_OPENCL))) {
                return moe_FAST_BACKEND_TVM_OPENCL;
            }
            break;
        case moe_BACKEND_METAL:
            if (moe_fast_backend_selected_name_is("metal_mps") &&
                (mask & (1u << moe_FAST_BACKEND_METAL_MPS))) {
                return moe_FAST_BACKEND_METAL_MPS;
            }
            if (moe_fast_backend_selected_name_is("tvm_metal") &&
                (mask & (1u << moe_FAST_BACKEND_TVM_METAL))) {
                return moe_FAST_BACKEND_TVM_METAL;
            }
            break;
        case moe_BACKEND_VULKAN:
            if (moe_fast_backend_selected_name_is("vulkan_coopmat") &&
                (mask & (1u << moe_FAST_BACKEND_VULKAN_COOPMAT))) {
                return moe_FAST_BACKEND_VULKAN_COOPMAT;
            }
            if (moe_fast_backend_selected_name_is("tvm_vulkan") &&
                (mask & (1u << moe_FAST_BACKEND_TVM_VULKAN))) {
                return moe_FAST_BACKEND_TVM_VULKAN;
            }
            break;
        case moe_BACKEND_OPENCL:
            if (moe_fast_backend_selected_name_is("opencl_clblast") &&
                (mask & (1u << moe_FAST_BACKEND_OPENCL_CLBLAST))) {
                return moe_FAST_BACKEND_OPENCL_CLBLAST;
            }
            if (moe_fast_backend_selected_name_is("tvm_opencl") &&
                (mask & (1u << moe_FAST_BACKEND_TVM_OPENCL))) {
                return moe_FAST_BACKEND_TVM_OPENCL;
            }
            break;
        case moe_BACKEND_CPU:
            if (moe_fast_backend_selected_name_is("cpu_native_f32") &&
                (mask & (1u << moe_FAST_BACKEND_CPU_ONEDNN))) {
                return moe_FAST_BACKEND_CPU_ONEDNN;
            }
            if (moe_fast_backend_selected_name_is("tvm_cpu") &&
                (mask & (1u << moe_FAST_BACKEND_TVM_CPU))) {
                return moe_FAST_BACKEND_TVM_CPU;
            }
            break;
        default:
            break;
    }
    return moe_FAST_BACKEND_NONE;
}


static uint32_t moe_fast_backend_mask_from_model_lib(const moe_pc_engine* e) {
    if (!e || !e->model_lib_handle || !e->model_lib_fast_backend_mask) {
        return 0u;
    }
    auto fn = reinterpret_cast<moe_model_lib_fast_backend_mask_fn>(
        e->model_lib_fast_backend_mask);
    return fn ? fn() : 0u;
}

static int moe_fast_backend_mask_supports_backend(moe_backend_t backend, uint32_t mask) {
    switch (backend) {
        case moe_BACKEND_CUDA:
            return (mask & ((1u << moe_FAST_BACKEND_CUDA_CUTLASS) |
                            (1u << moe_FAST_BACKEND_TVM_CUDA) |
                            (1u << moe_FAST_BACKEND_CUDA_CUBLASLT))) != 0u;
        case moe_BACKEND_HIP:
            return (mask & ((1u << moe_FAST_BACKEND_ROCM_HIPBLASLT) |
                            (1u << moe_FAST_BACKEND_ROCM_CK))) != 0u;
        case moe_BACKEND_METAL:
            return (mask & ((1u << moe_FAST_BACKEND_TVM_METAL) |
                            (1u << moe_FAST_BACKEND_METAL_MPS))) != 0u;
        case moe_BACKEND_LEVEL_ZERO:
        case moe_BACKEND_SYCL:
            return (mask & (1u << moe_FAST_BACKEND_INTEL_ONEDNN_SYCL)) != 0u;
        case moe_BACKEND_VULKAN:
            return (mask & ((1u << moe_FAST_BACKEND_TVM_VULKAN) |
                            (1u << moe_FAST_BACKEND_VULKAN_COOPMAT))) != 0u;
        case moe_BACKEND_OPENCL:
            return (mask & ((1u << moe_FAST_BACKEND_TVM_OPENCL) |
                            (1u << moe_FAST_BACKEND_OPENCL_CLBLAST))) != 0u;
        case moe_BACKEND_CPU:
            return (mask & ((1u << moe_FAST_BACKEND_TVM_CPU) |
                            (1u << moe_FAST_BACKEND_CPU_ONEDNN))) != 0u;
        default:
            return 0;
    }
}

static int moe_verified_backend_required(const moe_pc_engine* e) {
    return e && e->io_config.internal_verified_backend_only != 0;
}

static moe_fast_backend_kind_t choose_fast_backend(const moe_pc_engine* e) {
    if (!e) {
        return moe_FAST_BACKEND_NONE;
    }
    const uint32_t mask = moe_fast_backend_mask_from_model_lib(e);
    const moe_fast_backend_kind_t selected =
        moe_fast_backend_kind_from_selected_env(e->backend_caps.backend, mask);
    if (selected != moe_FAST_BACKEND_NONE) {
        return selected;
    }
    switch (e->backend_caps.backend) {
        case moe_BACKEND_CUDA:
            if (mask & (1u << moe_FAST_BACKEND_CUDA_CUTLASS)) {
                return moe_FAST_BACKEND_CUDA_CUTLASS;
            }
            if (mask & (1u << moe_FAST_BACKEND_TVM_CUDA)) {
                return moe_FAST_BACKEND_TVM_CUDA;
            }
            if (mask & (1u << moe_FAST_BACKEND_CUDA_CUBLASLT)) {
                return moe_FAST_BACKEND_CUDA_CUBLASLT;
            }
            break;
        case moe_BACKEND_HIP:
            if (mask & (1u << moe_FAST_BACKEND_ROCM_HIPBLASLT)) {
                return moe_FAST_BACKEND_ROCM_HIPBLASLT;
            }
            if (mask & (1u << moe_FAST_BACKEND_ROCM_CK)) {
                return moe_FAST_BACKEND_ROCM_CK;
            }
            break;
        case moe_BACKEND_METAL:
            if (mask & (1u << moe_FAST_BACKEND_TVM_METAL)) {
                return moe_FAST_BACKEND_TVM_METAL;
            }
            if (mask & (1u << moe_FAST_BACKEND_METAL_MPS)) {
                return moe_FAST_BACKEND_METAL_MPS;
            }
            break;
        case moe_BACKEND_LEVEL_ZERO:
        case moe_BACKEND_SYCL:
            if (mask & (1u << moe_FAST_BACKEND_INTEL_ONEDNN_SYCL)) {
                return moe_FAST_BACKEND_INTEL_ONEDNN_SYCL;
            }
            break;
        case moe_BACKEND_VULKAN:
            if (mask & (1u << moe_FAST_BACKEND_TVM_VULKAN)) {
                return moe_FAST_BACKEND_TVM_VULKAN;
            }
            if (mask & (1u << moe_FAST_BACKEND_VULKAN_COOPMAT)) {
                return moe_FAST_BACKEND_VULKAN_COOPMAT;
            }
            break;
        case moe_BACKEND_OPENCL:
            if (mask & (1u << moe_FAST_BACKEND_TVM_OPENCL)) {
                return moe_FAST_BACKEND_TVM_OPENCL;
            }
            if (mask & (1u << moe_FAST_BACKEND_OPENCL_CLBLAST)) {
                return moe_FAST_BACKEND_OPENCL_CLBLAST;
            }
            break;
        case moe_BACKEND_CPU:
            if (mask & (1u << moe_FAST_BACKEND_TVM_CPU)) {
                return moe_FAST_BACKEND_TVM_CPU;
            }
            if (mask & (1u << moe_FAST_BACKEND_CPU_ONEDNN)) {
                return moe_FAST_BACKEND_CPU_ONEDNN;
            }
            break;
        default:
            break;
    }
    return moe_FAST_BACKEND_NONE;
}

static int moe_fast_backend_available(const moe_pc_engine* e) {
    return choose_fast_backend(e) != moe_FAST_BACKEND_NONE;
}
