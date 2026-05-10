using moe_model_lib_fast_backend_mask_fn = uint32_t (*)();

static int moe_fast_backend_selected_name_is(const char* expected) {
    if (!expected || !expected[0]) {
        return 0;
    }
    const char* selected = std::getenv("STORAGELLM_SELECTED_FAST_BACKEND");
    if (!selected || !selected[0] ||
        std::strcmp(selected, "auto") == 0 ||
        std::strcmp(selected, "AUTO") == 0 ||
        std::strcmp(selected, "max") == 0 ||
        std::strcmp(selected, "MAX") == 0) {
        return 0;
    }
    if (std::strcmp(selected, expected) == 0) {
        return 1;
    }
    // Short aliases for user/runtime autotune output.
    if ((std::strcmp(expected, "cuda_cublaslt") == 0 && std::strcmp(selected, "cublaslt") == 0) ||
        (std::strcmp(expected, "cuda_cutlass") == 0 && std::strcmp(selected, "cutlass") == 0) ||
        (std::strcmp(expected, "cpu_native_f32") == 0 &&
            (std::strcmp(selected, "onednn") == 0 || std::strcmp(selected, "cpu_onednn") == 0 || std::strcmp(selected, "cpu") == 0)) ||
        (std::strncmp(expected, "tvm_", 4) == 0 && std::strcmp(selected, "tvm") == 0)) {
        return 1;
    }
    return 0;
}

static int moe_fast_backend_policy_is(const char* expected) {
    const char* policy = std::getenv("STORAGELLM_FAST_BACKEND_POLICY");
    return policy && expected && std::strcmp(policy, expected) == 0;
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

static uint32_t moe_fast_backend_mask_from_engine_adapters() {
    uint32_t mask = 0u;
#if defined(STORAGELLM_HAS_CUTLASS_MOE_ADAPTER)
    mask |= (1u << moe_FAST_BACKEND_CUDA_CUTLASS);
#endif
#if defined(STORAGELLM_HAS_CUBLASLT_MOE_ADAPTER)
    mask |= (1u << moe_FAST_BACKEND_CUDA_CUBLASLT);
#endif
#if defined(STORAGELLM_HAS_HIPBLASLT_MOE_ADAPTER)
    mask |= (1u << moe_FAST_BACKEND_ROCM_HIPBLASLT);
#endif
#if defined(STORAGELLM_HAS_CK_MOE_ADAPTER)
    mask |= (1u << moe_FAST_BACKEND_ROCM_CK);
#endif
#if defined(STORAGELLM_HAS_METAL_MPS_MOE_ADAPTER)
    mask |= (1u << moe_FAST_BACKEND_METAL_MPS);
#endif
#if defined(STORAGELLM_HAS_ONEDNN_SYCL_MOE_ADAPTER)
    mask |= (1u << moe_FAST_BACKEND_INTEL_ONEDNN_SYCL);
#endif
#if defined(STORAGELLM_HAS_VULKAN_COOPMAT_MOE_ADAPTER)
    mask |= (1u << moe_FAST_BACKEND_VULKAN_COOPMAT);
#endif
#if defined(STORAGELLM_HAS_CLBLAST_MOE_ADAPTER)
    mask |= (1u << moe_FAST_BACKEND_OPENCL_CLBLAST);
#endif
#if defined(STORAGELLM_HAS_ONEDNN_CPU_MOE_ADAPTER)
    mask |= (1u << moe_FAST_BACKEND_CPU_ONEDNN);
#endif
#if defined(STORAGELLM_HAS_TVM_CODEGEN_ADAPTER)
    // Current in-tree TVM codegen emits a CPU C ABI bridge.  Device TVM
    // adapters must not be advertised here until they compile a real device
    // wrapper; otherwise backend selection can pick an unavailable fast path.
    mask |= (1u << moe_FAST_BACKEND_TVM_CPU);
#endif
    return mask;
}

static int moe_fast_backend_mask_supports_backend(moe_backend_t backend, uint32_t mask) {
    switch (backend) {
        case moe_BACKEND_CUDA:
            return (mask & ((1u << moe_FAST_BACKEND_CUDA_CUTLASS) |
                            (1u << moe_FAST_BACKEND_CUDA_CUBLASLT))) != 0u;
        case moe_BACKEND_HIP:
            return (mask & ((1u << moe_FAST_BACKEND_ROCM_HIPBLASLT) |
                            (1u << moe_FAST_BACKEND_ROCM_CK))) != 0u;
        case moe_BACKEND_METAL:
            return (mask & (1u << moe_FAST_BACKEND_METAL_MPS)) != 0u;
        case moe_BACKEND_LEVEL_ZERO:
        case moe_BACKEND_SYCL:
            return (mask & (1u << moe_FAST_BACKEND_INTEL_ONEDNN_SYCL)) != 0u;
        case moe_BACKEND_VULKAN:
            return (mask & (1u << moe_FAST_BACKEND_VULKAN_COOPMAT)) != 0u;
        case moe_BACKEND_OPENCL:
            return (mask & (1u << moe_FAST_BACKEND_OPENCL_CLBLAST)) != 0u;
        case moe_BACKEND_CPU:
            return (mask & ((1u << moe_FAST_BACKEND_TVM_CPU) |
                            (1u << moe_FAST_BACKEND_CPU_ONEDNN))) != 0u;
        default:
            return 0;
    }
}

static int moe_fast_backend_kind_allowed_runtime(moe_fast_backend_kind_t kind) {
    if (kind == moe_FAST_BACKEND_NONE) return 0;
    if (moe_perf_backend_kind_allowed_for_strict(kind)) return 1;
    // In non-strict mode, prefer real kernels but allow explicit experimental
    // adapters only when the operator opts in.  Returning false here makes
    // backend selection fail closed instead of selecting a stub that just
    // returns 0 and then falling back silently.
    return 0;
}

static moe_fast_backend_kind_t moe_pick_first_runtime_allowed(
    uint32_t mask,
    const moe_fast_backend_kind_t* order,
    uint32_t count
) {
    if (!order) return moe_FAST_BACKEND_NONE;
    for (uint32_t i = 0; i < count; ++i) {
        if (moe_perf_fast_backend_supported(mask, order[i]) &&
            moe_fast_backend_kind_allowed_runtime(order[i])) {
            return order[i];
        }
    }
    return moe_FAST_BACKEND_NONE;
}

static int moe_verified_backend_required(const moe_pc_engine* e) {
    return e && e->io_config.internal_verified_backend_only != 0;
}

static moe_fast_backend_kind_t choose_fast_backend(const moe_pc_engine* e) {
    if (!e) {
        return moe_FAST_BACKEND_NONE;
    }
    // Fast-backend dispatch is engine-owned.  Model-lib masks are retained only
    // as a compatibility signal; compiled engine adapters are the source of
    // truth so generated model libraries never need to link backend symbols.
    const uint32_t mask = moe_fast_backend_mask_from_engine_adapters() |
                          moe_fast_backend_mask_from_model_lib(e);
    const moe_fast_backend_kind_t selected =
        moe_fast_backend_kind_from_selected_env(e->backend_caps.backend, mask);
    if (selected != moe_FAST_BACKEND_NONE) {
        return moe_fast_backend_kind_allowed_runtime(selected) ? selected : moe_FAST_BACKEND_NONE;
    }

    // Hardware/backend-aware default order.  This is deliberately centralized:
    // model-lib support mask decides what is actually available, while this
    // order decides which fast kernel wins when several are bundled.  Users or
    // an autotuner can override with STORAGELLM_SELECTED_FAST_BACKEND or with
    // a comma-separated STORAGELLM_FAST_BACKEND_ORDER.
    switch (e->backend_caps.backend) {
        case moe_BACKEND_CUDA: {
            // NVIDIA: cuBLASLt is the safest first pick for broad GEMM coverage;
            // CUTLASS can be faster for tuned grouped-MoE kernels and may be put
            // first via STORAGELLM_FAST_BACKEND_POLICY=cutlass_first or env order.
            const moe_fast_backend_kind_t order_cutlass[] = {
                moe_FAST_BACKEND_CUDA_CUTLASS,
                moe_FAST_BACKEND_CUDA_CUBLASLT
            };
            const moe_fast_backend_kind_t order_default[] = {
                moe_FAST_BACKEND_CUDA_CUBLASLT,
                moe_FAST_BACKEND_CUDA_CUTLASS
            };
            const bool cutlass_first = moe_fast_backend_policy_is("cutlass_first") ||
                                       moe_perf_env_truthy("STORAGELLM_CUTLASS_FIRST");
            return moe_pick_first_runtime_allowed(
                mask,
                cutlass_first ? order_cutlass : order_default,
                2u);
        }
        case moe_BACKEND_HIP: {
            const moe_fast_backend_kind_t order[] = {
                moe_FAST_BACKEND_ROCM_HIPBLASLT,
                moe_FAST_BACKEND_ROCM_CK
            };
            return moe_pick_first_runtime_allowed(mask, order, 2u);
        }
        case moe_BACKEND_METAL: {
            const moe_fast_backend_kind_t order[] = {
                moe_FAST_BACKEND_METAL_MPS
            };
            return moe_pick_first_runtime_allowed(mask, order, 1u);
        }
        case moe_BACKEND_LEVEL_ZERO:
        case moe_BACKEND_SYCL: {
            const moe_fast_backend_kind_t order[] = {
                moe_FAST_BACKEND_INTEL_ONEDNN_SYCL
            };
            return moe_pick_first_runtime_allowed(mask, order, 1u);
        }
        case moe_BACKEND_VULKAN: {
            const moe_fast_backend_kind_t order[] = {
                moe_FAST_BACKEND_VULKAN_COOPMAT
            };
            return moe_pick_first_runtime_allowed(mask, order, 1u);
        }
        case moe_BACKEND_OPENCL: {
            const moe_fast_backend_kind_t order[] = {
                moe_FAST_BACKEND_OPENCL_CLBLAST
            };
            return moe_pick_first_runtime_allowed(mask, order, 1u);
        }
        case moe_BACKEND_CPU: {
            const moe_fast_backend_kind_t order_tvm[] = {
                moe_FAST_BACKEND_TVM_CPU,
                moe_FAST_BACKEND_CPU_ONEDNN
            };
            const moe_fast_backend_kind_t order_default[] = {
                moe_FAST_BACKEND_CPU_ONEDNN,
                moe_FAST_BACKEND_TVM_CPU
            };
            const bool tvm_first = moe_fast_backend_policy_is("tvm_first") ||
                                   moe_perf_env_truthy("STORAGELLM_TVM_FIRST");
            return moe_pick_first_runtime_allowed(
                mask,
                tvm_first ? order_tvm : order_default,
                2u);
        }
        default:
            break;
    }
    return moe_FAST_BACKEND_NONE;
}

static int moe_fast_backend_available(const moe_pc_engine* e) {
    return choose_fast_backend(e) != moe_FAST_BACKEND_NONE;
}
