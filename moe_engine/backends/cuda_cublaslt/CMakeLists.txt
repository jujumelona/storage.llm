if(STORAGELLM_ENABLE_CUDA STREQUAL "OFF")
  return()
endif()

if(STORAGELLM_ENABLE_CUBLASLT STREQUAL "OFF")
  return()
endif()

set(STORAGELLM_CUBLASLT_MOE_ADAPTER_IMPLEMENTED ON)
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND AND TARGET CUDA::cublasLt AND TARGET CUDA::cublas AND TARGET CUDA::cudart)
  if(CUDAToolkit_VERSION VERSION_LESS 12.5)
    if(STORAGELLM_ENABLE_CUBLASLT STREQUAL "ON")
      message(FATAL_ERROR "cuBLAS grouped GEMM requires CUDA Toolkit 12.5 or newer.")
    else()
      message(WARNING "CUDA Toolkit ${CUDAToolkit_VERSION} lacks cuBLAS grouped GEMM. CUDA cuBLASLt fast backend disabled.")
    endif()
    return()
  endif()
  if(NOT STORAGELLM_CUBLASLT_MOE_ADAPTER_IMPLEMENTED)
    if(STORAGELLM_ENABLE_CUBLASLT STREQUAL "ON")
      message(FATAL_ERROR "cuBLASLt found, but cublaslt_grouped_moe_adapter.cu is still a stub. CUDA true-fast MoE disabled.")
    else()
      message(WARNING "cuBLASLt found, but cublaslt_grouped_moe_adapter.cu is still a stub. CUDA true-fast MoE disabled.")
    endif()
    return()
  endif()
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  else()
    message(WARNING "CUDA compiler not found. CUDA cuBLASLt fast backend disabled.")
    return()
  endif()
  add_library(storagellm_cuda_cublaslt STATIC
    cublaslt_grouped_moe_adapter.cu)
  target_link_libraries(storagellm_cuda_cublaslt PUBLIC
    storagellm_fast_backend_contract
    CUDA::cublasLt
    CUDA::cublas
    CUDA::cudart)
  target_compile_definitions(storagellm_cuda_cublaslt PRIVATE
    STORAGELLM_HAS_CUBLASLT=1
    STORAGELLM_HAS_CUBLASLT_MOE_ADAPTER=1
    STORAGELLM_MODEL_LIB_ENABLE_CUBLASLT_MOE=1)
else()
  message(STATUS "cuBLASLt not found. CUDA cuBLASLt fast backend disabled.")
endif()
