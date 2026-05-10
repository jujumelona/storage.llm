add_library(storagellm_fast_backend_contract INTERFACE)
target_include_directories(storagellm_fast_backend_contract INTERFACE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/moe_engine/include)

add_subdirectory(cuda_cutlass)
add_subdirectory(cuda_cublaslt)
add_subdirectory(rocm_hipblaslt)
add_subdirectory(rocm_ck)
add_subdirectory(metal_mps)
add_subdirectory(intel_onednn_sycl)
add_subdirectory(vulkan_coopmat)
add_subdirectory(opencl_clblast)
add_subdirectory(cpu_onednn)
add_subdirectory(tvm_codegen)
