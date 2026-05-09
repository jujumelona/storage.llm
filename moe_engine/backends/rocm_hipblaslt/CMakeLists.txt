if(STORAGELLM_ENABLE_ROCM STREQUAL "OFF")
  return()
endif()

if(STORAGELLM_ENABLE_HIPBLASLT STREQUAL "OFF")
  return()
endif()

set(STORAGELLM_HIPBLASLT_MOE_ADAPTER_IMPLEMENTED OFF)
find_package(hipblaslt QUIET)
if(hipblaslt_FOUND)
  if(NOT STORAGELLM_HIPBLASLT_MOE_ADAPTER_IMPLEMENTED)
    if(STORAGELLM_ENABLE_HIPBLASLT STREQUAL "ON")
      message(FATAL_ERROR "hipBLASLt found, but hipblaslt_grouped_moe_adapter.cpp is still a stub. ROCm true-fast MoE disabled.")
    else()
      message(WARNING "hipBLASLt found, but hipblaslt_grouped_moe_adapter.cpp is still a stub. ROCm true-fast MoE disabled.")
    endif()
    return()
  endif()
  add_library(storagellm_rocm_hipblaslt STATIC
    hipblaslt_grouped_moe_adapter.cpp)
  target_link_libraries(storagellm_rocm_hipblaslt PUBLIC
    storagellm_fast_backend_contract
    hipblaslt)
  target_compile_definitions(storagellm_rocm_hipblaslt PRIVATE
    STORAGELLM_HAS_HIPBLASLT=1
    STORAGELLM_HAS_HIPBLASLT_MOE_ADAPTER=1)
else()
  message(STATUS "hipBLASLt not found. ROCm hipBLASLt fast backend disabled.")
endif()
