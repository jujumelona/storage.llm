if(STORAGELLM_ENABLE_ROCM STREQUAL "OFF")
  return()
endif()

if(STORAGELLM_ENABLE_CK STREQUAL "OFF")
  return()
endif()

set(STORAGELLM_CK_MOE_ADAPTER_IMPLEMENTED OFF)
find_package(composable_kernel QUIET)
if(composable_kernel_FOUND)
  if(NOT STORAGELLM_CK_MOE_ADAPTER_IMPLEMENTED)
    if(STORAGELLM_ENABLE_CK STREQUAL "ON")
      message(FATAL_ERROR "Composable Kernel found, but ck_grouped_moe_adapter.cpp is still a stub. ROCm true-fast MoE disabled.")
    else()
      message(WARNING "Composable Kernel found, but ck_grouped_moe_adapter.cpp is still a stub. ROCm true-fast MoE disabled.")
    endif()
    return()
  endif()
  add_library(storagellm_rocm_ck STATIC
    ck_grouped_moe_adapter.cpp)
  target_link_libraries(storagellm_rocm_ck PUBLIC
    storagellm_fast_backend_contract
    composable_kernel::device_operations)
  target_compile_definitions(storagellm_rocm_ck PRIVATE
    STORAGELLM_HAS_CK=1
    STORAGELLM_HAS_CK_MOE_ADAPTER=1)
else()
  message(STATUS "Composable Kernel not found. ROCm CK fast backend disabled.")
endif()
