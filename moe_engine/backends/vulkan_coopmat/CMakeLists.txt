if(STORAGELLM_ENABLE_VULKAN STREQUAL "OFF")
  return()
endif()

set(STORAGELLM_VULKAN_COOPMAT_MOE_ADAPTER_IMPLEMENTED OFF)
find_package(Vulkan QUIET)
if(Vulkan_FOUND)
  if(NOT STORAGELLM_VULKAN_COOPMAT_MOE_ADAPTER_IMPLEMENTED)
    if(STORAGELLM_ENABLE_VULKAN STREQUAL "ON")
      message(FATAL_ERROR "Vulkan found, but vulkan_coopmat_moe_adapter.cpp is still a stub. Vulkan true-fast MoE disabled.")
    else()
      message(WARNING "Vulkan found, but vulkan_coopmat_moe_adapter.cpp is still a stub. Vulkan true-fast MoE disabled.")
    endif()
    return()
  endif()
  add_library(storagellm_vulkan_coopmat STATIC
    vulkan_coopmat_moe_adapter.cpp)
  target_link_libraries(storagellm_vulkan_coopmat PUBLIC
    storagellm_fast_backend_contract
    Vulkan::Vulkan)
  target_compile_definitions(storagellm_vulkan_coopmat PRIVATE
    STORAGELLM_HAS_VULKAN_COOPMAT=1
    STORAGELLM_HAS_VULKAN_COOPMAT_MOE_ADAPTER=1)
else()
  message(STATUS "Vulkan not found. Vulkan cooperative matrix fast backend disabled.")
endif()
