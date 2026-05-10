if(STORAGELLM_ENABLE_VULKAN STREQUAL "OFF")
  return()
endif()

find_package(Vulkan QUIET)
find_package(shaderc QUIET)

set(STORAGELLM_SHADERC_TARGET "")
if(TARGET shaderc::shaderc)
  set(STORAGELLM_SHADERC_TARGET shaderc::shaderc)
elseif(TARGET shaderc_shared)
  set(STORAGELLM_SHADERC_TARGET shaderc_shared)
elseif(TARGET shaderc_combined)
  set(STORAGELLM_SHADERC_TARGET shaderc_combined)
endif()

if(Vulkan_FOUND AND STORAGELLM_SHADERC_TARGET)
  add_library(storagellm_vulkan_coopmat STATIC
    vulkan_coopmat_moe_adapter.cpp)
  target_link_libraries(storagellm_vulkan_coopmat PUBLIC
    storagellm_fast_backend_contract
    Vulkan::Vulkan
    ${STORAGELLM_SHADERC_TARGET})
  target_compile_definitions(storagellm_vulkan_coopmat PRIVATE
    STORAGELLM_HAS_VULKAN_COOPMAT=1
    STORAGELLM_HAS_VULKAN_COOPMAT_MOE_ADAPTER=1)
  message(STATUS "Vulkan fused MoE adapter enabled (shaderc runtime SPIR-V compile path).")
else()
  if(STORAGELLM_ENABLE_VULKAN STREQUAL "ON")
    message(FATAL_ERROR "Vulkan fused MoE adapter requested, but Vulkan or shaderc was not found.")
  endif()
  message(STATUS "Vulkan fused MoE adapter disabled: Vulkan=${Vulkan_FOUND}, shaderc_target=${STORAGELLM_SHADERC_TARGET}")
endif()
