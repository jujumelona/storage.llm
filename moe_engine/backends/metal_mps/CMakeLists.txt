if(STORAGELLM_ENABLE_METAL STREQUAL "OFF")
  return()
endif()

set(STORAGELLM_METAL_MPS_MOE_ADAPTER_IMPLEMENTED OFF)
if(APPLE)
  if(NOT STORAGELLM_METAL_MPS_MOE_ADAPTER_IMPLEMENTED)
    if(STORAGELLM_ENABLE_METAL STREQUAL "ON")
      message(FATAL_ERROR "Metal/MPS is available, but metal_moe_adapter.mm is still a stub. Metal true-fast MoE disabled.")
    else()
      message(WARNING "Metal/MPS is available, but metal_moe_adapter.mm is still a stub. Metal true-fast MoE disabled.")
    endif()
    return()
  endif()
  add_library(storagellm_metal_mps STATIC
    metal_moe_adapter.mm)
  target_link_libraries(storagellm_metal_mps PUBLIC
    storagellm_fast_backend_contract
    "-framework Metal"
    "-framework MetalPerformanceShaders"
    "-framework MetalPerformanceShadersGraph")
  target_compile_definitions(storagellm_metal_mps PRIVATE
    STORAGELLM_HAS_METAL_MPS=1
    STORAGELLM_HAS_METAL_MPS_MOE_ADAPTER=1)
else()
  message(STATUS "Metal/MPS fast backend disabled: not building on Apple platform.")
endif()
