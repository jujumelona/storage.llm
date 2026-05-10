if(STORAGELLM_ENABLE_ONEDNN STREQUAL "OFF")
  return()
endif()

set(STORAGELLM_ONEDNN_SYCL_MOE_ADAPTER_IMPLEMENTED OFF)
find_package(dnnl QUIET)
if(dnnl_FOUND)
  if(NOT STORAGELLM_ONEDNN_SYCL_MOE_ADAPTER_IMPLEMENTED)
    if(STORAGELLM_ENABLE_ONEDNN STREQUAL "ON")
      message(FATAL_ERROR "oneDNN found, but onednn_sycl_moe_adapter.cpp is still a stub. Intel GPU true-fast MoE disabled.")
    else()
      message(WARNING "oneDNN found, but onednn_sycl_moe_adapter.cpp is still a stub. Intel GPU true-fast MoE disabled.")
    endif()
    return()
  endif()
  add_library(storagellm_intel_onednn_sycl STATIC
    onednn_sycl_moe_adapter.cpp)
  target_link_libraries(storagellm_intel_onednn_sycl PUBLIC
    storagellm_fast_backend_contract
    dnnl)
  target_compile_definitions(storagellm_intel_onednn_sycl PRIVATE
    STORAGELLM_HAS_ONEDNN_SYCL=1
    STORAGELLM_HAS_ONEDNN_SYCL_MOE_ADAPTER=1)
else()
  message(STATUS "oneDNN SYCL not found. Intel GPU oneDNN SYCL fast backend disabled.")
endif()
