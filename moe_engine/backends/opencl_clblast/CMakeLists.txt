if(STORAGELLM_ENABLE_OPENCL STREQUAL "OFF")
  return()
endif()

if(STORAGELLM_ENABLE_CLBLAST STREQUAL "OFF")
  return()
endif()

set(STORAGELLM_CLBLAST_MOE_ADAPTER_IMPLEMENTED ON)
find_package(OpenCL QUIET)
find_package(CLBlast QUIET)
if(OpenCL_FOUND AND CLBlast_FOUND)
  if(NOT STORAGELLM_CLBLAST_MOE_ADAPTER_IMPLEMENTED)
    if(STORAGELLM_ENABLE_CLBLAST STREQUAL "ON")
      message(FATAL_ERROR "CLBlast found, but the OpenCL fused MoE adapter is marked disabled. OpenCL true-fast MoE disabled.")
    else()
      message(WARNING "CLBlast found, but the OpenCL fused MoE adapter is marked disabled. OpenCL true-fast MoE disabled.")
    endif()
    return()
  endif()
  add_library(storagellm_opencl_clblast STATIC
    clblast_moe_adapter.cpp)
  target_link_libraries(storagellm_opencl_clblast PUBLIC
    storagellm_fast_backend_contract
    OpenCL::OpenCL
    clblast)
  target_compile_definitions(storagellm_opencl_clblast PRIVATE
    STORAGELLM_HAS_CLBLAST=1
    STORAGELLM_HAS_CLBLAST_MOE_ADAPTER=1)
else()
  message(STATUS "OpenCL/CLBlast not found. OpenCL CLBlast fast backend disabled.")
endif()
