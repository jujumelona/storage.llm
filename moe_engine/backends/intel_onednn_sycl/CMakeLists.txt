if(STORAGELLM_ENABLE_ONEDNN STREQUAL "OFF")
  return()
endif()

include(CheckCXXSourceCompiles)
set(_storagellm_sycl_save_flags "${CMAKE_REQUIRED_FLAGS}")
set(_storagellm_sycl_save_libs "${CMAKE_REQUIRED_LIBRARIES}")
set(STORAGELLM_SYCL_FLAGS "")

if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM|Clang")
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -fsycl")
  set(STORAGELLM_SYCL_FLAGS "-fsycl")
endif()

check_cxx_source_compiles("#include <sycl/sycl.hpp>
int main(){ sycl::queue q; return 0; }" STORAGELLM_HAS_SYCL_HEADER_AND_DRIVER)
set(CMAKE_REQUIRED_FLAGS "${_storagellm_sycl_save_flags}")
set(CMAKE_REQUIRED_LIBRARIES "${_storagellm_sycl_save_libs}")

if(STORAGELLM_HAS_SYCL_HEADER_AND_DRIVER)
  add_library(storagellm_intel_onednn_sycl STATIC
    onednn_sycl_moe_adapter.cpp)
  target_link_libraries(storagellm_intel_onednn_sycl PUBLIC
    storagellm_fast_backend_contract)
  if(STORAGELLM_SYCL_FLAGS)
    target_compile_options(storagellm_intel_onednn_sycl PRIVATE ${STORAGELLM_SYCL_FLAGS})
    target_link_options(storagellm_intel_onednn_sycl PRIVATE ${STORAGELLM_SYCL_FLAGS})
  endif()
  target_compile_definitions(storagellm_intel_onednn_sycl PRIVATE
    STORAGELLM_HAS_ONEDNN_SYCL=1
    STORAGELLM_HAS_ONEDNN_SYCL_MOE_ADAPTER=1)
  message(STATUS "Intel SYCL fused MoE adapter enabled.")
else()
  if(STORAGELLM_ENABLE_ONEDNN STREQUAL "ON")
    message(FATAL_ERROR "Intel SYCL fused MoE adapter requested, but <sycl/sycl.hpp> or SYCL compiler mode was not available.")
  endif()
  message(STATUS "Intel SYCL fused MoE adapter disabled: SYCL compiler/header unavailable.")
endif()
