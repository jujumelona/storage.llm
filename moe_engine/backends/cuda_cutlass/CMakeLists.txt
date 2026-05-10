if(STORAGELLM_ENABLE_CUDA STREQUAL "OFF")
  return()
endif()

if(STORAGELLM_ENABLE_CUTLASS STREQUAL "OFF")
  return()
endif()

set(_storagellm_cutlass_dir "${CMAKE_SOURCE_DIR}/third_party/cutlass")
set(STORAGELLM_CUTLASS_MOE_ADAPTER_IMPLEMENTED ON)
if(STORAGELLM_ENABLE_CUTLASS STREQUAL "FETCH" AND NOT EXISTS "${_storagellm_cutlass_dir}/include/cutlass/cutlass.h")
  include(FetchContent)
  FetchContent_Declare(
    cutlass
    GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
    GIT_TAG v4.4.0
    SOURCE_DIR "${_storagellm_cutlass_dir}")
  FetchContent_MakeAvailable(cutlass)
endif()

if(EXISTS "${_storagellm_cutlass_dir}/include/cutlass/cutlass.h")
  if(NOT STORAGELLM_CUTLASS_MOE_ADAPTER_IMPLEMENTED)
    message(WARNING "CUTLASS found/requested, but cutlass_grouped_moe_adapter.cu is still a fail-closed stub. It is not linked and will not be reported as a working fast backend.")
    return()
  endif()
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  else()
    message(WARNING "CUDA compiler not found. CUDA CUTLASS fast backend disabled.")
    return()
  endif()
  add_library(storagellm_cuda_cutlass STATIC
    cutlass_grouped_moe_adapter.cu)
  target_link_libraries(storagellm_cuda_cutlass PUBLIC
    storagellm_fast_backend_contract)
  target_include_directories(storagellm_cuda_cutlass PRIVATE
    "${_storagellm_cutlass_dir}/include"
    "${_storagellm_cutlass_dir}/tools/util/include")
  target_compile_definitions(storagellm_cuda_cutlass PRIVATE
    STORAGELLM_HAS_CUTLASS=1
    STORAGELLM_HAS_CUTLASS_MOE_ADAPTER=1
    STORAGELLM_MODEL_LIB_ENABLE_CUTLASS_MOE=1)
else()
  message(WARNING "CUTLASS not found. CUDA CUTLASS fast backend disabled.")
endif()
