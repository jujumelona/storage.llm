if(STORAGELLM_ENABLE_ONEDNN STREQUAL "OFF")
  return()
endif()

# Host CPU grouped-MoE adapter.
# The target name is kept for ABI/backward compatibility with the existing
# moe_FAST_BACKEND_CPU_ONEDNN enum.  When oneDNN is available, this target
# links real oneDNN matmul primitives; otherwise it keeps the native AVX path.
add_library(storagellm_cpu_onednn STATIC
  onednn_moe_adapter.cpp)
target_link_libraries(storagellm_cpu_onednn PUBLIC
  storagellm_fast_backend_contract)
target_compile_definitions(storagellm_cpu_onednn PUBLIC
  STORAGELLM_HAS_ONEDNN_CPU_MOE_ADAPTER=1
  STORAGELLM_MODEL_LIB_ENABLE_ONEDNN_CPU_MOE=1)

find_package(dnnl QUIET)
if(dnnl_FOUND)
  target_link_libraries(storagellm_cpu_onednn PUBLIC dnnl)
  target_compile_definitions(storagellm_cpu_onednn PUBLIC
    STORAGELLM_HAS_DNNL_PRIMITIVE=1
    STORAGELLM_MODEL_LIB_ENABLE_REAL_ONEDNN_CPU_MOE=1)
  message(STATUS "storage.llm CPU MoE: real oneDNN primitive path enabled")
else()
  message(STATUS "storage.llm CPU MoE: oneDNN not found; using native AVX CPU path")
endif()

if(STORAGELLM_ENABLE_OPENMP)
  find_package(OpenMP QUIET)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(storagellm_cpu_onednn PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(storagellm_cpu_onednn PUBLIC STORAGELLM_HAS_OPENMP=1)
  endif()
endif()

if(STORAGELLM_ENABLE_NATIVE_OPTIMIZE)
  if(MSVC)
    target_compile_options(storagellm_cpu_onednn PRIVATE /O2 /Oi /Ot)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(storagellm_cpu_onednn PRIVATE -O3 -march=native -mtune=native)
  endif()
endif()
