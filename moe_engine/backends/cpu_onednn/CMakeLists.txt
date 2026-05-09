if(STORAGELLM_ENABLE_ONEDNN STREQUAL "OFF")
  return()
endif()

# Host CPU native F32 grouped-MoE adapter.
# The target name is kept for ABI/backward compatibility with the existing
# moe_FAST_BACKEND_CPU_ONEDNN enum.  This is a real C++ CPU kernel path; when
# oneDNN is available later it can be added behind the same ABI.
add_library(storagellm_cpu_onednn STATIC
  onednn_moe_adapter.cpp)
target_link_libraries(storagellm_cpu_onednn PUBLIC
  storagellm_fast_backend_contract)
target_compile_definitions(storagellm_cpu_onednn PUBLIC
  STORAGELLM_HAS_ONEDNN_CPU_MOE_ADAPTER=1
  STORAGELLM_MODEL_LIB_ENABLE_ONEDNN_CPU_MOE=1)

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  target_link_libraries(storagellm_cpu_onednn PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(storagellm_cpu_onednn PUBLIC STORAGELLM_HAS_OPENMP=1)
endif()

if(STORAGELLM_ENABLE_NATIVE_OPTIMIZE)
  if(MSVC)
    target_compile_options(storagellm_cpu_onednn PRIVATE /O2 /Oi /Ot)
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    # AppleClang, especially on Apple Silicon CI runners, does not consistently
    # accept the same host-tuning flags as upstream Clang/GCC. Keep the safe
    # optimization level and avoid target-specific flags in portable builds.
    target_compile_options(storagellm_cpu_onednn PRIVATE -O3)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(storagellm_cpu_onednn PRIVATE -O3 -march=native -mtune=native)
  endif()
endif()
