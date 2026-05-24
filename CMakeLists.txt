cmake_minimum_required(VERSION 3.24)
project(storagellm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(STORAGELLM_BUILD_TESTS "Build tests" ON)
option(STORAGELLM_PPL_DIAG "Enable PPL diagnostics" OFF)
option(STORAGELLM_ENABLE_SANITIZERS "Enable sanitizers (ASan, UBSan)" OFF)
set(STORAGELLM_SANITIZER_SET "address,undefined" CACHE STRING "Comma-separated sanitizer set when STORAGELLM_ENABLE_SANITIZERS is ON")
option(STORAGELLM_ENABLE_NATIVE_OPTIMIZE "Optimize local C++ CPU kernels for this host during auto-build" ON)
set(STORAGELLM_ENABLE_CUDA "AUTO" CACHE STRING "OFF/AUTO/ON")
set_property(CACHE STORAGELLM_ENABLE_CUDA PROPERTY STRINGS OFF AUTO ON)
set(STORAGELLM_ENABLE_CUBLASLT "AUTO" CACHE STRING "OFF/AUTO/ON")
set_property(CACHE STORAGELLM_ENABLE_CUBLASLT PROPERTY STRINGS OFF AUTO ON)
set(STORAGELLM_ENABLE_CUTLASS "OFF" CACHE STRING "OFF/AUTO/FETCH/ON")
set_property(CACHE STORAGELLM_ENABLE_CUTLASS PROPERTY STRINGS OFF AUTO FETCH ON)
set(STORAGELLM_ENABLE_TVM_CODEGEN "PIP" CACHE STRING "OFF/AUTO/PIP")
set_property(CACHE STORAGELLM_ENABLE_TVM_CODEGEN PROPERTY STRINGS OFF AUTO PIP)
set(STORAGELLM_ENABLE_ROCM "AUTO" CACHE STRING "OFF/AUTO/ON")
set_property(CACHE STORAGELLM_ENABLE_ROCM PROPERTY STRINGS OFF AUTO ON)
set(STORAGELLM_ENABLE_HIPBLASLT "AUTO" CACHE STRING "OFF/AUTO/ON")
set_property(CACHE STORAGELLM_ENABLE_HIPBLASLT PROPERTY STRINGS OFF AUTO ON)
set(STORAGELLM_ENABLE_CK "AUTO" CACHE STRING "OFF/AUTO/ON")
set_property(CACHE STORAGELLM_ENABLE_CK PROPERTY STRINGS OFF AUTO ON)
set(STORAGELLM_ENABLE_METAL "AUTO" CACHE STRING "OFF/AUTO/ON")
set_property(CACHE STORAGELLM_ENABLE_METAL PROPERTY STRINGS OFF AUTO ON)
set(STORAGELLM_ENABLE_ONEDNN "AUTO" CACHE STRING "OFF/AUTO/ON")
set_property(CACHE STORAGELLM_ENABLE_ONEDNN PROPERTY STRINGS OFF AUTO ON)
set(STORAGELLM_ENABLE_VULKAN "AUTO" CACHE STRING "OFF/AUTO/ON")
set_property(CACHE STORAGELLM_ENABLE_VULKAN PROPERTY STRINGS OFF AUTO ON)
set(STORAGELLM_ENABLE_OPENCL "AUTO" CACHE STRING "OFF/AUTO/ON")
set_property(CACHE STORAGELLM_ENABLE_OPENCL PROPERTY STRINGS OFF AUTO ON)
set(STORAGELLM_ENABLE_CLBLAST "AUTO" CACHE STRING "OFF/AUTO/ON")
set_property(CACHE STORAGELLM_ENABLE_CLBLAST PROPERTY STRINGS OFF AUTO ON)


# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)

    if(STORAGELLM_ENABLE_SANITIZERS)
        if(STORAGELLM_SANITIZER_SET STREQUAL "")
            message(FATAL_ERROR "STORAGELLM_SANITIZER_SET must not be empty when sanitizers are enabled")
        endif()
        add_compile_options("-fsanitize=${STORAGELLM_SANITIZER_SET}" -fno-omit-frame-pointer)
        add_link_options("-fsanitize=${STORAGELLM_SANITIZER_SET}")
    endif()
endif()

# Diagnostics flag
if(STORAGELLM_PPL_DIAG)
    add_compile_definitions(STORAGELLM_PPL_DIAG=1)
else()
    add_compile_definitions(STORAGELLM_PPL_DIAG=0)
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/moe_engine/include)
include_directories(${CMAKE_SOURCE_DIR}/loader)
include_directories(${CMAKE_SOURCE_DIR}/engine_core)

add_subdirectory(moe_engine/backends)

# Loader library
add_library(storagellm_loader STATIC
    loader/file_read.cpp
    loader/json_find.cpp
    loader/json_match.cpp
    loader/json_value.cpp
    loader/manifest_key.cpp
    loader/manifest_lookup.cpp
    loader/manifest_projection.cpp
    loader/path_join.cpp
)

# Engine core KV library
add_library(storagellm_kv STATIC
    engine_core/kv/kv_qkv.cpp
    engine_core/kv/qkv_attention.cpp
    engine_core/kv/qkv_codebook.cpp
    engine_core/kv/qkv_dequantize.cpp
    engine_core/kv/qkv_helpers.cpp
    engine_core/kv/qkv_matrix.cpp
    engine_core/kv/qkv_packing.cpp
    engine_core/kv/qkv_quantize.cpp
    engine_core/kv/qkv_state.cpp
    engine_core/kv/qkv_thread_pool.cpp
)

# Engine core mmap loader
add_library(storagellm_mmap STATIC
    engine_core/core/mmap_loader.cpp
)

# Main MOE engine library
add_library(storagellm_engine STATIC
    moe_engine/src/moe_pc_engine.cpp
    moe_engine/src/hardcode_constants.cpp
    moe_engine/src/hardcode_layers.cpp
    moe_engine/src/hardcode_parts.cpp
    moe_engine/src/hardcode_projection_shapes.cpp
    moe_engine/src/hardcode_raw_spans.cpp
    moe_engine/src/hardcode_summary.cpp
    moe_engine/src/model_shape.cpp
    moe_engine/native/scale4.cpp
    moe_engine/src/tools/storage_f8.cpp
)

target_link_libraries(storagellm_engine
    PRIVATE storagellm_loader
    PRIVATE storagellm_kv
    PRIVATE storagellm_mmap
)

if(STORAGELLM_ENABLE_NATIVE_OPTIMIZE)
    if(MSVC)
        target_compile_options(storagellm_engine PRIVATE /O2 /Oi /Ot)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(storagellm_engine PRIVATE -O3 -march=native -mtune=native -UNDEBUG)
    endif()
endif()

# Optional fast backends are created by moe_engine/backends.  Link only targets
# that really configured successfully; do not require vendored TVM/CUTLASS trees.
if(TARGET storagellm_tvm_codegen)
    target_link_libraries(storagellm_engine PRIVATE storagellm_tvm_codegen)
    target_compile_definitions(storagellm_engine PRIVATE STORAGELLM_HAS_TVM_CODEGEN_ADAPTER=1)
endif()
if(TARGET storagellm_cuda_cublaslt)
    target_link_libraries(storagellm_engine PRIVATE storagellm_cuda_cublaslt)
    target_compile_definitions(storagellm_engine PRIVATE
        STORAGELLM_HAS_CUBLASLT_MOE_ADAPTER=1
        STORAGELLM_MODEL_LIB_ENABLE_CUBLASLT_MOE=1)
endif()
if(TARGET storagellm_cuda_cutlass)
    target_link_libraries(storagellm_engine PRIVATE storagellm_cuda_cutlass)
    target_compile_definitions(storagellm_engine PRIVATE
        STORAGELLM_HAS_CUTLASS_MOE_ADAPTER=1
        STORAGELLM_MODEL_LIB_ENABLE_CUTLASS_MOE=1)
endif()

if(TARGET storagellm_cpu_onednn)
    target_link_libraries(storagellm_engine PRIVATE storagellm_cpu_onednn)
    target_compile_definitions(storagellm_engine PRIVATE
        STORAGELLM_HAS_ONEDNN_CPU_MOE_ADAPTER=1
        STORAGELLM_MODEL_LIB_ENABLE_ONEDNN_CPU_MOE=1)
endif()
if(TARGET storagellm_rocm_hipblaslt)
    target_link_libraries(storagellm_engine PRIVATE storagellm_rocm_hipblaslt)
    target_compile_definitions(storagellm_engine PRIVATE
        STORAGELLM_HAS_HIPBLASLT_MOE_ADAPTER=1
        STORAGELLM_MODEL_LIB_ENABLE_HIPBLASLT_MOE=1)
endif()
if(TARGET storagellm_rocm_ck)
    target_link_libraries(storagellm_engine PRIVATE storagellm_rocm_ck)
    target_compile_definitions(storagellm_engine PRIVATE
        STORAGELLM_HAS_CK_MOE_ADAPTER=1
        STORAGELLM_MODEL_LIB_ENABLE_CK_MOE=1)
endif()
if(TARGET storagellm_metal_mps)
    target_link_libraries(storagellm_engine PRIVATE storagellm_metal_mps)
    target_compile_definitions(storagellm_engine PRIVATE
        STORAGELLM_HAS_METAL_MPS_MOE_ADAPTER=1
        STORAGELLM_MODEL_LIB_ENABLE_METAL_MPS_MOE=1)
endif()
if(TARGET storagellm_intel_onednn_sycl)
    target_link_libraries(storagellm_engine PRIVATE storagellm_intel_onednn_sycl)
    target_compile_definitions(storagellm_engine PRIVATE
        STORAGELLM_HAS_ONEDNN_SYCL_MOE_ADAPTER=1
        STORAGELLM_MODEL_LIB_ENABLE_ONEDNN_SYCL_MOE=1)
endif()
if(TARGET storagellm_vulkan_coopmat)
    target_link_libraries(storagellm_engine PRIVATE storagellm_vulkan_coopmat)
    target_compile_definitions(storagellm_engine PRIVATE
        STORAGELLM_HAS_VULKAN_COOPMAT_MOE_ADAPTER=1
        STORAGELLM_MODEL_LIB_ENABLE_VULKAN_COOPMAT_MOE=1)
endif()
if(TARGET storagellm_opencl_clblast)
    target_link_libraries(storagellm_engine PRIVATE storagellm_opencl_clblast)
    target_compile_definitions(storagellm_engine PRIVATE
        STORAGELLM_HAS_CLBLAST_MOE_ADAPTER=1
        STORAGELLM_MODEL_LIB_ENABLE_CLBLAST_MOE=1)
endif()


# Platform-specific libraries
if(WIN32)
    target_link_libraries(storagellm_engine PRIVATE ws2_32)
elseif(UNIX AND NOT APPLE)
    target_link_libraries(storagellm_engine PRIVATE pthread dl)
    # Check for io_uring
    find_library(URING_LIB uring)
    if(URING_LIB)
        target_link_libraries(storagellm_engine PRIVATE ${URING_LIB})
        target_compile_definitions(storagellm_engine PRIVATE MOE_HAVE_LIBURING=1)
    endif()
elseif(APPLE)
    target_link_libraries(storagellm_engine PRIVATE pthread)
endif()


# C++ host-local autotune executable.
# This owns automatic detection/measurement/selection. Python is used only as a
# helper for pip/TVM codegen because TVM exposes that path through Python.
add_executable(storagellm_host_autotune
    tools/autotune/main.cpp
    tools/autotune/process.cpp
    tools/autotune/files.cpp
    tools/autotune/host_detect.cpp
    tools/autotune/tvm_pipeline.cpp
    tools/autotune/benchmark.cpp
    tools/autotune/transfer_profile.cpp
    tools/autotune/select.cpp
)
target_include_directories(storagellm_host_autotune PRIVATE
    ${CMAKE_SOURCE_DIR}/tools/autotune
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/moe_engine/include
)
if(UNIX AND NOT APPLE)
    target_link_libraries(storagellm_host_autotune PRIVATE dl)
endif()
if(TARGET storagellm_cpu_onednn)
    target_link_libraries(storagellm_host_autotune PRIVATE storagellm_cpu_onednn)
    target_compile_definitions(storagellm_host_autotune PRIVATE STORAGELLM_AUTOTUNE_HAS_CPU_NATIVE=1)
endif()
if(TARGET storagellm_cuda_cublaslt)
    target_link_libraries(storagellm_host_autotune PRIVATE storagellm_cuda_cublaslt)
    target_compile_definitions(storagellm_host_autotune PRIVATE STORAGELLM_AUTOTUNE_HAS_CUBLASLT_NATIVE=1)
endif()
if(TARGET storagellm_cuda_cutlass)
    target_link_libraries(storagellm_host_autotune PRIVATE storagellm_cuda_cutlass)
    target_compile_definitions(storagellm_host_autotune PRIVATE STORAGELLM_AUTOTUNE_HAS_CUTLASS_NATIVE=1)
endif()
if(TARGET storagellm_cuda_cublaslt OR TARGET storagellm_cuda_cutlass)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND AND TARGET CUDA::cudart)
        target_link_libraries(storagellm_host_autotune PRIVATE CUDA::cudart)
        target_compile_definitions(storagellm_host_autotune PRIVATE STORAGELLM_AUTOTUNE_HAS_CUDA_RUNTIME=1)
    endif()
endif()
if(TARGET storagellm_rocm_hipblaslt)
    target_link_libraries(storagellm_host_autotune PRIVATE storagellm_rocm_hipblaslt)
    target_compile_definitions(storagellm_host_autotune PRIVATE
        STORAGELLM_AUTOTUNE_HAS_HIPBLASLT_NATIVE=1
        STORAGELLM_AUTOTUNE_HAS_HIP_RUNTIME=1)
    find_package(hip QUIET)
    if(TARGET hip::host)
        target_link_libraries(storagellm_host_autotune PRIVATE hip::host)
    endif()
endif()
if(TARGET storagellm_rocm_ck)
    target_link_libraries(storagellm_host_autotune PRIVATE storagellm_rocm_ck)
    target_compile_definitions(storagellm_host_autotune PRIVATE
        STORAGELLM_AUTOTUNE_HAS_CK_NATIVE=1
        STORAGELLM_AUTOTUNE_HAS_HIP_RUNTIME=1)
    find_package(hip QUIET)
    if(TARGET hip::host)
        target_link_libraries(storagellm_host_autotune PRIVATE hip::host)
    endif()
endif()
if(TARGET storagellm_metal_mps)
    target_link_libraries(storagellm_host_autotune PRIVATE storagellm_metal_mps)
    target_compile_definitions(storagellm_host_autotune PRIVATE STORAGELLM_AUTOTUNE_HAS_METAL_NATIVE=1)
endif()
if(TARGET storagellm_intel_onednn_sycl)
    target_link_libraries(storagellm_host_autotune PRIVATE storagellm_intel_onednn_sycl)
    target_compile_definitions(storagellm_host_autotune PRIVATE STORAGELLM_AUTOTUNE_HAS_ONEDNN_SYCL_NATIVE=1)
endif()
if(TARGET storagellm_vulkan_coopmat)
    target_link_libraries(storagellm_host_autotune PRIVATE storagellm_vulkan_coopmat)
    target_compile_definitions(storagellm_host_autotune PRIVATE STORAGELLM_AUTOTUNE_HAS_VULKAN_NATIVE=1)
endif()
if(TARGET storagellm_opencl_clblast)
    target_link_libraries(storagellm_host_autotune PRIVATE storagellm_opencl_clblast)
    target_compile_definitions(storagellm_host_autotune PRIVATE
        STORAGELLM_AUTOTUNE_HAS_CLBLAST_NATIVE=1
        STORAGELLM_AUTOTUNE_HAS_OPENCL_RUNTIME=1)
endif()

# Example server executable
add_executable(storagellm_server
    moe_engine/examples/pc_engine_server.cpp
)

target_link_libraries(storagellm_server
    PRIVATE storagellm_engine
)
if(STORAGELLM_ENABLE_NATIVE_OPTIMIZE)
    if(MSVC)
        target_compile_options(storagellm_server PRIVATE /O2 /Oi /Ot)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(storagellm_server PRIVATE -O3 -march=native -mtune=native -UNDEBUG)
    endif()
endif()

# Probe executable
add_executable(storagellm_probe
    moe_engine/examples/pc_engine_probe.cpp
)

target_link_libraries(storagellm_probe
    PRIVATE storagellm_engine
)

# Testing
if(STORAGELLM_BUILD_TESTS)
    enable_testing()

    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(
        NAME PythonJujuGenerationContractConfig
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/moe_engine/tests/run_juju_generation_contract_config.py
    )

    # Find Google Test
    find_package(GTest QUIET)

    if(NOT GTest_FOUND)
        set(STORAGELLM_BUNDLED_GTEST_SRC "" CACHE PATH "Optional local googletest source tree")
        if(STORAGELLM_BUNDLED_GTEST_SRC AND EXISTS "${STORAGELLM_BUNDLED_GTEST_SRC}/CMakeLists.txt")
            message(STATUS "Using googletest from ${STORAGELLM_BUNDLED_GTEST_SRC}")
            add_subdirectory("${STORAGELLM_BUNDLED_GTEST_SRC}" "${CMAKE_BINARY_DIR}/_deps/googletest-build" EXCLUDE_FROM_ALL)
        elseif(EXISTS "${CMAKE_SOURCE_DIR}/third_party/googletest/CMakeLists.txt")
            message(STATUS "Using bundled googletest from third_party/googletest")
            add_subdirectory("${CMAKE_SOURCE_DIR}/third_party/googletest" "${CMAKE_BINARY_DIR}/_deps/googletest-build" EXCLUDE_FROM_ALL)
        else()
            message(WARNING "Google Test not found locally; skipping C++ gtest targets. Install libgtest-dev or set STORAGELLM_BUNDLED_GTEST_SRC to enable them. Python contract tests remain registered.")
        endif()
    endif()

    if(TARGET GTest::gtest AND TARGET GTest::gtest_main)
    function(storagellm_add_gtest test_name)
        add_executable(${test_name} ${ARGN})
        target_link_libraries(${test_name}
            PRIVATE GTest::gtest
            PRIVATE GTest::gtest_main
        )
        add_test(NAME ${test_name} COMMAND ${test_name})
    endfunction()

    # Bug fixes tests
    add_executable(test_bug_fixes
        moe_engine/tests/test_bug_fixes.cpp
    )
    target_link_libraries(test_bug_fixes
        PRIVATE storagellm_engine
        PRIVATE GTest::gtest
        PRIVATE GTest::gtest_main
    )
    add_test(NAME BugFixes COMMAND test_bug_fixes)

    # JSON parser tests
    add_executable(test_json_parser
        moe_engine/tests/test_json_parser.cpp
    )
    target_link_libraries(test_json_parser
        PRIVATE GTest::gtest
        PRIVATE GTest::gtest_main
    )
    add_test(NAME JSONParser COMMAND test_json_parser)

    # Prefetch validation tests
    add_executable(test_prefetch_validation
        moe_engine/tests/test_prefetch_validation.cpp
    )
    target_link_libraries(test_prefetch_validation
        PRIVATE GTest::gtest
        PRIVATE GTest::gtest_main
    )
    add_test(NAME PrefetchValidation COMMAND test_prefetch_validation)

    # QKV parity tests (Bug B3)
    add_executable(test_qkv_parity
        moe_engine/tests/test_qkv_parity.cpp
    )
    target_link_libraries(test_qkv_parity
        PRIVATE storagellm_kv
        PRIVATE GTest::gtest
        PRIVATE GTest::gtest_main
    )
    add_test(NAME QKVParity COMMAND test_qkv_parity)


    # Module contract example tests
    storagellm_add_gtest(test_loader_manifest_examples
        moe_engine/tests/test_loader_manifest_examples.cpp
    )
    target_link_libraries(test_loader_manifest_examples
        PRIVATE storagellm_loader
    )

    storagellm_add_gtest(test_qkv_contract_examples
        moe_engine/tests/test_qkv_contract_examples.cpp
    )
    target_link_libraries(test_qkv_contract_examples
        PRIVATE storagellm_kv
    )

    storagellm_add_gtest(test_weight_contract_examples
        moe_engine/tests/test_weight_contract_examples.cpp
    )

    storagellm_add_gtest(test_ppl_math_contract_examples
        moe_engine/tests/test_ppl_math_contract_examples.cpp
    )

    storagellm_add_gtest(test_ppl_diag_contract_examples
        moe_engine/tests/test_ppl_diag_contract_examples.cpp
    )
    target_link_libraries(test_ppl_diag_contract_examples
        PRIVATE storagellm_kv
    )

    storagellm_add_gtest(test_engine_fixture_loading
        moe_engine/tests/test_engine_fixture_loading.cpp
    )
    target_link_libraries(test_engine_fixture_loading
        PRIVATE storagellm_engine
    )

    # Batch dot microbenchmark (Bug B4)
    add_executable(bench_batch_dot
        moe_engine/tests/bench_batch_dot.cpp
    )
    target_link_libraries(bench_batch_dot
        PRIVATE storagellm_engine
    )
    # Note: Benchmark is not added to CTest (run manually for performance testing)

    message(STATUS "Tests enabled. Run with: ctest --output-on-failure")
    message(STATUS "Benchmarks: ./build/bench_batch_dot")
    else()
        message(STATUS "C++ gtest targets disabled because Google Test was not found.")
        message(STATUS "Python contract tests enabled. Run with: ctest --output-on-failure -R PythonJujuGenerationContractConfig")
    endif()
endif()

# Installation
install(TARGETS storagellm_engine storagellm_server storagellm_probe storagellm_host_autotune
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY moe_engine/include/
    DESTINATION include/storagellm
    FILES_MATCHING PATTERN "*.h" PATTERN "*.inc"
)

# Print configuration summary
message(STATUS "")
message(STATUS "storageLLM Configuration Summary:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build tests: ${STORAGELLM_BUILD_TESTS}")
message(STATUS "  PPL diagnostics: ${STORAGELLM_PPL_DIAG}")
message(STATUS "  Sanitizers: ${STORAGELLM_ENABLE_SANITIZERS}")
message(STATUS "  CUDA: ${STORAGELLM_ENABLE_CUDA}")
message(STATUS "  cuBLASLt: ${STORAGELLM_ENABLE_CUBLASLT}")
message(STATUS "  CUTLASS: ${STORAGELLM_ENABLE_CUTLASS}")
message(STATUS "  ROCm: ${STORAGELLM_ENABLE_ROCM}")
message(STATUS "  Metal: ${STORAGELLM_ENABLE_METAL}")
message(STATUS "  Vulkan: ${STORAGELLM_ENABLE_VULKAN}")
message(STATUS "  OpenCL/CLBlast: ${STORAGELLM_ENABLE_OPENCL}/${STORAGELLM_ENABLE_CLBLAST}")
message(STATUS "  oneDNN/SYCL: ${STORAGELLM_ENABLE_ONEDNN}")
message(STATUS "  TVM codegen: ${STORAGELLM_ENABLE_TVM_CODEGEN}")
message(STATUS "")
