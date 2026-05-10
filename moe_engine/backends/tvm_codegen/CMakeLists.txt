set(STORAGELLM_ENABLE_TVM_CODEGEN "PIP" CACHE STRING "OFF/AUTO/PIP")
set_property(CACHE STORAGELLM_ENABLE_TVM_CODEGEN PROPERTY STRINGS OFF AUTO PIP)

if(STORAGELLM_ENABLE_TVM_CODEGEN STREQUAL "OFF")
  return()
endif()

message(STATUS "TVM codegen adapter enabled. TVM itself is resolved by pip via scripts/install_tvm_dependency.py, not third_party/tvm.")

add_library(storagellm_tvm_codegen STATIC
  tvm_grouped_moe_adapter.cpp)

target_link_libraries(storagellm_tvm_codegen PRIVATE
  storagellm_fast_backend_contract)

target_include_directories(storagellm_tvm_codegen PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/moe_engine/include)

target_compile_definitions(storagellm_tvm_codegen PRIVATE
  STORAGELLM_HAS_TVM_CODEGEN_ADAPTER=1)
