// Offload/JUJU model-root validation - FULLY MODULARIZED
//
// This file has been completely modularized into the following components:
//
// 1. model_contract_types.cpp.inc - Contract type definitions and helper functions
//    - moe_weight_encoding_from_family
//    - moe_offload_gguf_contract_t
//    - moe_offload_gguf_file_t
//    - moe_offload_gguf_scan_t
//    - moe_gguf_tensor_dir_entry_t
//    - moe_gguf_tensor_dir_t
//
// 2. metadata_parser.cpp.inc - Metadata JSON parsing
//    - moe_parse_juju_split_metadata_json
//    - moe_parse_offload_metadata_json
//
// 3. gguf_tensor_reader.cpp.inc - GGUF tensor reading
//    - moe_gguf_tensor_name_executable
//    - moe_read_gguf_tensor_infos
//    - moe_read_gguf_tensor_directory
//    - moe_scan_gguf_tensor_infos
//
// 4. model_file_readers.cpp.inc - GGUF/JUJU file readers
// 5. model_scan.cpp.inc - Model root scanning
// 6. model_helpers.cpp.inc - Model ID inference, QKV config
// 7. model_engine_state.cpp.inc - QKV cache reset, contract clearing
// 8. model_engine_contract.cpp.inc - Engine contract loading
// 9. model_validation_api.cpp.inc - Public API functions
// 10. gguf_types.cpp.inc - GGUF type definitions and I/O
// 11. json_parser.cpp.inc - JSON parsing utilities
// 12. juju_parser.cpp.inc - JUJU parsing, tensor index, metadata
// 13. platform_utils.cpp.inc - Platform utilities, string ops, path handling
// 14. platform_dir.cpp.inc - Cross-platform directory iteration
// 15. binary_reader.cpp.inc - Template-based binary reading
// 16. file_collector.cpp.inc - Generic file collection
//
// All includes are managed in moe_pc_engine.cpp in correct dependency order.
// Original file size: 464 lines → Current: ~38 lines (92% reduction!)
