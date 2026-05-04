// MODULARIZATION: Codec table split into logical sections for better maintainability
// Original 97KB file split into 4 files by functionality
// Each file contains one specific aspect of model loading and parsing

static uint64_t infer_weight_row_bytes(const moe_tensor_info_t& info, uint8_t format);
static void finalize_tensor_layout(moe_tensor_record* rec, uint8_t gguf_encoding_hint);
static uint64_t tensor_stream_bytes(const moe_tensor_record& rec);
static void apply_common_vram_reservation(moe_pc_engine* engine);
static void moe_engine_reset_qkv_cache(moe_pc_engine* engine);
static void moe_engine_clear_offload_gguf_contract(moe_pc_engine* engine);
static int moe_engine_load_offload_gguf_contract(
    moe_pc_engine* engine,
    const std::string& root,
    moe_offload_gguf_scan_t* out_scan);

#include "codec/model_file_io.cpp.inc"
#include "codec/tensor_index_loader.cpp.inc"
#include "codec/gguf_parser.cpp.inc"
#include "codec/juju_parser.cpp.inc"
