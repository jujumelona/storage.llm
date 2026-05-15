"""JUJU v2 reader/writer contract constants.

Kept outside the large notebook materializer so future model adapters can import
a single source of truth without editing the streamer.
"""

JUJU_IDX_FORMAT = "JUJU_IDX_JSON_V2"
JUJU_IDX_SCHEMA_VERSION = 6
JUJU_BINARY_WIRE_ID = "JUJU_V2_HEADER4096_SECTION96_BUNDLE_NATIVE"
GGUF_CODEC_REGISTRY_VERSION = "GGUF_CODEC_REGISTRY_V1"
JUJU_EXPERT_BUNDLE_TABLE_FORMAT = "JUJU_EXPERT_BUNDLE_TABLE_V2"
JUJU_FORMAT_CONTRACT_VERSION = 4

JUJU_REQUIRED_FEATURES = [
    "juju_idx_schema_v6",
    "juju_binary_wire_v2",
    "gguf_codec_registry_v1",
    "expert_bundle_table_v2",
    "bundle_native",
    "explicit_expert_layout",
    "graph_ir",
    "qkv_policy",
    "codec_registry_v1",
    "fail_closed_kernel_contract",
    "fail_closed_required_features",
    "exact_ppl_mode",
    "metadata_first_expert_resolution",
    "source_quant_rows_preserved",
    "deterministic_router_topk",
    "moe_router_topk_contract",
    "moe_router_topk_contract_aliases",
    "metadata_sidecar_config_extraction",
    "attention_scale_contract",
    "attention_head_dim_fallback_scale",
    "attention_query_pre_attn_scalar_contract",
    "storage_plan_contract",
    "runtime_tensor_index_contract",
]

# GGUF tensor types for which this project has an exact row-byte contract.
# Reader still fail-closes when a runtime dot/dequant kernel is missing.
GGUF_EXACT_ROW_BYTE_TYPES = {
    0, 1, 2, 3, 6, 7, 8, 9,
    10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 29, 30,
    34, 35, 39,
}
