import importlib.util
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAT_PATH = ROOT / "colab" / "juju_shard_materializer.py"
spec = importlib.util.spec_from_file_location("juju_shard_materializer", MAT_PATH)
mat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mat)


def test_runtime_arch_reads_nested_hf_metadata_files_config_json():
    contract = {
        "hf_metadata_files": {
            "config.json": {
                "model_type": "generic_moe",
                "text_config": {
                    "hidden_size": 2816,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 8,
                    "head_dim": 256,
                    "query_pre_attn_scalar": 256,
                    "num_experts": 128,
                    "top_k_experts": 8,
                    "moe_intermediate_size": 704,
                    "intermediate_size": 2112,
                    "norm_topk_prob": True,
                    "scoring_func": "softmax",
                    "rms_norm_eps": 1e-6,
                },
            }
        }
    }
    arch = mat.juju_runtime_arch_metadata(contract, {"gguf_runtime": {}})
    assert arch["query_pre_attn_scalar"] == 256.0
    assert abs(arch["attention_scale"] - 0.0625) < 1e-9
    assert arch["attention_scale_source"] == "query_pre_attn_scalar"
    assert arch["experts_per_moe_layer"] == 128
    assert arch["num_experts"] == 128
    assert arch["routed_experts_per_token"] == 8
    assert arch["top_k_experts"] == 8
    assert arch["expert_intermediate_size"] == 704
    assert arch["moe_intermediate_size"] == 704
    assert arch["norm_topk_prob"] is True
    assert arch["normalize_topk_prob"] is True
    assert arch["scoring_func"] == "softmax"


def test_layer_graph_ir_carries_attention_and_router_contracts():
    tensors = [
        {"name": "blk.0.attn_q.weight"},
        {"name": "blk.0.attn_k.weight"},
        {"name": "blk.0.attn_output.weight"},
        {"name": "blk.0.ffn_gate_inp.weight"},
        {"name": "blk.0.ffn_gate_inp.scale"},
        {"name": "blk.0.ffn_gate.weight"},
        {"name": "blk.0.ffn_up.weight"},
        {"name": "blk.0.ffn_down.weight"},
        {"name": "blk.0.ffn_gate_exps.weight"},
    ]
    arch = {
        "head_dim": 256,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "query_pre_attn_scalar": 256.0,
        "attention_scale": 0.0625,
        "experts_per_moe_layer": 128,
        "num_experts": 128,
        "routed_experts_per_token": 8,
        "top_k_experts": 8,
        "norm_topk_prob": True,
        "normalize_topk_prob": True,
        "scoring_func": "softmax",
    }
    row = mat._juju_layer_execution_contract_table(tensors, arch)[0]
    attn_row = mat._juju_attention_layer_contract_table(tensors, arch)[0]
    assert row["attention"]["query_pre_attn_scalar"] == 256.0
    assert row["attention"]["attention_scale"] == 0.0625
    assert attn_row["query_pre_attn_scalar"] == 256.0
    assert attn_row["attention_scale"] == 0.0625
    assert row["router"]["experts_per_moe_layer"] == 128
    assert row["router"]["num_experts"] == 128
    assert row["router"]["routed_experts_per_token"] == 8
    assert row["router"]["top_k_experts"] == 8
    assert row["router"]["norm_topk_prob"] is True
    assert row["router"]["normalize_topk_prob"] is True
    assert row["router"]["scoring_func"] == "softmax"
    assert row["router"]["router_scale"] == []
    assert row["router"].get("router_uses_hidden_when_raw_residual_or_internal_scale_contract", True) is True
    graph = mat.build_layer_graph_ir(0, tensors, arch)
    router_select = next(op for op in graph["ops"] if op["name"] == "router_input")
    assert router_select.get("raw_residual_contract", True) is True
    assert router_select.get("scale", []) == []



def test_cpp_reader_accepts_all_declared_juju_required_features():
    contract_path = ROOT / "colab" / "juju_modules" / "format_contract.py"
    spec = importlib.util.spec_from_file_location("format_contract", contract_path)
    fmt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fmt)

    parser_text = (ROOT / "moe_engine" / "src" / "parts" / "juju_parser.cpp.inc").read_text()
    missing = [
        feature for feature in fmt.JUJU_REQUIRED_FEATURES
        if f'"{feature.lower()}"' not in parser_text
    ]
    assert not missing, f"C++ JUJU reader does not accept required features: {missing}"


def test_materializer_emits_fail_closed_juju_reader_contract_fields():
    contract_path = ROOT / "colab" / "juju_modules" / "format_contract.py"
    spec = importlib.util.spec_from_file_location("format_contract", contract_path)
    fmt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fmt)

    assert list(mat.JUJU_REQUIRED_FEATURES) == list(fmt.JUJU_REQUIRED_FEATURES)
    assert mat.JUJU_IDX_FORMAT == fmt.JUJU_IDX_FORMAT
    assert mat.JUJU_IDX_SCHEMA_VERSION == fmt.JUJU_IDX_SCHEMA_VERSION
    assert mat.JUJU_BINARY_WIRE_ID == fmt.JUJU_BINARY_WIRE_ID
    assert mat.JUJU_EXPERT_BUNDLE_TABLE_FORMAT == fmt.JUJU_EXPERT_BUNDLE_TABLE_FORMAT
    assert mat.JUJU_FORMAT_CONTRACT_VERSION == fmt.JUJU_FORMAT_CONTRACT_VERSION

    materializer_text = MAT_PATH.read_text()
    assert '"required_features": list(JUJU_REQUIRED_FEATURES)' in materializer_text
    assert '"binary_wire_id": JUJU_BINARY_WIRE_ID' in materializer_text
    assert '"codec_registry_version": GGUF_CODEC_REGISTRY_VERSION' in materializer_text
    assert mat.GGUF_CODEC_REGISTRY_VERSION == fmt.GGUF_CODEC_REGISTRY_VERSION
    assert '"expert_bundle_table_format": JUJU_EXPERT_BUNDLE_TABLE_FORMAT' in materializer_text
    assert '"format_contract_version": JUJU_FORMAT_CONTRACT_VERSION' in materializer_text


def test_all_materializer_idx_constructions_use_canonical_contract_helper():
    text = MAT_PATH.read_text()
    assert 'def juju_required_idx_contract_fields' in text
    current_schema_sites = text.count('"schema_version": JUJU_IDX_SCHEMA_VERSION')
    helper_sites = text.count('**juju_required_idx_contract_fields(),')
    assert current_schema_sites > 0
    assert helper_sites >= current_schema_sites


def test_cpp_reader_requires_current_schema_and_contract_fields():
    parser_text = (ROOT / "moe_engine" / "src" / "parts" / "juju_parser.cpp.inc").read_text()
    assert 'schema_version != 6u' in parser_text
    assert 'missing JUJU required_features' in parser_text
    assert 'missing/wrong JUJU binary_wire_id' in parser_text
    assert 'missing/wrong JUJU codec_registry_version' in parser_text
    assert 'missing/wrong JUJU expert_bundle_table_format' in parser_text
    assert 'missing/wrong JUJU format_contract_version' in parser_text
    assert 'missing/wrong JUJU format_contract_source' in parser_text
    assert 'missing/wrong JUJU mutable_runtime_index' in parser_text
    assert 'moe_juju_required_feature_set_complete' in parser_text
    assert 'JUJU idx missing/invalid query_pre_attn_scalar' in parser_text
    assert 'JUJU idx attention scale mismatch' in parser_text


def test_engine_ingests_idx_attention_contract_into_runtime_metadata():
    parser_text = (ROOT / "moe_engine" / "src" / "parts" / "juju_parser.cpp.inc").read_text()
    main_parse_text = (ROOT / "moe_engine" / "src" / "parts" / "codec" / "juju_main_parsing.cpp.inc").read_text()
    readers_text = (ROOT / "moe_engine" / "src" / "parts" / "model_file_readers.cpp.inc").read_text()
    assert "moe_juju_extract_attention_contract" in parser_text
    assert "JUJU_RUNTIME_ATTENTION_CONTRACT_V1" in parser_text
    assert "engine->model_config_query_pre_attn_scalar = (float)query_pre_attn_scalar" in parser_text
    assert main_parse_text.count("moe_ingest_juju_attention_contract(engine, index_json);") >= 2
    assert "moe_juju_attention_contract_compact_json(index_json)" in readers_text
    assert "!moe_juju_index_schema_supported(index_json)" in readers_text


def test_cpp_reader_requires_every_declared_feature_not_just_unknown_filter():
    contract_path = ROOT / "colab" / "juju_modules" / "format_contract.py"
    spec = importlib.util.spec_from_file_location("format_contract", contract_path)
    fmt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fmt)
    parser_text = (ROOT / "moe_engine" / "src" / "parts" / "juju_parser.cpp.inc").read_text()
    assert 'moe_juju_required_feature_contract' in parser_text
    assert 'missing JUJU required feature group: %s' not in parser_text
    assert 'missing JUJU required feature: %s' in parser_text
    for feature in fmt.JUJU_REQUIRED_FEATURES:
        assert f'"{feature}"' in parser_text


def test_exact_ppl_eval_is_fail_closed_on_runtime_contract_and_fallbacks():
    eval_text = (ROOT / "moe_engine" / "src" / "parts" / "generation_eval.cpp.inc").read_text()
    state_text = (ROOT / "moe_engine" / "src" / "parts" / "engine_state.cpp.inc").read_text()
    parser_text = (ROOT / "moe_engine" / "src" / "parts" / "juju_parser.cpp.inc").read_text()
    assert "juju_exact_ppl_contract_ready" in state_text
    assert "juju_required_features_contract_ready" in state_text
    assert "juju_attention_contract_ready" in state_text
    assert "model_config_attention_scale" in state_text
    assert "moe_eval_exact_ppl_contract_ready" in eval_text
    assert "JUJU exact_ppl_mode required_features contract is not loaded" in eval_text
    assert "JUJU attention scale contract is not loaded" in eval_text
    assert "JUJU attention scale does not match runtime attention contract" in eval_text
    assert "moe_engine_contract_uses_unit_qk_norm_global(engine)" in eval_text
    assert "JUJU storage format plan is not available for exact PPL" in eval_text
    assert "JUJU expert index is not ready for exact PPL" in eval_text
    assert "selected-expert linear fallback was used" in eval_text
    assert "non-finite hidden state" in eval_text
    assert "non-finite lm_head logprob" in eval_text
    assert "moe_juju_required_feature_present(json, \"exact_ppl_mode\")" in parser_text
    assert "engine->model_config_attention_scale = (float)attention_scale" in parser_text


def test_gguf_byte_diagnostics_treats_alignment_padding_as_ok():
    diag = mat.gguf_tensor_byte_diagnostics([
        {
            "name": "blk.1.layer_output_scale.weight",
            "type": 0,
            "shape": [1],
            "exact_bytes": 4,
            "source_storage_bytes": 32,
            "bytes": 4,
            "source_padding_bytes": 28,
        }
    ])
    assert diag["mismatch_count"] == 0
    assert diag["alignment_padding_ok"] == 1
    assert diag["type_stats"][0]["alignment_padding_ok"] == 1


def test_file_size_prefers_range_total_over_head_content_length():
    class _Resp:
        def __init__(self, ok=True, headers=None):
            self.ok = ok
            self.headers = headers or {}
            self.closed = False

        def close(self):
            self.closed = True

    class _Session:
        def __init__(self):
            self.calls = []

        def get(self, url, headers=None, stream=True, timeout=None):
            self.calls.append(("GET", dict(headers or {})))
            return _Resp(headers={"Content-Range": "bytes 0-0/123456789"})

        def head(self, url, allow_redirects=True, headers=None, timeout=None):
            self.calls.append(("HEAD", dict(headers or {})))
            return _Resp(headers={"Content-Length": "42"})

    session = _Session()
    assert mat.file_size(session, "https://example.invalid/model.gguf", token="tok") == 123456789
    assert session.calls[0][0] == "GET"
    assert session.calls[0][1]["Range"] == "bytes=0-0"
    assert session.calls[0][1]["Authorization"] == "Bearer tok"


def test_cpp_juju_sidecar_is_authoritative_and_basename_tolerant():
    parser_text = (ROOT / "moe_engine" / "src" / "parts" / "juju_parser.cpp.inc").read_text()
    assert 'const std::string expected_weight = moe_juju_path_basename_local(juju_path);' in parser_text
    assert 'moe_juju_path_basename_local(weight_file) == expected_weight' in parser_text
    assert 'moe_juju_path_basename_local(index_file) == expected_weight + ".idx"' in parser_text
    assert 'const std::string sidecar_path = path + ".idx";' in parser_text
    assert 'malformed authoritative sidecar' in parser_text
    sidecar_pos = parser_text.index('const int sidecar_present = file_size_bytes(sidecar_path, &sidecar_bytes);')
    reject_pos = parser_text.index('malformed authoritative sidecar')
    embedded_pos = parser_text.index('moe_read_juju_json_section(path, entries')
    assert sidecar_pos < reject_pos < embedded_pos


def test_exact_ppl_builds_runtime_plan_before_readiness_check():
    eval_text = (ROOT / "moe_engine" / "src" / "parts" / "generation_eval.cpp.inc").read_text()
    entry_pos = eval_text.index('int moe_pc_engine_eval_token_ids_from')
    runtime_pos = eval_text.index('moe_cpu_storage_ensure_runtime(engine);', entry_pos)
    ready_pos = eval_text.index('moe_eval_exact_ppl_contract_ready(engine, out_stats)', entry_pos)
    assert runtime_pos < ready_pos


def test_gguf_byte_diagnostics_allows_large_declared_source_alignment_padding():
    diag = mat.gguf_tensor_byte_diagnostics([
        {
            "name": "blk.1.layer_output_scale.weight",
            "type": 0,
            "shape": [1],
            "exact_bytes": 4,
            "source_storage_bytes": 512,
            "bytes": 4,
            "source_padding_bytes": 508,
            "source_alignment_bytes": 512,
        }
    ])
    assert diag["mismatch_count"] == 0
    assert diag["alignment_padding_ok"] == 1


def test_cpp_juju_sidecar_presence_is_fail_closed_even_when_unreadable():
    parser_text = (ROOT / "moe_engine" / "src" / "parts" / "juju_parser.cpp.inc").read_text()
    assert "const int sidecar_present = file_size_bytes(sidecar_path, &sidecar_bytes);" in parser_text
    assert "unreadable authoritative sidecar" in parser_text
    sidecar_present_pos = parser_text.index("const int sidecar_present = file_size_bytes(sidecar_path, &sidecar_bytes);")
    unreadable_pos = parser_text.index("unreadable authoritative sidecar")
    embedded_pos = parser_text.index("moe_read_juju_json_section(path, entries")
    assert sidecar_present_pos < unreadable_pos < embedded_pos

def test_cpp_juju_tensor_index_is_fail_closed_on_malformed_records():
    parser_text = (ROOT / "moe_engine" / "src" / "parts" / "juju_parser.cpp.inc").read_text()
    assert "malformed JUJU tensor index rejected" in parser_text
    assert "JUJU runtime tensor count mismatch" in parser_text
    assert "source tensor_count differs from runtime objects" in parser_text
    assert "missing/invalid required name, juju_offset, or juju_bytes" in parser_text
    assert "missing codec registry/type contract" in parser_text
    assert "missing/invalid dims or shape" in parser_text
    assert "declared_runtime_tensor_count != tensor_object_count" in parser_text


def test_juju_runtime_loader_rejects_bad_payload_ranges_not_skip():
    parser_text = (ROOT / "moe_engine" / "src" / "parts" / "juju_parser.cpp.inc").read_text()
    main_parse_text = (ROOT / "moe_engine" / "src" / "parts" / "codec" / "juju_main_parsing.cpp.inc").read_text()
    assert "moe_juju_tensor_payload_range_valid" in parser_text
    assert "JUJU tensor index payload range rejected" in main_parse_text
    assert "JUJU tensor index payload range rejected during probe" in main_parse_text
    rejected_pos = main_parse_text.index("JUJU tensor index payload range rejected")
    return_pos = main_parse_text.index("return 0;", rejected_pos)
    assert rejected_pos < return_pos


def test_juju_runtime_loader_is_fail_closed_on_partial_load_and_duplicate_slots():
    main_parse_text = (ROOT / "moe_engine" / "src" / "parts" / "codec" / "juju_main_parsing.cpp.inc").read_text()
    assert "malformed JUJU artifact rejected during table read" in main_parse_text
    assert "missing/unreadable JUJU tensor index rejected" in main_parse_text
    assert "empty JUJU tensor index rejected during probe" in main_parse_text
    assert "empty JUJU tensor index rejected during load" in main_parse_text
    assert "too many JUJU tensor paths; refusing partial index load" in main_parse_text
    assert "JUJU bundle-native tensor has invalid member projection" in main_parse_text
    assert "JUJU expert tensor bytes are not divisible by expert_count" in main_parse_text
    assert "JUJU expert tensor math shape exceeds runtime slot width" in main_parse_text
    assert "moe_juju_claim_tensor_slot_unique" in main_parse_text
    assert "invalid JUJU tensor slot" in main_parse_text
    assert "duplicate JUJU tensor slot rejected" in main_parse_text
    claim_pos = main_parse_text.index("moe_juju_claim_tensor_slot_unique(engine, rec, entry.name.c_str())")
    push_pos = main_parse_text.index("engine->tensors.push_back(std::move(rec));", claim_pos)
    assert claim_pos < push_pos


def test_juju_aux_scale_helpers_skip_executable_projection_rejection():
    main_parse_text = (ROOT / "moe_engine" / "src" / "parts" / "codec" / "juju_main_parsing.cpp.inc").read_text()
    assert "moe_juju_tensor_name_is_aux_scale" in main_parse_text
    assert '".scale"' in main_parse_text
    assert '".scale2"' in main_parse_text
    assert '".scale4"' in main_parse_text
    helper_pos = main_parse_text.index("moe_juju_tensor_name_is_aux_scale(entry.name)")
    reject_pos = main_parse_text.index("JUJU expert tensor missing executable layer/projection contract")
    assert helper_pos < reject_pos


def test_juju_model_root_scan_rejects_invalid_juju_artifact_instead_of_skipping():
    scan_text = (ROOT / "moe_engine" / "src" / "parts" / "model_scan.cpp.inc").read_text()
    bad_read_pos = scan_text.index("if (!moe_read_offload_juju_file(path, &file))")
    reject_pos = scan_text.index("invalid JUJU artifact rejected during model scan", bad_read_pos)
    return_pos = scan_text.index("return 0;", reject_pos)
    push_pos = scan_text.index("out->files.push_back(std::move(file));", bad_read_pos)
    assert bad_read_pos < reject_pos < return_pos < push_pos


def test_juju_section_table_rejects_overlaps_and_duplicate_singleton_json_sections():
    parser_text = (ROOT / "moe_engine" / "src" / "parts" / "juju_parser.cpp.inc").read_text()
    assert "moe_juju_section_type_is_singleton" in parser_text
    assert "std::unordered_set<uint32_t> singleton_section_types" in parser_text
    assert "!singleton_section_types.insert(entry.type).second" in parser_text
    assert "std::vector<moe_juju_section_range_check_t> section_ranges" in parser_text
    assert "section_ranges[i].begin < section_ranges[i - 1u].end" in parser_text


def test_mixed_head_dim_qkv_config_does_not_free_all_persistent_head_states():
    qkv_text = (ROOT / "moe_engine" / "src" / "parts" / "raw_forward_qkv.cpp.inc").read_text()
    apply_pos = qkv_text.index("static void moe_qkv_apply_contract_config_locked")
    prepare_pos = qkv_text.index("static int moe_qkv_prepare_runtime_config", apply_pos)
    apply_body = qkv_text[apply_pos:prepare_pos]
    assert "Do not clear the whole persistent KV cache" in apply_body
    assert "moe_qkv_free_head_states_locked(engine);" not in apply_body
    assert "moe_qkv_layer_head_fingerprint(" in qkv_text
    assert "head_dim," in qkv_text[qkv_text.index("moe_qkv_layer_head_fingerprint("):qkv_text.index("static qkv_state_t* moe_qkv_ensure_layer_head_state_locked")]
    assert "engine->qkv_head_state_fingerprints[slot_index] == desired_fingerprint" in qkv_text


def test_qkv_decode_requires_appended_tokens_not_just_capacity():
    state_text = (ROOT / "moe_engine" / "src" / "parts" / "engine_state.cpp.inc").read_text()
    qkv_text = (ROOT / "moe_engine" / "src" / "parts" / "raw_forward_qkv.cpp.inc").read_text()
    assert "qkv_head_state_filled_tokens" in state_text
    assert "qkv_head_state_filled_tokens.resize(slot_count, 0)" in qkv_text
    assert "qkv_head_state_filled_tokens.clear()" in qkv_text
    assert "token_index + 1u" in qkv_text
    assert "engine->qkv_head_state_filled_tokens[slot_index] < context_tokens" in qkv_text
    decode_pos = qkv_text.index("static int moe_pc_engine_attention_decode_layer_head_qkv_f32")
    filled_pos = qkv_text.index("engine->qkv_head_state_filled_tokens[slot_index] < context_tokens", decode_pos)
    qkv_call_pos = qkv_text.index("qkv_attention_decode(query, slot", decode_pos)
    assert filled_pos < qkv_call_pos


def test_mxfp4_dot_and_cache_decode_use_physical_row_stride_from_index():
    kernel_text = (ROOT / "moe_engine" / "src" / "parts" / "tensor_kernels" / "dot_q8q4q5_kernels.cpp.inc").read_text()
    tensor_dot_text = (ROOT / "moe_engine" / "src" / "parts" / "tensor_dot.cpp.inc").read_text()
    raw_ops_text = (ROOT / "moe_engine" / "src" / "parts" / "raw_forward" / "forward_ops.cpp.inc").read_text()
    cache_text = (ROOT / "moe_engine" / "src" / "parts" / "raw_forward" / "forward_cache.cpp.inc").read_text()
    assert "moe_mxfp4_row_layout_for_bytes" in kernel_text
    assert "moe_mxfp4_row_bytes_for_block_cols(cols, 16u)" in kernel_text
    assert "layout.bytewise_codes ? (qs[index] & 0x0fu)" in kernel_text
    assert "moe_dot_gguf_mxfp4_row_strided" in kernel_text
    assert "moe_dot_gguf_mxfp4_two_rows_strided" in kernel_text
    assert "moe_dot_gguf_mxfp4_row_strided(packed, x, rec->info.cols, rec->weight_row_bytes)" in tensor_dot_text
    assert "moe_dot_gguf_mxfp4_row_strided(packed, x, view->cols, view->row_bytes)" in tensor_dot_text
    assert "moe_dot_gguf_mxfp4_two_rows_strided(pa, pb, x, a->cols, a->row_bytes, b->row_bytes" in tensor_dot_text
    assert "moe_dot_gguf_mxfp4_row_strided(row_ptr, hidden, hidden_count, row_bytes)" in raw_ops_text
    assert "moe_mxfp4_row_layout_for_bytes(out_count, row_bytes)" in cache_text


def test_qkv_single_token_attention_uses_paper_quantized_decode_path():
    attn_text = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "attention_decode.cpp.inc").read_text()
    assert "TurboQuant/QJL KV mode must use the same quantize/dequantize contract" in attn_text
    assert "sink_exact=%d current_exact=%d" in attn_text
    assert "qkv_sink_exact_current" in attn_text
    assert "qkv_current_value_exact" not in attn_text
    assert "if (qkv_current_value_exact && seq_len == 1u)" not in attn_text
    assert "const float* v = qkv_current_value_exact" not in attn_text
    append_pos = attn_text.index("moe_qkv_append_layer_head_token")
    fill_pos = attn_text.index("std::fill(s->attn_value.begin()", append_pos)
    qkv_decode_pos = attn_text.index("moe_pc_engine_attention_decode_layer_head_qkv_f32", fill_pos)
    assert append_pos < fill_pos < qkv_decode_pos


def test_rmsnorm_unit_offset_is_contract_driven_not_weight_stats():
    raw_ops_text = (ROOT / "moe_engine" / "src" / "parts" / "raw_forward_ops.cpp.inc").read_text()
    assert "unit-offset RMSNorm is architecture/config math" in raw_ops_text
    assert "Use only explicit metadata/GraphIR" in raw_ops_text
    assert "weight_stats_override_metadata_false" not in raw_ops_text
    assert "metadata_false_stats_inconclusive" not in raw_ops_text
    assert "mean_abs_raw < 0.75" not in raw_ops_text
    assert "raw_rms < 0.75" not in raw_ops_text
    assert "moe_rmsnorm_weight_implausible_sidecar_f32" not in raw_ops_text
    assert "moe_rmsnorm_unit_offset_from_json(engine->offload_gguf_metadata_json" in raw_ops_text
    assert "moe_rmsnorm_unit_offset_from_json(engine->offload_graph_ir_json" in raw_ops_text
    assert "moe_engine_contract_uses_rmsnorm_unit_offset(engine)" in raw_ops_text
    assert "moe_engine_contract_uses_direct_rmsnorm_weight(engine)" in raw_ops_text


def test_router_contract_reads_graph_ir_sigmoid_scale_and_norm_topk():
    router_text = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "router_topk.cpp.inc").read_text()
    assert "&engine->offload_graph_ir_json" in router_text
    assert "router_scoring_func" in router_text
    assert "router_score_function" in router_text
    assert "router_score_contract" in router_text
    assert "moe_router_json_get_double_any" in router_text
    assert "router_routed_scaling_factor" in router_text
    assert "moe_router_effective_routed_scaling_factor" in router_text
    assert "moe_router_json_get_bool_any(engine->offload_graph_ir_json" in router_text
    uses_pos = router_text.index("static int moe_router_uses_sigmoid_scores")
    scale_pos = router_text.index("static float moe_router_routed_scaling_factor")
    effective_pos = router_text.index("static float moe_router_effective_routed_scaling_factor")
    norm_pos = router_text.index("static int moe_router_norm_topk_prob")
    assert uses_pos < scale_pos < effective_pos < norm_pos


def test_router_routed_scaling_factor_does_not_parse_tensor_role_scale_names():
    router_text = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "router_topk.cpp.inc").read_text()
    scale_block = router_text[router_text.index("static float moe_router_routed_scaling_factor"):router_text.index("static int moe_router_norm_topk_prob")]
    keys_block = scale_block[scale_block.index("static const char* const scale_keys[]"):scale_block.index("};", scale_block.index("static const char* const scale_keys[]"))]
    assert "routed_scaling_factor" in keys_block
    assert "router_routed_scaling_factor" in keys_block
    assert "expert_routed_scaling_factor" in keys_block
    assert '"router_scale"' not in keys_block
    assert '"moe_router_scale"' not in keys_block
    assert '"expert_scale"' not in keys_block
    assert '"route_scale"' not in keys_block
    assert '"router_route_scale"' not in keys_block
    assert '"routed_scale"' not in keys_block
    assert '"moe_router_scaling_factor"' not in keys_block
    assert '"route_scaling_factor"' not in keys_block
    assert "softmax routers whose selected top-k weights are" in scale_block
    assert "moe_router_effective_routed_scaling_factor" in scale_block
    assert "norm_topk=1 produce weights greater than 1.0" in scale_block


def test_router_scale_tensor_uses_contract_specific_unit_offset_semantics():
    router_text = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "router_utils.cpp.inc").read_text()
    assert "moe_engine_contract_uses_direct_router_input_scale(engine, layer)" in router_text
    assert "? 0" in router_text
    assert ": moe_engine_rmsnorm_unit_offset(engine, scale_raw)" in router_text
    assert "const float effective_scale = unit_offset ? (1.0f + scale) : scale;" in router_text
    assert "router_scale_apply" in router_text
    assert "this tensor is a router input RMSNorm/scale weight" in router_text


def test_router_scale_separates_split_ffn_input_scale_from_weight_sidecars():
    router_text = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "router_utils.cpp.inc").read_text()
    suffix_block = router_text[router_text.index("static int moe_map_router_scale_tensor"):router_text.index("static int moe_prepare_scaled_router_input_f32")]
    reject_block = router_text[router_text.index("static int moe_router_scale_tensor_is_weight_sidecar_name_f32"):router_text.index("static const moe_gguf_common_tensor_record* moe_find_router_scale_record_by_role_f32")]
    assert '"ffn_gate_inp.scale"' in suffix_block
    assert '"ffn_gate_inp.scales"' in suffix_block
    assert '"ffn_gate_inp.weight.scale"' in reject_block
    assert '"ffn_gate_inp.weight.scales"' in reject_block
    assert '".weight.scale"' in reject_block
    assert '".weight.scales"' in reject_block
    assert "weight_scale_sidecar" in reject_block
    assert "quant_sidecar" in reject_block
    assert "mxfp" in reject_block
    assert "reason=weight_sidecar" in router_text
    assert "router_scale_contract_reject" in router_text
    reject_pos = router_text.index("router_scale_contract_reject")
    apply_pos = router_text.index("router_scale_apply", reject_pos)
    assert reject_pos < apply_pos


def test_router_input_mode_does_not_use_mxfp_sidecar_presence():
    mlp_text = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "mlp_forward.cpp.inc").read_text()
    first_start = mlp_text.index("const int router_has_internal_norm_scale =")
    first_end = mlp_text.index("moe_save_gate_input_snapshot", first_start)
    first_block = mlp_text[first_start:first_end]
    assert '"ffn_gate_inp.scale"' not in first_block
    assert '"router_norm.weight"' in first_block
    assert "router_uses_raw_residual_contract && router_has_plausible_input_scale" not in first_block
    assert "const int router_uses_raw_residual = router_uses_raw_residual_contract;" in first_block
    assert "(router_has_internal_norm_scale || router_uses_raw_residual) ?" in first_block
    assert "router_scale_available=%d" in mlp_text




def test_materializer_router_scale_sidecar_contract_is_not_required_by_engine_patch():
    # This patch is engine-only.  The runtime must distinguish bare split-FFN
    # router input scales from explicit weight sidecars without requiring a
    # Colab/materializer change.
    router_text = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "router_utils.cpp.inc").read_text()
    helper_text = (ROOT / "moe_engine" / "src" / "parts" / "raw_forward" / "forward_helpers.cpp.inc").read_text()
    assert '"ffn_gate_inp.scale"' in router_text
    assert '"ffn_gate_inp.weight.scale"' in router_text
    assert "moe_router_scale_record_is_weight_sidecar_f32" in router_text
    assert "moe_engine_contract_uses_split_ffn_norm(engine, layer)" in helper_text

def test_post_attention_norm_no_longer_falls_back_to_ffn_norm_aliases():
    desc_text = (ROOT / "moe_engine" / "src" / "parts" / "raw_forward_tensor_desc.cpp.inc").read_text()
    suffix_block = desc_text[desc_text.index("case moe_RAW_TENSOR_POST_ATTENTION_LAYER_NORM: {"):desc_text.index("case moe_RAW_TENSOR_Q_A_LAYER_NORM:")]
    assert '"ffn_norm.weight"' not in suffix_block
    assert '"mlp_norm.weight"' not in suffix_block
    assert '"pre_ffw_norm.weight"' not in suffix_block
    assert "GraphIR-required GLM/Gemma-MoE artifacts" in suffix_block
    role_block = desc_text[desc_text.index("case moe_RAW_TENSOR_POST_ATTENTION_LAYER_NORM:"):desc_text.index("case moe_RAW_TENSOR_Q_A_LAYER_NORM:", desc_text.index("case moe_RAW_TENSOR_POST_ATTENTION_LAYER_NORM:"))]
    assert 'out->push_back("ffn_norm")' not in role_block


def test_qkv_quantization_and_dequantization_fail_closed_on_nonfinite_values():
    q_text = (ROOT / "engine_core" / "kv" / "qkv_quantize.cpp").read_text()
    d_text = (ROOT / "engine_core" / "kv" / "qkv_dequantize.cpp").read_text()
    assert "return 0;" in q_text[q_text.index("if (!std::isfinite(input[i]))"):q_text.index("l2_norm += input[i]")]
    assert "silently store a zero vector" in q_text
    assert "fail closed instead of replacing the component with zero" in d_text
    assert "return 0;" in d_text[d_text.index("if (!std::isfinite(norm))"):d_text.index("if (norm < 1e-12f)")]
    assert "biased MSE-only reconstruction" in d_text


def test_shared_expert_suffixes_are_preferred_before_generic_dense_ffn_aliases():
    mlp_norm_text = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "mlp_normalization.cpp.inc").read_text()
    shared_block = mlp_norm_text[mlp_norm_text.index("static int moe_layer_shared_expert_dense_mlp_f32("):mlp_norm_text.index("static int moe_layer_dense_mlp_f32(")]
    generic_block = mlp_norm_text[mlp_norm_text.index("static int moe_layer_common_dense_mlp_f32("):mlp_norm_text.index("static int moe_layer_shared_expert_dense_mlp_f32(")]
    assert '"shared_expert.gate_proj.weight"' in shared_block
    assert '"shared_expert.gate_proj.weight"' not in generic_block
    assert '"ffn_gate.weight"' in generic_block


def test_attention_uses_unit_qk_norm_contract_scale_in_engine():
    attn_text = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "attention_prefill.cpp.inc").read_text()
    helper_text = (ROOT / "moe_engine" / "src" / "parts" / "raw_forward" / "forward_helpers.cpp.inc").read_text()
    assert "moe_engine_contract_uses_unit_qk_norm_global" in helper_text
    assert "moe_engine_is_gemma4_text_contract" not in helper_text
    assert "unit_qk_norm" in attn_text
    assert "resolved_source = 4" in attn_text
    assert "resolved_value = 1.0f" in attn_text
    contract_pos = attn_text.index("moe_layer_uses_unit_qk_norm_contract(engine, layer)")
    qpre_pos = attn_text.index("query_pre_attn_scalar > 0.0f", contract_pos)
    assert contract_pos < qpre_pos






def test_engine_runtime_contracts_do_not_use_model_name_metadata_keys():
    helper_text = (ROOT / "moe_engine" / "src" / "parts" / "raw_forward" / "forward_helpers.cpp.inc").read_text()
    attn_text = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "attention_prefill.cpp.inc").read_text()
    mla_text = (ROOT / "moe_engine" / "src" / "parts" / "generation_mla_constants.cpp.inc").read_text()
    router_text = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "router_utils.cpp.inc").read_text()
    for text in (helper_text, attn_text, mla_text, router_text):
        assert "moe_engine_is_gemma4_text_contract" not in text
        assert "gemma4." not in text
    assert '"full_rope_theta"' in attn_text
    assert '"sliding_rope_theta"' in attn_text

def test_contract_helpers_cover_remaining_non_name_runtime_axes():
    helper_text = (ROOT / "moe_engine" / "src" / "parts" / "raw_forward" / "forward_helpers.cpp.inc").read_text()
    raw_ops_text = (ROOT / "moe_engine" / "src" / "parts" / "raw_forward" / "forward_ops.cpp.inc").read_text()
    gen_text = (ROOT / "moe_engine" / "src" / "parts" / "generation_forward.cpp.inc").read_text()
    profile_text = (ROOT / "moe_engine" / "src" / "parts" / "profile_trace.cpp.inc").read_text()
    assert "moe_engine_contract_uses_direct_rmsnorm_weight" in helper_text
    assert "moe_engine_contract_uses_rmsnorm_unit_offset" in helper_text
    assert "moe_engine_contract_uses_gelu_tanh_mlp" in helper_text
    assert "moe_engine_is_gemma4_text_contract" not in helper_text
    assert "gelu_tanh_mlp_contract" in raw_ops_text
    assert "direct_rmsnorm=%d rmsnorm_unit_offset=%d gelu_tanh_mlp=%d" in gen_text
    assert '"activation_mode"' in profile_text
    assert '"rmsnorm_unit_offset"' in profile_text
    assert '"router_scale_apply"' in profile_text
    assert '"mlp_moe_router"' in profile_text
    assert '"mlp_moe_gate_up"' in profile_text

def test_routed_graph_suppresses_dense_fallback_even_with_split_ffn_aliases():
    mlp_text = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "mlp_forward.cpp.inc").read_text()
    assert "moe_layer_dense_mlp_should_run_f32" in mlp_text
    helper = mlp_text[mlp_text.index("static int moe_layer_has_common_dense_mlp_tensors_f32"):mlp_text.index("static int moe_layer_dense_mlp_should_run_f32")]
    block = mlp_text[mlp_text.index("static int moe_layer_dense_mlp_should_run_f32"):mlp_text.index("static int moe_layer_moe_mlp_f32")]
    assert '"ffn_gate.weight"' in helper
    assert '"ffn_up.weight"' in helper
    assert '"ffn_down.weight"' in helper
    assert "moe_engine_contract_uses_split_ffn_norm(engine, layer) &&" not in block
    assert "moe_layer_has_common_dense_mlp_tensors_f32(engine, layer)" not in block
    assert '"moe_expert_mlp"' in block
    assert "GraphIR routed roles own the FFN branch" in block
    routed_pos = block.index('"moe_expert_mlp"')
    suppress_pos = block.index("return 0;", routed_pos)
    dense_pos = block.index('"dense_mlp"')
    assert routed_pos < suppress_pos < dense_pos
    assert "split-FFN router input scale is required" in mlp_text


def test_ple_vocab_masking_allows_smaller_per_layer_vocab():
    mlp_common = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "mlp_common_tensors.cpp.inc").read_text()
    assert "token_vocab_size" in mlp_common
    assert "token_id < ple.token_vocab_size" in mlp_common
    assert "std::fill(s->ple_inputs.begin(), s->ple_inputs.end(), 0.0f)" in mlp_common
    assert "token_desc.rows == 0" in mlp_common


def test_expert_down_scale_is_legacy_runtime_scale_but_not_direct_contract_scale():
    norm_text = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "mlp_normalization.cpp.inc").read_text()
    fwd_text = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "mlp_forward.cpp.inc").read_text()
    for text in (norm_text, fwd_text):
        assert "direct_contract_suffixes" in text
        direct_block = text[text.index("static const char* const direct_contract_suffixes[]"):
                            text.index("const int direct_scale_contract", text.index("static const char* const direct_contract_suffixes[]"))]
        assert '"ffn_down_exps.scale"' not in direct_block
        assert '"per_expert_scale"' in direct_block
        assert '"experts.per_expert_scale"' in direct_block
        assert "moe_engine_contract_uses_direct_per_expert_scale(engine, layer)" in text
        legacy_scale_pos = text.index('"ffn_down_exps.scale"')
        direct_pos = text.index("static const char* const direct_contract_suffixes[]")
        assert legacy_scale_pos < direct_pos
        assert "runtime expert-down branch scale" in text


def test_raw_residual_router_input_is_split_ffn_or_explicit_graph_contract():
    norm_text = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "mlp_normalization.cpp.inc").read_text()
    block = norm_text[norm_text.index("static int moe_layer_router_uses_raw_residual_input"):norm_text.index("static int moe_layer_common_dense_mlp_f32")]
    assert "moe_engine_contract_uses_split_ffn_norm(engine, layer)" in block
    assert '"ffn_gate_inp.weight"' in block
    assert "moe_graph_ir_math_required(engine)" in block
    assert '"moe_router"' in block

def test_no_v_proj_contract_allows_k_equals_v_without_qkv_weight_changes():
    attn_text = (ROOT / "moe_engine/src/parts/generation/attention_decode.cpp.inc").read_text()
    allow_pos = attn_text.index("const int allow_v_equals_k")
    block = attn_text[allow_pos:attn_text.index("if (qkv_parallel_path)", allow_pos)]
    assert '"no_v_proj"' in block
    assert '"layers_with_no_v_proj"' in block
    assert '"no_value_projection"' in block
    assert "!has_v && moe_engine_contract_allows_missing_v_projection(engine, layer)" in block
    assert "QKV" not in block

def test_dense_branch_uses_graph_role_not_common_alias_presence():
    mlp_text = (ROOT / "moe_engine/src/parts/generation/mlp_forward.cpp.inc").read_text()
    helper_pos = mlp_text.index("static int moe_layer_has_common_dense_mlp_tensors_f32")
    helper_block = mlp_text[helper_pos:mlp_text.index("static int moe_layer_dense_mlp_should_run_f32", helper_pos)]
    for suffix in (
        '"ffn_gate.weight"', '"mlp.gate_proj.weight"', '"gate_proj.weight"',
        '"ffn_up.weight"', '"mlp.up_proj.weight"', '"up_proj.weight"',
        '"ffn_down.weight"', '"mlp.down_proj.weight"', '"down_proj.weight"',
    ):
        assert suffix in helper_block
    should_pos = mlp_text.index("static int moe_layer_dense_mlp_should_run_f32")
    should_block = mlp_text[should_pos:mlp_text.index("static int moe_layer_moe_mlp_f32", should_pos)]
    assert "moe_layer_has_common_dense_mlp_tensors_f32(engine, layer)" not in should_block
    assert '"dense_mlp"' in should_block
    assert '"dense_ffn"' in should_block
    assert '"moe_expert_mlp"' in should_block
    assert "return moe_graph_ir_layer_has_op_role_any" in should_block



def test_common_ffn_shared_branch_requires_explicit_shared_contract():
    mlp_text = (ROOT / "moe_engine/src/parts/generation/mlp_forward.cpp.inc").read_text()
    assert "moe_layer_common_ffn_is_shared_branch_f32" in mlp_text
    helper_pos = mlp_text.index("static int moe_layer_common_ffn_is_shared_branch_f32")
    helper_block = mlp_text[helper_pos:mlp_text.index("static int moe_layer_dense_mlp_should_run_f32", helper_pos)]
    assert "moe_layer_graph_shared_expert_declared_f32(engine, layer)" in helper_block
    assert "!moe_layer_graph_shared_expert_declared_f32(engine, layer)" in helper_block
    assert "moe_engine_contract_uses_split_ffn_norm(engine, layer)" not in helper_block
    assert "moe_layer_has_post_ffw_norm1_weight(engine, layer)" in helper_block
    assert "moe_layer_has_common_dense_mlp_tensors_f32(engine, layer)" in helper_block
    assert "common_enabled=%d common_ok=%d" in mlp_text
    assert "common FFN shared branch could not be mapped" in mlp_text


def test_embedding_scale_is_cast_to_embedding_tensor_dtype_not_model_name():
    ops_text = (ROOT / "moe_engine/src/parts/raw_forward/forward_ops.cpp.inc").read_text()
    emb_text = (ROOT / "moe_engine/src/parts/raw_forward_embedding.cpp.inc").read_text()
    assert "moe_round_f32_to_bf16_value" in ops_text
    assert "moe_round_f32_to_fp16_value" in ops_text
    assert "moe_cast_embedding_scale_to_tensor_dtype" in ops_text
    assert "scalar_encoding == moe_RAW_SCALAR_BF16" in ops_text
    assert "weight_format == moe_WEIGHT_ENCODING_RAW_BF16" in ops_text
    assert "scalar_encoding == moe_RAW_SCALAR_F16" in ops_text
    assert "weight_format == moe_WEIGHT_ENCODING_RAW_FP16" in ops_text
    assert "cast=%s applied_scale=%.9g base_scale=%.9g scalar=%u format=%u hidden=%u" in ops_text
    assert "moe_engine_is_gemma4_text_contract" not in ops_text
    assert "gemma4" not in ops_text.lower()
    assert "moe_apply_embedding_scale_f32(engine, c, out_hidden, common.scalar_encoding, common.weight_format)" in emb_text
    assert "moe_apply_embedding_scale_f32(engine, c, dst, common.scalar_encoding, common.weight_format)" in emb_text
    assert "moe_apply_embedding_scale_f32(engine, c, out_hidden, encoding, weight_format)" in emb_text
