import importlib.util
import math
import sys
import types
from pathlib import Path

try:
    import requests  # noqa: F401
except ModuleNotFoundError:
    sys.modules["requests"] = types.ModuleType("requests")

ROOT = Path(__file__).resolve().parents[2]


def _resolve_materializer_path():
    candidates = [
        ROOT / "colab" / "juju_shard_materializer.py",
        ROOT / "colab" / "juju_shard_materializer_PPL_FastRegen.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "missing JUJU materializer source; expected one of: "
        + ", ".join(str(x) for x in candidates)
    )


MAT_PATH = _resolve_materializer_path()
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


def test_qk_norm_layer_contract_declares_unit_attention_scale():
    tensors = [
        {"name": "blk.0.attn_q.weight"},
        {"name": "blk.0.attn_k.weight"},
        {"name": "blk.0.attn_v.weight"},
        {"name": "blk.0.attn_output.weight"},
        {"name": "blk.0.attn_q_norm.weight"},
        {"name": "blk.0.attn_k_norm.weight"},
        {"name": "blk.0.attn_v_norm.weight"},
    ]
    arch = {
        "head_dim": 256,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "query_pre_attn_scalar": 256.0,
        "attention_scale": 0.0625,
        "attention_scale_source": "head_dim",
    }
    row = mat._juju_layer_execution_contract_table(tensors, arch)[0]
    attn_row = mat._juju_attention_layer_contract_table(tensors, arch)[0]
    assert row["attention"]["unit_attention_scale"] is True
    assert row["attention"]["attention_scale"] == 1.0
    assert row["attention"]["attention_scale_source"] == "qk_norm_contract"
    assert attn_row["unit_attention_scale"] is True
    assert attn_row["attention_scale"] == 1.0
    assert attn_row["attention_scale_source"] == "qk_norm_contract"
    assert router_select.get("scale", []) == []


def test_gemma4_dual_dense_moe_branch_is_not_serialized_as_fallback_only():
    tensors = [
        {"name": "blk.0.ffn_norm.weight"},
        {"name": "blk.0.pre_ffw_norm_2.weight"},
        {"name": "blk.0.ffn_gate_inp.weight"},
        {"name": "blk.0.ffn_gate.weight"},
        {"name": "blk.0.ffn_up.weight"},
        {"name": "blk.0.ffn_down.weight"},
        {"name": "blk.0.ffn_gate_exps.weight"},
        {"name": "blk.0.ffn_up_exps.weight"},
        {"name": "blk.0.ffn_down_exps.weight"},
        {"name": "blk.0.post_ffw_norm_1.weight"},
        {"name": "blk.0.post_ffw_norm_2.weight"},
        {"name": "blk.0.post_ffw_norm.weight"},
    ]
    graph = mat.build_layer_graph_ir(0, tensors, {"num_experts": 128, "top_k_experts": 8})
    dense = next(op for op in graph["ops"] if op["op"] == "dense_mlp")
    assert dense["name"] == "dense_ffn_primary"
    assert dense["required"] is True
    assert dense["forbid_parallel_with_routed_moe"] is False
    assert "execute_only_when_no_moe" not in dense["fallback_semantics"]

    mlp_text = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "mlp_forward.cpp.inc").read_text(encoding="utf-8", errors="ignore")
    assert "structural_dense_moe_branch" in mlp_text
    assert "plan.routed.has_weights && !split_dense_branch_by_norm1" in mlp_text


def test_hidden_sized_ffn_gate_input_scale_is_router_weight_sidecar_contract():
    tensors = [
        {"name": "blk.0.ffn_gate_inp.weight", "shape": [2816, 128]},
        {"name": "blk.0.ffn_gate_inp.scale", "shape": [2816]},
        {"name": "blk.0.ffn_gate_exps.weight"},
    ]
    graph = mat.build_layer_graph_ir(0, tensors, {"hidden_size": 2816, "num_experts": 128, "top_k_experts": 8})
    router_select = next(op for op in graph["ops"] if op["name"] == "router_input")
    router_linear = next(op for op in graph["ops"] if op["name"] == "moe_router")
    assert router_select["scale"] == []
    assert router_select["weight_scale_sidecars"] == ["blk.0.ffn_gate_inp.scale"]
    assert router_linear["scale"] == []
    assert router_linear["weight_scale_sidecars"] == ["blk.0.ffn_gate_inp.scale"]

    router_scale_text = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "router_scale_inputs.cpp.inc").read_text(encoding="utf-8", errors="ignore")
    assert "legacy_weight_sidecar_as_input_scale" not in router_scale_text
    assert "weight_sidecar_dense_raw_rms_root" in router_scale_text
    assert "hidden[i] * norm * scale" in router_scale_text



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
    assert 'lower_source.find("head_dim")' not in parser_text
    assert 'has_q_norm_contract' not in parser_text
    extract_fn = parser_text[parser_text.index("static int moe_juju_extract_attention_contract"):parser_text.index("static std::string moe_juju_attention_contract_compact_json")]
    assert extract_fn.index("moe_juju_attention_contract_uses_unit_scale") < extract_fn.index("expected_attention_scale")
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
    assert "model_config_attention_unit_scale" in state_text
    assert "moe_eval_exact_ppl_contract_ready" in eval_text
    assert "JUJU exact_ppl_mode required_features contract is not loaded" in eval_text
    assert "JUJU attention scale contract is not loaded" in eval_text
    assert "JUJU attention scale does not match runtime attention contract" in eval_text
    assert "Non-unit numeric attention-scale contracts are checked" in eval_text
    assert "moe_engine_contract_uses_unit_qk_norm_global(engine)" not in eval_text
    assert "JUJU storage format plan is not available for exact PPL" in eval_text
    assert "JUJU expert index is not ready for exact PPL" in eval_text
    assert "selected-expert linear fallback was used" in eval_text
    assert "non-finite hidden state" in eval_text
    assert "non-finite lm_head logprob" in eval_text
    assert "moe_juju_required_feature_present(json, \"exact_ppl_mode\")" in parser_text
    assert "engine->model_config_attention_scale = (float)attention_scale" in parser_text
    assert "unit_attention_scale" in parser_text


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
    kernel_text = (ROOT / "moe_engine" / "src" / "parts" / "tensor_kernels" / "dot_q8q4q5_kernels.cpp.inc").read_text(encoding="utf-8")
    tensor_dot_text = (ROOT / "moe_engine" / "src" / "parts" / "tensor_dot.cpp.inc").read_text(encoding="utf-8")
    raw_ops_text = (ROOT / "moe_engine" / "src" / "parts" / "raw_forward" / "forward_ops.cpp.inc").read_text(encoding="utf-8")
    cache_text = (ROOT / "moe_engine" / "src" / "parts" / "raw_forward" / "forward_cache.cpp.inc").read_text(encoding="utf-8")
    materializer_text = (ROOT / "colab" / "juju_shard_materializer_PPL_FastRegen.py").read_text(encoding="utf-8")
    assert "moe_mxfp4_row_layout_for_bytes" in kernel_text
    assert "Standard GGML MXFP4 is a 32-value block" in kernel_text
    assert "low nibbles decode logical columns 0..15" in kernel_text
    assert '"split16_low_nibbles_first_half_high_nibbles_second_half"' in materializer_text
    assert "index row stride is the only reliable runtime discriminator" in kernel_text
    assert "moe_mxfp4_row_bytes_for_block_cols(cols, 16u)" in kernel_text
    assert "layout.block_cols = 16u" in kernel_text
    assert "layout.bytewise_codes = 1u" in kernel_text
    assert "layout.bytewise_codes = 0u" in kernel_text
    assert "moe_dot_gguf_mxfp4_row_strided" in kernel_text
    assert "moe_dot_gguf_mxfp4_two_rows_strided" in kernel_text
    assert "moe_dot_gguf_mxfp4_row_strided(packed, x, rec->info.cols, rec->weight_row_bytes)" in tensor_dot_text
    assert "moe_dot_gguf_mxfp4_row_strided(packed, x, view->cols, view->row_bytes)" in tensor_dot_text
    assert "moe_dot_gguf_mxfp4_two_rows_strided(pa, pb, x, a->cols, a->row_bytes, b->row_bytes" in tensor_dot_text
    assert "moe_dot_gguf_mxfp4_row_strided(row_ptr, hidden, hidden_count, row_bytes)" in raw_ops_text
    assert "moe_mxfp4_row_layout_for_bytes(out_count, row_bytes)" in cache_text


def test_mxfp4_e8m0_decode_uses_ggml_half_scale():
    kernel_text = (ROOT / "moe_engine" / "src" / "parts" / "tensor_kernels" / "dot_q8q4q5_kernels.cpp.inc").read_text(encoding="utf-8")
    model_lib_text = (ROOT / "moe_engine" / "src" / "parts" / "model_library_source.cpp.inc").read_text(encoding="utf-8")
    expected = "v < 2u ? (0x00200000u << v) : ((uint32_t)(v - 1u) << 23u)"
    assert expected in kernel_text
    assert expected in model_lib_text
    assert "v ? ((uint32_t)v << 23u) : 0x00400000u" not in kernel_text
    assert "v ? ((uint32_t)v << 23u) : 0x00400000u" not in model_lib_text


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
    raw_ops_text = (ROOT / "moe_engine" / "src" / "parts" / "raw_forward_ops.cpp.inc").read_text(encoding="utf-8", errors="ignore")
    assert "An explicit metadata/GraphIR" in raw_ops_text
    assert "false must not be undone" in raw_ops_text
    helper_text = (ROOT / "moe_engine" / "src" / "parts" / "raw_forward" / "forward_helpers.cpp.inc").read_text(encoding="utf-8", errors="ignore")
    assert "moe_engine_contract_uses_rmsnorm_unit_offset_tensor_contract" in helper_text
    assert "RMSNorm weight semantics are not inferable from tensor topology" in helper_text
    assert "return 0;" in helper_text[helper_text.index("moe_engine_contract_uses_rmsnorm_unit_offset_tensor_contract"):helper_text.index("moe_engine_contract_uses_direct_rmsnorm_weight")]
    assert "\"rmsnorm_unit_offset\": false" in helper_text
    assert "saw_explicit_unit_false" in helper_text
    assert "return unit_offset ? 1 : 0" not in helper_text
    assert "weight_stats_override_metadata_false" not in raw_ops_text
    assert "metadata_false_stats_inconclusive" not in raw_ops_text
    assert "mean_abs_raw < 0.75" not in raw_ops_text
    assert "raw_rms < 0.75" not in raw_ops_text
    assert "moe_rmsnorm_weight_implausible_sidecar_f32" not in raw_ops_text
    assert "moe_rmsnorm_unit_offset_from_json(engine->offload_gguf_metadata_json" in raw_ops_text
    assert "moe_rmsnorm_unit_offset_from_json(engine->offload_graph_ir_json" in raw_ops_text
    assert "moe_engine_contract_uses_rmsnorm_unit_offset(engine)" in raw_ops_text
    assert "moe_engine_contract_uses_direct_rmsnorm_weight(engine)" in raw_ops_text
    assert "cached_rmsnorm_unit_offset" in raw_ops_text
    assert "moe_engine_contract_model_family_uses_rmsnorm_unit_offset" in helper_text
    assert 'moe_ascii_contains_ci(engine->dynamic_architecture, "gemma")' in helper_text
    assert 'moe_ascii_contains_ci(engine->model_root, "gemma")' in helper_text
    assert 'moe_engine_graph_ir_mentions(engine, "post_ffw_norm_1")' in helper_text
    assert 'moe_engine_graph_ir_mentions(engine, "gelu_pytorch_tanh")' in helper_text
    assert 'moe_json_get_string_local(engine->offload_graph_ir_json, key, &value)' in helper_text
    assert '"source_repo_id"' in helper_text
    assert 'moe_engine_metadata_mentions(engine, "gemma")' in helper_text
    assert "unit_offset_contract_over_metadata_false" in raw_ops_text
    false_pos = helper_text.index("if (saw_explicit_unit_false)")
    family_pos = helper_text.index("moe_engine_contract_model_family_uses_rmsnorm_unit_offset(engine)")
    assert false_pos < family_pos


def test_runtime_arch_defaults_gemma_rmsnorm_to_unit_offset_contract():
    runtime = mat.juju_runtime_arch_metadata({
        "arch_meta": {
            "model_type": "gemma4_text",
            "rms_norm_eps": 1e-6,
        }
    })
    assert runtime["rms_norm_unit_offset"] is True
    assert runtime["rmsnorm_unit_offset"] is True
    assert runtime["rmsnorm_weight_semantics"] == "unit_offset"

    explicit_direct = mat.juju_runtime_arch_metadata({
        "arch_meta": {
            "model_type": "gemma4_text",
            "rms_norm_unit_offset": False,
        }
    })
    assert explicit_direct["rms_norm_unit_offset"] is False
    assert explicit_direct["rmsnorm_unit_offset"] is False
    assert explicit_direct["rmsnorm_weight_semantics"] == "direct"


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
    assert "moe_router_has_direct_input_scale_contract_f32(engine, layer)" in router_text
    assert "const int unit_offset = direct_input_scale" in router_text
    assert ": moe_engine_rmsnorm_unit_offset(engine, scale_raw)" in router_text
    assert "const float effective_scale = unit_offset ? (1.0f + scale) : scale;" in router_text
    assert "router_scale_apply" in router_text
    assert "this tensor is a router input RMSNorm/scale weight" in router_text


def test_router_scale_rejects_bare_ffn_gate_inp_scale_weight_sidecars():
    router_text = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "router_utils.cpp.inc").read_text()
    suffix_block = router_text[router_text.index("static int moe_map_router_scale_tensor"):router_text.index("static int moe_prepare_scaled_router_input_f32")]
    reject_block = router_text[router_text.index("static int moe_router_scale_tensor_is_weight_sidecar_name_f32"):router_text.index("static const moe_gguf_common_tensor_record* moe_find_router_scale_record_by_role_f32")]
    sidecar_array = reject_block[reject_block.index("static const char* const sidecar_suffixes[]"):reject_block.index("};", reject_block.index("static const char* const sidecar_suffixes[]"))]
    assert '"ffn_gate_inp.scale"' not in suffix_block
    assert '"ffn_gate_inp.scales"' not in suffix_block
    assert '"ffn_gate_inp.scale"' in sidecar_array
    assert '"ffn_gate_inp.scales"' in sidecar_array
    assert '"ffn_gate_inp.weight.scale"' in sidecar_array
    assert '"ffn_gate_inp.weight.scales"' in sidecar_array
    assert '".weight.scale"' in sidecar_array
    assert '".weight.scales"' in sidecar_array
    assert "weight_scale_sidecar" in reject_block
    assert "quant_sidecar" in reject_block
    assert "mxfp" in reject_block
    assert "reason=weight_sidecar" in router_text
    assert "router_scale_contract_reject" in router_text
    assert "router_scale_apply" in router_text


def test_router_hidden_input_rule_does_not_accept_generic_router_scale_names():
    mlp_norm_text = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "mlp_normalization.cpp.inc").read_text()
    start = mlp_norm_text.index("static int moe_router_contract_name_is_activation_scale_f32")
    end = mlp_norm_text.index("static int moe_router_contract_object_has_activation_scale_f32", start)
    block = mlp_norm_text[start:end]
    exact_block = block[block.index("static const char* const accepted_exact[]"):block.index("};", block.index("static const char* const accepted_exact[]"))]
    suffix_block = block[block.index("static const char* const accepted_suffixes[]"):block.index("};", block.index("static const char* const accepted_suffixes[]"))]
    assert '"ffn_gate_inp.scale"' not in exact_block
    assert '"ffn_gate_inp.scales"' not in exact_block
    assert '"router.input_scale"' in exact_block
    assert '"router_norm.weight"' in exact_block
    assert '"router.scale"' not in exact_block
    assert '"mlp.router.scale"' not in exact_block
    assert '"moe.gate.scale"' not in exact_block
    assert '".router.input_scale"' in suffix_block
    assert '".router.scale"' not in suffix_block
    assert "remain ambiguous" in block


def test_mlp_router_input_scale_filter_rejects_bare_ffn_gate_inp_scale_sidecar():
    mlp_norm_text = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "mlp_normalization.cpp.inc").read_text()
    sidecar_block = mlp_norm_text[mlp_norm_text.index("static int moe_router_contract_scale_name_is_weight_sidecar_f32"):mlp_norm_text.index("static int moe_router_contract_name_is_activation_scale_f32")]
    assert '"ffn_gate_inp.scale"' in sidecar_block
    assert '"ffn_gate_inp.scales"' in sidecar_block
    assert '"ffn_gate_inp.weight.scale"' in sidecar_block
    assert '"ffn_gate_inp.weight.scales"' in sidecar_block
    assert "ffn_gate_inp.scale/scales or" in mlp_norm_text


def test_router_input_mode_uses_ffn_gate_inp_sidecar_as_weight_scale_not_activation_scale():
    mlp_text = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "mlp_forward.cpp.inc").read_text()
    router_text = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "router_utils.cpp.inc").read_text()
    logits_block = router_text[router_text.index("static int moe_router_logits_f32"):
                               router_text.index("static float moe_router_sigmoid")]
    first_start = mlp_text.index("const int router_has_internal_norm_scale =")
    first_end = mlp_text.index("moe_save_gate_input_snapshot", first_start)
    first_block = mlp_text[first_start:first_end]
    assert "moe_router_has_direct_input_scale_contract_f32(engine, layer)" in first_block
    assert '"router_norm.weight"' in first_block
    assert "moe_router_has_internal_weight_scale_contract_f32" in first_block
    assert "router_uses_raw_residual_contract && router_has_plausible_input_scale" not in first_block
    assert "router_uses_raw_residual_contract || router_has_internal_weight_scale" not in first_block
    assert "moe_layer_router_should_use_raw_residual_f32(" in first_block
    assert "(router_has_internal_norm_scale || router_uses_raw_residual) ?" in first_block
    assert "router_scale_available=%d" in mlp_text
    internal_start = router_text.index("moe_router_has_internal_weight_scale_contract_f32")
    internal_block = router_text[internal_start:router_text.index("static const moe_gguf_common_tensor_record* moe_find_router_scale_record_by_role_f32", internal_start)]
    assert '"router_weight_scale_sidecar"' in internal_block
    assert '"ffn_gate_inp.scale"' in internal_block
    assert '"ffn_gate_inp.weight.scale"' in internal_block
    assert "moe_router_scale_record_is_weight_sidecar_f32(*rec)" in internal_block
    assert '"router_input_scale"' in internal_block
    assert '"router.input_scale"' in internal_block
    assert '"router.scale"' not in internal_block
    assert '"mlp.router.scale"' not in internal_block
    assert '"moe.gate.scale"' not in internal_block
    assert "ambiguous and require an explicit op-role record" in internal_block
    assert "common_is_dense_raw_matrix" in logits_block
    assert "router_weight_sidecar_skip" in logits_block
    assert "raw_router_matrix_already_scaled" in logits_block


def test_direct_router_input_scale_is_narrower_than_router_norm_gamma():
    helper_text = (ROOT / "moe_engine" / "src" / "parts" / "raw_forward" / "forward_helpers.cpp.inc").read_text()
    start = helper_text.index("static int moe_engine_contract_uses_direct_router_input_scale")
    end = helper_text.index("static int moe_engine_contract_uses_direct_per_expert_scale", start)
    block = helper_text[start:end]
    suffix_block = block[block.index("static const char* const direct_scale_suffixes[]"):block.index("};", block.index("static const char* const direct_scale_suffixes[]"))]
    assert '"ffn_gate_inp.scale"' not in suffix_block
    assert '"ffn_gate_inp.scales"' not in suffix_block
    assert '"router_input_scale"' in suffix_block
    assert '"router.input_scale"' in suffix_block
    assert '"router_norm.weight"' not in suffix_block
    assert '"router.scale"' not in suffix_block
    assert '"mlp.router.scale"' not in suffix_block
    assert '"moe.gate.scale"' not in suffix_block
    assert "RMSNorm gamma weights" in block


def test_router_direct_input_scale_helper_accepts_explicit_roles_not_norm_roles():
    router_text = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "router_utils.cpp.inc").read_text()
    start = router_text.index("static int moe_router_has_direct_input_scale_contract_f32")
    end = router_text.index("static int moe_map_router_scale_tensor", start)
    block = router_text[start:end]
    assert '"ffn_gate_inp.scale"' not in block
    assert '"ffn_gate_inp.scales"' not in block
    assert '"router_input_scale"' in block
    assert '"moe_router_scale"' in block
    assert '"router_scale"' in block
    assert '"router_norm"' not in block
    assert '"router_norm.weight"' not in block
    assert "moe_router_scale_record_is_weight_sidecar_f32" in block



def test_router_scale_mapper_prefers_direct_activation_scale_before_norm_gamma():
    router_text = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "router_utils.cpp.inc").read_text()
    start = router_text.index("static int moe_map_router_scale_tensor")
    end = router_text.index("static int moe_router_has_contract_input_scale_f32", start)
    block = router_text[start:end]
    suffix_start = block.index("static const char* const suffixes[]")
    suffix_end = block.index("};", suffix_start)
    suffix_block = block[suffix_start:suffix_end]
    assert '"ffn_gate_inp.scale"' not in suffix_block
    assert '"ffn_gate_inp.scales"' not in suffix_block
    assert suffix_block.index('"router_input_scale"') < suffix_block.index('"router_norm.weight"')
    assert "Bare ffn_gate_inp.scale/scales are router-weight sidecars" in suffix_block
    assert "not bare direct activation scales" in suffix_block
    assert "const int direct_input_scale = moe_router_has_direct_input_scale_contract_f32(engine, layer);" in router_text
    assert "const int unit_offset = direct_input_scale" in router_text


def test_materializer_router_scale_sidecar_contract_matches_engine():
    # The materializer and engine must both treat bare ffn_gate_inp.scale as a
    # router-weight quantization sidecar, not a router-input activation scale.
    router_text = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "router_utils.cpp.inc").read_text()
    helper_text = (ROOT / "moe_engine" / "src" / "parts" / "raw_forward" / "forward_helpers.cpp.inc").read_text()
    assert '"ffn_gate_inp.scale"' in router_text
    assert '"ffn_gate_inp.weight.scale"' in router_text
    assert "moe_router_scale_record_is_weight_sidecar_f32" in router_text
    router_helper = helper_text[helper_text.index("static int moe_engine_contract_has_router_input_scale"):helper_text.index("static int moe_engine_contract_uses_direct_router_input_scale")]
    assert '"ffn_gate_inp.scale"' not in router_helper
    assert '"ffn_gate_inp.weight.scale"' not in router_helper
    assert '"router.scale"' not in router_helper[router_helper.index("static const char* const router_scale_suffixes[]"):router_helper.index("};", router_helper.index("static const char* const router_scale_suffixes[]"))]
    assert '"mlp.router.scale"' not in router_helper[router_helper.index("static const char* const router_scale_suffixes[]"):router_helper.index("};", router_helper.index("static const char* const router_scale_suffixes[]"))]
    assert '"moe.gate.scale"' not in router_helper[router_helper.index("static const char* const router_scale_suffixes[]"):router_helper.index("};", router_helper.index("static const char* const router_scale_suffixes[]"))]
    assert "graph-wide text mentions" in router_helper


def test_router_weight_sidecar_rms_root_transform_requires_explicit_ir_transform():
    router_utils_text = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "router_utils.cpp.inc").read_text()
    router_scale_text = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "router_scale_inputs.cpp.inc").read_text()
    assert "moe_router_weight_sidecar_requires_rms_root_transform_f32" in router_utils_text
    assert '"router_input_rms_root_transform"' in router_utils_text
    assert '"ffn_gate_inp_scale_is_weight_scale_sidecar"' in router_utils_text
    assert "moe_router_row_has_rms_root_graph_signature_f32" not in router_utils_text
    assert "moe_router_row_contract_implies_sidecar_rms_root_transform_f32" in router_utils_text
    sidecar_infer = router_utils_text[router_utils_text.index("static int moe_router_row_contract_implies_sidecar_rms_root_transform_f32"):router_utils_text.index("static int moe_router_weight_sidecar_requires_rms_root_transform_uncached_f32")]
    assert "Split-MoE shape alone is" in sidecar_infer
    assert "return 1;" not in sidecar_infer
    assert "moe_router_scope_declares_rms_root_input_transform_f32" in router_utils_text
    start = router_scale_text.index("static int moe_prepare_router_weight_sidecar_input_f32")
    end = router_scale_text.index("static int moe_router_has_contract_input_scale_f32", start)
    block = router_scale_text[start:end]
    assert "moe_router_weight_sidecar_requires_rms_root_transform_f32(engine, layer)" in block
    assert "ss += v * v;" in block
    assert "std::sqrt((float)(ss / (double)hidden_size) + 1.0e-6f)" in block
    assert "1.0f / std::sqrt((float)hidden_size)" in block
    assert "hidden[i] * norm * scale" in block
    assert "weight_sidecar_ir_rms_root" in block
    assert "weight_sidecar\"" in block
    assert "gemma" not in block.lower()



def test_materializer_classifies_ffn_gate_inp_scale_as_router_weight_sidecar():
    mat = MAT_PATH.read_text()
    sidecar_block = mat[mat.index('elif suffix in {"ffn_gate_inp.scale", "ffn_gate_inp.scales"}'):mat.index('elif suffix in {"router.scale", "mlp.router.scale", "moe.gate.scale"}')]
    assert '"router_weight_scale_sidecar"' in sidecar_block
    router_scale_marker = 'feature_counts["layers_with_router_scale"] += 1'
    router_scale_start = mat.rindex('if any(x in suffixes for x in {', 0, mat.index(router_scale_marker))
    router_scale_end = mat.index(router_scale_marker, router_scale_start)
    router_scale_block = mat[router_scale_start:router_scale_end]
    assert '"ffn_gate_inp.scale"' not in router_scale_block
    assert '"router.scale"' in router_scale_block


def test_graphir_dense_fallback_does_not_run_next_to_routed_moe_without_primary_branch_norm():
    fwd = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "mlp_forward.cpp.inc").read_text()
    block = fwd[fwd.index("const int dense_uses_shared_path ="):fwd.index("plan.run_dense =", fwd.index("const int dense_uses_shared_path ="))]
    assert "dense_uses_shared_path ||" in block
    assert "moe_layer_has_common_dense_mlp_tensors_f32(engine, layer)" not in block
    assert "plan.routed.has_weights && plan.dense.has_weights && plan.post_norm1.has_weights" in block


def test_graphir_dense_branch_execution_uses_split_branch_structure_not_routed_presence():
    fwd = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "mlp_forward.cpp.inc").read_text()
    materializer = MAT_PATH.read_text()
    planner = fwd[fwd.index("static moe_graph_ir_mlp_layer_plan_f32"):fwd.index("plan.run_shared =")]
    assert "execute_only_when_no_moe_or_shared_expert_weights_on_layer_unless_post_ffw_norm_1_declares_primary_dense_branch" in materializer
    assert '"forbid_parallel_with_routed_moe": bool((moe_weights or shared_expert_weights) and not (moe_weights and post_ffw_norm1_weights and not shared_expert_weights))' in materializer
    assert "if (plan.routed.present && plan.dense.has_weights && !plan.shared.has_weights)" not in planner
    assert "graphir_dense_branch_by_norm1 =" not in planner
    assert "dense_mlp + post_ffw_norm_1 as the primary non-routed FFN" in planner
    assert "dense_forbidden_with_routed" in planner
    assert "plan.dense.forbid_parallel_with_routed_moe || plan.dense.fallback_only" in planner
    assert "((plan.dense.required && plan.dense.has_weights) || split_dense_branch_by_norm1)" in planner



def test_prefetch_plan_execute_preserves_router_selected_reason():
    text = (ROOT / "moe_engine" / "src" / "parts" / "prefetch_plan_execute.cpp.inc").read_text()
    call = text[text.index("const int queued = moe_pc_engine_prefetch_expert_triplet_impl"):text.index(");", text.index("const int queued = moe_pc_engine_prefetch_expert_triplet_impl"))]
    assert "items[i].reason" in call
    assert "prefetch_is_router_selected(items[i]) ? 1 : 0" in call

def test_post_attention_norm_no_longer_falls_back_to_ffn_norm_aliases():
    desc_text = (ROOT / "moe_engine" / "src" / "parts" / "raw_forward_tensor_desc.cpp.inc").read_text()
    suffix_block = desc_text[desc_text.index("case moe_RAW_TENSOR_POST_ATTENTION_LAYER_NORM: {"):desc_text.index("case moe_RAW_TENSOR_Q_A_LAYER_NORM:")]
    assert '"ffn_norm.weight"' not in suffix_block
    assert '"mlp_norm.weight"' not in suffix_block
    assert '"pre_ffw_norm.weight"' not in suffix_block
    assert "GraphIR-required artifacts" in suffix_block
    assert "GLM/Gemma" not in suffix_block
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


def test_attention_scale_uses_contract_without_model_name_branch():
    attn_text = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "attention_prefill.cpp.inc").read_text()
    helper_text = (ROOT / "moe_engine" / "src" / "parts" / "raw_forward" / "forward_helpers.cpp.inc").read_text()
    parser_text = (ROOT / "moe_engine" / "src" / "parts" / "juju_parser.cpp.inc").read_text()
    scale_fn = attn_text[attn_text.index("static float moe_standard_attention_score_scale"):attn_text.index("static float moe_attention_logit_softcap_value")]
    assert scale_fn.index('"attention_scale"') < scale_fn.index("model_config_attention_unit_scale")
    assert scale_fn.index('"query_pre_attn_scalar"') < scale_fn.index("model_config_attention_unit_scale")
    assert "moe_engine_contract_uses_unit_qk_norm_global" in helper_text
    assert "moe_engine_is_gemma4_text_contract" not in helper_text
    assert "unit_qk_norm" in attn_text
    assert "attention_scale_contract" in attn_text
    assert "query_pre_attn_scalar" in attn_text
    assert "attention_scale_source" in parser_text
    assert "qk_norm" in parser_text
    assert "post_attention_norm" in parser_text
    assert "layer_output_scale" in parser_text
    assert "moe_engine_is_gemma4_text_contract" not in attn_text






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

def test_routed_graph_runs_dense_only_when_ir_declares_primary_branch_shape():
    mlp_text = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "mlp_forward.cpp.inc").read_text()
    assert "moe_layer_dense_mlp_should_run_f32" in mlp_text
    helper = mlp_text[mlp_text.index("static int moe_layer_has_common_dense_mlp_tensors_f32"):mlp_text.index("static int moe_layer_dense_mlp_should_run_f32")]
    planner_start = mlp_text.index("static moe_graph_ir_mlp_layer_plan_f32")
    planner = mlp_text[planner_start:mlp_text.index("static int moe_layer_has_common_dense_mlp_tensors_f32", planner_start)]
    should_block = mlp_text[mlp_text.index("static int moe_layer_dense_mlp_should_run_f32"):mlp_text.index("static int moe_layer_moe_mlp_f32")]
    assert '"ffn_gate.weight"' in helper
    assert '"ffn_up.weight"' in helper
    assert '"ffn_down.weight"' in helper
    assert "moe_engine_contract_uses_split_ffn_norm(engine, layer) &&" not in should_block
    assert "moe_layer_has_common_dense_mlp_tensors_f32(engine, layer)" not in should_block
    assert "post_ffw_norm_1" in planner
    assert "A routed layer can still own a primary dense branch" in mlp_text
    assert "forbid_parallel_with_routed_moe" in mlp_text
    assert "fallback_semantics" in mlp_text
    assert "dense_forbidden_with_routed" in planner
    assert "split-FFN router input scale is required" in mlp_text


def test_ple_vocab_masking_allows_smaller_per_layer_vocab():
    mlp_common = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "mlp_common_tensors.cpp.inc").read_text()
    assert "token_vocab_size" in mlp_common
    assert "token_id < ple.token_vocab_size" in mlp_common
    assert "std::fill(s->ple_inputs.begin(), s->ple_inputs.end(), 0.0f)" in mlp_common
    assert "token_desc.rows == 0" in mlp_common
    assert '"per_layer_inp_gate.weight"' in mlp_common
    assert '"per_layer_proj.weight"' in mlp_common
    assert '"per_layer_post_norm.weight"' in mlp_common
    assert '"proj.weight"' in mlp_common
    assert '"post_norm.weight"' in mlp_common


def test_expert_down_scale_is_legacy_runtime_scale_but_not_direct_contract_scale():
    norm_text = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "mlp_normalization.cpp.inc").read_text()
    fwd_text = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "mlp_forward.cpp.inc").read_text()
    router_text = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "router_utils.cpp.inc").read_text()
    topk_text = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "router_topk.cpp.inc").read_text()
    materializer_text = MAT_PATH.read_text()
    for text in (norm_text, fwd_text):
        assert "direct_contract_suffixes" in text
        direct_block = text[text.index("static const char* const direct_contract_suffixes[]"):
                            text.index("const int direct_scale_contract", text.index("static const char* const direct_contract_suffixes[]"))]
        assert '"ffn_down_exps.scale"' not in direct_block
        assert '"per_expert_scale"' in direct_block
        assert '"experts.per_expert_scale"' in direct_block
        assert "moe_engine_contract_uses_direct_per_expert_scale(engine, layer)" in text
        if '"ffn_down_exps.scale"' in text:
            legacy_scale_pos = text.index('"ffn_down_exps.scale"')
            direct_pos = text.index("static const char* const direct_contract_suffixes[]")
            assert legacy_scale_pos < direct_pos
            assert "runtime expert-down branch scale" in text
    router_block = router_text[router_text.index("moe_map_router_per_expert_scale_tensor_f32"):
                               router_text.index("static int moe_router_apply_per_expert_scales_f32")]
    assert "moe_graph_ir_map_layer_op_role_any" in router_block
    assert "moe_graph_ir_map_layer_contract_tensor_role_f32" in router_block
    assert "legacy_generated_juju_suffixes" in router_block
    assert '"ffn_down_exps.scale"' in router_block
    assert '"router.per_expert_scale.weight"' in router_block
    assert "count != 1u && count != (uint64_t)experts" in router_block
    assert "moe_router_apply_per_expert_scales_f32" in topk_text
    assert topk_text.count("moe_router_apply_per_expert_scales_f32") >= 2
    graph_block = materializer_text[materializer_text.index("router_per_expert_scale_weights = bind("):
                                  materializer_text.index("expert_output_scale_weights = bind(")]
    assert '"ffn_down_exps.scale"' in graph_block
    experts_op = materializer_text[materializer_text.index('"name": "moe_experts"'):
                                   materializer_text.index('"name": "post_ffw_norm_2"', materializer_text.index('"name": "moe_experts"'))]
    assert "per_expert_output_scale" in experts_op
    assert "expert_output_scale_weights" in experts_op
    assert "ffn_down_exps.scale" not in experts_op


def test_raw_residual_router_input_requires_explicit_graph_contract():
    norm_text = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "mlp_normalization.cpp.inc").read_text()
    fwd_text = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "mlp_forward.cpp.inc").read_text()
    block = norm_text[norm_text.index("static int moe_layer_router_uses_raw_residual_input"):norm_text.index("static int moe_layer_common_dense_mlp_f32")]
    selector = fwd_text[fwd_text.index("static int moe_layer_router_should_use_raw_residual_f32"):
                        fwd_text.index("static int moe_layer_moe_mlp_f32")]
    assert "moe_engine_contract_uses_split_ffn_norm(engine, layer)" not in block
    assert '"ffn_gate_inp.weight"' not in block
    assert "ffn_gate_inp.weight" not in block
    assert "use_hidden_when_router_has_internal_scale_else_expert_ffn_input" not in block
    assert "return 0;" in block
    assert "router_has_internal_weight_scale" in selector
    assert "return 1;" in selector

def test_no_v_proj_contract_is_layer_local_not_graphwide_string():
    attn_text = (ROOT / "moe_engine/src/parts/generation/attention_decode.cpp.inc").read_text()
    helper_text = (ROOT / "moe_engine/src/parts/raw_forward/forward_helpers.cpp.inc").read_text()
    allow_pos = attn_text.index("const int allow_v_equals_k")
    block = attn_text[allow_pos:attn_text.index("if (qkv_parallel_path)", allow_pos)]
    assert '"no_v_proj"' not in block
    assert '"layers_with_no_v_proj"' not in block
    assert '"no_value_projection"' not in block
    assert "!has_v && moe_engine_contract_allows_missing_v_projection(engine, layer)" in block
    helper_block = helper_text[helper_text.index("static int moe_engine_contract_allows_missing_v_projection"):]
    assert "layers_with_no_v_proj must not hide" in helper_block
    assert "has_q && has_k && has_o && !has_v" in helper_block

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
    planner_block = mlp_text[mlp_text.index("static moe_graph_ir_mlp_layer_plan_f32"):should_pos]
    assert "moe_layer_has_common_dense_mlp_tensors_f32(engine, layer)" not in should_block
    assert '"dense_mlp"' in planner_block
    assert '"dense_ffn"' in planner_block
    assert '"moe_expert_mlp"' in planner_block
    assert "forbid_parallel_with_routed_moe" in mlp_text
    assert "fallback_semantics" in mlp_text
    assert "dense_forbidden_with_routed" in planner_block
    assert "return plan.run_dense ? 1 : 0;" in should_block



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


def test_post_ffw_norms_use_layer_local_suffix_fallback_in_graphir_mode():
    text = (ROOT / "moe_engine/src/parts/generation/mlp_post_ffw_norms.cpp.inc").read_text()
    for name, suffix in (
        ("moe_layer_post_ffw_norm_f32", '"post_ffw_norm.weight"'),
        ("moe_layer_post_ffw_norm1_f32", '"post_ffw_norm_1.weight"'),
        ("moe_layer_post_ffw_norm2_f32", '"post_ffw_norm_2.weight"'),
    ):
        block = text[text.index(f"static int {name}("):]
        block = block[:block.index("\nstatic int ", 1) if "\nstatic int " in block[1:] else len(block)]
        assert "moe_graph_ir_apply_rmsnorm_role_any_f32" in block
        assert "moe_graph_ir_map_layer_contract_tensor_role_f32" in block
        assert "layer_execution_contract_table" in text
        assert suffix in block
        assert "moe_layer_post_ffw_norm_suffixes_f32" in block
    for name, suffix in (
        ("moe_layer_has_post_ffw_norm_weight", '"post_ffw_norm.weight"'),
        ("moe_layer_has_post_ffw_norm1_weight", '"post_ffw_norm_1.weight"'),
        ("moe_layer_has_post_ffw_norm2_weight", '"post_ffw_norm_2.weight"'),
    ):
        block = text[text.index(f"static int {name}("):]
        block = block[:block.index("\nstatic int ", 1) if "\nstatic int " in block[1:] else len(block)]
        assert "moe_graph_ir_layer_has_op_role_any" in block
        assert "moe_graph_ir_layer_contract_declares_tensor_role" in block
        assert suffix in block
        assert "layers_with_" not in block


def test_layer_output_scale_plan_and_apply_use_role_or_layer_suffix_not_graphwide_count():
    common = (ROOT / "moe_engine/src/parts/generation/mlp_post_ffw_norms.cpp.inc").read_text()
    forward = (ROOT / "moe_engine/src/parts/generation_forward.cpp.inc").read_text()
    apply_block = common[common.index("static int moe_layer_output_scale_f32("):]
    assert "moe_graph_ir_map_layer_op_role_any" in apply_block
    assert "moe_graph_ir_map_layer_contract_tensor_role_f32" in apply_block
    assert '"layer_output_scale.weight"' in apply_block
    assert '"layer_scalar.weight"' in apply_block
    assert "layers_with_layer_output_scale" not in apply_block
    plan_block = forward[forward.index("plan.apply_layer_output_scale ="):forward.index("return plan;", forward.index("plan.apply_layer_output_scale ="))]
    assert "moe_graph_ir_layer_has_op_role_any" in plan_block
    assert "moe_graph_ir_layer_contract_declares_tensor_role" in plan_block
    assert '"layer_output_scale.weight"' in plan_block
    assert '"layer_scalar.weight"' in plan_block
    assert "layers_with_layer_output_scale" not in plan_block




def test_layer_execution_contract_table_norms_and_tail_are_runtime_bindings():
    common = (ROOT / "moe_engine/src/parts/generation/mlp_common_tensors.cpp.inc").read_text()
    post_ffw = (ROOT / "moe_engine/src/parts/generation/mlp_post_ffw_norms.cpp.inc").read_text()
    text = common + post_ffw
    helper_start = text.rindex("static int moe_graph_ir_layer_contract_tensor_names_for_role")
    helper_end = text.index("static int moe_graph_ir_apply_rmsnorm_role_any_f32", helper_start)
    helper = text[helper_start:helper_end]
    assert '"layer_execution_contract_table"' in helper
    assert "moe_json_get_string_array_slice" in helper
    assert "find_common_tensor_by_names" in helper or "moe_common_tensor_find(engine, name.c_str())" in helper
    assert "graphir_contract_tensor_bind" in helper
    assert '"norms"' in text
    assert '"tail"' in text
    assert '"post_ffw_norm_1"' in text
    assert '"post_ffw_norm_2"' in text
    assert '"post_ffw_norm"' in text
    assert '"layer_output_scale"' in text


def test_final_logit_softcap_uses_scalar_contract_not_executable_op_only():
    ops = (ROOT / "moe_engine/src/parts/raw_forward/forward_ops.cpp.inc").read_text()
    block = ops[ops.index("static float moe_final_logit_softcap_value"):ops.index("static float moe_apply_final_logit_softcap")]
    assert '"final_logit_softcapping"' in block
    assert '"final_logit_softcap"' in block
    assert '"logit_softcap"' in block
    assert "moe_graph_ir_declares_final_logit_softcap_execution" not in ops
    assert "final_logit_softcap_ignored" not in ops


def test_value_norm_contract_table_is_layer_local_runtime_contract():
    attn = (ROOT / "moe_engine/src/parts/generation/attention_prefill.cpp.inc").read_text()
    block = attn[attn.index("static int moe_layer_uses_unweighted_v_norm"):attn.index("static int moe_layer_has_qk_norm_scaled_standard_layout")]
    assert '"value_norm_contract_table"' in block
    assert '"unweighted_value_norm_is_contractual_when_declared"' in attn
    assert "graph-wide substring" in block
    assert "executable op declares it" not in block


def test_router_input_does_not_use_raw_residual_without_internal_scale():
    norm = (ROOT / "moe_engine/src/parts/generation/mlp_normalization.cpp.inc").read_text()
    fwd = (ROOT / "moe_engine/src/parts/generation/mlp_forward.cpp.inc").read_text()
    block = norm[norm.index("static int moe_layer_router_uses_raw_residual_input"):norm.index("static int moe_layer_common_dense_mlp_f32")]
    selector = fwd[fwd.index("static int moe_layer_router_should_use_raw_residual_f32"):
                   fwd.index("static int moe_layer_moe_mlp_f32")]
    assert '"ffn_gate_inp.weight"' not in block
    assert '"moe_router"' not in block
    assert "return 0;" in block
    assert "router_has_internal_weight_scale" in selector
    assert "router-column scale" in selector
    assert "router_has_internal_norm_scale || router_uses_raw_residual" in fwd


def test_common_tensor_role_lookup_preserves_execution_op_contract_fields():
    type_text = (ROOT / "moe_engine" / "src" / "parts" / "engine_types.cpp.inc").read_text()
    parse_text = (ROOT / "moe_engine" / "src" / "parts" / "codec" / "juju_main_parsing.cpp.inc").read_text()
    desc_text = (ROOT / "moe_engine" / "src" / "parts" / "raw_forward_tensor_desc.cpp.inc").read_text()
    assert "std::string execution_op;" in type_text
    assert "std::string graph_role_text;" in type_text
    assert "std::string bundle_member_role;" in type_text
    assert "common_rec.execution_op = moe_lower_ascii_copy(entry.execution_op);" in parse_text
    assert "common_rec.graph_role_text = moe_lower_ascii_copy(entry.graph_role_text);" in parse_text
    assert "common_rec.bundle_member_role = moe_lower_ascii_copy(entry.bundle_member_role);" in parse_text
    assert "moe_common_tensor_role_text_matches(rec.execution_op, role)" in desc_text
    assert "moe_common_tensor_role_text_matches(rec.bundle_member_role, role)" in desc_text


def test_shared_expert_role_scan_uses_execution_op_fields_without_model_names():
    text = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "mlp_normalization.cpp.inc").read_text()
    assert "moe_shared_expert_role_is_shared_f32(rec.execution_op)" in text
    assert "moe_shared_expert_role_is_shared_f32(rec.bundle_member_role)" in text
    assert "moe_shared_expert_text_is_shared_f32(rec.graph_role_text)" in text
    runtime_scope = text[text.index("static int moe_map_shared_expert_graph_role_f32"):text.index("static int moe_layer_shared_expert_dense_mlp_f32")]
    assert "Gemma" not in runtime_scope
    assert "GLM" not in runtime_scope
    assert "Qwen" not in runtime_scope


def test_standard_attention_skips_noop_rope_at_position_zero():
    attn_text = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "attention_decode.cpp.inc").read_text()
    assert "position 0 is mathematically identity" in attn_text
    assert "if (rope_dim >= 2u && position != 0u)" in attn_text
    assert "attn_standard_rope" in attn_text


def test_sliding_default_rope_uses_full_head_dim_not_full_attention_partial():
    arch = {
        "head_dim": 256,
        "global_head_dim": 512,
        "partial_rotary_factor": 0.25,
        "qk_rope_head_dim": 512,
        "rope_parameters": {
            "full_attention": {
                "rope_type": "proportional",
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.25,
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10000.0,
            },
        },
        "layer_types": ["sliding_attention", "full_attention"],
    }
    sliding = mat._juju_layer_rope_contract(0, arch)
    full = mat._juju_layer_rope_contract(1, arch)
    assert sliding["rope_dim"] == 256
    assert sliding["frequency_dim"] == 256
    assert sliding["partial_rotary_factor"] is None
    assert full["rope_dim"] == 128
    assert full["frequency_dim"] == 512
    attn_text = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "attention_prefill.cpp.inc").read_text()
    assert "sliding_default_rope" in attn_text
    assert "resolved.rope_dim = head_dim" in attn_text
    assert "!sliding_default_rope" in attn_text


def test_partial_rotate_half_pairs_inside_active_rope_block():
    rope_text = (ROOT / "moe_engine" / "src" / "parts" / "generation_rope.cpp.inc").read_text(encoding="utf-8")
    fn = rope_text[
        rope_text.index("static void moe_rope_rotate_half_partial_inplace"):
        rope_text.index("static void moe_standard_rope_apply_inplace_with_layout")
    ]
    assert "const uint32_t partner_stride = active_half_dim;" in fn
    assert "full_dim / 2u" not in fn


def test_router_contract_layer_slice_is_cached_and_cleared_on_reload_paths():
    state = (ROOT / "moe_engine" / "src" / "parts" / "engine_state.cpp.inc").read_text()
    router = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "router_topk.cpp.inc").read_text()
    reset = (ROOT / "moe_engine" / "src" / "parts" / "model_engine_state.cpp.inc").read_text()
    juju = (ROOT / "moe_engine" / "src" / "parts" / "codec" / "juju_main_parsing.cpp.inc").read_text()
    model_io = (ROOT / "moe_engine" / "src" / "parts" / "codec" / "model_file_io.cpp.inc").read_text()
    lifecycle = (ROOT / "moe_engine" / "src" / "parts" / "lifecycle_backend_api.cpp.inc").read_text()
    assert "router_layer_contract_slice_cache_mutex" in state
    assert "router_layer_contract_slice_cache" in state
    assert "moe_router_layer_contract_object_slice_cached_f32" in router
    assert "moe_router_layer_contract_object_slice_f32(engine->offload_graph_ir_json" not in router
    assert router.count("moe_router_layer_contract_object_slice_cached_f32(engine, layer") >= 4
    for text in (reset, juju, model_io, lifecycle):
        assert "router_layer_contract_slice_cache.clear();" in text


def test_router_direct_input_scale_roles_are_distinct_from_norm_gamma_suffixes():
    router = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "router_utils.cpp.inc").read_text()
    helper = router[router.index("static int moe_router_has_direct_input_scale_contract_f32"):router.index("static int moe_map_router_scale_tensor", router.index("static int moe_router_has_direct_input_scale_contract_f32"))]
    assert '"ffn_gate_inp.scale"' not in helper
    assert '"ffn_gate_inp.scales"' not in helper
    assert '"router_input_scale"' in helper
    assert '"moe_router_scale"' in helper
    assert '"router_scale"' in helper
    assert '"router_norm.weight"' not in helper
    assert '"mlp.router.scale"' not in helper
    prep = router[router.index("static int moe_prepare_scaled_router_input_f32"):]
    assert "const int direct_input_scale = moe_router_has_direct_input_scale_contract_f32(engine, layer);" in prep
    assert "const int unit_offset = direct_input_scale" in prep


def test_layer_execution_contract_rows_are_indexed_once_per_doc_not_per_layer():
    state = (ROOT / "moe_engine" / "src" / "parts" / "engine_state.cpp.inc").read_text()
    helper = (ROOT / "moe_engine" / "src" / "parts" / "raw_forward" / "forward_helpers.cpp.inc").read_text()
    assert "layer_contract_rows_cache_built_docs" in state
    assert "moe_contract_all_layer_rows_uncached_f32" in helper
    cached = helper[helper.index("static int moe_contract_layer_rows_cached_f32"):helper.index("static int moe_graph_ir_tensor_binding_name")]
    assert "layer_contract_rows_cache_built_docs" in cached
    assert "moe_contract_all_layer_rows_uncached_f32" in cached
    assert "moe_contract_layer_rows_uncached_f32(*doc, layer" in cached
    assert "engine->layer_contract_rows_cache_built_docs.insert(doc_index)" in cached


def test_graph_layer_ops_slices_are_indexed_once_per_doc_not_per_layer():
    state = (ROOT / "moe_engine" / "src" / "parts" / "engine_state.cpp.inc").read_text()
    router = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "router_topk.cpp.inc").read_text()
    assert "graph_layer_ops_slice_cache_built_docs" in state
    assert "moe_router_all_layer_ops_slices_f32" in router
    cached = router[router.index("static int moe_router_layer_ops_object_slice_cached_f32"):router.index("static int moe_router_op_slice_is_router_f32")]
    assert "graph_layer_ops_slice_cache_built_docs" in cached
    assert "moe_router_all_layer_ops_slices_f32" in cached
    assert "engine->graph_layer_ops_slice_cache_built_docs.insert(doc_index)" in cached


def test_ple_global_binding_has_suffix_fallback_for_prefixed_runtime_tensors():
    mlp_common = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "mlp_common_tensors.cpp.inc").read_text()
    helper = mlp_common[mlp_common.index("static const moe_gguf_common_tensor_record* moe_find_common_tensor_by_roles_or_names"):
                        mlp_common.index("static int moe_map_common_tensor_by_roles_or_names")]
    assert "find_common_tensor_by_op_role" in helper
    assert "find_common_tensor_by_names" in helper
    assert "lowered.compare(lowered.size() - suffix.size()" in helper
    assert "ple_global_tensor_suffix_bind" in helper
    assert '"per_layer_input_model_proj.weight"' in mlp_common
    assert '"per_layer_model_projection_norm.weight"' in mlp_common


def test_ple_empty_global_inputs_do_not_run_speculative_local_adapter():
    mlp_common = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "mlp_common_tensors.cpp.inc").read_text()
    apply = mlp_common[mlp_common.index("static int moe_layer_apply_ple_f32"):
                       mlp_common.index("static int moe_raw_tensor_scalar_at_f32")]
    assert "s->ple_inputs.empty()" in apply
    assert "moe_layer_has_explicit_ple_contract" in apply
    assert "ple_layer_missing_inputs" in apply
    assert "source=explicit_contract" in apply
    assert "ple_layer_apply_local" not in apply
    assert "s->gate[row] = v;" not in apply
    assert "per_layer_inp_gate.weight" not in apply[:apply.index("if (num_layers == 0 ||")]
    assert "per_layer_proj.weight" not in apply[:apply.index("if (num_layers == 0 ||")]


def test_ple_required_detection_reads_layer_contract_table_roles():
    mlp_common = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "mlp_common_tensors.cpp.inc").read_text()
    detector = mlp_common[mlp_common.index("static int moe_layer_has_ple_tensors"):
                          mlp_common.index("static int moe_map_layer_ple_tensor_role_or_suffix_f32")]
    assert "contract_declares_any" in detector
    assert "moe_graph_ir_layer_contract_declares_tensor_role" in detector
    assert '"per_layer_input_proj"' in detector
    assert '"per_layer_input_post_norm"' in detector
    explicit = mlp_common[mlp_common.index("static int moe_layer_has_explicit_ple_contract"):
                          mlp_common.index("static int moe_map_layer_ple_tensor_role_or_suffix_f32")]
    assert "moe_engine_has_common_tensor_suffix" not in explicit
    assert "moe_graph_ir_layer_has_op_role_any" in explicit


def test_conditional_router_hidden_rule_does_not_force_raw_input():
    mlp_norm = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "mlp_normalization.cpp.inc").read_text()
    helper = mlp_norm[mlp_norm.index("static int moe_layer_router_contract_has_hidden_input_f32"):
                      mlp_norm.index("static int moe_layer_execution_contract_router_uses_hidden_f32")]
    assert "moe_router_rule_uses_hidden_only_with_activation_scale_f32" in mlp_norm
    assert '"use_hidden_only_when_explicit_router_input_scale_present"' in mlp_norm
    assert "moe_json_get_string_slice(doc, obj_begin, obj_end, \"rule\", &rule)" in helper
    assert helper.index("moe_json_get_string_slice(doc, obj_begin, obj_end, \"rule\", &rule)") < helper.index("moe_router_contract_object_has_raw_hidden_input_f32")
    assert "return moe_router_contract_object_has_activation_scale_f32(doc, obj_begin, obj_end) ? 1 : 0;" in helper


def test_router_rms_root_sidecar_uses_raw_residual_input():
    mlp = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "mlp_forward.cpp.inc").read_text()
    helper = mlp[mlp.index("static int moe_layer_router_should_use_raw_residual_f32"):
                 mlp.index("static int moe_layer_moe_mlp_f32")]
    assert "router_sidecar_requires_rms_root" in helper
    assert "router_has_internal_weight_scale" in helper
    assert "Gemma4-style" in helper
    first = mlp[mlp.index("static int moe_layer_moe_mlp_f32"):
                mlp.index("static uint32_t moe_mlp_prefill_expert_batch_limit")]
    assert "moe_router_weight_sidecar_requires_rms_root_transform_f32(engine, layer)" in first
    assert "moe_layer_router_should_use_raw_residual_f32(" in first
    batch = mlp[mlp.index("static int moe_layer_moe_mlp_prefill_batch_f32"):]
    assert "moe_router_weight_sidecar_requires_rms_root_transform_f32(engine, layer)" in batch
    assert "moe_layer_router_should_use_raw_residual_f32(" in batch


def test_router_layer_row_does_not_block_graphir_op_score_or_norm_lookup():
    router = (ROOT / "moe_engine" / "src" / "parts" / "generation" / "router_topk.cpp.inc").read_text()
    score_block = router[router.index("static int moe_router_layer_score_contract_f32"):
                         router.index("static int moe_router_uses_sigmoid_scores_for_layer")]
    bool_block = router[router.index("static int moe_router_layer_bool_contract_f32"):
                        router.index("static float moe_router_routed_scaling_factor_for_layer")]
    double_block = router[router.index("static int moe_router_layer_double_contract_f32"):
                          router.index("static int moe_router_layer_bool_contract_f32")]
    u32_block = router[router.index("static int moe_router_layer_u32_contract_f32"):
                       router.index("static uint32_t moe_router_top_k_from_metadata_for_layer")]
    for block in (score_block, bool_block, double_block, u32_block):
        assert "if (!has_layer_contract_row && layer >= 0 && !engine->offload_graph_ir_json.empty())" not in block
        assert "if (!has_layer_contract_row && moe_router_layer_ops_object_slice_cached_f32" not in block
    assert "if (!has_layer_contract_row) {\n            int use_sigmoid" not in score_block
    assert "moe_router_layer_op_score_contract_f32" in score_block
    for block in (bool_block, double_block, u32_block):
        assert "if (moe_router_layer_ops_object_slice_cached_f32" in block

def test_layer_first_prefill_disabled_for_ple_contracts():
    path = ROOT / "moe_engine" / "src" / "parts" / "generation_kv_cache.cpp.inc"
    text = path.read_text(encoding="utf-8", errors="ignore")
    block = text[text.index("static int moe_generation_should_use_layer_first_prefill"):]
    block = block[:block.index("return 1;")]
    assert '"per_layer_inp_gate"' in block
    assert '"per_layer_proj"' in block
    assert '"per_layer_post_norm"' in block
    assert '"per_layer_model_projection"' in block
    assert '"per_layer_inp_gate.weight"' in block
    assert '"per_layer_proj.weight"' in block
    assert '"per_layer_post_norm.weight"' in block

def test_final_logit_softcap_lookup_is_cached_per_engine_generation():
    ops = (ROOT / "moe_engine/src/parts/raw_forward/forward_ops.cpp.inc").read_text()
    start = ops.index("static float moe_final_logit_softcap_value")
    end = ops.index("static float moe_apply_final_logit_softcap", start)
    helper = ops[start:end]
    assert "final_logit_softcap_cache_entry" in helper
    assert "static thread_local std::vector" in helper
    assert "engine->engine_generation" in helper
    assert '"final_logit_softcap"' in helper
    assert "moe_json_get_double_local(engine->offload_graph_ir_json" in helper

def test_final_norm_mapping_is_cached_per_engine_generation():
    mlp_common = (ROOT / "moe_engine/src/parts/generation/mlp_common_tensors.cpp.inc").read_text()
    start = mlp_common.index("static int moe_final_norm_f32")
    end = mlp_common.index("// Cache split/post-norm detection", start)
    helper = mlp_common[start:end]
    assert "final_norm_raw_cache_entry" in helper
    assert "static thread_local std::vector" in helper
    assert "engine->engine_generation" in helper
    assert "cache_final_norm_raw(raw);" in helper
    assert "moe_apply_rmsnorm_common_weight_f32(engine, e.raw" in helper

def test_layer_first_prefill_contract_guard_is_cached():
    kv = (ROOT / "moe_engine/src/parts/generation_kv_cache.cpp.inc").read_text()
    start = kv.index("static int moe_generation_should_use_layer_first_prefill")
    block = kv[start:]
    assert "layer_first_prefill_contract_cache_entry" in block
    assert "static thread_local std::vector" in block
    assert "engine->engine_generation" in block
    assert "unsafe_contract" in block
    assert '"per_layer_inp_gate"' in block
    assert '"layer_output_scale"' in block

def test_rmsnorm_suffix_mapping_is_cached_per_engine_generation():
    mlp_common = (ROOT / "moe_engine/src/parts/generation/mlp_common_tensors.cpp.inc").read_text()
    start = mlp_common.index("static int moe_layer_rmsnorm_suffixes_f32")
    end = mlp_common.index("static int moe_layer_ffn_norm_f32", start)
    helper = mlp_common[start:end]
    assert "rmsnorm_suffix_cache_entry" in helper
    assert "static thread_local std::vector" in helper
    assert "engine->engine_generation" in helper
    assert "suffix_key" in helper
    assert "cache_rmsnorm_suffix(1, &raw);" in helper
    assert "cache_rmsnorm_suffix(0, nullptr);" in helper



def test_missing_v_projection_contract_is_cached():
    src = (ROOT / "moe_engine" / "src" / "parts" / "raw_forward" / "forward_helpers.cpp.inc").read_text()
    assert "missing_v_projection_cache_entry" in src
    assert "missing_v_projection_cache" in src
    assert "engine_generation" in src
    assert "moe_engine_contract_allows_missing_v_projection_explicit(engine, layer)" in src
    assert "moe_engine_contract_has_layer_tensor_suffix_any(engine, layer, q_suffixes" in src

def test_router_bias_lookup_caches_absent_bias_per_generation():
    src = (ROOT / "moe_engine/src/parts/generation/router_utils.cpp.inc").read_text()
    start = src.index("static int moe_map_router_bias_tensor")
    end = src.index("static int moe_router_apply_bias_f32", start)
    helper = src[start:end]
    assert "router_bias_cache_entry" in helper
    assert "static thread_local std::vector" in helper
    assert "engine->engine_generation" in helper
    assert "entry.found = found_ok" in helper
    assert "if (!e.found || !e.raw.base)" in helper
    assert "moe_push_layer_tensor_candidates" in helper


def test_embedding_tensor_mapping_is_cached_per_generation():
    src = (ROOT / "moe_engine/src/parts/raw_forward_embedding.cpp.inc").read_text()
    assert "tls_embedding_map_cache" in src
    assert "engine_generation" in src
    assert "vocab_size" in src and "hidden_size" in src
    assert "cache_embedding_map(0, nullptr)" in src
    assert "*out = entry.raw" in src


def test_router_global_explicit_config_not_blocked_by_layer_contract_rows():
    router = (ROOT / "moe_engine/src/parts/generation/router_topk.cpp.inc").read_text()
    score_block = router[router.index("static int moe_router_layer_score_contract_f32"):
                         router.index("static int moe_router_uses_sigmoid_scores_for_layer")]
    exact_graph_pos = score_block.index("*graph_ir_doc_for_score, score_keys")
    needle_guard_pos = score_block.index("if (has_layer_contract_row) {", exact_graph_pos)
    assert exact_graph_pos < needle_guard_pos
    bool_block = router[router.index("static int moe_router_layer_bool_contract_f32"):
                        router.index("static float moe_router_routed_scaling_factor_for_layer")]
    double_block = router[router.index("static int moe_router_layer_double_contract_f32"):
                          router.index("static int moe_router_layer_bool_contract_f32")]
    u32_block = router[router.index("static int moe_router_layer_u32_contract_f32"):
                       router.index("static uint32_t moe_router_top_k_from_metadata_for_layer")]
    assert "!has_layer_contract_row && moe_router_json_get" not in bool_block
    assert "!has_layer_contract_row && moe_router_json_get" not in double_block
    assert "if (moe_router_json_get_bool_any(engine->offload_graph_ir_json" in bool_block
    assert "if (moe_router_json_get_double_any(engine->offload_graph_ir_json" in double_block
    assert "if (has_layer_contract_row)" not in u32_block[u32_block.rindex("for (size_t i = 0; i < key_count; ++i)"):]
    assert "moe_json_get_u64_local(engine->offload_graph_ir_json" in u32_block


def test_server_tokenizer_keeps_exact_eos_piece_over_generation_eot_ids():
    src = (ROOT / "moe_engine/examples/pc_engine_server.cpp").read_text(encoding="utf-8", errors="ignore")
    add_piece = src[src.index("static void tokenizer_add_piece"):
                    src.index("static void tokenizer_add_unigram_piece")]
    assert 'tokenizer_exact_piece_is_special(raw_piece, "eos")' in add_piece
    assert 'lowered.find("eot")' not in add_piece
    assert 'lowered.find("end")' not in add_piece

    runtime = src[src.index("static void tokenizer_apply_runtime_config"):
                  src.index("static void tokenizer_prepend_bos_if_needed")]
    assert "!tok->eos_token_id_from_piece" in runtime
    assert '"eos_token"' in runtime
    assert "tokenizer_set_special_from_piece(" in runtime
