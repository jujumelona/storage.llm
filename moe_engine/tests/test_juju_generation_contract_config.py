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
    assert "JUJU attention scale does not match query_pre_attn_scalar" in eval_text
    assert "JUJU storage format plan is not available for exact PPL" in eval_text
    assert "JUJU expert index is not ready for exact PPL" in eval_text
    assert "selected-expert linear fallback was used" in eval_text
    assert "non-finite hidden state" in eval_text
    assert "non-finite lm_head logprob" in eval_text
    assert "moe_juju_required_feature_present(json, \"exact_ppl_mode\")" in parser_text
    assert "engine->model_config_attention_scale = (float)attention_scale" in parser_text
