from __future__ import annotations

import csv
import gc
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent

def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

v1 = load_module("weight_field_v1", "real_smollm2_weight_field.py")
v2 = load_module("weight_field_v2", "real_smollm2_weight_field_v2.py")
v3 = load_module("weight_field_v3", "real_smollm2_weight_field_v3_blind.py")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
torch.set_num_threads(min(4, os.cpu_count() or 1))
torch.manual_seed(20260726)
np.random.seed(20260726)

ROOT = Path("out/real_smollm2_weight_field_v4_structural_blind")
ROOT.mkdir(parents=True, exist_ok=True)
BASE_ID = v1.BASE_ID
INSTRUCT_ID = v1.INSTRUCT_ID
BLIND_OFFSET = 192
BLIND_N = 64
WIKI_BLOCKS = 16
v1.WIKI_BLOCKS = WIKI_BLOCKS
v3.BLIND_OFFSET = BLIND_OFFSET
v3.BLIND_N = BLIND_N


def main():
    config_a = AutoConfig.from_pretrained(BASE_ID)
    config_b = AutoConfig.from_pretrained(INSTRUCT_ID)
    fields = [
        "hidden_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "vocab_size",
        "hidden_act", "rope_theta", "tie_word_embeddings",
    ]
    config_audit = {field: [getattr(config_a, field), getattr(config_b, field)] for field in fields}
    if any(pair[0] != pair[1] for pair in config_audit.values()):
        raise RuntimeError({"config_mismatch": config_audit})

    tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
    chat_tokenizer = AutoTokenizer.from_pretrained(INSTRUCT_ID)
    if tokenizer.get_vocab() != chat_tokenizer.get_vocab():
        raise RuntimeError("tokenizer mismatch")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_a = AutoModelForCausalLM.from_pretrained(BASE_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()
    model_b = AutoModelForCausalLM.from_pretrained(INSTRUCT_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()
    state_a = v1.clone_float_state(model_a)
    state_b = v1.clone_float_state(model_b)

    probe_prompts = [
        "A long time ago", "The chemical symbol for oxygen is", "Question: 8*7=", "class Queue:",
        v3.chat_prompt(chat_tokenizer, "State one fact about the Moon."),
    ]
    ref_a = v1.prompt_logits(model_a, tokenizer, probe_prompts)
    ref_b = v1.prompt_logits(model_b, tokenizer, probe_prompts)

    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    all_wiki = [row["text"] for row in wiki if row["text"].strip()]
    wiki_texts = all_wiki[1000:1700]
    arc = v3.arc_blind(chat_tokenizer)
    hella = v3.hella_blind()
    boolq = v3.boolq_blind(chat_tokenizer)
    if not (len(arc) == len(hella) == len(boolq) == BLIND_N):
        raise RuntimeError({"arc": len(arc), "hella": len(hella), "boolq": len(boolq)})

    results = [
        v3.evaluate_model("parent_base", model_a, tokenizer, wiki_texts, arc, hella, boolq),
        v3.evaluate_model("parent_instruct", model_b, tokenizer, wiki_texts, arc, hella, boolq),
    ]
    del model_a, model_b
    gc.collect()

    means, qdelta, scales, total_n, layer_energy, key_energy = v2.build_field_fp16(state_a, state_b)

    endpoint_a_state = v2.materialize_fn(means, qdelta, scales, lambda key: -1.0)
    endpoint_a = v1.load_from_state(config_a, endpoint_a_state)
    endpoint_a_audit = v1.compare_logits(ref_a, v1.prompt_logits(endpoint_a, tokenizer, probe_prompts))
    del endpoint_a, endpoint_a_state
    gc.collect()

    endpoint_b_state = v2.materialize_fn(means, qdelta, scales, lambda key: 1.0)
    endpoint_b = v1.load_from_state(config_a, endpoint_b_state)
    endpoint_b_audit = v1.compare_logits(ref_b, v1.prompt_logits(endpoint_b, tokenizer, probe_prompts))
    del endpoint_b, endpoint_b_state
    gc.collect()

    def is_shell(key: str) -> bool:
        return "embed_tokens" in key or key.startswith("lm_head") or key == "model.norm.weight"

    def is_attention(key: str) -> bool:
        return ".self_attn." in key

    def is_mlp(key: str) -> bool:
        return ".mlp." in key

    def is_layer_norm(key: str) -> bool:
        return "layernorm" in key.lower()

    def shell_body(body_q: float, norms_q: float | None = None):
        def fn(key: str) -> float:
            if is_shell(key):
                return -1.0
            if norms_q is not None and is_layer_norm(key):
                return norms_q
            return body_q
        return fn

    def shell_attn_mlp(attn_q: float, mlp_q: float, norm_q: float = -1.0):
        def fn(key: str) -> float:
            if is_shell(key):
                return -1.0
            if is_attention(key):
                return attn_q
            if is_mlp(key):
                return mlp_q
            if is_layer_norm(key):
                return norm_q
            return -1.0
        return fn

    candidates = {
        "struct_base_shell_body_q_m0p5": shell_body(-0.5),
        "struct_base_shell_body_q_m0p25": shell_body(-0.25),
        "struct_base_shell_body_q_zero": shell_body(0.0),
        "struct_base_shell_body_q_p0p25": shell_body(0.25),
        "struct_base_shell_base_norm_body_q_zero": shell_body(0.0, norms_q=-1.0),
        "struct_base_shell_attn_q_zero_mlp_base": shell_attn_mlp(0.0, -1.0, -1.0),
        "struct_base_shell_attn_base_mlp_q_zero": shell_attn_mlp(-1.0, 0.0, -1.0),
        "struct_base_shell_attn_q_m0p25_mlp_q_zero": shell_attn_mlp(-0.25, 0.0, -1.0),
        "struct_base_shell_attn_q_zero_mlp_q_m0p25": shell_attn_mlp(0.0, -0.25, -1.0),
    }

    for name, q_function in candidates.items():
        print("Materializing", name)
        state = v2.materialize_fn(means, qdelta, scales, q_function)
        model = v1.load_from_state(config_a, state)
        results.append(v3.evaluate_model(name, model, tokenizer, wiki_texts, arc, hella, boolq))
        del model, state
        gc.collect()
        (ROOT / "PARTIAL.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    nll_fields = ["wiki_nll", "arc_correct_nll", "hellaswag_correct_nll", "boolq_correct_nll"]
    acc_fields = ["arc_accuracy", "hellaswag_accuracy", "boolq_accuracy"]
    parent_rows = [row for row in results if row["model"].startswith("parent_")]
    best_nll = {field: min(row[field] for row in parent_rows) for field in nll_fields}
    best_acc = {field: max(row[field] for row in parent_rows) for field in acc_fields}
    for row in results:
        row["balanced_relative_nll"] = float(np.mean([row[field] / best_nll[field] for field in nll_fields]))
        row["nll_feasible"] = all(row[field] <= 1.03 * best_nll[field] for field in nll_fields)
        row["accuracy_feasible"] = all(row[field] >= best_acc[field] - 0.05 for field in acc_fields)

    best_parent = min(parent_rows, key=lambda row: row["balanced_relative_nll"])
    candidate_rows = [row for row in results if row["model"].startswith("struct_")]
    feasible = [row for row in candidate_rows if row["nll_feasible"] and row["accuracy_feasible"]]
    best_candidate = min(feasible, key=lambda row: row["balanced_relative_nll"]) if feasible else None
    endpoint_pass = endpoint_a_audit["relative_rms"] < 0.02 and endpoint_b_audit["relative_rms"] < 0.02
    promoted = bool(
        best_candidate is not None
        and best_candidate["balanced_relative_nll"] < best_parent["balanced_relative_nll"] - 0.001
        and endpoint_pass
    )

    payload_bytes = sum(v1.tensor_bytes(tensor) for tensor in means.values())
    payload_bytes += sum(v1.tensor_bytes(tensor) for tensor in qdelta.values())
    payload_bytes += sum(v1.tensor_bytes(tensor) for tensor in scales.values())
    parent_bf16_bytes = total_n * 2

    summary = {
        "status": "REAL_PUBLIC_STRUCTURAL_BLIND_PASS" if promoted else "REAL_PUBLIC_STRUCTURAL_BLIND_NOT_PROMOTED",
        "parents": [BASE_ID, INSTRUCT_ID],
        "config_audit": config_audit,
        "blind_protocol": {
            "mcq_offset": BLIND_OFFSET,
            "mcq_examples_each": BLIND_N,
            "wiki_text_offset": 1000,
            "wiki_blocks": WIKI_BLOCKS,
            "candidate_families_fixed_before_evaluation": True,
        },
        "structure": {
            "base_embedding_and_lm_head_fixed": True,
            "separate_attention_and_swiglu_coordinates": True,
            "runtime_parent_models": False,
            "post_merge_training": False,
            "input_router": False,
        },
        "weight_field": {
            "mean_dtype": "float16",
            "contrast_dtype": "int8_per_row",
            "payload_ratio_vs_bf16_parent": payload_bytes / parent_bf16_bytes,
            "endpoint_base_logit_audit": endpoint_a_audit,
            "endpoint_instruct_logit_audit": endpoint_b_audit,
        },
        "eligibility": {
            "nll_regression_limit": 0.03,
            "per_task_accuracy_regression_limit": 0.05,
            "minimum_balanced_relative_nll_improvement": 0.001,
        },
        "best_nll_by_parent": best_nll,
        "best_accuracy_by_parent": best_acc,
        "best_parent": best_parent["model"],
        "best_parent_balanced_relative_nll": best_parent["balanced_relative_nll"],
        "best_feasible_candidate": None if best_candidate is None else best_candidate["model"],
        "best_feasible_candidate_balanced_relative_nll": None if best_candidate is None else best_candidate["balanced_relative_nll"],
        "delta_candidate_vs_best_parent": None if best_candidate is None else best_candidate["balanced_relative_nll"] - best_parent["balanced_relative_nll"],
        "endpoint_pass": endpoint_pass,
        "promoted": promoted,
        "results": results,
    }
    (ROOT / "RESULTS.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fields_out = [
        "model", "wiki_nll", "arc_accuracy", "arc_correct_nll",
        "hellaswag_accuracy", "hellaswag_correct_nll", "boolq_accuracy",
        "boolq_correct_nll", "mean_mcq_accuracy", "balanced_relative_nll",
        "nll_feasible", "accuracy_feasible", "seconds",
    ]
    with (ROOT / "METRICS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields_out)
        writer.writeheader()
        for row in results:
            writer.writerow({field: row.get(field) for field in fields_out})

    report = [
        "# Structural Blind SmolLM2 Weight-Field Validation",
        "",
        f"Status: **{summary['status']}**",
        f"Best parent: **{best_parent['model']}** ({best_parent['balanced_relative_nll']:.6f})",
        f"Best feasible candidate: **{summary['best_feasible_candidate']}**",
        f"Candidate delta: **{summary['delta_candidate_vs_best_parent']}**",
        f"Field payload ratio: **{summary['weight_field']['payload_ratio_vs_bf16_parent']:.6f}x**",
        "",
        "All candidates are fixed standard checkpoints with base shell and separately recombined attention/SwiGLU blocks. No routing or calibration is used.",
    ]
    (ROOT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")

    if promoted and best_candidate is not None:
        state = v2.materialize_fn(means, qdelta, scales, candidates[best_candidate["model"]])
        model = v1.load_from_state(config_a, state)
        model_dir = ROOT / "PROMOTED_MODEL"
        model.save_pretrained(model_dir, safe_serialization=True)
        tokenizer.save_pretrained(model_dir)
        (model_dir / "PROMOTION.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
