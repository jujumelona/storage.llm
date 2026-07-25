from __future__ import annotations

import csv
import gc
import importlib.util
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
spec1 = importlib.util.spec_from_file_location("weight_field_v1", HERE / "real_smollm2_weight_field.py")
v1 = importlib.util.module_from_spec(spec1)
assert spec1.loader is not None
spec1.loader.exec_module(v1)
spec2 = importlib.util.spec_from_file_location("weight_field_v2", HERE / "real_smollm2_weight_field_v2.py")
v2 = importlib.util.module_from_spec(spec2)
assert spec2.loader is not None
spec2.loader.exec_module(v2)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
torch.set_num_threads(min(4, os.cpu_count() or 1))
torch.manual_seed(20260726)
np.random.seed(20260726)

ROOT = Path("out/real_smollm2_weight_field_v3_blind")
ROOT.mkdir(parents=True, exist_ok=True)
BASE_ID = v1.BASE_ID
INSTRUCT_ID = v1.INSTRUCT_ID
BLIND_OFFSET = 64
BLIND_N = 64
WIKI_BLOCKS = 16
v1.WIKI_BLOCKS = WIKI_BLOCKS


def chat_prompt(tokenizer, content: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def arc_blind(tokenizer):
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation")
    rows = []
    for row in ds:
        texts = list(row["choices"]["text"])
        labels = list(row["choices"]["label"])
        answer = str(row["answerKey"])
        mapping = {str(label): i for i, label in enumerate(labels)}
        if answer not in mapping or len(texts) < 2:
            continue
        letters = [chr(ord("A") + i) for i in range(len(texts))]
        content = "Answer the multiple-choice question with only the option letter.\n\n"
        content += "Question: " + row["question"] + "\n"
        content += "\n".join(f"{letters[i]}. {text}" for i, text in enumerate(texts))
        content += "\nAnswer:"
        rows.append({"prompt": chat_prompt(tokenizer, content), "choices": letters, "label": mapping[answer]})
    return rows[BLIND_OFFSET : BLIND_OFFSET + BLIND_N]


def hella_blind():
    ds = load_dataset("Rowan/hellaswag", split="validation")
    rows = []
    for row in ds:
        label = int(row["label"])
        if label < 0:
            continue
        rows.append({"prompt": row["ctx"].strip(), "choices": list(row["endings"]), "label": label})
    return rows[BLIND_OFFSET : BLIND_OFFSET + BLIND_N]


def boolq_blind(tokenizer):
    try:
        ds = load_dataset("google/boolq", split="validation")
    except Exception:
        ds = load_dataset("super_glue", "boolq", split="validation")
    rows = []
    for row in ds:
        content = (
            "Read the passage and answer the question with only yes or no.\n\n"
            f"Passage: {row['passage']}\n\nQuestion: {row['question']}\nAnswer:"
        )
        rows.append({
            "prompt": chat_prompt(tokenizer, content),
            "choices": ["yes", "no"],
            "label": 0 if bool(row["answer"]) else 1,
        })
    return rows[BLIND_OFFSET : BLIND_OFFSET + BLIND_N]


def evaluate_model(name, model, tokenizer, wiki_texts, arc, hella, boolq):
    start = time.time()
    wiki_nll = v1.evaluate_wikitext(model, tokenizer, wiki_texts)
    arc_acc, arc_nll = v1.evaluate_mcq(model, tokenizer, arc)
    hella_acc, hella_nll = v1.evaluate_mcq(model, tokenizer, hella)
    boolq_acc, boolq_nll = v1.evaluate_mcq(model, tokenizer, boolq)
    result = {
        "model": name,
        "wiki_nll": wiki_nll,
        "arc_accuracy": arc_acc,
        "arc_correct_nll": arc_nll,
        "hellaswag_accuracy": hella_acc,
        "hellaswag_correct_nll": hella_nll,
        "boolq_accuracy": boolq_acc,
        "boolq_correct_nll": boolq_nll,
        "mean_mcq_accuracy": float(np.mean([arc_acc, hella_acc, boolq_acc])),
        "seconds": time.time() - start,
    }
    print(json.dumps(result, indent=2))
    return result


def main():
    config_a = AutoConfig.from_pretrained(BASE_ID)
    config_b = AutoConfig.from_pretrained(INSTRUCT_ID)
    fields = [
        "hidden_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "vocab_size",
        "hidden_act", "rope_theta", "tie_word_embeddings",
    ]
    config_audit = {f: [getattr(config_a, f), getattr(config_b, f)] for f in fields}
    if any(pair[0] != pair[1] for pair in config_audit.values()):
        raise RuntimeError({"config_mismatch": config_audit})

    tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
    chat_tokenizer = AutoTokenizer.from_pretrained(INSTRUCT_ID)
    if tokenizer.get_vocab() != chat_tokenizer.get_vocab():
        raise RuntimeError("tokenizer vocab mismatch")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_a = AutoModelForCausalLM.from_pretrained(BASE_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()
    model_b = AutoModelForCausalLM.from_pretrained(INSTRUCT_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()
    state_a = v1.clone_float_state(model_a)
    state_b = v1.clone_float_state(model_b)

    probe_prompts = [
        "The quick brown fox", "Water freezes at", "Question: 7+6=", "def sort_values(xs):",
        chat_prompt(chat_tokenizer, "Explain gravity in one sentence."),
    ]
    ref_a = v1.prompt_logits(model_a, tokenizer, probe_prompts)
    ref_b = v1.prompt_logits(model_b, tokenizer, probe_prompts)

    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    all_wiki = [row["text"] for row in wiki if row["text"].strip()]
    wiki_texts = all_wiki[300:900]
    arc = arc_blind(chat_tokenizer)
    hella = hella_blind()
    boolq = boolq_blind(chat_tokenizer)
    if not (len(arc) == len(hella) == len(boolq) == BLIND_N):
        raise RuntimeError({"arc": len(arc), "hella": len(hella), "boolq": len(boolq)})

    results = [
        evaluate_model("parent_base", model_a, tokenizer, wiki_texts, arc, hella, boolq),
        evaluate_model("parent_instruct", model_b, tokenizer, wiki_texts, arc, hella, boolq),
    ]
    del model_a, model_b
    gc.collect()

    means, qdelta, scales, total_n, layer_energy, key_energy = v2.build_field_fp16(state_a, state_b)
    L = int(config_a.num_hidden_layers)
    t = np.linspace(1.0 / (L + 1), L / (L + 1), L)
    sorted_keys = sorted(key_energy, key=key_energy.get)
    key_percentile = {key: (rank + 0.5) / len(sorted_keys) for rank, key in enumerate(sorted_keys)}

    def endpoint_a_q(key):
        return -1.0

    def endpoint_b_q(key):
        return 1.0

    endpoint_a_state = v2.materialize_fn(means, qdelta, scales, endpoint_a_q)
    endpoint_a = v1.load_from_state(config_a, endpoint_a_state)
    endpoint_a_audit = v1.compare_logits(ref_a, v1.prompt_logits(endpoint_a, tokenizer, probe_prompts))
    del endpoint_a, endpoint_a_state
    gc.collect()

    endpoint_b_state = v2.materialize_fn(means, qdelta, scales, endpoint_b_q)
    endpoint_b = v1.load_from_state(config_a, endpoint_b_state)
    endpoint_b_audit = v1.compare_logits(ref_b, v1.prompt_logits(endpoint_b, tokenizer, probe_prompts))
    del endpoint_b, endpoint_b_state
    gc.collect()

    def global_q(value):
        return lambda key: value

    def high_delta_q(threshold, selected_q):
        def fn(key):
            if "embed_tokens" in key or key.startswith("lm_head") or key == "model.norm.weight":
                return -1.0
            return selected_q if key_percentile[key] >= threshold else -1.0
        return fn

    def late_layers_q(key):
        if "embed_tokens" in key or key.startswith("lm_head") or key == "model.norm.weight":
            return -1.0
        match = re.search(r"model\.layers\.(\d+)\.", key)
        if match and t[int(match.group(1))] >= 0.60:
            return -0.5
        return -1.0

    candidates = {
        "blind_global_q_m0p875": global_q(-0.875),
        "blind_global_q_m0p75": global_q(-0.75),
        "blind_global_q_m0p625": global_q(-0.625),
        "blind_global_q_m0p5": global_q(-0.5),
        "blind_high_delta_t60_q_m0p5": high_delta_q(0.60, -0.5),
        "blind_high_delta_t60_q_m0p25": high_delta_q(0.60, -0.25),
        "blind_high_delta_t70_q_zero": high_delta_q(0.70, 0.0),
        "blind_late_layers_q_m0p5": late_layers_q,
    }

    for name, q_function in candidates.items():
        print("Materializing", name)
        state = v2.materialize_fn(means, qdelta, scales, q_function)
        model = v1.load_from_state(config_a, state)
        results.append(evaluate_model(name, model, tokenizer, wiki_texts, arc, hella, boolq))
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
    candidate_rows = [row for row in results if row["model"].startswith("blind_")]
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
        "status": "REAL_PUBLIC_BLIND_FIXED_MODEL_PASS" if promoted else "REAL_PUBLIC_BLIND_NOT_PROMOTED",
        "parents": [BASE_ID, INSTRUCT_ID],
        "config_audit": config_audit,
        "blind_protocol": {
            "development_results_not_reused": True,
            "mcq_offset": BLIND_OFFSET,
            "mcq_examples_each": BLIND_N,
            "wiki_text_offset": 300,
            "wiki_blocks": WIKI_BLOCKS,
            "candidate_schedule_frozen_before_blind_evaluation": True,
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
        "# Blind Real SmolLM2 Fixed Weight-Field Validation",
        "",
        f"Status: **{summary['status']}**",
        f"Best parent: **{best_parent['model']}** ({best_parent['balanced_relative_nll']:.6f})",
        f"Best feasible candidate: **{summary['best_feasible_candidate']}**",
        f"Candidate delta: **{summary['delta_candidate_vs_best_parent']}**",
        f"Field payload ratio: **{summary['weight_field']['payload_ratio_vs_bf16_parent']:.6f}x**",
        f"Base endpoint relative RMS: **{endpoint_a_audit['relative_rms']:.6g}**",
        f"Instruct endpoint relative RMS: **{endpoint_b_audit['relative_rms']:.6g}**",
        "",
        "Promotion requires blind improvement, every NLL domain within 3%, and every MCQ accuracy within 5 percentage points of the better parent.",
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
