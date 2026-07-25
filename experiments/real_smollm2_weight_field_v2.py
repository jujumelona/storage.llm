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
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("weight_field_v1", HERE / "real_smollm2_weight_field.py")
v1 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(v1)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
torch.set_num_threads(min(4, os.cpu_count() or 1))
torch.manual_seed(20260726)
np.random.seed(20260726)

ROOT = Path("out/real_smollm2_weight_field_v2")
ROOT.mkdir(parents=True, exist_ok=True)
BASE_ID = v1.BASE_ID
INSTRUCT_ID = v1.INSTRUCT_ID
EVAL_N = 32
WIKI_BLOCKS = 12
v1.WIKI_BLOCKS = WIKI_BLOCKS


def build_field_fp16(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]):
    means = {}
    qdelta = {}
    scales = {}
    key_energy = {}
    layer_values: dict[int, list[float]] = {}
    total_n = 0
    for key in a:
        if key not in b or a[key].shape != b[key].shape:
            raise RuntimeError(f"state mismatch: {key}")
        wa = a[key]
        wb = b[key]
        mean = ((wa + wb) * 0.5).to(torch.float16)
        delta = (wb - wa) * 0.5
        q, scale = v1.quantize_contrast(delta)
        means[key] = mean
        qdelta[key] = q
        scales[key] = scale
        total_n += wa.numel()
        rel = float(delta.square().sum() / (mean.float().square().sum() + 1e-30))
        key_energy[key] = rel
        match = re.search(r"model\.layers\.(\d+)\.", key)
        if match:
            layer_values.setdefault(int(match.group(1)), []).append(rel)
    layer_energy = {i: float(np.mean(values)) for i, values in layer_values.items()}
    return means, qdelta, scales, total_n, layer_energy, key_energy


def materialize_fn(means, qdelta, scales, q_function):
    state = {}
    for key, mean in means.items():
        qvalue = float(np.clip(q_function(key), -1.0, 1.0))
        state[key] = mean.float() + qvalue * v1.dequantize_contrast(qdelta[key], scales[key])
    return state


def chat_prompt(chat_tokenizer, content: str) -> str:
    return chat_tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def arc_examples(chat_tokenizer):
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation")
    rows = []
    for row in ds:
        texts = list(row["choices"]["text"])
        labels = list(row["choices"]["label"])
        answer = str(row["answerKey"])
        mapping = {str(label): i for i, label in enumerate(labels)}
        if answer not in mapping or len(texts) < 2:
            continue
        canonical = [chr(ord("A") + i) for i in range(len(texts))]
        content = "Answer the multiple-choice question with only the option letter.\n\n"
        content += "Question: " + row["question"] + "\n"
        content += "\n".join(f"{canonical[i]}. {text}" for i, text in enumerate(texts))
        content += "\nAnswer:"
        rows.append({
            "prompt": chat_prompt(chat_tokenizer, content),
            "choices": canonical,
            "label": mapping[answer],
        })
        if len(rows) >= EVAL_N:
            break
    return rows


def boolq_examples(chat_tokenizer):
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
        label = 0 if bool(row["answer"]) else 1
        rows.append({
            "prompt": chat_prompt(chat_tokenizer, content),
            "choices": ["yes", "no"],
            "label": label,
        })
        if len(rows) >= EVAL_N:
            break
    return rows


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
    vocab_equal = tokenizer.get_vocab() == chat_tokenizer.get_vocab()
    probes = ["Hello world", "2 + 2 =", "Question: What is water?", "def add(a, b):"]
    probe_equal = all(v1.encode_no_special(tokenizer, p) == v1.encode_no_special(chat_tokenizer, p) for p in probes)
    if not vocab_equal or not probe_equal:
        raise RuntimeError({"vocab_equal": vocab_equal, "probe_equal": probe_equal})
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_a = AutoModelForCausalLM.from_pretrained(BASE_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()
    model_b = AutoModelForCausalLM.from_pretrained(INSTRUCT_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()
    state_a = v1.clone_float_state(model_a)
    state_b = v1.clone_float_state(model_b)

    raw_prompts = ["Once upon a time", "The capital of France is", "Question: 3+5=", "def fibonacci(n):"]
    chat_probe = chat_prompt(chat_tokenizer, "What is the capital of France? Answer briefly.")
    logit_prompts = raw_prompts + [chat_probe]
    ref_a = v1.prompt_logits(model_a, tokenizer, logit_prompts)
    ref_b = v1.prompt_logits(model_b, tokenizer, logit_prompts)

    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    wiki_texts = [row["text"] for row in wiki if row["text"].strip()][:300]
    arc = arc_examples(chat_tokenizer)
    hella = v1.hella_examples()
    boolq = boolq_examples(chat_tokenizer)

    results = [
        evaluate_model("parent_base", model_a, tokenizer, wiki_texts, arc, hella, boolq),
        evaluate_model("parent_instruct", model_b, tokenizer, wiki_texts, arc, hella, boolq),
    ]
    del model_a, model_b
    gc.collect()

    means, qdelta, scales, total_n, layer_energy, key_energy = build_field_fp16(state_a, state_b)
    parent_bf16_bytes = total_n * 2
    payload_bytes = sum(v1.tensor_bytes(t) for t in means.values())
    payload_bytes += sum(v1.tensor_bytes(t) for t in qdelta.values())
    payload_bytes += sum(v1.tensor_bytes(t) for t in scales.values())

    field_tensors = {}
    for key in means:
        safe = key.replace("::", "__")
        field_tensors[f"mean::{safe}"] = means[key]
        field_tensors[f"qdelta::{safe}"] = qdelta[key]
        field_tensors[f"scale::{safe}"] = scales[key]
    field_path = ROOT / "TQWF_REAL_FIELD_FP16.safetensors"
    save_file(field_tensors, str(field_path), metadata={"base": BASE_ID, "instruct": INSTRUCT_ID, "mean_dtype": "float16"})
    actual_field_bytes = field_path.stat().st_size

    L = int(config_a.num_hidden_layers)
    t = np.linspace(1.0 / (L + 1), L / (L + 1), L)
    layer_order = np.argsort(np.argsort(np.array([layer_energy[i] for i in range(L)])))
    layer_percentile = (layer_order + 0.5) / L
    sorted_keys = sorted(key_energy, key=key_energy.get)
    key_percentile = {key: (rank + 0.5) / len(sorted_keys) for rank, key in enumerate(sorted_keys)}

    def layer_q(key, values, embed_q, head_q):
        match = re.search(r"model\.layers\.(\d+)\.", key)
        if match:
            return float(values[int(match.group(1))])
        if "embed_tokens" in key:
            return embed_q
        if key.startswith("lm_head"):
            return head_q
        if key == "model.norm.weight":
            return head_q
        return 0.5 * (embed_q + head_q)

    candidate_functions = {
        "fixed_midpoint_fp16": lambda key: 0.0,
        "fixed_linear_base_to_instruct_fp16": lambda key: layer_q(key, -1.0 + 2.0 * t, -1.0, 1.0),
        "fixed_late_instruct_fp16": lambda key: layer_q(key, np.where(t < 0.60, -1.0, 1.0), -1.0, 1.0),
        "fixed_weight_layer_energy_fp16": lambda key: layer_q(key, -1.0 + 2.0 * layer_percentile, -1.0, 1.0),
        "fixed_high_delta_instruct_low_delta_base": lambda key: (-1.0 if ("embed_tokens" in key or key.startswith("lm_head")) else (1.0 if key_percentile[key] >= 0.60 else -1.0)),
        "fixed_high_delta_instruct_low_delta_midpoint": lambda key: (-1.0 if ("embed_tokens" in key or key.startswith("lm_head")) else (1.0 if key_percentile[key] >= 0.60 else 0.0)),
        "fixed_base_shell_instruct_core": lambda key: (-1.0 if ("embed_tokens" in key or key.startswith("lm_head")) else 1.0),
    }

    endpoint_a_state = materialize_fn(means, qdelta, scales, lambda key: -1.0)
    endpoint_a = v1.load_from_state(config_a, endpoint_a_state)
    endpoint_a_audit = v1.compare_logits(ref_a, v1.prompt_logits(endpoint_a, tokenizer, logit_prompts))
    del endpoint_a, endpoint_a_state
    gc.collect()

    endpoint_b_state = materialize_fn(means, qdelta, scales, lambda key: 1.0)
    endpoint_b = v1.load_from_state(config_a, endpoint_b_state)
    endpoint_b_audit = v1.compare_logits(ref_b, v1.prompt_logits(endpoint_b, tokenizer, logit_prompts))
    del endpoint_b, endpoint_b_state
    gc.collect()

    for name, q_function in candidate_functions.items():
        print("Materializing", name)
        state = materialize_fn(means, qdelta, scales, q_function)
        model = v1.load_from_state(config_a, state)
        results.append(evaluate_model(name, model, tokenizer, wiki_texts, arc, hella, boolq))
        del model, state
        gc.collect()
        (ROOT / "PARTIAL.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    nll_fields = ["wiki_nll", "arc_correct_nll", "hellaswag_correct_nll", "boolq_correct_nll"]
    parent_rows = [row for row in results if row["model"].startswith("parent_")]
    best_domain = {field: min(row[field] for row in parent_rows) for field in nll_fields}
    for row in results:
        row["balanced_relative_nll"] = float(np.mean([row[field] / best_domain[field] for field in nll_fields]))

    best_parent = min(parent_rows, key=lambda row: row["balanced_relative_nll"])
    fixed_rows = [row for row in results if row["model"].startswith("fixed_")]
    best_fixed = min(fixed_rows, key=lambda row: row["balanced_relative_nll"])
    no_large_regression = all(best_fixed[field] <= best_domain[field] * 1.03 for field in nll_fields)
    promoted = (
        best_fixed["balanced_relative_nll"] < best_parent["balanced_relative_nll"] - 0.001
        and no_large_regression
        and best_fixed["mean_mcq_accuracy"] >= best_parent["mean_mcq_accuracy"] - 0.01
        and endpoint_a_audit["relative_rms"] < 0.02
        and endpoint_b_audit["relative_rms"] < 0.02
    )

    summary = {
        "status": "REAL_PUBLIC_FIXED_MODEL_PASS" if promoted else "REAL_PUBLIC_NOT_PROMOTED",
        "version": 2,
        "parents": [BASE_ID, INSTRUCT_ID],
        "config_audit": config_audit,
        "tokenizer_vocab_equal": vocab_equal,
        "tokenizer_probe_equal": probe_equal,
        "evaluation": {
            "wiki_blocks": WIKI_BLOCKS,
            "arc_examples": len(arc),
            "hellaswag_examples": len(hella),
            "boolq_examples": len(boolq),
            "arc_and_boolq_chat_template": True,
        },
        "weight_field": {
            "mean_dtype": "float16",
            "contrast_dtype": "int8_per_row",
            "parameter_count": total_n,
            "parent_bf16_bytes": parent_bf16_bytes,
            "payload_bytes": payload_bytes,
            "actual_serialized_bytes": actual_field_bytes,
            "actual_ratio_vs_bf16_parent": actual_field_bytes / parent_bf16_bytes,
            "endpoint_base_logit_audit": endpoint_a_audit,
            "endpoint_instruct_logit_audit": endpoint_b_audit,
        },
        "best_parent": best_parent["model"],
        "best_fixed": best_fixed["model"],
        "best_parent_balanced_relative_nll": best_parent["balanced_relative_nll"],
        "best_fixed_balanced_relative_nll": best_fixed["balanced_relative_nll"],
        "delta_fixed_vs_best_parent": best_fixed["balanced_relative_nll"] - best_parent["balanced_relative_nll"],
        "no_domain_regression_over_3pct": no_large_regression,
        "promoted": promoted,
        "results": results,
    }
    (ROOT / "RESULTS.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fields = [
        "model", "wiki_nll", "arc_accuracy", "arc_correct_nll",
        "hellaswag_accuracy", "hellaswag_correct_nll", "boolq_accuracy",
        "boolq_correct_nll", "mean_mcq_accuracy", "balanced_relative_nll", "seconds",
    ]
    with (ROOT / "METRICS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({field: row.get(field) for field in fields})

    report = [
        "# Corrected Real SmolLM2 Weight-Field Validation",
        "",
        f"Status: **{summary['status']}**",
        f"Actual field ratio: **{summary['weight_field']['actual_ratio_vs_bf16_parent']:.6f}x**",
        f"Base endpoint relative RMS: **{endpoint_a_audit['relative_rms']:.6g}**",
        f"Instruct endpoint relative RMS: **{endpoint_b_audit['relative_rms']:.6g}**",
        f"Best parent: **{best_parent['model']}** ({best_parent['balanced_relative_nll']:.6f})",
        f"Best fixed model: **{best_fixed['model']}** ({best_fixed['balanced_relative_nll']:.6f})",
        f"Delta: **{summary['delta_fixed_vs_best_parent']:+.6f}**",
        "",
        "Only fixed, standard single-tensor candidates are eligible for promotion. Prompt-conditioned endpoint selection is excluded.",
    ]
    (ROOT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")

    if promoted:
        state = materialize_fn(means, qdelta, scales, candidate_functions[best_fixed["model"]])
        model = v1.load_from_state(config_a, state)
        model_dir = ROOT / "PROMOTED_MODEL"
        model.save_pretrained(model_dir, safe_serialization=True)
        tokenizer.save_pretrained(model_dir)
        (model_dir / "PROMOTION.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    field_path.unlink(missing_ok=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
