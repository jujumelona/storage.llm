from __future__ import annotations

import csv
import gc
import json
import math
import os
import re
import shutil
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
torch.set_num_threads(min(4, os.cpu_count() or 1))
torch.manual_seed(20260726)
np.random.seed(20260726)

ROOT = Path("out/real_smollm2_weight_field")
ROOT.mkdir(parents=True, exist_ok=True)
RESULT_JSON = ROOT / "RESULTS.json"
BASE_ID = "HuggingFaceTB/SmolLM2-135M"
INSTRUCT_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
EVAL_ARC = 32
EVAL_HELLA = 32
WIKI_BLOCKS = 12
BLOCK_SIZE = 256


def tensor_bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def clone_float_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().float().clone() for k, v in model.state_dict().items()}


def encode_no_special(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


@torch.inference_mode()
def prompt_logits(model, tokenizer, prompts: list[str]) -> list[torch.Tensor]:
    out = []
    for text in prompts:
        ids = encode_no_special(tokenizer, text)
        x = torch.tensor([ids], dtype=torch.long)
        out.append(model(input_ids=x, use_cache=False).logits.detach().cpu().float())
    return out


def quantize_contrast(d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if d.ndim >= 2:
        flat = d.reshape(d.shape[0], -1)
        scale = flat.abs().amax(dim=1).clamp_min(1e-12) / 127.0
        q = torch.round(flat / scale[:, None]).clamp(-127, 127).to(torch.int8)
        return q.reshape_as(d), scale.to(torch.float16)
    scale = d.abs().amax().clamp_min(1e-12).reshape(1) / 127.0
    q = torch.round(d / scale.float()).clamp(-127, 127).to(torch.int8)
    return q, scale.to(torch.float16)


def dequantize_contrast(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if q.ndim >= 2:
        view = [q.shape[0]] + [1] * (q.ndim - 1)
        return q.float() * scale.float().view(*view)
    return q.float() * scale.float()[0]


def build_field(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]):
    means: dict[str, torch.Tensor] = {}
    qdelta: dict[str, torch.Tensor] = {}
    scales: dict[str, torch.Tensor] = {}
    energy_by_layer: dict[int, list[float]] = {}
    total_n = 0
    for key in a:
        if key not in b or a[key].shape != b[key].shape:
            raise RuntimeError(f"state mismatch: {key}")
        wa = a[key]
        wb = b[key]
        m = ((wa + wb) * 0.5).to(torch.bfloat16)
        d = (wb - wa) * 0.5
        q, s = quantize_contrast(d)
        means[key] = m
        qdelta[key] = q
        scales[key] = s
        total_n += wa.numel()
        match = re.search(r"model\.layers\.(\d+)\.", key)
        if match:
            layer = int(match.group(1))
            rel = float(d.square().sum() / (m.float().square().sum() + 1e-30))
            energy_by_layer.setdefault(layer, []).append(rel)
    energy = {layer: float(np.mean(vals)) for layer, vals in energy_by_layer.items()}
    return means, qdelta, scales, total_n, energy


def q_for_key(key: str, schedule: list[float], embed_q: float, head_q: float) -> float:
    match = re.search(r"model\.layers\.(\d+)\.", key)
    if match:
        return float(schedule[int(match.group(1))])
    if "embed_tokens" in key:
        return float(embed_q)
    if key.startswith("lm_head") or key == "model.norm.weight":
        return float(head_q)
    return float((embed_q + head_q) * 0.5)


def materialize(
    means: dict[str, torch.Tensor],
    qdelta: dict[str, torch.Tensor],
    scales: dict[str, torch.Tensor],
    schedule: list[float],
    embed_q: float,
    head_q: float,
) -> dict[str, torch.Tensor]:
    result = {}
    for key, m in means.items():
        qv = q_for_key(key, schedule, embed_q, head_q)
        result[key] = m.float() + qv * dequantize_contrast(qdelta[key], scales[key])
    return result


def load_from_state(config, state: dict[str, torch.Tensor]):
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)
    missing, unexpected = model.load_state_dict(state, strict=False)
    illegal_missing = [x for x in missing if x != "lm_head.weight"]
    if illegal_missing or unexpected:
        raise RuntimeError({"missing": missing, "unexpected": unexpected})
    model.eval()
    return model


@torch.inference_mode()
def evaluate_wikitext(model, tokenizer, texts: list[str]) -> float:
    ids = encode_no_special(tokenizer, "\n\n".join(texts))
    usable = min(len(ids) - 1, WIKI_BLOCKS * BLOCK_SIZE)
    losses = []
    for start in range(0, usable, BLOCK_SIZE):
        chunk = ids[start : start + BLOCK_SIZE + 1]
        if len(chunk) < 32:
            continue
        x = torch.tensor([chunk[:-1]], dtype=torch.long)
        y = torch.tensor([chunk[1:]], dtype=torch.long)
        logits = model(input_ids=x, use_cache=False).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        losses.append(float(loss))
    return float(np.mean(losses))


@torch.inference_mode()
def option_nlls(model, tokenizer, prompt: str, choices: list[str]) -> list[float]:
    prompt_ids = encode_no_special(tokenizer, prompt)
    if not prompt_ids:
        prompt_ids = [tokenizer.eos_token_id]
    sequences = []
    starts = []
    for choice in choices:
        option_ids = encode_no_special(tokenizer, " " + choice)
        if not option_ids:
            option_ids = [tokenizer.eos_token_id]
        starts.append(len(prompt_ids))
        sequences.append(prompt_ids + option_ids)
    max_len = max(map(len, sequences))
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    batch = torch.full((len(sequences), max_len), pad, dtype=torch.long)
    mask = torch.zeros_like(batch)
    for i, seq in enumerate(sequences):
        batch[i, : len(seq)] = torch.tensor(seq)
        mask[i, : len(seq)] = 1
    logits = model(input_ids=batch, attention_mask=mask, use_cache=False).logits.float()
    logp = logits.log_softmax(-1)
    values = []
    for i, seq in enumerate(sequences):
        start = starts[i]
        token_positions = torch.arange(start, len(seq), dtype=torch.long)
        pred_positions = token_positions - 1
        targets = batch[i, token_positions]
        nll = -logp[i, pred_positions, targets].mean()
        values.append(float(nll))
    return values


def arc_examples() -> list[dict]:
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation")
    rows = []
    for row in ds:
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        mapping = {str(label): i for i, label in enumerate(labels)}
        answer = str(row["answerKey"])
        if answer not in mapping:
            continue
        prompt = "Question: " + row["question"] + "\nAnswer:"
        rows.append({"prompt": prompt, "choices": texts, "label": mapping[answer]})
        if len(rows) >= EVAL_ARC:
            break
    return rows


def hella_examples() -> list[dict]:
    ds = load_dataset("Rowan/hellaswag", split="validation")
    rows = []
    for row in ds:
        label = int(row["label"])
        if label < 0:
            continue
        prompt = row["ctx"].strip()
        rows.append({"prompt": prompt, "choices": list(row["endings"]), "label": label})
        if len(rows) >= EVAL_HELLA:
            break
    return rows


@torch.inference_mode()
def evaluate_mcq(model, tokenizer, examples: list[dict]) -> tuple[float, float]:
    correct = 0
    correct_nll = []
    for row in examples:
        nlls = option_nlls(model, tokenizer, row["prompt"], row["choices"])
        pred = int(np.argmin(nlls))
        correct += int(pred == row["label"])
        correct_nll.append(float(nlls[row["label"]]))
    return correct / len(examples), float(np.mean(correct_nll))


def evaluate_model(name: str, model, tokenizer, wiki_texts, arc, hella) -> dict:
    start = time.time()
    raw_nll = evaluate_wikitext(model, tokenizer, wiki_texts)
    arc_acc, arc_nll = evaluate_mcq(model, tokenizer, arc)
    hella_acc, hella_nll = evaluate_mcq(model, tokenizer, hella)
    result = {
        "model": name,
        "wiki_nll": raw_nll,
        "arc_accuracy": arc_acc,
        "arc_correct_nll": arc_nll,
        "hellaswag_accuracy": hella_acc,
        "hellaswag_correct_nll": hella_nll,
        "mean_mcq_accuracy": (arc_acc + hella_acc) * 0.5,
        "seconds": time.time() - start,
    }
    print(json.dumps(result, indent=2))
    return result


def compare_logits(reference: list[torch.Tensor], candidate: list[torch.Tensor]) -> dict:
    diffs = []
    refs = []
    for a, b in zip(reference, candidate):
        d = (a - b).reshape(-1)
        diffs.append(d)
        refs.append(a.reshape(-1))
    d = torch.cat(diffs)
    r = torch.cat(refs)
    return {
        "max_abs": float(d.abs().max()),
        "rms": float(d.square().mean().sqrt()),
        "relative_rms": float(d.square().mean().sqrt() / (r.square().mean().sqrt() + 1e-30)),
    }


def main() -> None:
    config_a = AutoConfig.from_pretrained(BASE_ID)
    config_b = AutoConfig.from_pretrained(INSTRUCT_ID)
    audit_fields = [
        "hidden_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "vocab_size",
        "hidden_act", "rope_theta", "tie_word_embeddings",
    ]
    config_audit = {f: [getattr(config_a, f), getattr(config_b, f)] for f in audit_fields}
    if any(v[0] != v[1] for v in config_audit.values()):
        raise RuntimeError({"config_mismatch": config_audit})

    tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
    tokenizer_b = AutoTokenizer.from_pretrained(INSTRUCT_ID)
    vocab_equal = tokenizer.get_vocab() == tokenizer_b.get_vocab()
    probes = ["Hello world", "2 + 2 =", "Question: What is water?", "def add(a, b):"]
    probe_equal = all(
        encode_no_special(tokenizer, p) == encode_no_special(tokenizer_b, p)
        for p in probes
    )
    if not vocab_equal or not probe_equal:
        raise RuntimeError({"vocab_equal": vocab_equal, "probe_equal": probe_equal})
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading public parents...")
    model_a = AutoModelForCausalLM.from_pretrained(BASE_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()
    model_b = AutoModelForCausalLM.from_pretrained(INSTRUCT_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()

    state_a = clone_float_state(model_a)
    state_b = clone_float_state(model_b)
    if state_a.keys() != state_b.keys():
        raise RuntimeError("state keys differ")

    logit_prompts = ["Once upon a time", "The capital of France is", "Question: 3+5=", "def fibonacci(n):"]
    ref_a = prompt_logits(model_a, tokenizer, logit_prompts)
    ref_b = prompt_logits(model_b, tokenizer, logit_prompts)

    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    wiki_texts = [x["text"] for x in wiki if x["text"].strip()][:300]
    arc = arc_examples()
    hella = hella_examples()

    results = []
    results.append(evaluate_model("parent_base", model_a, tokenizer, wiki_texts, arc, hella))
    results.append(evaluate_model("parent_instruct", model_b, tokenizer, wiki_texts, arc, hella))

    del model_a, model_b
    gc.collect()

    means, qdelta, scales, total_n, energy = build_field(state_a, state_b)
    parent_bf16_bytes = total_n * 2
    field_payload_bytes = sum(map(tensor_bytes, means.values())) + sum(map(tensor_bytes, qdelta.values())) + sum(map(tensor_bytes, scales.values()))

    # Write the actual mixed-precision field once to measure real serialized size.
    field_tensors = {}
    for key in means:
        safe = key.replace("::", "__")
        field_tensors[f"mean::{safe}"] = means[key]
        field_tensors[f"qdelta::{safe}"] = qdelta[key]
        field_tensors[f"scale::{safe}"] = scales[key]
    field_path = ROOT / "TQWF_REAL_FIELD.safetensors"
    save_file(field_tensors, str(field_path), metadata={"base": BASE_ID, "instruct": INSTRUCT_ID})
    actual_field_bytes = field_path.stat().st_size

    L = int(config_a.num_hidden_layers)
    t = np.linspace(1.0 / (L + 1), L / (L + 1), L)
    layer_energy = np.array([energy[i] for i in range(L)])
    order = np.argsort(np.argsort(layer_energy))
    percentile = (order + 0.5) / L

    schedules: dict[str, tuple[list[float], float, float]] = {
        "fixed_midpoint": ([0.0] * L, 0.0, 0.0),
        "fixed_linear_base_to_instruct": ((-1.0 + 2.0 * t).tolist(), -1.0, 1.0),
        "fixed_smooth_base_to_instruct": ((-1.0 + 2.0 * (3.0 * t**2 - 2.0 * t**3)).tolist(), -1.0, 1.0),
        "fixed_late_instruct": ((np.where(t < 0.60, -1.0, 1.0)).tolist(), -1.0, 1.0),
        "fixed_midpoint_to_instruct": (t.tolist(), 0.0, 1.0),
        "fixed_weight_energy_schedule": ((-1.0 + 2.0 * percentile).tolist(), -1.0, 1.0),
    }

    # Quantized endpoint fidelity.
    endpoint_a_state = materialize(means, qdelta, scales, [-1.0] * L, -1.0, -1.0)
    endpoint_a = load_from_state(config_a, endpoint_a_state)
    endpoint_a_logits = prompt_logits(endpoint_a, tokenizer, logit_prompts)
    endpoint_a_audit = compare_logits(ref_a, endpoint_a_logits)
    del endpoint_a, endpoint_a_state, endpoint_a_logits
    gc.collect()

    endpoint_b_state = materialize(means, qdelta, scales, [1.0] * L, 1.0, 1.0)
    endpoint_b = load_from_state(config_a, endpoint_b_state)
    endpoint_b_logits = prompt_logits(endpoint_b, tokenizer, logit_prompts)
    endpoint_b_audit = compare_logits(ref_b, endpoint_b_logits)
    del endpoint_b, endpoint_b_state, endpoint_b_logits
    gc.collect()

    for name, (schedule, embed_q, head_q) in schedules.items():
        print("Materializing", name)
        state = materialize(means, qdelta, scales, schedule, embed_q, head_q)
        model = load_from_state(config_a, state)
        row = evaluate_model(name, model, tokenizer, wiki_texts, arc, hella)
        row["schedule"] = schedule
        row["embed_q"] = embed_q
        row["head_q"] = head_q
        results.append(row)
        del model, state
        gc.collect()
        RESULT_JSON.write_text(json.dumps({"partial_results": results}, indent=2), encoding="utf-8")

    parent_rows = [r for r in results if r["model"].startswith("parent_")]
    best_domain = {
        "wiki_nll": min(r["wiki_nll"] for r in parent_rows),
        "arc_correct_nll": min(r["arc_correct_nll"] for r in parent_rows),
        "hellaswag_correct_nll": min(r["hellaswag_correct_nll"] for r in parent_rows),
    }
    for r in results:
        r["balanced_relative_nll"] = float(np.mean([
            r["wiki_nll"] / best_domain["wiki_nll"],
            r["arc_correct_nll"] / best_domain["arc_correct_nll"],
            r["hellaswag_correct_nll"] / best_domain["hellaswag_correct_nll"],
        ]))

    best_parent = min(parent_rows, key=lambda r: r["balanced_relative_nll"])
    fixed_rows = [r for r in results if r["model"].startswith("fixed_")]
    best_fixed = min(fixed_rows, key=lambda r: r["balanced_relative_nll"])
    no_large_regression = all(
        best_fixed[k] <= 1.03 * best_domain[k]
        for k in ["wiki_nll", "arc_correct_nll", "hellaswag_correct_nll"]
    )
    promoted = (
        best_fixed["balanced_relative_nll"] < best_parent["balanced_relative_nll"] - 0.001
        and no_large_regression
        and best_fixed["mean_mcq_accuracy"] >= best_parent["mean_mcq_accuracy"] - 0.01
        and endpoint_a_audit["relative_rms"] < 0.02
        and endpoint_b_audit["relative_rms"] < 0.02
    )

    summary = {
        "status": "REAL_PUBLIC_FIXED_MODEL_PASS" if promoted else "REAL_PUBLIC_NOT_PROMOTED",
        "parents": [BASE_ID, INSTRUCT_ID],
        "config_audit": config_audit,
        "tokenizer_vocab_equal": vocab_equal,
        "tokenizer_probe_equal": probe_equal,
        "evaluation": {"wiki_blocks": WIKI_BLOCKS, "arc_examples": len(arc), "hellaswag_examples": len(hella)},
        "weight_field": {
            "parameter_count": total_n,
            "parent_bf16_bytes": parent_bf16_bytes,
            "payload_bytes": field_payload_bytes,
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
    RESULT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    csv_path = ROOT / "METRICS.csv"
    fields = [
        "model", "wiki_nll", "arc_accuracy", "arc_correct_nll",
        "hellaswag_accuracy", "hellaswag_correct_nll", "mean_mcq_accuracy",
        "balanced_relative_nll", "seconds",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k) for k in fields})

    report = [
        "# Real SmolLM2 Weight-Field Validation",
        "",
        f"Status: **{summary['status']}**",
        "",
        f"Parents: `{BASE_ID}` and `{INSTRUCT_ID}`",
        f"Actual mixed-precision field ratio vs one BF16 parent: **{summary['weight_field']['actual_ratio_vs_bf16_parent']:.6f}x**",
        f"Endpoint base relative RMS logit error: **{endpoint_a_audit['relative_rms']:.6g}**",
        f"Endpoint instruct relative RMS logit error: **{endpoint_b_audit['relative_rms']:.6g}**",
        f"Best parent: **{best_parent['model']}** ({best_parent['balanced_relative_nll']:.6f})",
        f"Best fixed new model: **{best_fixed['model']}** ({best_fixed['balanced_relative_nll']:.6f})",
        f"Delta fixed vs best parent: **{summary['delta_fixed_vs_best_parent']:+.6f}**",
        "",
        "A prompt-conditioned endpoint selector is not counted as a new autonomous model. Promotion is based only on the fixed single-tensor candidates.",
    ]
    (ROOT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")

    if promoted:
        schedule, embed_q, head_q = schedules[best_fixed["model"]]
        state = materialize(means, qdelta, scales, schedule, embed_q, head_q)
        model = load_from_state(config_a, state)
        model_dir = ROOT / "PROMOTED_MODEL"
        model.save_pretrained(model_dir, safe_serialization=True)
        tokenizer.save_pretrained(model_dir)
        (model_dir / "PROMOTION.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        del model, state

    # Results artifact should remain small. The 1.5x field is measured then removed.
    field_path.unlink(missing_ok=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
