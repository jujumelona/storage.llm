from __future__ import annotations

import csv
import gc
import importlib.util
import json
import math
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


v1 = load_module("wf_v1", "real_smollm2_weight_field.py")
v5 = load_module("wf_v5", "real_smollm2_v5_strict.py")
v7 = load_module("wf_v7", "real_smollm2_v7_batched_replication.py")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
torch.set_num_threads(min(4, os.cpu_count() or 1))
torch.manual_seed(20260726)
np.random.seed(20260726)

ROOT = Path("out/real_smollm2_v8_proper_choice_loss")
ROOT.mkdir(parents=True, exist_ok=True)
BASE_ID = v1.BASE_ID
INSTRUCT_ID = v1.INSTRUCT_ID
N = 96
BOOTSTRAPS = 5000
BATCH_SEQUENCES = 16
v1.WIKI_BLOCKS = 16
v1.BLOCK_SIZE = 256


@torch.inference_mode()
def evaluate_mcq_proper(model, tokenizer, examples: list[dict]):
    flat = []
    for example_id, row in enumerate(examples):
        prompt_ids = v1.encode_no_special(tokenizer, row["prompt"])
        if not prompt_ids:
            prompt_ids = [tokenizer.eos_token_id]
        for choice_id, choice in enumerate(row["choices"]):
            option_ids = v1.encode_no_special(tokenizer, " " + choice)
            if not option_ids:
                option_ids = [tokenizer.eos_token_id]
            flat.append({
                "example_id": example_id,
                "choice_id": choice_id,
                "sequence": prompt_ids + option_ids,
                "start": len(prompt_ids),
            })

    scores = {i: {} for i in range(len(examples))}
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    for batch_start in range(0, len(flat), BATCH_SEQUENCES):
        items = flat[batch_start : batch_start + BATCH_SEQUENCES]
        max_len = max(len(item["sequence"]) for item in items)
        input_ids = torch.full((len(items), max_len), pad, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for i, item in enumerate(items):
            seq = item["sequence"]
            input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            attention_mask[i, : len(seq)] = 1
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits.float()
        logp = logits.log_softmax(-1)
        for i, item in enumerate(items):
            positions = torch.arange(item["start"], len(item["sequence"]), dtype=torch.long)
            pred_positions = positions - 1
            targets = input_ids[i, positions]
            token_nll = -logp[i, pred_positions, targets].mean()
            scores[item["example_id"]][item["choice_id"]] = float(token_nll)

    records = []
    for example_id, row in enumerate(examples):
        option_nll = np.array([scores[example_id][i] for i in range(len(row["choices"]))], dtype=np.float64)
        label = int(row["label"])
        prediction = int(option_nll.argmin())
        logits = -option_nll
        max_logit = float(logits.max())
        logsumexp = max_logit + math.log(float(np.exp(logits - max_logit).sum()))
        choice_ce = float(option_nll[label] + logsumexp)
        wrong = np.delete(option_nll, label)
        margin = float(wrong.min() - option_nll[label])
        records.append({
            "example_id": example_id,
            "correct_token_nll": float(option_nll[label]),
            "choice_ce": choice_ce,
            "margin": margin,
            "correct": int(prediction == label),
            "prediction": prediction,
            "label": label,
        })
    return records


def summarize(records_by_domain: dict[str, list[dict]]):
    result = {}
    for domain, rows in records_by_domain.items():
        if domain == "wikitext":
            result[domain] = {"n": len(rows), "loss": float(np.mean([row["nll"] for row in rows])), "wiki_nll": float(np.mean([row["nll"] for row in rows]))}
        else:
            result[domain] = {
                "n": len(rows),
                "loss": float(np.mean([row["choice_ce"] for row in rows])),
                "choice_ce": float(np.mean([row["choice_ce"] for row in rows])),
                "correct_token_nll": float(np.mean([row["correct_token_nll"] for row in rows])),
                "margin": float(np.mean([row["margin"] for row in rows])),
                "accuracy": float(np.mean([row["correct"] for row in rows])),
            }
    result["balanced_relative_source"] = "wiki_nll plus choice-normalized cross-entropy"
    result["mean_mcq_accuracy"] = float(np.mean([entry["accuracy"] for domain, entry in result.items() if domain != "wikitext" and isinstance(entry, dict) and "accuracy" in entry]))
    return result


def loss_array(rows: list[dict], domain: str):
    key = "nll" if domain == "wikitext" else "choice_ce"
    return np.array([row[key] for row in rows], dtype=np.float64)


def bootstrap_composite(candidate, parent, rng):
    values = np.empty(BOOTSTRAPS, dtype=np.float64)
    domains = list(candidate)
    for b in range(BOOTSTRAPS):
        relative = []
        for domain in domains:
            c = loss_array(candidate[domain], domain)
            p = loss_array(parent[domain], domain)
            idx = rng.integers(0, len(c), len(c))
            relative.append(c[idx].mean() / p[idx].mean() - 1.0)
        values[b] = float(np.mean(relative))
    return {
        "mean": float(values.mean()),
        "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "wins_fraction": float(np.mean(values < 0.0)),
    }


def bootstrap_domain(candidate_rows, parent_rows, domain, rng):
    c = loss_array(candidate_rows, domain)
    p = loss_array(parent_rows, domain)
    n = len(c)
    loss_delta = np.empty(BOOTSTRAPS, dtype=np.float64)
    accuracy_delta = None if domain == "wikitext" else np.empty(BOOTSTRAPS, dtype=np.float64)
    if accuracy_delta is not None:
        ca = np.array([row["correct"] for row in candidate_rows], dtype=np.float64)
        pa = np.array([row["correct"] for row in parent_rows], dtype=np.float64)
    for b in range(BOOTSTRAPS):
        idx = rng.integers(0, n, n)
        loss_delta[b] = c[idx].mean() - p[idx].mean()
        if accuracy_delta is not None:
            accuracy_delta[b] = ca[idx].mean() - pa[idx].mean()
    result = {
        "loss_delta_mean": float(loss_delta.mean()),
        "loss_delta_ci95": [float(np.quantile(loss_delta, 0.025)), float(np.quantile(loss_delta, 0.975))],
    }
    if accuracy_delta is not None:
        result["accuracy_delta_mean"] = float(accuracy_delta.mean())
        result["accuracy_delta_ci95"] = [float(np.quantile(accuracy_delta, 0.025)), float(np.quantile(accuracy_delta, 0.975))]
    return result


def main():
    config = AutoConfig.from_pretrained(BASE_ID)
    config_b = AutoConfig.from_pretrained(INSTRUCT_ID)
    fields = ["hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "vocab_size", "hidden_act", "rope_theta", "tie_word_embeddings"]
    config_audit = {field: [getattr(config, field), getattr(config_b, field)] for field in fields}
    if any(pair[0] != pair[1] for pair in config_audit.values()):
        raise RuntimeError({"config_mismatch": config_audit})

    tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
    chat_tokenizer = AutoTokenizer.from_pretrained(INSTRUCT_ID)
    if tokenizer.get_vocab() != chat_tokenizer.get_vocab():
        raise RuntimeError("tokenizer mismatch")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_base = AutoModelForCausalLM.from_pretrained(BASE_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()
    model_instruct = AutoModelForCausalLM.from_pretrained(INSTRUCT_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()
    state_base = v1.clone_float_state(model_base)
    state_instruct = v1.clone_float_state(model_instruct)
    candidate_state, _ = v5.fixed_candidate_state(state_base, state_instruct)
    model_candidate = v1.load_from_state(config, candidate_state)

    datasets = {
        "openbookqa_chat": v7.deterministic_sample(v7.openbookqa_rows(chat_tokenizer), N, 201),
        "commonsenseqa_chat": v7.deterministic_sample(v7.commonsenseqa_rows(chat_tokenizer), N, 202),
        "winogrande_chat": v7.deterministic_sample(v7.winogrande_rows(chat_tokenizer), N, 203),
        "piqa_chat": v7.deterministic_sample(v7.piqa_rows(chat_tokenizer), N, 204),
        "boolq_chat": v7.deterministic_sample(v7.boolq_rows(chat_tokenizer), N, 205),
    }
    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    nonempty = [row["text"] for row in wiki if row["text"].strip()]
    wiki_texts = nonempty[2200:2450]
    if not wiki_texts:
        raise RuntimeError({"reason": "empty_wikitext_range", "nonempty_count": len(nonempty)})

    models = {"parent_base": model_base, "parent_instruct": model_instruct, "fixed_candidate": model_candidate}
    all_records = {}
    summaries = {}
    for model_name, model in models.items():
        start = time.time()
        records = {"wikitext": v5.evaluate_wiki_blocks(model, tokenizer, wiki_texts)}
        if not records["wikitext"]:
            raise RuntimeError("No WikiText blocks were produced")
        for domain, rows in datasets.items():
            records[domain] = evaluate_mcq_proper(model, tokenizer, rows)
        all_records[model_name] = records
        summaries[model_name] = summarize(records)
        summaries[model_name]["seconds"] = time.time() - start
        print(model_name, json.dumps(summaries[model_name], indent=2))
        (ROOT / "PARTIAL.json").write_text(json.dumps({"summaries": summaries}, indent=2), encoding="utf-8")

    rng = np.random.default_rng(20260726)
    comparisons = {}
    for parent_name in ["parent_base", "parent_instruct"]:
        comparisons[parent_name] = {
            "composite_relative_proper_loss": bootstrap_composite(all_records["fixed_candidate"], all_records[parent_name], rng),
            "domains": {domain: bootstrap_domain(all_records["fixed_candidate"][domain], all_records[parent_name][domain], domain, rng) for domain in all_records["fixed_candidate"]},
        }

    domains = list(all_records["fixed_candidate"])
    mcq_domains = [domain for domain in domains if domain != "wikitext"]
    best_loss = {domain: min(summaries["parent_base"][domain]["loss"], summaries["parent_instruct"][domain]["loss"]) for domain in domains}
    best_accuracy = {domain: max(summaries["parent_base"][domain]["accuracy"], summaries["parent_instruct"][domain]["accuracy"]) for domain in mcq_domains}
    no_loss_regression = all(summaries["fixed_candidate"][domain]["loss"] <= 1.03 * best_loss[domain] for domain in domains)
    no_accuracy_regression = all(summaries["fixed_candidate"][domain]["accuracy"] >= best_accuracy[domain] - 0.05 for domain in mcq_domains)
    composite_significant = all(comparisons[parent]["composite_relative_proper_loss"]["ci95"][1] < 0.0 for parent in comparisons)

    prompts = ["A prime number is", "The Pacific Ocean is", "Question: 19 + 23 =", "def breadth_first_search(graph):", v7.chat_prompt(chat_tokenizer, "Give one fact about copper.")]
    rotated_state, rotated_config, rotation_audit = v5.canonical_rotation(candidate_state, config)
    model_rotated = AutoModelForCausalLM.from_config(rotated_config, torch_dtype=torch.float32)
    missing, unexpected = model_rotated.load_state_dict(rotated_state, strict=False)
    if missing or unexpected:
        raise RuntimeError({"missing": missing, "unexpected": unexpected})
    model_rotated.eval()
    rotation_logits = v5.compare_model_logits(model_candidate, model_rotated, tokenizer, prompts)
    copy_audit = v5.exact_copy_audit(rotated_state, {"base": state_base, "instruct": state_instruct})
    parent_params = int(sum(parameter.numel() for parameter in model_base.parameters()))
    rotated_params = int(sum(tensor.numel() for tensor in rotated_state.values()))
    parameter_ratio = rotated_params / parent_params
    rotation_pass = rotation_logits["relative_rms"] < 1e-5 and copy_audit["exact_copy_count"] == 0 and parameter_ratio <= 1.8
    promoted = bool(no_loss_regression and no_accuracy_regression and composite_significant and rotation_pass)

    result = {
        "status": "PROPER_CHOICE_LOSS_PASS" if promoted else "PROPER_CHOICE_LOSS_NOT_PROMOTED",
        "candidate_frozen": True,
        "objective": "WikiText token NLL plus option-normalized choice cross-entropy",
        "why_previous_correct_token_nll_is_invalid": "Absolute answer-token NLL can improve when all option probabilities rise together; it is not a proper within-question discrimination loss.",
        "evaluation": {"n_per_mcq_domain": N, "domains": list(datasets), "wikitext_blocks": len(all_records["fixed_candidate"]["wikitext"]), "bootstrap_resamples": BOOTSTRAPS},
        "config_audit": config_audit,
        "summaries": summaries,
        "comparisons": comparisons,
        "best_parent_loss": best_loss,
        "best_parent_accuracy": best_accuracy,
        "no_domain_proper_loss_regression_over_3pct": no_loss_regression,
        "no_mcq_accuracy_regression_over_5pp": no_accuracy_regression,
        "composite_proper_loss_significantly_beats_both_parents": composite_significant,
        "canonical_rotation": {**rotation_audit, "logit_audit": rotation_logits, "exact_copy_audit": copy_audit, "parameter_ratio": parameter_ratio, "pass": rotation_pass},
        "promoted": promoted,
    }
    (ROOT / "RESULTS.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    with (ROOT / "EXAMPLE_METRICS.csv").open("w", newline="", encoding="utf-8") as handle:
        fields_out = ["model", "domain", "example_id", "nll", "correct_token_nll", "choice_ce", "margin", "correct", "prediction", "label"]
        writer = csv.DictWriter(handle, fieldnames=fields_out)
        writer.writeheader()
        for model_name, domain_map in all_records.items():
            for domain, rows in domain_map.items():
                for row in rows:
                    writer.writerow({"model": model_name, "domain": domain, **row})

    (ROOT / "REPORT.md").write_text("\n".join([
        "# Proper Choice-Normalized Validation",
        "",
        f"Status: **{result['status']}**",
        f"Composite proper loss significant vs both parents: **{composite_significant}**",
        f"All proper-loss domains within 3%: **{no_loss_regression}**",
        f"All accuracies within 5pp: **{no_accuracy_regression}**",
        f"Rotation relative RMS: **{rotation_logits['relative_rms']:.8g}**",
        f"Exact copied tensors: **{copy_audit['exact_copy_count']}**",
        f"Parameter ratio: **{parameter_ratio:.6f}x**",
    ]), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
