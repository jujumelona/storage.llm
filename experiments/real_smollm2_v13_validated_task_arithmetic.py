from __future__ import annotations

import csv
import gc
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.set_num_threads(min(4, os.cpu_count() or 1))
torch.manual_seed(20260727)
np.random.seed(20260727)

ROOT = Path("out/real_smollm2_v13_validated_task_arithmetic")
ROOT.mkdir(parents=True, exist_ok=True)

BASE_ID = "HuggingFaceTB/SmolLM2-135M"
INSTRUCT_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
SQL_ID = "Ellight/code-smolLM2-135m-text-to-sql"

N_TARGET = 48
N_MCQ = 64
WIKI_BLOCKS = 12
BLOCK_SIZE = 256
MAX_LENGTH = 512
BATCH_SIZE = 8
BOOTSTRAPS = 4000
SEED = 20260727


def structural_signature(config: Any) -> dict[str, Any]:
    fields = [
        "model_type", "hidden_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "vocab_size", "hidden_act",
        "rope_theta", "rms_norm_eps", "attention_bias", "mlp_bias",
        "tie_word_embeddings",
    ]
    return {field: getattr(config, field, None) for field in fields}


def deterministic_sample(rows: list[Any], n: int, seed: int) -> list[Any]:
    if len(rows) < n:
        raise RuntimeError({"available": len(rows), "required": n})
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(rows), size=n, replace=False))
    return [rows[int(index)] for index in indices]


def tokenizer_audit(tokenizers: list[Any]) -> dict[str, Any]:
    probes = [
        "Hello world", "Question: 17 + 24 =", "SELECT name FROM users",
        "The chemical symbol for oxygen is", "Goal: open a jar",
    ]
    base_vocab = tokenizers[0].get_vocab()
    return {
        "vocab_equal": [tok.get_vocab() == base_vocab for tok in tokenizers],
        "probe_ids_equal": [
            [
                tok(text, add_special_tokens=False)["input_ids"]
                == tokenizers[0](text, add_special_tokens=False)["input_ids"]
                for text in probes
            ]
            for tok in tokenizers
        ],
        "special_ids": [
            [tok.bos_token_id, tok.eos_token_id, tok.pad_token_id]
            for tok in tokenizers
        ],
    }


def build_wikitext() -> list[str]:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [str(row["text"]) for row in dataset if str(row["text"]).strip()]
    return texts[100:1400]


def chat_prompt(tokenizer: Any, user_text: str) -> str:
    messages = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )


def build_instruction(tokenizer: Any) -> list[dict[str, str]]:
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    rows = []
    for row in dataset:
        instruction = str(row.get("instruction") or "").strip()
        extra = str(row.get("input") or "").strip()
        output = str(row.get("output") or "").strip()
        if not instruction or not output:
            continue
        user_text = instruction if not extra else instruction + "\n\nInput:\n" + extra
        rows.append({"prompt": chat_prompt(tokenizer, user_text), "target": output})
    return deterministic_sample(rows, N_TARGET, 1301)


def build_text2sql() -> list[dict[str, str]]:
    dataset = load_dataset("b-mc2/sql-create-context", split="train")
    rows = []
    for row in dataset:
        question = str(row.get("question") or "").strip()
        context = str(row.get("context") or "").strip()
        answer = str(row.get("answer") or "").strip()
        if not question or not context or not answer:
            continue
        prompt = (
            "Database schema:\n" + context
            + "\n\nQuestion: " + question
            + "\nSQL query:\n"
        )
        rows.append({"prompt": prompt, "target": answer})
    return deterministic_sample(rows, N_TARGET, 1302)


def build_openbookqa(tokenizer: Any) -> list[dict[str, Any]]:
    dataset = load_dataset("allenai/openbookqa", "main", split="validation")
    rows = []
    for row in dataset:
        labels = [str(value) for value in row["choices"]["label"]]
        choices = [str(value) for value in row["choices"]["text"]]
        mapping = {label: index for index, label in enumerate(labels)}
        answer = str(row["answerKey"])
        if answer not in mapping:
            continue
        prompt = chat_prompt(tokenizer, "Science question: " + str(row["question_stem"]) + "\nAnswer with the best choice.")
        rows.append({"prompt": prompt, "choices": choices, "label": mapping[answer]})
    return deterministic_sample(rows, N_MCQ, 1303)


def build_piqa(tokenizer: Any) -> list[dict[str, Any]]:
    dataset = load_dataset("ybisk/piqa", split="validation", trust_remote_code=True)
    rows = []
    for row in dataset:
        label = int(row["label"])
        if label not in (0, 1):
            continue
        prompt = chat_prompt(tokenizer, "Goal: " + str(row["goal"]) + "\nChoose the better solution.")
        rows.append({
            "prompt": prompt,
            "choices": [str(row["sol1"]), str(row["sol2"])],
            "label": label,
        })
    return deterministic_sample(rows, N_MCQ, 1304)


def build_boolq(tokenizer: Any) -> list[dict[str, Any]]:
    dataset = load_dataset("google/boolq", split="validation")
    rows = []
    for row in dataset:
        passage = str(row["passage"]).strip()
        question = str(row["question"]).strip()
        prompt = chat_prompt(tokenizer, "Passage:\n" + passage + "\n\nQuestion: " + question + "\nAnswer yes or no.")
        rows.append({
            "prompt": prompt,
            "choices": ["no", "yes"],
            "label": int(bool(row["answer"])),
        })
    return deterministic_sample(rows, N_MCQ, 1305)


@torch.inference_mode()
def evaluate_wiki(model: nn.Module, tokenizer: Any, texts: list[str]) -> list[dict[str, Any]]:
    ids = tokenizer("\n\n".join(texts), add_special_tokens=False)["input_ids"]
    usable = min(len(ids) - 1, WIKI_BLOCKS * BLOCK_SIZE)
    rows = []
    for block_id, start in enumerate(range(0, usable, BLOCK_SIZE)):
        chunk = ids[start : start + BLOCK_SIZE + 1]
        if len(chunk) < 64:
            continue
        x = torch.tensor([chunk[:-1]], dtype=torch.long)
        y = torch.tensor([chunk[1:]], dtype=torch.long)
        logits = model(input_ids=x, use_cache=False).logits.float()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        rows.append({"example_id": block_id, "loss": float(loss)})
    if not rows:
        raise RuntimeError("empty wikitext evaluation")
    return rows


def prepare_target(tokenizer: Any, prompt: str, target: str) -> tuple[list[int], int]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    if not target_ids:
        target_ids = [tokenizer.eos_token_id]
    target_ids = target_ids[: min(256, MAX_LENGTH - 1)]
    max_prompt = MAX_LENGTH - len(target_ids)
    prompt_ids = prompt_ids[-max_prompt:] if max_prompt > 0 else []
    if not prompt_ids:
        prompt_ids = [tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id]
    return prompt_ids + target_ids, len(prompt_ids)


@torch.inference_mode()
def evaluate_targets(model: nn.Module, tokenizer: Any, examples: list[dict[str, str]]) -> list[dict[str, Any]]:
    prepared = [prepare_target(tokenizer, row["prompt"], row["target"]) for row in examples]
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    records = []
    for batch_start in range(0, len(prepared), BATCH_SIZE):
        batch = prepared[batch_start : batch_start + BATCH_SIZE]
        max_len = max(len(sequence) for sequence, _ in batch)
        input_ids = torch.full((len(batch), max_len), pad, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for index, (sequence, _) in enumerate(batch):
            input_ids[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
            attention_mask[index, : len(sequence)] = 1
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits.float()
        logp = logits.log_softmax(-1)
        for index, (sequence, target_start) in enumerate(batch):
            positions = torch.arange(target_start, len(sequence), dtype=torch.long)
            targets = input_ids[index, positions]
            token_nll = -logp[index, positions - 1, targets]
            records.append({
                "example_id": batch_start + index,
                "loss": float(token_nll.mean()),
                "target_tokens": int(len(positions)),
            })
    return records


@torch.inference_mode()
def evaluate_mcq(model: nn.Module, tokenizer: Any, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat = []
    for example_id, row in enumerate(examples):
        prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        if not prompt_ids:
            prompt_ids = [tokenizer.eos_token_id]
        for choice_id, choice in enumerate(row["choices"]):
            choice_ids = tokenizer(" " + str(choice), add_special_tokens=False)["input_ids"]
            if not choice_ids:
                choice_ids = [tokenizer.eos_token_id]
            choice_ids = choice_ids[: min(256, MAX_LENGTH - 1)]
            prompt_trim = prompt_ids[-(MAX_LENGTH - len(choice_ids)) :]
            flat.append({
                "example_id": example_id,
                "choice_id": choice_id,
                "sequence": prompt_trim + choice_ids,
                "start": len(prompt_trim),
            })

    scores: dict[int, dict[int, float]] = {index: {} for index in range(len(examples))}
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    for batch_start in range(0, len(flat), BATCH_SIZE):
        items = flat[batch_start : batch_start + BATCH_SIZE]
        max_len = max(len(item["sequence"]) for item in items)
        input_ids = torch.full((len(items), max_len), pad, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for index, item in enumerate(items):
            input_ids[index, : len(item["sequence"])] = torch.tensor(item["sequence"], dtype=torch.long)
            attention_mask[index, : len(item["sequence"])] = 1
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits.float()
        logp = logits.log_softmax(-1)
        for index, item in enumerate(items):
            positions = torch.arange(item["start"], len(item["sequence"]), dtype=torch.long)
            targets = input_ids[index, positions]
            sequence_nll = -logp[index, positions - 1, targets].sum()
            scores[item["example_id"]][item["choice_id"]] = float(sequence_nll)

    records = []
    for example_id, row in enumerate(examples):
        nlls = np.array([scores[example_id][index] for index in range(len(row["choices"]))], dtype=np.float64)
        choice_logits = -nlls
        maximum = float(choice_logits.max())
        log_partition = maximum + math.log(float(np.exp(choice_logits - maximum).sum()))
        label = int(row["label"])
        prediction = int(np.argmin(nlls))
        records.append({
            "example_id": example_id,
            "loss": float(nlls[label] + log_partition),
            "correct": int(prediction == label),
            "prediction": prediction,
            "label": label,
        })
    return records


def evaluate_all(model: nn.Module, tokenizer: Any, datasets: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], float]:
    start = time.time()
    records = {
        "wikitext": evaluate_wiki(model, tokenizer, datasets["wikitext"]),
        "instruction": evaluate_targets(model, tokenizer, datasets["instruction"]),
        "text2sql": evaluate_targets(model, tokenizer, datasets["text2sql"]),
        "openbookqa": evaluate_mcq(model, tokenizer, datasets["openbookqa"]),
        "piqa": evaluate_mcq(model, tokenizer, datasets["piqa"]),
        "boolq": evaluate_mcq(model, tokenizer, datasets["boolq"]),
    }
    return records, time.time() - start


def summarize(records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    result = {}
    for domain, rows in records.items():
        entry = {"n": len(rows), "loss": float(np.mean([row["loss"] for row in rows]))}
        if "correct" in rows[0]:
            entry["accuracy"] = float(np.mean([row["correct"] for row in rows]))
        result[domain] = entry
    result["balanced_loss"] = float(np.mean([entry["loss"] for entry in result.values() if isinstance(entry, dict)]))
    return result


def paired_delta(candidate_rows: list[dict[str, Any]], parent_rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    candidate = np.array([row["loss"] for row in candidate_rows], dtype=np.float64)
    parent = np.array([row["loss"] for row in parent_rows], dtype=np.float64)
    if len(candidate) != len(parent):
        raise RuntimeError("paired length mismatch")
    rng = np.random.default_rng(seed)
    values = np.empty(BOOTSTRAPS, dtype=np.float64)
    for index in range(BOOTSTRAPS):
        sample = rng.integers(0, len(candidate), len(candidate))
        values[index] = float(candidate[sample].mean() - parent[sample].mean())
    return {
        "mean": float(candidate.mean() - parent.mean()),
        "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "wins_fraction": float(np.mean(values < 0.0)),
    }


def bootstrap_composite(candidate: dict[str, list[dict[str, Any]]], parent: dict[str, list[dict[str, Any]]], seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    domains = list(candidate)
    values = np.empty(BOOTSTRAPS, dtype=np.float64)
    for bootstrap_index in range(BOOTSTRAPS):
        relative = []
        for domain in domains:
            c = np.array([row["loss"] for row in candidate[domain]], dtype=np.float64)
            p = np.array([row["loss"] for row in parent[domain]], dtype=np.float64)
            sample = rng.integers(0, len(c), len(c))
            relative.append(c[sample].mean() / max(p[sample].mean(), 1e-30) - 1.0)
        values[bootstrap_index] = float(np.mean(relative))
    return {
        "mean": float(values.mean()),
        "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "wins_fraction": float(np.mean(values < 0.0)),
    }


def finite_audit(model: nn.Module) -> dict[str, Any]:
    failures = []
    with torch.inference_mode():
        for name, parameter in model.named_parameters():
            if not bool(torch.isfinite(parameter.detach()).all().item()):
                failures.append(name)
    return {"all_finite": not failures, "nonfinite_parameters": failures}


def merge_task_arithmetic(candidate: nn.Module, base: nn.Module, instruct: nn.Module, sql: nn.Module) -> dict[str, Any]:
    base_state = base.state_dict()
    instruct_state = instruct.state_dict()
    sql_state = sql.state_dict()
    candidate_state = candidate.state_dict()
    if list(base_state) != list(instruct_state) or list(base_state) != list(sql_state) or list(base_state) != list(candidate_state):
        raise RuntimeError("state key mismatch")

    max_formula_error = 0.0
    update_norm_sq = 0.0
    with torch.no_grad():
        for key in candidate_state:
            if tuple(base_state[key].shape) != tuple(instruct_state[key].shape) or tuple(base_state[key].shape) != tuple(sql_state[key].shape):
                raise RuntimeError({"shape_mismatch": key})
            merged = instruct_state[key].detach().float() + sql_state[key].detach().float() - base_state[key].detach().float()
            candidate_state[key].copy_(merged.to(candidate_state[key].dtype))
            update = merged - base_state[key].detach().float()
            update_norm_sq += float(update.square().sum())
            check = candidate_state[key].detach().float() - merged
            max_formula_error = max(max_formula_error, float(check.abs().max()))
    candidate.tie_weights()
    return {
        "formula": "W_new = W_instruct + W_sql - W_base",
        "coefficient_search": False,
        "max_abs_formula_error": max_formula_error,
        "merged_update_l2": math.sqrt(update_norm_sq),
    }


def main() -> None:
    ids = [BASE_ID, INSTRUCT_ID, SQL_ID]
    configs = [AutoConfig.from_pretrained(model_id) for model_id in ids]
    signatures = [structural_signature(config) for config in configs]
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise RuntimeError({"status": "CONFIG_MISMATCH", "signatures": signatures})

    tokenizers = [AutoTokenizer.from_pretrained(model_id) for model_id in ids]
    audit = tokenizer_audit(tokenizers)
    if not all(audit["vocab_equal"]) or not all(all(row) for row in audit["probe_ids_equal"]):
        raise RuntimeError({"status": "TOKENIZER_MISMATCH", "audit": audit})
    tokenizer = tokenizers[1]
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        raise RuntimeError("instruct tokenizer has no chat template")

    datasets = {
        "wikitext": build_wikitext(),
        "instruction": build_instruction(tokenizer),
        "text2sql": build_text2sql(),
        "openbookqa": build_openbookqa(tokenizer),
        "piqa": build_piqa(tokenizer),
        "boolq": build_boolq(tokenizer),
    }

    print("Loading aligned full checkpoints")
    base = AutoModelForCausalLM.from_pretrained(BASE_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()
    instruct = AutoModelForCausalLM.from_pretrained(INSTRUCT_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()
    sql = AutoModelForCausalLM.from_pretrained(SQL_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()
    candidate = AutoModelForCausalLM.from_pretrained(BASE_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()

    parent_parameter_count = int(sum(parameter.numel() for parameter in base.parameters()))
    merge_audit = merge_task_arithmetic(candidate, base, instruct, sql)
    candidate_parameter_count = int(sum(parameter.numel() for parameter in candidate.parameters()))

    models = {
        "parent_base": base,
        "parent_instruct": instruct,
        "parent_sql": sql,
        "task_arithmetic": candidate,
    }
    records = {}
    summaries = {}
    timings = {}
    for name, model in models.items():
        model_records, elapsed = evaluate_all(model, tokenizer, datasets)
        records[name] = model_records
        summaries[name] = summarize(model_records)
        timings[name] = elapsed
        print(name, json.dumps(summaries[name], indent=2))

    parent_gate_instruction = paired_delta(
        records["parent_instruct"]["instruction"],
        records["parent_base"]["instruction"],
        2101,
    )
    parent_gate_sql = paired_delta(
        records["parent_sql"]["text2sql"],
        records["parent_base"]["text2sql"],
        2102,
    )
    parent_gates = {
        "instruct_beats_base_on_instruction": parent_gate_instruction,
        "sql_beats_base_on_text2sql": parent_gate_sql,
        "pass": bool(parent_gate_instruction["ci95"][1] < 0.0 and parent_gate_sql["ci95"][1] < 0.0),
    }

    parents = ["parent_base", "parent_instruct", "parent_sql"]
    comparisons = {
        parent: bootstrap_composite(records["task_arithmetic"], records[parent], 2200 + index)
        for index, parent in enumerate(parents)
    }

    domains = list(records["task_arithmetic"])
    best_parent_by_domain = {
        domain: min(parents, key=lambda parent: summaries[parent][domain]["loss"])
        for domain in domains
    }
    virtual_best_records = {
        domain: records[best_parent_by_domain[domain]][domain]
        for domain in domains
    }
    comparison_virtual_best = bootstrap_composite(records["task_arithmetic"], virtual_best_records, 2301)

    best_parent_loss = {
        domain: summaries[best_parent_by_domain[domain]][domain]["loss"]
        for domain in domains
    }
    mcq_domains = ["openbookqa", "piqa", "boolq"]
    best_parent_accuracy = {
        domain: max(summaries[parent][domain]["accuracy"] for parent in parents)
        for domain in mcq_domains
    }
    loss_ratios = {
        domain: summaries["task_arithmetic"][domain]["loss"] / max(best_parent_loss[domain], 1e-30)
        for domain in domains
    }
    accuracy_deltas = {
        domain: summaries["task_arithmetic"][domain]["accuracy"] - best_parent_accuracy[domain]
        for domain in mcq_domains
    }

    no_loss_regression_over_2pct = all(value <= 1.02 for value in loss_ratios.values())
    no_accuracy_regression_over_2pp = all(value >= -0.02 for value in accuracy_deltas.values())
    beats_each_parent = all(result["ci95"][1] < 0.0 for result in comparisons.values())
    beats_virtual_best = comparison_virtual_best["ci95"][1] < 0.0
    structure_pass = bool(
        candidate_parameter_count == parent_parameter_count
        and finite_audit(candidate)["all_finite"]
        and candidate.config.tie_word_embeddings
        and hasattr(candidate, "lm_head")
        and merge_audit["max_abs_formula_error"] == 0.0
    )
    promoted = bool(
        parent_gates["pass"]
        and beats_each_parent
        and beats_virtual_best
        and no_loss_regression_over_2pct
        and no_accuracy_regression_over_2pp
        and structure_pass
    )

    result = {
        "status": "VALIDATED_TASK_ARITHMETIC_PASS" if promoted else "VALIDATED_TASK_ARITHMETIC_NOT_PROMOTED",
        "models": {"base": BASE_ID, "instruct": INSTRUCT_ID, "sql": SQL_ID},
        "method": {
            "name": "validated_parent_unit_task_arithmetic",
            "formula": "W_new = W_instruct + W_sql - W_base",
            "training": False,
            "labels_used_for_merge": False,
            "coefficient_search": False,
            "router": False,
            "probability_or_logit_mixture": False,
        },
        "config_signature": signatures[0],
        "tokenizer_audit": audit,
        "evaluation": {
            "domain_sizes": {domain: len(rows) for domain, rows in records["task_arithmetic"].items()},
            "mcq_metric": "choice-normalized total answer-sequence cross entropy",
            "target_metric": "mean target-token NLL",
            "bootstrap_resamples": BOOTSTRAPS,
        },
        "parent_gates": parent_gates,
        "merge_audit": merge_audit,
        "structure": {
            "parent_parameter_count": parent_parameter_count,
            "candidate_parameter_count": candidate_parameter_count,
            "parameter_ratio": candidate_parameter_count / parent_parameter_count,
            "single_checkpoint": True,
            "single_hidden_stream": True,
            "single_tied_lm_head": True,
            "finite_audit": finite_audit(candidate),
            "structure_pass": structure_pass,
        },
        "summaries": summaries,
        "timings_seconds": timings,
        "speed_ratio_vs_mean_parent": timings["task_arithmetic"] / float(np.mean([timings[parent] for parent in parents])),
        "comparisons_vs_each_parent": comparisons,
        "best_parent_by_domain": best_parent_by_domain,
        "comparison_vs_virtual_domain_best": comparison_virtual_best,
        "loss_ratio_vs_domain_best_parent": loss_ratios,
        "accuracy_delta_vs_domain_best_parent": accuracy_deltas,
        "no_loss_regression_over_2pct": no_loss_regression_over_2pct,
        "no_accuracy_regression_over_2pp": no_accuracy_regression_over_2pp,
        "significantly_beats_each_parent": beats_each_parent,
        "significantly_beats_virtual_domain_best": beats_virtual_best,
        "promoted": promoted,
    }

    (ROOT / "RESULTS.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (ROOT / "METRICS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "model", "domain", "example_id", "loss", "correct", "prediction", "label", "target_tokens"
        ])
        writer.writeheader()
        for model_name, domain_map in records.items():
            for domain, rows in domain_map.items():
                for row in rows:
                    writer.writerow({
                        "model": model_name,
                        "domain": domain,
                        "example_id": row.get("example_id"),
                        "loss": row.get("loss"),
                        "correct": row.get("correct"),
                        "prediction": row.get("prediction"),
                        "label": row.get("label"),
                        "target_tokens": row.get("target_tokens"),
                    })

    report = [
        "# SmolLM2 Validated-Parent Task Arithmetic",
        "",
        f"Status: **{result['status']}**",
        f"Parent specialist gate: **{parent_gates['pass']}**",
        f"Candidate parameter ratio: **{candidate_parameter_count / parent_parameter_count:.6f}x**",
        f"Speed ratio vs mean parent: **{result['speed_ratio_vs_mean_parent']:.6f}x**",
        f"Significantly beats each parent: **{beats_each_parent}**",
        f"Significantly beats virtual domain-best parent: **{beats_virtual_best}**",
        f"No domain loss regression over 2%: **{no_loss_regression_over_2pct}**",
        f"No MCQ accuracy regression over 2pp: **{no_accuracy_regression_over_2pp}**",
        "",
        "The merge is frozen before evaluation and has no learned or searched coefficients.",
    ]
    (ROOT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")

    if promoted:
        candidate.save_pretrained(ROOT / "VALIDATED_TASK_ARITHMETIC_MODEL", safe_serialization=True)
        tokenizer.save_pretrained(ROOT / "VALIDATED_TASK_ARITHMETIC_MODEL")

    del base, instruct, sql, candidate
    gc.collect()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
