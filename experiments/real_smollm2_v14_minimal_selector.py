from __future__ import annotations

import csv
import gc
import importlib.util
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

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "validated_task_arithmetic_v13", HERE / "real_smollm2_v13_validated_task_arithmetic.py"
)
v13 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(v13)

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.set_num_threads(min(4, os.cpu_count() or 1))
torch.manual_seed(20260727)
np.random.seed(20260727)

ROOT = Path("out/real_smollm2_v14_minimal_selector")
ROOT.mkdir(parents=True, exist_ok=True)

INSTRUCT_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
SQL_ID = "Ellight/code-smolLM2-135m-text-to-sql"
SEED = 20260727
CAL_N = 48
TEST_N = 48
WIKI_CAL_BLOCKS = 8
WIKI_TEST_BLOCKS = 8
BLOCK_SIZE = 256
MAX_FEATURE_TOKENS = 256
BOOTSTRAPS = 5000
DOMAIN_LOSS_TOLERANCE = 0.02
MCQ_ACCURACY_TOLERANCE = 0.02


def deterministic_partition(rows: list[Any], cal_n: int, test_n: int, seed: int) -> tuple[list[Any], list[Any]]:
    if len(rows) < cal_n + test_n:
        raise RuntimeError({"available": len(rows), "required": cal_n + test_n})
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(rows), size=cal_n + test_n, replace=False)
    return [rows[int(i)] for i in indices[:cal_n]], [rows[int(i)] for i in indices[cal_n:]]


def deterministic_sample(rows: list[Any], n: int, seed: int) -> list[Any]:
    if len(rows) < n:
        raise RuntimeError({"available": len(rows), "required": n})
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(rows), size=n, replace=False)
    return [rows[int(i)] for i in indices]


def instruction_rows(tokenizer: Any) -> list[dict[str, str]]:
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    rows = []
    for row in dataset:
        instruction = str(row.get("instruction") or "").strip()
        extra = str(row.get("input") or "").strip()
        output = str(row.get("output") or "").strip()
        if not instruction or not output:
            continue
        user_text = instruction if not extra else instruction + "\n\nInput:\n" + extra
        prompt = v13.chat_prompt(tokenizer, user_text)
        rows.append({"prompt": prompt, "target": output, "feature_text": prompt})
    return rows


def sql_rows() -> list[dict[str, str]]:
    dataset = load_dataset("b-mc2/sql-create-context", split="train")
    rows = []
    for row in dataset:
        question = str(row.get("question") or "").strip()
        context = str(row.get("context") or "").strip()
        answer = str(row.get("answer") or "").strip()
        if not question or not context or not answer:
            continue
        prompt = "Database schema:\n" + context + "\n\nQuestion: " + question + "\nSQL query:\n"
        rows.append({"prompt": prompt, "target": answer, "feature_text": prompt})
    return rows


def openbook_rows(tokenizer: Any, split: str) -> list[dict[str, Any]]:
    dataset = load_dataset("allenai/openbookqa", "main", split=split)
    rows = []
    for row in dataset:
        labels = [str(value) for value in row["choices"]["label"]]
        choices = [str(value) for value in row["choices"]["text"]]
        mapping = {label: index for index, label in enumerate(labels)}
        answer = str(row["answerKey"])
        if answer not in mapping:
            continue
        prompt = v13.chat_prompt(
            tokenizer,
            "Science question: " + str(row["question_stem"]) + "\nAnswer with the best choice.",
        )
        rows.append({"prompt": prompt, "choices": choices, "label": mapping[answer], "feature_text": prompt})
    return rows


def piqa_rows(tokenizer: Any, split: str) -> list[dict[str, Any]]:
    dataset = load_dataset("ybisk/piqa", split=split, trust_remote_code=True)
    rows = []
    for row in dataset:
        label = int(row["label"])
        if label not in (0, 1):
            continue
        prompt = v13.chat_prompt(tokenizer, "Goal: " + str(row["goal"]) + "\nChoose the better solution.")
        rows.append({
            "prompt": prompt,
            "choices": [str(row["sol1"]), str(row["sol2"])],
            "label": label,
            "feature_text": prompt,
        })
    return rows


def boolq_rows(tokenizer: Any, split: str) -> list[dict[str, Any]]:
    dataset = load_dataset("google/boolq", split=split)
    rows = []
    for row in dataset:
        passage = str(row["passage"]).strip()
        question = str(row["question"]).strip()
        prompt = v13.chat_prompt(
            tokenizer,
            "Passage:\n" + passage + "\n\nQuestion: " + question + "\nAnswer yes or no.",
        )
        rows.append({
            "prompt": prompt,
            "choices": ["no", "yes"],
            "label": int(bool(row["answer"])),
            "feature_text": prompt,
        })
    return rows


def wiki_blocks(tokenizer: Any, split: str, n: int, seed: int) -> list[dict[str, Any]]:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(str(row["text"]) for row in dataset if str(row["text"]).strip())
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    blocks = []
    for start in range(0, len(ids) - BLOCK_SIZE - 1, BLOCK_SIZE):
        chunk = ids[start : start + BLOCK_SIZE + 1]
        if len(chunk) != BLOCK_SIZE + 1:
            continue
        feature_text = tokenizer.decode(chunk[:MAX_FEATURE_TOKENS], skip_special_tokens=False)
        blocks.append({"ids": chunk, "feature_text": feature_text})
    return deterministic_sample(blocks, n, seed)


def build_datasets(tokenizer: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    instruction_cal, instruction_test = deterministic_partition(instruction_rows(tokenizer), CAL_N, TEST_N, 1401)
    sql_cal, sql_test = deterministic_partition(sql_rows(), CAL_N, TEST_N, 1402)
    calibration = {
        "wikitext": wiki_blocks(tokenizer, "train", WIKI_CAL_BLOCKS, 1400),
        "instruction": instruction_cal,
        "text2sql": sql_cal,
        "openbookqa": deterministic_sample(openbook_rows(tokenizer, "train"), CAL_N, 1403),
        "piqa": deterministic_sample(piqa_rows(tokenizer, "train"), CAL_N, 1404),
        "boolq": deterministic_sample(boolq_rows(tokenizer, "train"), CAL_N, 1405),
    }
    heldout = {
        "wikitext": wiki_blocks(tokenizer, "test", WIKI_TEST_BLOCKS, 2400),
        "instruction": instruction_test,
        "text2sql": sql_test,
        "openbookqa": deterministic_sample(openbook_rows(tokenizer, "validation"), TEST_N, 2403),
        "piqa": deterministic_sample(piqa_rows(tokenizer, "validation"), TEST_N, 2404),
        "boolq": deterministic_sample(boolq_rows(tokenizer, "validation"), TEST_N, 2405),
    }
    return calibration, heldout


@torch.inference_mode()
def evaluate_wiki_blocks(model: nn.Module, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for index, row in enumerate(examples):
        ids = row["ids"]
        x = torch.tensor([ids[:-1]], dtype=torch.long)
        y = torch.tensor([ids[1:]], dtype=torch.long)
        logits = model(input_ids=x, use_cache=False).logits.float()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        records.append({"example_id": index, "loss": float(loss)})
    return records


def evaluate_all(model: nn.Module, tokenizer: Any, datasets: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], float]:
    start = time.time()
    records = {
        "wikitext": evaluate_wiki_blocks(model, datasets["wikitext"]),
        "instruction": v13.evaluate_targets(model, tokenizer, datasets["instruction"]),
        "text2sql": v13.evaluate_targets(model, tokenizer, datasets["text2sql"]),
        "openbookqa": v13.evaluate_mcq(model, tokenizer, datasets["openbookqa"]),
        "piqa": v13.evaluate_mcq(model, tokenizer, datasets["piqa"]),
        "boolq": v13.evaluate_mcq(model, tokenizer, datasets["boolq"]),
    }
    return records, time.time() - start


@torch.inference_mode()
def prompt_features(
    embedding: torch.Tensor,
    tokenizer: Any,
    datasets: dict[str, Any],
) -> tuple[np.ndarray, list[tuple[str, int]]]:
    values = []
    keys = []
    weight = embedding.detach().cpu().float()
    for domain, examples in datasets.items():
        for index, row in enumerate(examples):
            ids = tokenizer(
                str(row["feature_text"]),
                add_special_tokens=False,
                truncation=True,
                max_length=MAX_FEATURE_TOKENS,
            )["input_ids"]
            if not ids:
                ids = [tokenizer.eos_token_id]
            token_ids = torch.tensor(ids, dtype=torch.long)
            vector = weight[token_ids].mean(dim=0)
            values.append(vector.numpy())
            keys.append((domain, index))
    return np.stack(values).astype(np.float64), keys


def oracle_labels(
    instruct_records: dict[str, list[dict[str, Any]]],
    sql_records: dict[str, list[dict[str, Any]]],
    keys: list[tuple[str, int]],
) -> np.ndarray:
    labels = []
    for domain, index in keys:
        li = float(instruct_records[domain][index]["loss"])
        ls = float(sql_records[domain][index]["loss"])
        labels.append(1.0 if ls < li else -1.0)
    return np.array(labels, dtype=np.float64)


def domain_equal_weights(keys: list[tuple[str, int]]) -> np.ndarray:
    counts: dict[str, int] = {}
    for domain, _ in keys:
        counts[domain] = counts.get(domain, 0) + 1
    weights = np.array([1.0 / counts[domain] for domain, _ in keys], dtype=np.float64)
    weights *= len(weights) / weights.sum()
    return weights


def fit_ridge_gcv(x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> dict[str, Any]:
    feature_mean = x.mean(axis=0)
    feature_std = x.std(axis=0)
    feature_std[feature_std < 1e-8] = 1.0
    z = (x - feature_mean) / feature_std
    design = np.concatenate([z, np.ones((len(z), 1), dtype=np.float64)], axis=1)

    sqrt_weight = np.sqrt(sample_weight)[:, None]
    xw = design * sqrt_weight
    yw = y * sqrt_weight[:, 0]
    u, singular_values, vh = np.linalg.svd(xw, full_matrices=False)
    scale = max(float(singular_values[0] ** 2), 1e-30)
    lambdas = scale * np.power(10.0, np.linspace(-8.0, 2.0, 41))

    best = None
    uy = u.T @ yw
    for lam in lambdas:
        shrink = singular_values / (singular_values ** 2 + lam)
        coefficient = vh.T @ (shrink * uy)
        fitted = xw @ coefficient
        residual = float(np.mean((yw - fitted) ** 2))
        degrees = float(np.sum(singular_values ** 2 / (singular_values ** 2 + lam)))
        denominator = max((1.0 - degrees / len(y)) ** 2, 1e-12)
        gcv = residual / denominator
        candidate = (gcv, float(lam), coefficient, degrees)
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    _, lam, coefficient, degrees = best
    calibration_scores = design @ coefficient
    calibration_accuracy = float(np.mean(np.sign(calibration_scores) == y))
    return {
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "coefficient": coefficient,
        "lambda": lam,
        "effective_degrees_of_freedom": degrees,
        "calibration_accuracy": calibration_accuracy,
        "gcv": float(best[0]),
    }


def selector_scores(x: np.ndarray, fit: dict[str, Any]) -> np.ndarray:
    z = (x - fit["feature_mean"]) / fit["feature_std"]
    design = np.concatenate([z, np.ones((len(z), 1), dtype=np.float64)], axis=1)
    return design @ fit["coefficient"]


def combine_records(
    instruct_records: dict[str, list[dict[str, Any]]],
    sql_records: dict[str, list[dict[str, Any]]],
    choices_by_key: dict[tuple[str, int], int],
) -> dict[str, list[dict[str, Any]]]:
    combined: dict[str, list[dict[str, Any]]] = {}
    for domain in instruct_records:
        rows = []
        for index, (ri, rs) in enumerate(zip(instruct_records[domain], sql_records[domain])):
            source = rs if choices_by_key[(domain, index)] == 1 else ri
            row = dict(source)
            row["selected_parent"] = "sql" if choices_by_key[(domain, index)] == 1 else "instruct"
            rows.append(row)
        combined[domain] = rows
    return combined


def oracle_records(
    instruct_records: dict[str, list[dict[str, Any]]],
    sql_records: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[tuple[str, int], int]]:
    choices = {}
    for domain in instruct_records:
        for index, (ri, rs) in enumerate(zip(instruct_records[domain], sql_records[domain])):
            choices[(domain, index)] = int(float(rs["loss"]) < float(ri["loss"]))
    return combine_records(instruct_records, sql_records, choices), choices


def domain_best_records(
    instruct_records: dict[str, list[dict[str, Any]]],
    sql_records: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    result = {}
    selected = {}
    for domain in instruct_records:
        li = float(np.mean([row["loss"] for row in instruct_records[domain]]))
        ls = float(np.mean([row["loss"] for row in sql_records[domain]]))
        if ls < li:
            result[domain] = [dict(row) for row in sql_records[domain]]
            selected[domain] = "sql"
        else:
            result[domain] = [dict(row) for row in instruct_records[domain]]
            selected[domain] = "instruct"
    return result, selected


def bootstrap_composite(
    candidate: dict[str, list[dict[str, Any]]],
    parent: dict[str, list[dict[str, Any]]],
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    domains = list(candidate)
    values = np.empty(BOOTSTRAPS, dtype=np.float64)
    for b in range(BOOTSTRAPS):
        relative = []
        for domain in domains:
            c = np.array([row["loss"] for row in candidate[domain]], dtype=np.float64)
            p = np.array([row["loss"] for row in parent[domain]], dtype=np.float64)
            index = rng.integers(0, len(c), len(c))
            relative.append(c[index].mean() / max(p[index].mean(), 1e-30) - 1.0)
        values[b] = float(np.mean(relative))
    return {
        "mean": float(values.mean()),
        "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "wins_fraction": float(np.mean(values < 0.0)),
    }


def summarize(records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    result = {}
    for domain, rows in records.items():
        entry = {"n": len(rows), "loss": float(np.mean([row["loss"] for row in rows]))}
        if "correct" in rows[0]:
            entry["accuracy"] = float(np.mean([row["correct"] for row in rows]))
        result[domain] = entry
    result["balanced_loss"] = float(np.mean([result[d]["loss"] for d in records]))
    return result


def route_statistics(
    choices: dict[tuple[str, int], int],
    oracle_choices: dict[tuple[str, int], int],
) -> dict[str, Any]:
    domains = sorted({domain for domain, _ in choices})
    result = {}
    total_correct = 0
    total = 0
    for domain in domains:
        keys = sorted([key for key in choices if key[0] == domain], key=lambda item: item[1])
        predicted = np.array([choices[key] for key in keys], dtype=np.int64)
        oracle = np.array([oracle_choices[key] for key in keys], dtype=np.int64)
        correct = int((predicted == oracle).sum())
        total_correct += correct
        total += len(keys)
        result[domain] = {
            "n": len(keys),
            "selector_oracle_accuracy": correct / len(keys),
            "selector_sql_fraction": float(predicted.mean()),
            "oracle_sql_fraction": float(oracle.mean()),
        }
    result["overall"] = {"n": total, "selector_oracle_accuracy": total_correct / total}
    return result


def save_single_checkpoint(
    instruct: nn.Module,
    sql: nn.Module,
    tokenizer: Any,
    fit: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    path = ROOT / "PROMOTED_DUAL_SELECTOR_CHECKPOINT.pt"
    payload = {
        "format": "smollm2_conditional_dual_body_v1",
        "architecture": "sequence-level selector; exactly one full parent executes per input",
        "model_ids_for_provenance_only": {"instruct": INSTRUCT_ID, "sql": SQL_ID},
        "config": instruct.config.to_dict(),
        "tokenizer_json": tokenizer.backend_tokenizer.to_str(),
        "tokenizer_special_tokens_map": tokenizer.special_tokens_map,
        "selector": {
            "feature_definition": "mean of instruct embedding vectors over first 256 prompt tokens",
            "positive_route": "sql",
            "feature_mean": torch.tensor(fit["feature_mean"], dtype=torch.float32),
            "feature_std": torch.tensor(fit["feature_std"], dtype=torch.float32),
            "coefficient": torch.tensor(fit["coefficient"], dtype=torch.float32),
            "lambda": float(fit["lambda"]),
        },
        "instruct_state_dict_fp16": {
            key: value.detach().cpu().to(torch.float16).contiguous()
            for key, value in instruct.state_dict().items()
        },
        "sql_state_dict_fp16": {
            key: value.detach().cpu().to(torch.float16).contiguous()
            for key, value in sql.state_dict().items()
        },
        "validation_summary": metrics,
    }
    torch.save(payload, path)
    return {"path": str(path), "size_bytes": path.stat().st_size}


def main() -> None:
    configs = [AutoConfig.from_pretrained(model_id) for model_id in [INSTRUCT_ID, SQL_ID]]
    signatures = [v13.structural_signature(config) for config in configs]
    if signatures[0] != signatures[1]:
        raise RuntimeError({"status": "CONFIG_MISMATCH", "signatures": signatures})

    tokenizers = [AutoTokenizer.from_pretrained(model_id) for model_id in [INSTRUCT_ID, SQL_ID]]
    tokenizer_audit = v13.tokenizer_audit(tokenizers)
    if not all(tokenizer_audit["vocab_equal"]) or not all(all(row) for row in tokenizer_audit["probe_ids_equal"]):
        raise RuntimeError({"status": "TOKENIZER_MISMATCH", "audit": tokenizer_audit})
    tokenizer = tokenizers[0]
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        raise RuntimeError("instruct tokenizer has no chat template")

    calibration, heldout = build_datasets(tokenizer)

    print("Loading two validated full checkpoints")
    instruct = AutoModelForCausalLM.from_pretrained(
        INSTRUCT_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True
    ).eval()
    sql = AutoModelForCausalLM.from_pretrained(
        SQL_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True
    ).eval()
    if {key: tuple(value.shape) for key, value in instruct.state_dict().items()} != {
        key: tuple(value.shape) for key, value in sql.state_dict().items()
    }:
        raise RuntimeError("state shape mismatch")

    calibration_records = {}
    heldout_records = {}
    timings = {}
    for name, model in [("instruct", instruct), ("sql", sql)]:
        calibration_records[name], timings[name + "_calibration_seconds"] = evaluate_all(model, tokenizer, calibration)
        heldout_records[name], timings[name + "_heldout_seconds"] = evaluate_all(model, tokenizer, heldout)
        print(name, summarize(calibration_records[name]), summarize(heldout_records[name]))

    calibration_oracle, calibration_oracle_choices = oracle_records(
        calibration_records["instruct"], calibration_records["sql"]
    )
    heldout_oracle, heldout_oracle_choices = oracle_records(
        heldout_records["instruct"], heldout_records["sql"]
    )
    heldout_domain_best, domain_best_parent = domain_best_records(
        heldout_records["instruct"], heldout_records["sql"]
    )

    oracle_bootstrap = {
        "vs_instruct": bootstrap_composite(heldout_oracle, heldout_records["instruct"], SEED + 1),
        "vs_sql": bootstrap_composite(heldout_oracle, heldout_records["sql"], SEED + 2),
    }
    oracle_gate = all(value["ci95"][1] < 0.0 for value in oracle_bootstrap.values())

    x_cal, keys_cal = prompt_features(
        instruct.model.embed_tokens.weight, tokenizer, calibration
    )
    y_cal = oracle_labels(
        calibration_records["instruct"], calibration_records["sql"], keys_cal
    )
    fit = fit_ridge_gcv(x_cal, y_cal, domain_equal_weights(keys_cal))

    x_test, keys_test = prompt_features(
        instruct.model.embed_tokens.weight, tokenizer, heldout
    )
    scores = selector_scores(x_test, fit)
    selector_choices = {
        key: int(score > 0.0) for key, score in zip(keys_test, scores)
    }
    selected_records = combine_records(
        heldout_records["instruct"], heldout_records["sql"], selector_choices
    )

    selector_bootstrap = {
        "vs_instruct": bootstrap_composite(selected_records, heldout_records["instruct"], SEED + 11),
        "vs_sql": bootstrap_composite(selected_records, heldout_records["sql"], SEED + 12),
        "vs_domain_best": bootstrap_composite(selected_records, heldout_domain_best, SEED + 13),
        "vs_oracle": bootstrap_composite(selected_records, heldout_oracle, SEED + 14),
    }

    summaries = {
        "instruct": summarize(heldout_records["instruct"]),
        "sql": summarize(heldout_records["sql"]),
        "domain_best": summarize(heldout_domain_best),
        "oracle": summarize(heldout_oracle),
        "selector": summarize(selected_records),
    }

    domain_gates = {}
    for domain in selected_records:
        selector_entry = summaries["selector"][domain]
        best_loss = min(summaries["instruct"][domain]["loss"], summaries["sql"][domain]["loss"])
        loss_pass = selector_entry["loss"] <= best_loss * (1.0 + DOMAIN_LOSS_TOLERANCE)
        accuracy_pass = True
        if "accuracy" in selector_entry:
            best_accuracy = max(
                summaries["instruct"][domain]["accuracy"],
                summaries["sql"][domain]["accuracy"],
            )
            accuracy_pass = selector_entry["accuracy"] >= best_accuracy - MCQ_ACCURACY_TOLERANCE
        domain_gates[domain] = {
            "loss_ratio_vs_best_parent": selector_entry["loss"] / max(best_loss, 1e-30),
            "loss_pass": bool(loss_pass),
            "accuracy_pass": bool(accuracy_pass),
        }

    selector_parent_gate = (
        selector_bootstrap["vs_instruct"]["ci95"][1] < 0.0
        and selector_bootstrap["vs_sql"]["ci95"][1] < 0.0
    )
    domain_gate = all(
        entry["loss_pass"] and entry["accuracy_pass"] for entry in domain_gates.values()
    )
    promoted = bool(oracle_gate and selector_parent_gate and domain_gate)

    parent_parameter_count = int(sum(parameter.numel() for parameter in instruct.parameters()))
    selector_parameter_count = int(len(fit["coefficient"]) + 2 * len(fit["feature_mean"]))
    compiled_parameter_count = 2 * parent_parameter_count + selector_parameter_count

    result = {
        "status": "MINIMAL_SELECTOR_PROMOTED" if promoted else "MINIMAL_SELECTOR_NOT_PROMOTED",
        "models": {"instruct": INSTRUCT_ID, "sql": SQL_ID},
        "architecture": {
            "single_checkpoint": bool(promoted),
            "sequence_level_selector": True,
            "one_parent_body_executed_per_input": True,
            "runtime_parent_dependencies": False if promoted else None,
            "logit_or_probability_ensemble": False,
            "persistent_hidden_streams": 1,
            "parameter_count": compiled_parameter_count,
            "parent_parameter_count": parent_parameter_count,
            "parameter_ratio": compiled_parameter_count / parent_parameter_count,
            "selector_parameter_count": selector_parameter_count,
            "expected_forward_compute_ratio": "approximately 1.0 parent forward plus mean-embedding selector",
        },
        "calibration_information": {
            "examples": len(keys_cal),
            "binary_oracle_labels": len(keys_cal),
            "minimum_supervision_bits_used": len(keys_cal),
            "domains": {domain: len(rows) for domain, rows in calibration.items()},
            "heldout_domains": {domain: len(rows) for domain, rows in heldout.items()},
            "no_heldout_labels_used_for_fit": True,
        },
        "selector_fit": {
            "lambda_gcv": float(fit["lambda"]),
            "effective_degrees_of_freedom": float(fit["effective_degrees_of_freedom"]),
            "calibration_oracle_label_accuracy": float(fit["calibration_accuracy"]),
            "gcv": float(fit["gcv"]),
        },
        "oracle_gate": {"pass": oracle_gate, "bootstrap": oracle_bootstrap},
        "selector_bootstrap": selector_bootstrap,
        "domain_gates": domain_gates,
        "route_statistics": route_statistics(selector_choices, heldout_oracle_choices),
        "domain_best_parent": domain_best_parent,
        "summaries": summaries,
        "timings": timings,
        "promotion_gates": {
            "oracle_significantly_beats_both_parents": oracle_gate,
            "selector_significantly_beats_both_parents": selector_parent_gate,
            "every_domain_within_2pct_loss_and_2pp_accuracy_of_best_parent": domain_gate,
            "all_pass": promoted,
        },
    }

    checkpoint = None
    if promoted:
        checkpoint = save_single_checkpoint(instruct, sql, tokenizer, fit, {
            "status": result["status"],
            "summaries": summaries,
            "selector_bootstrap": selector_bootstrap,
            "domain_gates": domain_gates,
        })
    result["checkpoint"] = checkpoint

    (ROOT / "RESULTS.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    metric_rows = []
    for model_name, summary in summaries.items():
        for domain, entry in summary.items():
            if domain == "balanced_loss":
                continue
            metric_rows.append({
                "model": model_name,
                "domain": domain,
                "n": entry["n"],
                "loss": entry["loss"],
                "accuracy": entry.get("accuracy", ""),
            })
    with (ROOT / "METRICS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "domain", "n", "loss", "accuracy"])
        writer.writeheader()
        writer.writerows(metric_rows)

    report = [
        "# SmolLM2 Minimal-Bit Sequence Selector",
        "",
        f"Status: **{result['status']}**",
        f"Calibration examples / minimum binary labels: **{len(keys_cal)}**",
        f"Parameter ratio: **{result['architecture']['parameter_ratio']:.6f}x**",
        f"Calibration oracle-label accuracy: **{fit['calibration_accuracy']:.4f}**",
        f"Held-out selector oracle-label accuracy: **{result['route_statistics']['overall']['selector_oracle_accuracy']:.4f}**",
        f"Oracle gate: **{oracle_gate}**",
        f"Selector beats both parents gate: **{selector_parent_gate}**",
        f"Every-domain preservation gate: **{domain_gate}**",
        "",
        "The selector receives only the input prompt. It never sees candidate logits, target labels, or both parent outputs at inference. Exactly one parent body executes per input.",
    ]
    (ROOT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")

    print(json.dumps(result, indent=2))
    del instruct, sql
    gc.collect()


if __name__ == "__main__":
    main()
