from __future__ import annotations

import csv
import gc
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "minimal_selector_v14", HERE / "real_smollm2_v14_minimal_selector.py"
)
v14 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(v14)

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.set_num_threads(min(4, os.cpu_count() or 1))
torch.manual_seed(20260727)
np.random.seed(20260727)

ROOT = Path("out/real_smollm2_v16_dual_prefix_selector")
ROOT.mkdir(parents=True, exist_ok=True)

INSTRUCT_ID = v14.INSTRUCT_ID
SQL_ID = v14.SQL_ID
SEED = 20260727
DEPTHS = (1, 2, 4)
MAX_DEPTH = max(DEPTHS)
FEATURE_BATCH_SIZE = 8
BOOTSTRAPS = 5000
DOMAIN_LOSS_TOLERANCE = 0.02
MCQ_ACCURACY_TOLERANCE = 0.02


def ordered_rows(datasets: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[tuple[str, int]]]:
    rows: list[dict[str, Any]] = []
    keys: list[tuple[str, int]] = []
    for domain, examples in datasets.items():
        for index, row in enumerate(examples):
            rows.append(row)
            keys.append((domain, index))
    return rows, keys


def tokenized_feature_inputs(
    tokenizer: Any,
    datasets: dict[str, list[dict[str, Any]]],
) -> tuple[list[list[int]], list[tuple[str, int]]]:
    rows, keys = ordered_rows(datasets)
    sequences: list[list[int]] = []
    for row in rows:
        ids = tokenizer(
            str(row["feature_text"]),
            add_special_tokens=False,
            truncation=True,
            max_length=v14.MAX_FEATURE_TOKENS,
        )["input_ids"]
        if not ids:
            ids = [tokenizer.eos_token_id]
        sequences.append(ids)
    return sequences, keys


@torch.inference_mode()
def prefix_features_for_model(
    model: nn.Module,
    tokenizer: Any,
    datasets: dict[str, list[dict[str, Any]]],
) -> tuple[dict[int, np.ndarray], list[tuple[str, int]]]:
    sequences, keys = tokenized_feature_inputs(tokenizer, datasets)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    values: dict[int, list[np.ndarray]] = {depth: [] for depth in DEPTHS}

    original_layers = model.model.layers
    model.model.layers = nn.ModuleList(list(original_layers[:MAX_DEPTH]))
    try:
        for batch_start in range(0, len(sequences), FEATURE_BATCH_SIZE):
            batch = sequences[batch_start : batch_start + FEATURE_BATCH_SIZE]
            max_len = max(len(sequence) for sequence in batch)
            input_ids = torch.full((len(batch), max_len), pad, dtype=torch.long)
            attention_mask = torch.zeros_like(input_ids)
            for index, sequence in enumerate(batch):
                input_ids[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
                attention_mask[index, : len(sequence)] = 1

            outputs = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states
            assert hidden_states is not None
            for depth in DEPTHS:
                # hidden_states[d] is the residual state after d blocks.  At the
                # truncation boundary MAX_DEPTH, last_hidden_state is already
                # final-RMS-normalized.  Apply the same final norm to shallower
                # states so every candidate depth has the same feature gauge.
                state = outputs.last_hidden_state if depth == MAX_DEPTH else model.model.norm(hidden_states[depth])
                state = state.detach().cpu().float()
                mask = attention_mask.detach().cpu().float()
                denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                mean_state = (state * mask[:, :, None]).sum(dim=1) / denom
                last_index = attention_mask.sum(dim=1).clamp_min(1) - 1
                last_state = state[torch.arange(len(batch)), last_index.cpu()]
                feature = torch.cat([mean_state, last_state], dim=1)
                values[depth].extend(feature.numpy().astype(np.float64))
    finally:
        model.model.layers = original_layers

    return {depth: np.stack(rows) for depth, rows in values.items()}, keys


def dual_features(
    instruct_features: dict[int, np.ndarray],
    sql_features: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    for depth in DEPTHS:
        hi = instruct_features[depth]
        hs = sql_features[depth]
        if hi.shape != hs.shape:
            raise RuntimeError("prefix feature shape mismatch")
        half = hi.shape[1] // 2
        mean_i, last_i = hi[:, :half], hi[:, half:]
        mean_s, last_s = hs[:, :half], hs[:, half:]

        mean_avg = 0.5 * (mean_i + mean_s)
        mean_diff = mean_i - mean_s
        last_avg = 0.5 * (last_i + last_s)
        last_diff = last_i - last_s

        scalar = np.stack(
            [
                np.linalg.norm(mean_diff, axis=1),
                np.linalg.norm(last_diff, axis=1),
                np.sum(mean_i * mean_s, axis=1)
                / np.maximum(np.linalg.norm(mean_i, axis=1) * np.linalg.norm(mean_s, axis=1), 1e-30),
                np.sum(last_i * last_s, axis=1)
                / np.maximum(np.linalg.norm(last_i, axis=1) * np.linalg.norm(last_s, axis=1), 1e-30),
            ],
            axis=1,
        )
        result[depth] = np.concatenate(
            [mean_avg, mean_diff, last_avg, last_diff, scalar], axis=1
        ).astype(np.float64)
    return result


def loss_differences(
    instruct_records: dict[str, list[dict[str, Any]]],
    sql_records: dict[str, list[dict[str, Any]]],
    keys: list[tuple[str, int]],
) -> np.ndarray:
    # Positive means SQL has lower loss and should be selected.
    return np.array(
        [
            float(instruct_records[domain][index]["loss"])
            - float(sql_records[domain][index]["loss"])
            for domain, index in keys
        ],
        dtype=np.float64,
    )


def regret_weights(differences: np.ndarray, keys: list[tuple[str, int]]) -> np.ndarray:
    base = v14.domain_equal_weights(keys)
    domains = sorted({domain for domain, _ in keys})
    normalized = np.ones_like(differences)
    for domain in domains:
        indices = np.array([i for i, key in enumerate(keys) if key[0] == domain], dtype=np.int64)
        gaps = np.abs(differences[indices])
        scale = float(np.median(gaps[gaps > 1e-12])) if np.any(gaps > 1e-12) else 1.0
        normalized[indices] = np.clip(gaps / max(scale, 1e-12), 0.1, 5.0)
    weight = base * normalized
    weight *= len(weight) / weight.sum()
    return weight


def weighted_regret_fraction(
    scores: np.ndarray,
    differences: np.ndarray,
    weights: np.ndarray,
) -> float:
    labels = np.where(differences > 0.0, 1.0, -1.0)
    wrong = np.sign(scores) != labels
    numerator = float(np.sum(weights * np.abs(differences) * wrong))
    denominator = float(np.sum(weights * np.abs(differences)))
    return numerator / max(denominator, 1e-30)


def leave_one_domain_out_depth(
    features: dict[int, np.ndarray],
    differences: np.ndarray,
    keys: list[tuple[str, int]],
) -> dict[str, Any]:
    domains = sorted({domain for domain, _ in keys})
    labels = np.where(differences > 0.0, 1.0, -1.0)
    all_weights = regret_weights(differences, keys)
    depth_rows = []
    for depth in DEPTHS:
        fold_regrets = []
        fold_accuracies = []
        for fold_index, domain in enumerate(domains):
            test_mask = np.array([key[0] == domain for key in keys], dtype=bool)
            train_mask = ~test_mask
            fit = v14.fit_ridge_gcv(
                features[depth][train_mask],
                labels[train_mask],
                all_weights[train_mask],
            )
            scores = v14.selector_scores(features[depth][test_mask], fit)
            fold_weights = all_weights[test_mask]
            fold_differences = differences[test_mask]
            fold_regrets.append(weighted_regret_fraction(scores, fold_differences, fold_weights))
            fold_accuracies.append(float(np.mean(np.sign(scores) == labels[test_mask])))
        depth_rows.append(
            {
                "depth": depth,
                "mean_leave_one_domain_out_regret_fraction": float(np.mean(fold_regrets)),
                "mean_leave_one_domain_out_accuracy": float(np.mean(fold_accuracies)),
                "fold_regret_fraction": fold_regrets,
                "fold_accuracy": fold_accuracies,
            }
        )
    selected = min(
        depth_rows,
        key=lambda row: (
            row["mean_leave_one_domain_out_regret_fraction"],
            -row["mean_leave_one_domain_out_accuracy"],
            row["depth"],
        ),
    )
    return {"selected_depth": int(selected["depth"]), "depth_rows": depth_rows}


def save_checkpoint(
    instruct: nn.Module,
    sql: nn.Module,
    tokenizer: Any,
    fit: dict[str, Any],
    selected_depth: int,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    path = ROOT / "PROMOTED_DUAL_PREFIX_SELECTOR_CHECKPOINT.pt"
    payload = {
        "format": "smollm2_dual_prefix_selector_v2",
        "architecture": (
            "run both parents through selected_depth blocks; route once; continue "
            "the selected parent's exact prefix state through its remaining blocks"
        ),
        "model_ids_for_provenance_only": {"instruct": INSTRUCT_ID, "sql": SQL_ID},
        "config": instruct.config.to_dict(),
        "tokenizer_json": tokenizer.backend_tokenizer.to_str(),
        "tokenizer_special_tokens_map": tokenizer.special_tokens_map,
        "selector": {
            "selected_depth": int(selected_depth),
            "positive_route": "sql",
            "feature_definition": (
                "concat symmetric/antisymmetric mean and last-token states from both "
                "parents after the selected prefix depth, plus four norm/cosine invariants"
            ),
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
    signatures = [v14.v13.structural_signature(config) for config in configs]
    if signatures[0] != signatures[1]:
        raise RuntimeError({"status": "CONFIG_MISMATCH", "signatures": signatures})

    tokenizers = [AutoTokenizer.from_pretrained(model_id) for model_id in [INSTRUCT_ID, SQL_ID]]
    tokenizer_audit = v14.v13.tokenizer_audit(tokenizers)
    if not all(tokenizer_audit["vocab_equal"]) or not all(
        all(row) for row in tokenizer_audit["probe_ids_equal"]
    ):
        raise RuntimeError({"status": "TOKENIZER_MISMATCH", "audit": tokenizer_audit})
    tokenizer = tokenizers[0]
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        raise RuntimeError("instruct tokenizer has no chat template")

    calibration, heldout = v14.build_datasets(tokenizer)

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

    calibration_records: dict[str, Any] = {}
    heldout_records: dict[str, Any] = {}
    timings: dict[str, float] = {}
    for name, model in [("instruct", instruct), ("sql", sql)]:
        calibration_records[name], timings[name + "_calibration_seconds"] = v14.evaluate_all(
            model, tokenizer, calibration
        )
        heldout_records[name], timings[name + "_heldout_seconds"] = v14.evaluate_all(
            model, tokenizer, heldout
        )
        print(name, v14.summarize(calibration_records[name]), v14.summarize(heldout_records[name]))

    calibration_oracle, _ = v14.oracle_records(
        calibration_records["instruct"], calibration_records["sql"]
    )
    heldout_oracle, heldout_oracle_choices = v14.oracle_records(
        heldout_records["instruct"], heldout_records["sql"]
    )
    heldout_domain_best, domain_best_parent = v14.domain_best_records(
        heldout_records["instruct"], heldout_records["sql"]
    )

    oracle_bootstrap = {
        "vs_instruct": v14.bootstrap_composite(
            heldout_oracle, heldout_records["instruct"], SEED + 1
        ),
        "vs_sql": v14.bootstrap_composite(heldout_oracle, heldout_records["sql"], SEED + 2),
    }
    oracle_gate = all(value["ci95"][1] < 0.0 for value in oracle_bootstrap.values())

    print("Extracting dual-parent early-prefix states")
    instruct_cal_by_depth, keys_cal_i = prefix_features_for_model(instruct, tokenizer, calibration)
    sql_cal_by_depth, keys_cal_s = prefix_features_for_model(sql, tokenizer, calibration)
    if keys_cal_i != keys_cal_s:
        raise RuntimeError("calibration feature key mismatch")
    calibration_features = dual_features(instruct_cal_by_depth, sql_cal_by_depth)

    differences_cal = loss_differences(
        calibration_records["instruct"], calibration_records["sql"], keys_cal_i
    )
    depth_selection = leave_one_domain_out_depth(
        calibration_features, differences_cal, keys_cal_i
    )
    selected_depth = int(depth_selection["selected_depth"])
    labels_cal = np.where(differences_cal > 0.0, 1.0, -1.0)
    weights_cal = regret_weights(differences_cal, keys_cal_i)
    fit = v14.fit_ridge_gcv(
        calibration_features[selected_depth], labels_cal, weights_cal
    )

    instruct_test_by_depth, keys_test_i = prefix_features_for_model(instruct, tokenizer, heldout)
    sql_test_by_depth, keys_test_s = prefix_features_for_model(sql, tokenizer, heldout)
    if keys_test_i != keys_test_s:
        raise RuntimeError("heldout feature key mismatch")
    heldout_features = dual_features(instruct_test_by_depth, sql_test_by_depth)
    scores = v14.selector_scores(heldout_features[selected_depth], fit)
    selector_choices = {
        key: int(score > 0.0) for key, score in zip(keys_test_i, scores)
    }
    selected_records = v14.combine_records(
        heldout_records["instruct"], heldout_records["sql"], selector_choices
    )

    selector_bootstrap = {
        "vs_instruct": v14.bootstrap_composite(
            selected_records, heldout_records["instruct"], SEED + 11
        ),
        "vs_sql": v14.bootstrap_composite(
            selected_records, heldout_records["sql"], SEED + 12
        ),
        "vs_domain_best": v14.bootstrap_composite(
            selected_records, heldout_domain_best, SEED + 13
        ),
        "vs_oracle": v14.bootstrap_composite(selected_records, heldout_oracle, SEED + 14),
    }

    summaries = {
        "instruct": v14.summarize(heldout_records["instruct"]),
        "sql": v14.summarize(heldout_records["sql"]),
        "domain_best": v14.summarize(heldout_domain_best),
        "oracle": v14.summarize(heldout_oracle),
        "selector": v14.summarize(selected_records),
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
    domain_best_gate = selector_bootstrap["vs_domain_best"]["ci95"][1] <= 0.0
    domain_gate = all(
        entry["loss_pass"] and entry["accuracy_pass"] for entry in domain_gates.values()
    )
    promoted = bool(oracle_gate and selector_parent_gate and domain_best_gate and domain_gate)

    differences_test = loss_differences(
        heldout_records["instruct"], heldout_records["sql"], keys_test_i
    )
    heldout_regret = weighted_regret_fraction(
        scores,
        differences_test,
        regret_weights(differences_test, keys_test_i),
    )

    parent_parameter_count = int(sum(parameter.numel() for parameter in instruct.parameters()))
    selector_parameter_count = int(len(fit["coefficient"]) + 2 * len(fit["feature_mean"]))
    compiled_parameter_count = 2 * parent_parameter_count + selector_parameter_count
    num_layers = int(instruct.config.num_hidden_layers)

    result = {
        "status": "DUAL_PREFIX_SELECTOR_PROMOTED" if promoted else "DUAL_PREFIX_SELECTOR_NOT_PROMOTED",
        "models": {"instruct": INSTRUCT_ID, "sql": SQL_ID},
        "architecture": {
            "single_checkpoint": bool(promoted),
            "sequence_level_selector": True,
            "two_prefixes_then_one_exact_parent_trajectory": True,
            "selected_depth": selected_depth,
            "one_parent_suffix_executed_per_input": True,
            "runtime_parent_dependencies": False if promoted else None,
            "logit_or_probability_ensemble": False,
            "persistent_hidden_streams_after_route": 1,
            "parameter_count": compiled_parameter_count,
            "parent_parameter_count": parent_parameter_count,
            "parameter_ratio": compiled_parameter_count / parent_parameter_count,
            "selector_parameter_count": selector_parameter_count,
            "expected_forward_compute_ratio": 1.0 + selected_depth / num_layers,
            "exact_parent_function_when_route_fixed": True,
        },
        "calibration_information": {
            "examples": len(keys_cal_i),
            "binary_oracle_labels": len(keys_cal_i),
            "domains": {domain: len(rows) for domain, rows in calibration.items()},
            "heldout_domains": {domain: len(rows) for domain, rows in heldout.items()},
            "no_heldout_labels_used_for_depth_or_fit": True,
            "training_weighting": "domain-equal multiplied by clipped normalized absolute parent loss gap",
        },
        "depth_selection": depth_selection,
        "selector_fit": {
            "lambda_gcv": float(fit["lambda"]),
            "effective_degrees_of_freedom": float(fit["effective_degrees_of_freedom"]),
            "calibration_oracle_label_accuracy": float(fit["calibration_accuracy"]),
            "gcv": float(fit["gcv"]),
            "heldout_regret_fraction": heldout_regret,
        },
        "oracle_gate": {"pass": oracle_gate, "bootstrap": oracle_bootstrap},
        "selector_bootstrap": selector_bootstrap,
        "domain_gates": domain_gates,
        "route_statistics": v14.route_statistics(selector_choices, heldout_oracle_choices),
        "domain_best_parent": domain_best_parent,
        "summaries": summaries,
        "timings": timings,
        "promotion_gates": {
            "oracle_significantly_beats_both_parents": oracle_gate,
            "selector_significantly_beats_both_parents": selector_parent_gate,
            "selector_not_worse_than_domain_best_bootstrap": domain_best_gate,
            "every_domain_within_2pct_loss_and_2pp_accuracy_of_best_parent": domain_gate,
            "all_pass": promoted,
        },
    }

    checkpoint = None
    if promoted:
        checkpoint = save_checkpoint(
            instruct,
            sql,
            tokenizer,
            fit,
            selected_depth,
            {
                "status": result["status"],
                "summaries": summaries,
                "selector_bootstrap": selector_bootstrap,
                "domain_gates": domain_gates,
                "depth_selection": depth_selection,
            },
        )
    result["checkpoint"] = checkpoint

    (ROOT / "RESULTS.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    metric_rows = []
    for model_name, summary in summaries.items():
        for domain, entry in summary.items():
            if domain == "balanced_loss":
                continue
            metric_rows.append(
                {
                    "model": model_name,
                    "domain": domain,
                    "n": entry["n"],
                    "loss": entry["loss"],
                    "accuracy": entry.get("accuracy", ""),
                }
            )
    with (ROOT / "METRICS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["model", "domain", "n", "loss", "accuracy"]
        )
        writer.writeheader()
        writer.writerows(metric_rows)

    report = [
        "# SmolLM2 Dual-Prefix Regret-Weighted Selector",
        "",
        f"Status: **{result['status']}**",
        f"Selected prefix depth: **{selected_depth}**",
        f"Expected forward compute ratio: **{result['architecture']['expected_forward_compute_ratio']:.6f}x**",
        f"Parameter ratio: **{result['architecture']['parameter_ratio']:.6f}x**",
        f"Held-out selector/oracle accuracy: **{result['route_statistics']['overall']['selector_oracle_accuracy']:.4f}**",
        f"Held-out weighted regret fraction: **{heldout_regret:.4f}**",
        f"Oracle gate: **{oracle_gate}**",
        f"Selector beats both parents gate: **{selector_parent_gate}**",
        f"Domain-best bootstrap gate: **{domain_best_gate}**",
        f"Every-domain preservation gate: **{domain_gate}**",
        "",
        "The two parents are evaluated only through the selected early prefix depth before routing. The selected parent's exact prefix state is continued through its remaining layers, so no cross-parent hidden-state splice occurs.",
    ]
    (ROOT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")

    print(json.dumps(result, indent=2))
    del instruct, sql
    gc.collect()


if __name__ == "__main__":
    main()
