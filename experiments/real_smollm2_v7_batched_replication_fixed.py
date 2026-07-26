from __future__ import annotations

import gc
import importlib.util
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v7 = load_module("wf_v7_source", "real_smollm2_v7_batched_replication.py")
ROOT = Path("out/real_smollm2_v7_batched_replication_fixed")
ROOT.mkdir(parents=True, exist_ok=True)
v7.ROOT = ROOT


@torch.inference_mode()
def evaluate_mcq_choice_normalized(
    model,
    tokenizer,
    examples: list[dict],
    batch_sequences: int = v7.BATCH_SEQUENCES,
):
    """Evaluate a finite answer set with a proper renormalized choice loss.

    For candidate answer string j, let
        s_j = -log P(answer_j | prompt)
    be the total sequence negative log likelihood. The choice probability is
        q_j = exp(-s_j) / sum_k exp(-s_k),
    and the reported loss is -log q_y. This measures discrimination among
    the stated answer choices; it cannot improve merely by raising every
    answer token probability together.
    """
    flat: list[dict] = []
    option_token_lengths: list[list[int]] = [[] for _ in examples]

    for example_id, row in enumerate(examples):
        prompt_ids = v7.v1.encode_no_special(tokenizer, row["prompt"])
        if not prompt_ids:
            prompt_ids = [tokenizer.eos_token_id]

        for choice_id, choice in enumerate(row["choices"]):
            option_ids = v7.v1.encode_no_special(tokenizer, " " + choice)
            if not option_ids:
                option_ids = [tokenizer.eos_token_id]
            option_token_lengths[example_id].append(len(option_ids))
            flat.append({
                "example_id": example_id,
                "choice_id": choice_id,
                "sequence": prompt_ids + option_ids,
                "start": len(prompt_ids),
            })

    sequence_nlls: dict[int, dict[int, float]] = {
        i: {} for i in range(len(examples))
    }
    pad = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    for start_batch in range(0, len(flat), batch_sequences):
        batch_items = flat[start_batch : start_batch + batch_sequences]
        max_len = max(len(item["sequence"]) for item in batch_items)
        input_ids = torch.full((len(batch_items), max_len), pad, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)

        for i, item in enumerate(batch_items):
            seq = item["sequence"]
            input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            attention_mask[i, : len(seq)] = 1

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits.float()
        logp = logits.log_softmax(-1)

        for i, item in enumerate(batch_items):
            seq = item["sequence"]
            token_positions = torch.arange(item["start"], len(seq), dtype=torch.long)
            pred_positions = token_positions - 1
            targets = input_ids[i, token_positions]
            sequence_nll = -logp[i, pred_positions, targets].sum()
            sequence_nlls[item["example_id"]][item["choice_id"]] = float(sequence_nll)

    records: list[dict] = []
    for example_id, row in enumerate(examples):
        nlls = np.array(
            [sequence_nlls[example_id][i] for i in range(len(row["choices"]))],
            dtype=np.float64,
        )
        label = int(row["label"])
        prediction = int(np.argmin(nlls))
        # Stable logsumexp(-nlls).
        logits = -nlls
        max_logit = float(np.max(logits))
        log_partition = max_logit + math.log(float(np.exp(logits - max_logit).sum()))
        choice_nll = float(nlls[label] + log_partition)

        records.append({
            "example_id": example_id,
            # Existing summary/bootstrap code reads correct_nll. It now means
            # the proper answer-set-normalized cross entropy, not raw token NLL.
            "correct_nll": choice_nll,
            "choice_nll": choice_nll,
            "raw_correct_sequence_nll": float(nlls[label]),
            "correct": int(prediction == label),
            "prediction": prediction,
            "label": label,
            "num_choices": int(len(nlls)),
            "option_token_lengths": option_token_lengths[example_id],
        })

    return records


v7.evaluate_mcq_batched = evaluate_mcq_choice_normalized


def main() -> None:
    # The source main performs the frozen candidate construction, benchmark
    # sampling, paired bootstrap, exact-copy audit, canonical rotation audit,
    # checkpoint reload audit, and writes raw records.
    v7.main()

    result_path = ROOT / "RESULTS.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    summaries = result["summaries"]
    comparisons = result["comparisons"]

    domains = [
        "wikitext",
        "openbookqa_chat",
        "commonsenseqa_chat",
        "winogrande_chat",
        "piqa_chat",
        "boolq_chat",
    ]
    mcq_domains = [d for d in domains if d != "wikitext"]

    for domain in domains:
        for model_name in ["parent_base", "parent_instruct", "fixed_candidate"]:
            entry = summaries[model_name][domain]
            if entry["n"] <= 0 or not np.isfinite(entry["nll"]):
                raise RuntimeError({
                    "status": "INVALID_EVALUATION",
                    "model": model_name,
                    "domain": domain,
                    "entry": entry,
                })

    best_parent_loss = {
        domain: min(
            summaries["parent_base"][domain]["nll"],
            summaries["parent_instruct"][domain]["nll"],
        )
        for domain in domains
    }
    best_parent_accuracy = {
        domain: max(
            summaries["parent_base"][domain]["accuracy"],
            summaries["parent_instruct"][domain]["accuracy"],
        )
        for domain in mcq_domains
    }

    # Frozen before this run: at most 1% loss regression and 2 percentage
    # points accuracy regression against the per-domain better parent.
    loss_within_1pct = all(
        summaries["fixed_candidate"][domain]["nll"]
        <= 1.01 * best_parent_loss[domain]
        for domain in domains
    )
    accuracy_within_2pp = all(
        summaries["fixed_candidate"][domain]["accuracy"]
        >= best_parent_accuracy[domain] - 0.02
        for domain in mcq_domains
    )
    composite_significant = all(
        comparisons[parent]["composite_relative_nll"]["ci95"][1] < 0.0
        for parent in ["parent_base", "parent_instruct"]
    )
    rotation_pass = bool(result["canonical_rotation"]["pass"])

    strict_promoted = bool(
        loss_within_1pct
        and accuracy_within_2pp
        and composite_significant
        and rotation_pass
    )

    result.update({
        "status": (
            "PROPER_CHOICE_NORMALIZED_PASS"
            if strict_promoted
            else "PROPER_CHOICE_NORMALIZED_NOT_PROMOTED"
        ),
        "metric_definition": {
            "mcq_sequence_score": "s_j = -log P(answer_j | prompt)",
            "mcq_choice_probability": "q_j = exp(-s_j) / sum_k exp(-s_k)",
            "mcq_loss": "-log q_correct",
            "wikitext_loss": "next-token cross entropy",
        },
        "strict_eligibility": {
            "max_per_domain_loss_regression": 0.01,
            "max_per_domain_accuracy_regression_pp": 0.02,
            "must_significantly_beat_both_whole_parents": True,
            "candidate_frozen_before_evaluation": True,
            "sample_selection_frozen_before_evaluation": True,
        },
        "best_parent_loss": best_parent_loss,
        "best_parent_accuracy": best_parent_accuracy,
        "all_losses_within_1pct": loss_within_1pct,
        "all_accuracies_within_2pp": accuracy_within_2pp,
        "composite_significantly_beats_both_parents": composite_significant,
        "promoted": strict_promoted,
    })
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if not strict_promoted:
        shutil.rmtree(ROOT / "CANONICAL_ROTATED_MODEL", ignore_errors=True)

    report_lines = [
        "# Proper Choice-Normalized Cross-Benchmark Validation",
        "",
        f"Status: **{result['status']}**",
        "",
        "MCQ metric:",
        "`s_j = -log P(answer_j | prompt)` and "
        "`loss = -log(exp(-s_y)/sum_j exp(-s_j))`.",
        "",
        f"Composite significantly beats both parents: **{composite_significant}**",
        f"Every loss domain within 1% of the better parent: **{loss_within_1pct}**",
        f"Every MCQ accuracy within 2pp of the better parent: **{accuracy_within_2pp}**",
        f"Canonical rotation audit passed: **{rotation_pass}**",
        f"Final promotion: **{strict_promoted}**",
    ]
    (ROOT / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(json.dumps(result, indent=2))

    gc.collect()


if __name__ == "__main__":
    main()
